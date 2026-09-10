"""Orquestração do fluxo RAG com LangGraph."""

import operator
import re
import unicodedata
from functools import lru_cache
from typing import Annotated, Literal, TypedDict, cast, NotRequired

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.generation import (
    format_documents,
    get_generation_chain,
    get_classification_chain,
)
from app.logger import monitor_node_execution
from app.nodes.prontuario import (
    COM_PACIENTE,
    SEM_PACIENTE,
    exames_pendentes_node,
    prontuario_node,
    rotear_por_paciente,
)
from app.rag import format_source, retrieve_documents

PRESSURE_PATTERN = re.compile(
    r"\b(?P<systolic>\d{2,3})\s*(?:/|x|por)\s*(?P<diastolic>\d{2,3})\b",
    flags=re.IGNORECASE,
)

ALERT_TERMS = {
    "dor toracica": "dor torácica",
    "dor no peito": "dor no peito",
    "aperto no peito": "aperto no peito",
    "pressao no peito": "pressão no peito",
    "peso no peito": "peso no peito",
    "desconforto toracico": "desconforto torácico",
    "dor irradiando para o braco": "dor irradiando para o braço",
    "dor irradiada para o braco": "dor irradiada para o braço",
    "dor irradiando para a mandibula": "dor irradiando para a mandíbula",
    "dor irradiada para a mandibula": "dor irradiada para a mandíbula",
    "dor subita e intensa": "dor súbita e intensa",
    "dor irradiando para o dorso": "dor irradiando para o dorso",
    "dor irradiada para o dorso": "dor irradiada para o dorso",
    "falta de ar": "falta de ar",
    "suor frio": "suor frio",
    "sudorese": "sudorese",
    "palidez": "palidez",
    "nausea": "náusea",
    "vomito": "vômito",
    "palpitacao": "palpitação",
    "confusao": "confusão",
    "alteracao neurologica": "alteração neurológica",
    "fraqueza em um lado": "fraqueza em um lado do corpo",
    "alteracao visual": "alteração visual",
    "visao turva": "visão turva",
    "desmaio": "desmaio",
    "sincope": "síncope",
    "convulsao": "convulsão",
}

INSTRUCTIONS = {
    "informacao_geral": "Responda esta pergunta que solicita informação",
    "frequencia": "Responda esta pergunta sobre a frequência de uma doença",
    "tratamento": "Responda esta pergunta que solicita informações de tratamento de uma doença",
    "sintomas": "Responda esta pergunta sobre os sintomas de uma doença",
    "diagnostico": "Responda esta pergunta sobre como realizar o diagnostico de uma doença.",
    "fora_escopo": "Responda que a pergunta esta fora do escopo.",
}

RAG_REQUIRED = {
    "tratamento": True,
    "diagnostico": True,
    "sintomas": True,
    "frequencia": False,
    "informacao_geral": False,
}

SafetyStatus = Literal["revisao_obrigatoria", "atencao", "emergencia"]

# Aviso colado na resposta quando o código informado não existe no prontuário.
# É determinístico de propósito: a LLM não pode ser a responsável por comunicar
# uma falha de sistema, porque ela pode simplesmente não mencionar.
AVISO_PACIENTE_NAO_ENCONTRADO = (
    "AVISO: o código de paciente informado não foi encontrado no prontuário. "
    "A resposta abaixo se baseia apenas na base de conhecimento e NÃO considera "
    "exames, medicações ou histórico deste paciente.\n\n"
)


class GraphInput(TypedDict):
    """Entrada pública do grafo."""

    question: str
    llm_provider: str
    patient_code: str


class GraphOutput(TypedDict):
    """Saída pública do grafo."""

    answer: str
    sources: NotRequired[list[str]]
    safety_alerts: list[str]
    safety_status: SafetyStatus
    requires_human_review: bool
    patient_summary: str
    patient_found: bool
    pending_exams: list[str]


class GraphState(TypedDict, total=False):
    """Estado compartilhado entre os nós do LangGraph."""

    question: str
    intent: str
    confidence: float
    instruction: str
    llm_provider: str
    patient_code: str
    patient_summary: str
    patient_found: bool
    pending_exams: list[str]
    documents: list[Document]
    answer: str
    # As fontes são preenchidas por mais de um nó (prontuário e recuperação),
    # então o operator.add junta as listas em vez de uma sobrescrever a outra.
    sources: Annotated[list[str], operator.add]
    safety_alerts: list[str]
    safety_status: SafetyStatus
    requires_human_review: bool


def _normalize_text(text: str) -> str:
    """Remove acentos e normaliza o texto para as regras determinísticas."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_accents.lower()


def detect_safety_alerts(question: str) -> tuple[bool, list[str]]:
    """Detecta pressão muito elevada e sintomas de alerta informados."""
    normalized_question = _normalize_text(question)
    symptoms = [
        label for term, label in ALERT_TERMS.items() if term in normalized_question
    ]

    severe_pressure = False
    for match in PRESSURE_PATTERN.finditer(normalized_question):
        systolic = int(match.group("systolic"))
        diastolic = int(match.group("diastolic"))
        if systolic >= 180 or diastolic >= 120:
            severe_pressure = True
            break

    return severe_pressure, symptoms


@monitor_node_execution
def retrieve_node(state: GraphState) -> GraphState:
    """Recupera os documentos relevantes e suas fontes."""
    documents = retrieve_documents(state["question"])
    # A fonte inclui a seção, então dois trechos do mesmo arquivo continuam
    # aparecendo separados em vez de virarem uma única linha no dedup.
    sources = list(dict.fromkeys(format_source(document) for document in documents))
    return {
        "documents": documents,
        "sources": sources,
    }


# @monitor_node_execution
def _classify_intent_fallback(question: str) -> dict:
    """Classificação por regras, usada quando a LLM não responde."""
    question_lower = question.lower()

    if any(x in question_lower for x in ["sintoma", "sintomas", "sinal"]):
        return {"intent": "sintomas", "confidence": 0.90}

    if any(
        x in question_lower
        for x in [
            "quantas pessoas",
            "prevalência",
            "incidência",
            "frequência",
            "frequencia",
        ]
    ):
        return {"intent": "frequencia", "confidence": 0.90}

    if any(x in question_lower for x in ["tratamento", "tratar", "terapia"]):
        return {"intent": "tratamento", "confidence": 0.90}

    return {"intent": "informacao_geral", "confidence": 0.50}


def should_use_rag(state: GraphState) -> GraphState:
    intent = state["intent"]

    if RAG_REQUIRED.get(intent, True):
        return "rag"

    return "generate"


@monitor_node_execution
def classify_question(state: GraphState) -> GraphState:
    """Classifica a intenção com a LLM; recorre às regras se a LLM falhar."""
    question = state["question"]
    provider = "ollama"

    try:
        result = get_classification_chain(provider).invoke({"question": question})
        intent = result.intent
        confidence = result.confidence
    except Exception:  # noqa: BLE001 - falha da LLM exige fallback determinístico
        fallback = _classify_intent_fallback(question)
        intent = fallback["intent"]
        confidence = fallback["confidence"]

    return {
        "intent": intent,
        "confidence": confidence,
        "instruction": INSTRUCTIONS[intent],
    }


@monitor_node_execution
def generate_node(state: GraphState) -> GraphState:
    """Gera uma resposta usando os documentos recuperados."""
    documents = state.get("documents", [])
    context = format_documents(documents)

    # O prontuário entra antes dos trechos da base para que a resposta seja
    # contextualizada com a situação atual do paciente.
    patient_summary = state.get("patient_summary", "")
    if patient_summary:
        context = f"[Prontuário do paciente]\n{patient_summary}\n\n{context}"

    answer = get_generation_chain(state["llm_provider"]).invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )
    return {"answer": answer}


@monitor_node_execution
def safety_node(state: GraphState) -> GraphState:
    """Aplica alertas determinísticos sem substituir a avaliação humana."""
    severe_pressure, symptoms = detect_safety_alerts(state["question"])
    answer = state.get("answer", "")
    alerts = symptoms.copy()

    # Aplicado antes dos avisos de segurança, para que um alerta de emergência
    # continue sendo a primeira coisa que o profissional lê.
    codigo_informado = str(state.get("patient_code", "")).strip()
    if codigo_informado and not state.get("patient_found", False):
        answer = AVISO_PACIENTE_NAO_ENCONTRADO + answer

    if severe_pressure:
        alerts.insert(0, "pressão arterial muito elevada")

    if symptoms:
        safety_status: SafetyStatus = "emergencia"
        # O texto fala com o profissional que conduz o caso, e não com o
        # paciente: quem lê é quem presta o atendimento.
        prefix = (
            "ALERTA DE SEGURANÇA: os dados informados contêm sinais "
            "potencialmente compatíveis com urgência clínica. Priorize a "
            "avaliação imediata do caso e considere acionar o protocolo de "
            "emergência da instituição. Repetir a aferição não deve atrasar a "
            "conduta.\n\n"
        )
        answer = prefix + answer
    elif severe_pressure:
        safety_status = "atencao"
        prefix = (
            "ATENÇÃO: foi informado um valor de pressão arterial muito elevado. "
            "O caso requer avaliação em caráter prioritário. Na presença de dor "
            "torácica, dispneia ou alterações neurológicas ou visuais, considere "
            "conduta de emergência.\n\n"
        )
        answer = prefix + answer
    else:
        safety_status = "revisao_obrigatoria"

    # A configuração pode dispensar a revisão em respostas de rotina, mas nunca
    # quando algum alerta foi disparado: esse piso não depende da configuração.
    requires_human_review = (
        get_settings().require_human_review or safety_status != "revisao_obrigatoria"
    )

    return {
        "answer": answer,
        "safety_alerts": alerts,
        "safety_status": safety_status,
        "requires_human_review": requires_human_review,
    }


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Monta e compila o fluxo do CardioAssist AI."""
    builder = StateGraph(
        GraphState,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )
    builder.add_node("prontuario", prontuario_node)
    builder.add_node("exames_pendentes", exames_pendentes_node)
    builder.add_node("classify", classify_question)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("safety", safety_node)

    # Só passa pelas etapas do prontuário quando um paciente é informado.
    builder.add_conditional_edges(
        START,
        rotear_por_paciente,
        {
            COM_PACIENTE: "prontuario",
            SEM_PACIENTE: "classify",
        },
    )
    builder.add_conditional_edges(
        "classify", should_use_rag, {"rag": "retrieve", "generate": "generate"}
    )
    builder.add_edge("prontuario", "exames_pendentes")
    builder.add_edge("exames_pendentes", "classify")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "safety")
    builder.add_edge("safety", END)

    return builder.compile()


def run_graph(
    question: str,
    llm_provider: str = "ollama",
    patient_code: str = "",
) -> GraphOutput:
    """Executa o fluxo completo e devolve somente a saída pública."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")
    if llm_provider not in {"ollama", "openai"}:
        raise ValueError("llm_provider deve ser 'ollama' ou 'openai'.")

    result = get_graph().invoke(
        {
            "question": normalized_question,
            "llm_provider": llm_provider,
            "patient_code": patient_code.strip(),
        }
    )
    return cast(GraphOutput, result)
