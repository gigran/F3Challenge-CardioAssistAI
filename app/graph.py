"""Orquestração do fluxo RAG com LangGraph."""

import re
import unicodedata
from functools import lru_cache
from typing import Literal, TypedDict, cast

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.generation import format_documents, get_generation_chain
from app.rag import retrieve_documents

PRESSURE_PATTERN = re.compile(
    r"\b(?P<systolic>\d{2,3})\s*(?:/|x|por)\s*(?P<diastolic>\d{2,3})\b",
    flags=re.IGNORECASE,
)

ALERT_TERMS = {
    "dor toracica": "dor torácica",
    "dor no peito": "dor no peito",
    "falta de ar": "falta de ar",
    "confusao": "confusão",
    "alteracao neurologica": "alteração neurológica",
    "fraqueza em um lado": "fraqueza em um lado do corpo",
    "alteracao visual": "alteração visual",
    "visao turva": "visão turva",
    "desmaio": "desmaio",
    "convulsao": "convulsão",
}

SafetyStatus = Literal["revisao_obrigatoria", "atencao", "emergencia"]


class GraphInput(TypedDict):
    """Entrada pública do grafo."""

    question: str


class GraphOutput(TypedDict):
    """Saída pública do grafo."""

    answer: str
    sources: list[str]
    safety_alerts: list[str]
    safety_status: SafetyStatus
    requires_human_review: bool


class GraphState(TypedDict, total=False):
    """Estado compartilhado entre os nós do LangGraph."""

    question: str
    documents: list[Document]
    answer: str
    sources: list[str]
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


def retrieve_node(state: GraphState) -> GraphState:
    """Recupera os documentos relevantes e suas fontes."""
    documents = retrieve_documents(state["question"])
    sources = list(
        dict.fromkeys(
            str(document.metadata.get("source", "fonte não informada"))
            for document in documents
        )
    )
    return {
        "documents": documents,
        "sources": sources,
    }


def generate_node(state: GraphState) -> GraphState:
    """Gera uma resposta usando os documentos recuperados."""
    documents = state.get("documents", [])
    context = format_documents(documents)
    answer = get_generation_chain().invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )
    return {"answer": answer}


def safety_node(state: GraphState) -> GraphState:
    """Aplica alertas determinísticos sem substituir a avaliação humana."""
    severe_pressure, symptoms = detect_safety_alerts(state["question"])
    answer = state.get("answer", "")
    alerts = symptoms.copy()

    if severe_pressure:
        alerts.insert(0, "pressão arterial muito elevada")

    if symptoms:
        safety_status: SafetyStatus = "emergencia"
        prefix = (
            "ALERTA DE SEGURANÇA: foram informados sinais potencialmente "
            "compatíveis com urgência clínica. Procure avaliação médica imediata "
            "ou um serviço de emergência. Uma nova aferição não deve atrasar o "
            "atendimento.\n\n"
        )
        answer = prefix + answer
    elif severe_pressure:
        safety_status = "atencao"
        prefix = (
            "ATENÇÃO: foi informado um valor de pressão arterial muito elevado. "
            "É necessária avaliação profissional rápida. Na presença de dor "
            "torácica, falta de ar, alterações neurológicas ou visuais, procure "
            "imediatamente um serviço de emergência.\n\n"
        )
        answer = prefix + answer
    else:
        safety_status = "revisao_obrigatoria"

    return {
        "answer": answer,
        "safety_alerts": alerts,
        "safety_status": safety_status,
        "requires_human_review": True,
    }


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Monta e compila o fluxo do CardioAssist AI."""
    builder = StateGraph(
        GraphState,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("safety", safety_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "safety")
    builder.add_edge("safety", END)

    return builder.compile()


def run_graph(question: str) -> GraphOutput:
    """Executa o fluxo completo e devolve somente a saída pública."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    result = get_graph().invoke({"question": normalized_question})
    return cast(GraphOutput, result)
