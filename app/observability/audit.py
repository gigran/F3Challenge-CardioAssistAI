"""Trilha de auditoria das consultas ao assistente.

Cada consulta recebe um identificador de rastreio (trace_id). Ele é guardado em
um ContextVar, que é uma variável isolada por requisição: assim todas as linhas
de log de uma mesma consulta carregam o mesmo identificador, mesmo com várias
requisições acontecendo ao mesmo tempo.
"""

import logging
import uuid
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

# A resposta completa deixaria a linha de log gigante, então o arquivo guarda um
# trecho inicial e o tamanho total.
TAMANHO_DA_PREVIA = 400


def iniciar_rastreio() -> str:
    """Gera um identificador novo para a consulta que está começando."""
    trace_id = uuid.uuid4().hex[:12]
    _trace_id.set(trace_id)
    return trace_id


def obter_trace_id() -> str:
    """Devolve o identificador da consulta atual."""
    return _trace_id.get()


def _resumir(texto: str) -> str:
    """Deixa o texto em uma linha só, para não quebrar o arquivo de log."""
    em_uma_linha = " ".join(texto.split())
    if len(em_uma_linha) <= TAMANHO_DA_PREVIA:
        return em_uma_linha
    return em_uma_linha[:TAMANHO_DA_PREVIA] + "..."


def registrar_pergunta(
    question: str,
    llm_provider: str,
    patient_code: str,
) -> None:
    """Registra a entrada da consulta, antes de o fluxo começar."""
    logger.info(
        "Consulta recebida",
        extra={
            "evento": "consulta_recebida",
            "llm_provider": llm_provider,
            "patient_code": patient_code or "-",
            "answer_preview": _resumir(question),
        },
    )


def registrar_resposta(resultado: dict) -> None:
    """Registra a saída da consulta, com o que é necessário para auditoria."""
    resposta = str(resultado.get("answer", ""))

    logger.info(
        "Consulta respondida",
        extra={
            "evento": "consulta_respondida",
            "safety_status": resultado.get("safety_status", "-"),
            "safety_alerts": resultado.get("safety_alerts", []),
            "requires_human_review": resultado.get("requires_human_review", True),
            "patient_found": resultado.get("patient_found", False),
            "pending_exams": resultado.get("pending_exams", []),
            "sources": resultado.get("sources", []),
            "answer_length": len(resposta),
            "answer_preview": _resumir(resposta),
        },
    )


def registrar_recusa(tipos_encontrados: list[str]) -> None:
    """Registra uma consulta recusada por conter dados pessoais.

    Grava apenas os tipos encontrados, nunca o texto: o log de auditoria não
    pode virar mais um lugar onde o dado pessoal fica guardado.
    """
    logger.warning(
        "Consulta recusada por conter dados pessoais",
        extra={
            "evento": "consulta_recusada",
            "dados_pessoais": tipos_encontrados,
        },
    )


def registrar_falha(erro: Exception) -> None:
    """Registra uma consulta que não pôde ser concluída."""
    logger.exception(
        "Consulta falhou",
        extra={"evento": "consulta_falhou", "error": str(erro)},
    )
