"""Ativação do LangSmith, que registra as cadeias do LangChain passo a passo.

As variáveis do LangSmith já chegam ao ambiente pelo arquivo .env, e o LangChain
as lê sozinho. O problema é que, com a chave de exemplo que vem no .env.example,
o envio falha em silêncio e nada indica se o rastreamento está ligado.

Este módulo confere a configuração antes de deixar o rastreamento ativo e diz no
log o que foi decidido.
"""

import logging
import os

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Valor de exemplo que acompanha o .env.example e não serve para autenticar.
PREFIXO_DE_CHAVE_DE_EXEMPLO = "substitua-"


def chave_e_valida(api_key: str) -> bool:
    """Confere se a chave foi realmente preenchida pelo usuário."""
    return bool(api_key) and not api_key.startswith(PREFIXO_DE_CHAVE_DE_EXEMPLO)


def configurar_langsmith(settings: Settings | None = None) -> bool:
    """Liga ou desliga o rastreamento e devolve se ele ficou ativo."""
    current_settings = settings or get_settings()

    if not current_settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info(
            "LangSmith desativado",
            extra={"evento": "langsmith_desativado"},
        )
        return False

    if not chave_e_valida(current_settings.langsmith_api_key):
        # Sem desligar aqui, toda chamada tentaria enviar e falharia em silêncio.
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.warning(
            "LangSmith pedido, mas LANGSMITH_API_KEY não foi configurada. "
            "O rastreamento foi desligado para evitar falhas silenciosas.",
            extra={"evento": "langsmith_sem_chave"},
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = current_settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = current_settings.langsmith_project

    logger.info(
        "LangSmith ativado",
        extra={
            "evento": "langsmith_ativado",
            "langsmith_project": current_settings.langsmith_project,
        },
    )
    return True
