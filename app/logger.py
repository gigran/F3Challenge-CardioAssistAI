"""Instrumentação dos nós do LangGraph.

O decorador registra início, fim, duração e resultado de cada nó do fluxo.
Os campos enviados em "extra" são gravados pelo formatador configurado em
app/observability/setup.py.
"""

import logging
import time
from functools import wraps

from app.observability.setup import configurar_logging

configurar_logging()

logger = logging.getLogger(__name__)


def monitor_node_execution(func):
    @wraps(func)
    def wrapper(state: dict) -> dict:
        start_time = time.time()
        node_name = func.__name__

        logger.info(
            f"Iniciando {node_name}",
            extra={
                "evento": "no_iniciado",
                "node": node_name,
            },
        )

        try:
            result = func(state)
            execution_time = time.time() - start_time

            logger.info(
                f"{node_name} concluído com sucesso",
                extra={
                    "evento": "no_concluido",
                    "node": node_name,
                    "status": "success",
                    "execution_time": round(execution_time, 3),
                },
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Erro em {node_name}",
                extra={
                    "evento": "no_com_erro",
                    "node": node_name,
                    "status": "error",
                    "execution_time": round(execution_time, 3),
                    "error": str(e),
                },
            )
            raise

    return wrapper
