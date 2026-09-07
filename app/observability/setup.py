"""Configuração central do logging.

O projeto registrava informações extras (nó, status, tempo de execução) usando o
parâmetro "extra" do logging, mas elas não apareciam no arquivo: um formato fixo
com esses campos derruba qualquer mensagem que não os traga, com KeyError.

A solução aqui é um formatador que acrescenta os campos de contexto apenas
quando eles existem no registro, e um filtro que carimba o identificador de
rastreio em todas as linhas.
"""

import logging
from logging.handlers import RotatingFileHandler

from app.config import PROJECT_ROOT

LOG_DIRECTORY = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIRECTORY / "cardioassist.log"

MAX_LOG_FILE_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3

BASE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Campos que os nós e a auditoria enviam pelo parâmetro "extra" do logging.
CAMPOS_DE_CONTEXTO = (
    "trace_id",
    "evento",
    "node",
    "status",
    "execution_time",
    "patient_code",
    "patient_found",
    "llm_provider",
    "safety_status",
    "safety_alerts",
    "pending_exams",
    "sources",
    "answer_length",
    "answer_preview",
    "requires_human_review",
    "langsmith_project",
    "dados_pessoais",
    "error",
)


class FormatadorComContexto(logging.Formatter):
    """Acrescenta à mensagem os campos de contexto presentes no registro."""

    def format(self, record: logging.LogRecord) -> str:
        mensagem = super().format(record)

        contexto = [
            f"{campo}={getattr(record, campo)}"
            for campo in CAMPOS_DE_CONTEXTO
            if hasattr(record, campo)
        ]

        if not contexto:
            return mensagem

        return mensagem + " | " + " | ".join(contexto)


class FiltroDeRastreio(logging.Filter):
    """Carimba o identificador de rastreio em todas as linhas de log."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Importado aqui dentro para evitar dependência circular entre os
        # módulos de configuração e de auditoria.
        from app.observability.audit import obter_trace_id

        if not hasattr(record, "trace_id"):
            record.trace_id = obter_trace_id()

        return True


def configurar_logging(level: int = logging.INFO) -> None:
    """Configura o logging da aplicação. Chamar mais de uma vez não duplica."""
    root_logger = logging.getLogger()

    if getattr(root_logger, "_cardioassist_configurado", False):
        return

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    formatador = FormatadorComContexto(BASE_FORMAT)
    filtro = FiltroDeRastreio()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatador)
    console_handler.addFilter(filtro)

    # A rotação evita que o arquivo cresça sem limite durante as demonstrações.
    arquivo_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_FILE_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    arquivo_handler.setFormatter(formatador)
    arquivo_handler.addFilter(filtro)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(arquivo_handler)
    root_logger._cardioassist_configurado = True
