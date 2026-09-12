"""Carregamento centralizado das configurações da aplicação."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_bool(variable_name: str, default: bool) -> bool:
    """Converte uma variável de ambiente textual para booleano."""
    value = os.getenv(variable_name)
    if value is None:
        return default

    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "sim", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "nao", "não", "off"}:
        return False

    raise ValueError(
        f"A variável {variable_name} deve conter um valor booleano válido."
    )


def _read_positive_int(variable_name: str, default: int) -> int:
    """Converte uma variável de ambiente para um número inteiro positivo."""
    raw_value = os.getenv(variable_name, str(default))

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"A variável {variable_name} deve conter um número inteiro."
        ) from error

    if value <= 0:
        raise ValueError(f"A variável {variable_name} deve ser maior que zero.")

    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Configurações utilizadas pelo CardioAssist AI."""

    app_name: str
    app_env: str
    app_host: str
    app_port: int
    llm_provider: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    openai_api_key: str
    openai_chat_model: str
    database_url: str
    rag_top_k: int
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str
    use_synthetic_data_only: bool
    require_human_review: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Carrega, valida e mantém em cache as configurações da aplicação."""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    supported_providers = {"ollama", "lora", "openai"}

    if llm_provider not in supported_providers:
        raise ValueError("LLM_PROVIDER deve ser 'ollama', 'lora' ou 'openai'.")

    return Settings(
        app_name=os.getenv("APP_NAME", "CardioAssist AI"),
        app_env=os.getenv("APP_ENV", "development"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=_read_positive_int("APP_PORT", 8000),
        llm_provider=llm_provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://cardioassist:senha_local@localhost:5432/cardioassist",
        ),
        rag_top_k=_read_positive_int("RAG_TOP_K", 2),
        langsmith_tracing=_read_bool("LANGSMITH_TRACING", False),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "cardioassist-ai"),
        use_synthetic_data_only=_read_bool("USE_SYNTHETIC_DATA_ONLY", True),
        require_human_review=_read_bool("REQUIRE_HUMAN_REVIEW", True),
    )
