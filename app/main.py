"""Ponto de entrada da API do CardioAssist AI."""

"""python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"""

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API de apoio à decisão clínica em cardiologia.",
    version="0.1.0",
)


@app.get("/", tags=["Geral"])
def read_root() -> dict[str, str]:
    """Apresenta a API e informa o endereço da documentação interativa."""
    return {
        "mensagem": "CardioAssist AI está em execução.",
        "documentacao": "/docs",
    }


@app.get("/health", tags=["Monitoramento"])
def read_health() -> dict[str, str]:
    """Informa se a API iniciou e carregou suas configurações."""
    return {
        "status": "saudável",
        "ambiente": settings.app_env,
        "provedor_llm": settings.llm_provider,
    }
