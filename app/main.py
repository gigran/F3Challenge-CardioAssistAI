"""Ponto de entrada da API do CardioAssist AI."""

from fastapi import FastAPI

from app.config import get_settings
from app.graph import run_graph
from app.schemas import AssistRequest, AssistResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API educacional de apoio à decisão clínica em cardiologia.",
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


@app.post("/assist", response_model=AssistResponse, tags=["Assistência"])
def assist(request: AssistRequest) -> AssistResponse:
    """Executa o fluxo RAG e devolve uma resposta para revisão humana."""
    result = run_graph(request.question)
    return AssistResponse.model_validate(result)
