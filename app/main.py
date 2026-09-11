"""Ponto de entrada da API do CardioAssist AI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.api.prontuario import router as prontuario_router
from app.config import get_settings
from app.database.seed import criar_dados_sinteticos
from app.graph import run_graph
from app.observability.audit import (
    iniciar_rastreio,
    registrar_falha,
    registrar_pergunta,
    registrar_recusa,
    registrar_resposta,
)
from app.observability.tracing import configurar_langsmith
from app.schemas import AssistRequest, AssistResponse
from app.security.personal_data import encontrar_dados_pessoais

settings = get_settings()
configurar_langsmith(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Prepara o banco antes de aceitar requisições."""
    if settings.use_synthetic_data_only:
        criar_dados_sinteticos()

    yield


app = FastAPI(
    title=settings.app_name,
    description="API educacional de apoio à decisão clínica em cardiologia.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(prontuario_router)


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
    iniciar_rastreio()

    if settings.use_synthetic_data_only:
        dados_pessoais = encontrar_dados_pessoais(request.question)
        if dados_pessoais:
            registrar_recusa(dados_pessoais)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "A pergunta parece conter dados pessoais "
                    f"({', '.join(dados_pessoais)}). Este protótipo aceita "
                    "somente informações sintéticas."
                ),
            )

    registrar_pergunta(
        request.question,
        request.llm_provider.value,
        request.patient_code,
    )

    try:
        result = run_graph(
            request.question,
            request.llm_provider.value,
            request.patient_code,
        )
    except Exception as erro:
        registrar_falha(erro)
        raise

    registrar_resposta(result)
    return AssistResponse.model_validate(result)
