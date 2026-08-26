"""Testes dos endpoints básicos da API."""

from fastapi.testclient import TestClient

from app.main import app, settings

client = TestClient(app)


def test_root_retorna_apresentacao_da_api() -> None:
    """Verifica a mensagem inicial e o caminho da documentação."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "mensagem": "CardioAssist AI está em execução.",
        "documentacao": "/docs",
    }


def test_health_retorna_configuracoes_basicas() -> None:
    """Verifica a saúde da API e as configurações públicas esperadas."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "saudável",
        "ambiente": settings.app_env,
        "provedor_llm": settings.llm_provider,
    }
