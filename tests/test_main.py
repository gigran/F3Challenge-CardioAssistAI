"""Testes dos endpoints da API."""

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import app.main as main_module
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


def test_assist_retorna_resposta_do_fluxo_simulado(monkeypatch: MonkeyPatch) -> None:
    """Verifica o contrato do endpoint sem executar Ollama ou OpenAI."""
    graph_result = {
        "answer": "O caso sintético exige avaliação médica imediata.",
        "sources": ["data/knowledge_base/hipertensao.md"],
        "safety_alerts": [
            "pressão arterial muito elevada",
            "dor torácica",
        ],
        "safety_status": "emergencia",
        "requires_human_review": True,
    }

    monkeypatch.setattr(
        main_module,
        "run_graph",
        lambda _question, _llm_provider, _patient_code: graph_result,
    )

    response = client.post(
        "/assist",
        json={
            "question": (
                "Paciente sintético com pressão 190/125 e dor torácica. "
                "Quais são os sinais de alerta?"
            ),
        },
    )

    assert response.status_code == 200
    # Sem código de paciente, os campos do prontuário voltam com os valores padrão.
    assert response.json() == graph_result | {
        "patient_found": False,
        "patient_summary": "",
        "pending_exams": [],
    }


def test_assist_repassa_pergunta_sem_espacos_externos(
    monkeypatch: MonkeyPatch,
) -> None:
    """Confirma que a pergunta é normalizada antes de chegar ao grafo."""
    received_requests: list[tuple[str, str]] = []
    graph_result = {
        "answer": "Resposta simulada.",
        "sources": [],
        "safety_alerts": [],
        "safety_status": "revisao_obrigatoria",
        "requires_human_review": True,
    }

    def fake_run_graph(
        question: str,
        llm_provider: str,
        patient_code: str,
    ) -> dict[str, object]:
        received_requests.append((question, llm_provider))
        return graph_result

    monkeypatch.setattr(main_module, "run_graph", fake_run_graph)

    response = client.post(
        "/assist",
        json={"question": "  Paciente sintético com hipertensão.  "},
    )

    assert response.status_code == 200
    assert received_requests == [("Paciente sintético com hipertensão.", "ollama")]


def test_assist_repassa_provedor_openai_ao_grafo(
    monkeypatch: MonkeyPatch,
) -> None:
    """Encaminha a seleção da interface sem alterar o provedor dos embeddings."""
    received_requests: list[tuple[str, str]] = []
    graph_result = {
        "answer": "Resposta simulada pela OpenAI.",
        "sources": [],
        "safety_alerts": [],
        "safety_status": "revisao_obrigatoria",
        "requires_human_review": True,
    }

    def fake_run_graph(
        question: str,
        llm_provider: str,
        patient_code: str,
    ) -> dict[str, object]:
        received_requests.append((question, llm_provider))
        return graph_result

    monkeypatch.setattr(main_module, "run_graph", fake_run_graph)

    response = client.post(
        "/assist",
        json={
            "question": "Paciente sintético com hipertensão.",
            "llm_provider": "openai",
        },
    )

    assert response.status_code == 200
    assert received_requests == [("Paciente sintético com hipertensão.", "openai")]


def test_assist_rejeita_provedor_desconhecido() -> None:
    """Restringe a seleção aos provedores suportados pela aplicação."""
    response = client.post(
        "/assist",
        json={
            "question": "Paciente sintético com hipertensão.",
            "llm_provider": "provedor-invalido",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "llm_provider"]


def test_assist_rejeita_pergunta_vazia() -> None:
    """Verifica a validação de uma pergunta composta apenas por espaços."""
    response = client.post("/assist", json={"question": "  "})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "question"]


def test_assist_rejeita_campo_desconhecido() -> None:
    """Evita a aceitação silenciosa de campos não previstos no contrato."""
    response = client.post(
        "/assist",
        json={
            "question": "Paciente sintético com hipertensão.",
            "patient_name": "Nome não permitido",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "patient_name"]
