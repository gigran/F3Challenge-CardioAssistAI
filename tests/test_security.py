"""Testes da recusa de dados pessoais e das configurações de segurança."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.main import app
from app.observability.tracing import chave_e_valida, configurar_langsmith
from app.security.personal_data import encontrar_dados_pessoais

client = TestClient(app)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("Paciente com CPF 123.456.789-00.", ["CPF"]),
        ("Contato pelo e-mail teste@hospital.com.", ["e-mail"]),
        ("Telefone (11) 98765-4321 para retorno.", ["telefone"]),
        ("Reside no CEP 01310-100.", ["CEP"]),
        ("CPF 12345678900 e fone 11987654321.", ["CPF", "telefone"]),
    ],
)
def test_encontra_identificadores_diretos(texto, esperado) -> None:
    """Identificadores objetivos precisam ser reconhecidos."""
    assert encontrar_dados_pessoais(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "Paciente com pressão 190/125 e dor torácica.",
        "Frequência cardíaca 88 bpm e creatinina 1,9 mg/dL.",
        "Fração de ejeção de 32% no ecocardiograma.",
        "Losartana 50 mg 2 vezes ao dia.",
    ],
)
def test_nao_confunde_valores_clinicos_com_dados_pessoais(texto) -> None:
    """Um falso positivo aqui bloquearia perguntas clínicas legítimas."""
    assert encontrar_dados_pessoais(texto) == []


def test_api_recusa_pergunta_com_dado_pessoal() -> None:
    """Com USE_SYNTHETIC_DATA_ONLY ligado, a API não aceita dado pessoal."""
    response = client.post(
        "/assist",
        json={"question": "Paciente de CPF 123.456.789-00 com dor torácica."},
    )

    assert response.status_code == 422
    assert "somente informações sintéticas" in response.json()["detail"]


def test_api_nao_repete_o_dado_pessoal_na_resposta_de_erro() -> None:
    """A mensagem de erro informa o tipo, nunca o valor encontrado."""
    response = client.post(
        "/assist",
        json={"question": "Paciente de CPF 123.456.789-00 com dor torácica."},
    )

    assert "123.456.789-00" not in response.text


def test_api_aceita_pergunta_quando_a_regra_esta_desligada(monkeypatch) -> None:
    """Desligar a configuração devolve o comportamento anterior."""
    monkeypatch.setattr(
        main_module,
        "settings",
        replace(get_settings(), use_synthetic_data_only=False),
    )
    monkeypatch.setattr(
        main_module,
        "run_graph",
        lambda _question, _llm_provider, _patient_code: {
            "answer": "Resposta.",
            "sources": [],
            "safety_alerts": [],
            "safety_status": "revisao_obrigatoria",
            "requires_human_review": True,
        },
    )

    response = client.post(
        "/assist",
        json={"question": "Paciente de CPF 123.456.789-00 com dor torácica."},
    )

    assert response.status_code == 200


def test_chave_de_exemplo_do_langsmith_nao_e_valida() -> None:
    """A chave que vem no .env.example não pode ser tratada como real."""
    assert chave_e_valida("substitua-pela-sua-chave-no-arquivo-env") is False
    assert chave_e_valida("") is False
    assert chave_e_valida("lsv2_pt_chave_de_verdade") is True


def test_langsmith_fica_desligado_sem_chave_valida() -> None:
    """Pedir rastreamento sem chave desliga, em vez de falhar em silêncio."""
    settings = replace(
        get_settings(),
        langsmith_tracing=True,
        langsmith_api_key="substitua-pela-sua-chave-no-arquivo-env",
    )

    assert configurar_langsmith(settings) is False


def test_langsmith_liga_com_chave_valida(monkeypatch) -> None:
    """Com tracing pedido e chave preenchida, o rastreamento fica ativo."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    settings = replace(
        get_settings(),
        langsmith_tracing=True,
        langsmith_api_key="lsv2_pt_chave_de_verdade",
        langsmith_project="cardioassist-teste",
    )

    assert configurar_langsmith(settings) is True


def test_revisao_humana_continua_obrigatoria_em_alerta(monkeypatch) -> None:
    """Desligar a configuração não pode furar o piso de segurança."""
    import app.graph as graph_module

    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: replace(get_settings(), require_human_review=False),
    )

    emergencia = graph_module.safety_node(
        {"question": "Paciente com dor torácica.", "answer": "Resposta."}
    )
    rotina = graph_module.safety_node(
        {"question": "O que é hipertensão?", "answer": "Resposta."}
    )

    assert emergencia["safety_status"] == "emergencia"
    assert emergencia["requires_human_review"] is True
    assert rotina["requires_human_review"] is False
