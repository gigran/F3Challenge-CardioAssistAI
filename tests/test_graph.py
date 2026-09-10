"""Testes da orquestração e da camada de segurança do LangGraph."""

import pytest
from langchain_core.documents import Document

from app import graph as graph_module
from app.graph import (
    GraphState,
    detect_safety_alerts,
    get_graph,
    run_graph,
    safety_node,
)


@pytest.mark.parametrize(
    "question",
    [
        "Paciente sintético com pressão 190/125.",
        "Paciente sintético com pressão 180 x 110.",
        "Paciente sintético com pressão 170 por 120.",
    ],
)
def test_detecta_pressao_muito_elevada_em_formatos_diferentes(
    question: str,
) -> None:
    """Reconhece pressão sistólica ou diastólica acima do limite de alerta."""
    severe_pressure, symptoms = detect_safety_alerts(question)

    assert severe_pressure is True
    assert symptoms == []


def test_detecta_sintomas_com_e_sem_acentos() -> None:
    """Normaliza o texto antes de procurar sinais de alerta."""
    severe_pressure, symptoms = detect_safety_alerts(
        "Paciente com dor torácica, confusao e visão turva."
    )

    assert severe_pressure is False
    assert symptoms == ["dor torácica", "confusão", "visão turva"]


@pytest.mark.parametrize(
    ("question", "expected_alert"),
    [
        ("Paciente sintético com aperto no peito.", "aperto no peito"),
        ("Paciente sintético com pressão no peito.", "pressão no peito"),
        (
            "Paciente sintético com dor irradiando para o braço.",
            "dor irradiando para o braço",
        ),
        ("Paciente sintético com suor frio.", "suor frio"),
        ("Paciente sintético com palidez e náusea.", "palidez"),
        ("Paciente sintético apresentou vômito.", "vômito"),
        ("Paciente sintético apresentou síncope.", "síncope"),
        (
            "Paciente sintético com dor súbita e intensa irradiando para o dorso.",
            "dor súbita e intensa",
        ),
    ],
)
def test_detecta_sinais_associados_a_dor_toracica(
    question: str,
    expected_alert: str,
) -> None:
    """Reconhece apresentações e sintomas associados a emergências torácicas."""
    severe_pressure, symptoms = detect_safety_alerts(question)

    assert severe_pressure is False
    assert expected_alert in symptoms


def test_classifica_aperto_no_peito_com_suor_frio_como_emergencia() -> None:
    """Não depende da LLM para priorizar sinais compatíveis com urgência."""
    state: GraphState = {
        "question": "Paciente sintético com aperto no peito e suor frio.",
        "answer": "Resposta original do modelo.",
    }

    result = safety_node(state)

    assert result["safety_status"] == "emergencia"
    assert result["safety_alerts"] == ["aperto no peito", "suor frio"]
    assert result["answer"].startswith("ALERTA DE SEGURANÇA")
    assert result["requires_human_review"] is True


def test_classifica_emergencia_e_prioriza_atendimento() -> None:
    """Gera alerta imediato quando a pergunta informa sintomas relevantes."""
    state: GraphState = {
        "question": "Paciente sintético com pressão 190/125 e dor torácica.",
        "answer": "Resposta original do modelo.",
    }

    result = safety_node(state)

    assert result["safety_status"] == "emergencia"
    assert result["safety_alerts"] == [
        "pressão arterial muito elevada",
        "dor torácica",
    ]
    assert result["answer"].startswith("ALERTA DE SEGURANÇA")
    assert "não deve atrasar a conduta" in result["answer"]
    assert result["requires_human_review"] is True


def test_classifica_atencao_quando_apenas_pressao_e_muito_elevada() -> None:
    """Diferencia pressão muito elevada sem sintomas declarados."""
    state: GraphState = {
        "question": "Paciente sintético com pressão 190/125 e sem outros dados.",
        "answer": "Resposta original do modelo.",
    }

    result = safety_node(state)

    assert result["safety_status"] == "atencao"
    assert result["safety_alerts"] == ["pressão arterial muito elevada"]
    assert result["answer"].startswith("ATENÇÃO")
    assert result["requires_human_review"] is True


def test_exige_revisao_mesmo_sem_alertas_detectados() -> None:
    """Mantém revisão humana obrigatória em respostas sem alerta automático."""
    state: GraphState = {
        "question": "Pergunta educacional sobre acompanhamento da hipertensão.",
        "answer": "Resposta original do modelo.",
    }

    result = safety_node(state)

    assert result == {
        "answer": "Resposta original do modelo.",
        "safety_alerts": [],
        "safety_status": "revisao_obrigatoria",
        "requires_human_review": True,
    }


def test_rejeita_pergunta_vazia_antes_de_executar_o_grafo() -> None:
    """Impede a execução do fluxo quando não existe uma pergunta."""
    with pytest.raises(ValueError, match="A pergunta não pode estar vazia"):
        run_graph("   ")


def test_rejeita_provedor_desconhecido_antes_de_executar_o_grafo() -> None:
    """Impede a execução com um provedor que não possui implementação."""
    with pytest.raises(ValueError, match="llm_provider deve ser"):
        run_graph("Pergunta sintética.", "provedor-invalido")


def test_grafo_compilado_contem_os_tres_nos() -> None:
    """Confirma a topologia principal do fluxo."""
    drawable_graph = get_graph().get_graph()

    assert {"retrieve", "generate", "safety"} <= set(drawable_graph.nodes)
    mermaid = drawable_graph.draw_mermaid()
    assert "retrieve" in mermaid
    assert "generate" in mermaid
    assert "safety" in mermaid


def test_executa_fluxo_com_recuperacao_e_geracao_simuladas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executa o LangGraph sem acessar modelos ou serviços externos."""
    document = Document(
        page_content="Contexto sintético sobre hipertensão.",
        metadata={"source": "data/knowledge_base/hipertensao.md"},
    )

    def fake_retrieve_node(_state: GraphState) -> GraphState:
        return {
            "documents": [document],
            "sources": ["data/knowledge_base/hipertensao.md"],
        }

    def fake_generate_node(_state: GraphState) -> GraphState:
        return {"answer": "Resposta simulada pelo teste."}

    def fake_classify_question(_state: GraphState) -> GraphState:
        return {
            "intent": "sintomas",
            "confidence": 1.0,
            "instruction": "Responda esta pergunta sobre os sintomas de uma doença",
        }

    monkeypatch.setattr(graph_module, "classify_question", fake_classify_question)
    monkeypatch.setattr(graph_module, "retrieve_node", fake_retrieve_node)
    monkeypatch.setattr(graph_module, "generate_node", fake_generate_node)
    get_graph.cache_clear()

    try:
        result = run_graph("Paciente sintético com pressão 190/125 e dor torácica.")
    finally:
        get_graph.cache_clear()

    assert result["answer"].startswith("ALERTA DE SEGURANÇA")
    assert "Resposta simulada pelo teste." in result["answer"]
    assert result["sources"] == ["data/knowledge_base/hipertensao.md"]
    assert result["safety_status"] == "emergencia"
    assert result["requires_human_review"] is True
