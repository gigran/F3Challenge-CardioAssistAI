"""Testes dos nós do fluxo que consultam o prontuário.

Os nós abrem a própria sessão do banco, então os testes trocam a função
get_session por uma que aponta para um SQLite temporário.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.graph as graph_module
from app.database.models import Base
from app.database.seed import ler_prontuarios, popular_banco
from app.graph import get_graph
from app.nodes import prontuario
from app.nodes.prontuario import (
    COM_PACIENTE,
    SEM_PACIENTE,
    exames_pendentes_node,
    prontuario_node,
    rotear_por_paciente,
)


@pytest.fixture
def banco(monkeypatch):
    """Aponta os nós para um banco temporário com os pacientes sintéticos.

    O StaticPool mantém a mesma conexão viva: sem ele, cada nova sessão abriria
    um banco em memória vazio.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        popular_banco(session, ler_prontuarios())

    monkeypatch.setattr(prontuario, "get_session", lambda: Session(engine))


def test_roteia_para_o_prontuario_quando_ha_paciente() -> None:
    """Com código informado, o fluxo passa pelas etapas do prontuário."""
    assert rotear_por_paciente({"patient_code": "PAC-001"}) == COM_PACIENTE


@pytest.mark.parametrize("estado", [{"patient_code": ""}, {"patient_code": "   "}, {}])
def test_roteia_direto_para_a_base_sem_paciente(estado) -> None:
    """Sem código, o assistente responde apenas com a base de conhecimento."""
    assert rotear_por_paciente(estado) == SEM_PACIENTE


def test_no_do_prontuario_devolve_resumo_e_fonte(banco) -> None:
    """O prontuário consultado precisa aparecer nas fontes da resposta."""
    resultado = prontuario_node({"patient_code": "pac-006"})

    assert resultado["patient_found"] is True
    assert resultado["sources"] == ["prontuario:PAC-006"]
    assert "Paciente PAC-006" in resultado["patient_summary"]


def test_no_do_prontuario_nao_quebra_com_codigo_desconhecido(banco) -> None:
    """Código inexistente vira aviso no resumo, não erro no fluxo."""
    resultado = prontuario_node({"patient_code": "PAC-999"})

    assert resultado["patient_found"] is False
    assert resultado["sources"] == []
    assert resultado["patient_summary"] == (
        "Nenhum paciente encontrado com o código PAC-999."
    )


def test_no_de_exames_lista_apenas_os_pendentes(banco) -> None:
    """Etapa citada no desafio: verificar exames pendentes do paciente."""
    resultado = exames_pendentes_node({"patient_code": "PAC-006"})

    assert resultado["pending_exams"] == [
        "Ecocardiograma transtorácico",
        "Polissonografia",
    ]


def test_no_de_exames_devolve_lista_vazia_sem_pendencias(banco) -> None:
    """O PAC-004 está com todos os exames concluídos."""
    assert exames_pendentes_node({"patient_code": "PAC-004"}) == {"pending_exams": []}


def test_no_de_exames_devolve_lista_vazia_para_paciente_desconhecido(banco) -> None:
    """Um código inválido não pode gerar erro no meio do fluxo."""
    assert exames_pendentes_node({"patient_code": "PAC-999"}) == {"pending_exams": []}


def test_grafo_registra_as_novas_etapas() -> None:
    """Confirma que o fluxo compilado contém os nós do prontuário."""
    nos = get_graph().get_graph().nodes

    assert "prontuario" in nos
    assert "exames_pendentes" in nos


def test_grafo_tem_ramificacao_condicional_na_entrada() -> None:
    """A entrada do fluxo decide entre consultar ou não o prontuário."""
    arestas = get_graph().get_graph().edges

    condicionais = {
        aresta.data: aresta.target
        for aresta in arestas
        if aresta.source == "__start__" and aresta.conditional
    }

    assert condicionais == {
        COM_PACIENTE: "prontuario",
        SEM_PACIENTE: "retrieve",
    }


def test_avisa_quando_o_codigo_informado_nao_existe() -> None:
    """O aviso é determinístico: não depende de a LLM mencionar a falha."""
    resultado = graph_module.safety_node(
        {
            "question": "Quais os sinais de alerta?",
            "answer": "Resposta baseada na base de conhecimento.",
            "patient_code": "PAC-999",
            "patient_found": False,
        }
    )

    assert resultado["answer"].startswith(graph_module.AVISO_PACIENTE_NAO_ENCONTRADO)
    # A resposta continua sendo entregue, apenas com a limitação declarada.
    assert "Resposta baseada na base de conhecimento." in resultado["answer"]


def test_nao_avisa_quando_o_paciente_foi_encontrado() -> None:
    """Paciente válido não pode receber aviso de ausência."""
    resultado = graph_module.safety_node(
        {
            "question": "Quais os sinais de alerta?",
            "answer": "Resposta.",
            "patient_code": "PAC-006",
            "patient_found": True,
        }
    )

    assert graph_module.AVISO_PACIENTE_NAO_ENCONTRADO not in resultado["answer"]


def test_nao_avisa_quando_nenhum_codigo_foi_informado() -> None:
    """Pergunta geral, sem paciente, é caso legítimo e não gera aviso."""
    resultado = graph_module.safety_node(
        {"question": "O que é fibrilação atrial?", "answer": "Resposta."}
    )

    assert graph_module.AVISO_PACIENTE_NAO_ENCONTRADO not in resultado["answer"]


def test_alerta_de_emergencia_vem_antes_do_aviso_de_paciente() -> None:
    """O que exige ação imediata precisa ser a primeira coisa lida."""
    resultado = graph_module.safety_node(
        {
            "question": "Paciente com dor torácica.",
            "answer": "Resposta.",
            "patient_code": "PAC-999",
            "patient_found": False,
        }
    )

    posicao_emergencia = resultado["answer"].index("ALERTA DE SEGURANÇA")
    posicao_aviso = resultado["answer"].index("AVISO: o código de paciente")

    assert posicao_emergencia < posicao_aviso
