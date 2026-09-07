"""Testes das consultas ao prontuário eletrônico.

Os testes usam um banco SQLite em memória, criado e descartado a cada teste,
para que não seja preciso ter o PostgreSQL em execução.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.models import Base
from app.database.repositories import (
    buscar_paciente,
    listar_atendimentos,
    listar_exames_pendentes,
    listar_medicacoes_ativas,
    listar_pacientes,
    montar_resumo_prontuario,
)
from app.database.seed import ler_prontuarios, popular_banco


@pytest.fixture
def session():
    """Cria um banco temporário já carregado com os prontuários sintéticos."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        popular_banco(session, ler_prontuarios())
        yield session


def test_carrega_todos_os_pacientes_sinteticos(session) -> None:
    """Confirma que o arquivo de dados sintéticos é lido por completo."""
    pacientes = listar_pacientes(session)

    assert len(pacientes) == 6
    assert pacientes[0].codigo == "PAC-001"


def test_busca_paciente_ignorando_maiusculas(session) -> None:
    """O código pode ser digitado de qualquer forma pelo profissional."""
    paciente = buscar_paciente(session, " pac-003 ")

    assert paciente is not None
    assert paciente.idade == 71


def test_devolve_none_para_paciente_inexistente(session) -> None:
    """Um código desconhecido não pode gerar erro, apenas ausência de dados."""
    assert buscar_paciente(session, "PAC-999") is None


def test_lista_apenas_exames_sem_resultado(session) -> None:
    """O fluxo do assistente precisa saber quais exames continuam pendentes."""
    paciente = buscar_paciente(session, "PAC-002")

    pendentes = listar_exames_pendentes(session, paciente.id)

    assert [exame.nome for exame in pendentes] == ["Teste ergométrico", "Troponina I"]
    assert all(exame.resultado is None for exame in pendentes)


def test_ignora_medicacoes_suspensas(session) -> None:
    """O paciente PAC-001 tem uma medicação suspensa, que não deve aparecer."""
    paciente = buscar_paciente(session, "PAC-001")

    medicacoes = listar_medicacoes_ativas(session, paciente.id)

    nomes = [medicacao.nome for medicacao in medicacoes]
    assert nomes == ["Atorvastatina", "Losartana"]
    assert "Hidroclorotiazida" not in nomes


def test_lista_atendimentos_do_mais_recente_para_o_mais_antigo(session) -> None:
    """A contextualização precisa priorizar a informação mais atual."""
    paciente = buscar_paciente(session, "PAC-006")

    atendimentos = listar_atendimentos(session, paciente.id)

    datas = [atendimento.data.isoformat() for atendimento in atendimentos]
    assert datas == ["2026-09-03", "2026-08-30"]


def test_respeita_o_limite_de_atendimentos(session) -> None:
    """O resumo não deve crescer sem controle conforme o histórico aumenta."""
    paciente = buscar_paciente(session, "PAC-001")

    atendimentos = listar_atendimentos(session, paciente.id, limite=1)

    assert len(atendimentos) == 1


def test_resumo_reune_as_secoes_do_prontuario(session) -> None:
    """O resumo é o texto que será entregue à LLM como contexto do paciente."""
    resumo = montar_resumo_prontuario(session, "PAC-006")

    assert "Paciente PAC-006 | 59 anos | masculino" in resumo
    assert "Medicações em uso:" in resumo
    assert "Anlodipino 10 mg" in resumo
    assert "Exames pendentes:" in resumo
    assert "Polissonografia" in resumo
    assert "Últimos atendimentos:" in resumo


def test_resumo_avisa_quando_o_paciente_nao_existe(session) -> None:
    """O assistente precisa declarar a limitação em vez de inventar dados."""
    resumo = montar_resumo_prontuario(session, "PAC-999")

    assert resumo == "Nenhum paciente encontrado com o código PAC-999."


def test_resumo_indica_ausencia_de_exames_pendentes(session) -> None:
    """O paciente PAC-004 está com todos os exames concluídos."""
    resumo = montar_resumo_prontuario(session, "PAC-004")

    assert "- nenhum exame pendente" in resumo
