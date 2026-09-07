"""Testes dos endpoints de manutenção do prontuário.

Os testes trocam a sessão do banco por um SQLite temporário, então rodam sem
PostgreSQL e não tocam nos dados de demonstração.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.prontuario import obter_sessao
from app.database.models import Base
from app.database.seed import ler_prontuarios, popular_banco
from app.main import app

PACIENTE_NOVO = {
    "codigo": "PAC-100",
    "idade": 66,
    "sexo": "feminino",
    "condicoes_cronicas": "Estenose aórtica moderada",
    "alergias": "Nenhuma alergia conhecida",
    "exames": [{"nome": "Ecocardiograma transtorácico"}],
    "medicacoes": [
        {"nome": "Bisoprolol", "dose": "5 mg", "frequencia": "1 vez ao dia"}
    ],
    "atendimentos": [
        {
            "data": "2026-09-01",
            "queixa": "Cansaço aos esforços.",
            "evolucao": "Solicitado ecocardiograma.",
            "profissional": "Cardiologista",
        }
    ],
}


@pytest.fixture
def client():
    """Cliente da API ligado a um banco temporário com os dados sintéticos."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        popular_banco(session, ler_prontuarios())

    def sessao_de_teste():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[obter_sessao] = sessao_de_teste
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_lista_os_pacientes_da_carga_inicial(client) -> None:
    """A listagem devolve os pacientes sintéticos já cadastrados."""
    resposta = client.get("/pacientes")

    assert resposta.status_code == 200
    assert [p["codigo"] for p in resposta.json()] == [
        "PAC-001",
        "PAC-002",
        "PAC-003",
        "PAC-004",
        "PAC-005",
        "PAC-006",
    ]


def test_detalha_o_prontuario_completo(client) -> None:
    """O detalhe traz exames, medicações e atendimentos juntos."""
    corpo = client.get("/pacientes/PAC-003").json()

    assert corpo["idade"] == 71
    assert len(corpo["exames"]) == 4
    assert len(corpo["medicacoes"]) == 4
    assert len(corpo["atendimentos"]) == 2


def test_cadastra_paciente_e_normaliza_o_codigo(client) -> None:
    """O código é gravado em maiúsculas, como o resto do sistema espera."""
    resposta = client.post("/pacientes", json=PACIENTE_NOVO | {"codigo": "pac-100"})

    assert resposta.status_code == 201
    assert resposta.json()["codigo"] == "PAC-100"
    assert client.get("/pacientes/PAC-100").status_code == 200


def test_recusa_codigo_ja_existente(client) -> None:
    """Dois pacientes com o mesmo código quebrariam a busca do assistente."""
    client.post("/pacientes", json=PACIENTE_NOVO)

    resposta = client.post("/pacientes", json=PACIENTE_NOVO)

    assert resposta.status_code == 409


def test_responde_404_para_paciente_desconhecido(client) -> None:
    """Código inexistente devolve erro claro, não erro interno."""
    assert client.get("/pacientes/PAC-999").status_code == 404


def test_conclui_exame_pendente(client) -> None:
    """Registrar o resultado tira o exame da lista de pendentes."""
    criado = client.post("/pacientes", json=PACIENTE_NOVO).json()
    exame_id = criado["exames"][0]["id"]

    resposta = client.patch(
        f"/pacientes/PAC-100/exames/{exame_id}",
        json={"resultado": "Gradiente médio de 28 mmHg", "data": "2026-09-05"},
    )

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "concluido"
    assert "- nenhum exame pendente" in client.get("/pacientes/PAC-100/resumo").json()["resumo"]


def test_recusa_exame_de_outro_paciente(client) -> None:
    """O exame precisa pertencer ao paciente informado na URL."""
    exame_de_outro = client.get("/pacientes/PAC-002").json()["exames"][0]["id"]

    resposta = client.patch(
        f"/pacientes/PAC-004/exames/{exame_de_outro}",
        json={"resultado": "Normal", "data": "2026-09-05"},
    )

    assert resposta.status_code == 404


def test_recusa_exame_concluido_sem_resultado(client) -> None:
    """Um exame concluído sem resultado deixaria o prontuário incoerente."""
    resposta = client.post(
        "/pacientes/PAC-001/exames",
        json={"nome": "Hemograma", "status": "concluido"},
    )

    assert resposta.status_code == 422


def test_recusa_exame_pendente_com_resultado(client) -> None:
    """O contrário também: resultado preenchido exige status concluído."""
    resposta = client.post(
        "/pacientes/PAC-001/exames",
        json={"nome": "Hemograma", "status": "pendente", "resultado": "Normal"},
    )

    assert resposta.status_code == 422


def test_recusa_dado_pessoal_no_cadastro(client) -> None:
    """O banco do protótipo precisa continuar só com dados sintéticos."""
    resposta = client.post(
        "/pacientes",
        json=PACIENTE_NOVO | {"alergias": "Contato do filho: (11) 98765-4321"},
    )

    assert resposta.status_code == 422
    assert "dados pessoais" in resposta.json()["detail"]


def test_recusa_dado_pessoal_no_atendimento(client) -> None:
    """A regra vale também para os textos livres do histórico."""
    resposta = client.post(
        "/pacientes/PAC-001/atendimentos",
        json={
            "data": "2026-09-06",
            "queixa": "Paciente de CPF 123.456.789-00 com dor torácica.",
            "evolucao": "Encaminhado.",
            "profissional": "Cardiologista",
        },
    )

    assert resposta.status_code == 422


def test_inclui_medicacao_no_prontuario(client) -> None:
    """Medicação nova aparece no resumo usado pelo assistente."""
    resposta = client.post(
        "/pacientes/PAC-004/medicacoes",
        json={"nome": "Espironolactona", "dose": "25 mg", "frequencia": "1 vez ao dia"},
    )

    assert resposta.status_code == 201
    assert "Espironolactona" in client.get("/pacientes/PAC-004/resumo").json()["resumo"]


def test_exclui_paciente_e_o_prontuario_junto(client) -> None:
    """A exclusão precisa levar exames, medicações e atendimentos."""
    client.post("/pacientes", json=PACIENTE_NOVO)

    assert client.delete("/pacientes/PAC-100").status_code == 204
    assert client.get("/pacientes/PAC-100").status_code == 404


def test_recusa_idade_invalida(client) -> None:
    """Idade fora da faixa aceitável é erro de entrada."""
    resposta = client.post("/pacientes", json=PACIENTE_NOVO | {"idade": 200})

    assert resposta.status_code == 422
