"""Testes do rastreamento e da auditoria."""

import logging

import pytest

from app.logger import monitor_node_execution
from app.observability.audit import (
    TAMANHO_DA_PREVIA,
    _resumir,
    iniciar_rastreio,
    obter_trace_id,
    registrar_falha,
    registrar_pergunta,
    registrar_resposta,
)
from app.observability.setup import FormatadorComContexto


def _registro(mensagem: str, **contexto) -> logging.LogRecord:
    """Monta um registro de log com campos de contexto, como o logging faz."""
    record = logging.LogRecord(
        name="teste",
        level=logging.INFO,
        pathname="teste.py",
        lineno=1,
        msg=mensagem,
        args=(),
        exc_info=None,
    )
    for campo, valor in contexto.items():
        setattr(record, campo, valor)
    return record


def test_formatador_acrescenta_os_campos_de_contexto() -> None:
    """Era o bug do projeto: os campos de "extra" não chegavam ao arquivo."""
    formatador = FormatadorComContexto("%(message)s")

    saida = formatador.format(
        _registro("Nó concluído", node="retrieve_node", status="success")
    )

    assert saida == "Nó concluído | node=retrieve_node | status=success"


def test_formatador_nao_quebra_sem_campos_de_contexto() -> None:
    """Uma mensagem comum, de qualquer biblioteca, não pode derrubar o log."""
    formatador = FormatadorComContexto("%(message)s")

    assert formatador.format(_registro("Mensagem simples")) == "Mensagem simples"


def test_rastreio_gera_identificador_por_consulta() -> None:
    """Cada consulta recebe um identificador próprio."""
    primeiro = iniciar_rastreio()
    segundo = iniciar_rastreio()

    assert primeiro != segundo
    assert obter_trace_id() == segundo


def test_previa_junta_linhas_e_corta_texto_longo() -> None:
    """A resposta vira uma linha só, para não quebrar o arquivo de log."""
    assert _resumir("linha um\nlinha dois") == "linha um linha dois"

    resumo = _resumir("a" * (TAMANHO_DA_PREVIA + 50))

    assert resumo.endswith("...")
    assert len(resumo) == TAMANHO_DA_PREVIA + 3


def test_registra_a_pergunta_recebida(caplog: pytest.LogCaptureFixture) -> None:
    """A entrada da consulta precisa aparecer na trilha de auditoria."""
    with caplog.at_level(logging.INFO, logger="app.observability.audit"):
        registrar_pergunta("Quais os sinais de alerta?", "ollama", "PAC-006")

    registro = caplog.records[-1]

    assert registro.evento == "consulta_recebida"
    assert registro.patient_code == "PAC-006"
    assert registro.llm_provider == "ollama"


def test_registra_sem_paciente_com_marcador(caplog: pytest.LogCaptureFixture) -> None:
    """Consulta sem paciente registra um traço, em vez de campo vazio."""
    with caplog.at_level(logging.INFO, logger="app.observability.audit"):
        registrar_pergunta("Pergunta geral", "ollama", "")

    assert caplog.records[-1].patient_code == "-"


def test_registra_a_resposta_com_os_dados_de_auditoria(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """O que o PDF pede para auditoria: fontes, status e revisão humana."""
    resultado = {
        "answer": "Resposta do assistente.",
        "sources": ["prontuario:PAC-006", "data/knowledge_base/hipertensao.md"],
        "safety_status": "emergencia",
        "safety_alerts": ["dor torácica"],
        "requires_human_review": True,
        "patient_found": True,
        "pending_exams": ["Polissonografia"],
    }

    with caplog.at_level(logging.INFO, logger="app.observability.audit"):
        registrar_resposta(resultado)

    registro = caplog.records[-1]

    assert registro.evento == "consulta_respondida"
    assert registro.safety_status == "emergencia"
    assert registro.sources == resultado["sources"]
    assert registro.pending_exams == ["Polissonografia"]
    assert registro.requires_human_review is True
    assert registro.answer_length == len("Resposta do assistente.")


def test_registra_falha_da_consulta(caplog: pytest.LogCaptureFixture) -> None:
    """Uma consulta que falha também precisa deixar rastro."""
    with caplog.at_level(logging.ERROR, logger="app.observability.audit"):
        registrar_falha(ValueError("banco indisponível"))

    registro = caplog.records[-1]

    assert registro.evento == "consulta_falhou"
    assert registro.error == "banco indisponível"


def test_decorador_registra_duracao_e_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """O decorador dos nós grava nome, status e tempo de execução."""

    @monitor_node_execution
    def no_de_teste(state: dict) -> dict:
        return {"resultado": state["entrada"]}

    with caplog.at_level(logging.INFO, logger="app.logger"):
        assert no_de_teste({"entrada": 1}) == {"resultado": 1}

    conclusao = caplog.records[-1]

    assert conclusao.node == "no_de_teste"
    assert conclusao.status == "success"
    assert conclusao.execution_time >= 0


def test_decorador_registra_erro_e_propaga(caplog: pytest.LogCaptureFixture) -> None:
    """Um nó que falha registra o erro antes de o fluxo interromper."""

    @monitor_node_execution
    def no_que_falha(state: dict) -> dict:
        raise RuntimeError("falha no nó")

    with (
        caplog.at_level(logging.ERROR, logger="app.logger"),
        pytest.raises(RuntimeError),
    ):
        no_que_falha({})

    registro = caplog.records[-1]

    assert registro.status == "error"
    assert registro.error == "falha no nó"
