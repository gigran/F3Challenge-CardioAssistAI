"""Testes da geração contextualizada de respostas."""

import os
from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app import generation
from app.config import get_settings
from app.generation import (
    SYSTEM_PROMPT,
    create_chat_model,
    format_documents,
    generate_answer,
)


def test_formata_documentos_com_conteudo_e_fonte() -> None:
    """Verifica a formatação do contexto enviado ao modelo."""
    documents = [
        Document(
            page_content="Trecho clínico sintético.",
            metadata={
                "titulo": "Documento de teste",
                "source": "data/knowledge_base/teste.md",
            },
        )
    ]

    context = format_documents(documents)

    assert "Trecho 1" in context
    assert "Documento de teste" in context
    assert "data/knowledge_base/teste.md" in context
    assert "Trecho clínico sintético." in context


def test_formata_lista_vazia_com_mensagem_explicita() -> None:
    """Verifica o contexto produzido quando nenhum documento é recuperado."""
    assert format_documents([]) == "Nenhum contexto foi recuperado."


def test_prompt_contem_regras_minimas_de_seguranca() -> None:
    """Confirma a presença das principais restrições no prompt de sistema."""
    normalized_prompt = SYSTEM_PROMPT.lower()

    assert "use somente" in normalized_prompt
    assert "não invente" in normalized_prompt
    assert "não prescreva" in normalized_prompt
    assert "sinais de alerta" in normalized_prompt
    assert "validação de um profissional" in normalized_prompt


def test_rejeita_openai_sem_chave_configurada() -> None:
    """Impede a criação do modelo OpenAI sem uma chave válida."""
    settings = replace(
        get_settings(),
        llm_provider="openai",
        openai_api_key="",
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY deve ser configurada"):
        create_chat_model(settings)


def test_rejeita_pergunta_vazia_sem_recuperar_documentos() -> None:
    """Impede a geração quando a pergunta não possui conteúdo."""
    with pytest.raises(ValueError, match="A pergunta não pode estar vazia"):
        generate_answer("   ")


def test_gera_resposta_estruturada_com_fontes_unicas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Testa recuperação e geração com uma cadeia simulada."""
    documents = [
        Document(
            page_content="Primeiro trecho sobre sinais de alerta.",
            metadata={"source": "data/knowledge_base/hipertensao.md"},
        ),
        Document(
            page_content="Segundo trecho da mesma fonte.",
            metadata={"source": "data/knowledge_base/hipertensao.md"},
        ),
    ]
    received_input: dict[str, str] = {}

    def fake_generation(values: dict[str, str]) -> str:
        received_input.update(values)
        return "Resposta simulada com validação profissional."

    fake_chain = RunnableLambda(fake_generation)
    monkeypatch.setattr(generation, "retrieve_documents", lambda _question: documents)
    monkeypatch.setattr(
        generation,
        "get_generation_chain",
        lambda _llm_provider: fake_chain,
    )

    result = generate_answer("Quais são os sinais de alerta?", "ollama")

    assert received_input["question"] == "Quais são os sinais de alerta?"
    assert "Primeiro trecho" in received_input["context"]
    assert "Segundo trecho" in received_input["context"]
    assert result == {
        "answer": "Resposta simulada com validação profissional.",
        "sources": ["data/knowledge_base/hipertensao.md"],
        "requires_human_review": True,
    }


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS", "false").lower() != "true",
    reason="Teste de integração com Ollama não habilitado.",
)
def test_gera_resposta_completa_com_ollama() -> None:
    """Verifica opcionalmente a recuperação e a geração usando o Ollama."""
    result = generate_answer(
        "Quais são os sinais de alerta associados à pressão muito elevada?"
    )

    assert result["answer"].strip()
    assert "data/knowledge_base/hipertensao.md" in result["sources"]
    assert result["requires_human_review"] is True
