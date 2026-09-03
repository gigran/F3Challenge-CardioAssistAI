"""Testes do carregamento, divisão e recuperação do RAG."""

import os
from dataclasses import replace

import pytest
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings
from app.rag import (
    _separate_front_matter,
    create_embeddings,
    load_knowledge_documents,
    retrieve_documents,
    split_documents,
)


def test_mantem_embeddings_no_ollama_quando_chat_usa_openai() -> None:
    """Desacopla os embeddings locais do provedor escolhido para geração."""
    settings = replace(get_settings(), llm_provider="openai")

    embeddings = create_embeddings(settings)

    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.model == settings.ollama_embedding_model


def test_separa_metadados_do_conteudo_markdown() -> None:
    """Verifica a leitura do front matter YAML simples."""
    content = """---
titulo: Documento de teste
tema: hipertensao
---

# Conteúdo clínico de teste
"""

    metadata, body = _separate_front_matter(content)

    assert metadata == {
        "titulo": "Documento de teste",
        "tema": "hipertensao",
    }
    assert body == "# Conteúdo clínico de teste"


def test_carrega_documento_da_base_com_metadados() -> None:
    """Verifica o carregamento do documento de hipertensão."""
    documents = load_knowledge_documents()

    hypertension_document = next(
        document
        for document in documents
        if document.metadata["file_name"] == "hipertensao.md"
    )

    assert hypertension_document.metadata["id"] == "cardio-hipertensao-001"
    assert hypertension_document.metadata["tema"] == "hipertensao_arterial"
    assert hypertension_document.metadata["requer_validacao_humana"] == "true"
    assert (
        hypertension_document.metadata["source"] == "data/knowledge_base/hipertensao.md"
    )
    assert "# Hipertensão arterial em adultos" in hypertension_document.page_content


def test_carrega_documento_de_dor_toracica_com_metadados() -> None:
    """Verifica o carregamento e a rastreabilidade da base de dor torácica."""
    documents = load_knowledge_documents()

    chest_pain_document = next(
        document
        for document in documents
        if document.metadata["file_name"] == "dor_toracica.md"
    )

    assert chest_pain_document.metadata["id"] == "cardio-dor-toracica-001"
    assert chest_pain_document.metadata["tema"] == "dor_toracica"
    assert chest_pain_document.metadata["requer_validacao_humana"] == "true"
    assert (
        chest_pain_document.metadata["source"] == "data/knowledge_base/dor_toracica.md"
    )
    assert "# Dor torácica em adultos" in chest_pain_document.page_content
    assert "## Sinais de alerta e segurança" in chest_pain_document.page_content


def test_divide_documentos_e_preserva_metadados() -> None:
    """Verifica a criação de chunks com rastreabilidade da fonte."""
    documents = load_knowledge_documents()

    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > len(documents)
    assert all(chunk.metadata["source"] for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (500, -1),
        (500, 500),
    ],
)
def test_rejeita_configuracao_invalida_de_chunks(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Verifica a validação dos tamanhos usados na divisão."""
    documents = load_knowledge_documents()

    with pytest.raises(ValueError):
        split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_rejeita_pergunta_vazia_sem_consultar_ollama() -> None:
    """Impede buscas sem conteúdo antes de acessar o modelo de embeddings."""
    with pytest.raises(ValueError, match="A pergunta não pode estar vazia"):
        retrieve_documents("   ")


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS", "false").lower() != "true",
    reason="Teste de integração com Ollama não habilitado.",
)
def test_recupera_sinais_de_alerta_com_ollama() -> None:
    """Verifica opcionalmente a recuperação semântica usando o Ollama local."""
    documents = retrieve_documents(
        "Quais são os sinais de alerta em uma pressão muito elevada?"
    )

    assert documents
    assert any(
        document.metadata["source"] == "data/knowledge_base/hipertensao.md"
        for document in documents
    )
    assert any(
        "sinais de alerta" in document.page_content.lower() for document in documents
    )


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS", "false").lower() != "true",
    reason="Teste de integração com Ollama não habilitado.",
)
def test_recupera_dor_toracica_com_ollama() -> None:
    """Recupera a nova fonte em uma consulta semântica sobre dor torácica."""
    documents = retrieve_documents(
        "Paciente com aperto no peito, suor frio e dor irradiando para o braço."
    )

    assert documents
    assert any(
        document.metadata["source"] == "data/knowledge_base/dor_toracica.md"
        for document in documents
    )
    assert any(
        "dor torácica" in document.page_content.lower()
        or "sinais de alerta" in document.page_content.lower()
        for document in documents
    )
