"""Testes do carregamento, divisão e recuperação do RAG."""

import os
from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings
from app.rag import (
    _separate_front_matter,
    create_embeddings,
    format_source,
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


def test_grava_a_secao_de_cada_trecho() -> None:
    """A fonte precisa apontar o lugar do documento, não só o arquivo."""
    documento = Document(
        page_content=(
            "# Título do documento\n\n"
            "Texto de abertura.\n\n"
            "## Sinais de alerta\n\n"
            "Conteúdo da seção de alerta.\n\n"
            "## Acompanhamento\n\n"
            "Conteúdo da seção de acompanhamento.\n"
        ),
        metadata={"source": "data/knowledge_base/teste.md", "file_name": "teste.md"},
    )

    chunks = split_documents([documento], chunk_size=60, chunk_overlap=0)
    secoes = [chunk.metadata["secao"] for chunk in chunks]

    assert "Sinais de alerta" in secoes
    assert "Acompanhamento" in secoes
    assert all(secao for secao in secoes)


def test_herda_a_secao_quando_o_trecho_comeca_no_meio() -> None:
    """Trecho que não começa em título herda a seção anterior."""
    corpo = "## Sinais de alerta\n\n" + ("Frase longa da seção. " * 30)
    documento = Document(
        page_content=corpo,
        metadata={"source": "data/knowledge_base/teste.md", "file_name": "teste.md"},
    )

    chunks = split_documents([documento], chunk_size=200, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.metadata["secao"] == "Sinais de alerta" for chunk in chunks)


def test_monta_a_fonte_com_arquivo_e_secao() -> None:
    """Formato usado na resposta da API e no prompt."""
    documento = Document(
        page_content="conteúdo",
        metadata={
            "source": "data/knowledge_base/hipertensao.md",
            "secao": "Sinais de alerta e segurança",
        },
    )

    assert format_source(documento) == (
        "data/knowledge_base/hipertensao.md § Sinais de alerta e segurança"
    )


def test_usa_apenas_o_arquivo_quando_nao_ha_secao() -> None:
    """Sem seção resolvida, a fonte não pode ficar com o separador solto."""
    documento = Document(
        page_content="conteúdo",
        metadata={"source": "data/knowledge_base/hipertensao.md", "secao": ""},
    )

    assert format_source(documento) == "data/knowledge_base/hipertensao.md"


def test_fontes_do_mesmo_arquivo_nao_se_fundem() -> None:
    """Antes, dois trechos do mesmo arquivo viravam uma única fonte."""
    documentos = [
        Document(
            page_content="a",
            metadata={"source": "base.md", "secao": "Seção A"},
        ),
        Document(
            page_content="b",
            metadata={"source": "base.md", "secao": "Seção B"},
        ),
    ]

    fontes = list(dict.fromkeys(format_source(d) for d in documentos))

    assert fontes == ["base.md § Seção A", "base.md § Seção B"]
