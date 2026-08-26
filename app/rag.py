"""Carregamento e recuperação de documentos da base de conhecimento."""

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import PROJECT_ROOT, Settings, get_settings

KNOWLEDGE_BASE_DIRECTORY = PROJECT_ROOT / "data" / "knowledge_base"


def _separate_front_matter(content: str) -> tuple[dict[str, str], str]:
    """Separa metadados YAML simples do conteúdo Markdown."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return {}, content

    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if ":" not in line:
            continue

        key, value = line.split(":", maxsplit=1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def load_knowledge_documents(
    directory: Path = KNOWLEDGE_BASE_DIRECTORY,
) -> list[Document]:
    """Carrega os arquivos Markdown da base como documentos LangChain."""
    if not directory.exists():
        raise FileNotFoundError(
            f"A pasta da base de conhecimento não foi encontrada: {directory}"
        )

    document_paths = sorted(directory.glob("*.md"))
    if not document_paths:
        raise FileNotFoundError(
            f"Nenhum documento Markdown foi encontrado em: {directory}"
        )

    documents: list[Document] = []
    for document_path in document_paths:
        content = document_path.read_text(encoding="utf-8")
        metadata, body = _separate_front_matter(content)
        metadata.update(
            {
                "source": document_path.relative_to(PROJECT_ROOT).as_posix(),
                "file_name": document_path.name,
            }
        )
        documents.append(Document(page_content=body, metadata=metadata))

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Divide documentos em trechos menores, preservando seus metadados."""
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap não pode ser negativo nem alcançar chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Cria o modelo de embeddings configurado para Ollama ou OpenAI."""
    current_settings = settings or get_settings()

    if current_settings.llm_provider == "ollama":
        return OllamaEmbeddings(
            model=current_settings.ollama_embedding_model,
            base_url=current_settings.ollama_base_url,
        )

    api_key = current_settings.openai_api_key
    if not api_key or api_key.startswith("substitua-"):
        raise ValueError(
            "OPENAI_API_KEY deve ser configurada para utilizar embeddings da OpenAI."
        )

    return OpenAIEmbeddings(
        model=current_settings.openai_embedding_model,
        api_key=api_key,
    )


def build_vector_store() -> InMemoryVectorStore:
    """Indexa os documentos e devolve um banco vetorial em memória."""
    documents = load_knowledge_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    return InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Cria uma única instância do recuperador durante a execução."""
    settings = get_settings()
    vector_store = build_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": settings.rag_top_k})


def retrieve_documents(question: str) -> list[Document]:
    """Recupera os trechos mais relacionados à pergunta informada."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    return get_retriever().invoke(normalized_question)
