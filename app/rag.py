"""Carregamento e recuperação de documentos da base de conhecimento."""

import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import PROJECT_ROOT, Settings, get_settings
from app.vector_store import create_in_memory_vector_store, create_retriever

from pypdf import PdfReader

KNOWLEDGE_BASE_DIRECTORY = PROJECT_ROOT / "data" / "knowledge_base"

TITULO_MARKDOWN = re.compile(r"^#{1,6}\s+(.+)$", flags=re.MULTILINE)

# Separa o arquivo da seção na fonte citada, como em
# "data/knowledge_base/hipertensao.md § Sinais de alerta e segurança".
SEPARADOR_DE_SECAO = " § "

# Para PDFs, a "seção" citada é o número da página.
PREFIXO_DE_PAGINA = "pág. "

def _carregar_markdown(document_path: Path) -> list[Document]:
    """Lê um arquivo Markdown como um único documento LangChain."""
    content = document_path.read_text(encoding="utf-8")
    metadata, body = _separate_front_matter(content)

    metadata.update(
        {
            "source": document_path.relative_to(PROJECT_ROOT).as_posix(),
            "file_name": document_path.name,
        }
    )

    return [Document(page_content=body, metadata=metadata)]


def _carregar_pdf(document_path: Path) -> list[Document]:
    """Lê um PDF como um documento LangChain por página.

    Cada página vira um documento próprio para que a fonte citada possa
    apontar a página exata, do mesmo modo que a seção aponta o título no
    Markdown.
    """
    source = document_path.relative_to(PROJECT_ROOT).as_posix()
    reader = PdfReader(str(document_path))

    documents: list[Document] = []
    for numero, page in enumerate(reader.pages, start=1):
        texto = (page.extract_text() or "").strip()
        if not texto:
            continue

        documents.append(
            Document(
                page_content=texto,
                metadata={
                    "source": source,
                    "file_name": document_path.name,
                    "pagina": numero,
                },
            )
        )

    return documents

_CARREGADORES_POR_EXTENSAO = {
    ".md": _carregar_markdown,
    ".pdf": _carregar_pdf,
}

def _resolver_secao(chunk: Document, conteudo_do_documento: str) -> str:
    """Descobre em que seção do documento original o trecho começa.

    O divisor corta preferencialmente nos títulos, então a maioria dos trechos
    já começa em um. Quando o trecho começa no meio de uma seção, a busca volta
    no documento até o título mais próximo acima, usando o start_index.
    """
    primeira_linha = chunk.page_content.lstrip().split("\n", maxsplit=1)[0]
    if primeira_linha.startswith("#"):
        return primeira_linha.lstrip("#").strip()

    inicio = int(chunk.metadata.get("start_index", 0))
    titulos = TITULO_MARKDOWN.findall(conteudo_do_documento[:inicio])

    if titulos:
        return titulos[-1].strip()

    pagina = chunk.metadata.get("pagina")
    return f"{PREFIXO_DE_PAGINA}{pagina}" if pagina else ""

def add_section_metadata(
    documents: list[Document],
    chunks: list[Document],
) -> list[Document]:
    """Grava em cada trecho a seção de onde ele veio."""
    # conteudo_por_arquivo = {
    #     str(document.metadata.get("file_name", "")): document.page_content
    #     for document in documents
    # }
    conteudo_por_arquivo = {
         str(document.metadata.get("source", "")): document.page_content
         for document in documents
     }

    for chunk in chunks:
        arquivo = str(chunk.metadata.get("source", ""))
        chunk.metadata["secao"] = _resolver_secao(
            chunk,
            conteudo_por_arquivo.get(arquivo, ""),
        )

    return chunks


def format_source(document: Document) -> str:
    """Monta a fonte no nível do trecho, e não apenas do arquivo."""
    arquivo = str(document.metadata.get("source", "fonte não informada"))
    secao = str(document.metadata.get("secao", "")).strip()

    return f"{arquivo}{SEPARADOR_DE_SECAO}{secao}" if secao else arquivo


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
    """Carrega os arquivos Markdown e PDF da base como documentos LangChain."""
    if not directory.exists():
        raise FileNotFoundError(
            f"A pasta da base de conhecimento não foi encontrada: {directory}"
        )

    document_paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in _CARREGADORES_POR_EXTENSAO
    )
    if not document_paths:
        raise FileNotFoundError(
            f"Nenhum documento Markdown ou PDF foi encontrado em: {directory}"
        )

    documents: list[Document] = []
    for document_path in document_paths:
        carregador = _CARREGADORES_POR_EXTENSAO[document_path.suffix.lower()]
        documents.extend(carregador(document_path))

    return documents

# def load_knowledge_documents(
#     directory: Path = KNOWLEDGE_BASE_DIRECTORY,
# ) -> list[Document]:
#     """Carrega os arquivos Markdown da base como documentos LangChain."""
#     if not directory.exists():
#         raise FileNotFoundError(
#             f"A pasta da base de conhecimento não foi encontrada: {directory}"
#         )

#     document_paths = sorted(directory.glob("*.md"))
#     if not document_paths:
#         raise FileNotFoundError(
#             f"Nenhum documento Markdown foi encontrado em: {directory}"
#         )

#     documents: list[Document] = []

#     for document_path in document_paths:
#         content = document_path.read_text(encoding="utf-8")
#         metadata, body = _separate_front_matter(content)

#         metadata.update(
#             {
#                 "source": document_path.relative_to(PROJECT_ROOT).as_posix(),
#                 "file_name": document_path.name,
#             }
#         )

#         documents.append(
#             Document(
#                 page_content=body,
#                 metadata=metadata,
#             )
#         )
    

#     return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Divide documentos em trechos menores, preservando seus metadados.

    Cada trecho recebe também a seção de origem, para que a resposta possa citar
    o lugar exato do documento em vez de apenas o nome do arquivo.
    """
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

    return add_section_metadata(documents, splitter.split_documents(documents))


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Cria o modelo local de embeddings do Ollama."""
    current_settings = settings or get_settings()
    return OllamaEmbeddings(
        model=current_settings.ollama_embedding_model,
        base_url=current_settings.ollama_base_url,
    )


def build_vector_store():
    """Indexa os documentos e devolve um banco vetorial em memória."""
    documents = load_knowledge_documents()
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    return create_in_memory_vector_store(
        documents=chunks,
        embeddings=embeddings,
    )


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Cria uma única instância do recuperador durante a execução."""
    settings = get_settings()
    vector_store = build_vector_store()

    return create_retriever(
        vector_store=vector_store,
        k=settings.rag_top_k,
    )


def retrieve_documents(question: str) -> list[Document]:
    """Recupera os trechos mais relacionados à pergunta informada."""
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    return get_retriever().invoke(normalized_question)