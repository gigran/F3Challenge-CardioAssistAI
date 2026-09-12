"""Abstrações para armazenamento e recuperação vetorial."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreRetriever,
)

def create_in_memory_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    batch_size: int = 64,
) -> InMemoryVectorStore:
    """Cria o vector store adicionando os documentos em lotes menores."""

    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")

    vector_store = InMemoryVectorStore(embedding=embeddings)

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vector_store.add_documents(batch)

    return vector_store


def create_retriever(
    vector_store: VectorStore,
    k: int,
) -> VectorStoreRetriever:
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    return vector_store.as_retriever(search_kwargs={"k": k})

# def create_in_memory_vector_store(
#     documents: list[Document],
#     embeddings: Embeddings,
# ) -> VectorStore:
#     """Cria um vector store em memória a partir dos documentos."""
#     return InMemoryVectorStore.from_documents(
#         documents=documents,
#         embedding=embeddings,
#     )


# def create_retriever(
#     vector_store: VectorStore,
#     k: int,
# ) -> VectorStoreRetriever:
#     """Cria um retriever configurado com a quantidade de resultados desejada."""
#     if k <= 0:
#         raise ValueError("k deve ser maior que zero.")

#     return vector_store.as_retriever(
#         search_kwargs={"k": k},
#     )
