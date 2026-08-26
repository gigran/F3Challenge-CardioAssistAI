"""Geração de respostas contextualizadas com LCEL."""

from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.rag import retrieve_documents

SYSTEM_PROMPT = """
Você é o CardioAssist AI, um assistente educacional de apoio à decisão clínica.

Regras obrigatórias:
- Use somente as informações presentes no contexto recuperado.
- Se o contexto não for suficiente, declare explicitamente essa limitação.
- Não invente dados do paciente, diagnósticos, exames ou fontes.
- Não prescreva, inicie, suspenda ou altere medicamentos.
- Destaque sinais de alerta e necessidade de avaliação imediata quando aplicável.
- Diferencie dados fornecidos, informações das fontes e limitações da análise.
- Cite as fontes usando o formato [Fonte: nome do arquivo].
- Finalize lembrando que a resposta exige validação de um profissional de saúde.

Responda em português do Brasil, de forma clara, objetiva e prudente.
""".strip()


class GeneratedAnswer(TypedDict):
    """Estrutura devolvida pela cadeia de geração."""

    answer: str
    sources: list[str]
    requires_human_review: bool


def format_documents(documents: list[Document]) -> str:
    """Formata os documentos recuperados para inclusão no prompt."""
    if not documents:
        return "Nenhum contexto foi recuperado."

    formatted_chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = str(document.metadata.get("source", "fonte não informada"))
        title = str(
            document.metadata.get(
                "titulo",
                document.metadata.get("file_name", source),
            )
        )
        formatted_chunks.append(
            f"[Trecho {index} | Fonte: {title} | Arquivo: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Cria o modelo de chat configurado para Ollama ou OpenAI."""
    current_settings = settings or get_settings()

    if current_settings.llm_provider == "ollama":
        return ChatOllama(
            model=current_settings.ollama_chat_model,
            base_url=current_settings.ollama_base_url,
            temperature=0,
        )

    api_key = current_settings.openai_api_key
    if not api_key or api_key.startswith("substitua-"):
        raise ValueError(
            "OPENAI_API_KEY deve ser configurada para utilizar o modelo da OpenAI."
        )

    return ChatOpenAI(
        model=current_settings.openai_chat_model,
        api_key=api_key,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_generation_chain() -> Runnable[dict[str, str], str]:
    """Monta e mantém em cache a cadeia LCEL de geração."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Pergunta:\n{question}\n\n"
                    "Contexto recuperado:\n{context}\n\n"
                    "Produza a resposta seguindo todas as regras de segurança."
                ),
            ),
        ]
    )

    return prompt | create_chat_model() | StrOutputParser()


def generate_answer(question: str) -> GeneratedAnswer:
    """Recupera o contexto e gera uma resposta acompanhada das fontes."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    documents = retrieve_documents(normalized_question)
    context = format_documents(documents)
    answer = get_generation_chain().invoke(
        {
            "question": normalized_question,
            "context": context,
        }
    )
    sources = list(
        dict.fromkeys(
            str(document.metadata.get("source", "fonte não informada"))
            for document in documents
        )
    )

    return {
        "answer": answer,
        "sources": sources,
        "requires_human_review": True,
    }
