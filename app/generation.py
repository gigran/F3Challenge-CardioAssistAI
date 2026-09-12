"""Geração de respostas contextualizadas com LCEL."""

from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from app.config import Settings, get_settings
from app.rag import format_source, retrieve_documents

SYSTEM_PROMPT = """
Você é o CardioAssist AI, um assistente educacional de apoio à decisão clínica.

Quem lê a sua resposta é um profissional de saúde conduzindo o caso, e não o
paciente. Escreva como quem apresenta um resumo técnico a um colega.

Regras obrigatórias:
- Use quando exista as informações presentes no contexto recuperado.
- Se o contexto não for suficiente, declare explicitamente essa limitação.
- Não invente dados do paciente, diagnósticos, exames ou fontes.
- Não prescreva, inicie, suspenda ou altere medicamentos.
- Destaque sinais de alerta e necessidade de avaliação imediata quando aplicável.
- Diferencie dados fornecidos, informações das fontes e limitações da análise.
- Cite as fontes usando o formato [Fonte: nome do arquivo].
- Finalize lembrando que a resposta é de apoio e não substitui o julgamento
  clínico do profissional responsável.

Regras de linguagem:
- Não oriente o leitor a procurar um médico, um serviço de emergência ou um
  especialista: é ele quem presta o atendimento. Descreva a conduta indicada.
- Não use saudações nem trate o leitor como paciente. Nada de "você deve",
  "seu médico" ou "procure ajuda".
- Refira-se ao paciente na terceira pessoa.

Responda em português do Brasil, de forma clara, objetiva e prudente.
""".strip()

CLASSIFIER_SYSTEM_PROMPT = """
Você classifica perguntas do CardioAssist AI em UMA categoria.

Categorias e quando usar cada uma:
- informacao_geral: pergunta solicitando informação conceitual na area de saúde e não se enquadra em nenhuma outra categoria.
- fora_escopo: pergunta ou informação que esta totalmente fora da áre de saúde.
- sintomas: sinais e sintomas de uma condição.
- frequencia: prevalência, incidência ou frequência de uma condição.
- tratamento: como tratar, terapia ou conduta terapêutica.
- diagnostico: como diagnosticar ou critérios diagnósticos.


Responda somente com a categoria e um grau de confiança entre 0 e 1.
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


SUPPORTED_LLM_PROVIDERS = {"ollama", "lora", "openai"}


def create_chat_model(
    settings: Settings | None = None,
    llm_provider: str | None = None,
) -> BaseChatModel:
    """Cria o modelo de chat configurado para Ollama ou OpenAI."""
    current_settings = settings or get_settings()
    selected_provider = llm_provider or current_settings.llm_provider

    if selected_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError("llm_provider deve ser 'ollama' ou 'openai'.")

    if selected_provider == "ollama":
        return ChatOllama(
            model=current_settings.ollama_chat_model,
            base_url=current_settings.ollama_base_url,
            temperature=0,
        )

    if selected_provider == "lora":
        from app.loramodel import load_lora_model, LoraChatModel
        model, tokenizer = load_lora_model()
        return LoraChatModel(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=64,
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


class IntentResult(BaseModel):
    """Classificação de intenção produzida pela LLM."""

    intent: Literal[
        "informacao_geral",
        "sintomas",
        "frequencia",
        "tratamento",
        "diagnostico",
        "protocolo",
        "procedimento",
        "fora_escopo",
    ] = Field(description="Categoria da pergunta.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Grau de confiança da classificação, entre 0 e 1.",
    )


@lru_cache(maxsize=1)
def get_classification_chain(
    llm_provider: str,
) -> Runnable[dict[str, str], IntentResult]:
    """Monta e mantém em cache a cadeia LCEL de classificação de intenção."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFIER_SYSTEM_PROMPT),
            ("human", "Pergunta:\n{question}"),
        ]
    )
    model = create_chat_model(llm_provider=llm_provider)
    return prompt | model.with_structured_output(IntentResult)


@lru_cache(maxsize=3)
def get_generation_chain(llm_provider: str) -> Runnable[dict[str, str], str]:
    """Monta e mantém em cache a cadeia LCEL de geração."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                (
                    "Instrução:\n{instruction}\n\n"
                    "Pergunta:\n{question}\n\n"
                    "Contexto recuperado:\n{context}\n\n"
                    "Produza a resposta seguindo todas as regras de segurança."
                ),
            ),
        ]
    )

    return prompt | create_chat_model(llm_provider=llm_provider) | StrOutputParser()


def generate_answer(
    question: str,
    llm_provider: str | None = None,
) -> GeneratedAnswer:
    """Recupera o contexto e gera uma resposta acompanhada das fontes."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    selected_provider = llm_provider or get_settings().llm_provider
    documents = retrieve_documents(normalized_question)
    context = format_documents(documents)
    answer = get_generation_chain(selected_provider).invoke(
        {
            "question": normalized_question,
            "context": context,
        }
    )
    sources = list(dict.fromkeys(format_source(document) for document in documents))

    return {
        "answer": answer,
        "sources": sources,
        "requires_human_review": True,
    }
