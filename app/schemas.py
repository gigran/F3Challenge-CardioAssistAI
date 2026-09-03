"""Contratos de entrada e saída da API."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SafetyStatus(StrEnum):
    """Níveis de segurança produzidos pelo fluxo clínico."""

    REQUIRED_REVIEW = "revisao_obrigatoria"
    ATTENTION = "atencao"
    EMERGENCY = "emergencia"


class LLMProvider(StrEnum):
    """Provedores disponíveis para geração de respostas."""

    OLLAMA = "ollama"
    OPENAI = "openai"


class AssistRequest(BaseModel):
    """Pergunta e contexto sintético enviados ao CardioAssist AI."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    question: str = Field(
        min_length=3,
        max_length=4000,
        description="Pergunta clínica contendo exclusivamente dados sintéticos.",
        examples=[
            (
                "Paciente sintético com pressão 190/125 e dor torácica. "
                "Quais são os sinais de alerta?"
            ),
        ],
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Provedor utilizado somente para gerar a resposta.",
    )


class AssistResponse(BaseModel):
    """Resposta contextualizada devolvida pelo fluxo LangGraph."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Resposta gerada a partir do contexto recuperado.",
    )
    sources: list[str] = Field(
        description="Arquivos da base utilizados na recuperação.",
    )
    safety_alerts: list[str] = Field(
        description="Alertas identificados por regras determinísticas.",
    )
    safety_status: SafetyStatus = Field(
        description="Classificação de segurança atribuída à resposta.",
    )
    requires_human_review: bool = Field(
        description="Indica que a resposta exige validação profissional.",
    )
