"""Contratos de entrada e saída dos endpoints do prontuário."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.database.models import StatusExame

CODIGO_PACIENTE = Field(
    min_length=3,
    max_length=20,
    description="Código do paciente sintético, como PAC-001.",
    examples=["PAC-007"],
)


class ExameEntrada(BaseModel):
    """Exame informado na criação de um paciente ou acrescentado depois."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    nome: str = Field(min_length=2, max_length=120)
    resultado: str | None = Field(default=None, max_length=2000)
    valor_referencia: str | None = Field(default=None, max_length=120)
    status: StatusExame = StatusExame.PENDENTE
    data: date | None = None

    @model_validator(mode="after")
    def validar_coerencia_do_status(self) -> "ExameEntrada":
        """Um exame concluído precisa de resultado; um pendente não pode ter."""
        if self.status == StatusExame.CONCLUIDO and not self.resultado:
            raise ValueError("Um exame concluído precisa informar o resultado.")

        if self.status == StatusExame.PENDENTE and self.resultado:
            raise ValueError(
                "Um exame com resultado deve ser registrado como concluído."
            )

        return self


class MedicacaoEntrada(BaseModel):
    """Medicação informada no prontuário."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    nome: str = Field(min_length=2, max_length=120)
    dose: str = Field(min_length=1, max_length=60)
    frequencia: str = Field(min_length=1, max_length=60)
    ativa: bool = True


class AtendimentoEntrada(BaseModel):
    """Consulta registrada no histórico do paciente."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    data: date
    queixa: str = Field(min_length=3, max_length=2000)
    evolucao: str = Field(min_length=3, max_length=2000)
    profissional: str = Field(min_length=2, max_length=120)


class PacienteEntrada(BaseModel):
    """Paciente sintético a ser criado, com o prontuário inicial dele."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    codigo: str = CODIGO_PACIENTE
    idade: int = Field(ge=0, le=130)
    sexo: str = Field(min_length=3, max_length=20, examples=["masculino"])
    condicoes_cronicas: str = Field(default="", max_length=1000)
    alergias: str = Field(default="", max_length=1000)

    exames: list[ExameEntrada] = Field(default_factory=list)
    medicacoes: list[MedicacaoEntrada] = Field(default_factory=list)
    atendimentos: list[AtendimentoEntrada] = Field(default_factory=list)


class ResultadoDeExame(BaseModel):
    """Resultado usado para concluir um exame que estava pendente."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    resultado: str = Field(min_length=1, max_length=2000)
    data: date
    valor_referencia: str | None = Field(default=None, max_length=120)


class ExameSaida(BaseModel):
    """Exame devolvido pela API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    nome: str
    resultado: str | None
    valor_referencia: str | None
    status: str
    data: date | None


class MedicacaoSaida(BaseModel):
    """Medicação devolvida pela API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    nome: str
    dose: str
    frequencia: str
    ativa: bool


class AtendimentoSaida(BaseModel):
    """Atendimento devolvido pela API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    data: date
    queixa: str
    evolucao: str
    profissional: str


class PacienteResumido(BaseModel):
    """Paciente sem o prontuário completo, usado na listagem."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    codigo: str
    idade: int
    sexo: str
    condicoes_cronicas: str
    alergias: str


class PacienteCompleto(PacienteResumido):
    """Paciente com exames, medicações e histórico de atendimentos."""

    exames: list[ExameSaida]
    medicacoes: list[MedicacaoSaida]
    atendimentos: list[AtendimentoSaida]
