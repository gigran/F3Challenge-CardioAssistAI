"""Modelos do prontuário eletrônico sintético.

As tabelas guardam apenas dados clínicos. Não existe coluna de nome, CPF,
endereço ou telefone: cada paciente é identificado por um código como
"PAC-001". O prontuário já nasce anonimizado por construção.
"""

from datetime import date
from enum import StrEnum

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Classe base de todos os modelos."""


class StatusExame(StrEnum):
    """Situação de um exame dentro do prontuário."""

    PENDENTE = "pendente"
    CONCLUIDO = "concluido"


class Paciente(Base):
    """Paciente sintético acompanhado pelo CardioAssist AI."""

    __tablename__ = "pacientes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    codigo: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    idade: Mapped[int] = mapped_column(Integer)
    sexo: Mapped[str] = mapped_column(String(20))
    condicoes_cronicas: Mapped[str] = mapped_column(Text, default="")
    alergias: Mapped[str] = mapped_column(Text, default="")

    exames: Mapped[list["Exame"]] = relationship(
        back_populates="paciente",
        cascade="all, delete-orphan",
    )
    medicacoes: Mapped[list["Medicacao"]] = relationship(
        back_populates="paciente",
        cascade="all, delete-orphan",
    )
    atendimentos: Mapped[list["Atendimento"]] = relationship(
        back_populates="paciente",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Paciente {self.codigo}>"


class Exame(Base):
    """Exame solicitado para um paciente, concluído ou ainda pendente."""

    __tablename__ = "exames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes.id"))
    nome: Mapped[str] = mapped_column(String(120))
    # Exames pendentes ainda não têm resultado nem data de conclusão.
    resultado: Mapped[str | None] = mapped_column(Text, default=None)
    valor_referencia: Mapped[str | None] = mapped_column(String(120), default=None)
    status: Mapped[str] = mapped_column(String(20), default=StatusExame.PENDENTE)
    data: Mapped[date | None] = mapped_column(Date, default=None)

    paciente: Mapped[Paciente] = relationship(back_populates="exames")

    def __repr__(self) -> str:
        return f"<Exame {self.nome} ({self.status})>"


class Medicacao(Base):
    """Medicação em uso ou já suspensa para um paciente."""

    __tablename__ = "medicacoes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes.id"))
    nome: Mapped[str] = mapped_column(String(120))
    dose: Mapped[str] = mapped_column(String(60))
    frequencia: Mapped[str] = mapped_column(String(60))
    ativa: Mapped[bool] = mapped_column(Boolean, default=True)

    paciente: Mapped[Paciente] = relationship(back_populates="medicacoes")

    def __repr__(self) -> str:
        return f"<Medicacao {self.nome} {self.dose}>"


class Atendimento(Base):
    """Registro de uma consulta, com a queixa e a evolução clínica."""

    __tablename__ = "atendimentos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes.id"))
    data: Mapped[date] = mapped_column(Date)
    queixa: Mapped[str] = mapped_column(Text)
    evolucao: Mapped[str] = mapped_column(Text)
    profissional: Mapped[str] = mapped_column(String(120))

    paciente: Mapped[Paciente] = relationship(back_populates="atendimentos")

    def __repr__(self) -> str:
        return f"<Atendimento {self.data} {self.profissional}>"
