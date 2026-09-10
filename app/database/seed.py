"""Carga dos prontuários sintéticos no PostgreSQL.

Lê o arquivo "data/synthetic/prontuarios.json" e grava os pacientes no banco.
A carga é repetível: os dados antigos são apagados antes da inserção, então
rodar o script duas vezes não duplica nada. Uso:

    python -m app.database.seed
"""

import json
from datetime import date
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT
from app.database.connection import create_tables, get_session
from app.database.models import Atendimento, Exame, Medicacao, Paciente

SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "prontuarios.json"


def ler_prontuarios(caminho: Path = SYNTHETIC_DATA_PATH) -> list[dict]:
    """Lê o arquivo JSON com os pacientes sintéticos."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de dados sintéticos não encontrado: {caminho}"
        )

    with caminho.open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _converter_data(valor: str | None) -> date | None:
    """Converte a data do JSON para o tipo date do Python."""
    return date.fromisoformat(valor) if valor else None


def montar_paciente(dados: dict) -> Paciente:
    """Monta um paciente com os exames, medicações e atendimentos dele."""
    return Paciente(
        codigo=dados["codigo"],
        idade=dados["idade"],
        sexo=dados["sexo"],
        condicoes_cronicas=dados["condicoes_cronicas"],
        alergias=dados["alergias"],
        exames=[
            Exame(
                nome=exame["nome"],
                resultado=exame["resultado"],
                valor_referencia=exame["valor_referencia"],
                status=exame["status"],
                data=_converter_data(exame["data"]),
            )
            for exame in dados["exames"]
        ],
        medicacoes=[
            Medicacao(
                nome=medicacao["nome"],
                dose=medicacao["dose"],
                frequencia=medicacao["frequencia"],
                ativa=medicacao["ativa"],
            )
            for medicacao in dados["medicacoes"]
        ],
        atendimentos=[
            Atendimento(
                data=_converter_data(atendimento["data"]),
                queixa=atendimento["queixa"],
                evolucao=atendimento["evolucao"],
                profissional=atendimento["profissional"],
            )
            for atendimento in dados["atendimentos"]
        ],
    )


def limpar_tabelas(session: Session) -> None:
    """Apaga os dados existentes, das tabelas filhas para a tabela principal."""
    session.execute(delete(Atendimento))
    session.execute(delete(Medicacao))
    session.execute(delete(Exame))
    session.execute(delete(Paciente))


def popular_banco(session: Session, prontuarios: list[dict]) -> int:
    """Grava os pacientes sintéticos e devolve quantos foram inseridos."""
    limpar_tabelas(session)
    session.add_all(montar_paciente(dados) for dados in prontuarios)
    session.commit()
    return len(prontuarios)


def main() -> None:
    """Cria as tabelas e carrega os prontuários sintéticos."""
    create_tables()
    prontuarios = ler_prontuarios()

    with get_session() as session:
        total = popular_banco(session, prontuarios)

    print(f"{total} pacientes sintéticos carregados a partir de {SYNTHETIC_DATA_PATH}")


if __name__ == "__main__":
    main()
