"""Conexão com o PostgreSQL do prontuário."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Cria uma única conexão com o banco durante a execução."""
    return create_engine(get_settings().database_url)


def create_tables() -> None:
    """Cria as tabelas do prontuário, se ainda não existirem."""
    Base.metadata.create_all(get_engine())


def get_session() -> Session:
    """Abre uma sessão do banco.

    Use sempre dentro de um "with", para que a sessão seja fechada no fim:

        with get_session() as session:
            paciente = buscar_paciente(session, "PAC-001")
    """
    return Session(get_engine())
