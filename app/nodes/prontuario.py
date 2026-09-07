"""Nós do fluxo que consultam o prontuário eletrônico.

São duas etapas separadas de propósito, para que o fluxo mostre cada decisão:

1. prontuario_node        busca o resumo clínico do paciente;
2. exames_pendentes_node  verifica quais exames ainda estão sem resultado.

Os nós recebem e devolvem o estado do LangGraph como um dicionário simples.
"""

from app.database.connection import get_session
from app.database.repositories import (
    buscar_paciente,
    listar_exames_pendentes,
    montar_resumo_prontuario,
)
from app.logger import monitor_node_execution

COM_PACIENTE = "com_paciente"
SEM_PACIENTE = "sem_paciente"


def rotear_por_paciente(state: dict) -> str:
    """Decide se o fluxo passa pelas etapas do prontuário.

    Quando nenhum código de paciente é informado, o assistente responde apenas
    com a base de conhecimento, sem consultar o banco.
    """
    codigo = str(state.get("patient_code", "")).strip()
    return COM_PACIENTE if codigo else SEM_PACIENTE


@monitor_node_execution
def prontuario_node(state: dict) -> dict:
    """Busca o prontuário do paciente e devolve o resumo clínico.

    Um código desconhecido não interrompe o fluxo: o resumo passa a conter o
    aviso de que o paciente não foi encontrado, para que o assistente declare
    a limitação em vez de inventar dados.
    """
    codigo = str(state.get("patient_code", "")).strip()

    with get_session() as session:
        paciente = buscar_paciente(session, codigo)
        resumo = montar_resumo_prontuario(session, codigo)

    encontrado = paciente is not None

    return {
        "patient_summary": resumo,
        "patient_found": encontrado,
        # A fonte entra na lista de fontes da resposta, para deixar explícito
        # que o prontuário foi consultado.
        "sources": [f"prontuario:{codigo.upper()}"] if encontrado else [],
    }


@monitor_node_execution
def exames_pendentes_node(state: dict) -> dict:
    """Verifica quais exames do paciente ainda estão sem resultado."""
    codigo = str(state.get("patient_code", "")).strip()

    with get_session() as session:
        paciente = buscar_paciente(session, codigo)
        exames = listar_exames_pendentes(session, paciente.id) if paciente else []
        nomes = [exame.nome for exame in exames]

    return {"pending_exams": nomes}
