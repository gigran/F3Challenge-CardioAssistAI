"""Consultas e escrita no prontuário eletrônico.

Este módulo concentra as buscas no banco e a montagem do resumo em texto que
será entregue à LLM como contexto do paciente.
"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Atendimento, Exame, Medicacao, Paciente, StatusExame

LIMITE_DE_ATENDIMENTOS = 3


def buscar_paciente(session: Session, codigo: str) -> Paciente | None:
    """Busca um paciente pelo código, como "PAC-001"."""
    consulta = select(Paciente).where(Paciente.codigo == codigo.strip().upper())
    return session.scalars(consulta).first()


def listar_pacientes(session: Session) -> list[Paciente]:
    """Lista todos os pacientes cadastrados, em ordem de código."""
    return list(session.scalars(select(Paciente).order_by(Paciente.codigo)))


def listar_exames_pendentes(session: Session, paciente_id: int) -> list[Exame]:
    """Lista os exames que ainda não têm resultado."""
    consulta = (
        select(Exame)
        .where(Exame.paciente_id == paciente_id)
        .where(Exame.status == StatusExame.PENDENTE)
        .order_by(Exame.nome)
    )
    return list(session.scalars(consulta))


def listar_exames_concluidos(session: Session, paciente_id: int) -> list[Exame]:
    """Lista os exames já concluídos, do mais recente para o mais antigo."""
    consulta = (
        select(Exame)
        .where(Exame.paciente_id == paciente_id)
        .where(Exame.status == StatusExame.CONCLUIDO)
        .order_by(Exame.data.desc())
    )
    return list(session.scalars(consulta))


def listar_medicacoes_ativas(session: Session, paciente_id: int) -> list[Medicacao]:
    """Lista as medicações que o paciente usa no momento."""
    consulta = (
        select(Medicacao)
        .where(Medicacao.paciente_id == paciente_id)
        .where(Medicacao.ativa.is_(True))
        .order_by(Medicacao.nome)
    )
    return list(session.scalars(consulta))


def listar_atendimentos(
    session: Session,
    paciente_id: int,
    limite: int = LIMITE_DE_ATENDIMENTOS,
) -> list[Atendimento]:
    """Lista os atendimentos mais recentes do paciente."""
    consulta = (
        select(Atendimento)
        .where(Atendimento.paciente_id == paciente_id)
        .order_by(Atendimento.data.desc())
        .limit(limite)
    )
    return list(session.scalars(consulta))


def montar_resumo_prontuario(session: Session, codigo: str) -> str:
    """Monta o resumo em texto do prontuário, pronto para virar contexto da LLM.

    Devolve uma mensagem clara quando o paciente não existe, em vez de falhar,
    para que o assistente possa declarar a limitação em vez de inventar dados.
    """
    paciente = buscar_paciente(session, codigo)
    if paciente is None:
        return f"Nenhum paciente encontrado com o código {codigo}."

    linhas = [
        f"Paciente {paciente.codigo} | {paciente.idade} anos | {paciente.sexo}",
        f"Condições crônicas: {paciente.condicoes_cronicas or 'nenhuma registrada'}",
        f"Alergias: {paciente.alergias or 'nenhuma registrada'}",
    ]

    medicacoes = listar_medicacoes_ativas(session, paciente.id)
    linhas.append("")
    linhas.append("Medicações em uso:")
    if medicacoes:
        linhas.extend(
            f"- {medicacao.nome} {medicacao.dose}, {medicacao.frequencia}"
            for medicacao in medicacoes
        )
    else:
        linhas.append("- nenhuma medicação ativa registrada")

    concluidos = listar_exames_concluidos(session, paciente.id)
    linhas.append("")
    linhas.append("Exames concluídos:")
    if concluidos:
        linhas.extend(
            f"- {exame.data}: {exame.nome} = {exame.resultado} "
            f"(referência: {exame.valor_referencia})"
            for exame in concluidos
        )
    else:
        linhas.append("- nenhum exame concluído registrado")

    pendentes = listar_exames_pendentes(session, paciente.id)
    linhas.append("")
    linhas.append("Exames pendentes:")
    if pendentes:
        linhas.extend(f"- {exame.nome}" for exame in pendentes)
    else:
        linhas.append("- nenhum exame pendente")

    atendimentos = listar_atendimentos(session, paciente.id)
    linhas.append("")
    linhas.append("Últimos atendimentos:")
    if atendimentos:
        for atendimento in atendimentos:
            linhas.append(
                f"- {atendimento.data} ({atendimento.profissional}): "
                f"queixa: {atendimento.queixa} | evolução: {atendimento.evolucao}"
            )
    else:
        linhas.append("- nenhum atendimento registrado")

    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Escrita no prontuário
# ---------------------------------------------------------------------------


def criar_paciente(session: Session, paciente: Paciente) -> Paciente:
    """Grava um paciente novo com tudo o que veio junto dele."""
    session.add(paciente)
    session.commit()
    session.refresh(paciente)
    return paciente


def remover_paciente(session: Session, paciente: Paciente) -> None:
    """Apaga o paciente e, junto, exames, medicações e atendimentos."""
    session.delete(paciente)
    session.commit()


def adicionar_exame(session: Session, exame: Exame) -> Exame:
    """Acrescenta um exame ao prontuário de um paciente."""
    session.add(exame)
    session.commit()
    session.refresh(exame)
    return exame


def adicionar_medicacao(session: Session, medicacao: Medicacao) -> Medicacao:
    """Acrescenta uma medicação ao prontuário de um paciente."""
    session.add(medicacao)
    session.commit()
    session.refresh(medicacao)
    return medicacao


def adicionar_atendimento(session: Session, atendimento: Atendimento) -> Atendimento:
    """Acrescenta um atendimento ao histórico de um paciente."""
    session.add(atendimento)
    session.commit()
    session.refresh(atendimento)
    return atendimento


def buscar_exame(session: Session, paciente_id: int, exame_id: int) -> Exame | None:
    """Busca um exame garantindo que ele pertence ao paciente informado."""
    consulta = (
        select(Exame)
        .where(Exame.id == exame_id)
        .where(Exame.paciente_id == paciente_id)
    )
    return session.scalars(consulta).first()


def registrar_resultado_exame(
    session: Session,
    exame: Exame,
    resultado: str,
    data: date,
    valor_referencia: str | None = None,
) -> Exame:
    """Conclui um exame pendente, gravando o resultado e a data."""
    exame.resultado = resultado
    exame.data = data
    exame.status = StatusExame.CONCLUIDO
    if valor_referencia is not None:
        exame.valor_referencia = valor_referencia

    session.commit()
    session.refresh(exame)
    return exame
