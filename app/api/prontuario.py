"""Endpoints de manutenção do prontuário sintético.

Permitem cadastrar pacientes e complementar o prontuário sem editar o arquivo
de carga inicial e rodar o seed de novo.

A mesma regra do endpoint de assistência vale aqui: com USE_SYNTHETIC_DATA_ONLY
ligado, textos com identificadores diretos são recusados. O banco deste
protótipo precisa continuar sendo só de dados sintéticos.
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.prontuario_schemas import (
    AtendimentoEntrada,
    AtendimentoSaida,
    ExameEntrada,
    ExameSaida,
    MedicacaoEntrada,
    MedicacaoSaida,
    PacienteCompleto,
    PacienteEntrada,
    PacienteResumido,
    ResultadoDeExame,
)
from app.config import get_settings
from app.database.connection import get_session
from app.database.models import Atendimento, Exame, Medicacao, Paciente
from app.database.repositories import (
    adicionar_atendimento,
    adicionar_exame,
    adicionar_medicacao,
    buscar_exame,
    buscar_paciente,
    criar_paciente,
    listar_pacientes,
    montar_resumo_prontuario,
    registrar_resultado_exame,
    remover_paciente,
)
from app.observability.audit import registrar_recusa
from app.security.personal_data import encontrar_dados_pessoais

router = APIRouter(prefix="/pacientes", tags=["Prontuário"])


def obter_sessao() -> Iterator[Session]:
    """Abre e fecha uma sessão do banco por requisição."""
    with get_session() as session:
        yield session


# Apelido da dependência: o FastAPI injeta a sessão em toda rota que o usa.
SessaoDoBanco = Annotated[Session, Depends(obter_sessao)]


def _garantir_dados_sinteticos(*textos: str | None) -> None:
    """Recusa a gravação quando algum texto contém identificador direto."""
    if not get_settings().use_synthetic_data_only:
        return

    encontrados = sorted(
        {tipo for texto in textos if texto for tipo in encontrar_dados_pessoais(texto)}
    )

    if not encontrados:
        return

    registrar_recusa(encontrados)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"O conteúdo enviado parece conter dados pessoais ({', '.join(encontrados)}). "
            "Este protótipo aceita somente prontuários sintéticos."
        ),
    )


def _obter_paciente(session: Session, codigo: str) -> Paciente:
    """Busca o paciente ou responde 404, em vez de deixar o erro subir."""
    paciente = buscar_paciente(session, codigo)

    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum paciente encontrado com o código {codigo}.",
        )

    return paciente


@router.get("", response_model=list[PacienteResumido])
def listar(session: SessaoDoBanco) -> list[Paciente]:
    """Lista os pacientes cadastrados, em ordem de código."""
    return listar_pacientes(session)


@router.get("/{codigo}", response_model=PacienteCompleto)
def detalhar(codigo: str, session: SessaoDoBanco) -> Paciente:
    """Devolve o prontuário completo de um paciente."""
    return _obter_paciente(session, codigo)


@router.get("/{codigo}/resumo")
def resumir(codigo: str, session: SessaoDoBanco) -> dict[str, str]:
    """Devolve o mesmo resumo em texto que o assistente usa como contexto."""
    _obter_paciente(session, codigo)
    return {
        "codigo": codigo.strip().upper(),
        "resumo": montar_resumo_prontuario(session, codigo),
    }


@router.post("", response_model=PacienteCompleto, status_code=status.HTTP_201_CREATED)
def cadastrar(
    entrada: PacienteEntrada,
    session: SessaoDoBanco,
) -> Paciente:
    """Cadastra um paciente sintético com o prontuário inicial dele."""
    _garantir_dados_sinteticos(entrada.condicoes_cronicas, entrada.alergias)

    codigo = entrada.codigo.strip().upper()
    if buscar_paciente(session, codigo) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Já existe um paciente com o código {codigo}.",
        )

    paciente = Paciente(
        codigo=codigo,
        idade=entrada.idade,
        sexo=entrada.sexo,
        condicoes_cronicas=entrada.condicoes_cronicas,
        alergias=entrada.alergias,
        exames=[Exame(**exame.model_dump()) for exame in entrada.exames],
        medicacoes=[
            Medicacao(**medicacao.model_dump()) for medicacao in entrada.medicacoes
        ],
        atendimentos=[
            Atendimento(**atendimento.model_dump())
            for atendimento in entrada.atendimentos
        ],
    )

    return criar_paciente(session, paciente)


@router.delete("/{codigo}", status_code=status.HTTP_204_NO_CONTENT)
def excluir(codigo: str, session: SessaoDoBanco) -> None:
    """Apaga o paciente e todo o prontuário dele."""
    remover_paciente(session, _obter_paciente(session, codigo))


@router.post(
    "/{codigo}/exames",
    response_model=ExameSaida,
    status_code=status.HTTP_201_CREATED,
)
def incluir_exame(
    codigo: str,
    entrada: ExameEntrada,
    session: SessaoDoBanco,
) -> Exame:
    """Solicita um exame novo ou registra um já concluído."""
    _garantir_dados_sinteticos(entrada.resultado)
    paciente = _obter_paciente(session, codigo)

    return adicionar_exame(
        session,
        Exame(paciente_id=paciente.id, **entrada.model_dump()),
    )


@router.patch("/{codigo}/exames/{exame_id}", response_model=ExameSaida)
def concluir_exame(
    codigo: str,
    exame_id: int,
    entrada: ResultadoDeExame,
    session: SessaoDoBanco,
) -> Exame:
    """Registra o resultado de um exame, tirando-o da lista de pendentes."""
    _garantir_dados_sinteticos(entrada.resultado)
    paciente = _obter_paciente(session, codigo)

    exame = buscar_exame(session, paciente.id, exame_id)
    if exame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"O exame {exame_id} não pertence ao paciente {paciente.codigo}.",
        )

    return registrar_resultado_exame(
        session,
        exame,
        resultado=entrada.resultado,
        data=entrada.data,
        valor_referencia=entrada.valor_referencia,
    )


@router.post(
    "/{codigo}/medicacoes",
    response_model=MedicacaoSaida,
    status_code=status.HTTP_201_CREATED,
)
def incluir_medicacao(
    codigo: str,
    entrada: MedicacaoEntrada,
    session: SessaoDoBanco,
) -> Medicacao:
    """Acrescenta uma medicação ao prontuário."""
    paciente = _obter_paciente(session, codigo)

    return adicionar_medicacao(
        session,
        Medicacao(paciente_id=paciente.id, **entrada.model_dump()),
    )


@router.post(
    "/{codigo}/atendimentos",
    response_model=AtendimentoSaida,
    status_code=status.HTTP_201_CREATED,
)
def incluir_atendimento(
    codigo: str,
    entrada: AtendimentoEntrada,
    session: SessaoDoBanco,
) -> Atendimento:
    """Registra uma consulta no histórico do paciente."""
    _garantir_dados_sinteticos(entrada.queixa, entrada.evolucao)
    paciente = _obter_paciente(session, codigo)

    return adicionar_atendimento(
        session,
        Atendimento(paciente_id=paciente.id, **entrada.model_dump()),
    )
