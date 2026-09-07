"""Testes da preparação do dataset de fine-tuning."""

import pandas as pd
import pytest

from finetuning import config
from finetuning.prepare_dataset import (
    ajustar_registro,
    anonymize_text,
    build_instruction,
    curate,
    filter_cardiology,
    format_dataset_into_model_input,
    remover_saudacao,
)


def test_remove_dados_pessoais_do_texto() -> None:
    """Confirma que identificadores diretos são substituídos por marcadores."""
    texto = (
        "Meu nome é Maria Silva, CPF 123.456.789-00, "
        "telefone (11) 98765-4321, e-mail maria@teste.com. "
        "Fiz o exame em 10/03/2021 e a cirurgia em 07022022."
    )

    resultado = anonymize_text(texto)

    assert "Maria Silva" not in resultado
    assert "123.456.789-00" not in resultado
    assert "98765-4321" not in resultado
    assert "maria@teste.com" not in resultado
    assert "10/03/2021" not in resultado
    assert "07022022" not in resultado


def test_preserva_valores_clinicos() -> None:
    """Garante que a anonimização não apaga números úteis ao modelo."""
    texto = "Pressão 190/125, frequência 88 bpm, colesterol 240 mg/dL."

    assert anonymize_text(texto) == texto


def test_seleciona_apenas_linhas_de_cardiologia() -> None:
    """A especialidade vem como texto com vários rótulos separados por vírgula."""
    data = pd.DataFrame(
        {
            "medical_specialty": [
                "Cardiologista",
                "Ortopedista - traumatologista",
                "Clínico geral, Cardiologista",
            ],
            "question": ["a", "b", "c"],
        }
    )

    resultado = filter_cardiology(data)

    assert resultado["question"].tolist() == ["a", "c"]


def test_descarta_respostas_curtas_e_perguntas_repetidas() -> None:
    """A curadoria mantém a resposta mais completa de cada pergunta."""
    resposta_longa = "R" * (config.MINIMUM_ANSWER_LENGTH + 20)
    resposta_media = "R" * (config.MINIMUM_ANSWER_LENGTH + 1)

    data = pd.DataFrame(
        {
            "question": ["O que é angina?", "O que é angina?", "O que é infarto?"],
            "answer": [resposta_media, resposta_longa, "Sim."],
        }
    )

    resultado = curate(data)

    assert resultado["question"].tolist() == ["O que é angina?"]
    assert resultado["answer"].tolist() == [resposta_longa]


def test_monta_instrucao_a_partir_do_tipo_de_pergunta() -> None:
    """O MedPT não traz a coluna instruction, que é derivada do question_type."""
    assert build_instruction("Tratamento") == (
        config.INSTRUCTION_BY_QUESTION_TYPE["Tratamento"]
    )
    assert build_instruction("tipo inexistente") == config.DEFAULT_INSTRUCTION


def test_gera_o_formato_esperado_pelo_treino() -> None:
    """O treino lê um JSON com as chaves instruction, input e output."""
    data = pd.DataFrame(
        {
            "question_type": ["Tratamento"],
            "question": ["Como tratar hipertensão?"],
            "answer": ["Com acompanhamento médico."],
        }
    )

    resultado = format_dataset_into_model_input(data)

    assert list(resultado.keys()) == ["instruction", "input", "output"]
    assert resultado["input"] == ["Como tratar hipertensão?"]
    assert resultado["output"] == ["Com acompanhamento médico."]


def test_anonimiza_nome_em_minusculas() -> None:
    """No texto livre do MedPT o paciente costuma escrever o nome sem maiúscula."""
    resultado = anonymize_text("me chamo joao e sinto dor no peito")

    assert resultado == "[NOME] e sinto dor no peito"


def test_nao_confunde_frase_comum_com_nome() -> None:
    """As expressões "sou a" e "sou o" só viram nome diante de um nome próprio."""
    frase = "sou a melhor pessoa para responder essa pergunta"

    assert anonymize_text(frase) == frase


def test_preserva_o_texto_clinico_apos_o_nome() -> None:
    """O marcador substitui apenas o nome, nunca o restante da queixa."""
    resultado = anonymize_text("Meu nome é Zilda tenho ostio na coronária esquerda")

    assert resultado == "[NOME] tenho ostio na coronária esquerda"


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("Olá, a angina estável é...", "a angina estável é..."),
        ("Boa tarde! O exame indica...", "O exame indica..."),
        ("Bom dia. Trata-se de...", "Trata-se de..."),
        ("A angina estável é...", "A angina estável é..."),
    ],
)
def test_remove_saudacao_ao_paciente(texto, esperado) -> None:
    """O corpus é de médico respondendo a paciente, e começa saudando."""
    assert remover_saudacao(texto) == esperado


def test_descarta_respostas_que_mandam_procurar_atendimento() -> None:
    """Num assistente para profissionais, o leitor é quem atende."""
    data = pd.DataFrame(
        {
            "answer": [
                "Procure um cardiologista para avaliar o caso.",
                "Converse com seu médico sobre a dosagem.",
                "A conduta indicada é otimizar a dose do betabloqueador.",
            ]
        }
    )

    resultado = ajustar_registro(data)

    assert resultado["answer"].tolist() == [
        "A conduta indicada é otimizar a dose do betabloqueador."
    ]


def test_instrucoes_falam_com_profissional() -> None:
    """As instruções enquadram a tarefa como apoio a quem conduz o caso."""
    instrucoes = [
        *config.INSTRUCTION_BY_QUESTION_TYPE.values(),
        config.DEFAULT_INSTRUCTION,
    ]

    for instrucao in instrucoes:
        assert instrucao.startswith(("Descreva", "Responda de forma técnica"))
        assert "procure" not in instrucao.lower()
