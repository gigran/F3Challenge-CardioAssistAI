"""Testes da avaliação do modelo com fine-tuning.

Só as funções que não dependem de GPU são testadas aqui: a leitura do histórico
de loss e a montagem do relatório. O carregamento dos modelos e a geração das
respostas exigem unsloth e CUDA, então rodam apenas no Colab.
"""

from finetuning import config
from finetuning.evaluate import (
    build_report,
    extract_loss_history,
    summarize_question,
)


def test_separa_a_loss_de_treino_da_loss_de_validacao() -> None:
    """O log_history do trainer mistura registros de treino, validação e resumo."""
    log_history = [
        {"loss": 2.10, "learning_rate": 0.0002, "step": 1},
        {"loss": 1.80, "learning_rate": 0.0002, "step": 2},
        {"eval_loss": 1.95, "eval_runtime": 12.3, "step": 2},
        {"train_runtime": 300.0, "total_flos": 1.0, "step": 2},
    ]

    historico = extract_loss_history(log_history)

    assert historico["treino"] == [(1, 2.10), (2, 1.80)]
    assert historico["validacao"] == [(2, 1.95)]


def test_historico_vazio_nao_quebra() -> None:
    """Antes do primeiro passo o histórico ainda não tem nenhuma loss."""
    historico = extract_loss_history([])

    assert historico == {"treino": [], "validacao": []}


def test_encurta_pergunta_longa_para_o_titulo() -> None:
    """As perguntas do MedPT são textos longos e não cabem num título."""
    pergunta = "Tenho 45 anos e sinto uma dor no peito que aparece quando subo escada"

    assert summarize_question(pergunta, limit=30) == "Tenho 45 anos e sinto uma dor…"


def test_mantem_pergunta_curta_inteira() -> None:
    """Uma pergunta que já cabe no título não é cortada."""
    assert summarize_question("O que é angina?") == "O que é angina?"


def test_relatorio_traz_as_tres_respostas_de_cada_exemplo() -> None:
    """O relatório precisa mostrar referência, modelo base e modelo treinado."""
    rows = [
        {
            "instrucao": "Responda esta pergunta sobre tratamento.",
            "pergunta": "Como tratar hipertensão?",
            "referencia": "Com acompanhamento médico.",
            "base": "Resposta genérica do modelo base.",
            "treinado": "Resposta do modelo com fine-tuning.",
        }
    ]

    relatorio = build_report(rows)

    assert config.BASE_MODEL_NAME in relatorio
    assert "## 1. Como tratar hipertensão?" in relatorio
    assert "Com acompanhamento médico." in relatorio
    assert "Resposta genérica do modelo base." in relatorio
    assert "Resposta do modelo com fine-tuning." in relatorio


def test_relatorio_numera_os_exemplos() -> None:
    """Cada exemplo vira uma seção numerada, para citar no vídeo da entrega."""
    rows = [
        {
            "instrucao": "i",
            "pergunta": f"pergunta {numero}",
            "referencia": "r",
            "base": "b",
            "treinado": "t",
        }
        for numero in range(1, 4)
    ]

    relatorio = build_report(rows)

    assert "## 1. pergunta 1" in relatorio
    assert "## 2. pergunta 2" in relatorio
    assert "## 3. pergunta 3" in relatorio
