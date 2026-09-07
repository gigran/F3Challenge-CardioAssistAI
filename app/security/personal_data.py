"""Detecção de dados pessoais no texto enviado à API.

O protótipo trabalha apenas com dados sintéticos, e a interface já avisa isso ao
profissional. Este módulo transforma o aviso em regra: quando a configuração
USE_SYNTHETIC_DATA_ONLY está ligada, a API recusa perguntas que contenham
identificadores diretos.

São verificados apenas identificadores objetivos. Nome de pessoa não entra na
lista de propósito: detectar nome em texto livre gera falso positivo demais e
bloquearia perguntas clínicas legítimas.
"""

import re

# A ordem não importa: todos os padrões são testados.
PADROES_DE_DADOS_PESSOAIS = {
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    # Exige de 10 a 11 dígitos, para não confundir com valores clínicos como
    # uma pressão de 190/125 ou uma frequência cardíaca.
    "telefone": re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b"),
    "CEP": re.compile(r"\b\d{5}-\d{3}\b"),
}


def encontrar_dados_pessoais(texto: str) -> list[str]:
    """Devolve os tipos de dado pessoal encontrados, em ordem alfabética.

    Devolve apenas o tipo, nunca o valor encontrado: assim o dado pessoal não é
    repetido na mensagem de erro nem no log.
    """
    return sorted(
        tipo
        for tipo, padrao in PADROES_DE_DADOS_PESSOAIS.items()
        if padrao.search(texto)
    )
