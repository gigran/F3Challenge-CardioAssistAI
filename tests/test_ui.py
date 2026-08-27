"""Testes da comunicação HTTP utilizada pela interface Streamlit."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app import ui
from app.ui import request_assistance


def create_response(
    status_code: int,
    *,
    json_data: Any | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    """Cria uma resposta HTTP associada a uma requisição simulada."""
    request = httpx.Request("POST", "http://127.0.0.1:8000/assist")

    if content is not None:
        return httpx.Response(status_code, content=content, request=request)

    return httpx.Response(status_code, json=json_data, request=request)


def test_envia_pergunta_e_retorna_resposta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica URL, corpo, timeout e resposta da chamada à API."""
    received_request: dict[str, object] = {}
    expected_result = {
        "answer": "Resposta simulada.",
        "sources": ["data/knowledge_base/hipertensao.md"],
        "safety_alerts": [],
        "safety_status": "revisao_obrigatoria",
        "requires_human_review": True,
    }

    def fake_post(
        url: str,
        *,
        json: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        received_request.update(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        return create_response(200, json_data=expected_result)

    monkeypatch.setattr(ui.httpx, "post", fake_post)

    result = request_assistance(
        "http://127.0.0.1:8000/",
        "Paciente sintético com hipertensão.",
    )

    assert result == expected_result
    assert received_request == {
        "url": "http://127.0.0.1:8000/assist",
        "json": {"question": "Paciente sintético com hipertensão."},
        "timeout": 300.0,
    }


def test_propaga_erro_http_da_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite que a interface trate respostas HTTP malsucedidas."""
    fake_post: Callable[..., httpx.Response] = lambda *args, **kwargs: create_response(
        500, json_data={"detail": "Erro interno simulado."}
    )
    monkeypatch.setattr(ui.httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        request_assistance("http://127.0.0.1:8000", "Pergunta sintética.")


def test_propaga_timeout_da_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Permite que a interface apresente a mensagem específica de timeout."""
    request = httpx.Request("POST", "http://127.0.0.1:8000/assist")

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ReadTimeout("Tempo limite simulado.", request=request)

    monkeypatch.setattr(ui.httpx, "post", fake_post)

    with pytest.raises(httpx.ReadTimeout, match="Tempo limite simulado"):
        request_assistance("http://127.0.0.1:8000", "Pergunta sintética.")


def test_rejeita_json_com_tipo_inesperado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejeita uma lista JSON quando o contrato exige um objeto."""
    fake_post: Callable[..., httpx.Response] = lambda *args, **kwargs: create_response(
        200, json_data=["resposta inválida"]
    )
    monkeypatch.setattr(ui.httpx, "post", fake_post)

    with pytest.raises(TypeError, match="JSON em formato inesperado"):
        request_assistance("http://127.0.0.1:8000", "Pergunta sintética.")


def test_rejeita_conteudo_que_nao_e_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejeita conteúdo textual que não possa ser interpretado como JSON."""
    fake_post: Callable[..., httpx.Response] = lambda *args, **kwargs: create_response(
        200, content=b"resposta sem JSON"
    )
    monkeypatch.setattr(ui.httpx, "post", fake_post)

    with pytest.raises(ValueError):
        request_assistance("http://127.0.0.1:8000", "Pergunta sintética.")
