"""Interface Streamlit do CardioAssist AI."""

import logging
from typing import Any

import httpx
import streamlit as st

from app.config import get_settings

settings = get_settings()
DEFAULT_API_URL = f"http://{settings.app_host}:{settings.app_port}"
REQUEST_TIMEOUT_SECONDS = 300.0
logger = logging.getLogger(__name__)


def request_assistance(api_url: str, question: str) -> dict[str, Any]:
    """Envia a pergunta para a API e devolve a resposta validada como JSON."""
    response = httpx.post(
        f"{api_url.rstrip('/')}/assist",
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    result = response.json()

    if not isinstance(result, dict):
        raise TypeError("A API retornou um JSON em formato inesperado.")

    return result


def render_result(result: dict[str, Any]) -> None:
    """Apresenta resposta, alertas, fontes e indicação de revisão humana."""
    safety_status = result.get("safety_status", "revisao_obrigatoria")
    safety_alerts = result.get("safety_alerts", [])

    if safety_status == "emergencia":
        st.error("Status de segurança: emergência")
    elif safety_status == "atencao":
        st.warning("Status de segurança: atenção")
    else:
        st.info("Status de segurança: revisão obrigatória")

    st.subheader("Resposta")
    st.markdown(result.get("answer", "A API não retornou uma resposta."))

    if safety_alerts:
        st.subheader("Alertas identificados")
        for alert in safety_alerts:
            st.warning(alert)

    st.subheader("Fontes consultadas")
    sources = result.get("sources", [])
    if sources:
        for source in sources:
            st.write(f"- `{source}`")
    else:
        st.write("Nenhuma fonte foi informada pela API.")

    if result.get("requires_human_review", True):
        st.warning("Esta resposta deve ser revisada por um profissional de saúde.")


def main() -> None:
    """Configura e executa a página principal da interface."""
    st.set_page_config(
        page_title="CardioAssist AI",
        page_icon="❤️",
        layout="centered",
    )

    st.title("CardioAssist AI")
    st.caption("Protótipo educacional de apoio à decisão clínica em cardiologia")

    st.error(
        "Utilize somente informações sintéticas. "
        "Não informe nomes, documentos ou outros dados de pacientes reais."
    )

    with st.sidebar:
        st.header("Configuração")
        api_url = st.text_input("Endereço da API", value=DEFAULT_API_URL)
        st.caption(f"Provedor configurado: {settings.llm_provider}")

    with st.form("assist_form"):
        question = st.text_area(
            "Dados sintéticos e pergunta clínica",
            placeholder=(
                "Exemplo: Paciente sintético com pressão 190/125 e dor "
                "torácica. Quais são os sinais de alerta?"
            ),
            height=180,
            max_chars=4000,
        )
        submitted = st.form_submit_button(
            "Analisar caso sintético",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    normalized_question = question.strip()
    if len(normalized_question) < 3:
        st.warning("Digite uma pergunta com pelo menos três caracteres.")
        return

    try:
        with st.spinner("Consultando o fluxo RAG e o modelo de linguagem..."):
            result = request_assistance(api_url, normalized_question)
    except httpx.ReadTimeout:
        logger.exception(
            "A API excedeu o limite de %.0f segundos para responder.",
            REQUEST_TIMEOUT_SECONDS,
        )
        st.error(
            "A análise ultrapassou o tempo máximo de 300 segundos. "
            "Confirme se o Ollama e o modelo configurado estão funcionando."
        )
        return
    except httpx.ConnectError:
        logger.exception("Não foi possível conectar à API em %s.", api_url)
        st.error(
            "Não foi possível conectar à API. Confirme se o FastAPI está "
            "em execução e se o endereço informado está correto."
        )
        return
    except httpx.HTTPStatusError as error:
        logger.exception(
            "A API respondeu com o código HTTP %s.",
            error.response.status_code,
        )
        st.error(
            "A API recusou ou não conseguiu processar a solicitação. "
            f"Código HTTP: {error.response.status_code}."
        )
        return
    except httpx.RequestError:
        logger.exception("Falha inesperada na comunicação com a API.")
        st.error("Ocorreu um erro ao processar a comunicação com a API.")
        return
    except (TypeError, ValueError):
        logger.exception("A resposta da API não contém o formato JSON esperado.")
        st.error("A API retornou uma resposta em formato inválido.")
        return

    render_result(result)


if __name__ == "__main__":
    main()
