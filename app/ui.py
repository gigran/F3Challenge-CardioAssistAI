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


def request_assistance(
    api_url: str,
    question: str,
    llm_provider: str,
    patient_code: str = "",
) -> dict[str, Any]:
    """Envia a pergunta para a API e devolve a resposta validada como JSON."""
    response = httpx.post(
        f"{api_url.rstrip('/')}/assist",
        json={
            "question": question,
            "llm_provider": llm_provider,
            "patient_code": patient_code,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    result = response.json()

    if not isinstance(result, dict):
        raise TypeError("A API retornou um JSON em formato inesperado.")

    return result


def render_result(result: dict[str, Any], patient_code: str = "") -> None:
    """Apresenta resposta, alertas, fontes e indicação de revisão humana."""
    # Um código informado que não existe precisa ficar visível: sem isso, a
    # resposta parece ter considerado o prontuário quando não considerou.
    if patient_code and not result.get("patient_found", False):
        st.error(
            f"O paciente {patient_code} não foi encontrado no prontuário. "
            "A resposta abaixo não considera dados deste paciente."
        )

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

    pending_exams = result.get("pending_exams", [])
    if pending_exams:
        st.subheader("Exames pendentes")
        for exam in pending_exams:
            st.write(f"- {exam}")

    patient_summary = result.get("patient_summary", "")
    if patient_summary:
        with st.expander("Prontuário usado como contexto"):
            st.text(patient_summary)

    st.subheader("Fontes consultadas")
    sources = result.get("sources", [])
    if sources:
        for source in sources:
            st.write(f"- `{source}`")
    else:
        st.write("Nenhuma fonte foi informada pela API.")

    if result.get("requires_human_review", True):
        st.warning(
            "Resposta de apoio à decisão: exige validação do profissional responsável."
        )


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
        provider_options = ["ollama", "lora", "openai"]
        default_provider_index = (
            provider_options.index(settings.llm_provider)
            if settings.llm_provider in provider_options
            else 0
        )
        llm_provider = st.selectbox(
            "Modelo de linguagem",
            options=provider_options,
            index=default_provider_index,
            format_func=lambda provider: {
                "ollama": "Ollama — Gratuita",
                "lora": "Ollama — treinada",
                "openai": "OpenAI — API paga",
            }[provider],
        )
        st.caption("Embeddings: Ollama — nomic-embed-text")

    with st.form("assist_form"):
        patient_code = st.text_input(
            "Código do paciente (opcional)",
            placeholder="Exemplo: PAC-006",
            max_chars=20,
            help=(
                "Quando informado, o assistente consulta o prontuário sintético "
                "e verifica os exames pendentes antes de responder."
            ),
        )
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
            result = request_assistance(
                api_url,
                normalized_question,
                llm_provider,
                patient_code.strip(),
            )
    except httpx.ReadTimeout:
        logger.exception(
            "A API excedeu o limite de %.0f segundos para responder.",
            REQUEST_TIMEOUT_SECONDS,
        )
        st.error(
            "A análise ultrapassou o tempo máximo de 300 segundos. "
            "Confirme se o provedor selecionado está configurado e disponível."
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

    render_result(result, patient_code.strip())


if __name__ == "__main__":
    main()
