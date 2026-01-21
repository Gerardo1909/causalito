"""
Punto de entrada para UI Streamlit del agente RAG.
"""

import streamlit as st

from main import build_agent
from agent.rag_agent import RAGAgent
from components import (
    init_chat_state,
    render_chat_history,
    append_user_message,
    append_assistant_message,
)


@st.cache_resource
def get_agent() -> RAGAgent:
    return build_agent()


st.set_page_config(
    page_title="Causalito | Agente RAG 🅱️",
    layout="wide",
)

st.title("🅱️ Causalito — Inferencia Causal Bayesiana")

init_chat_state()
render_chat_history()

question = st.chat_input("Haz una pregunta sobre inferencia causal...")

if question:
    append_user_message(question)

    with st.chat_message("assistant"):
        with st.spinner("Razonando..."):
            try:
                answer, sources = get_agent().answer(question)
            except Exception as e:
                answer = "Lo siento, ha ocurrido un error al procesar tu pregunta."
                sources = []
                st.error(f"Error: {e}")

        st.markdown(answer)

        if sources:
            with st.expander("📚 Fuentes"):
                for s in sources:
                    st.markdown(f"- **{s}**")

    append_assistant_message(answer, sources)