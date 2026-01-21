"""
Componentes visuales para la UI conversacional del agente RAG.
"""

from typing import List

import streamlit as st


def init_chat_state() -> None:
    """
    Inicializa el estado del chat si no existe.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    """
    Renderiza el historial de mensajes estilo chat.
    """
    for i, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        content = message.get("content")

        # Saltar mensajes corruptos
        if content is None:
            continue

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                sources = message.get("sources", [])
                if sources:
                    with st.expander("📚 Fuentes"):
                        for s in sources:
                            st.markdown(f"- **{s}**")


def append_user_message(content: str) -> None:
    """
    Agrega un mensaje de usuario al estado del chat.

    :param content: Contenido del mensaje del usuario.
    :type content: str
    """
    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(
    content: str,
    sources: List[str] | None = None,
) -> None:
    """
    Agrega un mensaje del asistente al estado del chat.

    :param content: Contenido del mensaje del asistente.
    :type content: str
    :param sources: Lista de fuentes asociadas al mensaje.
    :type sources: List[str] | None
    """
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "sources": sources or [],
        }
    )
