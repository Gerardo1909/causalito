import streamlit as st
from agent.rag_agent import RAGAgent
from main import build_agent


@st.cache_resource
def get_agent() -> RAGAgent:
    return build_agent()


st.set_page_config(
    page_title="Causalito | RAG",
    page_icon="📊",
    layout="centered",
)

# Sidebar
with st.sidebar:
    st.header("Causalito")
    st.caption("Agente RAG para inferencia causal bayesiana")
    if st.button("Limpiar conversacion"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**Fuentes:** PDFs academicos indexados")

st.title("Causalito")

# Estado inicial con mensaje de bienvenida
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola, soy Causalito. Puedo responder preguntas sobre inferencia causal bayesiana basandome en bibliografia academica. ¿En que puedo ayudarte?",
            "sources": [],
        }
    ]

# Render historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Fuentes"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Input
if question := st.chat_input("Pregunta sobre inferencia causal..."):
    # Mostrar mensaje usuario inmediatamente
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        # Mostrar status del retrieval
        with st.status("Buscando contexto relevante...", expanded=False) as status:
            try:
                agent = get_agent()
                docs, sources, scores = agent.retriever.retrieve(question)
                status.update(
                    label=f"Recuperados {len(docs)} fragmentos de {len(sources)} fuentes"
                )

                if docs:
                    prompt = agent.prompt_builder.build(question, docs)
                    answer = agent.llm.generate(prompt)
                else:
                    answer = (
                        "No encontre informacion relevante en los documentos indexados."
                    )
                    sources = []
            except Exception as e:
                answer = f"Error al procesar: {str(e)}"
                sources = []
                status.update(label="Error", state="error")

        st.markdown(answer)
        if sources:
            with st.expander("Fuentes"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
