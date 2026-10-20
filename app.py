"""
Streamlit application for the Legal Assistant RAG system.
Encapsulates UI rendering and state management into the LegalAssistantApp class.
"""

from typing import Optional
import streamlit as st

from src.rag_service import RAGService


@st.cache_resource
def get_rag_service() -> RAGService:
    """
    Instantiate and cache the RAGService instance across Streamlit reruns.

    Returns:
        Cached RAGService singleton instance.
    """
    return RAGService()


class LegalAssistantApp:
    """Streamlit web application for the Legal Assistant RAG system."""

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        """
        Initialize the Streamlit UI application.

        Args:
            rag_service: Optional RAGService instance. Defaults to the cached singleton.
        """
        self.rag_service = rag_service or get_rag_service()

    def run(self) -> None:
        """Execute the application rendering lifecycle."""
        self._setup_page_config()
        self._init_session_state()
        self._render_header()
        self._render_sidebar()
        self._render_main_layout()
        self._handle_user_input()
        self._render_footer()

    def _setup_page_config(self) -> None:
        """Configure Streamlit page title, icon, and wide layout."""
        st.set_page_config(
            page_title="Sistema RAG - Asistente Legal",
            page_icon="⚖️",
            layout="wide",
        )

    def _init_session_state(self) -> None:
        """Initialize chat message history in Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def _render_header(self) -> None:
        """Render the page title and separator."""
        st.title("⚖️ Sistema RAG - Asistente Legal")
        st.divider()

    def _render_sidebar(self) -> None:
        """Render sidebar system info, models, and clear chat button."""
        with st.sidebar:
            st.header("📋 Información del Sistema")

            retriever_info = self.rag_service.get_retriever_info()

            st.markdown("**🔍 Retriever:**")
            st.info(f"Tipo: {retriever_info['tipo']}")

            st.markdown("**🤖 Modelos:**")
            st.info(
                f"Consultas: {retriever_info['query_model']}\n"
                f"Respuestas: {retriever_info['generation_model']}"
            )

            st.divider()

            if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

    def _render_main_layout(self) -> None:
        """Render the two-column layout for chat and retrieved documents."""
        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_chat_column()

        with col2:
            self._render_documents_column()

    def _render_chat_column(self) -> None:
        """Render the conversation chat history."""
        st.markdown("### 💬 Chat")
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _render_documents_column(self) -> None:
        """Render the citations and retrieved contract chunks of the latest query."""
        st.markdown("### 📄 Documentos Relevantes")

        messages = st.session_state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if last_message.get("role") == "assistant" and "docs" in last_message:
                docs = last_message["docs"]
                if docs:
                    for doc in docs:
                        with st.expander(f"📄 Fragmento {doc['fragmento']}", expanded=False):
                            st.markdown(f"**Fuente:** {doc['fuente']}")
                            st.markdown(f"**Página:** {doc['pagina']}")
                            st.markdown("**Contenido:**")
                            st.text(doc["contenido"])

    def _handle_user_input(self) -> None:
        """Capture user input, query RAGService, and refresh chat."""
        if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
            # Append user message to history
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            st.session_state["messages"].append({"role": "user", "content": prompt})

            # Generate model answer with spinner indicator
            with st.spinner("🔍 Analizando..."):
                response, docs = self.rag_service.query(prompt)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response, "docs": docs}
                )

            # Reload to display the new messages immediately
            st.rerun()

    def _render_footer(self) -> None:
        """Render application footer."""
        st.divider()
        st.markdown(
            "<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con RAG</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app = LegalAssistantApp()
    app.run()
