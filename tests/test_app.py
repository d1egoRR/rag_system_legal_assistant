"""
Unit tests for app.py (LegalAssistantApp).
"""

from unittest.mock import MagicMock, patch
import pytest

from app import LegalAssistantApp


class MockSessionState(dict):
    """Mock session state supporting both dict key and attribute access like Streamlit."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def mock_rag_service():
    """Provides a mocked RAGService instance."""
    mock_service = MagicMock()
    mock_service.get_retriever_info.return_value = {
        "tipo": "MMR (MMR)",
        "search_type": "mmr",
        "search_k": 2,
        "hybrid_search": False,
        "query_model": "gpt-4o-mini",
        "generation_model": "gpt-4o",
    }
    mock_service.query.return_value = ("Test answer", [])
    return mock_service


@pytest.mark.unit
def test_app_initialization(mock_rag_service):
    """Verify LegalAssistantApp initializes with the provided RAGService."""
    app = LegalAssistantApp(rag_service=mock_rag_service)
    assert app.rag_service == mock_rag_service


@pytest.mark.unit
def test_app_init_session_state_creates_empty_list(mock_rag_service):
    """Verify _init_session_state sets up messages list when empty."""
    app = LegalAssistantApp(rag_service=mock_rag_service)
    mock_state = MockSessionState()
    with patch("app.st.session_state", mock_state):
        app._init_session_state()
        assert "messages" in mock_state
        assert mock_state["messages"] == []


@pytest.mark.unit
def test_app_init_session_state_preserves_existing(mock_rag_service):
    """Verify _init_session_state preserves existing messages in state."""
    app = LegalAssistantApp(rag_service=mock_rag_service)
    existing_messages = [{"role": "user", "content": "Hola"}]
    mock_state = MockSessionState(messages=existing_messages)
    with patch("app.st.session_state", mock_state):
        app._init_session_state()
        assert len(mock_state["messages"]) == 1
        assert mock_state["messages"][0]["content"] == "Hola"


@pytest.mark.unit
def test_app_run_lifecycle(mock_rag_service):
    """Verify app.run invokes all sub-rendering methods in the expected lifecycle."""
    app = LegalAssistantApp(rag_service=mock_rag_service)

    with (
        patch.object(app, "_setup_page_config") as mock_setup,
        patch.object(app, "_init_session_state") as mock_init_state,
        patch.object(app, "_render_header") as mock_header,
        patch.object(app, "_render_sidebar") as mock_sidebar,
        patch.object(app, "_render_main_layout") as mock_layout,
        patch.object(app, "_handle_user_input") as mock_input,
        patch.object(app, "_render_footer") as mock_footer,
    ):
        app.run()

        mock_setup.assert_called_once()
        mock_init_state.assert_called_once()
        mock_header.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_layout.assert_called_once()
        mock_input.assert_called_once()
        mock_footer.assert_called_once()
