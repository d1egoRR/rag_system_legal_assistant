"""
Unit tests for the src/rag_service.py module.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from src.rag_service import RAGService


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    """Ensure test settings provide a default fake OpenAI API key."""
    monkeypatch.setattr("src.settings.OPENAI_API_KEY", "sk-test-fake-key")


@pytest.mark.unit
def test_rag_service_initialization_missing_api_key(monkeypatch):
    """Verify ValueError is raised when OpenAI API key is missing."""
    monkeypatch.setattr("src.settings.OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OpenAI API key must be configured in settings."):
        RAGService()


@pytest.mark.unit
def test_rag_service_initialization_defaults(monkeypatch):
    """Verify RAGService initializes with settings defaults and builds components."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", True)

    with (
        patch("src.rag_service.OpenAIEmbeddings") as mock_embeddings,
        patch("src.rag_service.Chroma") as mock_chroma,
        patch("src.rag_service.ChatOpenAI") as mock_chat,
        patch("src.rag_service.MultiQueryRetriever.from_llm") as mock_multi_retriever,
        patch("src.rag_service.EnsembleRetriever") as mock_ensemble,
    ):
        mock_vectorstore_instance = MagicMock()
        mock_chroma.return_value = mock_vectorstore_instance
        mock_ensemble_instance = MagicMock()
        mock_ensemble.return_value = mock_ensemble_instance

        service = RAGService()

        assert service.api_key == "sk-test-fake-key"
        assert service.retriever == mock_ensemble_instance
        mock_embeddings.assert_called_once()
        mock_chroma.assert_called_once()
        assert mock_chat.call_count == 2
        mock_multi_retriever.assert_called_once()
        mock_ensemble.assert_called_once()


@pytest.mark.unit
def test_rag_service_initialization_without_hybrid_search(monkeypatch):
    """Verify RAGService returns MultiQueryRetriever when hybrid search is disabled."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", False)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm") as mock_multi_retriever,
        patch("src.rag_service.EnsembleRetriever") as mock_ensemble,
    ):
        mock_multi_instance = MagicMock()
        mock_multi_retriever.return_value = mock_multi_instance

        service = RAGService()

        assert service.retriever == mock_multi_instance
        mock_ensemble.assert_not_called()


@pytest.mark.unit
def test_format_documents():
    """Verify format_documents formats text, header, source, and page correctly."""
    docs = [
        Document(
            page_content="Cláusula Primera: Inmueble en Arriendo.",
            metadata={"source": "contracts/sample_lease.pdf", "page": 1},
        ),
        Document(
            page_content="Cláusula Segunda: Renta de 1200 euros.",
            metadata={"source": "another_contract.pdf"},
        ),
        Document(
            page_content="Cláusula Tercera: Depósito de fianza.",
            metadata={},
        ),
    ]

    formatted = RAGService.format_documents(docs)

    assert "[Fragmento 1] - Fuente: sample_lease.pdf - Pagina: 1" in formatted
    assert "Cláusula Primera: Inmueble en Arriendo." in formatted
    assert "[Fragmento 2] - Fuente: another_contract.pdf" in formatted
    assert "[Fragmento 3]" in formatted
    assert "Cláusula Tercera: Depósito de fianza." in formatted


@pytest.mark.unit
def test_format_sources():
    """Verify format_sources produces structured dictionaries and respects max_content_length."""
    long_content = "A" * 1500
    docs = [
        Document(
            page_content=long_content,
            metadata={"source": "dir/contract_one.pdf", "page": 2},
        ),
        Document(
            page_content="Short text",
            metadata={"source": "contract_two.pdf", "page": 5},
        ),
        Document(
            page_content="Third document",
            metadata={},
        ),
    ]

    # Test with max_k limit and custom length limit
    sources = RAGService.format_sources(docs, max_k=2, max_content_length=500)

    assert len(sources) == 2
    assert sources[0]["fragmento"] == 1
    assert len(sources[0]["contenido"]) == 500
    assert sources[0]["fuente"] == "contract_one.pdf"
    assert sources[0]["pagina"] == 2

    assert sources[1]["fragmento"] == 2
    assert sources[1]["contenido"] == "Short text"
    assert sources[1]["fuente"] == "contract_two.pdf"
    assert sources[1]["pagina"] == 5

    # Test default metadata fallback
    all_sources = RAGService.format_sources(docs)
    assert len(all_sources) == 3
    assert all_sources[2]["fuente"] == "Desconocida"
    assert all_sources[2]["pagina"] == "N/A"


@pytest.mark.unit
def test_rag_service_retrieve(monkeypatch):
    """Verify retrieve delegates directly to the underlying retriever."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", False)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm"),
    ):
        service = RAGService()
        service.retriever = MagicMock()
        expected_docs = [Document(page_content="Test chunk")]
        service.retriever.invoke.return_value = expected_docs

        result = service.retrieve("¿Cuál es la renta?")

        service.retriever.invoke.assert_called_once_with("¿Cuál es la renta?")
        assert result == expected_docs


@pytest.mark.unit
def test_rag_service_generate(monkeypatch):
    """Verify generate invokes generation chain with question and formatted context."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", False)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm"),
    ):
        service = RAGService()
        service.generation_chain = MagicMock()
        service.generation_chain.invoke.return_value = "La renta es de 1000 euros."

        result = service.generate(
            question="¿Cuál es la renta?",
            context="Cláusula Renta: 1000 euros.",
        )

        service.generation_chain.invoke.assert_called_once_with(
            {"question": "¿Cuál es la renta?", "context": "Cláusula Renta: 1000 euros."}
        )
        assert result == "La renta es de 1000 euros."


@pytest.mark.unit
def test_rag_service_query_single_pass_pipeline(monkeypatch):
    """
    Verify query coordinates retrieval, generation, and source formatting in a single pass.
    Ensure retrieve is called exactly once (preventing duplicate queries).
    """
    monkeypatch.setattr("src.settings.SEARCH_K", 1)
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", False)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm"),
    ):
        service = RAGService()

        test_doc = Document(
            page_content="Renta mensual: 800 EUR.",
            metadata={"source": "contrato.pdf", "page": 1},
        )
        service.retrieve = MagicMock(return_value=[test_doc])
        service.generate = MagicMock(return_value="La renta mensual es de 800 EUR.")

        answer, sources = service.query("¿Cuánto es la renta?")

        # Check single-pass execution: retrieve called exactly once!
        service.retrieve.assert_called_once_with("¿Cuánto es la renta?")
        service.generate.assert_called_once()
        assert answer == "La renta mensual es de 800 EUR."
        assert len(sources) == 1
        assert sources[0]["fuente"] == "contrato.pdf"
        assert sources[0]["contenido"] == "Renta mensual: 800 EUR."


@pytest.mark.unit
def test_rag_service_query_handles_exception(monkeypatch):
    """Verify query catches errors gracefully and returns an error message."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", False)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm"),
    ):
        service = RAGService()
        service.retrieve = MagicMock(side_effect=RuntimeError("Chroma connection timeout"))

        answer, sources = service.query("Cualquier pregunta")

        assert "Error al consultar el sistema RAG: Chroma connection timeout" in answer
        assert sources == []


@pytest.mark.unit
def test_rag_service_get_retriever_info(monkeypatch):
    """Verify get_retriever_info returns correct configuration metadata."""
    monkeypatch.setattr("src.settings.ENABLE_HYBRID_SEARCH", True)

    with (
        patch("src.rag_service.OpenAIEmbeddings"),
        patch("src.rag_service.Chroma"),
        patch("src.rag_service.ChatOpenAI"),
        patch("src.rag_service.MultiQueryRetriever.from_llm"),
        patch("src.rag_service.EnsembleRetriever"),
    ):
        service = RAGService()
        info = service.get_retriever_info()

        assert "Híbrido" in info["tipo"]
        assert info["hybrid_search"] is True
        assert "query_model" in info
        assert "generation_model" in info
