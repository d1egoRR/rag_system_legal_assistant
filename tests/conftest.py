"""
Pytest fixtures for Legal Assistant RAG test suite.
"""

from pathlib import Path
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_contract_text() -> str:
    """Provides a sample lease contract text snippet for testing."""
    return (
        "CONTRATO DE ARRENDAMIENTO DE VIVIENDA\n"
        "En Madrid, a 1 de enero de 2026.\n"
        "REUNIDOS: De una parte, D. Juan Pérez con DNI 12345678A como ARRENDADOR, "
        "y Dª. Ana Gómez con DNI 87654321B como ARRENDATARIA.\n"
        "CLÁUSULAS: Primera.- Objeto: Piso situado en Calle Mayor 10, 2ºB, Madrid.\n"
        "Segunda.- Renta: Mil euros (1.000 €) mensuales abonables del 1 al 5 de cada mes.\n"
        "Tercera.- Duración: Un (1) año prorrogable."
    )


@pytest.fixture
def sample_contract_document(sample_contract_text: str) -> Document:
    """Provides a Document instance with contract text and metadata."""
    return Document(
        page_content=sample_contract_text,
        metadata={
            "source": "contract_files/sample_test_contract.pdf",
            "page": 0,
            "total_pages": 1,
        },
    )


@pytest.fixture
def isolated_chroma_dir(tmp_path: Path) -> Path:
    """Provides a temporary, isolated directory for ChromaDB vector store testing."""
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir
