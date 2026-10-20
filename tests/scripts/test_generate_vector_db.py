"""
Unit tests for scripts/generate_vector_db.py.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.generate_vector_db import create_vector_database


@pytest.mark.unit
def test_create_vector_database_missing_contracts_dir(tmp_path: Path):
    """Verify FileNotFoundError is raised when contracts directory does not exist."""
    non_existent = tmp_path / "non_existent_contracts"
    with pytest.raises(FileNotFoundError):
        create_vector_database(contracts_dir=str(non_existent))


@pytest.mark.unit
def test_create_vector_database_missing_api_key(monkeypatch):
    """Verify ValueError is raised when no API key is provided or present in settings."""
    monkeypatch.setattr("src.settings.OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        create_vector_database(openai_api_key="")


@pytest.mark.unit
def test_create_vector_database_clears_existing_directory(tmp_path: Path):
    """Verify existing persistence directory contents are cleared prior to vector store generation."""
    test_persist = tmp_path / "existing_chroma_db"
    test_persist.mkdir(parents=True, exist_ok=True)
    marker_file = test_persist / "old_data.sqlite3"
    marker_file.write_text("stale_data")

    test_contracts = tmp_path / "contracts"
    test_contracts.mkdir()

    with (
        patch("scripts.generate_vector_db.PyPDFDirectoryLoader") as mock_loader,
        patch("scripts.generate_vector_db.OpenAIEmbeddings"),
        patch("scripts.generate_vector_db.Chroma.from_documents") as mock_chroma,
    ):
        mock_loader.return_value.load.return_value = []
        mock_chroma.return_value = MagicMock()

        create_vector_database(
            contracts_dir=str(test_contracts),
            persist_directory=str(test_persist),
            openai_api_key="sk-fake-test-key",
            run_sample_query=False,
        )

    # Old directory contents must be wiped clean
    assert not marker_file.exists()
