"""
Unit tests for src/settings.py module.
"""

from pathlib import Path
import pytest
from src import settings


@pytest.mark.unit
def test_settings_paths_are_valid_objects():
    """Verify that path settings are Path instances."""
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.CONTRACTS_DIR, Path)
    assert isinstance(settings.CHROMA_DB_PATH, Path)
    assert settings.BASE_DIR.exists()


@pytest.mark.unit
def test_settings_types():
    """Verify that settings types match their expected schemas."""
    assert isinstance(settings.CHUNK_SIZE, int)
    assert settings.CHUNK_SIZE > 0
    assert isinstance(settings.CHUNK_OVERLAP, int)
    assert settings.CHUNK_OVERLAP >= 0
    assert isinstance(settings.SEARCH_K, int)
    assert isinstance(settings.MMR_DIVERSITY_LAMBDA, float)
    assert isinstance(settings.SIMILARITY_THRESHOLD, float)
    assert isinstance(settings.ENABLE_HYBRID_SEARCH, bool)
    assert isinstance(settings.EMBEDDING_MODEL, str)
    assert len(settings.EMBEDDING_MODEL) > 0
