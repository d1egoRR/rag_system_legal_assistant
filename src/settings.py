"""
Centralized settings module for Legal Assistant RAG.
Loads and types environment variables from the root .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project root directory (one level up from src/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env located at the project root
load_dotenv(BASE_DIR / ".env")


def _get_bool(env_var: str, default: bool = False) -> bool:
    """Helper to parse boolean values from environment variables."""
    val = os.getenv(env_var)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y")


# --- Authentication & Credentials ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# --- System Paths ---
CONTRACTS_DIR: Path = (BASE_DIR / "contract_files").resolve()

# Normalize CHROMA_DB_PATH in case of relative '../' references
_chroma_env = os.getenv("CHROMA_DB_PATH", "chroma_db").strip().strip('"').strip("'")
_chroma_clean = _chroma_env.replace("../", "")
CHROMA_DB_PATH: Path = (BASE_DIR / _chroma_clean).resolve()

# --- LLM & Embedding Models ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip().strip('"').strip("'")
QUERY_MODEL: str = os.getenv("QUERY_MODEL", "gpt-4o-mini").strip().strip('"').strip("'")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt-4o").strip().strip('"').strip("'")

# --- Chunking Parameters ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "5000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "1000"))

# --- Retriever Configuration ---
SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "mmr").strip().strip('"').strip("'")
SEARCH_K: int = int(os.getenv("SEARCH_K", "2"))
MMR_DIVERSITY_LAMBDA: float = float(os.getenv("MMR_DIVERSITY_LAMBDA", "0.7"))
MMR_FETCH_K: int = int(os.getenv("MMR_FETCH_K", "20"))

# --- Hybrid Search ---
ENABLE_HYBRID_SEARCH: bool = _get_bool("ENABLE_HYBRID_SEARCH", default=True)
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
