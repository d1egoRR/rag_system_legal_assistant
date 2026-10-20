"""
Module to vectorize legal PDF contracts and persist them into a Chroma (SQLite) vector database.
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Ensure src and settings can be imported from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_vector_db")


def create_vector_database(
    contracts_dir: Optional[str] = None,
    persist_directory: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    run_sample_query: bool = True,
) -> Chroma:
    """
    Load PDF contracts from a directory, split their content into chunks,
    and persist them into a Chroma vector store.
    If the vector store already exists, it removes it before generating a fresh one.

    Args:
        contracts_dir: Directory containing PDF contracts. Defaults to settings.CONTRACTS_DIR.
        persist_directory: Destination path for Chroma DB. Defaults to settings.CHROMA_DB_PATH.
        chunk_size: Maximum character chunk size. Defaults to settings.CHUNK_SIZE.
        chunk_overlap: Character overlap between consecutive chunks. Defaults to settings.CHUNK_OVERLAP.
        embedding_model: OpenAI embedding model name. Defaults to settings.EMBEDDING_MODEL.
        openai_api_key: Optional API key override. Defaults to settings.OPENAI_API_KEY.
        run_sample_query: If True, executes a sample similarity query after indexing.

    Returns:
        Chroma: Configured and persisted vector store instance.
    """
    api_key = openai_api_key or settings.OPENAI_API_KEY
    if not api_key:
        error_msg = (
            "OPENAI_API_KEY not found. "
            "Please configure it in your .env file or pass it as an argument."
        )
        logger.error(f"{error_msg}")
        raise ValueError(error_msg)

    # Resolve contracts directory
    c_dir = contracts_dir or str(settings.CONTRACTS_DIR)
    contracts_path = Path(c_dir)
    if not contracts_path.is_absolute():
        contracts_path = (PROJECT_ROOT / contracts_path).resolve()

    # Resolve persist directory
    p_dir = persist_directory or str(settings.CHROMA_DB_PATH)
    persist_path = Path(p_dir)
    if not persist_path.is_absolute():
        persist_path = (PROJECT_ROOT / persist_path).resolve()

    c_size = chunk_size or settings.CHUNK_SIZE
    c_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    emb_model = embedding_model or settings.EMBEDDING_MODEL

    if not contracts_path.exists() or not contracts_path.is_dir():
        error_msg = f"Contracts directory '{contracts_path}' does not exist or is not a valid directory."
        logger.error(f"{error_msg}")
        raise FileNotFoundError(error_msg)

    # 1. Check if vector DB directory already exists and clear its contents for a clean rebuild
    if persist_path.exists():
        logger.info(
            f"Existing vector database directory detected at '{persist_path}'. Clearing contents for clean rebuild..."
        )
        for item in persist_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info("Previous vector database contents removed successfully.")
    else:
        persist_path.mkdir(parents=True, exist_ok=True)

    # 2. Load PDF documents
    logger.info(f"Loading PDF documents from '{contracts_path}'...")
    loader = PyPDFDirectoryLoader(str(contracts_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} document pages from '{contracts_path}'.")

    # 3. Chunk documents
    logger.info(
        f"Splitting documents into chunks (chunk_size={c_size}, chunk_overlap={c_overlap})..."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
    )
    docs_split = text_splitter.split_documents(docs)
    logger.info(f"Created {len(docs_split)} chunks after splitting.")

    # 4. Initialize embedding model
    logger.info(f"Initializing embedding client with model '{emb_model}'...")
    embeddings = OpenAIEmbeddings(
        model=emb_model,
        openai_api_key=api_key,
    )

    # 5. Persist to ChromaDB
    logger.info(f"Indexing chunks and persisting to ChromaDB at '{persist_path}'...")
    vector_store = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    logger.info(f"Vector store successfully created and persisted at '{persist_path}'.")

    # 6. Optional sample similarity query
    if run_sample_query:
        sample_query = "¿Donde se encuentra el local del contrato en el que participa Maria Jimenez Campos?"
        logger.info(f"Running sample similarity search: '{sample_query}'")
        results = vector_store.similarity_search(sample_query, k=settings.SEARCH_K)
        logger.info(f"Sample query results retrieved: {len(results)}")
        for i, doc in enumerate(results, start=1):
            logger.info(
                f"Result {i}:\nMetadata: {doc.metadata}\nContent:\n{doc.page_content}\n{'-' * 60}"
            )

    return vector_store


if __name__ == "__main__":
    try:
        create_vector_database()
    except Exception as exc:
        logger.critical(f"Vector database generation failed: {exc}", exc_info=True)
        sys.exit(1)
