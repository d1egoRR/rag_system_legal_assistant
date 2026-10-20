"""
Módulo para vectorizar contratos legales en PDF y almacenarlos
en una base de datos vectorial Chroma (SQLite).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración básica de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_vector_db")


def create_vector_database(
    contracts_dir: str = "contratos",
    persist_directory: str = "chroma_db",
    chunk_size: int = 5000,
    chunk_overlap: int = 1000,
    embedding_model: str = "text-embedding-3-large",
    openai_api_key: Optional[str] = None,
    run_sample_query: bool = True,
) -> Chroma:
    """
    Carga contratos en formato PDF desde un directorio, divide su contenido
    en fragmentos y genera una base de datos vectorial persistente con ChromaDB.

    Args:
        contracts_dir: Directorio donde residen los archivos PDF de contratos.
        persist_directory: Carpeta de destino donde se persistirá la base SQLite de Chroma.
        chunk_size: Tamaño máximo en caracteres de cada fragmento.
        chunk_overlap: Solapamiento entre fragmentos consecutivos.
        embedding_model: Modelo de embeddings de OpenAI a utilizar.
        openai_api_key: Clave API opcional (si no se especifica, se toma de OPENAI_API_KEY).
        run_sample_query: Si es True, realiza una consulta de prueba tras la indexación.

    Returns:
        Chroma: Instancia del vector store configurado y persistido.
    """
    load_dotenv()

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        error_msg = (
            "No se encontró la clave OPENAI_API_KEY. "
            "Configúrala en el archivo .env o pásala como parámetro."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    contracts_path = Path(contracts_dir)

    if not contracts_path.exists() or not contracts_path.is_dir():
        error_msg = f"El directorio de contratos '{contracts_dir}' no existe o no es un directorio válido."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Iniciando carga de documentos PDF desde '%s'...", contracts_dir)

    loader = PyPDFDirectoryLoader(str(contracts_path))
    docs = loader.load()

    logger.info("Se cargaron %d páginas de documentos desde '%s'.", len(docs), contracts_dir)

    logger.info(
        "Dividiendo documentos en fragmentos (chunk_size=%d, chunk_overlap=%d)...",
        chunk_size,
        chunk_overlap,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs_split = text_splitter.split_documents(docs)
    logger.info("Se crearon %d fragmentos tras la división.", len(docs_split))

    logger.info("Inicializando cliente de embeddings con modelo '%s'...", embedding_model)
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=api_key,
    )

    logger.info("Indexando fragmentos y persistiendo en ChromaDB en '%s'...", persist_directory)
    vector_store = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    logger.info("Vector store creado y persistido exitosamente en '%s'.", persist_directory)

    if run_sample_query:
        consulta = "¿Donde se encuentra el local del contrato en el que participa Maria Jimenez Campos?"
        logger.info("Ejecutando búsqueda semántica de prueba: '%s'", consulta)
        resultados = vector_store.similarity_search(consulta, k=2)
        logger.info("Resultados de prueba obtenidos: %d", len(resultados))
        
        for i, doc in enumerate(resultados, start=1):
            logger.info(
                "Resultado %d:\nMetadatos: %s\nContenido:\n%s\n%s",
                i,
                doc.metadata,
                doc.page_content,
                "-" * 60,
            )

    return vector_store


if __name__ == "__main__":
    try:
        create_vector_database()
    except Exception as exc:
        logger.critical("Fallo en la ejecución de create_vector_database: %s", exc, exc_info=True)
        sys.exit(1)
