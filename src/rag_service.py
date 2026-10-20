"""
Service module for the Legal Assistant RAG system.
Handles vector retrieval, MultiQuery / Ensemble combining, context formatting,
and LLM response generation for legal lease contracts.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src import settings
from src.prompts import MULTI_QUERY_PROMPT, RAG_TEMPLATE

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service responsible for document retrieval and question answering on legal contracts.
    Decoupled from UI frameworks like Streamlit for modularity and testability.
    """

    def __init__(self) -> None:
        """
        Initialize RAGService loading configuration directly from project settings.
        """
        self.api_key = settings.OPENAI_API_KEY.strip()
        if not self.api_key:
            raise ValueError("OpenAI API key must be configured in settings.")

        self.chroma_db_path = str(settings.CHROMA_DB_PATH)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.query_model = settings.QUERY_MODEL
        self.generation_model = settings.GENERATION_MODEL
        self.search_type = settings.SEARCH_TYPE
        self.search_k = settings.SEARCH_K
        self.fetch_k = settings.MMR_FETCH_K
        self.lambda_mult = settings.MMR_DIVERSITY_LAMBDA
        self.enable_hybrid_search = settings.ENABLE_HYBRID_SEARCH

        self._init_components()

    # --- Public Methods ---

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute full RAG pipeline for a question:
        Retrieves relevant documents once, formats prompt context, generates
        the response, and extracts structured source references.

        Args:
            question: User inquiry string.

        Returns:
            Tuple containing the answer string and list of formatted source dictionaries.
        """
        try:
            # Single retrieval execution avoids redundant LLM query expansions
            docs = self.retrieve(question)

            # Format documents for prompt context
            context = self.format_documents(docs)

            # Generate model response
            response = self.generate(question=question, context=context)

            # Format citation sources for display
            formatted_sources = self.format_sources(docs=docs, max_k=self.search_k)

            return response, formatted_sources

        except Exception as exc:
            logger.error("Error querying RAG system: %s", exc, exc_info=True)
            return f"Error al consultar el sistema RAG: {str(exc)}", []

    def retrieve(self, question: str) -> List[Document]:
        """
        Execute single-pass retrieval for a question.

        Args:
            question: User inquiry or query string.

        Returns:
            List of relevant Document chunks.
        """
        return self.retriever.invoke(question)

    def generate(self, question: str, context: str) -> str:
        """
        Generate answer from the LLM given the user question and contract context.

        Args:
            question: User query string.
            context: Formatted contract fragments.

        Returns:
            Generated response string.
        """
        return self.generation_chain.invoke({
            "question": question,
            "context": context,
        })

    @staticmethod
    def format_documents(docs: List[Document]) -> str:
        """
        Format retrieved document chunks with headers and metadata for the LLM prompt context.

        Args:
            docs: List of Document chunks.

        Returns:
            Concatenated formatted text string.
        """
        formatted_chunks: List[str] = []
        for i, doc in enumerate(docs, start=1):
            header = f"[Fragmento {i}]"

            if doc.metadata:
                if "source" in doc.metadata:
                    raw_source = str(doc.metadata["source"]).replace("\\", "/")

                    source_name = (
                        raw_source.split("/")[-1]
                        if "/" in raw_source
                        else raw_source
                    )

                    header += f" - Fuente: {source_name}"

                if "page" in doc.metadata:
                    header += f" - Pagina: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formatted_chunks.append(f"{header}\n{content}")

        return "\n\n".join(formatted_chunks)

    @staticmethod
    def format_sources(
        docs: List[Document],
        max_k: Optional[int] = None,
        max_content_length: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Extract and format source metadata from retrieved documents for display in the UI.

        Args:
            docs: List of retrieved Document chunks.
            max_k: Maximum number of sources to return. If None, returns all.
            max_content_length: Maximum length of the content snippet.

        Returns:
            List of source information dictionaries.
        """
        selected_docs = docs[:max_k] if max_k is not None else docs
        formatted_sources: List[Dict[str, Any]] = []

        for i, doc in enumerate(selected_docs, start=1):
            content = doc.page_content
            if len(content) > max_content_length:
                content = content[:max_content_length]

            raw_source = (
                doc.metadata.get("source", "Desconocida")
                if doc.metadata
                else "Desconocida"
            )

            normalized_source = str(raw_source).replace("\\", "/")

            source_file = (
                normalized_source.split("/")[-1]
                if "/" in normalized_source
                else normalized_source
            )

            page = doc.metadata.get("page", "N/A") if doc.metadata else "N/A"

            source_info: Dict[str, Any] = {
                "fragmento": i,
                "contenido": content,
                "fuente": source_file,
                "pagina": page,
            }

            formatted_sources.append(source_info)

        return formatted_sources

    def _init_components(self) -> None:
        """Initialize all internal models, vector stores, retrievers, and generation chains."""
        self.embeddings = self._build_embeddings()
        self.vector_store = self._build_vector_store()
        self.llm_queries, self.llm_generation = self._build_llms()
        self.retriever = self._build_retriever()
        self.generation_chain = self._build_generation_chain()

    def _build_embeddings(self) -> OpenAIEmbeddings:
        """Create and configure the OpenAIEmbeddings instance."""
        return OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.api_key,
        )

    def _build_vector_store(self) -> Chroma:
        """Connect to the persistent ChromaDB vector store."""
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.chroma_db_path,
        )

    def _build_llms(self) -> Tuple[ChatOpenAI, ChatOpenAI]:
        """Instantiate query expansion LLM and generation LLM."""
        llm_queries = ChatOpenAI(
            model=self.query_model,
            temperature=0,
            openai_api_key=self.api_key,
        )

        llm_generation = ChatOpenAI(
            model=self.generation_model,
            temperature=0,
            openai_api_key=self.api_key,
        )

        return llm_queries, llm_generation

    def _build_retriever(self) -> Any:
        """
        Build and assemble the retriever.
        Combines MMR retriever, MultiQueryRetriever, and optional EnsembleRetriever.
        """
        retriever_mmr = self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={
                "k": self.search_k,
                "fetch_k": self.fetch_k,
                "lambda_mult": self.lambda_mult,
            },
        )

        multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)
        mmr_multi_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever_mmr,
            llm=self.llm_queries,
            prompt=multi_query_prompt,
        )

        if self.enable_hybrid_search:
            similarity_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.search_k},
            )

            # EnsembleRetriever combines reciprocal rank fusion across retrievers
            final_retriever = EnsembleRetriever(
                retrievers=[mmr_multi_retriever, similarity_retriever],
                weights=[0.7, 0.3],
            )

            return final_retriever

        return mmr_multi_retriever

    def _build_generation_chain(self) -> Any:
        """Build the runnable prompt-to-response generation chain."""
        prompt = PromptTemplate.from_template(RAG_TEMPLATE)
        return prompt | self.llm_generation | StrOutputParser()
