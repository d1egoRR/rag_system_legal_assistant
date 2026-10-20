# Legal Assistant RAG

A lightweight, modular Python application designed to analyze and query legal lease contracts using Retrieval-Augmented Generation (RAG). Built with ChromaDB, OpenAI, and Streamlit, with containerized Docker orchestration.

[![Watch Demo Video](https://cdn.loom.com/sessions/thumbnails/03907b49f69045bebffc7d7642ab4a84-with-play.gif)](https://www.loom.com/share/03907b49f69045bebffc7d7642ab4a84)

> 📹 **Demo Walkthrough**: [Watch the demo on Loom](https://www.loom.com/share/03907b49f69045bebffc7d7642ab4a84)

---

## Features

- **Decoupled Architecture**: Clean separation between core RAG logic (`RAGService`), vector database indexing (`generate_vector_db.py`), and the user interface (`LegalAssistantApp`).
- **Hybrid & MultiQuery Retrieval**:
  - **MultiQueryRetriever**: Generates 3 query variations using `QUERY_MODEL` to enhance recall across legal terminology.
  - **MMR (Maximal Margin Relevance)**: Balances relevance with diversity across retrieved chunks.
  - **EnsembleRetriever**: Combines MMR and cosine similarity using reciprocal rank fusion.
- **Single-Pass Pipeline**: Avoids redundant ChromaDB queries and duplicate LLM query expansions by retrieving once and feeding context into the generation chain.
- **Interactive Web Interface**: Streamlit UI featuring a two-column layout (Chat history on the left, retrieved contract fragments and page citations on the right), with resource caching via `@st.cache_resource`.
- **Persistent Vector Store**: ChromaDB local directory persistence synchronized with the host machine.
- **Docker Ready**: Complete Docker Compose setup with real-time volume mounts for development and automated vector store indexing on container start.

---

## Project Structure

```text
rag_system_legal_assistant/
├── app.py                      # Streamlit application (LegalAssistantApp)
├── contract_files/             # Directory containing legal PDF contracts
├── chroma_db/                  # Local persistent ChromaDB vector store
├── scripts/
│   └── generate_vector_db.py   # Script to parse, chunk, and index PDFs into ChromaDB
├── src/
│   ├── __init__.py             # Exports RAGService
│   ├── rag_service.py          # Core RAGService class
│   ├── settings.py             # Typed settings loaded from .env
│   └── prompts.py              # Centralized prompt templates (RAG, MultiQuery)
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_app.py             # Streamlit app unit tests
│   ├── src/
│   │   ├── test_rag_service.py # RAGService unit tests
│   │   ├── test_prompts.py     # Prompt template tests
│   │   └── test_settings.py    # Environment settings tests
│   └── scripts/
│       └── test_generate_vector_db.py # Indexing script tests
├── Dockerfile                  # Production container image definition
├── docker-compose.yml          # Docker Compose orchestration
├── entrypoint.sh               # Container entrypoint (indexes DB then launches app)
├── requirements.txt            # Python dependencies
└── .env.example                # Template for environment variables
```

---

## Prerequisites

- **Python**: Version 3.12 or newer.
- **Docker & Docker Compose**: Recommended for containerized execution.
- **OpenAI API Key**: Required for embeddings and completions.

---

## Configuration

Copy the example environment file and configure your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | Your OpenAI secret API key | Required |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-large` |
| `QUERY_MODEL` | LLM used for multi-query expansions | `gpt-4o-mini` |
| `GENERATION_MODEL` | LLM used for answer generation | `gpt-4o` |
| `CHROMA_DB_PATH` | Directory for vector store persistence | `chroma_db` |
| `CHUNK_SIZE` | Maximum chunk character size | `5000` |
| `CHUNK_OVERLAP` | Overlap characters between chunks | `1000` |
| `SEARCH_TYPE` | Base retriever type | `mmr` |
| `SEARCH_K` | Number of chunks to retrieve for context | `2` |
| `MMR_FETCH_K` | Documents to fetch before MMR filtering | `20` |
| `MMR_DIVERSITY_LAMBDA`| Diversity factor (0 to 1) for MMR | `0.7` |
| `ENABLE_HYBRID_SEARCH`| Enable Ensemble (MMR + Similarity) | `True` |

---

## Running the Application

### Option 1: Using Docker Compose (Recommended)

Docker Compose handles vector store initialization and starts the Streamlit UI automatically:

1. Place your contract PDF documents in the `contract_files/` directory.
2. Start the container:
   ```bash
   docker compose up -d
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```
4. To stop the container:
   ```bash
   docker compose down
   ```

> [!NOTE]
> The container mounts `./src`, `./scripts`, `./tests`, and `app.py` in real-time, allowing immediate hot-reloading for code modifications without rebuilding the image.

---

### Option 2: Running Locally with Python

1. **Create and activate a virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the vector database**:
   ```bash
   python scripts/generate_vector_db.py
   ```

4. **Launch the Streamlit web application**:
   ```bash
   streamlit run app.py
   ```
   The UI will open automatically in your browser at `http://localhost:8501`.

---

## Testing

The project uses `pytest` for unit and integration testing. All external network calls (OpenAI and ChromaDB) are mocked in the test suite to ensure fast, deterministic execution without API costs.

Run the entire test suite:
```bash
pytest -v
```

Run specific test modules:
```bash
pytest tests/src/test_rag_service.py -v
pytest tests/test_app.py -v
```

---

## Code Quality and Design Principles

- **PEP 8 & Type Hints**: Strict type hints across all modules and services.
- **Separation of Concerns**: UI components never contain vector logic or prompt construction.
- **Fail-Safe Exception Handling**: Explicit domain error logging with graceful fallbacks in the UI.
