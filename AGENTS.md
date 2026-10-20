# AGENTS.md

## Project Overview

Legal Assistant RAG is a lightweight MVP software project developed in Python to analyze legal contracts using Retrieval-Augmented Generation (RAG).

The project prioritizes simplicity, minimal abstractions, testability, and fast iteration. It avoids complex orchestration frameworks (such as LangChain or LlamaIndex) in favor of straightforward, explicit Python modules and functions.

The application requirements and features will evolve over time. Always follow the latest approved specifications and explicit user instructions.

## Core Principles

- Follow Python best practices and PEP 8.
- Keep the design simple, lean, and minimal (MVP focus).
- Avoid unnecessary frameworks, abstractions, and premature optimizations.
- Favor plain Python modules and explicit functions over complex class hierarchies.
- Isolate external services (LLM APIs, ChromaDB, document parsers) cleanly.
- Use dependency injection or direct parameter passing to keep components testable.
- Minimize external dependencies.
- Design for unit and integration testability.
- Implement only what is explicitly requested.

## Language

- Use English exclusively for source code, identifiers, comments, technical documentation, and project structure.
- Use descriptive and consistent naming conventions.
- Follow Python naming conventions.
- Keep technical documentation clear and concise.

## Technology Stack

- Language: Python (3.12+).
- Vector Database: ChromaDB.
- User Interface: Streamlit.
- Containerization: Docker.
- Local Orchestration: Docker Compose.
- Testing: pytest.

Prefer standard library functionality whenever it is sufficient.

Avoid adding LLM frameworks (e.g., LangChain, LlamaIndex, Haystack) unless explicitly instructed. Keep logic within plain Python scripts.

## Architecture & Code Organization

Organize the application cleanly around functional responsibilities without heavy boilerplate:

### Core RAG Pipeline

- Document Loading & Parsing: Modules dedicated to extracting text and metadata from contracts (PDF, DOCX, TXT).
- Chunking & Preprocessing: Text splitting and chunk enrichment logic.
- Embedding & Vector Store: Pure Python integration with ChromaDB for indexing and querying chunks.
- Generation / LLM Service: Explicit API client handling prompt construction and completions.

### User Interface

- Built with Streamlit.
- Keeps UI logic separated from RAG pipeline execution.
- Interacts with RAG services by importing application modules directly.

### Separation Rules

- Core parsing, indexing, and retrieval logic must remain independent of the Streamlit UI.
- ChromaDB client management and collections should be isolated in dedicated helper modules.
- Configuration and API credentials must not leak into core logic.

Do not create empty packages, unnecessary layers, or speculative abstractions.

## Project Structure

- Keep the project structure flat, practical, and aligned with MVP goals.
- Group related functions and modules logically (e.g., parsing, embeddings, storage, retrieval, UI).
- Separate the Streamlit entry point from core business logic.
- Keep test files in a dedicated `tests/` directory matching the source structure.
- Avoid moving or renaming modules without a clear benefit.

## Python Coding Standards

- Follow PEP 8.
- Use type hints consistently across all functions.
- Prefer explicit and readable code over clever one-liners.
- Use Python 3.12+ features where appropriate.
- Handle exceptions explicitly (e.g., parsing errors, API rate limits, ChromaDB connection errors).
- Avoid broad exception handling (`except Exception:`) unless strictly justified.
- Avoid unnecessary global mutable state (especially within Streamlit reruns).
- Use dataclasses or typed dictionaries for structured data when beneficial.
- Document public functions and modules with concise docstrings.

## RAG & Vector Database Guidelines

When managing RAG operations and ChromaDB:

- Configure ChromaDB client explicitly (in-memory or persistent directory as required).
- Always include relevant metadata (e.g., document name, page number, clause section) in vector embeddings.
- Ensure text chunking strategies preserve legal context and contract clause boundaries where feasible.
- Handle API rate limits, timeouts, and retrieval errors gracefully.
- Support deterministic query-retrieval patterns for easier testing.

## UI Development (Streamlit)

When developing the Streamlit UI:

- Maintain a clean, intuitive interface for document upload, indexing status, and contract querying.
- Cache expensive resources (like ChromaDB clients or embedding models) using `@st.cache_resource`.
- Cache deterministic data transformations using `@st.cache_data` where appropriate.
- Never place core business rules or vector logic directly inside UI view scripts; delegate to backend modules.
- Present retrieved contract sources and citations clearly to the user.

## Docker and Deployment

- Use Docker for application containerization.
- Support running both the Streamlit UI and local ChromaDB persistence seamlessly.
- Use minimal production images with non-root users where possible.
- Avoid including secrets, API keys, or embedded databases in container images.
- Use environment variables for runtime configuration (e.g., LLM keys, DB paths).
- Use Docker Compose for local development and orchestration.
- Ensure smooth container shutdown and state cleanup.

## Configuration and Security

- Keep configuration separate from code using environment variables (e.g., `.env` files with python-dotenv).
- Never hardcode LLM provider keys, credentials, or sensitive legal data.
- Avoid logging confidential legal document contents or sensitive customer data.
- Validate and sanitize file uploads before processing.

## Testing

- Use pytest as the default testing framework.
- Write unit tests for chunking, text processing, and prompt-building logic.
- Test ChromaDB operations using isolated test collections or temporary directories.
- Mock or fake LLM provider API responses in tests to avoid external network calls and costs.
- Cover edge cases (e.g., empty contracts, malformed files, zero search results).
- Keep tests fast, independent, and deterministic.
- Ensure the complete test suite passes before considering any task complete.

## Code Quality

- Format code consistently.
- Prefer Ruff for fast linting and formatting.
- Avoid unnecessary code duplication.
- Remove dead code and unused imports.

## Documentation

- Maintain a README.md with setup, Docker, and execution instructions.
- Document how to set up environment variables and LLM keys.
- Document how contract documents are processed and indexed.
- Update documentation whenever architectural or execution details change.

## Requirements and Specifications

- Follow the latest approved specifications and explicit user instructions.
- Treat approved specifications as the source of truth.
- Do not implement features that are not requested.
- Prioritize MVP simplicity: implement working, clean solutions before refining.

## Development Workflow

Before implementing a task:

1. Inspect existing modules, dependencies, and tests.
2. Read the relevant specifications and instructions.
3. Identify affected components (e.g., parser, ChromaDB adapter, UI).
4. Propose a straightforward approach for substantial changes.
5. Implement only the required changes using plain Python.
6. Add or update unit/integration tests.
7. Run linting, formatting, and pytest suites.
8. Verify Docker configuration when changes impact deployment.
9. Update documentation when necessary.

## Agent Behavior
- **Do not execute code, run commands, or perform actions unless explicitly instructed by the user.**
- Do not introduce unrequested features or external frameworks (e.g., LangChain).
- Do not modify unrelated files.
- Do not add dependencies without justification.
- Do not make unnecessary architectural changes or introduce premature abstractions.
- Do not silently ignore errors, empty search results, or failing tests.
- Explain significant technical trade-offs when relevant.
- Ask for clarification when essential requirements are ambiguous.
- Prefer incremental, verifiable changes.

## Priority

When instructions conflict, apply the following order:

1. Explicit user instructions in the current task.
2. Latest approved project specifications.
3. Existing architectural decisions and project documentation.
4. This file's general development guidelines.
5. General coding conventions and best practices.

Always identify important conflicts instead of silently making assumptions.