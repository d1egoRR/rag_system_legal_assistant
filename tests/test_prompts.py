"""
Unit tests for prompts module templates.
"""

import pytest
from prompts import (
    RAG_TEMPLATE,
    MULTI_QUERY_PROMPT,
    RELEVANCE_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
)


@pytest.mark.unit
def test_rag_template_formatting():
    """Verify RAG_TEMPLATE formats correctly with context and question."""
    context_text = "Cláusula Primera: Inmueble en Calle Tetuán 15."
    question_text = "¿Cuál es la dirección del inmueble?"
    formatted = RAG_TEMPLATE.format(
        context=context_text,
        question=question_text,
    )
    assert context_text in formatted
    assert question_text in formatted
    assert "RESPUESTA:" in formatted


@pytest.mark.unit
def test_multi_query_prompt_formatting():
    """Verify MULTI_QUERY_PROMPT formats correctly with user question."""
    question = "¿Quién es la arrendataria?"
    formatted = MULTI_QUERY_PROMPT.format(question=question)
    assert f"Consulta original: {question}" in formatted


@pytest.mark.unit
def test_relevance_prompt_formatting():
    """Verify RELEVANCE_PROMPT formats correctly with document and question."""
    doc = "Renta mensual de 1.800 euros."
    query = "¿Cuál es el importe de la renta?"
    formatted = RELEVANCE_PROMPT.format(document=doc, question=query)
    assert doc in formatted
    assert query in formatted


@pytest.mark.unit
def test_entity_extraction_prompt_formatting():
    """Verify ENTITY_EXTRACTION_PROMPT formats correctly with text."""
    sample = "María Jiménez Campos arrendó el local a Tienda Verde, S.L."
    formatted = ENTITY_EXTRACTION_PROMPT.format(text=sample)
    assert sample in formatted
    assert "PERSONAS:" in formatted
