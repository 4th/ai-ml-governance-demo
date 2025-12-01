# src/generative/llm_rag_app.py
from __future__ import annotations

from typing import List, Tuple

from src.generative.llm_client import call_llm
from src.generative.retriever import simple_keyword_search
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def answer_with_rag(question: str) -> Tuple[str, List[str]]:
    """
    Use a trivial RAG pattern: retrieve top docs, build context prompt, call LLM.
    Returns an (answer, sources) tuple.
    """
    docs = simple_keyword_search(question)
    if not docs:
        logger.info("No docs retrieved for question '{}', sending directly to LLM.", question)
        return call_llm(question), []

    context = "\n\n".join(d["content"] for d in docs)
    sources = [d.get("title") or f"doc-{d.get('id')}" for d in docs]

    prompt = (
        "You are an AI assistant helping explain AI governance and ML lifecycle topics.\n"
        "Use ONLY the provided context to answer the user's question. If the answer "
        "cannot be derived from context, say you are not sure.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    answer = call_llm(prompt)
    logger.info("RAG answer generated for question '{}'", question)

    return answer, sources
