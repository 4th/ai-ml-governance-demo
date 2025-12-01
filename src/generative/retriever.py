# retriever
# src/generative/retriever.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

KB_PATH = Path("src/generative/knowledge_base.json")


def load_knowledge_base(path: Path | None = None) -> List[Dict]:
    kb_path = path or KB_PATH
    if not kb_path.exists():
        logger.error("Knowledge base file not found at {}", kb_path)
        return []

    with kb_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def simple_keyword_search(question: str, k: int = 3) -> List[Dict]:
    """
    Very simple keyword matching over a small in-repo knowledge base.
    """
    kb = load_knowledge_base()
    question_tokens = question.lower().split()
    scores_docs = []

    for doc in kb:
        content = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
        score = sum(token in content for token in question_tokens)
        if score > 0:
            scores_docs.append((score, doc))

    scores_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scores_docs[:k]]

    logger.info("Retriever returned {} docs for question '{}'", len(top_docs), question)
    return top_docs

