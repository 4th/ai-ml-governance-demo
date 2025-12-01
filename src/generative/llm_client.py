# llm client
# src/generative/llm_client.py
from __future__ import annotations

from typing import Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. LLM calls will use fallback behavior.")
        return None

    if OPENAI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    return client


def call_llm(prompt: str) -> str:
    """
    Call the configured LLM, with a safe fallback if configuration is missing.
    """
    client = get_client()
    if client is None:
        logger.info("Using fallback fake LLM response.")
        return f"[FAKE-LLM] {prompt}"

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error calling LLM: {}", e)
        return f"[LLM-ERROR-FALLBACK] {prompt}"
