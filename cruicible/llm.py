"""Shared Gemini client with retry logic for all tools."""

import os
import time
import logging

from google import genai

logger = logging.getLogger(__name__)

_client = None
_client_key = None


def get_client() -> genai.Client:
    global _client, _client_key
    current_key = os.environ.get("GOOGLE_API_KEY", "")
    if _client is None or _client_key != current_key:
        _client = genai.Client(api_key=current_key)
        _client_key = current_key
    return _client


def generate(prompt: str, retries: int = 3) -> str:
    """Call Gemini with retry on 503/429 errors."""
    client = get_client()
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Gemini %s — retry in %ds (attempt %d/%d)", err_str[:60], wait, attempt + 1, retries)
                time.sleep(wait)
            else:
                raise
    return ""
