"""OpenInference tracing → Phoenix Cloud.

Call `setup_tracing()` once at startup before creating the agent.
All Gemini calls via google-genai will be auto-instrumented.
"""

import os
import logging

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing() -> None:
    """Configure tracing to Phoenix Cloud using phoenix.otel.register."""
    global _initialized
    if _initialized:
        return

    phoenix_api_key = os.environ.get("PHOENIX_API_KEY", "")

    if not phoenix_api_key:
        logger.warning(
            "PHOENIX_API_KEY not set — traces will NOT be exported. "
            "Set it in .env to enable Phoenix observability."
        )
        _initialized = True
        return

    from phoenix.otel import register

    register(
        project_name="cruicible",
        auto_instrument=True,
    )

    _initialized = True
    logger.info("Tracing configured → Phoenix Cloud")
