"""Optional Gemini explanation layer. Deterministic rules remain authoritative."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = 0.0
    model: str | None = None
    enabled: bool = False
    note: str = "LLM disabled; deterministic explanations used."

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeminiExplainer:
    """Use Gemini only to explain evidence already validated by code."""

    def __init__(self):
        self.enabled = (
            os.getenv("ENABLE_GEMINI", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv(
            "GEMINI_MODEL", "gemini-3.8-flash"
        ).strip()

        self.input_rate = self._optional_float(
            "GEMINI_INPUT_USD_PER_MILLION_TOKENS"
        )
        self.output_rate = self._optional_float(
            "GEMINI_OUTPUT_USD_PER_MILLION_TOKENS"
        )

        self.usage = LLMUsage(
            model=self.model,
            enabled=self.enabled,
        )

        self.client = None

        if self.enabled:
            if not self.api_key:
                raise ValueError(
                    "Gemini is enabled but GEMINI_API_KEY is missing. "
                    "Set your key in .env."
                )

            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)

                self.usage.note = (
                    "Gemini enabled; explanations use validated evidence."
                )

            except ImportError as exc:
                raise RuntimeError(
                    "Install the Google GenAI SDK using "
                    "pip install -r requirements-ai.txt"
                ) from exc

    @staticmethod
    def _optional_float(name: str) -> float | None:
        value = os.getenv(name, "").strip()

        if not value:
            return None

        try:
            parsed = float(value)
            return parsed if parsed >= 0 else None
        except ValueError:
            return None

    @staticmethod
    def _fallback(note: dict) -> str:
        """Return an evidence-based explanation without an LLM."""

        return (
            f"Note {note['note_id']} dated {note['date'].date()} "
            f"provides supporting context: {note['note']} "
            "The explanation uses the supplied note; "
            "the numerical anomaly decision remains unchanged."
        )

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)

        input_tokens = int(
            getattr(usage, "prompt_token_count", 0) or 0
        )
        output_tokens = int(
            getattr(usage, "candidates_token_count", 0) or 0
        )

        self.usage.calls += 1
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens

        if self.input_rate is None or self.output_rate is None:
            self.usage.estimated_cost_usd = None
            self.usage.note = (
                "Token usage recorded. Configure published model "
                "input and output rates to estimate API cost."
            )
        else:
            self.usage.estimated_cost_usd = round(
                (
                    self.usage.input_tokens * self.input_rate
                    + self.usage.output_tokens * self.output_rate
                ) / 1_000_000,
                8,
            )

    def explain_validated_note(
        self,
        route: str,
        week_of: str,
        note: dict,
    ) -> str:

        if not self.enabled:
            return self._fallback(note)

        prompt = f"""
Write one concise, plain-English explanation for a shipping-cost anomaly.

Use only the validated source note below.
Do not invent causes, facts, dates, or claims.
Do not change the anomaly verdict.
Mention the exact note ID and its date.

Route: {route}
Week starting: {week_of}
Note ID: {note['note_id']}
Note date: {note['date'].date()}
Validated source note: {note['note']}

Return only the explanation text.
"""

        # Retry only transient server/rate-limit errors.
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0},
                )

                self._record_usage(response)

                text = (getattr(response, "text", None) or "").strip()

                if not text:
                    logger.warning(
                        "Gemini returned an empty response for %s",
                        note["note_id"],
                    )
                    return self._fallback(note)

                # Ensure the explanation retains its evidence citation.
                if note["note_id"] not in text:
                    text = (
                        f"Note {note['note_id']} dated "
                        f"{note['date'].date()} supports this finding. "
                        f"{text}"
                    )

                return text

            except Exception as exc:
                # Avoid depending on SDK-specific exception class names.
                status = getattr(exc, "code", None)

                if status is None:
                    status = getattr(exc, "status_code", None)

                message = str(exc).lower()

                transient = (
                    status in (429, 500, 502, 503, 504)
                    or any(
                        marker in message
                        for marker in (
                            "429",
                            "500 internal",
                            "502",
                            "503",
                            "504",
                            "resource_exhausted",
                            "unavailable",
                            "too many requests",
                        )
                    )
                )

                if not transient:
                    logger.warning(
                        "Gemini request failed: %s", exc
                    )
                    self.usage.note = (
                        "Gemini request failed; deterministic "
                        "fallback explanations may have been used."
                    )
                    return self._fallback(note)

                if attempt == 2:
                    logger.warning(
                        "Gemini remained unavailable after retries: %s",
                        exc,
                    )
                    self.usage.note = (
                        "Gemini was temporarily unavailable. "
                        "Deterministic fallback explanations were used."
                    )
                    return self._fallback(note)

                time.sleep(2 ** attempt)

        return self._fallback(note)