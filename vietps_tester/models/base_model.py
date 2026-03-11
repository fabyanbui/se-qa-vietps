"""
models/base_model.py — Abstract base class for all LLM adapters.

All adapters implement a single contract:
    predict(prompt: str) -> str  ("Có" or "Không")

This Adapter pattern makes it trivial to add new models and mock them in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


VALID_LABELS = {"Có", "Không"}
LABEL_YES = "Có"        # Model detected hallucination
LABEL_NO = "Không"      # Model found no hallucination


@dataclass
class AdapterConfig:
    """Common configuration shared by all adapters."""

    name: str
    api_key: str = ""
    endpoint: str = ""
    model_id: str = ""
    max_retries: int = 3
    timeout: int = 60
    temperature: float = 0.0
    max_tokens: int = 10


class BaseLLMAdapter(ABC):
    """
    Abstract adapter for any LLM used as a system under test.

    Subclasses must implement `_call_api` to handle the HTTP layer.
    The `predict` method handles normalization, retries, and error recovery.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Call the LLM API and return the raw text response.
        Raises an exception on network or API errors.
        """

    def predict(self, prompt: str) -> str:
        """
        Send a prompt and return a normalized label: 'Có' or 'Không'.

        Normalizes various response formats the LLM might use.
        Raises exceptions from _call_api so the caller can record error details.
        """
        raw = self._call_api(prompt)
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: str) -> str:
        """
        Normalize raw LLM output to 'Có' or 'Không'.

        LLMs may respond with different capitalizations, extra punctuation,
        or surrounding text. This extracts the binary label.
        """
        if not raw:
            return LABEL_NO

        text = raw.strip()

        # Direct match (most common)
        if text in VALID_LABELS:
            return text

        # Case-insensitive check
        upper = text.upper()
        if "CÓ" in upper or "CO" in upper or "YES" in upper or "TRUE" in upper:
            return LABEL_YES
        if "KHÔNG" in upper or "KHONG" in upper or "NO" in upper or "FALSE" in upper:
            return LABEL_NO

        # Default: treat any non-empty as Không (no hallucination detected)
        return LABEL_NO
