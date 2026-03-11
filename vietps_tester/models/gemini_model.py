"""
models/gemini_model.py — Google Gemini adapter (e.g. gemini-2.0-flash).
"""

from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_model import AdapterConfig, BaseLLMAdapter


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google Gemini GenerateContent API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._model_id = config.model_id or self.DEFAULT_MODEL

    @property
    def _endpoint(self) -> str:
        return (
            f"{self.BASE_URL}/{self._model_id}:generateContent"
            f"?key={self.config.api_key}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        response = requests.post(
            self._endpoint,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
