"""
models/openai_model.py — OpenAI GPT adapter (e.g. GPT-4o-mini).
"""

from __future__ import annotations

import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_model import AdapterConfig, BaseLLMAdapter


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI Chat Completions API."""

    DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        self._endpoint = config.endpoint or self.DEFAULT_ENDPOINT
        self._model_id = config.model_id or self.DEFAULT_MODEL

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
