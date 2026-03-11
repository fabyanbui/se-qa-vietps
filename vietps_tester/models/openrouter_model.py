"""
models/openrouter_model.py — OpenRouter adapter for DeepSeek V3, Claude 3.5 Haiku, etc.

OpenRouter provides unified API access to many closed-source models.
"""

from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_model import AdapterConfig, BaseLLMAdapter


class OpenRouterAdapter(BaseLLMAdapter):
    """Adapter for OpenRouter Chat Completions API (OpenAI-compatible)."""

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        if not config.model_id:
            raise ValueError("model_id is required for OpenRouterAdapter.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fabyanbui/se-qa-vietps",
        }
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = requests.post(
            self.ENDPOINT,
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
