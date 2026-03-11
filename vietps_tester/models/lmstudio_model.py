"""
models/lmstudio_model.py — Local LM Studio adapter (OpenAI-compatible local server).

LM Studio exposes a local OpenAI-compatible API at http://localhost:1234/v1.
"""

from __future__ import annotations

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_model import AdapterConfig, BaseLLMAdapter


class LMStudioAdapter(BaseLLMAdapter):
    """Adapter for locally-running LM Studio models."""

    DEFAULT_ENDPOINT = "http://localhost:1234/v1/chat/completions"

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        endpoint_base = config.endpoint.rstrip("/") if config.endpoint else "http://localhost:1234/v1"
        self._endpoint = f"{endpoint_base}/chat/completions"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.model_id:
            payload["model"] = self.config.model_id

        response = requests.post(
            self._endpoint,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
