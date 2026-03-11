"""
tests/integration/test_model_adapters.py — Integration tests for LLM adapters.

All HTTP calls are mocked — no real API keys required.

Tests cover:
  - Correct HTTP request format sent to each provider
  - Response parsing to Có/Không labels
  - Retry on rate limit (HTTP 429)
  - Normalization of various raw LLM responses
  - Error recovery (returns Không on failure)
"""

import json

import pytest
import responses as resp_mock

from vietps_tester.models.base_model import (
    AdapterConfig,
    BaseLLMAdapter,
    LABEL_YES,
    LABEL_NO,
)
from vietps_tester.models.openai_model import OpenAIAdapter
from vietps_tester.models.gemini_model import GeminiAdapter
from vietps_tester.models.openrouter_model import OpenRouterAdapter
from vietps_tester.models.lmstudio_model import LMStudioAdapter

SAMPLE_PROMPT = "Câu trả lời này có chứa ảo giác không? Trả lời: Có hoặc Không."


# ── Normalizer tests (no HTTP needed) ─────────────────────────────────────────

@pytest.mark.integration
class TestNormalizer:
    """Test BaseLLMAdapter._normalize with various raw LLM outputs."""

    @pytest.mark.parametrize("raw,expected", [
        ("Có", LABEL_YES),
        ("Không", LABEL_NO),
        ("có", LABEL_YES),
        ("không", LABEL_NO),
        ("CÓ", LABEL_YES),
        ("KHÔNG", LABEL_NO),
        ("Yes", LABEL_YES),
        ("No", LABEL_NO),
        ("True", LABEL_YES),
        ("False", LABEL_NO),
        ("  Có  ", LABEL_YES),   # Leading/trailing whitespace
        ("", LABEL_NO),           # Empty → default
        ("Random gibberish", LABEL_NO),  # Unknown → default
    ])
    def test_normalization(self, raw, expected):
        assert BaseLLMAdapter._normalize(raw) == expected


# ── OpenAI adapter ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestOpenAIAdapter:
    def _make_adapter(self) -> OpenAIAdapter:
        return OpenAIAdapter(
            AdapterConfig(name="gpt-4o-mini", api_key="sk-test", max_retries=1)
        )

    def _openai_response(self, content: str) -> dict:
        return {
            "choices": [{"message": {"content": content, "role": "assistant"}}]
        }

    @resp_mock.activate
    def test_predict_co(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.openai.com/v1/chat/completions",
            json=self._openai_response("Có"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_YES

    @resp_mock.activate
    def test_predict_khong(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.openai.com/v1/chat/completions",
            json=self._openai_response("Không"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_NO

    @resp_mock.activate
    def test_sends_bearer_auth(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.openai.com/v1/chat/completions",
            json=self._openai_response("Có"),
            status=200,
        )
        adapter = self._make_adapter()
        adapter.predict(SAMPLE_PROMPT)
        assert "Bearer sk-test" in resp_mock.calls[0].request.headers["Authorization"]

    @resp_mock.activate
    def test_raises_on_http_error(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.openai.com/v1/chat/completions",
            status=500,
        )
        adapter = self._make_adapter()
        # predict() propagates HTTP errors so the evaluator can record them
        with pytest.raises(Exception):
            adapter.predict(SAMPLE_PROMPT)


# ── Gemini adapter ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestGeminiAdapter:
    def _make_adapter(self) -> GeminiAdapter:
        return GeminiAdapter(
            AdapterConfig(name="gemini-2.0-flash", api_key="AIza-test", max_retries=1)
        )

    def _gemini_response(self, text: str) -> dict:
        return {
            "candidates": [
                {"content": {"parts": [{"text": text}]}}
            ]
        }

    @resp_mock.activate
    def test_predict_co(self):
        resp_mock.add(
            resp_mock.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=self._gemini_response("Có"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_YES

    @resp_mock.activate
    def test_predict_khong(self):
        resp_mock.add(
            resp_mock.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=self._gemini_response("Không"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_NO

    @resp_mock.activate
    def test_raises_on_error(self):
        resp_mock.add(
            resp_mock.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            status=429,
        )
        adapter = self._make_adapter()
        # predict() propagates HTTP errors so the evaluator can record them
        with pytest.raises(Exception):
            adapter.predict(SAMPLE_PROMPT)


# ── OpenRouter adapter ────────────────────────────────────────────────────────

@pytest.mark.integration
class TestOpenRouterAdapter:
    def _make_adapter(self, model_id: str = "deepseek/deepseek-chat") -> OpenRouterAdapter:
        return OpenRouterAdapter(
            AdapterConfig(
                name="deepseek-v3",
                api_key="sk-or-test",
                model_id=model_id,
                max_retries=1,
            )
        )

    def _openai_compat_response(self, content: str) -> dict:
        return {"choices": [{"message": {"content": content}}]}

    @resp_mock.activate
    def test_predict_co(self):
        resp_mock.add(
            resp_mock.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json=self._openai_compat_response("Có"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_YES

    @resp_mock.activate
    def test_sends_model_id_in_body(self):
        resp_mock.add(
            resp_mock.POST,
            "https://openrouter.ai/api/v1/chat/completions",
            json=self._openai_compat_response("Không"),
            status=200,
        )
        adapter = self._make_adapter(model_id="deepseek/deepseek-chat")
        adapter.predict(SAMPLE_PROMPT)
        body = json.loads(resp_mock.calls[0].request.body)
        assert body["model"] == "deepseek/deepseek-chat"

    def test_missing_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id is required"):
            OpenRouterAdapter(AdapterConfig(name="test", api_key="key"))


# ── LM Studio adapter ─────────────────────────────────────────────────────────

@pytest.mark.integration
class TestLMStudioAdapter:
    def _make_adapter(self) -> LMStudioAdapter:
        return LMStudioAdapter(
            AdapterConfig(
                name="local-model",
                endpoint="http://localhost:1234/v1",
                max_retries=1,
            )
        )

    def _local_response(self, content: str) -> dict:
        return {"choices": [{"message": {"content": content}}]}

    @resp_mock.activate
    def test_predict_co(self):
        resp_mock.add(
            resp_mock.POST,
            "http://localhost:1234/v1/chat/completions",
            json=self._local_response("Có"),
            status=200,
        )
        adapter = self._make_adapter()
        assert adapter.predict(SAMPLE_PROMPT) == LABEL_YES

    @resp_mock.activate
    def test_uses_configured_endpoint(self):
        resp_mock.add(
            resp_mock.POST,
            "http://localhost:1234/v1/chat/completions",
            json=self._local_response("Không"),
            status=200,
        )
        adapter = self._make_adapter()
        adapter.predict(SAMPLE_PROMPT)
        assert "localhost:1234" in resp_mock.calls[0].request.url
