"""
tests/conftest.py — Shared fixtures for all test layers.

Provides:
  - sample_test_cases: a small fixed list of HalluTestCase objects
  - mock_adapter: a BaseLLMAdapter that returns predetermined labels
  - submodule_path: path to the Public-Sector-Application submodule
"""

from __future__ import annotations

import pytest

from vietps_tester.dataset_loader import HalluTestCase
from vietps_tester.models.base_model import AdapterConfig, BaseLLMAdapter, LABEL_YES, LABEL_NO


# ── Constants ──────────────────────────────────────────────────────────────────

SUBMODULE_PATH = "Public-Sector-Application"


# ── Sample test cases ──────────────────────────────────────────────────────────

SAMPLE_TEST_CASES = [
    HalluTestCase(
        id="https://dichvucong.gov.vn/?id=1::hallucinated",
        link="https://dichvucong.gov.vn/?id=1",
        question="Thủ tục cấp thẻ nhà báo như thế nào?",
        answer="Thẻ nhà báo được cấp theo Thông tư số 99/2020/TT-BCA.",  # Wrong
        is_hallucinated=True,
        pattern=0,
        ministry="Bộ Thông tin và Truyền thông",
    ),
    HalluTestCase(
        id="https://dichvucong.gov.vn/?id=1::correct",
        link="https://dichvucong.gov.vn/?id=1",
        question="Thủ tục cấp thẻ nhà báo như thế nào?",
        answer="Thẻ nhà báo được cấp theo Thông tư số 49/2016/TT-BTTTT.",
        is_hallucinated=False,
        pattern=-1,
        ministry="Bộ Thông tin và Truyền thông",
    ),
    HalluTestCase(
        id="https://dichvucong.gov.vn/?id=2::hallucinated",
        link="https://dichvucong.gov.vn/?id=2",
        question="Phí cấp phép xây dựng là bao nhiêu?",
        answer="Phí xây dựng là 5 triệu đồng theo Nghị định 10/2010.",  # Fabricated
        is_hallucinated=True,
        pattern=2,
        ministry="Bộ Xây dựng",
    ),
    HalluTestCase(
        id="https://dichvucong.gov.vn/?id=2::correct",
        link="https://dichvucong.gov.vn/?id=2",
        question="Phí cấp phép xây dựng là bao nhiêu?",
        answer="Phí xây dựng được quy định tại Thông tư 172/2016/TT-BTC.",
        is_hallucinated=False,
        pattern=-1,
        ministry="Bộ Xây dựng",
    ),
]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_test_cases() -> list[HalluTestCase]:
    """Return a small deterministic set of test cases."""
    return SAMPLE_TEST_CASES.copy()


@pytest.fixture
def submodule_path() -> str:
    return SUBMODULE_PATH


class _PerfectAdapter(BaseLLMAdapter):
    """Mock adapter that always returns the correct label."""

    def _call_api(self, prompt: str) -> str:
        # Detect which test case this prompt belongs to by checking for known wrong text
        if "99/2020" in prompt or "5 triệu đồng" in prompt:
            return LABEL_YES   # Hallucinated → correctly detected
        return LABEL_NO        # Correct answer → correctly not flagged


class _AlwaysYesAdapter(BaseLLMAdapter):
    """Mock adapter that always predicts 'Có' (hallucination)."""

    def _call_api(self, prompt: str) -> str:
        return LABEL_YES


class _AlwaysNoAdapter(BaseLLMAdapter):
    """Mock adapter that always predicts 'Không' (no hallucination)."""

    def _call_api(self, prompt: str) -> str:
        return LABEL_NO


class _ErrorAdapter(BaseLLMAdapter):
    """Mock adapter that raises an exception on every call."""

    def _call_api(self, prompt: str) -> str:
        raise ConnectionError("Simulated network failure")


@pytest.fixture
def perfect_adapter() -> BaseLLMAdapter:
    return _PerfectAdapter(AdapterConfig(name="perfect-mock"))


@pytest.fixture
def always_yes_adapter() -> BaseLLMAdapter:
    return _AlwaysYesAdapter(AdapterConfig(name="always-yes-mock"))


@pytest.fixture
def always_no_adapter() -> BaseLLMAdapter:
    return _AlwaysNoAdapter(AdapterConfig(name="always-no-mock"))


@pytest.fixture
def error_adapter() -> BaseLLMAdapter:
    return _ErrorAdapter(AdapterConfig(name="error-mock"))
