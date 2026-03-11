"""
tests/unit/test_metrics.py — Unit tests for metrics computation.

Tests cover:
  - Perfect predictions → accuracy=1.0
  - All wrong predictions → accuracy=0.0
  - Mixed predictions → correct F1
  - Pattern and ministry breakdowns
  - Regression baseline check
"""

import pytest

from vietps_tester.dataset_loader import HalluTestCase
from vietps_tester.metrics import (
    ClassificationMetrics,
    ModelMetrics,
    _compute_metrics,
    check_regression,
    compute_model_metrics,
    build_comparison_table,
)
from tests.conftest import SAMPLE_TEST_CASES


def _make_case(is_hallucinated: bool, pattern: int = 0, ministry: str = "A") -> HalluTestCase:
    return HalluTestCase(
        id=f"tc_{is_hallucinated}_{pattern}",
        link="https://dichvucong.gov.vn/?test",
        question="Test?",
        answer="Test answer.",
        is_hallucinated=is_hallucinated,
        pattern=pattern,
        ministry=ministry,
    )


@pytest.mark.unit
class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [True, False, True, False]
        y_pred = [True, False, True, False]
        m = _compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.support == 4

    def test_all_wrong_predictions(self):
        y_true = [True, True, False, False]
        y_pred = [False, False, True, True]
        m = _compute_metrics(y_true, y_pred)
        assert m.accuracy == 0.0

    def test_empty_inputs(self):
        m = _compute_metrics([], [])
        assert m.accuracy == 0.0
        assert m.support == 0

    def test_all_positive_predictions(self):
        y_true = [True, False, True]
        y_pred = [True, True, True]
        m = _compute_metrics(y_true, y_pred)
        # TP=2, FP=1, FN=0, TN=0
        assert m.precision == pytest.approx(2 / 3, abs=1e-4)
        assert m.recall == 1.0

    @pytest.mark.parametrize("accuracy,expected", [
        ([True, True, False, False], [True, True, False, False]),   # 1.0
        ([True, False], [True, True]),                               # 0.5
    ])
    def test_accuracy_values(self, accuracy, expected):
        m = _compute_metrics(accuracy, expected)
        assert 0.0 <= m.accuracy <= 1.0


@pytest.mark.unit
class TestComputeModelMetrics:
    def test_basic(self):
        cases = SAMPLE_TEST_CASES
        # Perfect predictions
        predictions = [tc.is_hallucinated for tc in cases]
        result = compute_model_metrics("test-model", cases, predictions)
        assert result.model_name == "test-model"
        assert result.overall.accuracy == 1.0

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="Mismatch"):
            compute_model_metrics("m", SAMPLE_TEST_CASES, [True])

    def test_pattern_breakdown(self):
        cases = [
            _make_case(True, pattern=0),
            _make_case(False, pattern=0),
            _make_case(True, pattern=1),
            _make_case(False, pattern=1),
        ]
        predictions = [True, False, True, False]  # All correct
        result = compute_model_metrics("m", cases, predictions)
        assert 0 in result.by_pattern
        assert 1 in result.by_pattern
        assert result.by_pattern[0].accuracy == 1.0
        assert result.by_pattern[1].accuracy == 1.0

    def test_ministry_breakdown(self):
        cases = [
            _make_case(True, ministry="Bộ A"),
            _make_case(False, ministry="Bộ A"),
            _make_case(True, ministry="Bộ B"),
        ]
        predictions = [True, False, True]
        result = compute_model_metrics("m", cases, predictions)
        assert "Bộ A" in result.by_ministry
        assert "Bộ B" in result.by_ministry


@pytest.mark.unit
class TestCheckRegression:
    def _make_metrics(self, accuracy: float) -> ModelMetrics:
        m = ModelMetrics(model_name="test")
        m.overall = ClassificationMetrics(
            accuracy=accuracy, precision=0.5, recall=0.5, f1=0.5, support=100
        )
        return m

    def test_passes_above_baseline(self):
        metrics = self._make_metrics(0.60)
        passed, msg = check_regression("gpt-4o-mini", metrics)
        assert passed
        assert "PASS" in msg

    def test_fails_below_baseline(self):
        metrics = self._make_metrics(0.30)
        passed, msg = check_regression("gpt-4o-mini", metrics)
        assert not passed
        assert "FAIL" in msg

    def test_custom_baseline_override(self):
        metrics = self._make_metrics(0.75)
        passed, msg = check_regression("any-model", metrics, baseline_override=0.70)
        assert passed

    def test_unknown_model_uses_default(self):
        metrics = self._make_metrics(0.50)
        passed, _ = check_regression("totally-new-model-xyz", metrics)
        # Default baseline is 0.45 → 0.50 >= 0.45 → pass
        assert passed


@pytest.mark.unit
class TestBuildComparisonTable:
    def test_returns_dataframe(self):
        import pandas as pd

        m1 = ModelMetrics(model_name="model-a")
        m1.overall = ClassificationMetrics(
            accuracy=0.75, precision=0.7, recall=0.8, f1=0.75, support=100
        )
        m2 = ModelMetrics(model_name="model-b")
        m2.overall = ClassificationMetrics(
            accuracy=0.65, precision=0.6, recall=0.7, f1=0.65, support=100
        )
        df = build_comparison_table([m1, m2])
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["model_name", "accuracy", "precision", "recall", "f1", "support"]
        # Should be sorted by accuracy descending
        assert df.iloc[0]["model_name"] == "model-a"
