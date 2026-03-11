"""
tests/integration/test_evaluator_mock.py — Integration tests for the Evaluator.

Uses mock adapters to test the full evaluation pipeline without real API calls.

Tests cover:
  - Full run with perfect, always-yes, always-no, and error adapters
  - EvaluationRun structure and pass/fail counts
  - Metrics are computed correctly after a run
  - Multiple adapters run in a single call
  - Regression check on evaluation results
"""

import pytest

from vietps_tester.evaluator import Evaluator, EvaluationRun
from vietps_tester.metrics import check_regression

SUBMODULE = "Public-Sector-Application"


@pytest.mark.integration
class TestEvaluatorWithPerfectAdapter:
    def test_returns_list_of_runs(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        assert isinstance(runs, list)
        assert len(runs) == 1

    def test_run_is_evaluation_run(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        assert isinstance(runs[0], EvaluationRun)

    def test_perfect_adapter_all_pass(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        run = runs[0]
        assert run.passed_count == run.total_count
        assert run.failed_count == 0

    def test_perfect_adapter_accuracy_1(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        assert runs[0].metrics.overall.accuracy == 1.0

    def test_pass_rate_is_1(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        assert runs[0].pass_rate == 1.0


@pytest.mark.integration
class TestEvaluatorWithAlwaysYes:
    def test_always_yes_detects_all_hallucinated(self, sample_test_cases, always_yes_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [always_yes_adapter])
        run = runs[0]
        # always-yes: passes for is_hallucinated=True, fails for is_hallucinated=False
        truth_positive = sum(1 for tc in sample_test_cases if tc.is_hallucinated)
        assert run.passed_count == truth_positive

    def test_always_yes_recall_is_1(self, sample_test_cases, always_yes_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [always_yes_adapter])
        assert runs[0].metrics.overall.recall == 1.0


@pytest.mark.integration
class TestEvaluatorWithAlwaysNo:
    def test_always_no_misses_all_hallucinated(self, sample_test_cases, always_no_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [always_no_adapter])
        run = runs[0]
        # always-no: passes only for correct (non-hallucinated) answers
        truth_negative = sum(1 for tc in sample_test_cases if not tc.is_hallucinated)
        assert run.passed_count == truth_negative

    def test_always_no_recall_is_0(self, sample_test_cases, always_no_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [always_no_adapter])
        assert runs[0].metrics.overall.recall == 0.0


@pytest.mark.integration
class TestEvaluatorWithErrorAdapter:
    def test_error_adapter_returns_khong(self, sample_test_cases, error_adapter):
        """Error adapter should not raise; evaluator should handle gracefully."""
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [error_adapter])
        run = runs[0]
        assert run.total_count == len(sample_test_cases)
        # All predictions should be Không (safe default on error)
        for result in run.results:
            assert result.predicted_label == "Không"
            assert result.error is not None


@pytest.mark.integration
class TestEvaluatorMultipleAdapters:
    def test_multiple_adapters_returns_multiple_runs(
        self, sample_test_cases, perfect_adapter, always_yes_adapter
    ):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(
            sample_test_cases, [perfect_adapter, always_yes_adapter]
        )
        assert len(runs) == 2

    def test_runs_have_correct_model_names(
        self, sample_test_cases, perfect_adapter, always_yes_adapter
    ):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(
            sample_test_cases, [perfect_adapter, always_yes_adapter]
        )
        names = {r.model_name for r in runs}
        assert "perfect-mock" in names
        assert "always-yes-mock" in names


@pytest.mark.integration
class TestRegressionAfterRun:
    def test_perfect_adapter_passes_regression(self, sample_test_cases, perfect_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [perfect_adapter])
        passed, msg = check_regression(
            "any-model", runs[0].metrics, baseline_override=0.45
        )
        assert passed

    def test_always_no_may_fail_regression(self, sample_test_cases, always_no_adapter):
        evaluator = Evaluator(SUBMODULE, verbose=False)
        runs = evaluator.run(sample_test_cases, [always_no_adapter])
        # With balanced 50/50 dataset, always-no gets ~50% accuracy
        # Passes baseline=0.45 but we can test with a high threshold
        passed, _ = check_regression(
            "always-no", runs[0].metrics, baseline_override=0.99
        )
        assert not passed
