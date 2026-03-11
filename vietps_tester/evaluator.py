"""
evaluator.py — Core test runner: evaluates LLMs against VietPS-Hallu test cases.

This is the heart of the QA automation pipeline:
  1. Accepts a list of HalluTestCase objects (the test suite)
  2. Accepts a list of BaseLLMAdapter instances (the systems under test)
  3. For each (model, test_case) pair: builds a prompt, calls the model, records the result
  4. Returns EvaluationRun objects containing full results per model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from .dataset_loader import HalluTestCase
from .metrics import ModelMetrics, compute_model_metrics
from .models.base_model import BaseLLMAdapter, LABEL_YES
from .prompt_builder import EvaluationMode, ModelType, PromptBuilder


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """The result of evaluating a single test case with a single model."""

    test_case: HalluTestCase
    model_name: str
    predicted_label: str        # "Có" or "Không"
    predicted_hallucinated: bool
    ground_truth: bool
    passed: bool                # predicted_hallucinated == ground_truth
    prompt_text: str
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class EvaluationRun:
    """Full results for one model over a batch of test cases."""

    model_name: str
    mode: EvaluationMode
    results: list[TestResult] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / self.total_count


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs a test suite (list of HalluTestCase) against one or more LLM adapters.

    Usage::

        from vietps_tester.evaluator import Evaluator
        from vietps_tester.models import OpenAIAdapter, AdapterConfig

        adapter = OpenAIAdapter(AdapterConfig(name="gpt-4o-mini", api_key="sk-..."))
        evaluator = Evaluator(submodule_path="Public-Sector-Application")

        runs = evaluator.run(
            test_cases=loader.load_primary(sample_size=50),
            adapters=[adapter],
            mode="without_knowledge",
        )
        for run in runs:
            print(f"{run.model_name}: {run.pass_rate:.2%}")
    """

    def __init__(
        self,
        submodule_path: str = "Public-Sector-Application",
        model_type: ModelType = "close_source",
        verbose: bool = True,
    ) -> None:
        self._prompt_builder = PromptBuilder(submodule_path)
        self._model_type = model_type
        self._verbose = verbose

    def run(
        self,
        test_cases: list[HalluTestCase],
        adapters: list[BaseLLMAdapter],
        mode: EvaluationMode = "without_knowledge",
        knowledge_map: Optional[dict[str, str]] = None,
    ) -> list[EvaluationRun]:
        """
        Evaluate all adapters on all test cases.

        Args:
            test_cases: List of HalluTestCase objects to use as the test suite.
            adapters: List of LLM adapters (systems under test).
            mode: "without_knowledge" or "with_knowledge".
            knowledge_map: Maps test_case.link → TTHC knowledge text.
                           Required when mode="with_knowledge".

        Returns:
            One EvaluationRun per adapter, with full results and computed metrics.
        """
        runs: list[EvaluationRun] = []

        for adapter in adapters:
            run = self._run_single(
                test_cases=test_cases,
                adapter=adapter,
                mode=mode,
                knowledge_map=knowledge_map or {},
            )
            runs.append(run)

        return runs

    def _run_single(
        self,
        test_cases: list[HalluTestCase],
        adapter: BaseLLMAdapter,
        mode: EvaluationMode,
        knowledge_map: dict[str, str],
    ) -> EvaluationRun:
        """Run a single adapter against all test cases."""
        run = EvaluationRun(model_name=adapter.name, mode=mode)

        iterator = tqdm(
            test_cases,
            desc=f"Evaluating {adapter.name}",
            disable=not self._verbose,
        )

        for tc in iterator:
            knowledge = knowledge_map.get(tc.link, "") if mode == "with_knowledge" else ""
            prompt = self._prompt_builder.build(
                question=tc.question,
                answer=tc.answer,
                mode=mode,
                model_type=self._model_type,
                pattern=tc.pattern,
                knowledge=knowledge,
            )

            t0 = time.monotonic()
            try:
                predicted_label = adapter.predict(prompt.text)
                error = None
            except Exception as exc:
                predicted_label = "Không"
                error = str(exc)

            latency_ms = (time.monotonic() - t0) * 1000
            predicted_hallucinated = predicted_label == LABEL_YES

            result = TestResult(
                test_case=tc,
                model_name=adapter.name,
                predicted_label=predicted_label,
                predicted_hallucinated=predicted_hallucinated,
                ground_truth=tc.is_hallucinated,
                passed=(predicted_hallucinated == tc.is_hallucinated),
                prompt_text=prompt.text,
                latency_ms=round(latency_ms, 2),
                error=error,
            )
            run.results.append(result)

        # Compute metrics
        predictions = [r.predicted_hallucinated for r in run.results]
        run.metrics = compute_model_metrics(
            model_name=adapter.name,
            test_cases=test_cases,
            predictions=predictions,
        )

        return run
