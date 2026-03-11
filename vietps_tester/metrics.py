"""
metrics.py — Compute evaluation metrics for LLM hallucination detection.

Computes:
  - accuracy, precision, recall, F1 (per model overall)
  - Breakdowns by: pattern (0-3), ministry, evaluation mode
  - Regression comparison against historical baselines
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    """Binary classification metrics for hallucination detection."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0  # Total number of samples


@dataclass
class ModelMetrics:
    """Full metrics for a single model evaluation run."""

    model_name: str
    overall: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    by_pattern: dict[int, ClassificationMetrics] = field(default_factory=dict)
    by_ministry: dict[str, ClassificationMetrics] = field(default_factory=dict)


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: list[bool], y_pred: list[bool]
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall, F1 from binary lists."""
    if not y_true:
        return ClassificationMetrics()

    support = len(y_true)
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))

    accuracy = (tp + tn) / support if support else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        support=support,
    )


def compute_model_metrics(
    model_name: str,
    test_cases: list,           # list[TestCase]
    predictions: list[bool],    # True = hallucination detected
) -> ModelMetrics:
    """
    Compute full metrics for a single model over a batch of test cases.

    Args:
        model_name: Identifier string for the model.
        test_cases: List of TestCase objects (ground truth via .is_hallucinated).
        predictions: List of booleans (True = model predicted hallucinated).

    Returns:
        ModelMetrics with overall, by-pattern, and by-ministry breakdowns.
    """
    if len(test_cases) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(test_cases)} test cases vs {len(predictions)} predictions."
        )

    y_true = [tc.is_hallucinated for tc in test_cases]
    overall = _compute_metrics(y_true, predictions)

    # Group by pattern
    pattern_groups: dict[int, tuple[list[bool], list[bool]]] = defaultdict(
        lambda: ([], [])
    )
    for tc, pred in zip(test_cases, predictions):
        pattern_groups[tc.pattern][0].append(tc.is_hallucinated)
        pattern_groups[tc.pattern][1].append(pred)

    by_pattern = {
        pat: _compute_metrics(truths, preds)
        for pat, (truths, preds) in pattern_groups.items()
    }

    # Group by ministry
    ministry_groups: dict[str, tuple[list[bool], list[bool]]] = defaultdict(
        lambda: ([], [])
    )
    for tc, pred in zip(test_cases, predictions):
        if tc.ministry:
            ministry_groups[tc.ministry][0].append(tc.is_hallucinated)
            ministry_groups[tc.ministry][1].append(pred)

    by_ministry = {
        min_name: _compute_metrics(truths, preds)
        for min_name, (truths, preds) in ministry_groups.items()
    }

    return ModelMetrics(
        model_name=model_name,
        overall=overall,
        by_pattern=by_pattern,
        by_ministry=by_ministry,
    )


# ── Regression comparison ─────────────────────────────────────────────────────

HISTORICAL_BASELINES: dict[str, float] = {
    "gpt-4o-mini": 0.50,
    "gemini-2.0-flash": 0.50,
    "deepseek-v3": 0.50,
    "claude-3.5-haiku": 0.50,
    "llama-3-7b": 0.48,
    "mistral-7b": 0.48,
    "qwen2.5-7b": 0.50,
    "vicuna-7b": 0.48,
    "wizardlm-2-7b": 0.52,
    "qwen-viet": 0.52,
    # Default for unknown models
    "_default": 0.45,
}


def check_regression(
    model_name: str,
    metrics: ModelMetrics,
    baseline_override: Optional[float] = None,
    submodule_path: str = "Public-Sector-Application",
) -> tuple[bool, str]:
    """
    Check if model accuracy meets or exceeds the known baseline.

    Args:
        model_name: Model identifier (must match HISTORICAL_BASELINES keys
                    or provide baseline_override).
        metrics: Computed ModelMetrics.
        baseline_override: Custom minimum accuracy threshold.
        submodule_path: Path to the submodule (unused currently, for future
                        live CSV comparison).

    Returns:
        (passed: bool, message: str)
    """
    if baseline_override is not None:
        baseline = baseline_override
    else:
        # Fuzzy match: check if any key is a substring of model_name
        baseline = HISTORICAL_BASELINES["_default"]
        for key, value in HISTORICAL_BASELINES.items():
            if key != "_default" and key.lower() in model_name.lower():
                baseline = value
                break

    actual = metrics.overall.accuracy
    passed = actual >= baseline
    direction = ">=" if passed else "<"
    msg = (
        f"[{'PASS' if passed else 'FAIL'}] {model_name}: "
        f"accuracy={actual:.4f} {direction} baseline={baseline:.4f}"
    )
    return passed, msg


def build_comparison_table(all_metrics: list[ModelMetrics]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing all evaluated models.

    Columns: model_name, accuracy, precision, recall, f1, support
    """
    rows = []
    for m in all_metrics:
        rows.append(
            {
                "model_name": m.model_name,
                "accuracy": m.overall.accuracy,
                "precision": m.overall.precision,
                "recall": m.overall.recall,
                "f1": m.overall.f1,
                "support": m.overall.support,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)
