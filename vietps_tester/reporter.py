"""
reporter.py — Persist evaluation results and generate comparison summaries.

Results are saved as:
  - JSON: full structured output (results/gpt-4o-mini_20250311_131727.json)
  - CSV:  summary comparison table (results/summary_20250311_131727.csv)

Also applies Allure annotations when allure-pytest is installed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .evaluator import EvaluationRun, TestResult
from .metrics import ModelMetrics, build_comparison_table


def _result_to_dict(r: TestResult) -> dict:
    return {
        "id": r.test_case.id,
        "link": r.test_case.link,
        "is_hallucinated_truth": r.ground_truth,
        "pattern": r.test_case.pattern,
        "ministry": r.test_case.ministry,
        "predicted_label": r.predicted_label,
        "predicted_hallucinated": r.predicted_hallucinated,
        "passed": r.passed,
        "latency_ms": r.latency_ms,
        "error": r.error,
    }


def _metrics_to_dict(m: ModelMetrics) -> dict:
    return {
        "overall": {
            "accuracy": m.overall.accuracy,
            "precision": m.overall.precision,
            "recall": m.overall.recall,
            "f1": m.overall.f1,
            "support": m.overall.support,
        },
        "by_pattern": {
            str(k): {
                "accuracy": v.accuracy,
                "precision": v.precision,
                "recall": v.recall,
                "f1": v.f1,
                "support": v.support,
            }
            for k, v in m.by_pattern.items()
        },
        "by_ministry": {
            k: {
                "accuracy": v.accuracy,
                "f1": v.f1,
                "support": v.support,
            }
            for k, v in m.by_ministry.items()
        },
    }


class Reporter:
    """
    Saves evaluation results to disk and generates summary reports.

    Usage::

        reporter = Reporter(results_dir="results")
        paths = reporter.save(runs)
        print(f"Saved to: {paths}")
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        runs: list[EvaluationRun],
        tag: Optional[str] = None,
    ) -> dict[str, list[Path]]:
        """
        Persist all evaluation runs to disk.

        Args:
            runs: List of EvaluationRun objects from the Evaluator.
            tag: Optional label appended to filenames (e.g. "nightly").

        Returns:
            Dict with keys "json_files" and "summary_csv".
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""
        json_paths: list[Path] = []

        for run in runs:
            filename = f"{run.model_name}{suffix}_{timestamp}.json"
            path = self.results_dir / filename
            data = {
                "model_name": run.model_name,
                "mode": run.mode,
                "timestamp": timestamp,
                "pass_rate": run.pass_rate,
                "passed": run.passed_count,
                "failed": run.failed_count,
                "total": run.total_count,
                "metrics": _metrics_to_dict(run.metrics) if run.metrics else {},
                "results": [_result_to_dict(r) for r in run.results],
            }
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            json_paths.append(path)

        # Summary CSV
        all_metrics = [r.metrics for r in runs if r.metrics]
        summary_path = None
        if all_metrics:
            df = build_comparison_table(all_metrics)
            summary_filename = f"summary{suffix}_{timestamp}.csv"
            summary_path = self.results_dir / summary_filename
            df.to_csv(summary_path, index=False)

        return {
            "json_files": json_paths,
            "summary_csv": [summary_path] if summary_path else [],
        }

    def list_runs(self) -> list[dict]:
        """
        Return metadata for all stored evaluation runs, newest first.

        Returns a list of dicts with: model_name, timestamp, accuracy, path.
        """
        runs = []
        for path in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                runs.append(
                    {
                        "model_name": data.get("model_name", "unknown"),
                        "timestamp": data.get("timestamp", ""),
                        "mode": data.get("mode", ""),
                        "accuracy": data.get("metrics", {})
                        .get("overall", {})
                        .get("accuracy", 0.0),
                        "pass_rate": data.get("pass_rate", 0.0),
                        "total": data.get("total", 0),
                        "path": str(path),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return runs

    def load_run(self, path: str) -> dict:
        """Load a single stored evaluation run from a JSON file."""
        return json.loads(Path(path).read_text())
