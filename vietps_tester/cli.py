"""
vietps_tester/cli.py — Command-line interface for running evaluations.

Usage:
    python -m vietps_tester.cli --config config.yaml
    python -m vietps_tester.cli --config config.yaml --sample-size 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def build_adapters(model_configs: list[dict]):
    """Build LLM adapter instances from config dicts."""
    from vietps_tester.models import (
        AdapterConfig,
        OpenAIAdapter,
        GeminiAdapter,
        OpenRouterAdapter,
        LMStudioAdapter,
    )

    adapters = []
    provider_map = {
        "openai": OpenAIAdapter,
        "gemini": GeminiAdapter,
        "openrouter": OpenRouterAdapter,
        "lmstudio": LMStudioAdapter,
    }

    for cfg in model_configs:
        provider = cfg.get("provider", "openai")
        cls = provider_map.get(provider)
        if cls is None:
            print(f"[WARN] Unknown provider '{provider}', skipping {cfg.get('name')}")
            continue

        adapter_cfg = AdapterConfig(
            name=cfg.get("name", provider),
            api_key=cfg.get("api_key", ""),
            endpoint=cfg.get("endpoint", ""),
            model_id=cfg.get("model_id", ""),
        )
        try:
            adapters.append(cls(adapter_cfg))
        except ValueError as e:
            print(f"[WARN] Skipping {cfg.get('name')}: {e}")

    return adapters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VietPS-Hallu LLM Hallucination Testing Runner"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--sample-size", type=int, help="Override sample_size from config")
    parser.add_argument("--mode", choices=["without_knowledge", "with_knowledge", "both"])
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.sample_size:
        cfg.setdefault("evaluation", {})["sample_size"] = args.sample_size
    if args.mode:
        cfg.setdefault("evaluation", {})["mode"] = args.mode

    submodule_path = cfg["dataset"]["submodule_path"]
    sample_size = cfg.get("evaluation", {}).get("sample_size", 50)
    mode = cfg.get("evaluation", {}).get("mode", "without_knowledge")
    results_dir = cfg.get("output", {}).get("results_dir", "results")

    # Build components
    from vietps_tester.dataset_loader import DatasetLoader
    from vietps_tester.evaluator import Evaluator
    from vietps_tester.reporter import Reporter
    from vietps_tester.metrics import check_regression

    loader = DatasetLoader(submodule_path)
    evaluator = Evaluator(submodule_path, verbose=True)
    reporter = Reporter(results_dir)
    adapters = build_adapters(cfg.get("models", []))

    if not adapters:
        print("[ERROR] No valid adapters configured. Check config.yaml.")
        return 1

    print(f"\n📋 Loading {sample_size} test cases...")
    test_cases = loader.load_primary(sample_size=sample_size)
    print(f"   Loaded {len(test_cases)} test cases (×2 correct+hallucinated per row)")

    modes = ["without_knowledge", "with_knowledge"] if mode == "both" else [mode]
    all_runs = []

    for eval_mode in modes:
        print(f"\n🚀 Running evaluation — mode: {eval_mode}")
        runs = evaluator.run(test_cases, adapters, mode=eval_mode)
        all_runs.extend(runs)

        print("\n📊 Results:")
        for run in runs:
            passed, msg = check_regression(run.model_name, run.metrics)
            status = "✅" if passed else "⚠️"
            print(f"  {status} {msg}")

    paths = reporter.save(all_runs)
    print(f"\n💾 Saved results to: {results_dir}/")
    for p in paths.get("json_files", []):
        print(f"   {p}")
    if paths.get("summary_csv"):
        print(f"   Summary: {paths['summary_csv'][0]}")

    print("\n✔ Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
