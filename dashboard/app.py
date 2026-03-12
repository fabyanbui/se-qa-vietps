"""
dashboard/app.py — Streamlit dashboard for VietPS-Hallu LLM evaluation results.

Features:
  - View all past evaluation runs
  - Compare accuracy/F1 across models
  - Breakdown by hallucination pattern (0-3)
  - Breakdown by ministry (bộ ngành)
  - Run a new evaluation from the UI
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VietPS-Hallu — LLM Test Dashboard",
    page_icon="🧪",
    layout="wide",
)

RESULTS_DIR = Path("results")
PATTERN_NAMES = {
    "-1": "Correct answer",
    "0": "Pattern 0: Entity substitution",
    "1": "Pattern 1: Contradictory info",
    "2": "Pattern 2: Unverifiable claims",
    "3": "Pattern 3: Factual errors",
}


# ── Load results ───────────────────────────────────────────────────────────────

def load_all_runs() -> list[dict]:
    """Load metadata for all stored evaluation runs."""
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            runs.append(
                {
                    "filename": path.name,
                    "model_name": data.get("model_name", "unknown"),
                    "timestamp": data.get("timestamp", ""),
                    "mode": data.get("mode", ""),
                    "accuracy": data.get("metrics", {})
                    .get("overall", {})
                    .get("accuracy", 0.0),
                    "f1": data.get("metrics", {})
                    .get("overall", {})
                    .get("f1", 0.0),
                    "precision": data.get("metrics", {})
                    .get("overall", {})
                    .get("precision", 0.0),
                    "recall": data.get("metrics", {})
                    .get("overall", {})
                    .get("recall", 0.0),
                    "support": data.get("total", 0),
                    "pass_rate": data.get("pass_rate", 0.0),
                    "path": str(path),
                    "_raw": data,
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("🧪 VietPS-Hallu — LLM Hallucination Testing Dashboard")
st.caption(
    "Automated QA evaluation of LLMs on the VietPS-Hallu dataset "
    "(Vietnamese public service hallucination detection)."
)

runs = load_all_runs()

if not runs:
    st.warning(
        "No evaluation results found in `results/`. "
        "Run an evaluation first using:\n\n"
        "```bash\npython -m vietps_tester.cli --config config.yaml\n```"
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────

st.sidebar.header("🔍 Filters")
all_models = sorted({r["model_name"] for r in runs})
selected_models = st.sidebar.multiselect(
    "Filter by model", all_models, default=all_models
)
all_modes = sorted({r["mode"] for r in runs if r["mode"]})
selected_mode = st.sidebar.selectbox(
    "Evaluation mode",
    ["all"] + all_modes,
    index=0,
)

filtered = [
    r
    for r in runs
    if r["model_name"] in selected_models
    and (selected_mode == "all" or r["mode"] == selected_mode)
]

# ── Overview table ────────────────────────────────────────────────────────────

st.subheader("📊 Model Comparison Overview")

if filtered:
    overview_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Mode": r["mode"],
                "Accuracy": f"{r['accuracy']:.1%}",
                "Precision": f"{r['precision']:.1%}",
                "Recall": f"{r['recall']:.1%}",
                "F1": f"{r['f1']:.1%}",
                "Samples": r["support"],
                "Timestamp": r["timestamp"],
            }
            for r in filtered
        ]
    ).drop_duplicates(subset=["Model", "Mode"])

    st.dataframe(overview_df, width="stretch")

    # Accuracy bar chart
    st.subheader("📈 Accuracy by Model")
    chart_df = pd.DataFrame(
        [{"Model": r["model_name"], "Accuracy": r["accuracy"]} for r in filtered]
    ).drop_duplicates("Model")

    st.bar_chart(chart_df.set_index("Model")["Accuracy"])

# ── Pattern breakdown ─────────────────────────────────────────────────────────

st.subheader("🔬 Pattern Breakdown")
st.caption(
    "Accuracy per hallucination pattern across all selected models."
)

pattern_rows = []
for r in filtered:
    by_pattern = r["_raw"].get("metrics", {}).get("by_pattern", {})
    for pat_key, pat_metrics in by_pattern.items():
        if pat_key == "-1":
            continue
        pattern_rows.append(
            {
                "Model": r["model_name"],
                "Pattern": PATTERN_NAMES.get(pat_key, f"Pattern {pat_key}"),
                "Accuracy": pat_metrics.get("accuracy", 0.0),
                "F1": pat_metrics.get("f1", 0.0),
                "Samples": pat_metrics.get("support", 0),
            }
        )

if pattern_rows:
    pattern_df = pd.DataFrame(pattern_rows)
    pivot = pattern_df.pivot_table(
        index="Model", columns="Pattern", values="Accuracy"
    ).fillna(0)
    st.dataframe(pivot.style.format("{:.1%}"), width="stretch")

# ── Ministry breakdown ────────────────────────────────────────────────────────

st.subheader("🏛️ Ministry (Bộ Ngành) Breakdown")

if filtered:
    selected_run_name = st.selectbox(
        "Select model run to inspect:",
        [f"{r['model_name']} @ {r['timestamp']}" for r in filtered],
    )
    idx = next(
        (
            i
            for i, r in enumerate(filtered)
            if f"{r['model_name']} @ {r['timestamp']}" == selected_run_name
        ),
        0,
    )
    by_ministry = filtered[idx]["_raw"].get("metrics", {}).get("by_ministry", {})
    if by_ministry:
        ministry_df = pd.DataFrame(
            [
                {
                    "Ministry": k,
                    "Accuracy": v.get("accuracy", 0.0),
                    "F1": v.get("f1", 0.0),
                    "Samples": v.get("support", 0),
                }
                for k, v in by_ministry.items()
            ]
        ).sort_values("Accuracy", ascending=False)
        st.dataframe(
            ministry_df.style.format({"Accuracy": "{:.1%}", "F1": "{:.1%}"}),
            width="stretch",
        )
    else:
        st.info("No ministry-level data in this run (gold dataset required).")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built on VietPS-Hallu — "
    "[GitHub](https://github.com/fabyanbui/se-qa-vietps) | "
    "Data: Public-Sector-Application submodule"
)
