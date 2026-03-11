# VietPS-Hallu LLM Hallucination Testing App
### Full Documentation & Demo Tutorial

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Core Idea: Dataset as Test Suite](#2-the-core-idea-dataset-as-test-suite)
3. [Repository Structure](#3-repository-structure)
4. [The Dataset (VietPS-Hallu)](#4-the-dataset-vietps-hallu)
5. [Prerequisites & Installation](#5-prerequisites--installation)
6. [Running the Test Suite (QA Demo)](#6-running-the-test-suite-qa-demo)
7. [Configuration Reference](#7-configuration-reference)
8. [Running a Live LLM Evaluation (CLI)](#8-running-a-live-llm-evaluation-cli)
9. [Results & the Streamlit Dashboard](#9-results--the-streamlit-dashboard)
10. [Module Reference](#10-module-reference)
11. [Adding a New LLM Adapter](#11-adding-a-new-llm-adapter)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [QA Concepts Demonstrated](#13-qa-concepts-demonstrated)

---

## 1. Project Overview

This repository is a **QA Automation Portfolio Project** built on top of an NLP research dataset.

The research dataset — **VietPS-Hallu** (submodule `Public-Sector-Application/`) — was originally
created to study hallucination in Vietnamese LLMs answering public-sector administrative questions.
It contains **3,717 Q&A pairs**, each with a correct answer and a manufactured hallucinated
counterpart, categorised into 4 hallucination patterns.

This project **repurposes that dataset as an automated test suite**, treating each LLM as a
**System Under Test (SUT)** and evaluating whether the model can correctly detect hallucinated answers.

**What it demonstrates for a QA Engineering role:**

| QA Concept | Implementation |
|---|---|
| Test suite design | 7,434 test cases derived from structured CSV data |
| System Under Test (SUT) abstraction | `BaseLLMAdapter` — swap any LLM without changing test logic |
| Parameterised testing | Filter by hallucination pattern, ministry, sample size |
| Metrics-driven pass/fail | Accuracy, Precision, Recall, F1 per model |
| Regression testing | Compare accuracy against historical baselines |
| Test data management | `DatasetLoader` + pandera schema validation |
| CI/CD automation | GitHub Actions runs tests on every push |
| Results reporting | JSON + CSV output, Streamlit dashboard |

---

## 2. The Core Idea: Dataset as Test Suite

```
          VietPS-Hallu Dataset
          ┌─────────────────────────────────────────────┐
          │  Q&A pair (row)                              │
          │  ├─ question        = "Thủ tục cấp thẻ...?" │
          │  ├─ cauTraLoi       = correct answer         │
          │  ├─ cauTraLoiAoGiac = hallucinated answer     │
          │  └─ pattern         = 0 | 1 | 2 | 3          │
          └─────────────────────────────────────────────┘
                         │
                         ▼ DatasetLoader.load_primary()
          ┌─────────────────────────────────────────────┐
          │  HalluTestCase (×2 per row)                  │
          │  ├─ ::correct      is_hallucinated = False   │
          │  └─ ::hallucinated is_hallucinated = True    │
          └─────────────────────────────────────────────┘
                         │
                         ▼ PromptBuilder.build()
          ┌─────────────────────────────────────────────┐
          │  Prompt text (Vietnamese)                    │
          │  = system instruction (evaluate_context)     │
          │  + question/answer data (evaluate_template)  │
          └─────────────────────────────────────────────┘
                         │
                         ▼ LLMAdapter.predict()
          ┌─────────────────────────────────────────────┐
          │  LLM responds: "Có" (yes) or "Không" (no)   │
          │  = "does this answer contain hallucination?" │
          └─────────────────────────────────────────────┘
                         │
                         ▼ Evaluator compares to ground truth
          ┌─────────────────────────────────────────────┐
          │  TestResult: passed / failed + latency_ms   │
          └─────────────────────────────────────────────┘
                         │
                         ▼ metrics.compute_model_metrics()
          ┌─────────────────────────────────────────────┐
          │  Accuracy, Precision, Recall, F1             │
          │  Breakdown: by pattern, by ministry          │
          │  Regression: vs. historical baseline         │
          └─────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
se-qa-vietps/
│
├── Public-Sector-Application/       ← git submodule (VietPS-Hallu dataset)
│   ├── C_Generate/
│   │   └── postgenerate_gpt.csv     ← 3,717 generated Q&A pairs (primary test data)
│   ├── CH_Annotate/
│   │   └── annotated_data/
│   │       ├── human1.csv           ← 300 human-annotated gold samples
│   │       └── human2.csv
│   └── DK_Evaluate/
│       ├── Template/template.csv    ← Vietnamese evaluation prompt templates
│       ├── Close_source/            ← Baseline eval CSVs (GPT-4o-mini, Gemini, etc.)
│       └── Open_source/             ← Baseline eval CSVs (LLaMA, Mistral, Qwen, etc.)
│
├── vietps_tester/                   ← Main application package
│   ├── __init__.py
│   ├── dataset_loader.py            ← Loads & validates CSV → HalluTestCase objects
│   ├── prompt_builder.py            ← Renders evaluation prompts from templates
│   ├── evaluator.py                 ← Core test runner (SUT × test suite loop)
│   ├── metrics.py                   ← accuracy / precision / recall / F1 + regression
│   ├── reporter.py                  ← Saves results as JSON + CSV
│   ├── cli.py                       ← python -m vietps_tester.cli entry point
│   └── models/
│       ├── base_model.py            ← Abstract BaseLLMAdapter + _normalize()
│       ├── openai_model.py          ← OpenAI GPT adapter
│       ├── gemini_model.py          ← Google Gemini adapter
│       ├── openrouter_model.py      ← OpenRouter (DeepSeek, Claude, etc.)
│       └── lmstudio_model.py        ← LM Studio local inference adapter
│
├── tests/
│   ├── conftest.py                  ← Fixtures: sample_test_cases, mock adapters
│   ├── unit/
│   │   ├── test_dataset_loader.py   ← 20 tests: schema, sampling, filtering
│   │   ├── test_prompt_builder.py   ← 15 tests: template rendering, modes, patterns
│   │   └── test_metrics.py          ← 15 tests: metrics correctness, edge cases
│   └── integration/
│       ├── test_model_adapters.py   ← 25 tests: HTTP-mocked API calls per adapter
│       └── test_evaluator_mock.py   ← 14 tests: full pipeline with mock adapters
│
├── dashboard/
│   └── app.py                       ← Streamlit dashboard (results visualisation)
│
├── .github/workflows/
│   ├── ci.yml                       ← Run tests on push / PR
│   └── evaluate.yml                 ← Scheduled live LLM evaluation
│
├── DOCS.md                          ← This file
├── README.md                        ← Project overview and quick-start
├── config.example.yaml              ← Copy → config.yaml, fill API keys
├── requirements.txt                 ← All Python dependencies
├── pytest.ini                       ← pytest config + markers + coverage settings
└── results/                         ← Evaluation output (JSON + CSV), gitignored
```

---

## 4. The Dataset (VietPS-Hallu)

The submodule `Public-Sector-Application/` contains the **VietPS-Hallu** dataset used as a
thesis research artefact. Here is what is relevant to the testing app.

### Primary Dataset — `C_Generate/postgenerate_gpt.csv`

| Column | Description |
|---|---|
| `link` | Source URL on dichvucong.gov.vn |
| `cauHoi` | The administrative question (Vietnamese) |
| `cauTraLoi` | A correct answer (ground truth: **not hallucinated**) |
| `cauTraLoiAoGiac` | A manufactured hallucinated answer (ground truth: **hallucinated**) |
| `pattern` | Integer 0–3 (hallucination type — see below) |
| `boNganh` | Originating ministry/department |

**Size:** 3,717 rows → **7,434 test cases** (2 per row: correct + hallucinated).

### Gold Dataset — `CH_Annotate/annotated_data/human1.csv`

300 manually selected and human-annotated samples. Same structure as the primary dataset
plus `phanLoai` (category), `TTHCLienQuan`, `cauHoiLienQuan` (related procedure links).

Use the gold dataset for highest-confidence regression testing.

### Hallucination Patterns

| Pattern | Name | Description | Prompt hint |
|---|---|---|---|
| **0** | Entity substitution | Wrong organisation name, document number, date, or person | *"Chú ý đến sự chính xác của các thực thể..."* |
| **1** | Contradictory information | Statement contradicts real-world facts | *"Chú ý đến các thông tin mâu thuẫn..."* |
| **2** | Unverifiable claims | Claims with no legal basis | *"Chú ý đến các thông tin không thể xác minh..."* |
| **3** | Factual errors | Wrong numbers, deadlines, or procedures | *"Chú ý đến các lỗi thực tế..."* |

### Historical Baselines (from research)

25 baseline evaluation CSV files from 11 LLMs live in `DK_Evaluate/`:

| Tier | Models |
|---|---|
| Close-source | GPT-4o-mini, Gemini 2.0 Flash, DeepSeek V3, Claude 3.5 Haiku |
| Open-source (general) | LLaMA-3-7B, Mistral-7B, WizardLM-2-7B, Qwen2.5-7B |
| Open-source (Vietnamese) | Qwen2.5-Viet-SFT, Vistral-7B |

---

## 5. Prerequisites & Installation

### System Requirements

- Python 3.10+
- Git (with submodule support)

### Step 1 — Clone with submodule

```bash
git clone --recurse-submodules https://github.com/fabyanbui/se-qa-vietps.git
cd se-qa-vietps
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Verify the submodule is populated:

```bash
ls Public-Sector-Application/C_Generate/postgenerate_gpt.csv
# Should print the file path — not an error
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Role |
|---|---|
| `pandas` | CSV loading and data manipulation |
| `pandera` | DataFrame schema validation (data quality gates) |
| `pydantic` | Config and model validation |
| `pytest` + `pytest-cov` | Test runner + coverage |
| `responses` | HTTP mocking for adapter integration tests |
| `PyYAML` | Config file parsing |
| `tenacity` | Retry logic in LLM adapters |
| `tqdm` | Progress bars during evaluation |
| `streamlit` | Results dashboard |
| `requests` | HTTP calls to LLM APIs |

---

## 6. Running the Test Suite (QA Demo)

This is the main thing to show as a QA demo — all 89 tests run with **no API keys needed**.

### Run all tests

```bash
pytest tests/ -v
```

### Run only unit tests (fastest, ~3 s)

```bash
pytest tests/unit/ -v
```

### Run with coverage report

```bash
pytest tests/ --cov=vietps_tester --cov-report=term-missing
```

Expected output:

```
tests/integration/test_evaluator_mock.py ..............         [ 15%]
tests/integration/test_model_adapters.py .........................   [ 43%]
tests/unit/test_dataset_loader.py ....................           [ 66%]
tests/unit/test_metrics.py ...............                       [ 83%]
tests/unit/test_prompt_builder.py ...............                [100%]

Name                                       Stmts   Miss  Cover
--------------------------------------------------------------
vietps_tester/dataset_loader.py               86      3    97%
vietps_tester/evaluator.py                    71      1    99%
vietps_tester/metrics.py                      68      0   100%
vietps_tester/models/base_model.py            41      0   100%
...
TOTAL                                        521    115    78%

89 passed in 14s
```

### Run tests filtered by marker

```bash
# Only unit tests
pytest tests/ -m unit

# Only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"
```

### Test structure explained

```
tests/
├── conftest.py                 ← Shared fixtures (no submodule needed)
│   ├── sample_test_cases       ← 4 hard-coded HalluTestCase objects
│   ├── perfect_adapter         ← Mock that always returns the right label
│   ├── always_yes_adapter      ← Mock that always says "Có" (hallucinated)
│   ├── always_no_adapter       ← Mock that always says "Không" (not hallucinated)
│   └── error_adapter           ← Mock that always raises ConnectionError
│
├── unit/test_dataset_loader    ← Loads from actual submodule CSV
├── unit/test_prompt_builder    ← Renders from actual template.csv
├── unit/test_metrics           ← Pure computation, no I/O
└── integration/
    ├── test_model_adapters     ← responses library mocks all HTTP calls
    └── test_evaluator_mock     ← Full pipeline using in-memory mock adapters
```

### Key test highlights

**1. Schema validation (data quality gate)**
```python
# test_dataset_loader.py
def test_load_primary_returns_list(self):
    loader = DatasetLoader("Public-Sector-Application")
    cases = loader.load_primary(sample_size=10)
    assert len(cases) == 20   # 10 rows × 2 cases each
```

**2. Deterministic sampling (reproducibility)**
```python
def test_load_primary_reproducible_with_seed(self):
    cases1 = loader.load_primary(sample_size=10, seed=42)
    cases2 = loader.load_primary(sample_size=10, seed=42)
    assert [tc.id for tc in cases1] == [tc.id for tc in cases2]
```

**3. Perfect adapter baseline (sanity check)**
```python
# test_evaluator_mock.py
def test_perfect_adapter_100_pass_rate(self, sample_test_cases, perfect_adapter):
    evaluator = Evaluator(SUBMODULE, verbose=False)
    runs = evaluator.run(sample_test_cases, [perfect_adapter])
    assert runs[0].pass_rate == 1.0
```

**4. Error resilience (fault tolerance)**
```python
def test_error_adapter_returns_khong(self, sample_test_cases, error_adapter):
    runs = evaluator.run(sample_test_cases, [error_adapter])
    for result in runs[0].results:
        assert result.predicted_label == "Không"   # safe default
        assert result.error is not None             # error is recorded
```

**5. Regression test (preventing quality degradation)**
```python
# test_metrics.py
def test_regression_pass_at_baseline(self):
    metrics = _make_metrics("gpt-4o-mini", accuracy=0.55)
    passed, msg = check_regression("gpt-4o-mini", metrics)
    assert passed   # 0.55 >= 0.50 baseline

def test_regression_fail_below_baseline(self):
    metrics = _make_metrics("gpt-4o-mini", accuracy=0.40)
    passed, msg = check_regression("gpt-4o-mini", metrics)
    assert not passed   # 0.40 < 0.50 baseline
```

---

## 7. Configuration Reference

Copy the example config and fill in your API keys:

```bash
cp config.example.yaml config.yaml
```

Full `config.yaml` structure:

```yaml
dataset:
  submodule_path: "Public-Sector-Application"   # Path to submodule

evaluation:
  sample_size: 100      # Rows to sample (null = all 3,717)
  pattern: null         # Filter to one pattern: 0 | 1 | 2 | 3 | null
  mode: "without_knowledge"
  # "without_knowledge" — model sees only question + answer
  # "with_knowledge"    — model also gets TTHC procedure context
  # "both"              — run both modes in sequence

models:
  - name: "gpt-4o-mini"
    provider: "openai"
    api_key: "sk-..."

  - name: "gemini-2.0-flash"
    provider: "gemini"
    api_key: "AIza..."

  - name: "deepseek-v3"
    provider: "openrouter"
    api_key: "sk-or-..."
    model_id: "deepseek/deepseek-chat"

  - name: "claude-3.5-haiku"
    provider: "openrouter"
    api_key: "sk-or-..."
    model_id: "anthropic/claude-3-5-haiku"

  - name: "local-llama"
    provider: "lmstudio"
    endpoint: "http://localhost:1234/v1"
    model_id: "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"

output:
  results_dir: "results"
```

### Supported providers

| `provider` value | API target | Required fields |
|---|---|---|
| `openai` | OpenAI Chat Completions | `api_key` |
| `gemini` | Google Generative Language | `api_key` |
| `openrouter` | OpenRouter unified API | `api_key`, `model_id` |
| `lmstudio` | LM Studio local server | `endpoint`, `model_id` |

---

## 8. Running a Live LLM Evaluation (CLI)

### Minimal example (OpenAI, 20 samples)

```bash
# 1. Create config
cat > config.yaml << 'EOF'
dataset:
  submodule_path: "Public-Sector-Application"
evaluation:
  sample_size: 20
  mode: "without_knowledge"
models:
  - name: "gpt-4o-mini"
    provider: "openai"
    api_key: "sk-..."
output:
  results_dir: "results"
EOF

# 2. Run
python -m vietps_tester.cli --config config.yaml
```

### CLI output

```
📋 Loading 20 test cases...
   Loaded 40 test cases (×2 correct+hallucinated per row)

🚀 Running evaluation — mode: without_knowledge
Evaluating gpt-4o-mini: 100%|████████████████| 40/40 [01:23<00:00]

📊 Results:
  ✅ [PASS] gpt-4o-mini: accuracy=0.5750 >= baseline=0.5000

💾 Saved results to: results/
   results/gpt-4o-mini_20260311_140000.json
   Summary: results/summary_20260311_140000.csv

✔ Evaluation complete.
```

### Override CLI flags

```bash
# Override sample size from config
python -m vietps_tester.cli --config config.yaml --sample-size 50

# Run both evaluation modes
python -m vietps_tester.cli --config config.yaml --mode both
```

### Multi-model comparison

Put multiple models in `config.yaml` and they all run in sequence:

```bash
# config.yaml has gpt-4o-mini, gemini-2.0-flash, deepseek-v3
python -m vietps_tester.cli --config config.yaml --sample-size 100
```

Output (results CSV snippet):

```
model_name,accuracy,precision,recall,f1,support
gpt-4o-mini,0.5750,0.5652,0.6500,0.6047,40
gemini-2.0-flash,0.5500,0.5400,0.6750,0.6000,40
deepseek-v3,0.5250,0.5100,0.5100,0.5100,40
```

### Use the gold dataset (Python API)

```python
from vietps_tester.dataset_loader import DatasetLoader
from vietps_tester.evaluator import Evaluator

loader = DatasetLoader("Public-Sector-Application")
# 300 human-annotated gold samples, both correct + hallucinated answers
gold_cases = loader.load_gold(annotator=1, sample_size=50)

evaluator = Evaluator("Public-Sector-Application")
# ... pass your adapter
```

### Full Python API

```python
from vietps_tester.dataset_loader import DatasetLoader
from vietps_tester.evaluator import Evaluator
from vietps_tester.reporter import Reporter
from vietps_tester.metrics import check_regression
from vietps_tester.models import OpenAIAdapter, AdapterConfig

# 1. Load test cases
loader = DatasetLoader("Public-Sector-Application")
cases = loader.load_primary(sample_size=50, pattern=0)   # only Pattern 0

# 2. Create adapter(s)
adapter = OpenAIAdapter(AdapterConfig(
    name="gpt-4o-mini",
    api_key="sk-...",
))

# 3. Run evaluation
evaluator = Evaluator("Public-Sector-Application")
runs = evaluator.run(cases, [adapter], mode="without_knowledge")

# 4. Check regression
for run in runs:
    passed, msg = check_regression(run.model_name, run.metrics)
    print(msg)   # [PASS] gpt-4o-mini: accuracy=0.5750 >= baseline=0.5000

# 5. Save results
reporter = Reporter("results")
reporter.save(runs)
```

---

## 9. Results & the Streamlit Dashboard

### Results files

After each evaluation run, two files are written to `results/`:

**JSON** (`results/gpt-4o-mini_20260311_140000.json`):
```json
{
  "model_name": "gpt-4o-mini",
  "mode": "without_knowledge",
  "timestamp": "20260311_140000",
  "pass_rate": 0.575,
  "passed": 23,
  "failed": 17,
  "total": 40,
  "metrics": {
    "overall": { "accuracy": 0.575, "precision": 0.5652, "recall": 0.65, "f1": 0.6047, "support": 40 },
    "by_pattern": {
      "0": { "accuracy": 0.6, "precision": 0.6, "recall": 0.75, "f1": 0.6667, "support": 10 },
      "1": { "accuracy": 0.5, "precision": 0.5, "recall": 0.6, "f1": 0.5455, "support": 10 }
    },
    "by_ministry": {
      "Bộ Thông tin và Truyền thông": { "accuracy": 0.7, "f1": 0.72, "support": 10 }
    }
  },
  "results": [
    {
      "id": "https://dichvucong.gov.vn/...::hallucinated",
      "is_hallucinated_truth": true,
      "pattern": 0,
      "predicted_label": "Có",
      "predicted_hallucinated": true,
      "passed": true,
      "latency_ms": 824.5,
      "error": null
    }
  ]
}
```

**CSV summary** (`results/summary_20260311_140000.csv`):
```csv
model_name,accuracy,precision,recall,f1,support
gpt-4o-mini,0.575,0.5652,0.65,0.6047,40
```

### Start the dashboard

```bash
pip install streamlit   # if not already installed
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501`. The dashboard shows:

- **Model Comparison Overview** — accuracy / precision / recall / F1 table
- **Accuracy Bar Chart** — visual comparison across all evaluated models
- **Pattern Breakdown** — pivot table: each model × each hallucination pattern
- **Ministry Breakdown** — per-ministry accuracy for any selected run

Filter controls in the sidebar let you narrow by model name and evaluation mode.

> **Note:** The dashboard reads from `results/*.json`. Run at least one evaluation first.

---

## 10. Module Reference

### `dataset_loader.py`

**Classes:**
- `DatasetLoader(submodule_path)` — main loader
  - `load_primary(sample_size, pattern, ministry, seed)` → `list[HalluTestCase]`
  - `load_gold(annotator, sample_size, pattern, ministry, seed)` → `list[HalluTestCase]`
  - `available_ministries()` → `list[str]`
- `HalluTestCase` (frozen dataclass) — `id, link, question, answer, is_hallucinated, pattern, ministry`
- `PostGenerateSchema` (pandera) — validates primary CSV columns
- `AnnotatedSchema` (pandera) — validates gold CSV columns

Schema validation runs automatically on load. If the submodule CSV is corrupted or
columns are missing, a `pandera.errors.SchemaError` is raised before any test cases are built.

### `prompt_builder.py`

**Classes:**
- `PromptBuilder(submodule_path)` — loads `DK_Evaluate/Template/template.csv`
  - `build(question, answer, mode, model_type, pattern, knowledge)` → `Prompt`
  - `build_from_test_case(...)` → `Prompt` (convenience wrapper)
- `Prompt` (frozen dataclass) — `text, model_type, mode, pattern`

**Two-part prompt structure:**
1. `evaluate_context` (system instruction) — tells the model its role and how to respond
2. `evaluate_template` (data block) — fills in `{knowledge}`, `{question}`, `{answer}`

Final `Prompt.text` = `evaluate_context.format(pattern=...)` + `"\n\n"` + `evaluate_template.format(...)`.

### `evaluator.py`

**Classes:**
- `Evaluator(submodule_path, model_type, verbose)`
  - `run(test_cases, adapters, mode, knowledge_map)` → `list[EvaluationRun]`
- `EvaluationRun` — `model_name, mode, results, metrics, pass_rate, total_count`
- `TestResult` — `test_case, model_name, predicted_label, passed, latency_ms, error`

If `adapter.predict()` raises any exception, `error=str(exc)` is stored on `TestResult`,
`predicted_label` defaults to `"Không"`, and evaluation continues without crashing.

### `metrics.py`

**Functions:**
- `compute_model_metrics(model_name, test_cases, predictions)` → `ModelMetrics`
- `check_regression(model_name, metrics, baseline_override)` → `(bool, str)`
- `build_comparison_table(all_metrics)` → `pd.DataFrame`

**Classes:**
- `ModelMetrics` — `model_name, overall, by_pattern, by_ministry`
- `ClassificationMetrics` — `accuracy, precision, recall, f1, support`

### `models/base_model.py`

- `BaseLLMAdapter(ABC)` — abstract; subclasses implement `_call_api(prompt: str) -> str`
  - `predict(prompt: str) -> str` — calls `_call_api`, then normalises output
  - `_normalize(raw: str) -> str` — maps LLM free-text to `"Có"` / `"Không"`
- `AdapterConfig` (dataclass) — `name, api_key, endpoint, model_id, max_retries, timeout, temperature, max_tokens`

Normalisation accepts: `"Có"`, `"Không"`, case-insensitive variants, `"yes"`/`"no"`,
`"true"`/`"false"`. Defaults to `"Không"` for unrecognised output.

### `reporter.py`

- `Reporter(results_dir)`
  - `save(runs, tag)` → `{ "json_files": [...], "summary_csv": [...] }`
  - `list_runs()` → list of metadata dicts (used by dashboard)
  - `load_run(path)` → raw dict (for inspection or re-analysis)

---

## 11. Adding a New LLM Adapter

To test a model not yet supported, add one file. Example: **Anthropic Claude** (direct API):

**`vietps_tester/models/claude_model.py`**:

```python
from __future__ import annotations
import requests
from .base_model import BaseLLMAdapter, AdapterConfig


class ClaudeAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude (direct Messages API)."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise ValueError("ClaudeAdapter requires api_key")

    def _call_api(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.model_id or "claude-3-5-haiku-20241022",
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
```

**Register it** in `vietps_tester/models/__init__.py`:

```python
from .claude_model import ClaudeAdapter  # add this line
```

**Add it to the CLI** in `vietps_tester/cli.py`:

```python
provider_map = {
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "openrouter": OpenRouterAdapter,
    "lmstudio": LMStudioAdapter,
    "claude": ClaudeAdapter,   # add this line
}
```

**Write an integration test** in `tests/integration/test_model_adapters.py`:

```python
@pytest.mark.integration
class TestClaudeAdapter:
    def _make_adapter(self):
        return ClaudeAdapter(AdapterConfig(
            name="claude-test", api_key="test-key",
            model_id="claude-3-5-haiku-20241022"
        ))

    @resp_mock.activate
    def test_returns_co_when_model_says_co(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.anthropic.com/v1/messages",
            json={"content": [{"text": "Có"}]},
        )
        assert self._make_adapter().predict(SAMPLE_PROMPT) == LABEL_YES

    @resp_mock.activate
    def test_raises_on_http_error(self):
        resp_mock.add(
            resp_mock.POST,
            "https://api.anthropic.com/v1/messages",
            status=429,
        )
        with pytest.raises(Exception):
            self._make_adapter().predict(SAMPLE_PROMPT)
```

That's the full extension pattern — **one file, one registration, one test class**.

---

## 12. CI/CD Pipeline

### `ci.yml` — Tests on every push

Triggered on: `push` to `main`/`develop`, `pull_request` to `main`.

Steps:
1. Checkout repository **with submodule** (`submodules: recursive`)
2. Set up Python 3.12 with pip cache
3. `pip install -r requirements.txt`
4. `pytest tests/unit tests/integration --cov=vietps_tester --cov-report=xml`
5. Upload `coverage.xml` to Codecov

> All 89 tests run without any API keys — adapters are HTTP-mocked.

### `evaluate.yml` — Scheduled LLM evaluation

Triggered on: weekly schedule (or manual `workflow_dispatch`).

This workflow requires secrets set in GitHub → Settings → Secrets:

```
OPENAI_API_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
```

Steps:
1. Checkout with submodule
2. Install dependencies
3. Build `config.yaml` from secrets (env substitution)
4. Run `python -m vietps_tester.cli --config config.yaml --sample-size 200`
5. Upload `results/` as a workflow artifact

This gives you a **historical record** of LLM performance over time — viewable in
GitHub Actions artifacts.

---

## 13. QA Concepts Demonstrated

### The Adapter Pattern as SUT Abstraction

```python
# BaseLLMAdapter defines the contract
class BaseLLMAdapter(ABC):
    @abstractmethod
    def _call_api(self, prompt: str) -> str: ...

# Any class that implements _call_api becomes a testable SUT
class OpenAIAdapter(BaseLLMAdapter): ...
class MockAdapter(BaseLLMAdapter):
    def _call_api(self, prompt): return "Có"
```

The test suite does not know (or care) whether it is talking to GPT-4o or a mock.
This is the **Adapter pattern** applied to test infrastructure.

### Schema Validation as a Data Quality Gate

```python
class PostGenerateSchema(pa.DataFrameModel):
    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    cauHoi: Series[str] = pa.Field(nullable=False)
    cauTraLoi: Series[str] = pa.Field(nullable=False)
    pattern: Series[int] = pa.Field(ge=0, le=3)
```

Before any test case is created, pandera validates the CSV against this schema.
If the data changes (column renamed, pattern out of range, null values), the test fails
with a clear `SchemaError` — not a confusing `KeyError` deep in the pipeline.

### Regression Testing against Historical Baselines

The research paper established accuracy baselines for 11 LLMs on the same dataset.
`check_regression()` compares any new run against those baselines, and the CI pipeline
will fail if a model's accuracy drops below its known floor.

```python
HISTORICAL_BASELINES = {
    "gpt-4o-mini": 0.50,
    "wizardlm-2-7b": 0.52,
    "_default": 0.45,
}
```

This prevents **quality regression** — if someone changes the prompt template in a way
that degrades detection accuracy, the CI check catches it.

### Deterministic Test Data (Reproducibility)

```python
loader.load_primary(sample_size=100, seed=42)
```

Every evaluation run uses an explicit random seed for the sampling step. Two runs with
the same `seed` and `sample_size` will use exactly the same test cases. This makes
failures reproducible and enables fair A/B comparison between models.

### Isolation: No Real APIs in Unit/Integration Tests

All 89 tests pass with zero API calls. The integration tests use the `responses` library
to intercept HTTP calls and return pre-configured JSON responses. This means:

- Tests run in < 15 seconds total
- Tests are free (no API costs)
- Tests work offline and are not flaky due to network issues

```python
@resp_mock.activate
def test_returns_co_on_success(self):
    resp_mock.add(
        resp_mock.POST,
        "https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "Có"}}]},
    )
    assert adapter.predict(SAMPLE_PROMPT) == LABEL_YES
```

### Fault Tolerance by Design

When a model API call fails (network error, rate limit, etc.), the evaluator records
the error and continues — it does not crash the entire evaluation run:

```python
# evaluator.py
try:
    predicted_label = adapter.predict(prompt.text)
    error = None
except Exception as exc:
    predicted_label = "Không"   # safe default
    error = str(exc)            # captured for inspection
```

This mirrors real production QA thinking: a failing probe should report, not explode.

---

*Built on VietPS-Hallu — Vietnamese Public Sector Hallucination Detection Dataset.*
