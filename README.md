# VietPS-Hallu — LLM Hallucination Testing App

> **QA Automation Testing Portfolio Project**

A Python-based test automation framework that evaluates multiple Large Language Models (LLMs) on the **VietPS-Hallu dataset** — a Vietnamese public service hallucination detection benchmark built by Nguyễn Tiến Nhật & Bùi Đình Bảo (thesis, 2025).

---

## Project Concept

| QA Role Concept | How It Maps Here |
|-----------------|-----------------|
| **Test suite** | VietPS-Hallu dataset (~12,000 Q&A pairs with ground truth labels) |
| **System Under Test (SUT)** | LLMs: GPT-4o-mini, Gemini 2.0, DeepSeek V3, Claude 3.5 Haiku, local models |
| **Test cases** | Each Q&A sample with a correct/hallucinated answer and expected label |
| **Assertions** | Model label (`Có`/`Không`) must match ground truth |
| **Test runner** | `vietps_tester/evaluator.py` |
| **Test reports** | JSON results + Streamlit dashboard |
| **Regression testing** | Accuracy must stay ≥ known baseline per model |
| **CI/CD** | GitHub Actions — push-triggered tests + scheduled weekly evaluation |

---

## Repository Structure

```
se-qa-vietps/
├── vietps_tester/              # Core application package
│   ├── dataset_loader.py       # Loads test cases from VietPS-Hallu CSV files
│   ├── prompt_builder.py       # Renders evaluation prompts from templates
│   ├── evaluator.py            # Test runner (send prompt → assert label)
│   ├── metrics.py              # Accuracy, Precision, Recall, F1 + regression check
│   ├── reporter.py             # Save results to JSON/CSV
│   ├── cli.py                  # Command-line interface
│   └── models/                 # LLM adapters (systems under test)
│       ├── base_model.py       # Abstract adapter interface
│       ├── openai_model.py     # GPT-4o-mini
│       ├── gemini_model.py     # Gemini 2.0 Flash
│       ├── openrouter_model.py # DeepSeek V3, Claude 3.5 Haiku via OpenRouter
│       └── lmstudio_model.py   # Local models via LM Studio
│
├── tests/
│   ├── conftest.py             # Shared fixtures + mock adapters
│   ├── unit/
│   │   ├── test_dataset_loader.py   # Schema validation, sampling, filtering
│   │   ├── test_prompt_builder.py   # Template rendering, all patterns
│   │   └── test_metrics.py          # Accuracy/F1 with known inputs
│   └── integration/
│       ├── test_model_adapters.py   # HTTP mocked adapter tests
│       └── test_evaluator_mock.py   # Full pipeline with mock LLMs
│
├── dashboard/app.py            # Streamlit results visualization
├── results/                    # Stored evaluation outputs (JSON + CSV)
├── .github/workflows/
│   ├── ci.yml                  # pytest on every push/PR
│   └── evaluate.yml            # Scheduled weekly LLM evaluation
├── pytest.ini                  # Test configuration
├── config.example.yaml         # Example configuration (no real keys)
└── Public-Sector-Application/  # Submodule: VietPS-Hallu dataset
```

---

## Setup

### Prerequisites

- Python 3.12+
- Git (with submodule support)

### Install

```bash
# 1. Clone with submodule
git clone --recursive https://github.com/fabyanbui/se-qa-vietps.git
cd se-qa-vietps

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml — add your API keys
```

---

## Running Tests

```bash
# Run all tests (unit + integration, mocked — no API keys needed)
pytest

# Run only unit tests
pytest tests/unit -v

# Run only integration tests
pytest tests/integration -v

# Run with coverage report
pytest --cov=vietps_tester --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Running an Evaluation

```bash
# Evaluate configured LLMs on 100 samples
python -m vietps_tester.cli --config config.yaml --sample-size 100

# With knowledge context (RAG mode)
python -m vietps_tester.cli --config config.yaml --mode with_knowledge

# View results in dashboard
streamlit run dashboard/app.py
```

---

## Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Features:
- 📊 **Model comparison table** (accuracy, precision, recall, F1)
- 📈 **Accuracy bar chart** across models
- 🔬 **Pattern breakdown** (performance per hallucination type)
- 🏛️ **Ministry breakdown** (performance by Vietnamese government ministry)

---

## Adding a New LLM

1. Create `vietps_tester/models/my_new_model.py`:
   ```python
   from .base_model import AdapterConfig, BaseLLMAdapter

   class MyNewAdapter(BaseLLMAdapter):
       def _call_api(self, prompt: str) -> str:
           # Call your LLM API here and return raw text
           ...
   ```
2. Export it from `vietps_tester/models/__init__.py`
3. Add provider entry in `vietps_tester/cli.py` `provider_map`
4. Add model config to `config.yaml`

---

## QA Methodology

### Test Pyramid

```
    ┌─────────────┐
    │  Dashboard  │  (manual verification)
    ├─────────────┤
    │ Integration │  test_evaluator_mock, test_model_adapters
    ├─────────────┤
    │    Unit     │  test_dataset_loader, test_prompt_builder, test_metrics
    └─────────────┘
```

### Hallucination Patterns Tested

| Pattern | Description |
|---------|-------------|
| 0 | **Entity substitution** — wrong agency/law names cited |
| 1 | **Contradictory info** — answer contradicts facts |
| 2 | **Unverifiable claims** — fabricated references |
| 3 | **Factual errors** — wrong numbers/dates/procedures |

### Regression Baseline

All models must achieve ≥ 45–52% accuracy (known baseline from thesis evaluation).
Models below baseline trigger a `[FAIL]` regression warning in CI.

---

## Dataset Credit

Data sourced from the **VietPS-Hallu** submodule (`Public-Sector-Application/`).

- Source: [dichvucong.gov.vn](https://dichvucong.gov.vn)
- 12,473 Q&A pairs + 90,241 administrative procedures (TTHC)
- ~12,000 hallucinated responses across 4 patterns
- 1,000 human-annotated gold samples

> **Authors:** Nguyễn Tiến Nhật (21120108) & Bùi Đình Bảo (21120201)
> **Supervisors:** Nguyễn Tiến Huy, Lê Thanh Tùng — HCMUS, 2025
