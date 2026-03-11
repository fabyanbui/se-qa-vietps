"""
prompt_builder.py — Build evaluation prompts from templates and test cases.

Reuses the prompt templates from:
    Public-Sector-Application/DK_Evaluate/Template/template.csv

Two modes:
  - without_knowledge: model decides based on answer alone
  - with_knowledge:    model is given TTHC context as supporting knowledge
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


EvaluationMode = Literal["without_knowledge", "with_knowledge"]
ModelType = Literal["open_source", "close_source"]

# Pattern-specific instructions (from COMPREHENSIVE_DOCUMENTATION.md)
PATTERN_INSTRUCTIONS: dict[int, str] = {
    0: (
        "Hãy chú ý đến sự chính xác của các thực thể (entity) trong câu trả lời, "
        "bao gồm tên cơ quan, số văn bản, ngày ban hành và tên người."
    ),
    1: (
        "Hãy chú ý đến các thông tin mâu thuẫn hoặc trái ngược với kiến thức "
        "thực tế trong câu trả lời."
    ),
    2: (
        "Hãy chú ý đến các thông tin không thể xác minh hoặc không có cơ sở "
        "pháp lý rõ ràng trong câu trả lời."
    ),
    3: (
        "Hãy chú ý đến các lỗi thực tế (factual errors) trong câu trả lời, "
        "bao gồm các con số, thời hạn và quy trình không đúng."
    ),
}


@dataclass(frozen=True)
class Prompt:
    """A fully rendered evaluation prompt ready to send to an LLM."""

    text: str
    model_type: ModelType
    mode: EvaluationMode
    pattern: int


class PromptBuilder:
    """
    Renders evaluation prompts using templates from the submodule.

    Usage::

        builder = PromptBuilder("Public-Sector-Application")
        prompt = builder.build(
            question="Thủ tục cấp thẻ nhà báo như thế nào?",
            answer="...",
            mode="without_knowledge",
            model_type="close_source",
            pattern=0,
        )
    """

    def __init__(self, submodule_path: str = "Public-Sector-Application") -> None:
        self.base = Path(submodule_path)
        self._templates = self._load_templates()

    def _load_templates(self) -> pd.DataFrame:
        template_path = self.base / "DK_Evaluate" / "Template" / "template.csv"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template file not found: {template_path}. "
                "Run: git submodule update --init"
            )
        return pd.read_csv(template_path, encoding="utf-8-sig")

    def _get_template(self, model_type: ModelType) -> tuple[str, str]:
        """Return (system_template, data_template) for the given model type."""
        row = self._templates[self._templates.iloc[:, 0] == model_type]
        if row.empty:
            raise ValueError(
                f"No template found for model_type='{model_type}'. "
                f"Available: {self._templates.iloc[:, 0].tolist()}"
            )
        context = row.iloc[0]["evaluate_context"]
        data_tmpl = row.iloc[0]["evaluate_template"]
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"evaluate_context for '{model_type}' is empty.")
        if not isinstance(data_tmpl, str) or not data_tmpl.strip():
            raise ValueError(f"evaluate_template for '{model_type}' is empty.")
        return context, data_tmpl

    def build(
        self,
        question: str,
        answer: str,
        mode: EvaluationMode = "without_knowledge",
        model_type: ModelType = "close_source",
        pattern: int = -1,
        knowledge: str = "",
    ) -> Prompt:
        """
        Build an evaluation prompt.

        The final prompt combines:
          1. evaluate_context (system instruction with pattern hint)
          2. evaluate_template (question + answer + optional knowledge)

        Args:
            question: The Q&A question text.
            answer: The answer to evaluate (correct or hallucinated).
            mode: "without_knowledge" or "with_knowledge".
            model_type: "open_source" or "close_source".
            pattern: Hallucination pattern index (0-3), or -1 for correct answers.
            knowledge: TTHC context text (required when mode="with_knowledge").
        """
        if mode == "with_knowledge" and not knowledge:
            raise ValueError(
                "knowledge must be provided when mode='with_knowledge'."
            )

        system_tmpl, data_tmpl = self._get_template(model_type)
        pattern_instruction = (
            PATTERN_INSTRUCTIONS.get(pattern, "") if pattern >= 0 else ""
        )

        system_part = system_tmpl.format(pattern=pattern_instruction)
        data_part = data_tmpl.format(
            knowledge=knowledge,
            question=question,
            answer=answer,
        )
        text = system_part + "\n\n" + data_part
        return Prompt(
            text=text,
            model_type=model_type,
            mode=mode,
            pattern=pattern,
        )

    def build_from_test_case(
        self,
        answer: str,
        question: str = "",
        mode: EvaluationMode = "without_knowledge",
        model_type: ModelType = "close_source",
        pattern: int = -1,
        knowledge: str = "",
    ) -> Prompt:
        """Convenience wrapper accepting the same signature as TestCase fields."""
        return self.build(
            question=question,
            answer=answer,
            mode=mode,
            model_type=model_type,
            pattern=pattern,
            knowledge=knowledge,
        )
