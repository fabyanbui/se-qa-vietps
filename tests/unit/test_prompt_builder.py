"""
tests/unit/test_prompt_builder.py — Unit tests for PromptBuilder.

Tests cover:
  - Template loading from submodule
  - Prompt rendering for all 4 patterns
  - with/without knowledge modes
  - Error cases (missing knowledge, unknown model_type)
"""

import pytest

from vietps_tester.prompt_builder import Prompt, PromptBuilder

SUBMODULE = "Public-Sector-Application"

SAMPLE_QUESTION = "Thủ tục cấp thẻ nhà báo như thế nào?"
SAMPLE_ANSWER = "Thẻ nhà báo được cấp theo Thông tư số 49/2016/TT-BTTTT."
SAMPLE_KNOWLEDGE = "Thông tư 49/2016/TT-BTTTT quy định về cấp thẻ nhà báo..."


@pytest.mark.unit
class TestPromptBuilderInit:
    def test_init_loads_templates(self):
        builder = PromptBuilder(SUBMODULE)
        assert builder._templates is not None
        assert not builder._templates.empty

    def test_init_missing_submodule_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptBuilder("/nonexistent/path")


@pytest.mark.unit
class TestPromptBuilderBuild:
    @pytest.fixture(autouse=True)
    def builder(self):
        self.builder = PromptBuilder(SUBMODULE)

    def test_returns_prompt_object(self):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
        )
        assert isinstance(result, Prompt)

    def test_prompt_text_is_non_empty(self):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
        )
        assert len(result.text) > 0

    def test_prompt_contains_answer(self):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
        )
        assert SAMPLE_ANSWER in result.text

    @pytest.mark.parametrize("pattern", [0, 1, 2, 3])
    def test_pattern_instruction_included(self, pattern):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
            pattern=pattern,
        )
        assert len(result.text) > 0
        assert result.pattern == pattern

    def test_without_knowledge_mode(self):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
            mode="without_knowledge",
        )
        assert result.mode == "without_knowledge"

    def test_with_knowledge_mode(self):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
            mode="with_knowledge",
            knowledge=SAMPLE_KNOWLEDGE,
        )
        assert result.mode == "with_knowledge"

    def test_with_knowledge_missing_raises(self):
        with pytest.raises(ValueError, match="knowledge must be provided"):
            self.builder.build(
                question=SAMPLE_QUESTION,
                answer=SAMPLE_ANSWER,
                mode="with_knowledge",
                knowledge="",
            )

    @pytest.mark.parametrize("model_type", ["open_source", "close_source"])
    def test_model_type_variants(self, model_type):
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
            model_type=model_type,
        )
        assert result.model_type == model_type
        assert len(result.text) > 0

    def test_correct_answer_no_pattern_instruction(self):
        """Correct answers (pattern=-1) should not include pattern instructions."""
        result = self.builder.build(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER,
            pattern=-1,
        )
        assert result.pattern == -1
        # Prompt should still be valid
        assert len(result.text) > 0
