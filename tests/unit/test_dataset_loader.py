"""
tests/unit/test_dataset_loader.py — Unit tests for DatasetLoader.

Tests cover:
  - Schema validation (pandera) on both primary and gold datasets
  - Sampling by size and pattern
  - HalluTestCase structure correctness
  - FileNotFoundError on missing submodule
"""

import pytest

from vietps_tester.dataset_loader import DatasetLoader, HalluTestCase

SUBMODULE = "Public-Sector-Application"


@pytest.mark.unit
class TestDatasetLoaderPrimary:
    """Tests for load_primary() using C_Generate/postgenerate_gpt.csv."""

    def test_load_primary_returns_list(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=10)
        assert isinstance(cases, list)
        assert len(cases) == 20  # 10 rows × 2 (correct + hallucinated)

    def test_load_primary_all_are_test_case_instances(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=5)
        for tc in cases:
            assert isinstance(tc, HalluTestCase)

    def test_load_primary_has_valid_links(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=20)
        for tc in cases:
            assert tc.link.startswith("https://dichvucong.gov.vn"), (
                f"Invalid URL: {tc.link}"
            )

    def test_load_primary_pattern_filter(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=50, pattern=0)
        # Hallucinated cases should all have pattern=0
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        for tc in hallucinated:
            assert tc.pattern == 0, f"Expected pattern 0, got {tc.pattern}"

    @pytest.mark.parametrize("pattern", [0, 1, 2, 3])
    def test_load_primary_all_patterns_present(self, pattern):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=50, pattern=pattern)
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        assert len(hallucinated) > 0, f"No hallucinated cases for pattern {pattern}"

    def test_load_primary_reproducible_with_seed(self):
        loader = DatasetLoader(SUBMODULE)
        cases1 = loader.load_primary(sample_size=10, seed=42)
        cases2 = loader.load_primary(sample_size=10, seed=42)
        assert [tc.id for tc in cases1] == [tc.id for tc in cases2]

    def test_load_primary_different_seeds_differ(self):
        loader = DatasetLoader(SUBMODULE)
        cases1 = loader.load_primary(sample_size=20, seed=1)
        cases2 = loader.load_primary(sample_size=20, seed=99)
        # Very unlikely to be identical with different seeds
        assert [tc.id for tc in cases1] != [tc.id for tc in cases2]

    def test_load_primary_each_row_yields_two_cases(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=10)
        # Each row → 1 correct + 1 hallucinated
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        correct = [tc for tc in cases if not tc.is_hallucinated]
        assert len(hallucinated) == len(correct)

    def test_load_primary_unique_ids(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=30)
        ids = [tc.id for tc in cases]
        assert len(ids) == len(set(ids)), "Duplicate HalluTestCase IDs found"


@pytest.mark.unit
class TestDatasetLoaderGold:
    """Tests for load_gold() using CH_Annotate/annotated_data/human*.csv."""

    def test_load_gold_annotator1(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_gold(annotator=1, sample_size=10)
        assert len(cases) == 20  # 10 rows × 2

    def test_load_gold_annotator2(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_gold(annotator=2, sample_size=10)
        assert len(cases) == 20

    def test_load_gold_invalid_annotator(self):
        loader = DatasetLoader(SUBMODULE)
        with pytest.raises(ValueError, match="annotator must be 1 or 2"):
            loader.load_gold(annotator=3)

    def test_load_gold_has_questions(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_gold(annotator=1, sample_size=10)
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        for tc in hallucinated:
            assert len(tc.question) > 0, "Gold set case has empty question"

    def test_load_gold_has_ministry(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_gold(annotator=1, sample_size=10)
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        for tc in hallucinated:
            assert len(tc.ministry) > 0, "Gold set case has empty ministry"

    def test_load_gold_pattern_range(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_gold(annotator=1, sample_size=50)
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        for tc in hallucinated:
            assert 0 <= tc.pattern <= 3, f"Pattern out of range: {tc.pattern}"


@pytest.mark.unit
class TestDatasetLoaderErrors:
    """Tests for error handling in DatasetLoader."""

    def test_missing_submodule_raises(self):
        with pytest.raises(FileNotFoundError, match="Submodule not found"):
            DatasetLoader("/nonexistent/path")

    def test_available_ministries_returns_list(self):
        loader = DatasetLoader(SUBMODULE)
        ministries = loader.available_ministries()
        assert isinstance(ministries, list)
        assert len(ministries) > 0
        assert all(isinstance(m, str) for m in ministries)
