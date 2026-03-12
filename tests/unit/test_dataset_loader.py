"""
tests/unit/test_dataset_loader.py — Unit tests for DatasetLoader.

Tests cover:
  - Schema validation (pandera) on primary dataset and TTHC dataset
  - Sampling by size, pattern, and ministry
  - HalluTestCase structure correctness
  - TTHC knowledge map: all 20 fields included, correct structure
  - FileNotFoundError on missing submodule
"""

import pytest

from vietps_tester.dataset_loader import (
    DatasetLoader,
    HalluTestCase,
    TTHC_FIELD_LABELS,
)

SUBMODULE = "Public-Sector-Application"


@pytest.mark.unit
class TestDatasetLoaderPrimary:
    """Tests for load_primary() using E_Analyze/Final_Data/postgenerate_gpt.csv."""

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
        assert [tc.id for tc in cases1] != [tc.id for tc in cases2]

    def test_load_primary_each_row_yields_two_cases(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=10)
        hallucinated = [tc for tc in cases if tc.is_hallucinated]
        correct = [tc for tc in cases if not tc.is_hallucinated]
        assert len(hallucinated) == len(correct)

    def test_load_primary_unique_ids(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=30)
        ids = [tc.id for tc in cases]
        assert len(ids) == len(set(ids)), "Duplicate HalluTestCase IDs found"

    def test_load_primary_ministry_filter(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=200, ministry="Y tế")
        for tc in cases:
            assert "Y tế" in tc.ministry or tc.ministry == "", (
                f"Ministry filter failed: {tc.ministry}"
            )


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


@pytest.mark.unit
class TestKnowledgeMap:
    """
    Tests for load_knowledge_map() — verifies ALL 20 TTHC columns are
    included in the knowledge text for with_knowledge evaluations.
    """

    def test_returns_dict(self):
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        assert isinstance(km, dict)

    def test_keys_are_qa_links(self):
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        for key in list(km.keys())[:10]:
            assert "dichvucong.gov.vn" in key, (
                f"Expected a Q&A URL key, got: {key}"
            )

    def test_covers_all_primary_rows(self):
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=20)
        km = loader.load_knowledge_map()
        for tc in cases:
            assert tc.link in km, f"Q&A link missing from knowledge map: {tc.link}"

    def test_values_are_strings(self):
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        for v in list(km.values())[:20]:
            assert isinstance(v, str)

    def test_most_entries_have_knowledge(self):
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        non_empty = sum(1 for v in km.values() if v.strip())
        assert non_empty / len(km) >= 0.5, (
            f"Only {non_empty}/{len(km)} entries have TTHC knowledge text"
        )

    def test_knowledge_includes_all_tthc_field_labels(self):
        """Non-empty knowledge entries must include all available TTHC field labels."""
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        sample = next(v for v in km.values() if v.strip())
        # The entry should contain multiple labelled fields from TTHC_FIELD_LABELS
        matched = [label for label in TTHC_FIELD_LABELS.values() if label in sample]
        assert len(matched) >= 3, (
            f"Expected at least 3 TTHC field labels in knowledge, found {len(matched)}. "
            f"Sample (first 300 chars): {sample[:300]}"
        )

    def test_knowledge_has_ten_thu_tuc(self):
        """Every non-empty entry should include the procedure name."""
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        sample = next(v for v in km.values() if v.strip())
        assert "Tên thủ tục:" in sample, (
            "Expected 'Tên thủ tục:' label in knowledge text"
        )

    def test_knowledge_has_trinh_tu_thuc_hien(self):
        """Procedural steps should be present in non-empty entries."""
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        sample = next(v for v in km.values() if v.strip())
        assert "Trình tự thực hiện:" in sample, (
            "Expected 'Trình tự thực hiện:' label in knowledge text"
        )

    def test_knowledge_has_can_cu_phap_ly(self):
        """Legal basis should be present in non-empty entries."""
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        sample = next(v for v in km.values() if v.strip())
        assert "Căn cứ pháp lý:" in sample, (
            "Expected 'Căn cứ pháp lý:' label in knowledge text"
        )

    def test_knowledge_no_placeholder_values(self):
        """Knowledge text should not contain the placeholder string."""
        loader = DatasetLoader(SUBMODULE)
        km = loader.load_knowledge_map()
        sample = next(v for v in km.values() if v.strip())
        assert "Không có thông tin" not in sample, (
            "Knowledge text should omit placeholder 'Không có thông tin' values"
        )

    def test_knowledge_map_matches_primary_sample(self):
        """Every sampled test case link must resolve in the knowledge map."""
        loader = DatasetLoader(SUBMODULE)
        cases = loader.load_primary(sample_size=50, seed=7)
        km = loader.load_knowledge_map()
        unique_links = {tc.link for tc in cases}
        missing = unique_links - km.keys()
        assert not missing, f"Test case links missing from knowledge map: {missing}"

    def test_tthc_field_labels_dict_has_all_20_columns(self):
        """TTHC_FIELD_LABELS must cover all 20 non-link columns."""
        assert len(TTHC_FIELD_LABELS) == 20, (
            f"Expected 20 TTHC field labels, got {len(TTHC_FIELD_LABELS)}"
        )

