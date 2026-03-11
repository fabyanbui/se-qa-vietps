"""
dataset_loader.py — Load and sample test cases from VietPS-Hallu dataset.

Primary data source: E_Analyze/Final_Data/ (the final, enriched dataset).

The Q&A dataset (postgenerate_gpt.csv) contains pairs where each row has:
  - A correct answer (cauTraLoi)
  - A hallucinated answer (cauTraLoiAoGiac)
  - A question (cauHoi)
  - A ministry (boNganh)
  - A pattern label (0-3)
  - A list of related TTHC procedure URLs (TTHCLienQuan)

The TTHC dataset (postprocessed_tthc.csv) contains 1,820 administrative
procedure entries that provide "knowledge" context for with_knowledge
evaluations. Each Q&A row links to one or more TTHC entries via TTHCLienQuan.

Each Q&A row yields TWO HalluTestCase objects: one correct, one hallucinated.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# ── Schema definitions (pandera) ──────────────────────────────────────────────

class PostGenerateSchema(pa.DataFrameModel):
    """Schema for E_Analyze/Final_Data/postgenerate_gpt.csv"""

    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    boNganh: Series[str] = pa.Field(nullable=True)
    cauHoi: Series[str] = pa.Field(nullable=False)
    cauTraLoi: Series[str] = pa.Field(nullable=False)
    cauTraLoiAoGiac: Series[str] = pa.Field(nullable=False)
    TTHCLienQuan: Series[str] = pa.Field(nullable=True)
    pattern: Series[int] = pa.Field(ge=0, le=3)

    class Config:
        coerce = True
        strict = "filter"


class TTHCSchema(pa.DataFrameModel):
    """Schema for E_Analyze/Final_Data/postprocessed_tthc.csv"""

    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    tenThuTuc: Series[str] = pa.Field(nullable=True)
    linhVuc: Series[str] = pa.Field(nullable=True)
    trinhTuThucHien: Series[str] = pa.Field(nullable=True)
    canCuPhapLy: Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = "filter"


class AnnotatedSchema(pa.DataFrameModel):
    """Schema for CH_Annotate/annotated_data/human*.csv"""

    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    phanLoai: Series[str] = pa.Field(nullable=True)
    boNganh: Series[str] = pa.Field(nullable=True)
    cauHoi: Series[str] = pa.Field(nullable=False)
    cauTraLoi: Series[str] = pa.Field(nullable=False)
    cauTraLoiAoGiac: Series[str] = pa.Field(nullable=False)
    pattern: Series[float] = pa.Field(ge=0, le=3)

    class Config:
        coerce = True
        strict = "filter"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HalluTestCase:
    """A single evaluation test case derived from the dataset."""

    id: str                  # Unique identifier (link + is_hallucinated)
    link: str                # Source URL
    question: str            # The Q&A question (may be empty in postgenerate CSV)
    answer: str              # The answer to evaluate
    is_hallucinated: bool    # Ground truth label
    pattern: int             # Hallucination pattern (0-3); -1 for correct answers
    ministry: str            # Originating ministry (boNganh)


# ── Loader ────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads VietPS-Hallu test cases from the submodule CSV files.

    Primary source: E_Analyze/Final_Data/ (the final, enriched dataset).
    All 3,717 Q&A rows are linked to TTHC procedures via TTHCLienQuan,
    enabling with_knowledge evaluations via load_knowledge_map().

    Usage::

        loader = DatasetLoader("Public-Sector-Application")
        cases = loader.load_primary(sample_size=200, pattern=1)
        knowledge = loader.load_knowledge_map()
        # pass knowledge to Evaluator.run(..., knowledge_map=knowledge)
    """

    # Paths relative to submodule root
    _PRIMARY_CSV = Path("E_Analyze") / "Final_Data" / "postgenerate_gpt.csv"
    _TTHC_CSV = Path("E_Analyze") / "Final_Data" / "postprocessed_tthc.csv"
    _GOLD_CSV_TEMPLATE = Path("CH_Annotate") / "annotated_data" / "human{n}.csv"

    def __init__(self, submodule_path: str = "Public-Sector-Application") -> None:
        self.base = Path(submodule_path)
        self._validate_submodule()

    def _validate_submodule(self) -> None:
        if not self.base.exists():
            raise FileNotFoundError(
                f"Submodule not found at '{self.base}'. "
                "Run: git submodule update --init"
            )

    # ── Primary dataset (3,717 final Q&A pairs) ──────────────────────────────

    def load_primary(
        self,
        sample_size: Optional[int] = None,
        pattern: Optional[int] = None,
        ministry: Optional[str] = None,
        seed: int = 42,
    ) -> list[HalluTestCase]:
        """
        Load from E_Analyze/Final_Data/postgenerate_gpt.csv.

        Each row yields two HalluTestCase objects:
          - ``::correct``      — correct answer (is_hallucinated=False)
          - ``::hallucinated`` — hallucinated answer (is_hallucinated=True)

        Args:
            sample_size: Maximum number of rows to sample. None = all rows.
            pattern: Filter to a specific hallucination pattern (0-3).
            ministry: Filter by ministry name substring (boNganh).
            seed: Random seed for reproducibility.
        """
        csv_path = self.base / self._PRIMARY_CSV
        df = self._read_csv(csv_path)
        df = PostGenerateSchema.validate(df)

        if pattern is not None:
            df = df[df["pattern"] == pattern]
        if ministry is not None:
            df = df[df["boNganh"].str.contains(ministry, na=False)]
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=seed)

        cases: list[HalluTestCase] = []
        for _, row in df.iterrows():
            question = str(row.get("cauHoi", ""))
            min_name = str(row.get("boNganh", ""))
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::correct",
                    link=row["link"],
                    question=question,
                    answer=row["cauTraLoi"],
                    is_hallucinated=False,
                    pattern=-1,
                    ministry=min_name,
                )
            )
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::hallucinated",
                    link=row["link"],
                    question=question,
                    answer=row["cauTraLoiAoGiac"],
                    is_hallucinated=True,
                    pattern=int(row["pattern"]),
                    ministry=min_name,
                )
            )
        return cases

    # ── TTHC knowledge map (for with_knowledge evaluations) ───────────────────

    def load_knowledge_map(
        self,
        max_chars: int = 1500,
    ) -> dict[str, str]:
        """
        Build a mapping from Q&A link → TTHC knowledge text.

        Each Q&A row in the primary dataset has a ``TTHCLienQuan`` column
        containing a list of related TTHC procedure URLs. This method resolves
        those URLs against postprocessed_tthc.csv and concatenates the
        procedure name + steps + legal basis as a single knowledge string.

        The result can be passed directly to ``Evaluator.run()`` as
        ``knowledge_map`` when using ``mode="with_knowledge"``.

        Args:
            max_chars: Truncate each knowledge string to this many characters
                       to avoid very long prompts.

        Returns:
            Dict mapping Q&A ``link`` → combined TTHC knowledge text.
        """
        qa_path = self.base / self._PRIMARY_CSV
        tthc_path = self.base / self._TTHC_CSV

        qa_df = self._read_csv(qa_path)
        tthc_df = self._read_csv(tthc_path)
        tthc_index = tthc_df.set_index("link")

        knowledge_map: dict[str, str] = {}

        for _, row in qa_df.iterrows():
            qa_link = row["link"]
            raw = row.get("TTHCLienQuan", "[]")
            try:
                tthc_links: list[str] = ast.literal_eval(str(raw)) if raw else []
            except (ValueError, SyntaxError):
                tthc_links = []

            parts: list[str] = []
            for tthc_link in tthc_links:
                if tthc_link not in tthc_index.index:
                    continue
                tthc_row = tthc_index.loc[tthc_link]
                # Compose knowledge from the most informative fields
                name = str(tthc_row.get("tenThuTuc", "") or "").strip()
                steps = str(tthc_row.get("trinhTuThucHien", "") or "").strip()
                legal = str(tthc_row.get("canCuPhapLy", "") or "").strip()
                field = str(tthc_row.get("linhVuc", "") or "").strip()
                section = []
                if name:
                    section.append(f"Thủ tục: {name}")
                if field:
                    section.append(f"Lĩnh vực: {field}")
                if steps:
                    section.append(f"Trình tự thực hiện: {steps[:600]}")
                if legal:
                    section.append(f"Căn cứ pháp lý: {legal[:300]}")
                if section:
                    parts.append("\n".join(section))

            combined = "\n\n---\n\n".join(parts)
            knowledge_map[qa_link] = combined[:max_chars] if combined else ""

        return knowledge_map

    # ── Gold standard dataset (1000 human-annotated samples) ─────────────────

    def load_gold(
        self,
        annotator: int = 1,
        sample_size: Optional[int] = None,
        pattern: Optional[int] = None,
        ministry: Optional[str] = None,
        seed: int = 42,
    ) -> list[HalluTestCase]:
        """
        Load from CH_Annotate/annotated_data/human{annotator}.csv.

        Args:
            annotator: 1 or 2 (selects human1.csv or human2.csv).
            sample_size: Max rows to sample.
            pattern: Filter by pattern (0-3).
            ministry: Filter by ministry name.
            seed: Random seed.
        """
        if annotator not in (1, 2):
            raise ValueError("annotator must be 1 or 2")

        csv_path = (
            self.base / "CH_Annotate" / "annotated_data" / f"human{annotator}.csv"
        )
        df = self._read_csv(csv_path)
        df = AnnotatedSchema.validate(df)

        if pattern is not None:
            df = df[df["pattern"] == float(pattern)]
        if ministry is not None:
            df = df[df["boNganh"].str.contains(ministry, na=False)]
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=seed)

        cases: list[HalluTestCase] = []
        for _, row in df.iterrows():
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::hallucinated",
                    link=row["link"],
                    question=str(row.get("cauHoi", "")),
                    answer=row["cauTraLoiAoGiac"],
                    is_hallucinated=True,
                    pattern=int(row["pattern"]),
                    ministry=str(row.get("boNganh", "")),
                )
            )
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::correct",
                    link=row["link"],
                    question=str(row.get("cauHoi", "")),
                    answer=row["cauTraLoi"],
                    is_hallucinated=False,
                    pattern=-1,
                    ministry=str(row.get("boNganh", "")),
                )
            )
        return cases

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path, encoding="utf-8-sig")

    def available_ministries(self) -> list[str]:
        """Return sorted list of unique ministry names from the primary dataset."""
        csv_path = self.base / self._PRIMARY_CSV
        df = self._read_csv(csv_path)
        return sorted(df["boNganh"].dropna().unique().tolist())
