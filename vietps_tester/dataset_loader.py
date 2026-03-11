"""
dataset_loader.py — Load and sample test cases from VietPS-Hallu dataset.

The dataset contains Q&A pairs where each row has:
  - A correct answer (cauTraLoi)
  - A hallucinated answer (cauTraLoiAoGiac)
  - A question (cauHoi)
  - A ministry (boNganh)
  - A pattern label (0-3)

Each row yields TWO HalluTestCase objects: one correct, one hallucinated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# ── Schema definitions (pandera) ──────────────────────────────────────────────

class PostGenerateSchema(pa.DataFrameModel):
    """Schema for C_Generate/postgenerate_gpt.csv"""

    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    boNganh: Series[str] = pa.Field(nullable=True)
    cauHoi: Series[str] = pa.Field(nullable=False)
    cauTraLoi: Series[str] = pa.Field(nullable=False)
    cauTraLoiAoGiac: Series[str] = pa.Field(nullable=False)
    pattern: Series[int] = pa.Field(ge=0, le=3)

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

    Usage::

        loader = DatasetLoader("Public-Sector-Application")
        cases = loader.load_primary(sample_size=200, pattern=1)
    """

    def __init__(self, submodule_path: str = "Public-Sector-Application") -> None:
        self.base = Path(submodule_path)
        self._validate_submodule()

    def _validate_submodule(self) -> None:
        if not self.base.exists():
            raise FileNotFoundError(
                f"Submodule not found at '{self.base}'. "
                "Run: git submodule update --init"
            )

    # ── Primary dataset (full ~12k generated samples) ────────────────────────

    def load_primary(
        self,
        sample_size: Optional[int] = None,
        pattern: Optional[int] = None,
        ministry: Optional[str] = None,
        seed: int = 42,
    ) -> list[HalluTestCase]:
        """
        Load from C_Generate/postgenerate_gpt.csv.

        Each row yields one hallucinated HalluTestCase (correct-answer rows are
        included as negative controls).

        Args:
            sample_size: Maximum number of rows to sample. None = all rows.
            pattern: Filter to a specific hallucination pattern (0-3).
            ministry: Filter by ministry name (boNganh).
            seed: Random seed for reproducibility.
        """
        csv_path = self.base / "C_Generate" / "postgenerate_gpt.csv"
        df = self._read_csv(csv_path)
        df = PostGenerateSchema.validate(df)

        if pattern is not None:
            df = df[df["pattern"] == float(pattern)]
        if sample_size is not None:
            df = df.sample(n=min(sample_size, len(df)), random_state=seed)

        cases: list[HalluTestCase] = []
        for _, row in df.iterrows():
            question = str(row.get("cauHoi", ""))
            ministry = str(row.get("boNganh", ""))
            # Correct answer (ground truth: not hallucinated)
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::correct",
                    link=row["link"],
                    question=question,
                    answer=row["cauTraLoi"],
                    is_hallucinated=False,
                    pattern=-1,
                    ministry=ministry,
                )
            )
            # Hallucinated answer (ground truth: is hallucinated)
            cases.append(
                HalluTestCase(
                    id=f"{row['link']}::hallucinated",
                    link=row["link"],
                    question=question,
                    answer=row["cauTraLoiAoGiac"],
                    is_hallucinated=True,
                    pattern=int(row["pattern"]),
                    ministry=ministry,
                )
            )
        return cases

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
        """Return sorted list of unique ministry names from the gold dataset."""
        csv_path = self.base / "CH_Annotate" / "annotated_data" / "human1.csv"
        df = self._read_csv(csv_path)
        return sorted(df["boNganh"].dropna().unique().tolist())
