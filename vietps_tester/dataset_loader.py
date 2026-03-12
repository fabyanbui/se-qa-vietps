"""
dataset_loader.py — Load and sample test cases from VietPS-Hallu dataset.

Single data source: E_Analyze/Final_Data/ (the final, enriched dataset).

Files used:
  postgenerate_gpt.csv   — 3,717 Q&A pairs (question + correct + hallucinated answer)
  postprocessed_tthc.csv — 1,820 TTHC administrative procedure entries

Each Q&A row links to one or more TTHC entries via TTHCLienQuan.
For with_knowledge evaluations, load_knowledge_map() resolves those links
and formats ALL 20 TTHC fields as structured knowledge text.

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
    """Schema for E_Analyze/Final_Data/postprocessed_tthc.csv — all 21 columns."""

    link: Series[str] = pa.Field(str_startswith="https://dichvucong.gov.vn")
    maThuTuc: Series[str] = pa.Field(nullable=True)
    soQuyetDinh: Series[str] = pa.Field(nullable=True)
    tenThuTuc: Series[str] = pa.Field(nullable=True)
    capThucHien: Series[str] = pa.Field(nullable=True)
    loaiThuTuc: Series[str] = pa.Field(nullable=True)
    linhVuc: Series[str] = pa.Field(nullable=True)
    trinhTuThucHien: Series[str] = pa.Field(nullable=True)
    cachThucThucHien: Series[str] = pa.Field(nullable=True)
    thanhPhanHoSo: Series[str] = pa.Field(nullable=True)
    doiTuongThucHien: Series[str] = pa.Field(nullable=True)
    coQuanThucHien: Series[str] = pa.Field(nullable=True)
    coQuanCoThamQuyen: Series[str] = pa.Field(nullable=True)
    diaChiTiepNhanHoSo: Series[str] = pa.Field(nullable=True)
    coQuanDuocUyQuyen: Series[str] = pa.Field(nullable=True)
    coQuanPhoiHop: Series[str] = pa.Field(nullable=True)
    ketQuaThucHien: Series[str] = pa.Field(nullable=True)
    canCuPhapLy: Series[str] = pa.Field(nullable=True)
    yeuCauDieuKienThucHien: Series[str] = pa.Field(nullable=True)
    tuKhoa: Series[str] = pa.Field(nullable=True)
    moTa: Series[str] = pa.Field(nullable=True)

    class Config:
        coerce = True
        strict = "filter"


# Human-readable Vietnamese labels for all TTHC columns (used in knowledge text)
TTHC_FIELD_LABELS: dict[str, str] = {
    "maThuTuc":               "Mã thủ tục",
    "soQuyetDinh":            "Số quyết định",
    "tenThuTuc":              "Tên thủ tục",
    "capThucHien":            "Cấp thực hiện",
    "loaiThuTuc":             "Loại thủ tục",
    "linhVuc":                "Lĩnh vực",
    "trinhTuThucHien":        "Trình tự thực hiện",
    "cachThucThucHien":       "Cách thức thực hiện",
    "thanhPhanHoSo":          "Thành phần hồ sơ",
    "doiTuongThucHien":       "Đối tượng thực hiện",
    "coQuanThucHien":         "Cơ quan thực hiện",
    "coQuanCoThamQuyen":      "Cơ quan có thẩm quyền",
    "diaChiTiepNhanHoSo":     "Địa chỉ tiếp nhận hồ sơ",
    "coQuanDuocUyQuyen":      "Cơ quan được ủy quyền",
    "coQuanPhoiHop":          "Cơ quan phối hợp",
    "ketQuaThucHien":         "Kết quả thực hiện",
    "canCuPhapLy":            "Căn cứ pháp lý",
    "yeuCauDieuKienThucHien": "Yêu cầu/Điều kiện thực hiện",
    "tuKhoa":                 "Từ khóa",
    "moTa":                   "Mô tả",
}

# Placeholder values that carry no real information — treated as empty
_EMPTY_PLACEHOLDERS = {"không có thông tin", "nan", "none", ".", ""}


def _is_empty(value: str) -> bool:
    return value.strip().lower() in _EMPTY_PLACEHOLDERS


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HalluTestCase:
    """A single evaluation test case derived from the dataset."""

    id: str                  # Unique identifier (link::correct or link::hallucinated)
    link: str                # Source Q&A URL on dichvucong.gov.vn
    question: str            # The administrative question (cauHoi)
    answer: str              # The answer to evaluate
    is_hallucinated: bool    # Ground truth label
    pattern: int             # Hallucination pattern (0-3); -1 for correct answers
    ministry: str            # Originating ministry/department (boNganh)


# ── Loader ────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads VietPS-Hallu test cases from E_Analyze/Final_Data/.

    Two files are used:
      - postgenerate_gpt.csv   → load_primary()
      - postprocessed_tthc.csv → load_knowledge_map()

    Usage::

        loader = DatasetLoader("Public-Sector-Application")

        # Load test cases (no_knowledge or with_knowledge)
        cases = loader.load_primary(sample_size=200, pattern=1)

        # Build knowledge map for with_knowledge evaluations
        knowledge = loader.load_knowledge_map()
        runs = evaluator.run(cases, adapters,
                             mode="with_knowledge",
                             knowledge_map=knowledge)
    """

    _PRIMARY_CSV = Path("E_Analyze") / "Final_Data" / "postgenerate_gpt.csv"
    _TTHC_CSV    = Path("E_Analyze") / "Final_Data" / "postprocessed_tthc.csv"

    def __init__(self, submodule_path: str = "Public-Sector-Application") -> None:
        self.base = Path(submodule_path)
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
        Load Q&A test cases from E_Analyze/Final_Data/postgenerate_gpt.csv.

        Each CSV row yields two HalluTestCase objects:
          - ``<link>::correct``      — correct answer  (is_hallucinated=False, pattern=-1)
          - ``<link>::hallucinated`` — hallucinated answer (is_hallucinated=True)

        Args:
            sample_size: Max rows to sample before doubling. None = all 3,717 rows.
            pattern:     Filter to hallucination pattern 0-3. None = all patterns.
            ministry:    Substring filter on boNganh. None = all ministries.
            seed:        Random seed for reproducible sampling.
        """
        df = self._read_and_validate(self._PRIMARY_CSV, PostGenerateSchema)

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
            cases.append(HalluTestCase(
                id=f"{row['link']}::correct",
                link=row["link"],
                question=question,
                answer=row["cauTraLoi"],
                is_hallucinated=False,
                pattern=-1,
                ministry=min_name,
            ))
            cases.append(HalluTestCase(
                id=f"{row['link']}::hallucinated",
                link=row["link"],
                question=question,
                answer=row["cauTraLoiAoGiac"],
                is_hallucinated=True,
                pattern=int(row["pattern"]),
                ministry=min_name,
            ))
        return cases

    # ── TTHC knowledge map (for with_knowledge evaluations) ───────────────────

    def load_knowledge_map(self) -> dict[str, str]:
        """
        Build a mapping from Q&A link → full TTHC knowledge text.

        For each Q&A row, the ``TTHCLienQuan`` column contains a list of
        related TTHC procedure URLs. This method resolves every URL against
        postprocessed_tthc.csv and formats ALL 20 non-link columns of each
        matched TTHC entry as labelled key-value text. Empty or placeholder
        values ("Không có thông tin", NaN) are omitted.

        Multiple linked TTHC entries are separated by "---".

        Returns:
            Dict mapping each Q&A ``link`` → combined TTHC knowledge string.
            Value is ``""`` for rows with no resolvable TTHC links.
        """
        qa_df = self._read_and_validate(self._PRIMARY_CSV, PostGenerateSchema)
        tthc_df = self._read_and_validate(self._TTHC_CSV, TTHCSchema)
        tthc_index = tthc_df.set_index("link")

        knowledge_map: dict[str, str] = {}

        for _, row in qa_df.iterrows():
            qa_link = row["link"]
            raw = row.get("TTHCLienQuan", "[]")
            try:
                tthc_links: list[str] = ast.literal_eval(str(raw)) if raw else []
            except (ValueError, SyntaxError):
                tthc_links = []

            entry_blocks: list[str] = []
            for tthc_link in tthc_links:
                if tthc_link not in tthc_index.index:
                    continue
                tthc_row = tthc_index.loc[tthc_link]
                # Handle duplicate index entries (take first if multiple)
                if isinstance(tthc_row, pd.DataFrame):
                    tthc_row = tthc_row.iloc[0]

                lines: list[str] = []
                for col, label in TTHC_FIELD_LABELS.items():
                    value = str(tthc_row.get(col, "") or "").strip()
                    if not _is_empty(value):
                        lines.append(f"{label}: {value}")

                if lines:
                    entry_blocks.append("\n".join(lines))

            knowledge_map[qa_link] = "\n\n---\n\n".join(entry_blocks)

        return knowledge_map

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_and_validate(self, relative_path: Path, schema) -> pd.DataFrame:
        path = self.base / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        return schema.validate(df)

    def available_ministries(self) -> list[str]:
        """Return sorted list of unique ministry names from the primary dataset."""
        df = self._read_and_validate(self._PRIMARY_CSV, PostGenerateSchema)
        return sorted(df["boNganh"].dropna().unique().tolist())

