"""Dataset loader and strict validator for the golden evaluation dataset."""

import json
import logging
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"lookup", "multi_hop", "no_answer", "ambiguous"}


class GoldenTestCase(BaseModel):
    """Pydantic model representing a single evaluation test case."""

    id: str = Field(..., description="Unique identifier for the test case")
    category: Literal["lookup", "multi_hop", "no_answer", "ambiguous"] = Field(
        ..., description="Evaluation category"
    )
    question: str = Field(..., description="Evaluation question")
    golden_answer: str = Field(..., description="Expected ground-truth answer or 'NO_ANSWER'")
    source_docs: list[str] = Field(
        default_factory=list, description="List of expected document file basenames"
    )
    source_sections: list[str] = Field(
        default_factory=list, description="List of expected section headings"
    )


def get_available_corpus_docs() -> set[str]:
    """Inspect sample_docs and data/raw to find all indexed document basenames."""
    known_docs: set[str] = set()

    # Search in sample_docs_dir
    if settings.sample_docs_dir.exists():
        for f in settings.sample_docs_dir.rglob("*"):
            if f.is_file():
                known_docs.add(f.name.lower())

    # Search in raw_data_dir
    if settings.raw_data_dir.exists():
        for f in settings.raw_data_dir.rglob("*"):
            if f.is_file():
                # Raw files may be prefixed with hash: {hash}_{filename}
                cleaned_name = f.name
                if "_" in f.name:
                    parts = f.name.split("_", 1)
                    if len(parts[0]) == 16:
                        cleaned_name = parts[1]
                known_docs.add(cleaned_name.lower())

    return known_docs


def load_golden_dataset(file_path: Path | str = "eval/golden_dataset.jsonl") -> list[GoldenTestCase]:
    """Load and validate the golden evaluation dataset.

    Fails loudly on:
    - Malformed JSON lines
    - Pydantic schema validation failures
    - Duplicate test case IDs
    - Reference to source_docs that do not exist in the indexed corpus
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset file not found at: {path}")

    cases: list[GoldenTestCase] = []
    seen_ids: set[str] = set()
    category_counts: dict[str, int] = {c: 0 for c in VALID_CATEGORIES}
    corpus_docs = get_available_corpus_docs()

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, start=1):
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            continue

        try:
            data = json.loads(clean_line)
        except json.JSONDecodeError as jde:
            raise ValueError(
                f"Malformed JSON on line {line_idx} of {path.name}: {jde}\nLine content: {clean_line}"
            ) from jde

        try:
            case = GoldenTestCase(**data)
        except ValidationError as ve:
            raise ValueError(
                f"Validation error on line {line_idx} ({data.get('id', 'unknown')}): {ve}"
            ) from ve

        # Check duplicate IDs
        if case.id in seen_ids:
            raise ValueError(
                f"Duplicate test case ID '{case.id}' detected on line {line_idx} of {path.name}!"
            )
        seen_ids.add(case.id)

        # Check source_docs existence in corpus (skip no_answer which legitimately has empty source_docs)
        if case.category != "no_answer":
            for doc in case.source_docs:
                base_name = Path(doc).name.lower()
                if base_name not in corpus_docs:
                    raise ValueError(
                        f"Line {line_idx} (id={case.id}) references source_doc '{doc}' "
                        f"which does not exist in the indexed corpus! Available corpus files: {sorted(list(corpus_docs))}"
                    )

        # Validate NO_ANSWER consistency
        if case.category == "no_answer" and case.golden_answer != "NO_ANSWER":
            raise ValueError(
                f"Line {line_idx} (id={case.id}) has category 'no_answer' but golden_answer is not 'NO_ANSWER'!"
            )

        category_counts[case.category] += 1
        cases.append(case)

    logger.info(
        f"Loaded {len(cases)} test cases from {path.name}: "
        + ", ".join(f"{cat}={count}" for cat, count in category_counts.items())
    )
    return cases

