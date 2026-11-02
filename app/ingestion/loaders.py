"""Multi-format document loader supporting Markdown, Plain Text, HTML, and PDF.

Extracts normalized plaintext, structural sections (with headings and page numbers),
and persists raw and processed representations to data directories.
"""

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import settings
from app.ingestion.models import Document, Section

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


def compute_file_hash(content_bytes: bytes) -> str:
    """Calculate a stable SHA-256 hash string for raw file bytes."""
    return hashlib.sha256(content_bytes).hexdigest()[:16]


def clean_whitespace(text: str) -> str:
    """Normalize irregular whitespace, preserving paragraph breaks."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_html(content: str) -> tuple[str, list[Section]]:
    """Parse HTML content, stripping scripts/styles/nav and extracting sections by heading."""
    soup = BeautifulSoup(content, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    sections: list[Section] = []
    current_heading: Optional[str] = None
    current_paragraphs: list[str] = []

    # Traverse elements to group text under headings
    for elem in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "pre", "blockquote"]):
        tag_name = elem.name.lower()
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            # Flush previous section if it has content
            if current_paragraphs:
                sec_text = clean_whitespace(" ".join(current_paragraphs))
                if sec_text:
                    sections.append(
                        Section(
                            section_heading=current_heading or "Overview",
                            page_number=None,
                            text=sec_text,
                        )
                    )
                current_paragraphs = []
            heading_text = elem.get_text(strip=True)
            if heading_text:
                current_heading = heading_text
        else:
            txt = elem.get_text(strip=True)
            if txt:
                current_paragraphs.append(txt)

    if current_paragraphs:
        sec_text = clean_whitespace(" ".join(current_paragraphs))
        if sec_text:
            sections.append(
                Section(
                    section_heading=current_heading or "Overview",
                    page_number=None,
                    text=sec_text,
                )
            )

    # If no heading-based sections were formed, create one default section
    if not sections:
        full_text = clean_whitespace(body.get_text(separator="\n", strip=True))
        if full_text:
            sections.append(Section(section_heading="Overview", page_number=None, text=full_text))

    full_doc_text = "\n\n".join(
        f"{s.section_heading}\n{s.text}" if s.section_heading else s.text for s in sections
    )
    return full_doc_text, sections


def parse_markdown(content: str) -> tuple[str, list[Section]]:
    """Parse Markdown content into structured sections based on markdown headers."""
    lines = content.splitlines()
    sections: list[Section] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = heading_pattern.match(line.strip())
        if match:
            # Flush previous section
            if current_lines:
                sec_text = clean_whitespace("\n".join(current_lines))
                if sec_text:
                    sections.append(
                        Section(
                            section_heading=current_heading or "Introduction",
                            page_number=None,
                            text=sec_text,
                        )
                    )
                current_lines = []
            current_heading = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sec_text = clean_whitespace("\n".join(current_lines))
        if sec_text:
            sections.append(
                Section(
                    section_heading=current_heading or "Introduction",
                    page_number=None,
                    text=sec_text,
                )
            )

    if not sections:
        full_text = clean_whitespace(content)
        if full_text:
            sections.append(Section(section_heading="Introduction", page_number=None, text=full_text))

    full_doc_text = "\n\n".join(
        f"{s.section_heading}\n{s.text}" if s.section_heading else s.text for s in sections
    )
    return full_doc_text, sections


def parse_pdf(file_path: Path) -> tuple[str, list[Section]]:
    """Parse a PDF document using pypdf, falling back to pdfplumber if needed."""
    sections: list[Section] = []

    try:
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = clean_whitespace(page_text)
            if not cleaned:
                continue

            # Try to infer section heading from first line
            first_line = cleaned.split("\n")[0][:100].strip()
            heading = first_line if first_line else f"Page {page_idx + 1}"

            sections.append(
                Section(
                    section_heading=heading,
                    page_number=page_idx + 1,
                    text=cleaned,
                )
            )
    except Exception as e:
        logger.warning(
            f"pypdf extraction failed on {file_path.name} ({e}), attempting pdfplumber fallback."
        )
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    extracted = page.extract_text() or ""
                    cleaned = clean_whitespace(extracted)
                    if not cleaned:
                        continue
                    first_line = cleaned.split("\n")[0][:100].strip()
                    heading = first_line if first_line else f"Page {page_idx + 1}"
                    sections.append(
                        Section(
                            section_heading=heading,
                            page_number=page_idx + 1,
                            text=cleaned,
                        )
                    )
        except Exception as fallback_err:
            logger.error(f"pdfplumber fallback also failed for {file_path}: {fallback_err}")
            raise fallback_err

    full_doc_text = "\n\n".join(s.text for s in sections)
    return full_doc_text, sections


def parse_plaintext(content: str, default_heading: str) -> tuple[str, list[Section]]:
    """Parse a plain text file into clean normalized text and sections."""
    cleaned = clean_whitespace(content)
    sections = [
        Section(
            section_heading=default_heading,
            page_number=None,
            text=cleaned,
        )
    ]
    return cleaned, sections


def load_document(file_path: Path | str) -> Optional[Document]:
    """Load and normalize a single document file into a Document instance.

    Persists original file copy to data/raw/ and parsed representation to data/processed/.
    Skips unreadable or unsupported files with a warning and returns None.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"File not found: {file_path}")
        return None

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Skipping unsupported file extension '{ext}': {path.name}")
        return None

    try:
        raw_bytes = path.read_bytes()
        doc_id = compute_file_hash(raw_bytes)

        if ext == ".pdf":
            full_text, sections = parse_pdf(path)
        elif ext in {".html", ".htm"}:
            full_text, sections = parse_html(raw_bytes.decode("utf-8", errors="replace"))
        elif ext in {".md", ".markdown"}:
            full_text, sections = parse_markdown(raw_bytes.decode("utf-8", errors="replace"))
        elif ext == ".txt":
            full_text, sections = parse_plaintext(
                raw_bytes.decode("utf-8", errors="replace"), default_heading=path.stem
            )
        else:
            logger.warning(f"Unsupported extension {ext} for {path.name}")
            return None

        if not full_text.strip():
            logger.warning(f"Document {path.name} produced empty extracted text. Skipping.")
            return None

        document = Document(
            doc_id=doc_id,
            source_file=str(path.as_posix()),
            text=full_text,
            sections=sections,
            metadata={
                "filename": path.name,
                "file_extension": ext,
                "file_size_bytes": len(raw_bytes),
                "section_count": len(sections),
            },
        )

        # Persist raw copy to data/raw/
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        raw_target = settings.raw_data_dir / f"{doc_id}_{path.name}"
        if not raw_target.exists():
            shutil.copy2(path, raw_target)

        # Persist processed JSON to data/processed/
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_target = settings.processed_data_dir / f"{doc_id}.json"
        processed_target.write_text(document.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"Successfully loaded and persisted document {path.name} (doc_id={doc_id})")
        return document

    except Exception as exc:
        logger.warning(f"Failed to read or parse file {path.name}: {exc}", exc_info=True)
        return None


def load_documents_from_directory(dir_path: Path | str) -> list[Document]:
    """Recursively search and load all supported documents from a directory.

    Unsupported or unreadable files are skipped with logged warnings.
    """
    path = Path(dir_path)
    if not path.is_dir():
        logger.warning(f"Provided path is not a valid directory: {dir_path}")
        return []

    documents: list[Document] = []
    # Sort files for deterministic processing
    all_files = sorted(path.rglob("*"))
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            doc = load_document(file_path)
            if doc is not None:
                documents.append(doc)

    logger.info(f"Loaded {len(documents)} document(s) from directory: {dir_path}")
    return documents

