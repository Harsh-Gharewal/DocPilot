"""Unit tests for multi-format document loader."""

from pathlib import Path
from unittest.mock import patch
import pytest

from app.ingestion.loaders import (
    compute_file_hash,
    load_document,
    load_documents_from_directory,
    parse_html,
    parse_markdown,
    parse_plaintext,
)


def test_compute_file_hash():
    """Verify deterministic hash generation."""
    h1 = compute_file_hash(b"hello world")
    h2 = compute_file_hash(b"hello world")
    h3 = compute_file_hash(b"different content")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16


def test_parse_markdown():
    """Verify markdown section extraction and clean text."""
    md_content = """# Architecture Overview
This is the system architecture.

## Database Design
We use PostgreSQL and ChromaDB for storage.

## API Layer
FastAPI handles async HTTP endpoints.
"""
    full_text, sections = parse_markdown(md_content)
    assert len(sections) == 3
    assert sections[0].section_heading == "Architecture Overview"
    assert "This is the system architecture." in sections[0].text
    assert sections[0].page_number is None

    assert sections[1].section_heading == "Database Design"
    assert "ChromaDB" in sections[1].text

    assert sections[2].section_heading == "API Layer"
    assert "FastAPI" in sections[2].text


def test_parse_html():
    """Verify HTML cleaning strips script, style, nav, and extracts heading sections."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DocPilot Docs</title>
        <style>body { color: red; }</style>
        <script>alert('malicious');</script>
    </head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <header>Header info</header>
        <h1>Authentication Service</h1>
        <p>The authentication service issues JWT tokens.</p>
        <h2>Rate Limiting</h2>
        <p>Limits are 100 requests per minute per IP.</p>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """
    full_text, sections = parse_html(html_content)

    assert "alert('malicious')" not in full_text
    assert "Home" not in full_text
    assert "Copyright" not in full_text

    headings = [s.section_heading for s in sections]
    assert "Authentication Service" in headings
    assert "Rate Limiting" in headings

    for s in sections:
        assert s.page_number is None


def test_parse_plaintext():
    """Verify plain text loading into sections."""
    txt_content = "Simple plaintext content with multiple paragraphs.\n\nSecond paragraph."
    full_text, sections = parse_plaintext(txt_content, default_heading="Notes")
    assert len(sections) == 1
    assert sections[0].section_heading == "Notes"
    assert "Second paragraph" in sections[0].text
    assert sections[0].page_number is None


def test_load_document_unsupported_and_missing(tmp_path: Path):
    """Verify unsupported file types and missing files log warnings and return None."""
    missing_file = tmp_path / "non_existent.md"
    assert load_document(missing_file) is None

    unsupported_file = tmp_path / "archive.zip"
    unsupported_file.write_bytes(b"PK000fakezip")
    assert load_document(unsupported_file) is None


def test_load_document_saves_raw_and_processed(tmp_path: Path, monkeypatch):
    """Verify loaded documents persist raw file and processed JSON."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    from app.config import settings

    monkeypatch.setattr(settings, "raw_data_dir", raw_dir)
    monkeypatch.setattr(settings, "processed_data_dir", processed_dir)

    doc_file = tmp_path / "guide.md"
    doc_file.write_text("# Setup Guide\nRun `pip install` to get started.", encoding="utf-8")

    doc = load_document(doc_file)
    assert doc is not None
    assert doc.doc_id is not None
    assert doc.source_file == str(doc_file.as_posix())

    # Raw file should exist in raw_dir
    raw_files = list(raw_dir.glob(f"*{doc_file.name}"))
    assert len(raw_files) == 1

    # Processed JSON should exist in processed_dir
    processed_file = processed_dir / f"{doc.doc_id}.json"
    assert processed_file.exists()


def test_load_documents_from_directory(tmp_path: Path, monkeypatch):
    """Verify batch loading from a directory skips unsupported files and loads valid ones."""
    monkeypatch.setattr("app.config.settings.raw_data_dir", tmp_path / "raw")
    monkeypatch.setattr("app.config.settings.processed_data_dir", tmp_path / "proc")

    (tmp_path / "doc1.md").write_text("# Intro\nDoc 1 content", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Doc 2 plain text content", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01\x02")

    docs = load_documents_from_directory(tmp_path)
    assert len(docs) == 2
    names = [Path(d.source_file).name for d in docs]
    assert "doc1.md" in names
    assert "doc2.txt" in names

