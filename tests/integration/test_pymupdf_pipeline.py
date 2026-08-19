"""Integration tests for the real PyMuPDF extraction and chunking pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pymupdf
import pytest

from domain import ExtractedEntry, PublicError
from ingest import (
    ChunkingService,
    PdfExtractor,
    WarningCollector,
    discover_pdf_entries,
    sha256_file,
)


def _create_pdf(path: Path, page_texts: Sequence[str | None]) -> None:
    """Create a minimal PDF; ``None`` represents a physically blank page."""

    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text is not None:
                page.insert_text((72, 72), text)
        document.save(path)


def _entry(path: Path, ordinal: int, display_name: str) -> ExtractedEntry:
    return ExtractedEntry(
        ordinal=ordinal,
        path=path,
        display_name=display_name,
        size=path.stat().st_size,
    )


def _direct_page_texts(path: Path) -> tuple[str, ...]:
    with pymupdf.open(path) as document:
        return tuple(
            document.load_page(index).get_text("text")
            for index in range(document.page_count)
        )


def _assert_clean_spool(spool_root: Path) -> None:
    assert spool_root.is_dir()
    assert tuple(spool_root.iterdir()) == ()


def test_real_pdfs_cover_textual_empty_and_mixed_pages_with_exact_spool_round_trip(
    tmp_path: Path,
) -> None:
    textual_path = tmp_path / "textual.pdf"
    empty_path = tmp_path / "empty.pdf"
    mixed_path = tmp_path / "mixed.pdf"
    _create_pdf(textual_path, ["Procedimento de cadastro no ERP."])
    _create_pdf(empty_path, [None, None])
    _create_pdf(mixed_path, ["Primeira etapa.", None, "Terceira etapa."])

    entries = (
        _entry(textual_path, 0, "textual.pdf"),
        _entry(empty_path, 1, "digitalizado.pdf"),
        _entry(mixed_path, 2, "misto.pdf"),
    )
    warnings = WarningCollector()
    spool_root = tmp_path / "spool"

    documents = PdfExtractor(warnings).extract_all(entries, spool_root)

    assert [document.display_name for document in documents] == [
        "textual.pdf",
        "digitalizado.pdf",
        "misto.pdf",
    ]
    assert [document.document_id for document in documents] == [
        sha256_file(textual_path),
        sha256_file(empty_path),
        sha256_file(mixed_path),
    ]
    expected_by_name = {
        "textual.pdf": _direct_page_texts(textual_path),
        "digitalizado.pdf": _direct_page_texts(empty_path),
        "misto.pdf": _direct_page_texts(mixed_path),
    }
    for document in documents:
        assert tuple(page.human_page for page in document.pages) == tuple(
            range(1, len(document.pages) + 1)
        )
        assert tuple(page.text for page in document.pages) == expected_by_name[
            document.display_name
        ]

    chunker = ChunkingService(chunk_size=16, chunk_overlap=4)
    chunks = [
        chunk
        for document in documents
        for chunk in chunker.split_document(document, transaction_id="tx-real-pdfs")
    ]

    assert len(documents) == 3
    assert sum(len(document.pages) for document in documents) == 6
    assert len(chunks) == sum(
        len(chunker.split_document(document, transaction_id="tx-real-pdfs"))
        for document in documents
    )
    assert not any(chunk.document_id == documents[1].document_id for chunk in chunks)
    assert {(chunk.display_name, chunk.human_page) for chunk in chunks} == {
        ("textual.pdf", 1),
        ("misto.pdf", 1),
        ("misto.pdf", 3),
    }
    assert warnings.warnings == (
        "digitalizado.pdf, página 1: nenhum caractere não branco foi extraído. "
        "O MVP não executa OCR; aplique OCR antes de uma nova importação se a "
        "página contiver texto em imagem.",
        "digitalizado.pdf, página 2: nenhum caractere não branco foi extraído. "
        "O MVP não executa OCR; aplique OCR antes de uma nova importação se a "
        "página contiver texto em imagem.",
        "misto.pdf, página 2: nenhum caractere não branco foi extraído. O MVP "
        "não executa OCR; aplique OCR antes de uma nova importação se a página "
        "contiver texto em imagem.",
    )
    _assert_clean_spool(spool_root)


def test_corrupt_pdf_is_skipped_beside_valid_pdf_with_sanitized_ordered_warnings(
    tmp_path: Path,
) -> None:
    ignored_path = tmp_path / "notes.txt"
    ignored_path.write_text("não PDF", encoding="utf-8")
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"not-a-pdf")
    valid_path = tmp_path / "valid.pdf"
    _create_pdf(valid_path, ["Campo Cliente", None])

    entries = (
        _entry(valid_path, 2, "manuais/cafe\u0301\u0001.PdF"),
        _entry(ignored_path, 0, "extras/no\u0002tas.txt"),
        _entry(corrupt_path, 1, "guias/corrompi\u0301do.PDF"),
    )
    warnings = WarningCollector()
    candidates = discover_pdf_entries(entries, warnings)
    spool_root = tmp_path / "spool"

    documents = PdfExtractor(warnings).extract_all(candidates, spool_root)

    assert [entry.ordinal for entry in candidates] == [1, 2]
    assert [entry.display_name for entry in candidates] == [
        "guias/corrompído.PDF",
        "manuais/café.PdF",
    ]
    assert len(documents) == 1
    assert documents[0].display_name == "manuais/café.PdF"
    assert documents[0].document_id == sha256_file(valid_path)
    chunks = ChunkingService(8, 2).split_document(documents[0], "tx-valid")
    assert chunks
    assert {chunk.human_page for chunk in chunks} == {1}

    expected_warnings = (
        "extras/notas.txt: arquivo ignorado porque não possui a extensão .pdf.",
        "guias/corrompído.PDF: arquivo ignorado por falha de leitura; "
        "substitua-o ou remova-o antes de reenviar.",
        "manuais/café.PdF, página 2: nenhum caractere não branco foi extraído. "
        "O MVP não executa OCR; aplique OCR antes de uma nova importação se a "
        "página contiver texto em imagem.",
    )
    assert warnings.warnings == expected_warnings
    assert warnings.add("pdf_read_failed", 1, None, "duplicado") is False
    assert warnings.add("pdf_page_empty", 2, 2, "duplicado") is False
    assert warnings.warnings == expected_warnings
    _assert_clean_spool(spool_root)


def test_all_unreadable_pdfs_raise_after_ordered_warnings_and_leave_no_spool(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"corrupt-one")
    second.write_bytes(b"corrupt-two")
    warnings = WarningCollector()
    spool_root = tmp_path / "spool"

    with pytest.raises(PublicError) as captured:
        PdfExtractor(warnings).extract_all(
            (
                _entry(first, 4, "primeiro.pdf"),
                _entry(second, 9, "subpasta/segundo.pdf"),
            ),
            spool_root,
        )

    assert captured.value.code == "no_readable_pdfs"
    assert captured.value.http_status == 422
    assert captured.value.message == (
        "Nenhum PDF do arquivo pôde ser lido. Substitua ou remova os arquivos "
        "com problema."
    )
    assert warnings.warnings == (
        "primeiro.pdf: arquivo ignorado por falha de leitura; substitua-o ou "
        "remova-o antes de reenviar.",
        "subpasta/segundo.pdf: arquivo ignorado por falha de leitura; "
        "substitua-o ou remova-o antes de reenviar.",
    )
    _assert_clean_spool(spool_root)
