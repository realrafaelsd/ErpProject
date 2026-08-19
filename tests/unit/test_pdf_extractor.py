"""Focused tests for page-by-page PDF extraction and faithful spooling."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from domain import ExtractedEntry, PublicError
from ingest import PdfExtractor, WarningCollector


def _create_pdf(path: Path, page_texts: list[str | None]) -> None:
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


def _texts_directly_from_pymupdf(path: Path) -> tuple[str, ...]:
    with pymupdf.open(path) as document:
        return tuple(
            document.load_page(index).get_text("text")
            for index in range(document.page_count)
        )


def test_extract_preserves_page_text_numbers_empty_page_and_cleans_spool(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    _create_pdf(pdf_path, ["Informação do ERP: ação concluída.", None])
    expected_texts = _texts_directly_from_pymupdf(pdf_path)
    warnings = WarningCollector()
    extractor = PdfExtractor(warnings)
    spool_root = tmp_path / "spool"

    document = extractor.extract(
        _entry(pdf_path, 4, "guias/manual.pdf"),
        spool_root,
        document_id="document-id",
    )

    assert document is not None
    assert document.document_id == "document-id"
    assert document.display_name == "guias/manual.pdf"
    assert tuple(page.human_page for page in document.pages) == (1, 2)
    assert tuple(page.text for page in document.pages) == expected_texts
    assert document.pages[1].text.strip() == ""
    assert warnings.warnings == (
        "guias/manual.pdf, página 2: nenhum caractere não branco foi "
        "extraído. O MVP não executa OCR; aplique OCR antes de uma nova "
        "importação se a página contiver texto em imagem.",
    )
    assert spool_root.is_dir()
    assert tuple(spool_root.iterdir()) == ()


def test_fully_empty_pdf_remains_readable_and_counts_all_pages(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "digitalizado.pdf"
    _create_pdf(pdf_path, [None, None])
    warnings = WarningCollector()
    extractor = PdfExtractor(warnings)

    documents = extractor.extract_all(
        [_entry(pdf_path, 0, "digitalizado.pdf")],
        tmp_path / "spool",
        document_ids={0: "empty-document"},
    )

    assert len(documents) == 1
    assert documents[0].document_id == "empty-document"
    assert len(documents[0].pages) == 2
    assert all(page.text.strip() == "" for page in documents[0].pages)
    assert len(warnings.warnings) == 2
    assert "página 1" in warnings.warnings[0]
    assert "página 2" in warnings.warnings[1]


def test_unreadable_pdf_is_discarded_while_readable_peer_continues(
    tmp_path: Path,
) -> None:
    corrupt_path = tmp_path / "corrompido.pdf"
    corrupt_path.write_bytes(b"isto nao e um PDF")
    valid_path = tmp_path / "valido.pdf"
    _create_pdf(valid_path, ["Conteúdo legível."])
    warnings = WarningCollector()
    extractor = PdfExtractor(warnings)
    spool_root = tmp_path / "spool"

    documents = extractor.extract_all(
        [
            _entry(corrupt_path, 0, "corrompido.pdf"),
            _entry(valid_path, 1, "valido.pdf"),
        ],
        spool_root,
        document_ids={0: "corrupt", 1: "valid"},
    )

    assert tuple(document.document_id for document in documents) == ("valid",)
    assert tuple(document.display_name for document in documents) == (
        "valido.pdf",
    )
    assert warnings.warnings == (
        "corrompido.pdf: arquivo ignorado por falha de leitura; substitua-o "
        "ou remova-o antes de reenviar.",
    )
    assert tuple(spool_root.iterdir()) == ()


def test_extract_all_rejects_only_after_every_pdf_is_unreadable(
    tmp_path: Path,
) -> None:
    first = tmp_path / "primeiro.pdf"
    second = tmp_path / "segundo.pdf"
    first.write_bytes(b"corrupt-1")
    second.write_bytes(b"corrupt-2")
    warnings = WarningCollector()
    extractor = PdfExtractor(warnings)
    spool_root = tmp_path / "spool"

    with pytest.raises(PublicError) as caught:
        extractor.extract_all(
            [
                _entry(first, 0, "primeiro.pdf"),
                _entry(second, 1, "segundo.pdf"),
            ],
            spool_root,
        )

    assert caught.value.code == "no_readable_pdfs"
    assert caught.value.http_status == 422
    assert caught.value.message == (
        "Nenhum PDF do arquivo pôde ser lido. Substitua ou remova os "
        "arquivos com problema."
    )
    assert len(warnings.warnings) == 2
    assert warnings.warnings[0].startswith("primeiro.pdf:")
    assert warnings.warnings[1].startswith("segundo.pdf:")
    assert tuple(spool_root.iterdir()) == ()
