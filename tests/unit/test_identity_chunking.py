"""Focused tests for deterministic document/chunk identity and chunking."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pymupdf
import pytest

from domain import ChunkingProfile, ExtractedDocument, ExtractedEntry, PdfPage
from ingest import ChunkingService, PdfExtractor, make_chunk_id, sha256_file


def test_sha256_file_depends_only_on_original_bytes_across_stream_blocks(
    tmp_path: Path,
) -> None:
    content = bytes(range(256)) * 300 + b"final"
    first = tmp_path / "first.pdf"
    second = tmp_path / "nested" / "renamed.pdf"
    second.parent.mkdir()
    first.write_bytes(content)
    second.write_bytes(content)

    expected = sha256(content).hexdigest()
    assert sha256_file(first) == expected
    assert sha256_file(second) == expected


def test_pdf_extractor_derives_identity_before_text_processing_when_omitted(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    with pymupdf.open() as source:
        page = source.new_page()
        page.insert_text((72, 72), "Conteúdo original")
        source.save(pdf_path)

    expected = sha256(pdf_path.read_bytes()).hexdigest()
    entry = ExtractedEntry(
        ordinal=0,
        path=pdf_path,
        display_name="manual.pdf",
        size=pdf_path.stat().st_size,
    )

    document = PdfExtractor().extract(entry, tmp_path / "spool")

    assert document is not None
    assert document.document_id == expected


def test_make_chunk_id_uses_exact_char_v1_nul_separated_payload() -> None:
    document_id = "a" * 64
    expected_payload = b"\0".join(
        (b"char-v1", document_id.encode("ascii"), b"3", b"12")
    )
    expected = "chk_" + sha256(expected_payload).hexdigest()

    assert make_chunk_id(document_id, 3, 12, "char-v1") == expected
    assert make_chunk_id(document_id, 3, 12) == expected


def test_chunking_preserves_slices_overlap_coverage_metadata_and_determinism() -> None:
    text = "0123456789ABC"
    page = PdfPage(human_page=2, text=text)
    profile = ChunkingProfile(size=5, overlap=2, schema_version="char-v1")
    service = ChunkingService(profile)

    chunks = service.split_page(page, "b" * 64, "guias/manual.pdf", "tx-1")

    assert [chunk.start_offset for chunk in chunks] == [0, 3, 6, 9]
    assert [chunk.text for chunk in chunks] == [
        text[0:5],
        text[3:8],
        text[6:11],
        text[9:13],
    ]
    assert all(0 < len(chunk.text) <= profile.size for chunk in chunks)
    assert all(
        previous.text[-profile.overlap :] == current.text[: profile.overlap]
        for previous, current in zip(chunks, chunks[1:])
    )
    covered = {
        index
        for chunk in chunks
        for index in range(
            chunk.start_offset,
            chunk.start_offset + len(chunk.text),
        )
    }
    assert covered == set(range(len(text)))
    assert chunks[-1].start_offset + len(chunks[-1].text) == len(text)
    assert all(chunk.document_id == "b" * 64 for chunk in chunks)
    assert all(chunk.display_name == "guias/manual.pdf" for chunk in chunks)
    assert all(chunk.human_page == 2 for chunk in chunks)
    assert all(chunk.transaction_id == "tx-1" for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [
        make_chunk_id("b" * 64, 2, start) for start in (0, 3, 6, 9)
    ]
    assert service.split_page(page, "b" * 64, "guias/manual.pdf", "tx-1") == chunks


def test_chunking_handles_short_whitespace_zero_overlap_and_page_boundaries() -> None:
    service = ChunkingService(chunk_size=4, chunk_overlap=0)
    whitespace = PdfPage(human_page=1, text=" \t\n\u2003")
    short_text = " ç\n"

    assert service.split_page(whitespace, "c" * 64, "manual.pdf", "tx") == []
    short_chunks = service.split_page(
        PdfPage(human_page=1, text=short_text),
        "c" * 64,
        "manual.pdf",
        "tx",
    )
    assert len(short_chunks) == 1
    assert short_chunks[0].text == short_text
    assert short_chunks[0].start_offset == 0

    document = ExtractedDocument(
        document_id="c" * 64,
        display_name="manual.pdf",
        pages=(
            PdfPage(human_page=1, text="abcdef"),
            whitespace,
            PdfPage(human_page=3, text="WXYZ1"),
        ),
    )
    chunks = service.split_document(document, "tx")

    assert [(chunk.human_page, chunk.start_offset, chunk.text) for chunk in chunks] == [
        (1, 0, "abcd"),
        (1, 4, "ef"),
        (3, 0, "WXYZ"),
        (3, 4, "1"),
    ]
    assert all(chunk.transaction_id == "tx" for chunk in chunks)

    with pytest.raises(ValueError):
        ChunkingService(chunk_size=4, chunk_overlap=4)
