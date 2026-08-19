"""Unit coverage for PDF-page chunking invariants and traceable metadata."""

from __future__ import annotations

from hashlib import sha256

import pytest

from domain import ChunkingProfile, ExtractedDocument, PdfPage
from ingest import ChunkingService, make_chunk_id


_DOCUMENT_ID = "d" * 64


@pytest.mark.parametrize(
    ("text", "expected_starts"),
    [
        (" \t\n\u2003", []),
        ("á", [0]),
        ("áβ🙂中Z", [0]),
        ("áβ🙂中Z!", [0, 3]),
    ],
)
def test_chunking_handles_empty_short_exact_and_just_over_boundary_pages(
    text: str,
    expected_starts: list[int],
) -> None:
    service = ChunkingService(
        ChunkingProfile(size=5, overlap=2, schema_version="char-v1")
    )
    page = PdfPage(human_page=3, text=text)

    chunks = service.split_page(
        page,
        document_id=_DOCUMENT_ID,
        display_name="guias/configuração.pdf",
        transaction_id="tx-boundary",
    )

    assert [chunk.start_offset for chunk in chunks] == expected_starts
    assert [chunk.text for chunk in chunks] == [
        text[start : start + 5] for start in expected_starts
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        make_chunk_id(_DOCUMENT_ID, 3, start) for start in expected_starts
    ]
    assert all(chunk.document_id == _DOCUMENT_ID for chunk in chunks)
    assert all(chunk.display_name == "guias/configuração.pdf" for chunk in chunks)
    assert all(chunk.human_page == 3 for chunk in chunks)
    assert all(chunk.transaction_id == "tx-boundary" for chunk in chunks)


def test_long_unicode_page_preserves_slices_overlap_coverage_ids_and_determinism() -> None:
    text = "Tela Cadastro🙂\nCampo ação 漢字 — confirme no ERP."
    profile = ChunkingProfile(size=11, overlap=4, schema_version="char-v1")
    service = ChunkingService(profile)
    page = PdfPage(human_page=7, text=text)

    first = service.split_page(
        page,
        document_id=_DOCUMENT_ID,
        display_name="manual/fluxo.pdf",
        transaction_id="tx-unicode",
    )
    second = service.split_page(
        page,
        document_id=_DOCUMENT_ID,
        display_name="manual/fluxo.pdf",
        transaction_id="tx-unicode",
    )

    assert first == second
    assert first
    assert first[0].start_offset == 0
    assert all(0 < len(chunk.text) <= profile.size for chunk in first)
    assert all(
        current.start_offset - previous.start_offset
        == profile.size - profile.overlap
        for previous, current in zip(first, first[1:])
    )
    assert all(
        chunk.text
        == text[chunk.start_offset : chunk.start_offset + len(chunk.text)]
        for chunk in first
    )
    assert all(
        previous.text[-profile.overlap :] == current.text[: profile.overlap]
        for previous, current in zip(first, first[1:])
    )

    covered_positions = {
        position
        for chunk in first
        for position in range(
            chunk.start_offset,
            chunk.start_offset + len(chunk.text),
        )
    }
    assert covered_positions == set(range(len(text)))
    assert first[-1].start_offset + len(first[-1].text) == len(text)
    assert [chunk.chunk_id for chunk in first] == [
        make_chunk_id(_DOCUMENT_ID, 7, chunk.start_offset) for chunk in first
    ]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_zero_overlap_keeps_pages_disjoint_and_preserves_unicode_exactly() -> None:
    service = ChunkingService(chunk_size=4, chunk_overlap=0)
    document = ExtractedDocument(
        document_id="e" * 64,
        display_name="manual/misto.pdf",
        pages=(
            PdfPage(human_page=1, text="á🙂中Z"),
            PdfPage(human_page=2, text="\n\t\u2003"),
            PdfPage(human_page=3, text="ABC\nç🙂XYZ"),
        ),
    )

    chunks = service.split_document(document, transaction_id="tx-pages")

    assert [(chunk.human_page, chunk.start_offset, chunk.text) for chunk in chunks] == [
        (1, 0, "á🙂中Z"),
        (3, 0, "ABC\n"),
        (3, 4, "ç🙂XY"),
        (3, 8, "Z"),
    ]
    assert all(chunk.human_page != 2 for chunk in chunks)
    for page in (document.pages[0], document.pages[2]):
        page_chunks = [chunk for chunk in chunks if chunk.human_page == page.human_page]
        assert "".join(chunk.text for chunk in page_chunks) == page.text
        assert all(
            previous.start_offset + len(previous.text) == current.start_offset
            for previous, current in zip(page_chunks, page_chunks[1:])
        )
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.display_name == document.display_name for chunk in chunks)
    assert all(chunk.transaction_id == "tx-pages" for chunk in chunks)


def test_chunk_identity_uses_only_schema_document_page_and_offset() -> None:
    expected_payload = b"\0".join((b"char-v1", b"f" * 64, b"2", b"9"))
    expected = "chk_" + sha256(expected_payload).hexdigest()

    assert make_chunk_id("f" * 64, page=2, start=9) == expected
    assert make_chunk_id("f" * 64, page=2, start=9) == expected
    assert make_chunk_id("f" * 64, page=2, start=10) != expected
    assert make_chunk_id("f" * 64, page=3, start=9) != expected
