"""Property test for content-faithful, page-local character chunking."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from domain import Chunk, ChunkingProfile, ExtractedDocument, PdfPage
from ingest import ChunkingService, make_chunk_id


_TEXT_CHARACTERS = st.characters(blacklist_categories=("Cs",))
_NON_WHITESPACE_CHARACTERS = st.sampled_from(("A", "á", "β", "中", "🙂"))
_WHITESPACE_CHARACTERS = st.sampled_from(
    (
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\u0085",
        "\u00a0",
        "\u1680",
        "\u2003",
        "\u2028",
        "\u2029",
        "\u202f",
        "\u205f",
        "\u3000",
    )
)
_SAFE_STEM_CHARACTERS = st.sampled_from(tuple("abcXYZ019áç中"))


@dataclass(frozen=True, slots=True)
class _ChunkingCase:
    profile: ChunkingProfile
    texts: tuple[str, str]
    whitespace_text: str
    blank_page: int
    document_id: str
    display_name: str
    transaction_id: str


def _draw_length(draw, size: int, stride: int) -> int:
    boundary = draw(st.sampled_from(("short", "exact", "just_over", "multiple")))
    if boundary == "short":
        return draw(st.integers(min_value=1, max_value=size))
    if boundary == "exact":
        return size
    if boundary == "just_over":
        return size + 1

    steps = draw(st.integers(min_value=1, max_value=5))
    tail = draw(st.integers(min_value=1, max_value=size))
    return size + (steps - 1) * stride + tail


def _draw_non_whitespace_text(draw, length: int) -> str:
    characters = draw(
        st.lists(
            _TEXT_CHARACTERS,
            min_size=length,
            max_size=length,
        )
    )
    anchor = draw(st.integers(min_value=0, max_value=length - 1))
    characters[anchor] = draw(_NON_WHITESPACE_CHARACTERS)
    return "".join(characters)


@st.composite
def _chunking_cases(draw):
    size = draw(st.integers(min_value=1, max_value=48))
    overlap_kind = draw(st.sampled_from(("zero", "maximum", "arbitrary")))
    if overlap_kind == "zero" or size == 1:
        overlap = 0
    elif overlap_kind == "maximum":
        overlap = size - 1
    else:
        overlap = draw(st.integers(min_value=0, max_value=size - 1))

    stride = size - overlap
    first_length = _draw_length(draw, size, stride)
    second_length = _draw_length(draw, size, stride)
    return _ChunkingCase(
        profile=ChunkingProfile(
            size=size,
            overlap=overlap,
            schema_version="char-v1",
        ),
        texts=(
            _draw_non_whitespace_text(draw, first_length),
            _draw_non_whitespace_text(draw, second_length),
        ),
        whitespace_text="".join(
            draw(
                st.lists(
                    _WHITESPACE_CHARACTERS,
                    min_size=1,
                    max_size=24,
                )
            )
        ),
        blank_page=draw(st.integers(min_value=1, max_value=3)),
        document_id=draw(st.binary(min_size=32, max_size=32)).hex(),
        display_name=(
            "guias/"
            + draw(st.text(_SAFE_STEM_CHARACTERS, min_size=1, max_size=20))
            + ".pdf"
        ),
        transaction_id="tx-" + str(draw(st.uuids())),
    )


def _expected_starts(text_length: int, profile: ChunkingProfile) -> list[int]:
    starts: list[int] = []
    start = 0
    stride = profile.size - profile.overlap
    while start < text_length:
        starts.append(start)
        if start + profile.size >= text_length:
            break
        start += stride
    return starts


def _assert_page_invariants(
    chunks: list[Chunk],
    *,
    text: str,
    page: int,
    case: _ChunkingCase,
) -> None:
    profile = case.profile
    starts = [chunk.start_offset for chunk in chunks]
    expected_starts = _expected_starts(len(text), profile)

    assert chunks
    assert starts == expected_starts
    assert starts[0] == 0
    assert [chunk.text for chunk in chunks] == [
        text[start : start + profile.size] for start in expected_starts
    ]
    assert all(0 < len(chunk.text) <= profile.size for chunk in chunks)
    assert all(
        chunk.text
        == text[
            chunk.start_offset : chunk.start_offset + len(chunk.text)
        ]
        for chunk in chunks
    )

    stride = profile.size - profile.overlap
    for previous, current in zip(chunks, chunks[1:]):
        assert len(previous.text) == profile.size
        assert current.start_offset - previous.start_offset == stride
        assert (
            previous.start_offset + len(previous.text) - current.start_offset
            == profile.overlap
        )
        if profile.overlap == 0:
            assert (
                previous.start_offset + len(previous.text)
                == current.start_offset
            )
        else:
            assert (
                previous.text[-profile.overlap :]
                == current.text[: profile.overlap]
            )

    covered = {
        position
        for chunk in chunks
        for position in range(
            chunk.start_offset,
            chunk.start_offset + len(chunk.text),
        )
    }
    assert covered == set(range(len(text)))
    ends = [chunk.start_offset + len(chunk.text) for chunk in chunks]
    assert ends[-1] == len(text)
    assert all(end < len(text) for end in ends[:-1])

    assert all(chunk.document_id == case.document_id for chunk in chunks)
    assert all(chunk.display_name == case.display_name for chunk in chunks)
    assert all(chunk.human_page == page for chunk in chunks)
    assert all(chunk.transaction_id == case.transaction_id for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [
        make_chunk_id(case.document_id, page, start)
        for start in expected_starts
    ]

    if len(text) <= profile.size:
        assert len(chunks) == 1
        assert chunks[0].start_offset == 0
        assert chunks[0].text == text
    else:
        assert len(chunks) >= 2


@settings(max_examples=200, deadline=None)
@given(case=_chunking_cases())
def test_property_09_chunking_preserves_content_overlap_coverage_and_origin(
    case: _ChunkingCase,
) -> None:
    # Feature: erp-ai-support, Property 9: Chunking preserva conteúdo, overlap, cobertura e origem
    # **Validates: Requirements 7.9, 7.14, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10**
    service = ChunkingService(case.profile)
    assert service.profile == case.profile
    assert case.whitespace_text.strip() == ""

    text_iterator = iter(case.texts)
    pages = tuple(
        PdfPage(
            human_page=human_page,
            text=(
                case.whitespace_text
                if human_page == case.blank_page
                else next(text_iterator)
            ),
        )
        for human_page in range(1, 4)
    )
    document = ExtractedDocument(
        document_id=case.document_id,
        display_name=case.display_name,
        pages=pages,
    )

    expected_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = service.split_page(
            page,
            document_id=case.document_id,
            display_name=case.display_name,
            transaction_id=case.transaction_id,
        )
        if page.human_page == case.blank_page:
            assert page_chunks == []
        else:
            _assert_page_invariants(
                page_chunks,
                text=page.text,
                page=page.human_page,
                case=case,
            )
            expected_chunks.extend(page_chunks)

    chunks = service.split_document(document, case.transaction_id)
    assert chunks == expected_chunks
    assert all(chunk.human_page != case.blank_page for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    fresh_service = ChunkingService(case.profile)
    assert fresh_service.split_document(document, case.transaction_id) == chunks
    assert service.split_document(document, case.transaction_id) == chunks
    assert service.split_page(
        PdfPage(human_page=4, text=""),
        case.document_id,
        case.display_name,
        case.transaction_id,
    ) == []
