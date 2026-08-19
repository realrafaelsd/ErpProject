"""Property test for ordered PDF discovery and safe display names."""

from __future__ import annotations

import ntpath
from pathlib import Path, PurePosixPath
from unicodedata import is_normalized, normalize

from hypothesis import given, settings, strategies as st

from domain import ExtractedEntry
from ingest import WarningCollector, discover_pdf_entries


_CONTROL_CHARACTERS = tuple(
    chr(codepoint)
    for codepoint in (*range(0x20), 0x7F, *range(0x80, 0xA0))
)
_SAFE_CHARACTER = st.characters(
    blacklist_categories=("Cc", "Cs"),
    blacklist_characters=("/", "\\", ":"),
)
_UNICODE_CORE = st.one_of(
    st.sampled_from(
        (
            "configurac\u0327a\u0303o",
            "e\u0301",
            "A\u030Angstro\u0308m",
            "漢字",
            "Δοκιμή",
            "поддержка",
            "دليل",
            "🙂",
        )
    ),
    st.text(_SAFE_CHARACTER, min_size=1, max_size=12).filter(
        lambda value: value not in {".", ".."}
    ),
)
_PDF_EXTENSIONS = st.sampled_from(
    (".pdf", ".pdF", ".pDf", ".pDF", ".Pdf", ".PdF", ".PDf", ".PDF")
)
_NON_PDF_EXTENSIONS = st.sampled_from((".txt", ".md", ".zip", ".pd", ".pdf.bak"))
_SEPARATORS = st.sampled_from(("/", "//", "\\", "/./", "\\.\\"))
_PREFIXES = st.sampled_from(("", "./", ".//", ".\\"))
_INTERNAL_ROOT = Path("/private/internal-upload-staging-canary")


def _is_control(character: str) -> bool:
    codepoint = ord(character)
    return codepoint < 0x20 or 0x7F <= codepoint <= 0x9F


@st.composite
def _decorated_segment(draw):
    """Generate a safe Unicode segment with removable controls."""

    characters = list(draw(_UNICODE_CORE))
    for _ in range(draw(st.integers(min_value=0, max_value=3))):
        index = draw(st.integers(min_value=0, max_value=len(characters)))
        characters.insert(index, draw(st.sampled_from(_CONTROL_CHARACTERS)))
    return "".join(characters)


@st.composite
def _raw_name(draw, is_pdf: bool):
    directory_count = draw(st.integers(min_value=0, max_value=3))
    segments = [draw(_decorated_segment()) for _ in range(directory_count)]
    extension = draw(_PDF_EXTENSIONS if is_pdf else _NON_PDF_EXTENSIONS)
    segments.append(f"{draw(_decorated_segment())}{extension}")

    value = f"{draw(_PREFIXES)}{segments[0]}"
    for segment in segments[1:]:
        value = f"{value}{draw(_SEPARATORS)}{segment}"
    return value


@st.composite
def _ordered_name_cases(draw):
    count = draw(st.integers(min_value=2, max_value=10))
    flags = draw(
        st.lists(st.booleans(), min_size=count, max_size=count)
    )
    pdf_index = draw(st.integers(min_value=0, max_value=count - 1))
    non_pdf_index = draw(
        st.integers(min_value=0, max_value=count - 1).filter(
            lambda index: index != pdf_index
        )
    )
    flags[pdf_index] = True
    flags[non_pdf_index] = False
    return tuple((draw(_raw_name(is_pdf)), is_pdf) for is_pdf in flags)


def _expected_display_name(raw_name: str) -> str:
    """Apply the display-name rules as a small independent oracle."""

    segments: list[str] = []
    for raw_segment in raw_name.replace("\\", "/").split("/"):
        if raw_segment in {"", "."}:
            continue
        without_controls = "".join(
            character for character in raw_segment if not _is_control(character)
        )
        segments.append(normalize("NFC", without_controls))
    return "/".join(segments)


def _assert_safe_relative_name(display_name: str) -> None:
    segments = display_name.split("/")
    assert display_name
    assert not PurePosixPath(display_name).is_absolute()
    assert "\\" not in display_name
    assert all(segment not in {"", ".", ".."} for segment in segments)
    assert all(not ntpath.splitdrive(segment)[0] for segment in segments)
    assert all(not _is_control(character) for character in display_name)
    assert is_normalized("NFC", display_name)
    assert not display_name.startswith(str(_INTERNAL_ROOT))


@settings(max_examples=150, deadline=None)
@given(cases=_ordered_name_cases())
def test_property_04_pdf_discovery_is_safe_and_deterministic(cases) -> None:
    # Feature: erp-ai-support, Property 4: Descoberta e nomes de documentos são seguros e determinísticos
    # **Validates: Requirements 7.1, 7.2, 7.13, 18.9**
    entries = tuple(
        ExtractedEntry(
            ordinal=ordinal,
            path=_INTERNAL_ROOT / f"entry-{ordinal}.bin",
            display_name=raw_name,
            size=ordinal + 1,
        )
        for ordinal, (raw_name, _) in enumerate(cases)
    )
    expected_names = tuple(
        _expected_display_name(raw_name) for raw_name, _ in cases
    )

    first_warnings = WarningCollector()
    second_warnings = WarningCollector()
    first_result = discover_pdf_entries(entries, first_warnings)
    second_result = discover_pdf_entries(entries, second_warnings)

    expected_result = tuple(
        (
            ordinal,
            _INTERNAL_ROOT / f"entry-{ordinal}.bin",
            expected_names[ordinal],
            ordinal + 1,
        )
        for ordinal, (_, is_pdf) in enumerate(cases)
        if is_pdf
    )
    actual_result = tuple(
        (entry.ordinal, entry.path, entry.display_name, entry.size)
        for entry in first_result
    )
    expected_warnings = tuple(
        f"{expected_names[ordinal]}: arquivo ignorado porque não possui a extensão .pdf."
        for ordinal, (_, is_pdf) in enumerate(cases)
        if not is_pdf
    )

    assert actual_result == expected_result
    assert second_result == first_result
    assert first_warnings.warnings == expected_warnings
    assert second_warnings.warnings == expected_warnings
    assert len(first_warnings.warnings) == sum(
        not is_pdf for _, is_pdf in cases
    )
    assert all(
        expected_names[ordinal].casefold().endswith(".pdf") is is_pdf
        for ordinal, (_, is_pdf) in enumerate(cases)
    )
    for display_name in expected_names:
        _assert_safe_relative_name(display_name)
