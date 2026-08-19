"""Unit tests for ordered PDF discovery and safe display metadata."""

from __future__ import annotations

from pathlib import Path
from unicodedata import is_normalized

import pytest

from domain import ExtractedEntry, PublicError
from ingest import (
    WarningCollector,
    discover_pdf_entries,
    normalize_display_name,
)


def _entry(root: Path, ordinal: int, display_name: str) -> ExtractedEntry:
    path = root / f"entry-{ordinal}"
    payload = f"payload-{ordinal}".encode()
    path.write_bytes(payload)
    return ExtractedEntry(
        ordinal=ordinal,
        path=path,
        display_name=display_name,
        size=len(payload),
    )


def test_discovers_nested_pdfs_case_insensitively_in_zip_order(tmp_path: Path) -> None:
    warnings = WarningCollector()
    entries = [
        _entry(tmp_path, 3, "extras/notas.TXT"),
        _entry(tmp_path, 2, "subpasta\\Manual.PdF"),
        _entry(tmp_path, 0, "guias/configurac\u0327a\u0303o.PDF"),
        _entry(tmp_path, 1, "extras/leia-me.md"),
    ]

    discovered = discover_pdf_entries(entries, warnings)

    assert [entry.ordinal for entry in discovered] == [0, 2]
    assert [entry.display_name for entry in discovered] == [
        "guias/configuração.PDF",
        "subpasta/Manual.PdF",
    ]
    assert all(is_normalized("NFC", entry.display_name) for entry in discovered)
    assert warnings.warnings == (
        "extras/leia-me.md: arquivo ignorado porque não possui a extensão .pdf.",
        "extras/notas.TXT: arquivo ignorado porque não possui a extensão .pdf.",
    )


def test_normalize_display_name_removes_controls_and_never_uses_system_path() -> None:
    display_name = normalize_display_name("./manuais//ca\u0000dastro/e\u0301.PDF")

    assert display_name == "manuais/cadastro/é.PDF"
    assert display_name.startswith("/") is False
    assert ".." not in display_name.split("/")
    assert "staging" not in display_name


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "/tmp/staging/manual.pdf",
        "../manual.pdf",
        "docs/../manual.pdf",
        r"C:\staging\manual.pdf",
        r"\\server\share\manual.pdf",
        "docs/C:/manual.pdf",
        "docs/.\u0000./manual.pdf",
    ],
)
def test_normalize_display_name_rejects_absolute_and_dangerous_segments(
    unsafe_name: str,
) -> None:
    with pytest.raises(PublicError) as captured:
        normalize_display_name(unsafe_name)

    assert captured.value.code == "unsafe_zip_entry"
    assert captured.value.http_status == 422
    assert unsafe_name not in captured.value.message


def test_warning_collector_deduplicates_by_structural_key_in_first_order() -> None:
    warnings = WarningCollector()

    assert warnings.add("non_pdf", 4, None, "primeiro") is True
    assert warnings.add("non_pdf", 4, None, "texto posterior") is False
    assert warnings.add("empty_page", 4, 1, "segundo") is True
    assert warnings.add("non_pdf", 5, None, "terceiro") is True

    assert len(warnings) == 3
    assert warnings.as_tuple() == ("primeiro", "segundo", "terceiro")
    assert tuple(warnings) == warnings.warnings


def test_rejects_upload_without_pdf_after_collecting_each_non_pdf_warning(
    tmp_path: Path,
) -> None:
    warnings = WarningCollector()
    entries = [
        _entry(tmp_path, 1, "dados.csv"),
        _entry(tmp_path, 0, "notas.txt"),
    ]

    with pytest.raises(PublicError) as captured:
        discover_pdf_entries(entries, warnings)

    assert captured.value.code == "no_pdfs"
    assert captured.value.http_status == 422
    assert captured.value.message == "O arquivo ZIP não contém documentos PDF."
    assert warnings.warnings == (
        "notas.txt: arquivo ignorado porque não possui a extensão .pdf.",
        "dados.csv: arquivo ignorado porque não possui a extensão .pdf.",
    )
