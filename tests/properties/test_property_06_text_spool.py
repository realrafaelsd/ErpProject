"""Property test for exact Unicode preservation by the production text spool."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from ingest import _read_utf8_spool, _write_utf8_spool


_SENTINEL_CHARACTERS = st.sampled_from(
    tuple("\0\r\n\t \u00a0\u0301\u2028\u2029\ufeffé漢🙂")
)
_UNICODE_TEXT = st.text(
    alphabet=st.one_of(st.characters(codec="utf-8"), _SENTINEL_CHARACTERS),
    max_size=2_048,
)


@settings(max_examples=150, deadline=None)
@given(text=_UNICODE_TEXT)
def test_property_06_text_spool_preserves_exact_code_points(text: str) -> None:
    # Feature: erp-ai-support, Property 6: Spool textual preserva exatamente o texto extraído
    # **Validates: Requirements 7.7**
    with TemporaryDirectory(prefix="erp-ai-support-spool-property-") as directory:
        spool_path = Path(directory) / "page.txt"

        _write_utf8_spool(spool_path, text)
        restored = _read_utf8_spool(spool_path)

    assert restored == text
    assert tuple(map(ord, restored)) == tuple(map(ord, text))
