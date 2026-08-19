"""Property test for deterministic chunk identity."""

from __future__ import annotations

from hashlib import sha256

from hypothesis import given, settings, strategies as st

from ingest import make_chunk_id


_DOCUMENT_IDENTITIES = st.binary(min_size=32, max_size=32).map(bytes.hex)
_HUMAN_PAGES = st.integers(min_value=1)
_START_OFFSETS = st.integers(min_value=0)
_VALID_SCHEMA_VERSIONS = st.sampled_from(("char-v1",))


@settings(max_examples=150, deadline=None)
@given(
    document_id=_DOCUMENT_IDENTITIES,
    human_page=_HUMAN_PAGES,
    start_offset=_START_OFFSETS,
    schema_version=_VALID_SCHEMA_VERSIONS,
)
def test_property_08_chunk_identity_is_deterministic(
    document_id: str,
    human_page: int,
    start_offset: int,
    schema_version: str,
) -> None:
    # Feature: erp-ai-support, Property 8: Identidade de chunk é determinística
    # **Validates: Requirements 8.9**
    first = make_chunk_id(
        document_id,
        human_page,
        start_offset,
        schema_version,
    )
    repeated = make_chunk_id(
        document_id,
        human_page,
        start_offset,
        schema_version,
    )
    expected_payload = b"\0".join(
        (
            schema_version.encode("utf-8"),
            document_id.encode("ascii"),
            str(human_page).encode("ascii"),
            str(start_offset).encode("ascii"),
        )
    )
    expected = "chk_" + sha256(expected_payload).hexdigest()

    assert repeated == first
    assert first == expected
    assert len(first) == len("chk_") + 64
    assert first.startswith("chk_")
    assert set(first.removeprefix("chk_")) <= set("0123456789abcdef")
