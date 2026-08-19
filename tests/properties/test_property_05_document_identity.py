"""Property test for content-only document identity."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from hypothesis import given, settings, strategies as st

from ingest import sha256_file


_FILESYSTEM_COMPONENT = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cc", "Cs"),
        blacklist_characters="/\\",
    ),
    min_size=1,
    max_size=24,
).filter(lambda value: value not in {".", ".."})


@settings(max_examples=150, deadline=None)
@given(
    content=st.binary(max_size=131_072),
    first_upload=_FILESYSTEM_COMPONENT,
    second_upload=_FILESYSTEM_COMPONENT,
    first_subpath=st.lists(_FILESYSTEM_COMPONENT, min_size=0, max_size=3),
    second_subpath=st.lists(_FILESYSTEM_COMPONENT, min_size=0, max_size=3),
    first_name=_FILESYSTEM_COMPONENT,
    second_name=_FILESYSTEM_COMPONENT,
)
def test_property_05_document_identity_depends_only_on_bytes(
    content: bytes,
    first_upload: str,
    second_upload: str,
    first_subpath: list[str],
    second_subpath: list[str],
    first_name: str,
    second_name: str,
) -> None:
    # Feature: erp-ai-support, Property 5: Identidade de documento depende somente dos bytes
    # **Validates: Requirements 8.1, 8.2**
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="erp-ai-support-property-05-") as directory:
        root = Path(directory)
        first_path = root.joinpath(
            f"upload-a-{first_upload}",
            *first_subpath,
            f"{first_name}.pdf",
        )
        second_path = root.joinpath(
            f"upload-b-{second_upload}",
            *second_subpath,
            f"{second_name}.pdf",
        )
        first_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        second_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        first_path.write_bytes(content)
        second_path.write_bytes(content)

        expected_identity = sha256(content).hexdigest()

        assert first_path != second_path
        assert sha256_file(first_path) == expected_identity
        assert sha256_file(second_path) == expected_identity
        assert sha256_file(first_path) == sha256_file(second_path)
