"""Property test for ZIP declared-metadata acceptance limits."""

from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from hypothesis import given, settings, strategies as st

from domain import AppConfig, PublicError
from ingest import ZipValidator


_EXTRACTION_ROOT = Path("/__erp_ai_support_property_02_root__").resolve(
    strict=False
)


@dataclass(frozen=True, slots=True)
class _LimitProfile:
    max_entries: int
    max_entry_bytes: int
    max_total_bytes: int
    max_ratio: int
    compressed_unit: int
    positive_zero_compressed_size: int


@st.composite
def _limit_profiles(draw):
    max_entries = draw(st.integers(min_value=2, max_value=12))
    max_ratio = draw(st.integers(min_value=1, max_value=50))
    compressed_unit = draw(st.integers(min_value=1, max_value=128))
    ratio_boundary = max_ratio * compressed_unit

    # Keep ratio N+1 below the per-entry limit, then keep entry N+1 at or
    # below the total limit. This isolates the category under test.
    max_entry_bytes = draw(
        st.integers(
            min_value=ratio_boundary + 1,
            max_value=ratio_boundary + 1_024,
        )
    )
    max_total_bytes = draw(
        st.integers(
            min_value=max_entry_bytes + 1,
            max_value=(2 * max_entry_bytes) - 1,
        )
    )
    positive_zero_compressed_size = draw(
        st.integers(min_value=1, max_value=max_entry_bytes)
    )
    return _LimitProfile(
        max_entries=max_entries,
        max_entry_bytes=max_entry_bytes,
        max_total_bytes=max_total_bytes,
        max_ratio=max_ratio,
        compressed_unit=compressed_unit,
        positive_zero_compressed_size=positive_zero_compressed_size,
    )


def _config(profile: _LimitProfile) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_collection",
        upload_folder=Path("/unused/uploads"),
        embedding_model="unused/property-model",
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=profile.max_entries,
        max_zip_entry_bytes=profile.max_entry_bytes,
        max_uncompressed_bytes=profile.max_total_bytes,
        max_compression_ratio=float(profile.max_ratio),
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _zip_infos(metadata: Sequence[tuple[int, int]]) -> tuple[zipfile.ZipInfo, ...]:
    infos: list[zipfile.ZipInfo] = []
    for index, (declared_size, compressed_size) in enumerate(metadata):
        info = zipfile.ZipInfo(f"entry-{index:04d}.pdf")
        info.external_attr = (stat.S_IFREG | 0o600) << 16
        info.file_size = declared_size
        info.compress_size = compressed_size
        infos.append(info)
    return tuple(infos)


def _assert_declared_case(
    validator: ZipValidator,
    profile: _LimitProfile,
    metadata: Sequence[tuple[int, int]],
    expected_error: str | None,
    label: str,
) -> None:
    try:
        plan = validator._build_plan(_zip_infos(metadata), _EXTRACTION_ROOT)
    except PublicError as error:
        assert expected_error is not None, (
            f"{label}: unexpected rejection {error.code}"
        )
        assert error.code == expected_error, label
        assert error.http_status == 422, label
        return

    assert expected_error is None, (
        f"{label}: expected rejection {expected_error}, but plan was accepted"
    )
    assert len(plan.entries) == len(metadata) <= profile.max_entries, label
    assert plan.declared_total_bytes == sum(size for size, _ in metadata), label
    assert plan.declared_total_bytes <= profile.max_total_bytes, label
    assert tuple(
        (entry.declared_size, entry.compressed_size) for entry in plan.entries
    ) == tuple(metadata), label

    ratio_limit = Decimal(profile.max_ratio)
    for entry in plan.entries:
        assert entry.declared_size <= profile.max_entry_bytes, label
        if entry.declared_size > 0:
            assert entry.compressed_size > 0, label
        if entry.compressed_size > 0:
            assert (
                Decimal(entry.declared_size)
                <= ratio_limit * Decimal(entry.compressed_size)
            ), label


def _stored_metadata_for_total(
    total: int,
    max_entry_bytes: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    first_size = min(total, max_entry_bytes)
    second_size = total - first_size
    return (
        (first_size, first_size),
        (second_size, second_size),
    )


@settings(max_examples=150, deadline=None)
@given(profile=_limit_profiles())
def test_property_02_declared_zip_limits_are_acceptance_invariants(
    profile: _LimitProfile,
) -> None:
    # Feature: erp-ai-support, Property 2: Limites ZIP declarados são invariantes de aceitação
    # **Validates: Requirements 6.4, 6.5, 6.6, 6.7**
    validator = ZipValidator(_config(profile))

    for entry_count in (
        profile.max_entries - 1,
        profile.max_entries,
        profile.max_entries + 1,
    ):
        _assert_declared_case(
            validator,
            profile,
            [(0, 0)] * entry_count,
            (
                "zip_entry_count_exceeded"
                if entry_count > profile.max_entries
                else None
            ),
            f"entry count {entry_count}/{profile.max_entries}",
        )

    for declared_size in (
        profile.max_entry_bytes - 1,
        profile.max_entry_bytes,
        profile.max_entry_bytes + 1,
    ):
        _assert_declared_case(
            validator,
            profile,
            [(declared_size, declared_size)],
            (
                "zip_entry_size_exceeded"
                if declared_size > profile.max_entry_bytes
                else None
            ),
            f"entry bytes {declared_size}/{profile.max_entry_bytes}",
        )

    for declared_total in (
        profile.max_total_bytes - 1,
        profile.max_total_bytes,
        profile.max_total_bytes + 1,
    ):
        _assert_declared_case(
            validator,
            profile,
            _stored_metadata_for_total(
                declared_total,
                profile.max_entry_bytes,
            ),
            (
                "zip_total_size_exceeded"
                if declared_total > profile.max_total_bytes
                else None
            ),
            f"total bytes {declared_total}/{profile.max_total_bytes}",
        )

    ratio_boundary = profile.max_ratio * profile.compressed_unit
    for declared_size in (
        ratio_boundary - 1,
        ratio_boundary,
        ratio_boundary + 1,
    ):
        _assert_declared_case(
            validator,
            profile,
            [(declared_size, profile.compressed_unit)],
            (
                "zip_compression_ratio_exceeded"
                if declared_size > ratio_boundary
                else None
            ),
            (
                "compression ratio numerator "
                f"{declared_size}/{ratio_boundary}"
            ),
        )

    _assert_declared_case(
        validator,
        profile,
        [(profile.positive_zero_compressed_size, 0)],
        "zip_compression_ratio_exceeded",
        "positive declared size with zero compressed size",
    )
