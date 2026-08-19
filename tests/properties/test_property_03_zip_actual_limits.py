"""Property test for actual byte limits during streamed ZIP extraction."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

import ingest
from domain import AppConfig, ArchiveEntryPlan, PublicError


@dataclass(frozen=True, slots=True)
class _ExtractionCase:
    blocks: tuple[bytes, ...]
    entry_limit: int
    total_limit: int
    initial_total: int
    write_quantum: int


@dataclass(frozen=True, slots=True)
class _ReadEvent:
    returned_size: int
    position_after: int


@dataclass(frozen=True, slots=True)
class _WriteEvent:
    data: bytes
    source_position: int


@dataclass(frozen=True, slots=True)
class _ExtractionOutcome:
    error_code: str | None
    returned_counts: tuple[int, int] | None
    sink_data: bytes
    source_data: bytes
    read_events: tuple[_ReadEvent, ...]
    write_events: tuple[_WriteEvent, ...]
    source_closed: bool
    created_file_count: int


@st.composite
def _extraction_cases(draw):
    blocks = tuple(
        draw(
            st.lists(
                st.binary(min_size=2, max_size=32),
                min_size=1,
                max_size=8,
            )
        )
    )
    payload_size = sum(len(block) for block in blocks)
    scenario = draw(
        st.sampled_from(
            ("accepted", "entry_overflow", "total_overflow", "both", "arbitrary")
        )
    )

    if scenario == "accepted":
        entry_limit = payload_size + draw(st.integers(min_value=0, max_value=16))
        total_capacity = payload_size + draw(st.integers(min_value=0, max_value=16))
    elif scenario == "entry_overflow":
        entry_limit = draw(st.integers(min_value=1, max_value=payload_size - 1))
        total_capacity = payload_size + draw(st.integers(min_value=0, max_value=16))
    elif scenario == "total_overflow":
        entry_limit = payload_size + draw(st.integers(min_value=0, max_value=16))
        total_capacity = draw(st.integers(min_value=0, max_value=payload_size - 1))
    elif scenario == "both":
        entry_limit = draw(st.integers(min_value=1, max_value=payload_size - 1))
        total_capacity = draw(st.integers(min_value=0, max_value=payload_size - 1))
    else:
        entry_limit = draw(st.integers(min_value=1, max_value=payload_size + 16))
        total_capacity = draw(st.integers(min_value=0, max_value=payload_size + 16))

    # Keep the generated AppConfig valid: the configured cumulative maximum is
    # always at least the configured per-entry maximum. ``initial_total`` then
    # models bytes already written by preceding archive entries.
    minimum_initial = max(0, entry_limit - total_capacity)
    initial_total = draw(
        st.integers(
            min_value=minimum_initial,
            max_value=minimum_initial + 32,
        )
    )
    return _ExtractionCase(
        blocks=blocks,
        entry_limit=entry_limit,
        total_limit=initial_total + total_capacity,
        initial_total=initial_total,
        write_quantum=draw(st.integers(min_value=1, max_value=17)),
    )


class _BlockStream:
    """A valid short-read stream that preserves generated block boundaries."""

    def __init__(self, blocks: tuple[bytes, ...]) -> None:
        self._blocks = blocks
        self._block_index = 0
        self._block_offset = 0
        self.returned = bytearray()
        self.events: list[_ReadEvent] = []
        self.closed = False

    def __enter__(self) -> _BlockStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def read(self, size: int) -> bytes:
        assert type(size) is int and size > 0
        if self._block_index == len(self._blocks):
            self.events.append(_ReadEvent(returned_size=0, position_after=len(self.returned)))
            return b""

        block = self._blocks[self._block_index]
        end = min(self._block_offset + size, len(block))
        result = block[self._block_offset : end]
        self._block_offset = end
        if self._block_offset == len(block):
            self._block_index += 1
            self._block_offset = 0

        self.returned.extend(result)
        self.events.append(
            _ReadEvent(
                returned_size=len(result),
                position_after=len(self.returned),
            )
        )
        return result


class _FakeArchive:
    def __init__(self, source: _BlockStream) -> None:
        self._source = source
        self.open_calls = 0

    def open(self, info: object, mode: str = "r") -> _BlockStream:
        del info
        assert mode == "r"
        self.open_calls += 1
        return self._source


class _RecordingWriter:
    """Record actual sink bytes while deliberately allowing short writes."""

    def __init__(
        self,
        source: _BlockStream,
        write_quantum: int,
        real_write,
    ) -> None:
        self._source = source
        self._write_quantum = write_quantum
        self._real_write = real_write
        self.events: list[_WriteEvent] = []

    def __call__(self, descriptor: int, data: object) -> int:
        offered = memoryview(data)[: self._write_quantum]
        amount = self._real_write(descriptor, offered)
        assert 0 < amount <= len(offered)
        self.events.append(
            _WriteEvent(
                data=bytes(offered[:amount]),
                source_position=len(self._source.returned),
            )
        )
        return amount


def _config(root: Path, entry_limit: int, total_limit: int) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused",
        chroma_path=root / "chroma",
        chroma_collection="property_collection",
        upload_folder=root / "uploads",
        embedding_model="unused",
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=10,
        max_zip_entry_bytes=entry_limit,
        max_uncompressed_bytes=total_limit,
        max_compression_ratio=100.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _exercise(case: _ExtractionCase) -> _ExtractionOutcome:
    payload_size = sum(len(block) for block in case.blocks)
    total_capacity = case.total_limit - case.initial_total
    declared_size = min(payload_size, case.entry_limit, total_capacity)

    with TemporaryDirectory(prefix="zip-actual-limits-") as directory:
        root = Path(directory)
        target = root / "payload.bin"
        validator = ingest.ZipValidator(
            _config(root, case.entry_limit, case.total_limit)
        )
        entry = ArchiveEntryPlan(
            ordinal=0,
            archive_name="payload.bin",
            relative_path="payload.bin",
            resolved_target=target,
            is_directory=False,
            declared_size=declared_size,
            compressed_size=max(1, declared_size),
        )
        info = zipfile.ZipInfo("payload.bin")
        source = _BlockStream(case.blocks)
        archive = _FakeArchive(source)
        writer = _RecordingWriter(source, case.write_quantum, os.write)
        created_files: list[Path] = []

        error_code: str | None = None
        returned_counts: tuple[int, int] | None = None
        try:
            with patch.object(ingest.os, "write", writer):
                returned_counts = validator._extract_regular_file(
                    archive=archive,
                    info=info,
                    entry=entry,
                    total_written=case.initial_total,
                    created_files=created_files,
                )
        except PublicError as error:
            error_code = error.code

        assert archive.open_calls == 1
        return _ExtractionOutcome(
            error_code=error_code,
            returned_counts=returned_counts,
            sink_data=target.read_bytes(),
            source_data=bytes(source.returned),
            read_events=tuple(source.events),
            write_events=tuple(writer.events),
            source_closed=source.closed,
            created_file_count=len(created_files),
        )


@settings(max_examples=200, deadline=None)
@given(case=_extraction_cases())
def test_property_03_extraction_never_writes_beyond_actual_limits(case) -> None:
    # Feature: erp-ai-support, Property 3: Extração nunca grava além dos limites reais
    # **Validates: Requirements 6.8, 6.9**
    payload = b"".join(case.blocks)
    total_capacity = case.total_limit - case.initial_total
    effective_capacity = min(case.entry_limit, total_capacity)
    outcome = _exercise(case)

    assert outcome.source_closed
    assert outcome.created_file_count == 1
    assert outcome.sink_data == b"".join(
        event.data for event in outcome.write_events
    )

    actual_written = 0
    for event in outcome.write_events:
        actual_written += len(event.data)
        # The sink's observed count is the ground truth for both production
        # counters at every write boundary, including forced short writes.
        assert actual_written <= case.entry_limit
        assert case.initial_total + actual_written <= case.total_limit
        assert event.source_position <= effective_capacity

    assert actual_written == len(outcome.sink_data)
    assert outcome.sink_data == payload[:actual_written]

    if len(payload) <= effective_capacity:
        assert outcome.error_code is None
        assert outcome.returned_counts == (
            actual_written,
            case.initial_total + actual_written,
        )
        assert actual_written == len(payload)
        assert outcome.source_data == payload
        assert sum(event.returned_size == 0 for event in outcome.read_events) == 1
    else:
        expected_code = (
            "zip_entry_size_exceeded"
            if case.entry_limit <= total_capacity
            else "zip_total_size_exceeded"
        )
        assert outcome.error_code == expected_code
        assert outcome.returned_counts is None
        assert outcome.source_data == payload[: effective_capacity + 1]
        assert outcome.read_events[-1].position_after == effective_capacity + 1
        assert outcome.read_events[-1].returned_size > 0
        assert all(
            event.position_after <= effective_capacity
            for event in outcome.read_events[:-1]
        )
        assert all(event.returned_size > 0 for event in outcome.read_events)
        assert actual_written <= effective_capacity
