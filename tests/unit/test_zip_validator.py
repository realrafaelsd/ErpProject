"""Unit tests for complete ZIP inspection and bounded extraction."""

from __future__ import annotations

import os
import stat
import zipfile
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

import ingest as ingest_module
from domain import AppConfig, PublicError
from ingest import ZipValidator


def _config(root: Path, **changes: object) -> AppConfig:
    config = AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generation-model",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local/test-embedding",
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=10,
        max_zip_entry_bytes=1_048_576,
        max_uncompressed_bytes=2_097_152,
        max_compression_ratio=100.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )
    return replace(config, **changes)


def _write_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _regular_info(
    name: str,
    *,
    declared_size: int,
    compressed_size: int,
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.file_size = declared_size
    info.compress_size = compressed_size
    return info


def _install_fake_archive(
    monkeypatch: pytest.MonkeyPatch,
    infos: tuple[zipfile.ZipInfo, ...],
    stream_factories: dict[str, Callable[[], object]] | None = None,
) -> None:
    factories = stream_factories or {}

    class FakeZipFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "FakeZipFile":
            return self

        def __exit__(self, *args: object) -> bool:
            del args
            self.close()
            return False

        def infolist(self) -> list[zipfile.ZipInfo]:
            return list(infos)

        def open(self, info: zipfile.ZipInfo, mode: str = "r") -> object:
            assert mode == "r"
            return factories[info.filename]()

        def close(self) -> None:
            return None

    monkeypatch.setattr(ingest_module.zipfile, "ZipFile", FakeZipFile)


class _ChunkedStream:
    """A member stream that preserves caller-controlled block boundaries."""

    def __init__(self, blocks: list[bytes]) -> None:
        self._blocks = list(blocks)

    def __enter__(self) -> "_ChunkedStream":
        return self

    def __exit__(self, *args: object) -> bool:
        del args
        return False

    def read(self, size: int = -1) -> bytes:
        if not self._blocks:
            return b""
        block = self._blocks[0]
        if size < 0 or len(block) <= size:
            self._blocks.pop(0)
            return block
        result = block[:size]
        self._blocks[0] = block[size:]
        return result


class _RecordingOS:
    """Delegate to the real OS while recording only extraction writes."""

    def __init__(self) -> None:
        self.writes: list[int] = []

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def write(self, descriptor: int, data: object) -> int:
        amount = os.write(descriptor, data)  # type: ignore[arg-type]
        self.writes.append(amount)
        return amount


def _assert_public_error(captured: pytest.ExceptionInfo[PublicError], code: str) -> None:
    assert captured.value.code == code
    assert captured.value.http_status == 422
    assert captured.value.message


def test_inspect_materializes_valid_archive_without_creating_output(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "valid.zip"
    extraction_root = tmp_path / "extracted"
    _write_archive(
        archive_path,
        [
            ("manual.pdf", b"manual"),
            ("guias/config.pdf", b"configuracao"),
        ],
    )

    plan = ZipValidator(_config(tmp_path)).inspect(archive_path, extraction_root)

    assert [entry.ordinal for entry in plan.entries] == [0, 1]
    assert [entry.relative_path for entry in plan.entries] == [
        "manual.pdf",
        "guias/config.pdf",
    ]
    assert plan.declared_total_bytes == len(b"manualconfiguracao")
    assert all(
        entry.resolved_target.is_relative_to(extraction_root.resolve())
        for entry in plan.entries
    )
    assert not extraction_root.exists()


def test_invalid_enumeration_is_rejected_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenEnumerationArchive:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> "BrokenEnumerationArchive":
            return self

        def __exit__(self, *args: object) -> bool:
            del args
            return False

        def infolist(self) -> list[zipfile.ZipInfo]:
            raise zipfile.BadZipFile("private enumeration detail")

    monkeypatch.setattr(
        ingest_module.zipfile,
        "ZipFile",
        BrokenEnumerationArchive,
    )
    extraction_root = tmp_path / "extracted"

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(
            tmp_path / "broken.zip",
            extraction_root,
        )

    _assert_public_error(captured, "invalid_zip")
    assert captured.value.message == (
        "O arquivo não é um ZIP válido. Envie um arquivo ZIP válido."
    )
    assert "private enumeration detail" not in captured.value.message
    assert not extraction_root.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.pdf",
        "safe/../../escape.pdf",
        r"..\escape.pdf",
        "/absolute.pdf",
        r"C:\escape.pdf",
        r"C:escape.pdf",
        r"\\server\share\escape.pdf",
        r"\\?\C:\escape.pdf",
    ],
)
def test_rejects_posix_windows_and_unc_traversal_before_writing(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    extraction_root = tmp_path / "extracted"
    _write_archive(
        archive_path,
        [("safe.pdf", b"safe"), (unsafe_name, b"malicious")],
    )

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(archive_path, extraction_root)

    _assert_public_error(captured, "unsafe_zip_entry")
    assert unsafe_name not in captured.value.message
    assert not extraction_root.exists()


@pytest.mark.parametrize(
    "special_mode",
    [
        stat.S_IFLNK | 0o777,
        stat.S_IFIFO | 0o600,
        stat.S_IFCHR | 0o600,
        stat.S_IFBLK | 0o600,
    ],
    ids=["symlink", "fifo", "character-device", "block-device"],
)
def test_rejects_symlink_fifo_and_device_metadata(
    tmp_path: Path,
    special_mode: int,
) -> None:
    archive_path = tmp_path / "special.zip"
    extraction_root = tmp_path / "extracted"
    special = zipfile.ZipInfo("special-entry")
    special.create_system = 3
    special.external_attr = special_mode << 16

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("safe.pdf", b"safe")
        archive.writestr(special, b"target" if stat.S_ISLNK(special_mode) else b"")

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(archive_path, extraction_root)

    _assert_public_error(captured, "unsafe_zip_entry")
    assert "special-entry" not in captured.value.message
    assert not extraction_root.exists()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("docs/manual.pdf", "docs/./manual.pdf"),
        ("docs/manual.pdf", "docs//manual.pdf"),
        ("docs/manual.pdf", r"docs\manual.pdf"),
    ],
)
def test_rejects_duplicate_normalized_destinations(
    tmp_path: Path,
    first: str,
    second: str,
) -> None:
    archive_path = tmp_path / "duplicate.zip"
    extraction_root = tmp_path / "extracted"
    _write_archive(archive_path, [(first, b"one"), (second, b"two")])

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(archive_path, extraction_root)

    _assert_public_error(captured, "unsafe_zip_entry")
    assert first not in captured.value.message
    assert second not in captured.value.message
    assert not extraction_root.exists()


@pytest.mark.parametrize(
    ("entry_count", "accepted"),
    [(2, True), (3, True), (4, False)],
    ids=["N-1", "N", "N+1"],
)
def test_declared_entry_count_boundaries(
    tmp_path: Path,
    entry_count: int,
    accepted: bool,
) -> None:
    archive_path = tmp_path / f"count-{entry_count}.zip"
    root = tmp_path / f"count-{entry_count}"
    _write_archive(
        archive_path,
        [(f"entry-{index}.pdf", b"") for index in range(entry_count)],
    )
    validator = ZipValidator(_config(tmp_path, max_zip_entries=3))

    if accepted:
        assert len(validator.inspect(archive_path, root).entries) == entry_count
    else:
        with pytest.raises(PublicError) as captured:
            validator.inspect(archive_path, root)
        _assert_public_error(captured, "zip_entry_count_exceeded")
    assert not root.exists()


@pytest.mark.parametrize(
    ("entry_size", "accepted"),
    [(7, True), (8, True), (9, False)],
    ids=["N-1", "N", "N+1"],
)
def test_declared_per_entry_size_boundaries(
    tmp_path: Path,
    entry_size: int,
    accepted: bool,
) -> None:
    archive_path = tmp_path / f"entry-size-{entry_size}.zip"
    root = tmp_path / f"entry-size-{entry_size}"
    _write_archive(archive_path, [("manual.pdf", b"x" * entry_size)])
    validator = ZipValidator(
        _config(
            tmp_path,
            max_zip_entry_bytes=8,
            max_uncompressed_bytes=32,
        )
    )

    if accepted:
        plan = validator.inspect(archive_path, root)
        assert plan.entries[0].declared_size == entry_size
    else:
        with pytest.raises(PublicError) as captured:
            validator.inspect(archive_path, root)
        _assert_public_error(captured, "zip_entry_size_exceeded")
    assert not root.exists()


@pytest.mark.parametrize(
    ("declared_total", "accepted"),
    [(7, True), (8, True), (9, False)],
    ids=["N-1", "N", "N+1"],
)
def test_declared_total_size_boundaries(
    tmp_path: Path,
    declared_total: int,
    accepted: bool,
) -> None:
    archive_path = tmp_path / f"total-{declared_total}.zip"
    root = tmp_path / f"total-{declared_total}"
    _write_archive(
        archive_path,
        [("first.pdf", b"a" * 3), ("second.pdf", b"b" * (declared_total - 3))],
    )
    validator = ZipValidator(
        _config(
            tmp_path,
            max_zip_entry_bytes=16,
            max_uncompressed_bytes=8,
        )
    )

    if accepted:
        assert validator.inspect(archive_path, root).declared_total_bytes == declared_total
    else:
        with pytest.raises(PublicError) as captured:
            validator.inspect(archive_path, root)
        _assert_public_error(captured, "zip_total_size_exceeded")
    assert not root.exists()


@pytest.mark.parametrize(
    ("declared_size", "accepted"),
    [(3, True), (4, True), (5, False)],
    ids=["N-1", "N", "N+1"],
)
def test_declared_compression_ratio_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_size: int,
    accepted: bool,
) -> None:
    info = _regular_info(
        "manual.pdf",
        declared_size=declared_size,
        compressed_size=1,
    )
    _install_fake_archive(monkeypatch, (info,))
    root = tmp_path / f"ratio-{declared_size}"
    validator = ZipValidator(
        _config(
            tmp_path,
            max_zip_entry_bytes=100,
            max_uncompressed_bytes=100,
            max_compression_ratio=4.0,
        )
    )

    if accepted:
        assert validator.inspect(tmp_path / "metadata.zip", root).entries
    else:
        with pytest.raises(PublicError) as captured:
            validator.inspect(tmp_path / "metadata.zip", root)
        _assert_public_error(captured, "zip_compression_ratio_exceeded")
    assert not root.exists()


def test_positive_declared_size_with_zero_compressed_bytes_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _regular_info("manual.pdf", declared_size=1, compressed_size=0)
    _install_fake_archive(monkeypatch, (info,))
    root = tmp_path / "zero-compressed"

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(
            tmp_path / "metadata.zip",
            root,
        )

    _assert_public_error(captured, "zip_compression_ratio_exceeded")
    assert not root.exists()


def test_actual_entry_limit_never_writes_the_n_plus_one_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 64 * 1024
    info = _regular_info(
        "manual.pdf",
        declared_size=limit,
        compressed_size=limit,
    )
    _install_fake_archive(
        monkeypatch,
        (info,),
        {"manual.pdf": lambda: BytesIO(b"x" * (limit + 1))},
    )
    recording_os = _RecordingOS()
    monkeypatch.setattr(ingest_module, "os", recording_os)
    root = tmp_path / "actual-entry-limit"
    validator = ZipValidator(
        _config(
            tmp_path,
            max_zip_entry_bytes=limit,
            max_uncompressed_bytes=limit * 2,
        )
    )
    plan = validator.inspect(tmp_path / "stream.zip", root)

    with pytest.raises(PublicError) as captured:
        validator.extract(tmp_path / "stream.zip", plan)

    _assert_public_error(captured, "zip_entry_size_exceeded")
    assert sum(recording_os.writes) == limit
    assert all(
        sum(recording_os.writes[: index + 1]) <= limit
        for index in range(len(recording_os.writes))
    )
    assert not root.exists()


def test_actual_total_limit_never_writes_the_n_plus_one_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    half = 32 * 1024
    total_limit = half * 2
    infos = (
        _regular_info("first.pdf", declared_size=half, compressed_size=half),
        _regular_info("second.pdf", declared_size=half, compressed_size=half),
    )
    _install_fake_archive(
        monkeypatch,
        infos,
        {
            "first.pdf": lambda: BytesIO(b"a" * half),
            "second.pdf": lambda: _ChunkedStream([b"b" * half, b"!"]),
        },
    )
    recording_os = _RecordingOS()
    monkeypatch.setattr(ingest_module, "os", recording_os)
    root = tmp_path / "actual-total-limit"
    validator = ZipValidator(
        _config(
            tmp_path,
            max_zip_entry_bytes=total_limit * 2,
            max_uncompressed_bytes=total_limit,
        )
    )
    plan = validator.inspect(tmp_path / "stream.zip", root)

    with pytest.raises(PublicError) as captured:
        validator.extract(tmp_path / "stream.zip", plan)

    _assert_public_error(captured, "zip_total_size_exceeded")
    assert sum(recording_os.writes) == total_limit
    assert all(
        sum(recording_os.writes[: index + 1]) <= total_limit
        for index in range(len(recording_os.writes))
    )
    assert not root.exists()
