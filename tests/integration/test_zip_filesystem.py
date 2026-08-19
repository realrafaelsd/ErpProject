"""Filesystem integration tests for secure ZIP extraction and cleanup."""

from __future__ import annotations

import stat
import zipfile
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pytest

from domain import AppConfig, PublicError
from ingest import IngestionService, ZipValidator


def _config(root: Path) -> AppConfig:
    upload_folder = root / "uploads"
    upload_folder.mkdir(exist_ok=True)
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generation-model",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=upload_folder,
        embedding_model="local/test-embedding",
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=20,
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


def _write_stored_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _corrupt_member_payload(path: Path, payload: bytes) -> None:
    archive_bytes = bytearray(path.read_bytes())
    assert archive_bytes.count(payload) == 1
    offset = archive_bytes.index(payload) + len(payload) // 2
    archive_bytes[offset] ^= 0xFF
    path.write_bytes(archive_bytes)


def test_valid_zip_extracts_regular_files_in_order_with_private_permissions(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "valid.zip"
    extraction_root = tmp_path / "extracted"
    members = [
        ("manual.pdf", b"root-manual"),
        ("guias/", b""),
        ("guias/admin.PDF", b"nested-manual"),
        ("notas.txt", b"notes"),
    ]
    _write_stored_archive(archive_path, members)
    validator = ZipValidator(_config(tmp_path))

    plan = validator.inspect(archive_path, extraction_root)
    assert not extraction_root.exists()
    extracted = validator.extract(archive_path, plan)

    assert [entry.ordinal for entry in extracted] == [0, 2, 3]
    assert [entry.display_name for entry in extracted] == [
        "manual.pdf",
        "guias/admin.PDF",
        "notas.txt",
    ]
    assert [entry.path.read_bytes() for entry in extracted] == [
        b"root-manual",
        b"nested-manual",
        b"notes",
    ]
    assert all(entry.path.is_file() for entry in extracted)
    assert all(entry.path.is_relative_to(extraction_root) for entry in extracted)

    assert stat.S_IMODE(extraction_root.stat().st_mode) == 0o700
    assert stat.S_IMODE((extraction_root / "guias").stat().st_mode) == 0o700
    for entry in extracted:
        mode = entry.path.stat().st_mode
        assert stat.S_ISREG(mode)
        assert stat.S_IMODE(mode) == 0o600
        assert mode & 0o111 == 0


def test_truncated_central_directory_is_invalid_before_output_exists(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "truncated.zip"
    extraction_root = tmp_path / "extracted"
    _write_stored_archive(archive_path, [("manual.pdf", b"content")])
    content = archive_path.read_bytes()
    end_record = content.rfind(b"PK\x05\x06")
    assert end_record > 0
    archive_path.write_bytes(content[:end_record])

    with pytest.raises(PublicError) as captured:
        ZipValidator(_config(tmp_path)).inspect(archive_path, extraction_root)

    assert captured.value.code == "invalid_zip"
    assert captured.value.http_status == 422
    assert captured.value.message == (
        "O arquivo não é um ZIP válido. Envie um arquivo ZIP válido."
    )
    assert str(archive_path) not in captured.value.message
    assert not extraction_root.exists()


class _NeverEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_ready(self) -> object:
        self.calls.append("ensure_ready")
        raise AssertionError("embedding must not run after ZIP extraction failure")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        self.calls.append("embed_documents")
        raise AssertionError("embedding must not run after ZIP extraction failure")

    def embed_query(self, text: str) -> list[float]:
        del text
        raise AssertionError("ingestion must not embed queries")


class _NeverManifest:
    def __init__(self) -> None:
        self.calls = 0

    def is_duplicate(self, key: object) -> bool:
        del key
        self.calls += 1
        raise AssertionError("manifest must not run after ZIP extraction failure")


class _NeverVectorStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _unexpected(self, name: str) -> None:
        self.calls.append(name)
        raise AssertionError("vector store must not run after ZIP extraction failure")

    def ingestion_guard(self) -> object:
        self._unexpected("ingestion_guard")

    def ensure_compatible(self, space: object, profile: object) -> None:
        del space, profile
        self._unexpected("ensure_compatible")

    def existing_chunks(self, ids: Sequence[str]) -> dict[str, object]:
        del ids
        self._unexpected("existing_chunks")

    def commit_chunks(self, plan: object) -> None:
        del plan
        self._unexpected("commit_chunks")


class _RecordingPdfExtractor:
    def __init__(self) -> None:
        self.entries: list[object] = []

    def extract(self, entry: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.entries.append(entry)
        raise AssertionError("no partial extraction list may reach the PDF extractor")


def test_crc_failure_publishes_no_partial_list_and_cleans_complete_staging(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    upload_id = uuid4()
    staging = config.upload_folder / f"upload-{upload_id}"
    staging.mkdir(mode=0o700)
    archive_path = staging / "knowledge.zip"
    first_payload = b"FIRST-PDF-PAYLOAD-UNIQUE"
    second_payload = b"SECOND-PDF-PAYLOAD-UNIQUE"
    _write_stored_archive(
        archive_path,
        [
            ("first.pdf", first_payload),
            ("second.pdf", second_payload),
        ],
    )
    _corrupt_member_payload(archive_path, second_payload)

    # Establish that this is a late extraction failure: the first member is
    # readable, while the second reaches a CRC error.
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("first.pdf") == first_payload
        with pytest.raises(zipfile.BadZipFile):
            archive.read("second.pdf")

    embeddings = _NeverEmbeddingProvider()
    manifest = _NeverManifest()
    vector_store = _NeverVectorStore()
    pdf_extractor = _RecordingPdfExtractor()
    service = IngestionService(
        config,
        embeddings,  # type: ignore[arg-type]
        manifest,
        vector_store,  # type: ignore[arg-type]
        pdf_extractor=pdf_extractor,  # type: ignore[arg-type]
    )

    with pytest.raises(PublicError) as captured:
        service.ingest(archive_path, upload_id)

    assert captured.value.code == "zip_extraction_failed"
    assert captured.value.http_status == 422
    assert captured.value.message == (
        "Não foi possível extrair o arquivo ZIP com segurança."
    )
    assert "second.pdf" not in captured.value.message
    assert pdf_extractor.entries == []
    assert embeddings.calls == []
    assert manifest.calls == 0
    assert vector_store.calls == []
    assert not staging.exists()
    assert tuple(config.upload_folder.iterdir()) == ()
