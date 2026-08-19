"""Unit tests for the SQLite manifest and recovery journal."""

from __future__ import annotations

import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Sequence

import pytest

from domain import (
    AppConfig,
    Chunk,
    ChunkingProfile,
    CommitPlan,
    DocumentManifest,
    DocumentManifestKey,
    PublicError,
    StoredChunk,
    VectorSpace,
)
from rag import IngestionMutex, LifecycleLock, ManifestStore, VisibilityLock


def test_lifecycle_lock_rejects_a_second_instance_and_can_be_reacquired(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "lock-root"
    lock_root.mkdir()
    first = LifecycleLock(lock_root)
    second = LifecycleLock(lock_root)

    first.acquire()
    try:
        assert first.acquired is True
        assert stat.S_IMODE(
            (lock_root / ".erp-ai-support.lock").stat().st_mode
        ) == 0o600
        with pytest.raises(PublicError) as captured:
            second.acquire()

        assert captured.value.code == "application_already_running"
        assert captured.value.http_status == 500
        assert str(lock_root) not in captured.value.message
    finally:
        first.release()

    try:
        second.acquire()
        assert second.acquired is True
    finally:
        second.release()
    assert first.acquired is False
    assert second.acquired is False


def test_ingestion_mutex_is_nonblocking_and_reports_public_contention() -> None:
    mutex = IngestionMutex()

    assert mutex.acquire() is True
    try:
        assert mutex.locked() is True
        assert mutex.acquire() is False
        with pytest.raises(PublicError) as captured:
            with mutex.hold():
                pytest.fail("a contended ingestion mutex must not be entered")

        assert captured.value.code == "ingestion_in_progress"
        assert captured.value.http_status == 409
        assert "thread" not in captured.value.message.casefold()
    finally:
        mutex.release()

    assert mutex.locked() is False
    with pytest.raises(ValueError):
        mutex.acquire(blocking=True)


def test_visibility_lock_hides_an_exclusive_window_from_readers() -> None:
    visibility = VisibilityLock()
    writer_started = Event()
    writer_acquired = Event()
    release_writer = Event()
    reader_started = Event()
    reader_acquired = Event()

    def write() -> None:
        writer_started.set()
        with visibility.exclusive():
            writer_acquired.set()
            assert release_writer.wait(timeout=2)

    def read() -> None:
        reader_started.set()
        with visibility.shared():
            reader_acquired.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        visibility.acquire_shared()
        try:
            writer = executor.submit(write)
            assert writer_started.wait(timeout=2)
            assert not writer_acquired.wait(timeout=0.05)
        finally:
            visibility.release_shared()

        assert writer_acquired.wait(timeout=2)
        reader = executor.submit(read)
        assert reader_started.wait(timeout=2)
        try:
            assert not reader_acquired.wait(timeout=0.05)
        finally:
            release_writer.set()

        writer.result(timeout=2)
        assert reader_acquired.wait(timeout=2)
        reader.result(timeout=2)


def test_journal_rejects_illegal_transitions_without_exposing_identifiers(
    tmp_path: Path,
) -> None:
    store = ManifestStore(_config(tmp_path))
    plan = _plan("tx-INTERNAL-JOURNAL-CANARY")
    store.prepare(plan)

    with pytest.raises(PublicError) as premature_commit:
        store.commit_manifests(plan.transaction_id, plan.manifests)

    assert premature_commit.value.code == "vector_store_write_failed"
    assert plan.transaction_id not in premature_commit.value.message
    assert store.get_journal(plan.transaction_id).state == "PREPARED"

    store.mark_chroma_committed(plan.transaction_id)
    store.commit_manifests(plan.transaction_id, plan.manifests)
    with pytest.raises(PublicError) as committed_abort:
        store.abort(plan.transaction_id)

    assert committed_abort.value.code == "vector_store_write_failed"
    assert plan.transaction_id not in committed_abort.value.message
    assert store.get_journal(plan.transaction_id).state == "COMMITTED"
    assert store.is_duplicate(plan.manifests[0].key) is True


def test_startup_recovery_compensates_chroma_committed_journal(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    initial = ManifestStore(config)
    plan = _plan("tx-chroma-committed-recovery")
    initial.prepare(plan)
    initial.mark_chroma_committed(plan.transaction_id)

    reopened = ManifestStore(config)
    assert reopened.requires_recovery is True
    vector_store = _RecordingRecoveryStore()

    reopened.recover_incomplete(vector_store)

    assert vector_store.rollback_calls == [plan.new_chunk_ids]
    assert reopened.requires_recovery is False
    assert reopened.get_journal(plan.transaction_id).state == "ABORTED"
    assert reopened.is_duplicate(plan.manifests[0].key) is False


def _config(root: Path) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generation-model",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local/test-model",
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


def _space() -> VectorSpace:
    return VectorSpace(
        model="local/test-model",
        dimension=3,
        normalized=True,
        metric="cosine",
    )


def _profile() -> ChunkingProfile:
    return ChunkingProfile(size=800, overlap=150, schema_version="char-v1")


def _plan(
    transaction_id: str,
    *,
    display_name: str = "manual.pdf",
    text: str = "conteúdo do chunk",
    warnings: tuple[str, ...] = (),
) -> CommitPlan:
    space = _space()
    profile = _profile()
    chunk_id = f"chk_{transaction_id}"
    manifest = DocumentManifest(
        key=DocumentManifestKey(
            document_id="a" * 64,
            vector_fingerprint=space.fingerprint,
            chunk_size=profile.size,
            chunk_overlap=profile.overlap,
            chunk_schema_version=profile.schema_version,
        ),
        first_display_name=display_name,
        page_count=2,
        chunk_count=1,
        transaction_id=transaction_id,
    )
    stored = StoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="a" * 64,
            display_name=display_name,
            human_page=1,
            start_offset=0,
            text=text,
            transaction_id=transaction_id,
        ),
        embedding=(1.0, 2.0, 3.0),
    )
    return CommitPlan(
        transaction_id=transaction_id,
        chunks=(stored,),
        manifests=(manifest,),
        new_chunk_ids=(chunk_id,),
        documents=1,
        pages=2,
        warnings=warnings,
    )


class _RecordingRecoveryStore:
    def __init__(self) -> None:
        self.rollback_calls: list[tuple[str, ...]] = []

    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        self.rollback_calls.append(tuple(ids))


class _FailingRecoveryStore:
    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        raise RuntimeError("RECOVERY_INTERNAL_CANARY")


def test_registers_and_reopens_vector_contract_with_sqlite_safety_pragmas(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = ManifestStore(config)
    space = _space()
    profile = _profile()

    store.ensure_vector_space(space, profile)

    assert store.database_path.is_file()
    assert store.get_vector_space() == (space, profile)
    with store._read_connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2

    reopened = ManifestStore(config)
    assert reopened.get_vector_space() == (space, profile)
    assert reopened.requires_recovery is False


@pytest.mark.parametrize(
    ("space_changes", "profile_changes", "expected_code"),
    [
        ({"dimension": 4}, {}, "vector_space_mismatch"),
        ({"model": "local/other-model"}, {}, "vector_space_mismatch"),
        ({}, {"size": 900}, "chunk_profile_mismatch"),
        ({}, {"overlap": 149}, "chunk_profile_mismatch"),
    ],
)
def test_rejects_incompatible_persisted_contract_without_overwriting_it(
    tmp_path: Path,
    space_changes: dict[str, object],
    profile_changes: dict[str, object],
    expected_code: str,
) -> None:
    store = ManifestStore(_config(tmp_path))
    original_space = _space()
    original_profile = _profile()
    store.ensure_vector_space(original_space, original_profile)

    incompatible_space = replace(original_space, **space_changes)
    incompatible_profile = replace(original_profile, **profile_changes)
    with pytest.raises(PublicError) as captured:
        store.ensure_vector_space(incompatible_space, incompatible_profile)

    assert captured.value.code == expected_code
    assert captured.value.http_status == 503
    assert str(tmp_path) not in captured.value.message
    assert store.get_vector_space() == (original_space, original_profile)


def test_journal_advances_idempotently_and_only_committed_documents_are_duplicates(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = ManifestStore(config)
    plan = _plan("tx-committed")
    key = plan.manifests[0].key

    first = store.prepare(plan)
    second = store.prepare(plan)
    assert first == second
    assert first.state == "PREPARED"
    assert store.is_duplicate(key) is False

    store.mark_chroma_committed(plan.transaction_id)
    store.mark_chroma_committed(plan.transaction_id)
    assert store.get_journal(plan.transaction_id).state == "CHROMA_COMMITTED"
    assert store.is_duplicate(key) is False

    store.commit_manifests(plan.transaction_id, plan.manifests)
    store.commit_manifests(plan.transaction_id, plan.manifests)
    assert store.get_journal(plan.transaction_id).state == "COMMITTED"
    assert store.is_duplicate(key) is True

    reopened = ManifestStore(config)
    assert reopened.requires_recovery is False
    assert reopened.is_duplicate(key) is True
    assert reopened.get_journal(plan.transaction_id).state == "COMMITTED"


def test_aborted_journal_never_qualifies_as_duplicate_and_cannot_be_committed(
    tmp_path: Path,
) -> None:
    store = ManifestStore(_config(tmp_path))
    plan = _plan("tx-aborted")
    store.prepare(plan)

    store.abort(plan.transaction_id)
    store.abort(plan.transaction_id)

    assert store.get_journal(plan.transaction_id).state == "ABORTED"
    assert store.is_duplicate(plan.manifests[0].key) is False
    with pytest.raises(PublicError) as captured:
        store.mark_chroma_committed(plan.transaction_id)
    assert captured.value.code == "vector_store_write_failed"
    assert captured.value.http_status == 503


def test_startup_recovery_rolls_back_explicit_ids_then_marks_journal_aborted(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    initial = ManifestStore(config)
    plan = _plan("tx-recover")
    initial.prepare(plan)

    reopened = ManifestStore(config)
    assert reopened.requires_recovery is True
    with pytest.raises(PublicError) as blocked:
        reopened.is_duplicate(plan.manifests[0].key)
    assert blocked.value.code == "recovery_required"

    vector_store = _RecordingRecoveryStore()
    reopened.recover_incomplete(vector_store)

    assert vector_store.rollback_calls == [plan.new_chunk_ids]
    assert reopened.requires_recovery is False
    assert reopened.get_journal(plan.transaction_id).state == "ABORTED"
    assert reopened.is_duplicate(plan.manifests[0].key) is False


def test_failed_startup_recovery_is_public_sanitized_and_retryable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    initial = ManifestStore(config)
    plan = _plan("tx-retry-recovery")
    initial.prepare(plan)
    reopened = ManifestStore(config)

    with pytest.raises(PublicError) as captured:
        reopened.recover_incomplete(_FailingRecoveryStore())

    assert captured.value.code == "recovery_required"
    assert captured.value.http_status == 503
    assert "RECOVERY_INTERNAL_CANARY" not in captured.value.message
    assert str(tmp_path) not in captured.value.message
    assert reopened.requires_recovery is True

    successful = _RecordingRecoveryStore()
    reopened.recover_incomplete(successful)
    assert successful.rollback_calls == [plan.new_chunk_ids]
    assert reopened.requires_recovery is False


def test_malformed_journal_payload_maps_to_recovery_required(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = ManifestStore(config)
    plan = _plan("tx-malformed")
    store.prepare(plan)
    with store._write_connection() as connection:
        connection.execute(
            """
            UPDATE ingestion_transactions
            SET new_chunk_ids_json = ?
            WHERE transaction_id = ?
            """,
            ("JOURNAL_PAYLOAD_CANARY", plan.transaction_id),
        )

    reopened = ManifestStore(config)
    with pytest.raises(PublicError) as captured:
        reopened.incomplete_journals()

    assert captured.value.code == "recovery_required"
    assert captured.value.http_status == 503
    assert "JOURNAL_PAYLOAD_CANARY" not in captured.value.message


def test_manifest_uses_parameters_and_never_persists_chunk_or_conversation_text(
    tmp_path: Path,
) -> None:
    store = ManifestStore(_config(tmp_path))
    plan = _plan(
        "tx-private",
        display_name="manual d'usuário.pdf",
        text="CHUNK_TEXT_CANARY",
        warnings=("QUESTION_CANARY", "ANSWER_CANARY"),
    )

    store.prepare(plan)
    store.mark_chroma_committed(plan.transaction_id)
    store.commit_manifests(plan.transaction_id, plan.manifests)

    with store._read_connection() as connection:
        transaction = connection.execute(
            """
            SELECT new_chunk_ids_json, manifests_json, plan_checksum
            FROM ingestion_transactions
            WHERE transaction_id = ?
            """,
            (plan.transaction_id,),
        ).fetchone()
        document = connection.execute(
            """
            SELECT first_display_name, page_count, chunk_count
            FROM documents
            WHERE transaction_id = ?
            """,
            (plan.transaction_id,),
        ).fetchone()
        columns = {
            row[1]
            for table in ("vector_space", "ingestion_transactions", "documents")
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    persisted = repr(tuple(transaction)) + repr(tuple(document))
    assert document["first_display_name"] == "manual d'usuário.pdf"
    assert "CHUNK_TEXT_CANARY" not in persisted
    assert "QUESTION_CANARY" not in persisted
    assert "ANSWER_CANARY" not in persisted
    assert {"question", "answer", "chunk_text", "prompt", "context"}.isdisjoint(
        columns
    )


def test_manifest_initialization_failure_uses_public_vector_store_error(
    tmp_path: Path,
) -> None:
    invalid_path = tmp_path / "not-a-directory"
    invalid_path.write_text("FILESYSTEM_INTERNAL_CANARY", encoding="utf-8")
    config = replace(_config(tmp_path), chroma_path=invalid_path)

    with pytest.raises(PublicError) as captured:
        ManifestStore(config)

    assert captured.value.code == "vector_store_unavailable"
    assert captured.value.http_status == 503
    assert str(invalid_path) not in captured.value.message
    assert "FILESYSTEM_INTERNAL_CANARY" not in captured.value.message
