"""End-to-end ingestion transaction failure, recovery, and concurrency tests.

The suite uses real ZIP/PDF processing, the pinned embedded Chroma adapter, and
its SQLite manifest. Embeddings are deterministic in-memory vectors, so no
model or network access is required.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event

import pymupdf
import pytest

from domain import (
    AppConfig,
    ChunkingProfile,
    DocumentManifestKey,
    PublicError,
    RawCandidate,
    VectorSpace,
)
from ingest import IngestionService
from rag import ChromaVectorStore, ManifestStore


_MODEL = "local/integration-embedding"
_QUERY_VECTOR = (1.0, 0.0, 0.0)


class _DeterministicEmbeddings:
    """Small local embedding provider with an injectable pre-write failure."""

    def __init__(self) -> None:
        self.space = VectorSpace(
            model=_MODEL,
            dimension=3,
            normalized=True,
            metric="cosine",
        )
        self.failure_marker: str | None = None

    def ensure_ready(self) -> VectorSpace:
        return self.space

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        materialized = tuple(texts)
        if self.failure_marker is not None and any(
            self.failure_marker in text for text in materialized
        ):
            raise PublicError(
                code="embedding_failed",
                message=(
                    f"A geração local de embeddings com {_MODEL} falhou. "
                    "Verifique a disponibilidade local do modelo e tente "
                    "novamente."
                ),
                http_status=503,
            )
        return [list(_QUERY_VECTOR) for _ in materialized]

    def embed_query(self, text: str) -> list[float]:
        assert isinstance(text, str)
        return list(_QUERY_VECTOR)


@dataclass(frozen=True, slots=True)
class _Upload:
    transaction_id: str
    archive_path: Path
    staging_path: Path
    document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _VisibleSnapshot:
    count: int
    chunk_ids: frozenset[str]
    document_ids: frozenset[str]


def _config(root: Path) -> AppConfig:
    upload_folder = root / "uploads"
    upload_folder.mkdir(parents=True)
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-local-generator",
        chroma_path=root / "chroma",
        chroma_collection="ingestion_transactions",
        upload_folder=upload_folder,
        embedding_model=_MODEL,
        top_k=6,
        chunk_size=500,
        chunk_overlap=100,
        relevance_threshold=0.3,
        max_upload_bytes=2_097_152,
        max_zip_entries=20,
        max_zip_entry_bytes=1_048_576,
        max_uncompressed_bytes=4_194_304,
        max_compression_ratio=1_000.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _profile(config: AppConfig) -> ChunkingProfile:
    return ChunkingProfile(
        size=config.chunk_size,
        overlap=config.chunk_overlap,
        schema_version="char-v1",
    )


def _pdf_bytes(text: str) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), text)
        return document.tobytes()


def _create_upload(
    config: AppConfig,
    transaction_id: str,
    documents: Sequence[tuple[str, str]],
) -> _Upload:
    staging_path = config.upload_folder / f"upload-{transaction_id}"
    staging_path.mkdir(mode=0o700)
    archive_path = staging_path / "knowledge.zip"
    document_ids: list[str] = []
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for display_name, text in documents:
            payload = _pdf_bytes(text)
            document_ids.append(sha256(payload).hexdigest())
            archive.writestr(display_name, payload)
    return _Upload(
        transaction_id=transaction_id,
        archive_path=archive_path,
        staging_path=staging_path,
        document_ids=tuple(document_ids),
    )


def _manifest_key(
    config: AppConfig,
    space: VectorSpace,
    document_id: str,
) -> DocumentManifestKey:
    return DocumentManifestKey(
        document_id=document_id,
        vector_fingerprint=space.fingerprint,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chunk_schema_version="char-v1",
    )


def _ingest(
    service: IngestionService,
    upload: _Upload,
):
    return service.ingest(upload.archive_path, upload.transaction_id)


def _chat_snapshot(store: ChromaVectorStore) -> _VisibleSnapshot:
    """Read count and retrieval candidates under one shared chat-read window."""

    with store.visibility_lock.shared():
        count = store.count_chunks()
        candidates: list[RawCandidate]
        if count == 0:
            candidates = []
        else:
            candidates = store.query(_QUERY_VECTOR, limit=count)

    assert len(candidates) == count
    return _VisibleSnapshot(
        count=count,
        chunk_ids=frozenset(candidate.chunk_id for candidate in candidates),
        document_ids=frozenset(
            candidate.metadata["document_id"] for candidate in candidates
        ),
    )


def _seed_confirmed_state(
    config: AppConfig,
    service: IngestionService,
    store: ChromaVectorStore,
) -> tuple[_Upload, _VisibleSnapshot]:
    seed = _create_upload(
        config,
        "tx-seed",
        (("base/confirmado.pdf", "Estado confirmado anterior da base."),),
    )
    result = _ingest(service, seed)
    assert (result.documents, result.pages, result.chunks) == (1, 1, 1)
    assert not seed.staging_path.exists()
    snapshot = _chat_snapshot(store)
    assert snapshot.count == 1
    assert snapshot.document_ids == frozenset(seed.document_ids)
    return seed, snapshot


def _recover_then_snapshot(
    config: AppConfig,
    transaction_id: str,
) -> tuple[bool, str | None, _VisibleSnapshot]:
    """Restart, run recovery explicitly, then perform the first count/query."""

    manifest = ManifestStore(config)
    recovery_was_pending = manifest.requires_recovery
    store = ChromaVectorStore(
        config,
        manifest,
        recover_on_startup=False,
    )
    try:
        store.recover_incomplete()
        store.ensure_compatible(
            VectorSpace(
                model=_MODEL,
                dimension=3,
                normalized=True,
                metric="cosine",
            ),
            _profile(config),
        )
        journal = manifest.get_journal(transaction_id)
        snapshot = _chat_snapshot(store)
        return (
            recovery_was_pending,
            None if journal is None else journal.state,
            snapshot,
        )
    finally:
        store.close()


def test_failure_before_first_persistent_write_preserves_confirmed_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _DeterministicEmbeddings()
    manifest = ManifestStore(config)
    store = ChromaVectorStore(config, manifest)
    service = IngestionService(config, embeddings, manifest, store)

    try:
        _, baseline = _seed_confirmed_state(config, service, store)
        failed = _create_upload(
            config,
            "tx-before-write",
            (("falha.pdf", "FALHAANTES da primeira escrita persistente."),),
        )
        embeddings.failure_marker = "FALHAANTES"

        with pytest.raises(PublicError) as captured:
            _ingest(service, failed)

        assert captured.value.code == "embedding_failed"
        assert captured.value.http_status == 503
        assert not failed.staging_path.exists()
        assert manifest.get_journal(failed.transaction_id) is None
        assert _chat_snapshot(store) == baseline
        assert all(
            not manifest.is_duplicate(
                _manifest_key(config, embeddings.space, document_id)
            )
            for document_id in failed.document_ids
        )
    finally:
        store.close()

    pending, journal_state, restarted = _recover_then_snapshot(
        config,
        failed.transaction_id,
    )
    assert pending is False
    assert journal_state is None
    assert restarted == baseline


@pytest.mark.parametrize(
    "failure_stage",
    ("during_chroma", "between_chroma_and_manifest", "in_manifest"),
)
def test_commit_failures_are_compensated_and_preserve_confirmed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    config = _config(tmp_path)
    embeddings = _DeterministicEmbeddings()
    manifest = ManifestStore(config)
    store = ChromaVectorStore(config, manifest)
    service = IngestionService(config, embeddings, manifest, store)
    observed: dict[str, object] = {}

    try:
        _, baseline = _seed_confirmed_state(config, service, store)
        failed = _create_upload(
            config,
            f"tx-{failure_stage}",
            (
                ("novo-a.pdf", f"Conteúdo A para {failure_stage}."),
                ("novo-b.pdf", f"Conteúdo B para {failure_stage}."),
            ),
        )
        collection = store._collection  # noqa: SLF001 - fault inspection
        assert collection is not None

        if failure_stage == "during_chroma":
            original_add = store._add_new_records  # noqa: SLF001

            def fail_during_chroma(collection_arg, chunks) -> None:
                assert len(chunks) == 2
                original_add(collection_arg, chunks[:1])
                observed["physical_count"] = collection_arg.count()
                raise OSError("injected failure during Chroma mutation")

            monkeypatch.setattr(store, "_add_new_records", fail_during_chroma)
        elif failure_stage == "between_chroma_and_manifest":

            def fail_between_chroma_and_manifest(transaction_id: str) -> None:
                journal = manifest.get_journal(transaction_id)
                observed["journal_state"] = journal.state if journal else None
                observed["physical_count"] = collection.count()
                raise OSError("injected failure between Chroma and manifest")

            monkeypatch.setattr(
                manifest,
                "mark_chroma_committed",
                fail_between_chroma_and_manifest,
            )
        else:

            def fail_in_manifest(transaction_id: str, manifests) -> None:
                del manifests
                journal = manifest.get_journal(transaction_id)
                observed["journal_state"] = journal.state if journal else None
                observed["physical_count"] = collection.count()
                raise OSError("injected manifest commit failure")

            monkeypatch.setattr(
                manifest,
                "commit_manifests",
                fail_in_manifest,
            )

        with pytest.raises(PublicError) as captured:
            _ingest(service, failed)

        assert captured.value.code == "vector_store_write_failed"
        assert captured.value.http_status == 503
        assert not failed.staging_path.exists()
        journal = manifest.get_journal(failed.transaction_id)
        assert journal is not None and journal.state == "ABORTED"
        assert _chat_snapshot(store) == baseline
        assert collection.count() == baseline.count
        assert all(
            not manifest.is_duplicate(
                _manifest_key(config, embeddings.space, document_id)
            )
            for document_id in failed.document_ids
        )

        if failure_stage == "during_chroma":
            assert observed == {"physical_count": baseline.count + 1}
        elif failure_stage == "between_chroma_and_manifest":
            assert observed == {
                "journal_state": "PREPARED",
                "physical_count": baseline.count + 2,
            }
        else:
            assert observed == {
                "journal_state": "CHROMA_COMMITTED",
                "physical_count": baseline.count + 2,
            }
    finally:
        store.close()

    pending, journal_state, restarted = _recover_then_snapshot(
        config,
        failed.transaction_id,
    )
    assert pending is False
    assert journal_state == "ABORTED"
    assert restarted == baseline


def test_compensation_failure_requires_restart_recovery_before_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    embeddings = _DeterministicEmbeddings()
    manifest = ManifestStore(config)
    store = ChromaVectorStore(config, manifest)
    service = IngestionService(config, embeddings, manifest, store)

    try:
        _, baseline = _seed_confirmed_state(config, service, store)
        failed = _create_upload(
            config,
            "tx-compensation-failure",
            (
                ("pendente-a.pdf", "Registro pendente A."),
                ("pendente-b.pdf", "Registro pendente B."),
            ),
        )
        collection = store._collection  # noqa: SLF001 - crash residue check
        assert collection is not None

        def fail_manifest_commit(transaction_id: str, manifests) -> None:
            del transaction_id, manifests
            raise OSError("injected final manifest failure")

        def fail_compensation(collection_arg, ids) -> None:
            del collection_arg, ids
            raise OSError("injected compensation failure")

        monkeypatch.setattr(manifest, "commit_manifests", fail_manifest_commit)
        monkeypatch.setattr(store, "_delete_explicit_ids", fail_compensation)

        with pytest.raises(PublicError) as captured:
            _ingest(service, failed)

        assert captured.value.code == "recovery_required"
        assert captured.value.http_status == 503
        assert not failed.staging_path.exists()
        assert manifest.requires_recovery is True
        journal = manifest.get_journal(failed.transaction_id)
        assert journal is not None and journal.state == "CHROMA_COMMITTED"
        assert collection.count() == baseline.count + 2

        with pytest.raises(PublicError) as blocked_count:
            store.count_chunks()
        assert blocked_count.value.code == "recovery_required"
        with pytest.raises(PublicError) as blocked_query:
            store.query(_QUERY_VECTOR, limit=1)
        assert blocked_query.value.code == "recovery_required"
    finally:
        store.close()

    pending, journal_state, recovered = _recover_then_snapshot(
        config,
        failed.transaction_id,
    )
    assert pending is True
    assert journal_state == "ABORTED"
    assert recovered == baseline

    recovered_manifest = ManifestStore(config)
    assert all(
        not recovered_manifest.is_duplicate(
            _manifest_key(config, embeddings.space, document_id)
        )
        for document_id in failed.document_ids
    )


def test_concurrent_ingestion_returns_409_and_chat_reads_never_see_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    embeddings = _DeterministicEmbeddings()
    manifest = ManifestStore(config)
    store = ChromaVectorStore(config, manifest)
    service = IngestionService(config, embeddings, manifest, store)
    release_partial_write = Barrier(2)
    partial_write_visible_raw = Event()
    chat_reader_started = Event()
    physical_counts: list[int] = []

    try:
        _, baseline = _seed_confirmed_state(config, service, store)
        first = _create_upload(
            config,
            "tx-first-concurrent",
            (
                ("primeiro-a.pdf", "Novo estado final A."),
                ("primeiro-b.pdf", "Novo estado final B."),
            ),
        )
        second = _create_upload(
            config,
            "tx-second-concurrent",
            (("segundo.pdf", "Esta ingestão concorrente deve falhar."),),
        )
        collection = store._collection  # noqa: SLF001 - raw partial proof
        assert collection is not None
        original_add = store._add_new_records  # noqa: SLF001

        def add_one_pause_then_finish(collection_arg, chunks) -> None:
            assert len(chunks) == 2
            assert chunks[0].chunk.transaction_id == first.transaction_id
            original_add(collection_arg, chunks[:1])
            physical_counts.append(collection_arg.count())
            partial_write_visible_raw.set()
            release_partial_write.wait(timeout=10)
            original_add(collection_arg, chunks[1:])

        def read_as_chat() -> _VisibleSnapshot:
            chat_reader_started.set()
            return _chat_snapshot(store)

        monkeypatch.setattr(
            store,
            "_add_new_records",
            add_one_pause_then_finish,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            first_future = executor.submit(_ingest, service, first)
            assert partial_write_visible_raw.wait(timeout=10)
            assert physical_counts == [baseline.count + 1]

            reader_future = executor.submit(read_as_chat)
            assert chat_reader_started.wait(timeout=5)
            with pytest.raises(FutureTimeout):
                reader_future.result(timeout=0.2)

            second_future = executor.submit(_ingest, service, second)
            try:
                with pytest.raises(PublicError) as busy:
                    second_future.result(timeout=10)
                assert busy.value.code == "ingestion_in_progress"
                assert busy.value.http_status == 409
                assert not second.staging_path.exists()
                assert manifest.get_journal(second.transaction_id) is None
            finally:
                try:
                    release_partial_write.wait(timeout=10)
                except BrokenBarrierError:
                    pass

            first_result = first_future.result(timeout=10)
            during_commit_read = reader_future.result(timeout=10)

        assert (first_result.documents, first_result.pages, first_result.chunks) == (
            2,
            2,
            2,
        )
        assert not first.staging_path.exists()
        final = _chat_snapshot(store)
        assert final.count == baseline.count + 2
        assert final.document_ids == (
            baseline.document_ids | frozenset(first.document_ids)
        )
        assert during_commit_read == final
        assert during_commit_read.count not in {
            baseline.count + 1,
        }
        assert collection.count() == final.count
        assert all(
            manifest.is_duplicate(
                _manifest_key(config, embeddings.space, document_id)
            )
            for document_id in first.document_ids
        )
        assert all(
            not manifest.is_duplicate(
                _manifest_key(config, embeddings.space, document_id)
            )
            for document_id in second.document_ids
        )
    finally:
        store.close()
