"""Property test for observable ingestion-state atomicity."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, RLock
from typing import Literal, cast

import pytest
from hypothesis import given, settings, strategies as st

from domain import (
    AppConfig,
    Chunk,
    ChunkingProfile,
    CommitPlan,
    DocumentManifest,
    DocumentManifestKey,
    PublicError,
    StoredChunk,
    TransactionJournal,
    VectorSpace,
)
from rag import ChromaVectorStore, StorageLocks


_SPACE = VectorSpace(
    model="local/property-atomicity",
    dimension=3,
    normalized=True,
    metric="cosine",
)
_PROFILE = ChunkingProfile(size=800, overlap=150, schema_version="char-v1")
_FAILURE_POINTS = (
    "commit",
    "before_write",
    "partial_write",
    "after_write",
    "journal_transition",
    "manifest_commit",
    "collision",
)
FailurePoint = Literal[
    "commit",
    "before_write",
    "partial_write",
    "after_write",
    "journal_transition",
    "manifest_commit",
    "collision",
]
_TEXT = st.text(
    alphabet=st.sampled_from(tuple("ABCDE abcde0123áçõ-_/")),
    min_size=1,
    max_size=24,
)
_EMBEDDING = st.tuples(
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10),
).map(lambda values: tuple(value / 10.0 for value in values))


@dataclass(frozen=True, slots=True)
class _AtomicCase:
    initial_chunks: tuple[StoredChunk, ...]
    initial_manifests: tuple[DocumentManifest, ...]
    plan: CommitPlan
    new_chunks: tuple[StoredChunk, ...]
    new_manifests: tuple[DocumentManifest, ...]
    failure_point: FailurePoint
    partial_count: int


@dataclass(slots=True)
class _MemoryJournal:
    state: str
    new_chunk_ids: tuple[str, ...]
    manifests: tuple[DocumentManifest, ...]


@dataclass(frozen=True, slots=True)
class _ObservedState:
    count: int
    chunks: Mapping[str, StoredChunk]


def _config() -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_atomicity",
        upload_folder=Path("/unused/uploads"),
        embedding_model=_SPACE.model,
        top_k=6,
        chunk_size=_PROFILE.size,
        chunk_overlap=_PROFILE.overlap,
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


def _stored_chunk(
    *,
    transaction_id: str,
    chunk_id: str,
    document_id: str,
    display_name: str,
    text: str,
    embedding: tuple[float, float, float],
    page: int,
    start_offset: int,
) -> StoredChunk:
    return StoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            display_name=display_name,
            human_page=page,
            start_offset=start_offset,
            text=text,
            transaction_id=transaction_id,
        ),
        embedding=embedding,
    )


def _manifest(stored: StoredChunk, transaction_id: str) -> DocumentManifest:
    return DocumentManifest(
        key=DocumentManifestKey(
            document_id=stored.chunk.document_id,
            vector_fingerprint=_SPACE.fingerprint,
            chunk_size=_PROFILE.size,
            chunk_overlap=_PROFILE.overlap,
            chunk_schema_version=_PROFILE.schema_version,
        ),
        first_display_name=stored.chunk.display_name,
        page_count=stored.chunk.human_page,
        chunk_count=1,
        transaction_id=transaction_id,
    )


def _plan(
    transaction_id: str,
    chunks: Sequence[StoredChunk],
    manifests: Sequence[DocumentManifest],
) -> CommitPlan:
    materialized_chunks = tuple(chunks)
    materialized_manifests = tuple(manifests)
    return CommitPlan(
        transaction_id=transaction_id,
        chunks=materialized_chunks,
        manifests=materialized_manifests,
        new_chunk_ids=tuple(
            stored.chunk.chunk_id for stored in materialized_chunks
        ),
        documents=len(materialized_manifests),
        pages=sum(manifest.page_count for manifest in materialized_manifests),
        warnings=(),
    )


def _for_transaction(stored: StoredChunk, transaction_id: str) -> StoredChunk:
    return replace(
        stored,
        chunk=replace(stored.chunk, transaction_id=transaction_id),
    )


def _divergent_collision(
    stored: StoredChunk,
    transaction_id: str,
    field: str,
) -> StoredChunk:
    chunk = replace(stored.chunk, transaction_id=transaction_id)
    if field == "text":
        chunk = replace(chunk, text=f"{chunk.text} divergente")
    elif field == "document_id":
        chunk = replace(chunk, document_id=f"{chunk.document_id}-divergente")
    elif field == "display_name":
        chunk = replace(chunk, display_name=f"outro/{chunk.display_name}")
    else:
        chunk = replace(chunk, human_page=chunk.human_page + 1)
    return replace(stored, chunk=chunk)


@st.composite
def _atomic_cases(draw):
    initial_count = draw(st.integers(min_value=1, max_value=3))
    new_count = draw(st.integers(min_value=2, max_value=4))
    reused_count = draw(st.integers(min_value=0, max_value=initial_count))
    failure_point = cast(FailurePoint, draw(st.sampled_from(_FAILURE_POINTS)))
    total = initial_count + new_count
    texts = draw(st.lists(_TEXT, min_size=total, max_size=total))
    embeddings = draw(
        st.lists(_EMBEDDING, min_size=total, max_size=total)
    )
    pages = draw(
        st.lists(
            st.integers(min_value=1, max_value=8),
            min_size=total,
            max_size=total,
        )
    )
    offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=2_000),
            min_size=total,
            max_size=total,
        )
    )

    initial_chunks = tuple(
        _stored_chunk(
            transaction_id="tx-seed",
            chunk_id=f"chunk-seed-{index}",
            document_id=f"document-seed-{index}",
            display_name=f"base/manual-{index}.pdf",
            text=texts[index],
            embedding=embeddings[index],
            page=pages[index],
            start_offset=offsets[index],
        )
        for index in range(initial_count)
    )
    new_chunks = tuple(
        _stored_chunk(
            transaction_id="tx-candidate",
            chunk_id=f"chunk-new-{index}",
            document_id=f"document-new-{index}",
            display_name=f"novos/manual-{index}.pdf",
            text=texts[initial_count + index],
            embedding=embeddings[initial_count + index],
            page=pages[initial_count + index],
            start_offset=offsets[initial_count + index],
        )
        for index in range(new_count)
    )
    initial_manifests = tuple(
        _manifest(stored, "tx-seed") for stored in initial_chunks
    )
    new_manifests = tuple(
        _manifest(stored, "tx-candidate") for stored in new_chunks
    )

    reused = tuple(
        _for_transaction(stored, "tx-candidate")
        for stored in initial_chunks[:reused_count]
    )
    if failure_point == "collision":
        collision_field = draw(
            st.sampled_from(("text", "document_id", "display_name", "page"))
        )
        collision = _divergent_collision(
            initial_chunks[0],
            "tx-candidate",
            collision_field,
        )
        reused_without_collision = tuple(
            stored
            for stored in reused
            if stored.chunk.chunk_id != collision.chunk.chunk_id
        )
        candidate_chunks = (collision, *reused_without_collision, *new_chunks)
    else:
        candidate_chunks = (*reused, *new_chunks)

    partial_count = (
        draw(st.integers(min_value=1, max_value=new_count - 1))
        if failure_point == "partial_write"
        else 1
    )
    return _AtomicCase(
        initial_chunks=initial_chunks,
        initial_manifests=initial_manifests,
        plan=_plan("tx-candidate", candidate_chunks, new_manifests),
        new_chunks=new_chunks,
        new_manifests=new_manifests,
        failure_point=failure_point,
        partial_count=partial_count,
    )


class _MemoryManifestStore:
    """In-memory journal with the same externally visible state transitions."""

    def __init__(self, case: _AtomicCase) -> None:
        self.requires_recovery = False
        self._failure_point = case.failure_point
        self._lock = RLock()
        self._documents = {
            manifest.key: manifest for manifest in case.initial_manifests
        }
        self._owners = {
            manifest.key: manifest.transaction_id
            for manifest in case.initial_manifests
        }
        self._journals: dict[str, _MemoryJournal] = {}

    def get_vector_space(self) -> tuple[VectorSpace, ChunkingProfile]:
        return _SPACE, _PROFILE

    def prepare(self, plan: CommitPlan) -> TransactionJournal:
        with self._lock:
            assert plan.transaction_id not in self._journals
            journal = _MemoryJournal(
                state="PREPARED",
                new_chunk_ids=tuple(plan.new_chunk_ids),
                manifests=tuple(plan.manifests),
            )
            self._journals[plan.transaction_id] = journal
            return TransactionJournal(
                transaction_id=plan.transaction_id,
                state="PREPARED",
                new_chunk_ids=journal.new_chunk_ids,
                plan_checksum="in-memory-property-checksum",
            )

    def mark_chroma_committed(self, transaction_id: str) -> None:
        with self._lock:
            journal = self._journals[transaction_id]
            assert journal.state == "PREPARED"
            journal.state = "CHROMA_COMMITTED"
            if self._failure_point == "journal_transition":
                raise OSError("injected journal transition failure")

    def commit_manifests(
        self,
        transaction_id: str,
        manifests: Sequence[DocumentManifest],
    ) -> None:
        with self._lock:
            journal = self._journals[transaction_id]
            assert journal.state == "CHROMA_COMMITTED"
            materialized = tuple(manifests)
            if self._failure_point == "manifest_commit":
                for manifest in materialized[:1]:
                    self._documents[manifest.key] = manifest
                    self._owners[manifest.key] = transaction_id
                raise OSError("injected partial manifest failure")
            for manifest in materialized:
                assert manifest.key not in self._documents
                self._documents[manifest.key] = manifest
                self._owners[manifest.key] = transaction_id
            journal.state = "COMMITTED"

    def abort(self, transaction_id: str) -> None:
        with self._lock:
            journal = self._journals[transaction_id]
            assert journal.state in {"PREPARED", "CHROMA_COMMITTED"}
            for key, owner in tuple(self._owners.items()):
                if owner == transaction_id:
                    self._owners.pop(key)
                    self._documents.pop(key)
            journal.state = "ABORTED"

    def _set_recovery_pending(self, pending: bool) -> None:
        self.requires_recovery = pending

    def snapshot(self) -> dict[DocumentManifestKey, DocumentManifest]:
        with self._lock:
            return dict(self._documents)

    def journal(self, transaction_id: str) -> _MemoryJournal | None:
        with self._lock:
            return self._journals.get(transaction_id)


class _MemoryCollection:
    """Columnar collection fake with failure injection inside one write call."""

    def __init__(self, case: _AtomicCase) -> None:
        self.metadata = ChromaVectorStore._collection_metadata(  # noqa: SLF001
            _SPACE,
            _PROFILE,
        )
        self.configuration_json = {"hnsw": {"space": "cosine"}}
        self._failure_point = case.failure_point
        self._partial_count = case.partial_count
        self._records = {
            stored.chunk.chunk_id: stored for stored in case.initial_chunks
        }
        self._lock = RLock()
        self.probe: Callable[[Mapping[str, StoredChunk]], None] | None = None

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def get(self, **kwargs: object) -> object:
        requested = cast(Sequence[str], kwargs["ids"])
        with self._lock:
            records = [
                self._records[chunk_id]
                for chunk_id in requested
                if chunk_id in self._records
            ]
        return {
            "ids": [stored.chunk.chunk_id for stored in records],
            "documents": [stored.chunk.text for stored in records],
            "metadatas": [
                ChromaVectorStore._record_metadata(stored)  # noqa: SLF001
                for stored in records
            ],
            "embeddings": [list(stored.embedding) for stored in records],
        }

    def query(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("atomicity property does not query by similarity")

    def add(self, **kwargs: object) -> None:
        ids = list(cast(Sequence[str], kwargs["ids"]))
        embeddings = list(cast(Sequence[Sequence[float]], kwargs["embeddings"]))
        metadatas = list(
            cast(Sequence[Mapping[str, object]], kwargs["metadatas"])
        )
        documents = list(cast(Sequence[str], kwargs["documents"]))

        if self._failure_point == "before_write":
            self._publish_probe()
            raise OSError("injected failure before vector write")

        write_count = (
            self._partial_count
            if self._failure_point == "partial_write"
            else len(ids)
        )
        with self._lock:
            for chunk_id, embedding, metadata, document in zip(
                ids[:write_count],
                embeddings[:write_count],
                metadatas[:write_count],
                documents[:write_count],
                strict=True,
            ):
                assert chunk_id not in self._records
                self._records[chunk_id] = StoredChunk(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        document_id=cast(str, metadata["document_id"]),
                        display_name=cast(str, metadata["display_name"]),
                        human_page=cast(int, metadata["page"]),
                        start_offset=cast(int, metadata["start_offset"]),
                        text=document,
                        transaction_id=cast(str, metadata["transaction_id"]),
                    ),
                    embedding=tuple(float(value) for value in embedding),
                )

        self._publish_probe()
        if self._failure_point == "partial_write":
            raise OSError("injected partial vector write failure")
        if self._failure_point == "after_write":
            raise OSError("injected failure after vector write")

    def delete(self, **kwargs: object) -> object:
        ids = cast(Sequence[str], kwargs["ids"])
        with self._lock:
            for chunk_id in ids:
                self._records.pop(chunk_id, None)
        return None

    def snapshot(self) -> dict[str, StoredChunk]:
        with self._lock:
            return dict(self._records)

    def _publish_probe(self) -> None:
        if self.probe is not None:
            self.probe(self.snapshot())


class _MemoryClient:
    def __init__(self, collection: _MemoryCollection) -> None:
        self._collection = collection

    def get_collection(self, **kwargs: object) -> _MemoryCollection:
        del kwargs
        return self._collection

    def get_or_create_collection(self, **kwargs: object) -> _MemoryCollection:
        del kwargs
        return self._collection

    def get_max_batch_size(self) -> int:
        return 1_000


class _VisibilityProbe:
    """Attempt a real read while the production visibility lock is exclusive."""

    def __init__(self, store: ChromaVectorStore, ids: Sequence[str]) -> None:
        self._store = store
        self._ids = tuple(ids)
        self._trigger = Event()
        self._attempted = Event()
        self._completed = Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future[_ObservedState] = self._executor.submit(self._read)
        self.blocked: list[bool] = []
        self.physical_snapshots: list[Mapping[str, StoredChunk]] = []

    def checkpoint(self, physical: Mapping[str, StoredChunk]) -> None:
        self.physical_snapshots.append(dict(physical))
        self._trigger.set()
        assert self._attempted.wait(timeout=2)
        self.blocked.append(not self._completed.wait(timeout=0.02))

    def result(self) -> _ObservedState:
        return self._future.result(timeout=2)

    def close(self) -> None:
        self._trigger.set()
        self._executor.shutdown(wait=True)

    def _read(self) -> _ObservedState:
        if not self._trigger.wait(timeout=2):
            raise AssertionError("commit never reached the visibility checkpoint")
        self._attempted.set()
        try:
            return _observe(self._store, self._ids)
        finally:
            self._completed.set()


def _memory_store(
    case: _AtomicCase,
) -> tuple[ChromaVectorStore, _MemoryManifestStore, _MemoryCollection]:
    manifest = _MemoryManifestStore(case)
    collection = _MemoryCollection(case)
    client = _MemoryClient(collection)
    locks = StorageLocks(Path("/unused/property-atomicity"))

    store = object.__new__(ChromaVectorStore)
    store._config = _config()  # noqa: SLF001
    store._manifest_store = manifest  # type: ignore[assignment]  # noqa: SLF001
    store.locks = locks
    store.lifecycle_lock = locks.lifecycle_lock
    store.ingestion_mutex = locks.ingestion_mutex
    store.visibility_lock = locks.visibility_lock
    store._compatibility_lock = RLock()  # noqa: SLF001
    store._compatible_space = _SPACE  # noqa: SLF001
    store._compatible_profile = _PROFILE  # noqa: SLF001
    store._collection = collection  # type: ignore[assignment]  # noqa: SLF001
    store._client = client  # type: ignore[assignment]  # noqa: SLF001
    return store, manifest, collection


def _observe(store: ChromaVectorStore, ids: Sequence[str]) -> _ObservedState:
    return _ObservedState(
        count=store.count_chunks(),
        chunks=dict(store.existing_chunks(ids)),
    )


@settings(max_examples=120, deadline=None)
@given(case=_atomic_cases())
def test_property_12_confirmation_or_failure_preserves_state_atomicity(
    case: _AtomicCase,
) -> None:
    # Feature: erp-ai-support, Property 12: Confirmação ou falha preserva atomicidade do estado
    # **Validates: Requirements 11.5, 11.6, 11.7, 11.12, 11.13, 16.14**
    store, manifest, collection = _memory_store(case)
    initial_chunks = {
        stored.chunk.chunk_id: stored for stored in case.initial_chunks
    }
    expected_committed_chunks = {
        **initial_chunks,
        **{stored.chunk.chunk_id: stored for stored in case.new_chunks},
    }
    initial_manifests = {
        item.key: item for item in case.initial_manifests
    }
    expected_committed_manifests = {
        **initial_manifests,
        **{item.key: item for item in case.new_manifests},
    }
    all_ids = tuple(
        dict.fromkeys(
            (*initial_chunks, *(stored.chunk.chunk_id for stored in case.new_chunks))
        )
    )

    if case.failure_point == "collision":
        with pytest.raises(PublicError) as captured:
            store.commit_chunks(case.plan)

        assert captured.value.code == "chunk_identity_conflict"
        assert captured.value.http_status == 409
        assert _observe(store, all_ids) == _ObservedState(
            count=len(initial_chunks),
            chunks=initial_chunks,
        )
        assert collection.snapshot() == initial_chunks
        assert manifest.snapshot() == initial_manifests
        assert manifest.journal(case.plan.transaction_id) is None
        return

    probe = _VisibilityProbe(store, all_ids)
    collection.probe = probe.checkpoint
    try:
        if case.failure_point == "commit":
            store.commit_chunks(case.plan)
        else:
            with pytest.raises(PublicError) as captured:
                store.commit_chunks(case.plan)
            assert captured.value.code == "vector_store_write_failed"
            assert captured.value.http_status == 503
        concurrent_observation = probe.result()
    finally:
        probe.close()

    assert probe.blocked == [True]
    assert len(probe.physical_snapshots) == 1
    if case.failure_point == "before_write":
        assert probe.physical_snapshots[0] == initial_chunks
    elif case.failure_point == "partial_write":
        expected_partial = {
            **initial_chunks,
            **{
                stored.chunk.chunk_id: stored
                for stored in case.new_chunks[: case.partial_count]
            },
        }
        assert initial_chunks.keys() < expected_partial.keys()
        assert expected_partial.keys() < expected_committed_chunks.keys()
        assert probe.physical_snapshots[0] == expected_partial
    else:
        assert probe.physical_snapshots[0] == expected_committed_chunks

    journal = manifest.journal(case.plan.transaction_id)
    assert journal is not None
    assert journal.new_chunk_ids == tuple(
        stored.chunk.chunk_id for stored in case.new_chunks
    )

    if case.failure_point == "commit":
        expected_state = _ObservedState(
            count=len(expected_committed_chunks),
            chunks=expected_committed_chunks,
        )
        assert journal.state == "COMMITTED"
        assert manifest.snapshot() == expected_committed_manifests
    else:
        expected_state = _ObservedState(
            count=len(initial_chunks),
            chunks=initial_chunks,
        )
        assert journal.state == "ABORTED"
        assert manifest.snapshot() == initial_manifests

    assert concurrent_observation == expected_state
    assert _observe(store, all_ids) == expected_state
    assert collection.snapshot() == dict(expected_state.chunks)
