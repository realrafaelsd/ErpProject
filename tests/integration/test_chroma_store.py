"""Integration tests for the pinned embedded Chroma persistence adapter.

These tests use ChromaDB 1.5.9 against disposable directories.  They provide
application embeddings directly and never load a model or access the network.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from threading import Event

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
from rag import ChromaVectorStore, ManifestStore


_SPACE = VectorSpace(
    model="local/integration-embedding",
    dimension=3,
    normalized=True,
    metric="cosine",
)
_PROFILE = ChunkingProfile(size=800, overlap=150, schema_version="char-v1")


def _config(root: Path) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="local-generation-model",
        chroma_path=root / "chroma",
        chroma_collection="integration_collection",
        upload_folder=root / "uploads",
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
    transaction_id: str,
    chunk_id: str,
    document_id: str,
    display_name: str,
    text: str,
    embedding: tuple[float, float, float],
    *,
    page: int = 1,
    start_offset: int = 0,
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


def _plan(
    transaction_id: str,
    chunks: Sequence[StoredChunk],
    *,
    include_manifests: bool = True,
) -> CommitPlan:
    materialized = tuple(chunks)
    manifests: list[DocumentManifest] = []
    if include_manifests:
        grouped: dict[str, list[StoredChunk]] = {}
        for stored in materialized:
            grouped.setdefault(stored.chunk.document_id, []).append(stored)
        for document_id, document_chunks in grouped.items():
            first = document_chunks[0].chunk
            manifests.append(
                DocumentManifest(
                    key=DocumentManifestKey(
                        document_id=document_id,
                        vector_fingerprint=_SPACE.fingerprint,
                        chunk_size=_PROFILE.size,
                        chunk_overlap=_PROFILE.overlap,
                        chunk_schema_version=_PROFILE.schema_version,
                    ),
                    first_display_name=first.display_name,
                    page_count=max(
                        stored.chunk.human_page for stored in document_chunks
                    ),
                    chunk_count=len(document_chunks),
                    transaction_id=transaction_id,
                )
            )

    return CommitPlan(
        transaction_id=transaction_id,
        chunks=materialized,
        manifests=tuple(manifests),
        new_chunk_ids=tuple(stored.chunk.chunk_id for stored in materialized),
        documents=len(manifests),
        pages=sum(manifest.page_count for manifest in manifests),
        warnings=(),
    )


@contextmanager
def _open_store(
    config: AppConfig,
    manifest: ManifestStore,
    *,
    recover_on_startup: bool = True,
) -> Iterator[ChromaVectorStore]:
    store = ChromaVectorStore(
        config,
        manifest,
        recover_on_startup=recover_on_startup,
    )
    try:
        yield store
    finally:
        store.close()


def _expected_metadata(stored: StoredChunk) -> dict[str, object]:
    chunk = stored.chunk
    return {
        "record_type": "chunk",
        "document_id": chunk.document_id,
        "display_name": chunk.display_name,
        "page": chunk.human_page,
        "start_offset": chunk.start_offset,
        "transaction_id": chunk.transaction_id,
        "chunk_schema_version": "char-v1",
    }


def test_pinned_cosine_collection_round_trips_add_get_query_and_reopen(
    tmp_path: Path,
) -> None:
    assert version("chromadb") == "1.5.9"
    config = _config(tmp_path)
    first = _stored_chunk(
        "tx-round-trip",
        "chunk-alpha",
        "document-alpha",
        "guias/cadastro.pdf",
        "Abra a tela Cadastro de Clientes.",
        (1.0, 0.0, 0.0),
        page=2,
        start_offset=17,
    )
    second = _stored_chunk(
        "tx-round-trip",
        "chunk-beta",
        "document-beta",
        "guias/financeiro.pdf",
        "Use o menu Financeiro para consultar títulos.",
        (0.0, 1.0, 0.0),
        page=4,
        start_offset=31,
    )
    plan = _plan("tx-round-trip", (first, second))

    manifest = ManifestStore(config)
    with _open_store(config, manifest) as store:
        store.ensure_compatible(_SPACE, _PROFILE)
        collection = store._collection  # noqa: SLF001 - integration contract
        assert collection is not None
        assert collection.configuration_json["hnsw"]["space"] == "cosine"
        assert collection.metadata == {
            "schema_version": 1,
            "embedding_model": _SPACE.model,
            "embedding_dimension": _SPACE.dimension,
            "embedding_normalized": True,
            "distance_metric": "cosine",
            "chunk_schema_version": "char-v1",
            "chunk_size": 800,
            "chunk_overlap": 150,
        }

        store.commit_chunks(plan)

        assert store.count_chunks() == 2
        assert store.existing_chunks(("chunk-alpha", "chunk-beta")) == {
            "chunk-alpha": first,
            "chunk-beta": second,
        }
        candidates = store.query((1.0, 0.0, 0.0), limit=2)
        assert [candidate.chunk_id for candidate in candidates] == [
            "chunk-alpha",
            "chunk-beta",
        ]
        assert candidates[0].document == first.chunk.text
        assert candidates[0].metadata == _expected_metadata(first)
        assert candidates[0].distance == pytest.approx(0.0, abs=1e-6)
        assert candidates[1].distance == pytest.approx(1.0, abs=1e-6)
        journal = manifest.get_journal(plan.transaction_id)
        assert journal is not None and journal.state == "COMMITTED"

    assert config.chroma_path.is_dir()
    assert any(config.chroma_path.iterdir())

    reopened_manifest = ManifestStore(config)
    with _open_store(config, reopened_manifest) as reopened:
        reopened.ensure_compatible(_SPACE, _PROFILE)

        assert reopened.count_chunks() == 2
        assert reopened.existing_chunks(("chunk-alpha", "chunk-beta")) == {
            "chunk-alpha": first,
            "chunk-beta": second,
        }
        reopened_candidates = reopened.query((1.0, 0.0, 0.0), limit=2)
        assert [candidate.chunk_id for candidate in reopened_candidates] == [
            "chunk-alpha",
            "chunk-beta",
        ]
        assert reopened_manifest.is_duplicate(plan.manifests[0].key)
        assert reopened_manifest.is_duplicate(plan.manifests[1].key)


def test_identical_recommit_is_idempotent_and_collision_preserves_record(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    original = _stored_chunk(
        "tx-original",
        "chunk-stable",
        "document-stable",
        "manual.pdf",
        "Conteúdo confirmado e imutável.",
        (1.0, 0.0, 0.0),
    )
    manifest = ManifestStore(config)

    with _open_store(config, manifest) as store:
        store.ensure_compatible(_SPACE, _PROFILE)
        store.commit_chunks(_plan("tx-original", (original,)))

        identical = _stored_chunk(
            "tx-idempotent",
            original.chunk.chunk_id,
            original.chunk.document_id,
            original.chunk.display_name,
            original.chunk.text,
            original.embedding,
            page=original.chunk.human_page,
            start_offset=original.chunk.start_offset,
        )
        store.commit_chunks(
            _plan(
                "tx-idempotent",
                (identical,),
                include_manifests=False,
            )
        )

        assert store.count_chunks() == 1
        assert store.existing_chunks((original.chunk.chunk_id,)) == {
            original.chunk.chunk_id: original
        }
        idempotent_journal = manifest.get_journal("tx-idempotent")
        assert (
            idempotent_journal is not None
            and idempotent_journal.state == "COMMITTED"
            and idempotent_journal.new_chunk_ids == ()
        )

        divergent = _stored_chunk(
            "tx-conflict",
            original.chunk.chunk_id,
            original.chunk.document_id,
            original.chunk.display_name,
            "Conteúdo divergente que não pode substituir o original.",
            original.embedding,
        )
        with pytest.raises(PublicError) as captured:
            store.commit_chunks(
                _plan(
                    "tx-conflict",
                    (divergent,),
                    include_manifests=False,
                )
            )

        assert captured.value.code == "chunk_identity_conflict"
        assert captured.value.http_status == 409
        assert manifest.get_journal("tx-conflict") is None
        assert store.count_chunks() == 1
        assert store.existing_chunks((original.chunk.chunk_id,)) == {
            original.chunk.chunk_id: original
        }


def test_partial_add_is_invisible_compensated_and_excludes_second_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    seed = _stored_chunk(
        "tx-seed",
        "chunk-seed",
        "document-seed",
        "seed.pdf",
        "Estado confirmado anterior.",
        (0.0, 0.0, 1.0),
    )
    pending = (
        _stored_chunk(
            "tx-partial",
            "chunk-new-a",
            "document-new-a",
            "new-a.pdf",
            "Primeiro registro ainda não confirmado.",
            (1.0, 0.0, 0.0),
        ),
        _stored_chunk(
            "tx-partial",
            "chunk-new-b",
            "document-new-b",
            "new-b.pdf",
            "Segundo registro ainda não confirmado.",
            (0.0, 1.0, 0.0),
        ),
    )
    partial_plan = _plan("tx-partial", pending)
    busy_plan = _plan(
        "tx-busy",
        (
            _stored_chunk(
                "tx-busy",
                "chunk-busy",
                "document-busy",
                "busy.pdf",
                "Esta transação concorrente deve ser rejeitada.",
                (1.0, 0.0, 0.0),
            ),
        ),
    )
    manifest = ManifestStore(config)

    with _open_store(config, manifest) as store:
        store.ensure_compatible(_SPACE, _PROFILE)
        store.commit_chunks(_plan("tx-seed", (seed,)))

        first_record_written = Event()
        allow_failure = Event()
        reader_started = Event()
        reader_finished = Event()
        original_add = store._add_new_records  # noqa: SLF001

        def add_one_then_fail(
            collection: object,
            chunks: Sequence[StoredChunk],
        ) -> None:
            original_add(collection, chunks[:1])  # type: ignore[arg-type]
            first_record_written.set()
            if not allow_failure.wait(timeout=5):
                raise TimeoutError("test did not release the injected failure")
            raise OSError("injected Chroma batch failure")

        def read_visible_count() -> int:
            reader_started.set()
            try:
                return store.count_chunks()
            finally:
                reader_finished.set()

        monkeypatch.setattr(store, "_add_new_records", add_one_then_fail)

        with ThreadPoolExecutor(max_workers=2) as executor:
            commit_future = executor.submit(store.commit_chunks, partial_plan)
            assert first_record_written.wait(timeout=5)
            reader_future = executor.submit(read_visible_count)
            assert reader_started.wait(timeout=5)
            try:
                assert not reader_finished.wait(timeout=0.1)
                with pytest.raises(PublicError) as busy:
                    store.commit_chunks(busy_plan)
                assert busy.value.code == "ingestion_in_progress"
                assert manifest.get_journal("tx-busy") is None
            finally:
                allow_failure.set()

            with pytest.raises(PublicError) as failed_commit:
                commit_future.result(timeout=5)
            assert failed_commit.value.code == "vector_store_write_failed"
            assert reader_future.result(timeout=5) == 1

        journal = manifest.get_journal(partial_plan.transaction_id)
        assert journal is not None and journal.state == "ABORTED"
        assert store.count_chunks() == 1
        assert store.existing_chunks(("chunk-new-a", "chunk-new-b")) == {}
        assert store.existing_chunks((seed.chunk.chunk_id,)) == {
            seed.chunk.chunk_id: seed
        }


@pytest.mark.parametrize(
    ("failure_stage", "expected_journal_state", "expected_physical_count"),
    [
        ("during_add", "PREPARED", 1),
        ("after_chroma", "CHROMA_COMMITTED", 2),
    ],
)
def test_restart_recovers_real_uncompensated_incomplete_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_journal_state: str,
    expected_physical_count: int,
) -> None:
    config = _config(tmp_path)
    chunks = (
        _stored_chunk(
            "tx-recovery",
            "chunk-recovery-a",
            "document-recovery-a",
            "recovery-a.pdf",
            "Registro temporário A.",
            (1.0, 0.0, 0.0),
        ),
        _stored_chunk(
            "tx-recovery",
            "chunk-recovery-b",
            "document-recovery-b",
            "recovery-b.pdf",
            "Registro temporário B.",
            (0.0, 1.0, 0.0),
        ),
    )
    plan = _plan("tx-recovery", chunks)
    manifest = ManifestStore(config)

    with _open_store(config, manifest) as store:
        store.ensure_compatible(_SPACE, _PROFILE)
        original_add = store._add_new_records  # noqa: SLF001

        if failure_stage == "during_add":

            def fail_during_add(
                collection: object,
                records: Sequence[StoredChunk],
            ) -> None:
                original_add(collection, records[:1])  # type: ignore[arg-type]
                raise OSError("injected partial add failure")

            monkeypatch.setattr(store, "_add_new_records", fail_during_add)
        else:

            def fail_manifest_commit(
                transaction_id: str,
                manifests: Sequence[DocumentManifest],
            ) -> None:
                raise OSError("injected manifest finalization failure")

            monkeypatch.setattr(
                manifest,
                "commit_manifests",
                fail_manifest_commit,
            )

        def fail_compensation(
            collection: object,
            ids: Sequence[str],
        ) -> None:
            raise OSError("injected compensation failure")

        monkeypatch.setattr(store, "_delete_explicit_ids", fail_compensation)

        with pytest.raises(PublicError) as captured:
            store.commit_chunks(plan)

        assert captured.value.code == "recovery_required"
        assert manifest.requires_recovery
        journal = manifest.get_journal(plan.transaction_id)
        assert journal is not None and journal.state == expected_journal_state
        collection = store._collection  # noqa: SLF001 - inspect crash residue
        assert collection is not None
        assert collection.count() == expected_physical_count
        with pytest.raises(PublicError) as blocked:
            store.count_chunks()
        assert blocked.value.code == "recovery_required"

    restarted_manifest = ManifestStore(config)
    assert restarted_manifest.requires_recovery
    with _open_store(config, restarted_manifest) as recovered:
        assert not restarted_manifest.requires_recovery
        recovered.ensure_compatible(_SPACE, _PROFILE)

        assert recovered.count_chunks() == 0
        assert recovered.existing_chunks(
            ("chunk-recovery-a", "chunk-recovery-b")
        ) == {}
        recovered_journal = restarted_manifest.get_journal(
            plan.transaction_id
        )
        assert recovered_journal is not None
        assert recovered_journal.state == "ABORTED"
