"""Focused orchestration tests for the complete ingestion transaction."""

from __future__ import annotations

import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence
from uuid import UUID, uuid4

import pymupdf
import pytest

from domain import (
    AppConfig,
    Chunk,
    CommitPlan,
    DocumentManifestKey,
    PublicError,
    StoredChunk,
    VectorSpace,
)
from ingest import IngestionService, make_chunk_id


class _EmbeddingProvider:
    def __init__(self, model: str) -> None:
        self.space = VectorSpace(
            model=model,
            dimension=3,
            normalized=True,
            metric="cosine",
        )
        self.ready_calls = 0
        self.document_calls: list[tuple[str, ...]] = []

    def ensure_ready(self) -> VectorSpace:
        self.ready_calls += 1
        return self.space

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        materialized = tuple(texts)
        self.document_calls.append(materialized)
        return [[1.0, 0.0, 0.0] for _ in materialized]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("ingestion must not embed a query")


class _Manifest:
    def __init__(self) -> None:
        self.committed: set[str] = set()
        self.race_duplicates: set[str] = set()
        self.calls: list[tuple[str, bool]] = []
        self.guard_active: Callable[[], bool] = lambda: False

    def is_duplicate(self, key: DocumentManifestKey) -> bool:
        active = self.guard_active()
        self.calls.append((key.document_id, active))
        return key.document_id in self.committed or (
            active and key.document_id in self.race_duplicates
        )


class _Guard:
    def __init__(self, store: "_VectorStore") -> None:
        self._store = store

    def __enter__(self) -> None:
        if self._store.reject_guard:
            raise PublicError(
                code="ingestion_in_progress",
                message=(
                    "Outra importação está modificando a base de conhecimento. "
                    "Tente novamente após a conclusão."
                ),
                http_status=409,
            )
        assert not self._store.guard_is_active
        self._store.guard_is_active = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._store.guard_is_active = False
        return False


class _VectorStore:
    def __init__(self) -> None:
        self.guard_is_active = False
        self.reject_guard = False
        self.existing: dict[str, StoredChunk] = {}
        self.compatibility_calls: list[tuple[VectorSpace, object]] = []
        self.existing_calls: list[tuple[str, ...]] = []
        self.commits: list[CommitPlan] = []
        self.commit_observer: Callable[[], None] | None = None

    def ingestion_guard(self) -> _Guard:
        return _Guard(self)

    def ensure_compatible(self, space: VectorSpace, profile: object) -> None:
        assert self.guard_is_active
        self.compatibility_calls.append((space, profile))

    def existing_chunks(
        self, ids: Sequence[str]
    ) -> Mapping[str, StoredChunk]:
        assert self.guard_is_active
        materialized = tuple(ids)
        self.existing_calls.append(materialized)
        return {
            chunk_id: self.existing[chunk_id]
            for chunk_id in materialized
            if chunk_id in self.existing
        }

    def commit_chunks(self, plan: CommitPlan) -> None:
        assert self.guard_is_active
        if self.commit_observer is not None:
            self.commit_observer()
        self.commits.append(plan)

    def count_chunks(self) -> int:
        return len(self.existing)

    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self.existing.pop(chunk_id, None)

    def query(self, embedding: Sequence[float], limit: int) -> list[object]:
        raise AssertionError("ingestion must not query")


def _config(root: Path) -> AppConfig:
    upload_folder = root / "uploads"
    upload_folder.mkdir()
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generator",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=upload_folder,
        embedding_model="local/test-embedding",
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


def _pdf_bytes(*page_texts: str | None) -> bytes:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text is not None:
                page.insert_text((72, 72), text)
        return document.tobytes()


def _archive(
    config: AppConfig,
    upload_id: UUID,
    entries: Sequence[tuple[str, bytes]],
) -> tuple[Path, Path]:
    staging = config.upload_folder / f"upload-{upload_id}"
    staging.mkdir(mode=0o700)
    archive = staging / "knowledge.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, content in entries:
            output.writestr(name, content)
    return archive, staging


def _service(
    config: AppConfig,
    embeddings: _EmbeddingProvider,
    manifest: _Manifest,
    vector: _VectorStore,
) -> IngestionService:
    manifest.guard_active = lambda: vector.guard_is_active
    return IngestionService(config, embeddings, manifest, vector)


def test_orchestrates_batch_dedup_empty_pdf_warnings_and_cleanup(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _VectorStore()
    first_pdf = _pdf_bytes("Procedimento Alpha")
    same_name_different_bytes = _pdf_bytes("Procedimento Beta")
    empty_pdf = _pdf_bytes(None)
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (
            ("caf\u00e9.pdf", first_pdf),
            ("cafe\u0301.pdf", same_name_different_bytes),
            ("copias/repetido.pdf", first_pdf),
            ("digitalizado.pdf", empty_pdf),
            ("notas.txt", b"ignorado"),
        ),
    )
    vector.commit_observer = lambda: (
        None if not staging.exists() else pytest.fail("staging survived until commit")
    )

    result = _service(config, embeddings, manifest, vector).ingest(
        archive, upload_id
    )

    assert result.success is True
    assert (result.documents, result.pages, result.chunks) == (3, 3, 2)
    assert not staging.exists()
    assert embeddings.ready_calls == 1
    assert len(embeddings.document_calls) == 1
    assert len(embeddings.document_calls[0]) == 2
    assert len(vector.commits) == 1
    plan = vector.commits[0]
    assert (plan.documents, plan.pages, len(plan.new_chunk_ids)) == (3, 3, 2)
    assert [manifest.first_display_name for manifest in plan.manifests] == [
        "caf\u00e9.pdf",
        "caf\u00e9.pdf",
        "digitalizado.pdf",
    ]
    assert [manifest.chunk_count for manifest in plan.manifests] == [1, 1, 0]
    assert len({manifest.key.document_id for manifest in plan.manifests}) == 3
    assert any("notas.txt: arquivo ignorado" in warning for warning in result.warnings)
    assert any(
        "copias/repetido.pdf: documento duplicado" in warning
        for warning in result.warnings
    )
    assert any(
        "digitalizado.pdf, página 1" in warning and "OCR" in warning
        for warning in result.warnings
    )
    assert len(result.warnings) == 3
    assert plan.warnings == result.warnings
    assert all(active is False for _, active in manifest.calls[:4])
    assert all(active is True for _, active in manifest.calls[4:])


def test_confirmed_duplicate_skips_pdf_embeddings_and_commit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _VectorStore()
    pdf = _pdf_bytes("Documento já confirmado")
    document_id = sha256(pdf).hexdigest()
    manifest.committed.add(document_id)
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (("outra-pasta/manual.pdf", pdf),),
    )

    result = _service(config, embeddings, manifest, vector).ingest(
        archive, upload_id
    )

    assert (result.documents, result.pages, result.chunks) == (0, 0, 0)
    assert result.warnings == (
        "outra-pasta/manual.pdf: documento duplicado; nenhuma nova "
        "indexação foi realizada.",
    )
    assert embeddings.document_calls == []
    assert vector.existing_calls == []
    assert vector.commits == []
    assert len(vector.compatibility_calls) == 1
    assert not staging.exists()


def test_revalidates_duplicate_under_guard_and_discards_prepared_document(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _VectorStore()
    pdf = _pdf_bytes("Documento confirmado por transação concorrente")
    document_id = sha256(pdf).hexdigest()
    manifest.race_duplicates.add(document_id)
    upload_id = uuid4()
    archive, staging = _archive(config, upload_id, (("manual.pdf", pdf),))

    result = _service(config, embeddings, manifest, vector).ingest(
        archive, upload_id
    )

    assert manifest.calls == [(document_id, False), (document_id, True)]
    assert len(embeddings.document_calls) == 1
    assert (result.documents, result.pages, result.chunks) == (0, 0, 0)
    assert result.warnings == (
        "manual.pdf: documento duplicado; nenhuma nova indexação foi realizada.",
    )
    assert vector.existing_calls == []
    assert vector.commits == []
    assert not staging.exists()


def test_busy_ingestion_returns_409_after_preparation_without_persisting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _VectorStore()
    vector.reject_guard = True
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (("manual.pdf", _pdf_bytes("Conteúdo preparado")),),
    )

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == "ingestion_in_progress"
    assert captured.value.http_status == 409
    assert len(embeddings.document_calls) == 1
    assert vector.compatibility_calls == []
    assert vector.commits == []
    assert not staging.exists()


def test_divergent_chunk_collision_is_rejected_under_guard(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _VectorStore()
    pdf = _pdf_bytes("Conteúdo novo")
    document_id = sha256(pdf).hexdigest()
    chunk_id = make_chunk_id(document_id, 1, 0)
    vector.existing[chunk_id] = StoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            display_name="manual.pdf",
            human_page=1,
            start_offset=0,
            text="conteúdo divergente",
            transaction_id="tx-confirmada",
        ),
        embedding=(1.0, 0.0, 0.0),
    )
    upload_id = uuid4()
    archive, staging = _archive(config, upload_id, (("manual.pdf", pdf),))

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == "chunk_identity_conflict"
    assert captured.value.http_status == 409
    assert vector.existing_calls == [(chunk_id,)]
    assert vector.commits == []
    assert not staging.exists()


class _StatefulVectorStore(_VectorStore):
    """In-memory atomic store used to observe confirmed orchestration state."""

    def __init__(self, manifest: _Manifest) -> None:
        super().__init__()
        self._manifest = manifest
        self.compatibility_error: Exception | None = None
        self.lookup_error: Exception | None = None
        self.commit_error: Exception | None = None
        self.commit_attempts: list[CommitPlan] = []
        self.upsert_calls: list[tuple[str, ...]] = []

    def ensure_compatible(self, space: VectorSpace, profile: object) -> None:
        assert self.guard_is_active
        self.compatibility_calls.append((space, profile))
        if self.compatibility_error is not None:
            raise self.compatibility_error

    def existing_chunks(
        self, ids: Sequence[str]
    ) -> Mapping[str, StoredChunk]:
        assert self.guard_is_active
        materialized = tuple(ids)
        self.existing_calls.append(materialized)
        if self.lookup_error is not None:
            raise self.lookup_error
        return {
            chunk_id: self.existing[chunk_id]
            for chunk_id in materialized
            if chunk_id in self.existing
        }

    def commit_chunks(self, plan: CommitPlan) -> None:
        assert self.guard_is_active
        self.commit_attempts.append(plan)
        if self.commit_observer is not None:
            self.commit_observer()
        if self.commit_error is not None:
            raise self.commit_error

        # Build the next state before publishing either vectors or manifests,
        # mirroring the atomic contract expected from the concrete adapter.
        next_existing = dict(self.existing)
        for stored in plan.chunks:
            next_existing[stored.chunk.chunk_id] = stored

        self.existing = next_existing
        self._manifest.committed.update(
            item.key.document_id for item in plan.manifests
        )
        if plan.new_chunk_ids:
            self.upsert_calls.append(tuple(plan.new_chunk_ids))
        self.commits.append(plan)


class _FailingEmbeddingProvider(_EmbeddingProvider):
    """Embedding fake that fails at one selected orchestration phase."""

    def __init__(self, model: str, phase: str) -> None:
        super().__init__(model)
        self._phase = phase

    def ensure_ready(self) -> VectorSpace:
        if self._phase == "ready":
            self.ready_calls += 1
            raise PublicError(
                code="embedding_model_missing",
                message=(
                    f"O modelo de embeddings {self.space.model} não está "
                    "disponível localmente."
                ),
                http_status=503,
            )
        return super().ensure_ready()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._phase == "documents":
            materialized = tuple(texts)
            self.document_calls.append(materialized)
            raise RuntimeError("falha técnica de embedding")
        return super().embed_documents(texts)


def _raw_archive(
    config: AppConfig,
    upload_id: UUID,
    content: bytes,
) -> tuple[Path, Path]:
    staging = config.upload_folder / f"upload-{upload_id}"
    staging.mkdir(mode=0o700)
    archive = staging / "knowledge.zip"
    archive.write_bytes(content)
    return archive, staging


def _seed_confirmed_state(
    manifest: _Manifest,
    vector: _StatefulVectorStore,
) -> None:
    document_id = sha256(b"estado confirmado anterior").hexdigest()
    chunk_id = make_chunk_id(document_id, 1, 0)
    manifest.committed.add(document_id)
    vector.existing[chunk_id] = StoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            display_name="anterior.pdf",
            human_page=1,
            start_offset=0,
            text="Conhecimento confirmado anteriormente.",
            transaction_id="tx-anterior",
        ),
        embedding=(0.0, 1.0, 0.0),
    )


def _confirmed_snapshot(
    manifest: _Manifest,
    vector: _StatefulVectorStore,
) -> tuple[frozenset[str], dict[str, StoredChunk]]:
    return frozenset(manifest.committed), dict(vector.existing)


def test_repeated_upload_is_deduplicated_without_new_embeddings_or_upserts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    service = _service(config, embeddings, manifest, vector)
    pdf = _pdf_bytes("Cadastro de clientes no menu Cadastros")

    first_id = uuid4()
    first_archive, first_staging = _archive(
        config,
        first_id,
        (("originais/manual.pdf", pdf),),
    )
    first = service.ingest(first_archive, first_id)
    state_after_first = _confirmed_snapshot(manifest, vector)
    calls_after_first = tuple(embeddings.document_calls)
    upserts_after_first = tuple(vector.upsert_calls)

    second_id = uuid4()
    second_archive, second_staging = _archive(
        config,
        second_id,
        (("copias/manual-renomeado.pdf", pdf),),
    )
    second = service.ingest(second_archive, second_id)

    assert (first.documents, first.pages, first.chunks) == (1, 1, 1)
    assert first.warnings == ()
    assert (second.documents, second.pages, second.chunks) == (0, 0, 0)
    assert second.warnings == (
        "copias/manual-renomeado.pdf: documento duplicado; nenhuma nova "
        "indexação foi realizada.",
    )
    assert _confirmed_snapshot(manifest, vector) == state_after_first
    assert tuple(embeddings.document_calls) == calls_after_first
    assert tuple(vector.upsert_calls) == upserts_after_first
    assert len(embeddings.document_calls) == 1
    assert len(vector.upsert_calls) == 1
    assert len(vector.commits) == 1
    assert sha256(pdf).hexdigest() in manifest.committed
    assert not first_staging.exists()
    assert not second_staging.exists()


def test_zero_chunk_document_is_manifested_and_later_skipped_without_upsert(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    service = _service(config, embeddings, manifest, vector)
    pdf = _pdf_bytes(None)

    first_id = uuid4()
    first_archive, first_staging = _archive(
        config,
        first_id,
        (("digitalizado.pdf", pdf),),
    )
    first = service.ingest(first_archive, first_id)
    state_after_first = _confirmed_snapshot(manifest, vector)

    second_id = uuid4()
    second_archive, second_staging = _archive(
        config,
        second_id,
        (("novamente/digitalizado.pdf", pdf),),
    )
    second = service.ingest(second_archive, second_id)

    assert (first.documents, first.pages, first.chunks) == (1, 1, 0)
    assert len(first.warnings) == 1
    assert "digitalizado.pdf, página 1" in first.warnings[0]
    assert "OCR" in first.warnings[0]
    assert (second.documents, second.pages, second.chunks) == (0, 0, 0)
    assert second.warnings == (
        "novamente/digitalizado.pdf: documento duplicado; nenhuma nova "
        "indexação foi realizada.",
    )
    assert embeddings.document_calls == []
    assert vector.upsert_calls == []
    assert len(vector.commits) == 1
    assert vector.commits[0].chunks == ()
    assert len(vector.commits[0].manifests) == 1
    assert _confirmed_snapshot(manifest, vector) == state_after_first
    assert not first_staging.exists()
    assert not second_staging.exists()


def test_unreadable_pdf_warns_while_valid_sibling_is_confirmed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    valid_pdf = _pdf_bytes("Procedimento válido")
    unreadable_pdf = b"this is not a PDF container"
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (
            ("corrompido.pdf", unreadable_pdf),
            ("manual.pdf", valid_pdf),
        ),
    )

    result = _service(config, embeddings, manifest, vector).ingest(
        archive, upload_id
    )

    assert (result.documents, result.pages, result.chunks) == (1, 1, 1)
    assert result.warnings == (
        "corrompido.pdf: arquivo ignorado por falha de leitura; substitua-o "
        "ou remova-o antes de reenviar.",
    )
    assert len(embeddings.document_calls) == 1
    assert len(embeddings.document_calls[0]) == 1
    assert len(vector.upsert_calls) == 1
    assert len(vector.commits) == 1
    assert [item.first_display_name for item in vector.commits[0].manifests] == [
        "manual.pdf"
    ]
    assert sha256(unreadable_pdf).hexdigest() not in manifest.committed
    assert sha256(valid_pdf).hexdigest() in manifest.committed
    assert not staging.exists()


def test_invalid_zip_cleans_staging_and_preserves_confirmed_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    _seed_confirmed_state(manifest, vector)
    before = _confirmed_snapshot(manifest, vector)
    upload_id = uuid4()
    archive, staging = _raw_archive(config, upload_id, b"not a ZIP")

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == "invalid_zip"
    assert _confirmed_snapshot(manifest, vector) == before
    assert embeddings.ready_calls == 0
    assert embeddings.document_calls == []
    assert vector.compatibility_calls == []
    assert vector.commit_attempts == []
    assert vector.upsert_calls == []
    assert not staging.exists()


def test_all_unreadable_pdfs_clean_staging_and_preserve_confirmed_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    _seed_confirmed_state(manifest, vector)
    before = _confirmed_snapshot(manifest, vector)
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (
            ("um.pdf", b"PDF corrompido 1"),
            ("dois.PDF", b"PDF corrompido 2"),
        ),
    )

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == "no_readable_pdfs"
    assert captured.value.http_status == 422
    assert _confirmed_snapshot(manifest, vector) == before
    assert embeddings.ready_calls == 1
    assert embeddings.document_calls == []
    assert vector.compatibility_calls == []
    assert vector.commit_attempts == []
    assert vector.upsert_calls == []
    assert not staging.exists()


@pytest.mark.parametrize(
    ("failure_phase", "expected_code"),
    (
        ("ready", "embedding_model_missing"),
        ("documents", "embedding_failed"),
    ),
    ids=("model-readiness", "document-embedding"),
)
def test_embedding_phase_failure_preserves_state_and_cleans_staging(
    tmp_path: Path,
    failure_phase: str,
    expected_code: str,
) -> None:
    config = _config(tmp_path)
    embeddings = _FailingEmbeddingProvider(
        config.embedding_model,
        failure_phase,
    )
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    _seed_confirmed_state(manifest, vector)
    before = _confirmed_snapshot(manifest, vector)
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (("manual.pdf", _pdf_bytes("Conteúdo para embedding")),),
    )

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == expected_code
    assert _confirmed_snapshot(manifest, vector) == before
    assert len(embeddings.document_calls) == (failure_phase == "documents")
    assert vector.compatibility_calls == []
    assert vector.commit_attempts == []
    assert vector.upsert_calls == []
    assert not staging.exists()


@pytest.mark.parametrize(
    ("failure_phase", "expected_code"),
    (
        ("guard", "ingestion_in_progress"),
        ("compatibility", "vector_space_mismatch"),
        ("lookup", "vector_store_unavailable"),
        ("collision", "chunk_identity_conflict"),
        ("commit", "vector_store_write_failed"),
    ),
)
def test_confirmation_phase_failure_never_changes_confirmed_state(
    tmp_path: Path,
    failure_phase: str,
    expected_code: str,
) -> None:
    config = _config(tmp_path)
    embeddings = _EmbeddingProvider(config.embedding_model)
    manifest = _Manifest()
    vector = _StatefulVectorStore(manifest)
    _seed_confirmed_state(manifest, vector)
    pdf = _pdf_bytes("Conteúdo novo para confirmação")

    if failure_phase == "guard":
        vector.reject_guard = True
    elif failure_phase == "compatibility":
        vector.compatibility_error = PublicError(
            code="vector_space_mismatch",
            message="A coleção usa um espaço vetorial incompatível.",
            http_status=503,
        )
    elif failure_phase == "lookup":
        vector.lookup_error = OSError("falha técnica de leitura")
    elif failure_phase == "commit":
        vector.commit_error = PublicError(
            code="vector_store_write_failed",
            message="Não foi possível gravar na base vetorial local.",
            http_status=503,
        )
    elif failure_phase == "collision":
        document_id = sha256(pdf).hexdigest()
        chunk_id = make_chunk_id(document_id, 1, 0)
        vector.existing[chunk_id] = StoredChunk(
            chunk=Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                display_name="manual.pdf",
                human_page=1,
                start_offset=0,
                text="conteúdo divergente já confirmado",
                transaction_id="tx-colisao-anterior",
            ),
            embedding=(1.0, 0.0, 0.0),
        )

    before = _confirmed_snapshot(manifest, vector)
    upload_id = uuid4()
    archive, staging = _archive(
        config,
        upload_id,
        (("manual.pdf", pdf),),
    )

    with pytest.raises(PublicError) as captured:
        _service(config, embeddings, manifest, vector).ingest(archive, upload_id)

    assert captured.value.code == expected_code
    assert _confirmed_snapshot(manifest, vector) == before
    assert len(embeddings.document_calls) == 1
    assert vector.commits == []
    assert vector.upsert_calls == []
    assert len(vector.commit_attempts) == (failure_phase == "commit")
    assert not vector.guard_is_active
    assert not staging.exists()
