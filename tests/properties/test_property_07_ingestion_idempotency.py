"""Property test for idempotent ingestion over an in-memory confirmed state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import pytest
from hypothesis import given, settings, strategies as st

from domain import (
    AppConfig,
    Chunk,
    CommitPlan,
    DocumentManifest,
    DocumentManifestKey,
    ExtractedDocument,
    ExtractedEntry,
    PdfPage,
    PublicError,
    StoredChunk,
    VectorSpace,
)
from ingest import ChunkingService, IngestionService


_DUPLICATE_WARNING = (
    "{document}: documento duplicado; nenhuma nova indexação foi realizada."
)
_READ_FAILURE_WARNING = (
    "{document}: arquivo ignorado por falha de leitura; substitua-o ou "
    "remova-o antes de reenviar."
)
_SAFE_STEM = st.text(
    alphabet=tuple("abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=1,
    max_size=12,
)


@dataclass(frozen=True, slots=True)
class _Occurrence:
    payload: bytes
    display_name: str
    readable: bool

    @property
    def document_id(self) -> str:
        return sha256(self.payload).hexdigest()


@dataclass(frozen=True, slots=True)
class _IngestionCase:
    base_values: tuple[int, ...]
    replayed_base_indexes: tuple[int, ...]
    first_new_value: int
    second_new_value: int
    unreadable_first_occurrences: int
    repeated_first_occurrences: int
    repeated_second_occurrences: int
    stem: str
    failure_stage: str


@dataclass(slots=True)
class _ConfirmedState:
    chunks: dict[str, StoredChunk] = field(default_factory=dict)
    manifests: dict[DocumentManifestKey, DocumentManifest] = field(
        default_factory=dict
    )


class _MemoryManifest:
    def __init__(self, state: _ConfirmedState) -> None:
        self._state = state

    def is_duplicate(self, key: DocumentManifestKey) -> bool:
        return key in self._state.manifests


class _IngestionGuard:
    def __init__(self, store: "_MemoryVectorStore") -> None:
        self._store = store

    def __enter__(self) -> None:
        assert not self._store.guard_active
        self._store.guard_active = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._store.guard_active = False
        return False


class _MemoryVectorStore:
    """Atomically apply commit plans to the shared confirmed-state model."""

    def __init__(self, state: _ConfirmedState) -> None:
        self._state = state
        self.guard_active = False
        self.fail_next_commit = False
        self.commit_attempts: list[CommitPlan] = []
        self.successful_commits: list[CommitPlan] = []
        self.existing_calls: list[tuple[str, ...]] = []

    def ingestion_guard(self) -> _IngestionGuard:
        return _IngestionGuard(self)

    def ensure_compatible(self, space: VectorSpace, profile: object) -> None:
        assert self.guard_active
        assert space.normalized is True
        assert space.metric == "cosine"
        assert profile == _chunker_profile()

    def existing_chunks(
        self, ids: Sequence[str]
    ) -> Mapping[str, StoredChunk]:
        assert self.guard_active
        materialized = tuple(ids)
        self.existing_calls.append(materialized)
        return {
            chunk_id: self._state.chunks[chunk_id]
            for chunk_id in materialized
            if chunk_id in self._state.chunks
        }

    def commit_chunks(self, plan: CommitPlan) -> None:
        assert self.guard_active
        self.commit_attempts.append(plan)
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise PublicError(
                code="vector_store_write_failed",
                message=(
                    "Não foi possível gravar na base vetorial local. Os dados "
                    "não foram confirmados; verifique o armazenamento configurado."
                ),
                http_status=503,
            )

        next_chunks = dict(self._state.chunks)
        next_manifests = dict(self._state.manifests)
        expected_new_ids: list[str] = []
        for stored in plan.chunks:
            chunk_id = stored.chunk.chunk_id
            current = next_chunks.get(chunk_id)
            if current is None:
                expected_new_ids.append(chunk_id)
                next_chunks[chunk_id] = stored
            else:
                assert _logical_chunk(current) == _logical_chunk(stored)

        assert plan.new_chunk_ids == tuple(expected_new_ids)
        for manifest in plan.manifests:
            current = next_manifests.get(manifest.key)
            assert current is None or current == manifest
            next_manifests[manifest.key] = manifest

        self._state.chunks.clear()
        self._state.chunks.update(next_chunks)
        self._state.manifests.clear()
        self._state.manifests.update(next_manifests)
        self.successful_commits.append(plan)

    def count_chunks(self) -> int:
        return len(self._state.chunks)

    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self._state.chunks.pop(chunk_id, None)

    def query(self, embedding: Sequence[float], limit: int) -> list[object]:
        raise AssertionError("ingestion must not query the vector store")


class _MemoryEmbeddingProvider:
    def __init__(self, model: str) -> None:
        self.space = VectorSpace(
            model=model,
            dimension=3,
            normalized=True,
            metric="cosine",
        )
        self.fail_next_documents = False
        self.document_calls: list[tuple[str, ...]] = []

    def ensure_ready(self) -> VectorSpace:
        return self.space

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        materialized = tuple(texts)
        self.document_calls.append(materialized)
        if self.fail_next_documents:
            self.fail_next_documents = False
            raise PublicError(
                code="embedding_failed",
                message="A geração local de embeddings falhou.",
                http_status=503,
            )
        return [list(_embedding(text)) for text in materialized]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("ingestion must not embed a query")


class _MemoryZipValidator:
    """Publish generated occurrences without invoking a ZIP implementation."""

    def __init__(self, occurrences: tuple[_Occurrence, ...]) -> None:
        self._occurrences = occurrences
        self._plan = object()

    def inspect(self, archive_path: Path, extraction_root: Path) -> object:
        assert archive_path.is_file()
        assert not extraction_root.exists()
        return self._plan

    def extract(
        self, archive_path: Path, plan: object
    ) -> list[ExtractedEntry]:
        assert archive_path.is_file()
        assert plan is self._plan
        extraction_root = archive_path.parent / ".model-extracted"
        extraction_root.mkdir(mode=0o700)
        entries: list[ExtractedEntry] = []
        for ordinal, occurrence in enumerate(self._occurrences):
            path = extraction_root / f"entry-{ordinal}.pdf"
            path.write_bytes(occurrence.payload)
            entries.append(
                ExtractedEntry(
                    ordinal=ordinal,
                    path=path,
                    display_name=occurrence.display_name,
                    size=len(occurrence.payload),
                )
            )
        return entries


class _MemoryPdfExtractor:
    def __init__(self, occurrences: tuple[_Occurrence, ...]) -> None:
        self._occurrences = occurrences
        self.calls: list[int] = []

    def extract(
        self,
        entry: ExtractedEntry,
        spool_root: Path,
        document_id: str | None = None,
        warnings: object | None = None,
    ) -> ExtractedDocument | None:
        del spool_root, warnings
        occurrence = self._occurrences[entry.ordinal]
        self.calls.append(entry.ordinal)
        assert document_id == occurrence.document_id
        if not occurrence.readable:
            return None
        return ExtractedDocument(
            document_id=occurrence.document_id,
            display_name=occurrence.display_name,
            pages=(
                PdfPage(
                    human_page=1,
                    text=_document_text(occurrence.payload),
                ),
            ),
        )


@st.composite
def _ingestion_cases(draw) -> _IngestionCase:
    values = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**32 - 1),
            min_size=3,
            max_size=6,
            unique=True,
        )
    )
    base_values = tuple(values[:-2])
    return _IngestionCase(
        base_values=base_values,
        replayed_base_indexes=tuple(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=len(base_values) - 1),
                    min_size=1,
                    max_size=4,
                )
            )
        ),
        first_new_value=values[-2],
        second_new_value=values[-1],
        unreadable_first_occurrences=draw(
            st.integers(min_value=1, max_value=3)
        ),
        repeated_first_occurrences=draw(
            st.integers(min_value=1, max_value=3)
        ),
        repeated_second_occurrences=draw(
            st.integers(min_value=0, max_value=3)
        ),
        stem=draw(_SAFE_STEM),
        failure_stage=draw(st.sampled_from(("embedding", "commit"))),
    )


def _payload(value: int) -> bytes:
    return b"document:" + value.to_bytes(4, "big")


def _document_text(payload: bytes) -> str:
    return "conteudo-" + payload.hex()


def _embedding(text: str) -> tuple[float, float, float]:
    digest = sha256(text.encode("utf-8")).digest()
    return tuple(component / 255.0 for component in digest[:3])  # type: ignore[return-value]


def _config(root: Path) -> AppConfig:
    upload_folder = root / "uploads"
    upload_folder.mkdir()
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generator",
        chroma_path=root / "chroma",
        chroma_collection="property_collection",
        upload_folder=upload_folder,
        embedding_model="local/property-embedding",
        top_k=6,
        chunk_size=32,
        chunk_overlap=8,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
        max_zip_entries=100,
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


def _chunker_profile():
    return ChunkingService(32, 8).profile


def _document_records(
    config: AppConfig,
    space: VectorSpace,
    *,
    payload: bytes,
    display_name: str,
    transaction_id: str,
) -> tuple[DocumentManifest, tuple[StoredChunk, ...]]:
    document = ExtractedDocument(
        document_id=sha256(payload).hexdigest(),
        display_name=display_name,
        pages=(PdfPage(human_page=1, text=_document_text(payload)),),
    )
    chunks = ChunkingService(config).split_document(document, transaction_id)
    stored = tuple(
        StoredChunk(chunk=chunk, embedding=_embedding(chunk.text))
        for chunk in chunks
    )
    manifest = DocumentManifest(
        key=DocumentManifestKey(
            document_id=document.document_id,
            vector_fingerprint=space.fingerprint,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            chunk_schema_version="char-v1",
        ),
        first_display_name=display_name,
        page_count=1,
        chunk_count=len(stored),
        transaction_id=transaction_id,
    )
    return manifest, stored


def _seed_confirmed_document(
    state: _ConfirmedState,
    config: AppConfig,
    space: VectorSpace,
    *,
    payload: bytes,
    display_name: str,
    transaction_id: str,
) -> None:
    manifest, stored = _document_records(
        config,
        space,
        payload=payload,
        display_name=display_name,
        transaction_id=transaction_id,
    )
    state.manifests[manifest.key] = manifest
    for record in stored:
        assert record.chunk.chunk_id not in state.chunks
        state.chunks[record.chunk.chunk_id] = record


def _logical_chunk(stored: StoredChunk) -> tuple[object, ...]:
    chunk = stored.chunk
    return (
        chunk.chunk_id,
        chunk.document_id,
        chunk.display_name,
        chunk.human_page,
        chunk.start_offset,
        chunk.text,
    )


def _state_snapshot(state: _ConfirmedState) -> tuple[object, ...]:
    chunks = tuple(
        sorted(
            (
                chunk_id,
                *_logical_chunk(stored),
                stored.embedding,
                stored.chunk.transaction_id,
            )
            for chunk_id, stored in state.chunks.items()
        )
    )
    manifests = tuple(
        sorted(
            (
                key.document_id,
                key.vector_fingerprint,
                key.chunk_size,
                key.chunk_overlap,
                key.chunk_schema_version,
                manifest.first_display_name,
                manifest.page_count,
                manifest.chunk_count,
                manifest.transaction_id,
            )
            for key, manifest in state.manifests.items()
        )
    )
    return chunks, manifests


def _occurrences(case: _IngestionCase) -> tuple[_Occurrence, ...]:
    first_payload = _payload(case.first_new_value)
    second_payload = _payload(case.second_new_value)
    shared_name = f"guias/{case.stem}.pdf"
    occurrences = [
        _Occurrence(
            payload=_payload(case.base_values[index]),
            display_name=f"reaplicacoes/{position}-{case.stem}.pdf",
            readable=True,
        )
        for position, index in enumerate(case.replayed_base_indexes)
    ]
    occurrences.extend(
        _Occurrence(
            payload=first_payload,
            display_name=f"falhas/{index}-{case.stem}.pdf",
            readable=False,
        )
        for index in range(case.unreadable_first_occurrences)
    )
    occurrences.append(
        _Occurrence(
            payload=first_payload,
            display_name=shared_name,
            readable=True,
        )
    )
    occurrences.extend(
        _Occurrence(
            payload=first_payload,
            display_name=f"copias-a/{index}-{case.stem}.pdf",
            readable=True,
        )
        for index in range(case.repeated_first_occurrences)
    )
    occurrences.append(
        _Occurrence(
            payload=second_payload,
            display_name=shared_name,
            readable=True,
        )
    )
    occurrences.extend(
        _Occurrence(
            payload=second_payload,
            display_name=f"copias-b/{index}-{case.stem}.pdf",
            readable=True,
        )
        for index in range(case.repeated_second_occurrences)
    )
    return tuple(occurrences)


def _model_first_application(
    initially_confirmed_ids: set[str],
    occurrences: tuple[_Occurrence, ...],
) -> tuple[tuple[int, ...], tuple[_Occurrence, ...], tuple[str, ...]]:
    seen = set(initially_confirmed_ids)
    extractor_calls: list[int] = []
    processed: list[_Occurrence] = []
    warnings: list[str] = []
    for ordinal, occurrence in enumerate(occurrences):
        if occurrence.document_id in seen:
            warnings.append(
                _DUPLICATE_WARNING.format(document=occurrence.display_name)
            )
            continue
        extractor_calls.append(ordinal)
        if not occurrence.readable:
            warnings.append(
                _READ_FAILURE_WARNING.format(document=occurrence.display_name)
            )
            continue
        seen.add(occurrence.document_id)
        processed.append(occurrence)
    return tuple(extractor_calls), tuple(processed), tuple(warnings)


def _upload_id(case: _IngestionCase, phase: int) -> UUID:
    case_digest = sha256(repr(case).encode("utf-8")).digest()
    base = int.from_bytes(case_digest[:16], "big")
    return UUID(int=(base + phase) % (2**128))


def _run_upload(
    service: IngestionService,
    config: AppConfig,
    upload_id: UUID,
):
    staging = config.upload_folder / f"upload-{upload_id}"
    staging.mkdir(mode=0o700)
    archive = staging / "knowledge.zip"
    archive.write_bytes(b"in-memory archive boundary")
    try:
        return service.ingest(archive, upload_id)
    finally:
        assert not staging.exists()


@settings(max_examples=100, deadline=None)
@given(case=_ingestion_cases())
def test_property_07_repeated_ingestion_is_idempotent_and_preserves_state(
    case: _IngestionCase,
) -> None:
    # Feature: erp-ai-support, Property 7: Ingestão repetida é idempotente e preserva estado anterior
    # **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.10, 11.11**
    with TemporaryDirectory(prefix="erp-ai-support-property-07-") as directory:
        config = _config(Path(directory))
        state = _ConfirmedState()
        embeddings = _MemoryEmbeddingProvider(config.embedding_model)
        manifest = _MemoryManifest(state)
        vector_store = _MemoryVectorStore(state)

        for index, value in enumerate(case.base_values):
            _seed_confirmed_document(
                state,
                config,
                embeddings.space,
                payload=_payload(value),
                display_name=f"confirmados/{index}-{case.stem}.pdf",
                transaction_id=f"confirmed-{index}",
            )

        occurrences = _occurrences(case)
        pdf_extractor = _MemoryPdfExtractor(occurrences)
        service = IngestionService(
            config,
            embeddings,
            manifest,
            vector_store,
            zip_validator=_MemoryZipValidator(occurrences),
            pdf_extractor=pdf_extractor,
        )
        initially_confirmed_ids = {
            key.document_id for key in state.manifests
        }
        expected_calls, expected_processed, expected_warnings = (
            _model_first_application(initially_confirmed_ids, occurrences)
        )
        assert len(expected_processed) == 2
        assert expected_processed[0].display_name == expected_processed[1].display_name
        assert expected_processed[0].document_id != expected_processed[1].document_id

        initial_snapshot = _state_snapshot(state)
        initial_chunks = dict(state.chunks)
        if case.failure_stage == "embedding":
            embeddings.fail_next_documents = True
            expected_error = "embedding_failed"
        else:
            vector_store.fail_next_commit = True
            expected_error = "vector_store_write_failed"

        with pytest.raises(PublicError) as captured:
            _run_upload(service, config, _upload_id(case, 1))

        assert captured.value.code == expected_error
        assert _state_snapshot(state) == initial_snapshot
        assert pdf_extractor.calls == list(expected_calls)
        assert not any(
            occurrence.document_id in {
                key.document_id for key in state.manifests
            }
            for occurrence in expected_processed
        )

        successful_result = _run_upload(
            service,
            config,
            _upload_id(case, 2),
        )

        assert pdf_extractor.calls == list(expected_calls) * 2
        assert successful_result.documents == len(expected_processed)
        assert successful_result.pages == len(expected_processed)
        assert successful_result.warnings == expected_warnings
        assert len(successful_result.warnings) == (
            len(occurrences) - len(expected_processed)
        )

        successful_plan = vector_store.successful_commits[-1]
        assert successful_result.chunks == len(successful_plan.new_chunk_ids)
        assert len(successful_plan.manifests) == len(expected_processed)
        assert [
            item.first_display_name for item in successful_plan.manifests
        ] == [item.display_name for item in expected_processed]
        assert len(
            {item.key.document_id for item in successful_plan.manifests}
        ) == len(expected_processed)
        assert all(
            state.chunks[chunk_id] == stored
            for chunk_id, stored in initial_chunks.items()
        )

        processed_ids = {item.document_id for item in expected_processed}
        assert processed_ids.issubset(
            {key.document_id for key in state.manifests}
        )
        assert {
            manifest_item.first_display_name
            for key, manifest_item in state.manifests.items()
            if key.document_id in processed_ids
        } == {expected_processed[0].display_name}
        assert {
            stored.chunk.document_id
            for stored in state.chunks.values()
            if stored.chunk.document_id in processed_ids
        } == processed_ids
        assert len(state.chunks) == len(set(state.chunks))

        committed_snapshot = _state_snapshot(state)
        extractor_calls_before_replay = tuple(pdf_extractor.calls)
        embedding_calls_before_replay = tuple(embeddings.document_calls)
        commit_attempts_before_replay = tuple(vector_store.commit_attempts)
        repeated_result = _run_upload(
            service,
            config,
            _upload_id(case, 3),
        )

        assert (repeated_result.documents, repeated_result.pages, repeated_result.chunks) == (
            0,
            0,
            0,
        )
        assert repeated_result.warnings == tuple(
            _DUPLICATE_WARNING.format(document=item.display_name)
            for item in occurrences
        )
        assert len(repeated_result.warnings) == len(occurrences)
        assert tuple(pdf_extractor.calls) == extractor_calls_before_replay
        assert tuple(embeddings.document_calls) == embedding_calls_before_replay
        assert tuple(vector_store.commit_attempts) == commit_attempts_before_replay
        assert _state_snapshot(state) == committed_snapshot
