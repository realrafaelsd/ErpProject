"""Property test for upload counts and structurally unique warnings."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from hypothesis import given, settings, strategies as st

from domain import (
    AppConfig,
    ArchivePlan,
    CommitPlan,
    DocumentManifestKey,
    ExtractedDocument,
    ExtractedEntry,
    PdfPage,
    StoredChunk,
    VectorSpace,
)
from ingest import ChunkingService, IngestionService, WarningCollector


_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 100
_BLANK_TEXT = st.sampled_from(
    ("", " ", "\t", "\n", "\r\n", "\u00a0", "\u2003", " \t\n\u00a0")
)
_TEXT_ALPHABET = st.sampled_from(tuple("abcXYZ019 áç中\n"))


@dataclass(frozen=True, slots=True)
class _UploadCase:
    documents: tuple[tuple[str, ...], ...]
    fully_blank_document: int
    non_pdf_count: int
    pdf_positions: tuple[int, ...]
    existing_mode: str


@dataclass(frozen=True, slots=True)
class _MaterializedUpload:
    archive_path: Path
    entries: tuple[ExtractedEntry, ...]
    documents_by_ordinal: Mapping[int, ExtractedDocument]
    documents: tuple[ExtractedDocument, ...]


@st.composite
def _non_blank_text(draw) -> str:
    length = draw(
        st.one_of(
            st.sampled_from((1, 499, 500, 501, 899, 900, 901, 1_200)),
            st.integers(min_value=1, max_value=1_200),
        )
    )
    tail = draw(
        st.text(
            alphabet=_TEXT_ALPHABET,
            min_size=length - 1,
            max_size=length - 1,
        )
    )
    return "A" + tail


_PAGE_TEXT = st.one_of(_BLANK_TEXT, _non_blank_text())


@st.composite
def _upload_cases(draw) -> _UploadCase:
    document_count = draw(st.integers(min_value=1, max_value=4))
    fully_blank_document = draw(
        st.integers(min_value=0, max_value=document_count - 1)
    )

    documents: list[tuple[str, ...]] = []
    for document_index in range(document_count):
        page_count = draw(st.integers(min_value=1, max_value=4))
        page_strategy = (
            _BLANK_TEXT
            if document_index == fully_blank_document
            else _PAGE_TEXT
        )
        documents.append(
            tuple(
                draw(
                    st.lists(
                        page_strategy,
                        min_size=page_count,
                        max_size=page_count,
                    )
                )
            )
        )

    non_pdf_count = draw(st.integers(min_value=0, max_value=3))
    entry_count = document_count + non_pdf_count
    pdf_positions = tuple(
        sorted(
            draw(
                st.sets(
                    st.integers(min_value=0, max_value=entry_count - 1),
                    min_size=document_count,
                    max_size=document_count,
                )
            )
        )
    )
    return _UploadCase(
        documents=tuple(documents),
        fully_blank_document=fully_blank_document,
        non_pdf_count=non_pdf_count,
        pdf_positions=pdf_positions,
        existing_mode=draw(
            st.sampled_from(("none", "all", "alternating", "first"))
        ),
    )


def _config(root: Path) -> AppConfig:
    upload_folder = root / "uploads"
    upload_folder.mkdir(mode=0o700)
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generator",
        chroma_path=root / "chroma",
        chroma_collection="property_collection",
        upload_folder=upload_folder,
        embedding_model="local/property-embedding",
        top_k=6,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        relevance_threshold=0.3,
        max_upload_bytes=1_048_576,
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


def _materialize_upload(
    config: AppConfig,
    case: _UploadCase,
) -> _MaterializedUpload:
    staging = config.upload_folder / "upload-property-10"
    files_root = staging / "published"
    files_root.mkdir(mode=0o700, parents=True)
    archive_path = staging / "knowledge.zip"
    archive_path.write_bytes(b"synthetic archive handled by the in-memory adapter")

    document_at_position = {
        position: document_index
        for document_index, position in enumerate(case.pdf_positions)
    }
    entries: list[ExtractedEntry] = []
    documents_by_ordinal: dict[int, ExtractedDocument] = {}
    documents: list[ExtractedDocument] = []
    non_pdf_index = 0

    for ordinal in range(len(case.documents) + case.non_pdf_count):
        path = files_root / f"entry-{ordinal}.bin"
        if ordinal in document_at_position:
            document_index = document_at_position[ordinal]
            payload = f"property-document-{document_index}".encode("ascii")
            display_name = f"manuais/documento-{document_index}.pdf"
            pages = tuple(
                PdfPage(human_page=page, text=text)
                for page, text in enumerate(
                    case.documents[document_index],
                    start=1,
                )
            )
            document = ExtractedDocument(
                document_id=sha256(payload).hexdigest(),
                display_name=display_name,
                pages=pages,
            )
            documents_by_ordinal[ordinal] = document
            documents.append(document)
        else:
            payload = f"ignored-file-{non_pdf_index}".encode("ascii")
            display_name = f"anexos/ignorado-{non_pdf_index}.txt"
            non_pdf_index += 1

        path.write_bytes(payload)
        entries.append(
            ExtractedEntry(
                ordinal=ordinal,
                path=path,
                display_name=display_name,
                size=len(payload),
            )
        )

    return _MaterializedUpload(
        archive_path=archive_path,
        entries=tuple(entries),
        documents_by_ordinal=documents_by_ordinal,
        documents=tuple(documents),
    )


def _empty_page_warning(document: str, page: int) -> str:
    return (
        f"{document}, página {page}: nenhum caractere não branco foi "
        "extraído. O MVP não executa OCR; aplique OCR antes de uma nova "
        "importação se a página contiver texto em imagem."
    )


def _expected_chunk_count(text: str) -> int:
    if text.strip() == "":
        return 0
    if len(text) <= _CHUNK_SIZE:
        return 1
    stride = _CHUNK_SIZE - _CHUNK_OVERLAP
    return 1 + (len(text) - _CHUNK_SIZE + stride - 1) // stride


def _is_existing(index: int, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "alternating":
        return index % 2 == 0
    if mode == "first":
        return index == 0
    return False


class _ZipAdapter:
    def __init__(self, entries: tuple[ExtractedEntry, ...]) -> None:
        self._entries = entries

    def inspect(self, archive_path: Path, extraction_root: Path) -> ArchivePlan:
        del archive_path, extraction_root
        return ArchivePlan(entries=(), declared_total_bytes=0)

    def extract(
        self,
        archive_path: Path,
        plan: ArchivePlan,
    ) -> list[ExtractedEntry]:
        del archive_path, plan
        return list(self._entries)


class _DocumentAdapter:
    def __init__(
        self,
        documents_by_ordinal: Mapping[int, ExtractedDocument],
    ) -> None:
        self._documents = documents_by_ordinal
        self.calls: list[int] = []
        self.duplicate_warning_attempts = 0

    def extract(
        self,
        entry: ExtractedEntry,
        spool_root: Path,
        document_id: str | None = None,
        warnings: WarningCollector | None = None,
    ) -> ExtractedDocument:
        del spool_root
        assert warnings is not None
        document = self._documents[entry.ordinal]
        assert document_id == document.document_id
        assert entry.display_name == document.display_name
        self.calls.append(entry.ordinal)

        for page in document.pages:
            if page.text.strip() != "":
                continue
            message = _empty_page_warning(
                document.display_name,
                page.human_page,
            )
            assert warnings.add(
                "pdf_page_empty",
                entry.ordinal,
                page.human_page,
                message,
            )
            self.duplicate_warning_attempts += 1
            assert not warnings.add(
                "pdf_page_empty",
                entry.ordinal,
                page.human_page,
                "duplicate event must not replace the first warning",
            )

        return document


class _EmbeddingAdapter:
    def __init__(self, model: str) -> None:
        self.space = VectorSpace(
            model=model,
            dimension=3,
            normalized=True,
            metric="cosine",
        )
        self.document_calls: list[tuple[str, ...]] = []

    def ensure_ready(self) -> VectorSpace:
        return self.space

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        materialized = tuple(texts)
        self.document_calls.append(materialized)
        return [[1.0, 0.0, 0.0] for _ in materialized]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError(f"ingestion must not embed query: {text}")


class _ManifestAdapter:
    def is_duplicate(self, key: DocumentManifestKey) -> bool:
        del key
        return False


class _Guard:
    def __init__(self, store: "_VectorAdapter") -> None:
        self._store = store

    def __enter__(self) -> None:
        assert not self._store.guard_active
        self._store.guard_active = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self._store.guard_active = False
        return False


class _VectorAdapter:
    def __init__(self, existing: Mapping[str, StoredChunk]) -> None:
        self.existing = dict(existing)
        self.guard_active = False
        self.commits: list[CommitPlan] = []

    def ingestion_guard(self) -> _Guard:
        return _Guard(self)

    def ensure_compatible(self, space: VectorSpace, profile: object) -> None:
        del space, profile
        assert self.guard_active

    def existing_chunks(
        self,
        ids: Sequence[str],
    ) -> Mapping[str, StoredChunk]:
        assert self.guard_active
        return {
            chunk_id: self.existing[chunk_id]
            for chunk_id in ids
            if chunk_id in self.existing
        }

    def commit_chunks(self, plan: CommitPlan) -> None:
        assert self.guard_active
        self.commits.append(plan)

    def count_chunks(self) -> int:
        return len(self.existing)

    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self.existing.pop(chunk_id, None)

    def query(self, embedding: Sequence[float], limit: int) -> list[object]:
        del embedding, limit
        raise AssertionError("ingestion must not query the vector store")


@settings(max_examples=120, deadline=None)
@given(case=_upload_cases())
def test_property_10_upload_counts_and_warnings_come_from_confirmed_plan(
    case: _UploadCase,
) -> None:
    # Feature: erp-ai-support, Property 10: Contagens e avisos são derivados do plano confirmado
    # **Validates: Requirements 9.11, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8**
    with TemporaryDirectory(prefix="erp-ai-support-property-10-") as directory:
        root = Path(directory)
        config = _config(root)
        materialized = _materialize_upload(config, case)
        transaction_id = "property-10-transaction"
        chunker = ChunkingService(config)

        chunks_by_document = tuple(
            tuple(chunker.split_document(document, transaction_id))
            for document in materialized.documents
        )
        expected_document_chunk_counts = tuple(
            sum(_expected_chunk_count(page.text) for page in document.pages)
            for document in materialized.documents
        )
        assert tuple(map(len, chunks_by_document)) == expected_document_chunk_counts

        all_chunks = tuple(
            chunk
            for document_chunks in chunks_by_document
            for chunk in document_chunks
        )
        existing = {
            chunk.chunk_id: StoredChunk(
                chunk=chunk,
                embedding=(1.0, 0.0, 0.0),
            )
            for index, chunk in enumerate(all_chunks)
            if _is_existing(index, case.existing_mode)
        }

        document_adapter = _DocumentAdapter(
            materialized.documents_by_ordinal
        )
        embeddings = _EmbeddingAdapter(config.embedding_model)
        vector = _VectorAdapter(existing)
        service = IngestionService(
            config,
            embeddings,
            _ManifestAdapter(),
            vector,
            zip_validator=_ZipAdapter(materialized.entries),  # type: ignore[arg-type]
            pdf_extractor=document_adapter,  # type: ignore[arg-type]
            chunking_service=chunker,
        )

        result = service.ingest(
            materialized.archive_path,
            transaction_id,
        )

        expected_pages = sum(
            len(document.pages) for document in materialized.documents
        )
        expected_new_chunks = len(all_chunks) - len(existing)
        non_pdf_entries = tuple(
            entry
            for entry in materialized.entries
            if not entry.display_name.casefold().endswith(".pdf")
        )
        expected_warnings = tuple(
            f"{entry.display_name}: arquivo ignorado porque não possui a "
            "extensão .pdf."
            for entry in non_pdf_entries
        ) + tuple(
            _empty_page_warning(document.display_name, page.human_page)
            for document in materialized.documents
            for page in document.pages
            if page.text.strip() == ""
        )
        expected_warning_keys = {
            ("non_pdf", entry.ordinal, None) for entry in non_pdf_entries
        } | {
            ("pdf_page_empty", ordinal, page.human_page)
            for ordinal, document in materialized.documents_by_ordinal.items()
            for page in document.pages
            if page.text.strip() == ""
        }

        assert result.success is True
        assert result.documents == len(materialized.documents)
        assert result.pages == expected_pages
        assert result.chunks == expected_new_chunks
        assert result.warnings == expected_warnings
        assert len(result.warnings) == len(expected_warning_keys)
        assert len(result.warnings) == len(set(result.warnings))

        assert len(vector.commits) == 1
        plan = vector.commits[0]
        assert plan.documents == result.documents
        assert plan.pages == result.pages
        assert len(plan.new_chunk_ids) == result.chunks
        assert plan.warnings == result.warnings
        assert len(plan.chunks) == len(all_chunks)
        assert tuple(
            manifest.page_count for manifest in plan.manifests
        ) == tuple(len(document.pages) for document in materialized.documents)
        assert tuple(
            manifest.chunk_count for manifest in plan.manifests
        ) == expected_document_chunk_counts

        blank_document = materialized.documents[case.fully_blank_document]
        blank_manifest = next(
            manifest
            for manifest in plan.manifests
            if manifest.first_display_name == blank_document.display_name
        )
        assert all(page.text.strip() == "" for page in blank_document.pages)
        assert blank_manifest.page_count == len(blank_document.pages)
        assert blank_manifest.chunk_count == 0
        assert all(
            _empty_page_warning(blank_document.display_name, page.human_page)
            in result.warnings
            for page in blank_document.pages
        )

        pdf_ordinals = [
            entry.ordinal
            for entry in materialized.entries
            if entry.display_name.casefold().endswith(".pdf")
        ]
        assert document_adapter.calls == pdf_ordinals
        assert document_adapter.duplicate_warning_attempts == sum(
            1
            for document in materialized.documents
            for page in document.pages
            if page.text.strip() == ""
        )
        if all_chunks:
            assert len(embeddings.document_calls) == 1
            assert len(embeddings.document_calls[0]) == len(all_chunks)
        else:
            assert embeddings.document_calls == []
        assert not materialized.archive_path.parent.exists()
