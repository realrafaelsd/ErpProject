"""Shared, dependency-free domain contracts for ERP AI Support.

This module intentionally contains only immutable data transfer objects, narrow
structural protocols, and the public error type shared by the application
modules.  It must not depend on Flask or on the concrete ingestion, storage,
embedding, or generation implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable


DistanceMetric: TypeAlias = Literal["cosine"]
ChunkSchemaVersion: TypeAlias = Literal["char-v1"]
TransactionState: TypeAlias = Literal[
    "PREPARED", "CHROMA_COMMITTED", "COMMITTED", "ABORTED"
]


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete configuration made available only after central validation."""

    ollama_url: str
    ollama_model: str
    chroma_path: Path
    chroma_collection: str
    upload_folder: Path
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    relevance_threshold: float
    max_upload_bytes: int
    max_zip_entries: int
    max_zip_entry_bytes: int
    max_uncompressed_bytes: int
    max_compression_ratio: float
    max_question_chars: int
    ollama_timeout_seconds: int
    max_answer_tokens: int
    flask_host: str
    flask_port: int
    flask_debug: bool


@dataclass(frozen=True, slots=True)
class PublicError(Exception):
    """An allowlisted error safe to serialize at an HTTP boundary.

    Concrete modules may chain an internal exception with ``raise ... from``;
    only these three public fields belong in a client response.
    """

    code: str
    message: str
    http_status: int

    def __post_init__(self) -> None:
        # Keep normal exception behaviour while ensuring ``str(error)`` contains
        # only the already-sanitized public message.
        Exception.__init__(self, self.message)


@dataclass(frozen=True, slots=True)
class VectorSpace:
    """Identity of the embedding space accepted by one vector collection."""

    model: str
    dimension: int
    normalized: bool
    metric: DistanceMetric

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint for persistence compatibility checks."""

        canonical = dumps(
            {
                "dimension": self.dimension,
                "metric": self.metric,
                "model": self.model,
                "normalized": self.normalized,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkingProfile:
    """Character chunking parameters fixed for a vector collection."""

    size: int
    overlap: int
    schema_version: ChunkSchemaVersion


@dataclass(frozen=True, slots=True)
class UploadDescriptor:
    """Validated, non-content upload metadata."""

    filename: str
    mimetype: str


@dataclass(frozen=True, slots=True)
class ArchiveEntryPlan:
    """One ZIP member approved during inspection, before extraction."""

    ordinal: int
    archive_name: str
    relative_path: str
    resolved_target: Path
    is_directory: bool
    declared_size: int
    compressed_size: int


@dataclass(frozen=True, slots=True)
class ArchivePlan:
    """Fully materialized ZIP extraction plan in declaration order."""

    entries: tuple[ArchiveEntryPlan, ...]
    declared_total_bytes: int


@dataclass(frozen=True, slots=True)
class ExtractedEntry:
    """A regular file safely extracted from an approved ZIP member."""

    ordinal: int
    path: Path
    display_name: str
    size: int


@dataclass(frozen=True, slots=True)
class PdfPage:
    """Text extracted from one physical PDF page."""

    human_page: int
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A completely readable PDF and all of its extracted pages."""

    document_id: str
    display_name: str
    pages: tuple[PdfPage, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A traceable, contiguous text slice from one PDF page."""

    chunk_id: str
    document_id: str
    display_name: str
    human_page: int
    start_offset: int
    text: str
    transaction_id: str


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk paired with its validated application-provided embedding."""

    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DocumentManifestKey:
    """Compatibility-qualified identity used for document deduplication."""

    document_id: str
    vector_fingerprint: str
    chunk_size: int
    chunk_overlap: int
    chunk_schema_version: ChunkSchemaVersion


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    """Technical metadata for a document in a committed ingestion."""

    key: DocumentManifestKey
    first_display_name: str
    page_count: int
    chunk_count: int
    transaction_id: str


@dataclass(frozen=True, slots=True)
class UploadResult:
    """Successful upload result exposed by the HTTP boundary."""

    success: Literal[True]
    documents: int
    pages: int
    chunks: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RawCandidate:
    """Validated vector-store candidate before relevance filtering."""

    original_index: int
    chunk_id: str
    document: str
    metadata: dict[str, object]
    distance: float


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Chunk retained for generation after deterministic retrieval filtering."""

    chunk_id: str
    text: str
    document_id: str
    display_name: str
    human_page: int
    score: float


@dataclass(frozen=True, slots=True)
class Source:
    """Public provenance for a document/page pair."""

    document: str
    page: int


@dataclass(frozen=True, slots=True)
class RagResult:
    """Answer and independently derived sources returned by the RAG service."""

    answer: str
    sources: tuple[Source, ...]


@dataclass(frozen=True, slots=True)
class CommitPlan:
    """Complete ingestion payload prepared before the first persistent write."""

    transaction_id: str
    chunks: tuple[StoredChunk, ...]
    manifests: tuple[DocumentManifest, ...]
    new_chunk_ids: tuple[str, ...]
    documents: int
    pages: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransactionJournal:
    """Persistent recovery state for one ingestion transaction."""

    transaction_id: str
    state: TransactionState
    new_chunk_ids: tuple[str, ...]
    plan_checksum: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Boundary implemented by the local embedding service."""

    def ensure_ready(self) -> VectorSpace:
        """Load/validate the local model and return its vector space."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts in the configured vector space."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one question in the same configured vector space."""


@runtime_checkable
class VectorStore(Protocol):
    """Persistence and retrieval boundary for chunk records."""

    def ensure_compatible(
        self, space: VectorSpace, profile: ChunkingProfile
    ) -> None:
        """Reject a collection incompatible with the supplied contracts."""

    def count_chunks(self) -> int:
        """Return the number of visible, confirmed chunks."""

    def existing_chunks(
        self, ids: Sequence[str]
    ) -> Mapping[str, StoredChunk]:
        """Return records already stored for the requested identifiers."""

    def commit_chunks(self, plan: CommitPlan) -> None:
        """Apply a prepared commit plan atomically at the adapter boundary."""

    def rollback_new_chunks(self, ids: Sequence[str]) -> None:
        """Remove only records newly introduced by an incomplete transaction."""

    def query(
        self, embedding: Sequence[float], limit: int
    ) -> list[RawCandidate]:
        """Return at most ``limit`` raw candidates for one embedding."""


@runtime_checkable
class GeneratorClient(Protocol):
    """Boundary implemented by the configured local generation client."""

    def generate(
        self, prompt: str, max_tokens: int, timeout_seconds: int
    ) -> str:
        """Generate one independent answer from the supplied prompt."""


__all__ = [
    "AppConfig",
    "ArchiveEntryPlan",
    "ArchivePlan",
    "Chunk",
    "ChunkingProfile",
    "ChunkSchemaVersion",
    "CommitPlan",
    "DistanceMetric",
    "DocumentManifest",
    "DocumentManifestKey",
    "EmbeddingProvider",
    "ExtractedDocument",
    "ExtractedEntry",
    "GeneratorClient",
    "PdfPage",
    "PublicError",
    "RagResult",
    "RawCandidate",
    "RetrievedChunk",
    "Source",
    "StoredChunk",
    "TransactionJournal",
    "TransactionState",
    "UploadDescriptor",
    "UploadResult",
    "VectorSpace",
    "VectorStore",
]
