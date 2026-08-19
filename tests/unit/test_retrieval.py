"""Unit tests for deterministic semantic retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from domain import (
    AppConfig,
    ChunkingProfile,
    PublicError,
    RawCandidate,
    RetrievedChunk,
    VectorSpace,
)
from rag import RetrievalService, cosine_distance_to_relevance


class _EmbeddingFake:
    def __init__(self) -> None:
        self.space = VectorSpace(
            model="local-test-embedding",
            dimension=3,
            normalized=True,
            metric="cosine",
        )
        self.ready_calls = 0
        self.query_calls: list[str] = []

    def ensure_ready(self) -> VectorSpace:
        self.ready_calls += 1
        return self.space

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.2, -0.1, 0.7]


class _VectorStoreFake:
    def __init__(
        self,
        *,
        count: int,
        candidates: tuple[RawCandidate, ...] = (),
    ) -> None:
        self.count = count
        self.candidates = candidates
        self.compatibility_calls: list[tuple[VectorSpace, ChunkingProfile]] = []
        self.count_calls = 0
        self.query_calls: list[tuple[tuple[float, ...], int]] = []

    def ensure_compatible(
        self,
        space: VectorSpace,
        profile: ChunkingProfile,
    ) -> None:
        self.compatibility_calls.append((space, profile))

    def count_chunks(self) -> int:
        self.count_calls += 1
        return self.count

    def query(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[RawCandidate]:
        self.query_calls.append((tuple(embedding), limit))
        return list(self.candidates)


def _config(
    root: Path,
    *,
    top_k: int = 6,
    threshold: float = 0.5,
) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="local-test-generation",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local-test-embedding",
        top_k=top_k,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=threshold,
        max_upload_bytes=1_048_576,
        max_zip_entries=10,
        max_zip_entry_bytes=1_048_576,
        max_uncompressed_bytes=2_097_152,
        max_compression_ratio=100.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=2,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _candidate(index: int, name: str, distance: float) -> RawCandidate:
    return RawCandidate(
        original_index=index,
        chunk_id=f"chunk-{name}",
        document=f"texto-{name}",
        metadata={
            "record_type": "chunk",
            "document_id": f"document-{name}",
            "display_name": f"{name}.pdf",
            "page": index + 1,
            "start_offset": index * 100,
            "transaction_id": "transaction-1",
            "chunk_schema_version": "char-v1",
        },
        distance=distance,
    )


@pytest.mark.parametrize(
    ("available", "expected_limit"),
    ((0, None), (3, 3), (6, 6), (9, 6)),
)
def test_retrieve_embeds_in_the_registered_space_and_requests_minimum_cardinality(
    tmp_path: Path,
    available: int,
    expected_limit: int | None,
) -> None:
    embedding = _EmbeddingFake()
    store = _VectorStoreFake(count=available)
    service = RetrievalService(_config(tmp_path), embedding, store)

    result = service.retrieve("pergunta validada")

    assert result == ()
    assert embedding.ready_calls == 1
    assert embedding.query_calls == ["pergunta validada"]
    assert store.compatibility_calls == [
        (
            embedding.space,
            ChunkingProfile(size=800, overlap=150, schema_version="char-v1"),
        )
    ]
    assert store.count_calls == 1
    if expected_limit is None:
        assert store.query_calls == []
    else:
        assert store.query_calls == [((0.2, -0.1, 0.7), expected_limit)]


@pytest.mark.parametrize(("top_k", "accepted"), ((4, False), (5, True), (8, True), (9, False)))
def test_retrieval_enforces_the_configured_top_k_range(
    tmp_path: Path,
    top_k: int,
    accepted: bool,
) -> None:
    arguments = (_config(tmp_path, top_k=top_k), _EmbeddingFake(), _VectorStoreFake(count=0))

    if accepted:
        assert isinstance(RetrievalService(*arguments), RetrievalService)
    else:
        with pytest.raises(ValueError, match="top_k"):
            RetrievalService(*arguments)


@pytest.mark.parametrize(
    ("distance", "expected"),
    ((-0.5, 1.0), (0.0, 1.0), (0.25, 0.75), (1.0, 0.0), (2.5, -1.0)),
)
def test_cosine_distance_is_converted_and_clamped(
    distance: float,
    expected: float,
) -> None:
    assert cosine_distance_to_relevance(distance) == pytest.approx(expected)


@pytest.mark.parametrize("distance", (float("nan"), float("inf"), float("-inf"), True))
def test_cosine_distance_rejects_non_finite_or_non_numeric_values(
    distance: float,
) -> None:
    with pytest.raises(ValueError, match="finite real"):
        cosine_distance_to_relevance(distance)


def test_retrieve_orders_stably_filters_at_threshold_and_confines_context(
    tmp_path: Path,
) -> None:
    candidates = (
        _candidate(0, "tie-first", 0.5),
        _candidate(1, "highest", 0.1),
        _candidate(2, "tie-second", 0.5),
        _candidate(3, "below-threshold", 0.500_001),
    )
    store = _VectorStoreFake(count=len(candidates), candidates=candidates)
    service = RetrievalService(
        _config(tmp_path, threshold=0.5),
        _EmbeddingFake(),
        store,
    )

    result = service.retrieve("qual trecho é relevante?")

    assert result == (
        RetrievedChunk(
            chunk_id="chunk-highest",
            text="texto-highest",
            document_id="document-highest",
            display_name="highest.pdf",
            human_page=2,
            score=pytest.approx(0.9),
        ),
        RetrievedChunk(
            chunk_id="chunk-tie-first",
            text="texto-tie-first",
            document_id="document-tie-first",
            display_name="tie-first.pdf",
            human_page=1,
            score=pytest.approx(0.5),
        ),
        RetrievedChunk(
            chunk_id="chunk-tie-second",
            text="texto-tie-second",
            document_id="document-tie-second",
            display_name="tie-second.pdf",
            human_page=3,
            score=pytest.approx(0.5),
        ),
    )
    assert all(chunk.chunk_id != "chunk-below-threshold" for chunk in result)
    assert store.query_calls == [((0.2, -0.1, 0.7), 4)]


def test_retrieve_maps_malformed_distance_to_vector_store_unavailable(
    tmp_path: Path,
) -> None:
    store = _VectorStoreFake(
        count=1,
        candidates=(_candidate(0, "invalid", float("nan")),),
    )
    service = RetrievalService(_config(tmp_path), _EmbeddingFake(), store)

    with pytest.raises(PublicError) as captured:
        service.retrieve("pergunta")

    assert captured.value.code == "vector_store_unavailable"
    assert captured.value.http_status == 503
