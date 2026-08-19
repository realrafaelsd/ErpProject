"""Property test for retrieval cardinality, scoring, ordering, and filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hypothesis import given, settings, strategies as st

from domain import (
    AppConfig,
    ChunkingProfile,
    RawCandidate,
    RetrievedChunk,
    VectorSpace,
)
from rag import RetrievalService, cosine_distance_to_relevance


_SPACE = VectorSpace(
    model="local/property-retrieval",
    dimension=3,
    normalized=True,
    metric="cosine",
)
_PROFILE = ChunkingProfile(size=800, overlap=150, schema_version="char-v1")
_QUERY_EMBEDDING = (0.25, -0.5, 0.75)
_THRESHOLDS = st.sampled_from(
    (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
)
_FINITE_DISTANCES = st.one_of(
    st.sampled_from((-100.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 100.0)),
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)


@dataclass(frozen=True, slots=True)
class _RetrievalCase:
    case_id: int
    top_k: int
    threshold: float
    distances: tuple[float, ...]


@st.composite
def _retrieval_cases(draw) -> _RetrievalCase:
    top_k = draw(st.integers(min_value=5, max_value=8))
    available_count = draw(
        st.one_of(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=5, max_value=16),
        )
    )
    threshold = draw(_THRESHOLDS)
    distances = draw(
        st.lists(
            _FINITE_DISTANCES,
            min_size=available_count,
            max_size=available_count,
        )
    )
    requested = min(top_k, available_count)

    # Keep deliberate boundary cases inside the portion returned by the fake
    # store while retaining arbitrary finite distances elsewhere.
    if requested >= 3:
        distances[2] = 1.0 - threshold
    if requested >= 4:
        distances[3] = -5.0
    if requested >= 5:
        distances[4] = 5.0
    if requested >= 2:
        tied_distance = draw(_FINITE_DISTANCES)
        distances[0] = tied_distance
        distances[1] = tied_distance

    return _RetrievalCase(
        case_id=draw(st.integers(min_value=0, max_value=1_000_000)),
        top_k=top_k,
        threshold=threshold,
        distances=tuple(distances),
    )


def _config(case: _RetrievalCase) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_retrieval",
        upload_folder=Path("/unused/uploads"),
        embedding_model=_SPACE.model,
        top_k=case.top_k,
        chunk_size=_PROFILE.size,
        chunk_overlap=_PROFILE.overlap,
        relevance_threshold=case.threshold,
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


def _candidates(case: _RetrievalCase) -> tuple[RawCandidate, ...]:
    return tuple(
        RawCandidate(
            original_index=index,
            chunk_id=f"chunk-{case.case_id}-{index}",
            document=f"texto exclusivo {case.case_id}:{index}",
            metadata={
                "record_type": "chunk",
                "document_id": f"document-{case.case_id}-{index}",
                "display_name": f"manuais/manual-{case.case_id}-{index}.pdf",
                "page": index + 1,
                "start_offset": index * 100,
                "transaction_id": f"transaction-{case.case_id}",
                "chunk_schema_version": "char-v1",
            },
            distance=distance,
        )
        for index, distance in enumerate(case.distances)
    )


class _EmbeddingFake:
    def __init__(self) -> None:
        self.ready_calls = 0
        self.query_calls: list[str] = []

    def ensure_ready(self) -> VectorSpace:
        self.ready_calls += 1
        return _SPACE

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return list(_QUERY_EMBEDDING)


class _VectorStoreFake:
    def __init__(self, candidates: Sequence[RawCandidate]) -> None:
        self._candidates = tuple(candidates)
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
        return len(self._candidates)

    def query(
        self,
        embedding: Sequence[float],
        limit: int,
    ) -> list[RawCandidate]:
        self.query_calls.append((tuple(float(value) for value in embedding), limit))
        return list(self._candidates[:limit])


def _expected_score(distance: float) -> float:
    return max(-1.0, min(1.0, 1.0 - distance))


def _retrieved(candidate: RawCandidate) -> RetrievedChunk:
    metadata = candidate.metadata
    return RetrievedChunk(
        chunk_id=candidate.chunk_id,
        text=candidate.document,
        document_id=cast(str, metadata["document_id"]),
        display_name=cast(str, metadata["display_name"]),
        human_page=cast(int, metadata["page"]),
        score=_expected_score(candidate.distance),
    )


@settings(max_examples=150, deadline=None)
@given(case=_retrieval_cases())
def test_property_13_retrieval_respects_cardinality_score_order_and_threshold(
    case: _RetrievalCase,
) -> None:
    # Feature: erp-ai-support, Property 13: Recuperação respeita cardinalidade, score, ordem e limiar
    # **Validates: Requirements 12.2, 12.3, 12.4, 12.5, 12.6, 12.9**
    candidates = _candidates(case)
    embeddings = _EmbeddingFake()
    store = _VectorStoreFake(candidates)
    service = RetrievalService(_config(case), embeddings, store)

    context = service.retrieve("pergunta independente")

    requested = min(case.top_k, len(candidates))
    queried = candidates[:requested]
    ranked = sorted(
        queried,
        key=lambda candidate: _expected_score(candidate.distance),
        reverse=True,
    )
    expected_context = tuple(
        _retrieved(candidate)
        for candidate in ranked
        if _expected_score(candidate.distance) >= case.threshold
    )

    assert all(math.isfinite(distance) for distance in case.distances)
    assert all(
        cosine_distance_to_relevance(distance) == _expected_score(distance)
        for distance in case.distances
    )
    assert embeddings.ready_calls == 1
    assert embeddings.query_calls == ["pergunta independente"]
    assert store.compatibility_calls == [(_SPACE, _PROFILE)]
    assert store.count_calls == 1
    assert store.query_calls == [(_QUERY_EMBEDDING, requested)]

    # Exact equality checks order, filtering, score, metadata projection, and
    # that no text from an unqueried candidate is appended to the context.
    assert context == expected_context
    assert len(context) <= requested
    assert all(-1.0 <= chunk.score <= 1.0 for chunk in context)
    assert all(chunk.score >= case.threshold for chunk in context)
    assert {chunk.chunk_id for chunk in context} == {
        candidate.chunk_id
        for candidate in queried
        if _expected_score(candidate.distance) >= case.threshold
    }
    assert {chunk.chunk_id for chunk in context}.isdisjoint(
        candidate.chunk_id for candidate in candidates[requested:]
    )

    accepted_scores = {
        _expected_score(candidate.distance)
        for candidate in queried
        if _expected_score(candidate.distance) >= case.threshold
    }
    for score in accepted_scores:
        assert [chunk.chunk_id for chunk in context if chunk.score == score] == [
            candidate.chunk_id
            for candidate in queried
            if _expected_score(candidate.distance) == score
        ]
