"""Property test for generation short-circuit on empty retrieval context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hypothesis import given, settings, strategies as st

from domain import AppConfig, ChunkingProfile, RawCandidate, VectorSpace
from rag import (
    INSUFFICIENT_ANSWER,
    RAGService,
    RetrievalService,
    cosine_distance_to_relevance,
)


_THRESHOLD = 0.3
_SPACE = VectorSpace(
    model="local/property-empty-context",
    dimension=3,
    normalized=True,
    metric="cosine",
)
_Mode = Literal["empty_collection", "empty_candidates", "below_threshold"]
_QUESTION = st.text(
    alphabet=st.sampled_from(tuple("ERP cadastro consulta usuário áéíóúç 0123456789?")),
    min_size=1,
    max_size=80,
).filter(lambda value: bool(value.strip()))


@dataclass(frozen=True, slots=True)
class _EmptyContextCase:
    mode: _Mode
    top_k: int
    collection_size: int
    candidates: tuple[RawCandidate, ...]
    question: str


@st.composite
def _empty_context_cases(draw):
    mode = draw(
        st.sampled_from(
            ("empty_collection", "empty_candidates", "below_threshold")
        )
    )
    top_k = draw(st.integers(min_value=5, max_value=8))
    question = draw(_QUESTION)

    if mode == "empty_collection":
        return _EmptyContextCase(
            mode=mode,
            top_k=top_k,
            collection_size=0,
            candidates=(),
            question=question,
        )

    collection_size = draw(st.integers(min_value=1, max_value=16))
    limit = min(top_k, collection_size)
    if mode == "empty_candidates":
        return _EmptyContextCase(
            mode=mode,
            top_k=top_k,
            collection_size=collection_size,
            candidates=(),
            question=question,
        )

    candidate_count = draw(st.integers(min_value=1, max_value=limit))
    margins = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_300),
            min_size=candidate_count,
            max_size=candidate_count,
        )
    )
    candidates = tuple(
        RawCandidate(
            original_index=index,
            chunk_id=f"chunk-{index}",
            document=f"conteúdo recuperado {index}",
            metadata={
                "record_type": "chunk",
                "document_id": f"document-{index}",
                "display_name": f"manuais/manual-{index}.pdf",
                "page": index + 1,
                "start_offset": index * 100,
                "transaction_id": "tx-property-14",
                "chunk_schema_version": "char-v1",
            },
            distance=0.7 + (margin / 1_000),
        )
        for index, margin in enumerate(margins)
    )
    return _EmptyContextCase(
        mode=mode,
        top_k=top_k,
        collection_size=collection_size,
        candidates=candidates,
        question=question,
    )


def _config(top_k: int) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_empty_context",
        upload_folder=Path("/unused/uploads"),
        embedding_model=_SPACE.model,
        top_k=top_k,
        chunk_size=800,
        chunk_overlap=150,
        relevance_threshold=_THRESHOLD,
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


class _EmbeddingSpy:
    def __init__(self) -> None:
        self.ready_calls = 0
        self.query_calls: list[str] = []

    def ensure_ready(self) -> VectorSpace:
        self.ready_calls += 1
        return _SPACE

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0, 0.0]


class _VectorStoreFake:
    def __init__(self, case: _EmptyContextCase) -> None:
        self._case = case
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
        return self._case.collection_size

    def query(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[RawCandidate]:
        self.query_calls.append((tuple(embedding), limit))
        return list(self._case.candidates)


class _GeneratorSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:
        self.calls.append((prompt, max_tokens, timeout_seconds))
        raise AssertionError("generation must not run for empty context")


@settings(max_examples=150, deadline=None)
@given(case=_empty_context_cases())
def test_property_14_empty_context_stops_generation(
    case: _EmptyContextCase,
) -> None:
    # Feature: erp-ai-support, Property 14: Contexto vazio interrompe geração
    # **Validates: Requirements 12.7, 12.8, 12.11**
    config = _config(case.top_k)
    embeddings = _EmbeddingSpy()
    vector_store = _VectorStoreFake(case)
    generator = _GeneratorSpy()
    retrieval = RetrievalService(config, embeddings, vector_store)
    service = RAGService(config, retrieval, generator)

    assert all(
        cosine_distance_to_relevance(candidate.distance) < _THRESHOLD
        for candidate in case.candidates
    )

    result = service.answer(case.question)

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == ()
    assert generator.calls == []
    assert embeddings.ready_calls == 1
    assert embeddings.query_calls == [case.question]
    assert vector_store.count_calls == 1
    assert len(vector_store.compatibility_calls) == 1

    if case.collection_size == 0:
        assert vector_store.query_calls == []
    else:
        assert vector_store.query_calls == [
            ((1.0, 0.0, 0.0), min(case.top_k, case.collection_size))
        ]
