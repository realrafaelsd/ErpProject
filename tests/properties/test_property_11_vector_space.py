"""Property test for vector-space consistency at external boundaries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from hypothesis import given, settings, strategies as st

from config import validate_vector_space_compatibility
from domain import AppConfig, DistanceMetric, PublicError, VectorSpace
from rag import LocalEmbeddingService


_MODEL_NAME = "local/property-model"
_FINITE_COMPONENTS = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_NON_FINITE_COMPONENTS = st.sampled_from(
    [float("nan"), float("inf"), float("-inf")]
)


def _config(dimension: int) -> AppConfig:
    """Build a complete config; only the embedding identity is exercised."""

    del dimension
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="unused-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="property_collection",
        upload_folder=Path("/unused/uploads"),
        embedding_model=_MODEL_NAME,
        top_k=6,
        chunk_size=800,
        chunk_overlap=150,
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


class _FakeModel:
    """Return one generated vector without loading a model or using network."""

    def __init__(self, dimension: int, vector: tuple[float, ...]) -> None:
        self._dimension = dimension
        self._vector = vector

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        del kwargs
        return [list(self._vector) for _ in texts]


class _RecordingBoundary:
    """Record only vectors that passed every production guard."""

    def __init__(self) -> None:
        self.queries: list[tuple[float, ...]] = []
        self.persisted: list[tuple[float, ...]] = []

    def query(self, vector: list[float]) -> None:
        self.queries.append(tuple(vector))

    def persist(self, vector: list[float]) -> None:
        self.persisted.append(tuple(vector))


@st.composite
def _vector_cases(draw):
    registered_dimension = draw(st.integers(min_value=1, max_value=8))
    vector_is_compatible = draw(st.booleans())
    space_is_compatible = draw(st.booleans())

    if vector_is_compatible:
        vector = draw(
            st.lists(
                _FINITE_COMPONENTS,
                min_size=registered_dimension,
                max_size=registered_dimension,
            )
        )
    else:
        invalid_vector_kind = draw(st.sampled_from(("dimension", "non_finite")))
        if invalid_vector_kind == "dimension":
            invalid_dimension = draw(
                st.sampled_from(
                    (registered_dimension - 1, registered_dimension + 1)
                )
            )
            vector = draw(
                st.lists(
                    _FINITE_COMPONENTS,
                    min_size=invalid_dimension,
                    max_size=invalid_dimension,
                )
            )
        else:
            vector = draw(
                st.lists(
                    _FINITE_COMPONENTS,
                    min_size=registered_dimension,
                    max_size=registered_dimension,
                )
            )
            index = draw(st.integers(min_value=0, max_value=registered_dimension - 1))
            vector[index] = draw(_NON_FINITE_COMPONENTS)

    registered_space = VectorSpace(
        model=_MODEL_NAME,
        dimension=registered_dimension,
        normalized=True,
        metric="cosine",
    )
    if space_is_compatible:
        recorded_space = registered_space
    else:
        mismatch = draw(
            st.sampled_from(("model", "dimension", "normalization", "metric"))
        )
        recorded_space = VectorSpace(
            model=(
                f"{_MODEL_NAME}-other"
                if mismatch == "model"
                else registered_space.model
            ),
            dimension=(
                registered_dimension + 1
                if mismatch == "dimension"
                else registered_dimension
            ),
            normalized=False if mismatch == "normalization" else True,
            metric=cast(
                DistanceMetric,
                "l2" if mismatch == "metric" else "cosine",
            ),
        )

    return (
        registered_space,
        recorded_space,
        tuple(vector),
        vector_is_compatible,
        space_is_compatible,
    )


def _recorded_metadata(space: VectorSpace) -> dict[str, object]:
    return {
        "embedding_model": space.model,
        "embedding_dimension": space.dimension,
        "embedding_normalized": space.normalized,
        "distance_metric": space.metric,
    }


def _exercise_boundaries(
    registered_space: VectorSpace,
    recorded_space: VectorSpace,
    generated_vector: tuple[float, ...],
) -> tuple[_RecordingBoundary, tuple[str | None, str | None]]:
    boundary = _RecordingBoundary()
    errors: list[str | None] = []

    for operation in ("query", "persist"):
        model = _FakeModel(registered_space.dimension, generated_vector)
        service = LocalEmbeddingService(
            _config(registered_space.dimension),
            model_factory=lambda *args, _model=model, **kwargs: _model,
        )
        try:
            validate_vector_space_compatibility(
                registered_space,
                _recorded_metadata(recorded_space),
            )
            assert service.ensure_ready() == registered_space
            if operation == "query":
                boundary.query(service.embed_query("pergunta"))
            else:
                boundary.persist(service.embed_documents(["chunk"])[0])
        except PublicError as error:
            errors.append(error.code)
        else:
            errors.append(None)

    return boundary, (errors[0], errors[1])


@settings(max_examples=150, deadline=None)
@given(case=_vector_cases())
def test_property_11_every_accepted_vector_belongs_to_configured_space(case) -> None:
    # Feature: erp-ai-support, Property 11: Todo vetor aceito pertence ao espaço configurado
    (
        registered_space,
        recorded_space,
        generated_vector,
        vector_is_compatible,
        space_is_compatible,
    ) = case

    assert (
        recorded_space.fingerprint == registered_space.fingerprint
    ) is space_is_compatible
    assert (
        len(generated_vector) == registered_space.dimension
        and all(math.isfinite(component) for component in generated_vector)
    ) is vector_is_compatible

    boundary, errors = _exercise_boundaries(
        registered_space,
        recorded_space,
        generated_vector,
    )
    accepted = vector_is_compatible and space_is_compatible

    if accepted:
        expected = [tuple(float(value) for value in generated_vector)]
        assert errors == (None, None)
        assert boundary.queries == expected
        assert boundary.persisted == expected
    else:
        expected_code = (
            "vector_space_mismatch"
            if not space_is_compatible
            else "embedding_failed"
        )
        assert errors == (expected_code, expected_code)
        assert boundary.queries == []
        assert boundary.persisted == []
