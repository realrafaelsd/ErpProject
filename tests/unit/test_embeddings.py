"""Unit tests for the strictly local embedding service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest

from domain import AppConfig, PublicError, VectorSpace
from rag import LocalEmbeddingService


def _config(model_name: str = "local/test-model") -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generation-model",
        chroma_path=Path("/unused/chroma"),
        chroma_collection="test_collection",
        upload_folder=Path("/unused/uploads"),
        embedding_model=model_name,
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
    def __init__(self, dimension: int = 3, output: object | None = None) -> None:
        self.dimension = dimension
        self.output = output
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> object:
        self.encode_calls.append((list(texts), dict(kwargs)))
        if self.output is not None:
            return self.output
        return [[float(index + 1) for index in range(self.dimension)] for _ in texts]


def test_loads_once_and_uses_normalized_local_encoding_for_documents_and_query() -> None:
    model = _FakeModel()
    factory_calls: list[tuple[str, dict[str, object]]] = []

    def factory(model_name: str, **kwargs: object) -> _FakeModel:
        factory_calls.append((model_name, dict(kwargs)))
        return model

    service = LocalEmbeddingService(_config(), model_factory=factory)

    expected_space = VectorSpace(
        model="local/test-model",
        dimension=3,
        normalized=True,
        metric="cosine",
    )
    assert service.ensure_ready() == expected_space
    assert service.ensure_ready() is service.ensure_ready()
    assert service.embed_documents(["primeiro", "segundo"]) == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
    assert service.embed_query("pergunta") == [1.0, 2.0, 3.0]

    assert factory_calls == [
        (
            "local/test-model",
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert model.encode_calls == [
        (
            ["primeiro", "segundo"],
            {
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        ),
        (
            ["pergunta"],
            {
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        ),
    ]


def test_concurrent_readiness_publishes_only_one_loaded_model() -> None:
    entered_factory = Event()
    release_factory = Event()
    counter_lock = Lock()
    factory_calls = 0

    def factory(model_name: str, **kwargs: object) -> _FakeModel:
        nonlocal factory_calls
        with counter_lock:
            factory_calls += 1
        entered_factory.set()
        assert release_factory.wait(timeout=2)
        return _FakeModel(dimension=2)

    service = LocalEmbeddingService(_config(), model_factory=factory)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(service.ensure_ready) for _ in range(4)]
        assert entered_factory.wait(timeout=2)
        release_factory.set()
        spaces = [future.result(timeout=2) for future in futures]

    assert factory_calls == 1
    assert spaces == [spaces[0]] * 4
    assert all(space.dimension == 2 for space in spaces)


def test_missing_local_model_is_mapped_without_internal_details() -> None:
    def unavailable_factory(model_name: str, **kwargs: object) -> object:
        raise OSError("INTERNAL_PATH_CANARY")

    service = LocalEmbeddingService(
        _config("local/missing-model"),
        model_factory=unavailable_factory,
    )

    with pytest.raises(PublicError) as captured:
        service.ensure_ready()

    assert captured.value.code == "embedding_model_missing"
    assert captured.value.http_status == 503
    assert "local/missing-model" in captured.value.message
    assert "INTERNAL_PATH_CANARY" not in captured.value.message


@pytest.mark.parametrize(
    "output",
    [
        [],
        [[1.0, 2.0]],
        [[1.0, 2.0, float("nan")]],
        [[1.0, 2.0, float("inf")]],
        [[1.0, 2.0, "3.0"]],
        [[1.0, 2.0, True]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, float("nan")]],
    ],
)
def test_rejects_invalid_quantity_components_and_dimensions(output: object) -> None:
    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: _FakeModel(output=output),
    )

    with pytest.raises(PublicError) as captured:
        service.embed_documents(["texto"])

    assert captured.value.code == "embedding_failed"
    assert captured.value.http_status == 503


@pytest.mark.parametrize("dimension", [0, -1, True, None])
def test_rejects_invalid_reported_dimension(dimension: object) -> None:
    model = _FakeModel()
    model.dimension = dimension  # type: ignore[assignment]
    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: model,
    )

    with pytest.raises(PublicError) as captured:
        service.ensure_ready()

    assert captured.value.code == "embedding_failed"


def test_inference_failure_returns_no_partial_result() -> None:
    class FailingModel(_FakeModel):
        def encode(self, texts: list[str], **kwargs: object) -> object:
            raise RuntimeError("PARTIAL_VECTOR_CANARY")

    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: FailingModel(),
    )

    with pytest.raises(PublicError) as captured:
        service.embed_query("pergunta")

    assert captured.value.code == "embedding_failed"
    assert "PARTIAL_VECTOR_CANARY" not in captured.value.message


@pytest.mark.parametrize(
    "invalid_texts",
    ["texto isolado", b"bytes", ["válido", 7]],
)
def test_rejects_invalid_document_input_before_loading_model(
    invalid_texts: object,
) -> None:
    factory_calls = 0

    def factory(model_name: str, **kwargs: object) -> _FakeModel:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeModel()

    service = LocalEmbeddingService(_config(), model_factory=factory)

    with pytest.raises(PublicError) as captured:
        service.embed_documents(invalid_texts)  # type: ignore[arg-type]

    assert captured.value.code == "embedding_failed"
    assert captured.value.http_status == 503
    assert factory_calls == 0


def test_rejects_invalid_query_type_before_loading_model() -> None:
    factory_calls = 0

    def factory(model_name: str, **kwargs: object) -> _FakeModel:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeModel()

    service = LocalEmbeddingService(_config(), model_factory=factory)

    with pytest.raises(PublicError) as captured:
        service.embed_query(42)  # type: ignore[arg-type]

    assert captured.value.code == "embedding_failed"
    assert factory_calls == 0


@pytest.mark.parametrize(
    "output",
    [
        object(),
        {"vector": [1.0, 2.0, 3.0]},
        [object()],
        [[1.0, 2.0, 3.0, 4.0]],
        [[1.0, 2.0, float("-inf")]],
    ],
)
def test_rejects_malformed_vector_containers(output: object) -> None:
    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: _FakeModel(output=output),
    )

    with pytest.raises(PublicError) as captured:
        service.embed_query("pergunta")

    assert captured.value.code == "embedding_failed"
    assert captured.value.http_status == 503


def test_dimension_discovery_failure_is_sanitized() -> None:
    class BrokenDimensionModel(_FakeModel):
        def get_sentence_embedding_dimension(self) -> int:
            raise RuntimeError("MODEL_CACHE_PATH_CANARY")

    service = LocalEmbeddingService(
        _config("local/broken-model"),
        model_factory=lambda *args, **kwargs: BrokenDimensionModel(),
    )

    with pytest.raises(PublicError) as captured:
        service.ensure_ready()

    assert captured.value.code == "embedding_failed"
    assert captured.value.http_status == 503
    assert "local/broken-model" in captured.value.message
    assert "MODEL_CACHE_PATH_CANARY" not in captured.value.message


def test_empty_document_batch_never_invokes_inference() -> None:
    model = _FakeModel()
    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: model,
    )

    assert service.embed_documents([]) == []
    assert model.encode_calls == []


def test_failed_model_load_is_not_published_and_a_retry_can_succeed() -> None:
    model = _FakeModel(dimension=2)
    factory_calls: list[dict[str, object]] = []

    def factory(model_name: str, **kwargs: object) -> _FakeModel:
        factory_calls.append(dict(kwargs))
        if len(factory_calls) == 1:
            raise OSError("MODEL_CACHE_INTERNAL_CANARY")
        return model

    service = LocalEmbeddingService(_config(), model_factory=factory)

    with pytest.raises(PublicError) as captured:
        service.ensure_ready()

    assert captured.value.code == "embedding_model_missing"
    assert "MODEL_CACHE_INTERNAL_CANARY" not in captured.value.message
    assert service.ensure_ready().dimension == 2
    assert service.ensure_ready().dimension == 2
    assert factory_calls == [
        {"local_files_only": True, "trust_remote_code": False},
        {"local_files_only": True, "trust_remote_code": False},
    ]


def test_rejects_document_batch_cardinality_mismatch_without_partial_result() -> None:
    model = _FakeModel(output=[[1.0, 2.0, 3.0]])
    service = LocalEmbeddingService(
        _config(),
        model_factory=lambda *args, **kwargs: model,
    )

    with pytest.raises(PublicError) as captured:
        service.embed_documents(["primeiro", "segundo"])

    assert captured.value.code == "embedding_failed"
    assert captured.value.http_status == 503
    assert model.encode_calls[0][0] == ["primeiro", "segundo"]
