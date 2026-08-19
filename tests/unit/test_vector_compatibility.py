"""Unit tests for vector fingerprints and collection compatibility guards."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from config import (
    validate_chunking_profile_compatibility,
    validate_collection_compatibility,
    validate_vector_space_compatibility,
)
from domain import AppConfig, ChunkingProfile, PublicError, VectorSpace
from rag import ChromaVectorStore, ManifestStore


def test_vector_space_fingerprint_changes_with_the_distance_metric() -> None:
    space = _space()
    incompatible_metric = replace(space, metric="l2")  # type: ignore[arg-type]

    assert incompatible_metric.fingerprint != space.fingerprint


def test_pure_guards_classify_missing_vector_and_profile_metadata() -> None:
    missing_vector = _metadata()
    del missing_vector["embedding_dimension"]
    with pytest.raises(PublicError) as vector_error:
        validate_vector_space_compatibility(_space(), missing_vector)
    assert vector_error.value.code == "vector_space_mismatch"

    missing_profile = _metadata()
    del missing_profile["chunk_overlap"]
    with pytest.raises(PublicError) as profile_error:
        validate_chunking_profile_compatibility(_profile(), missing_profile)
    assert profile_error.value.code == "chunk_profile_mismatch"


@pytest.mark.parametrize(
    "missing_field",
    ["chunk_schema_version", "chunk_size", "chunk_overlap"],
)
def test_chroma_classifies_missing_profile_metadata_without_manifest_write(
    tmp_path: Path,
    missing_field: str,
) -> None:
    config = _config(tmp_path)
    metadata = _metadata()
    del metadata[missing_field]
    client = _FakeClient(_FakeCollection(metadata))
    manifest = ManifestStore(config)
    store = ChromaVectorStore(
        config,
        manifest,
        client_factory=lambda **kwargs: client,
        recover_on_startup=False,
    )

    with pytest.raises(PublicError) as captured:
        store.ensure_compatible(_space(), _profile())

    assert captured.value.code == "chunk_profile_mismatch"
    assert captured.value.http_status == 503
    assert manifest.get_vector_space() is None


def test_chroma_open_failure_is_public_and_does_not_write_manifest(
    tmp_path: Path,
) -> None:
    class FailingClient:
        def get_or_create_collection(self, **kwargs: object) -> object:
            raise RuntimeError("COLLECTION_INTERNAL_CANARY")

    config = _config(tmp_path)
    manifest = ManifestStore(config)
    store = ChromaVectorStore(
        config,
        manifest,
        client_factory=lambda **kwargs: FailingClient(),  # type: ignore[arg-type]
        recover_on_startup=False,
    )

    with pytest.raises(PublicError) as captured:
        store.ensure_compatible(_space(), _profile())

    assert captured.value.code == "vector_store_unavailable"
    assert captured.value.http_status == 503
    assert "COLLECTION_INTERNAL_CANARY" not in captured.value.message
    assert str(tmp_path) not in captured.value.message
    assert manifest.get_vector_space() is None


def test_chroma_client_factory_failure_is_public_and_sanitized(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = ManifestStore(config)

    def failing_factory(**kwargs: object) -> object:
        raise OSError(f"CLIENT_FACTORY_CANARY:{tmp_path}")

    with pytest.raises(PublicError) as captured:
        ChromaVectorStore(
            config,
            manifest,
            client_factory=failing_factory,  # type: ignore[arg-type]
            recover_on_startup=False,
        )

    assert captured.value.code == "vector_store_unavailable"
    assert captured.value.http_status == 503
    assert "CLIENT_FACTORY_CANARY" not in captured.value.message
    assert str(tmp_path) not in captured.value.message


def _config(root: Path) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="test-generation-model",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local/test-model",
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


def _space() -> VectorSpace:
    return VectorSpace(
        model="local/test-model",
        dimension=3,
        normalized=True,
        metric="cosine",
    )


def _profile() -> ChunkingProfile:
    return ChunkingProfile(size=800, overlap=150, schema_version="char-v1")


def _metadata(
    space: VectorSpace | None = None,
    profile: ChunkingProfile | None = None,
) -> dict[str, object]:
    actual_space = space or _space()
    actual_profile = profile or _profile()
    return {
        "schema_version": 1,
        "embedding_model": actual_space.model,
        "embedding_dimension": actual_space.dimension,
        "embedding_normalized": actual_space.normalized,
        "distance_metric": actual_space.metric,
        "chunk_schema_version": actual_profile.schema_version,
        "chunk_size": actual_profile.size,
        "chunk_overlap": actual_profile.overlap,
    }


class _FakeCollection:
    def __init__(
        self,
        metadata: dict[str, object],
        *,
        distance_space: str = "cosine",
    ) -> None:
        self.metadata = metadata
        self.configuration_json = {"hnsw": {"space": distance_space}}


class _FakeClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection
        self.create_calls: list[dict[str, object]] = []

    def get_or_create_collection(self, **kwargs: object) -> _FakeCollection:
        self.create_calls.append(dict(kwargs))
        return self.collection

    def get_collection(self, **kwargs: object) -> _FakeCollection:
        return self.collection

    def get_max_batch_size(self) -> int:
        return 100


def test_vector_space_fingerprint_is_canonical_stable_and_contract_sensitive() -> None:
    space = _space()
    canonical = (
        '{"dimension":3,"metric":"cosine","model":"local/test-model",'
        '"normalized":true}'
    )

    assert space.fingerprint == sha256(canonical.encode("utf-8")).hexdigest()
    assert space.fingerprint == _space().fingerprint
    assert len(space.fingerprint) == 64
    assert len(
        {
            space.fingerprint,
            replace(space, model="local/other-model").fingerprint,
            replace(space, dimension=4).fingerprint,
            replace(space, normalized=False).fingerprint,
        }
    ) == 4


def test_pure_compatibility_guards_accept_exact_metadata() -> None:
    config = _config(Path("/unused"))
    space = _space()
    profile = _profile()
    metadata = _metadata(space, profile)

    validate_vector_space_compatibility(space, metadata)
    validate_chunking_profile_compatibility(profile, metadata)
    validate_collection_compatibility(config, space, metadata)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("embedding_model", "local/PERSISTED_METADATA_CANARY"),
        ("embedding_dimension", 4),
        ("embedding_dimension", True),
        ("embedding_normalized", False),
        ("embedding_normalized", 1),
        ("distance_metric", "l2"),
    ],
)
def test_vector_metadata_mismatch_is_a_sanitized_public_error(
    field: str,
    bad_value: object,
) -> None:
    metadata = _metadata()
    metadata[field] = bad_value

    with pytest.raises(PublicError) as captured:
        validate_vector_space_compatibility(_space(), metadata)

    assert captured.value.code == "vector_space_mismatch"
    assert captured.value.http_status == 503
    assert "PERSISTED_METADATA_CANARY" not in captured.value.message


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("chunk_size", 801),
        ("chunk_size", True),
        ("chunk_overlap", 149),
        ("chunk_schema_version", "future-v2"),
    ],
)
def test_chunk_metadata_mismatch_is_a_sanitized_public_error(
    field: str,
    bad_value: object,
) -> None:
    metadata = _metadata()
    metadata[field] = bad_value

    with pytest.raises(PublicError) as captured:
        validate_chunking_profile_compatibility(_profile(), metadata)

    assert captured.value.code == "chunk_profile_mismatch"
    assert captured.value.http_status == 503
    assert "future-v2" not in captured.value.message


def test_collection_guard_checks_configured_model_and_chunk_profile() -> None:
    metadata = _metadata()
    config = _config(Path("/unused"))

    with pytest.raises(PublicError) as model_error:
        validate_collection_compatibility(
            replace(config, embedding_model="local/other-model"),
            _space(),
            metadata,
        )
    assert model_error.value.code == "vector_space_mismatch"

    metadata["chunk_overlap"] = 0
    with pytest.raises(PublicError) as profile_error:
        validate_collection_compatibility(config, _space(), metadata)
    assert profile_error.value.code == "chunk_profile_mismatch"


def test_fake_chroma_client_receives_cosine_and_immutable_contract_metadata(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    space = _space()
    profile = _profile()
    collection = _FakeCollection(_metadata(space, profile))
    client = _FakeClient(collection)
    factory_paths: list[str] = []

    def factory(*, path: str) -> _FakeClient:
        factory_paths.append(path)
        return client

    manifest = ManifestStore(config)
    store = ChromaVectorStore(
        config,
        manifest,
        client_factory=factory,
        recover_on_startup=False,
    )
    store.ensure_compatible(space, profile)

    assert factory_paths == [str(config.chroma_path)]
    assert client.create_calls == [
        {
            "name": config.chroma_collection,
            "configuration": {"hnsw": {"space": "cosine"}},
            "metadata": _metadata(space, profile),
            "embedding_function": None,
        }
    ]
    assert manifest.get_vector_space() == (space, profile)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing_schema", "vector_space_mismatch"),
        ("dimension", "vector_space_mismatch"),
        ("normalization_type", "vector_space_mismatch"),
        ("distance", "vector_space_mismatch"),
        ("chunk_size", "chunk_profile_mismatch"),
    ],
)
def test_fake_chroma_rejects_incompatible_collection_before_manifest_write(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    config = _config(tmp_path)
    metadata = _metadata()
    distance_space = "cosine"
    if case == "missing_schema":
        del metadata["schema_version"]
    elif case == "dimension":
        metadata["embedding_dimension"] = 4
    elif case == "normalization_type":
        metadata["embedding_normalized"] = 1
    elif case == "distance":
        distance_space = "l2"
    elif case == "chunk_size":
        metadata["chunk_size"] = 900

    client = _FakeClient(
        _FakeCollection(metadata, distance_space=distance_space)
    )
    manifest = ManifestStore(config)
    store = ChromaVectorStore(
        config,
        manifest,
        client_factory=lambda **kwargs: client,
        recover_on_startup=False,
    )

    with pytest.raises(PublicError) as captured:
        store.ensure_compatible(_space(), _profile())

    assert captured.value.code == expected_code
    assert captured.value.http_status == 503
    assert manifest.get_vector_space() is None
