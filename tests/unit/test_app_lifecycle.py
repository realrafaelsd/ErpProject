"""Focused tests for Flask composition, lifecycle, and public error handling."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from flask import Flask
from werkzeug.exceptions import RequestEntityTooLarge

import app as app_module
from domain import AppConfig, PublicError


def _config(root: Path) -> AppConfig:
    return AppConfig(
        ollama_url="http://localhost:11434",
        ollama_model="local-generator",
        chroma_path=root / "chroma",
        chroma_collection="app_lifecycle",
        upload_folder=root / "uploads",
        embedding_model="local/test-embedding",
        top_k=6,
        chunk_size=500,
        chunk_overlap=100,
        relevance_threshold=0.3,
        max_upload_bytes=1_024,
        max_zip_entries=20,
        max_zip_entry_bytes=1_048_576,
        max_uncompressed_bytes=4_194_304,
        max_compression_ratio=100.0,
        max_question_chars=2_000,
        ollama_timeout_seconds=120,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_432,
        flask_debug=False,
    )


def _injected_services() -> app_module.Services:
    # The injection seam deliberately accepts already-built service instances;
    # these inert sentinels prove handler tests create no concrete dependency.
    return app_module.Services(
        embedding_service=object(),
        manifest_store=object(),
        vector_store=object(),
        ingestion_service=object(),
        retrieval_service=object(),
        ollama_client=object(),
        rag_service=object(),
    )


def test_create_app_composes_each_service_once_recovers_before_routes_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    events: list[str] = []
    instances: dict[str, object] = {}

    class FakeLifecycleLock:
        def acquire(self) -> None:
            events.append("lifecycle.acquire")

        def release(self) -> None:
            events.append("lifecycle.release")

    class FakeLocks:
        def __init__(self, path: Path) -> None:
            assert path == config.chroma_path
            events.append("locks")
            self.lifecycle_lock = FakeLifecycleLock()

    class FakeVectorStore:
        def __init__(
            self,
            supplied_config: AppConfig,
            manifest_store: object,
            *,
            recover_on_startup: bool,
        ) -> None:
            assert supplied_config is config
            assert manifest_store is instances["manifest"]
            assert recover_on_startup is False
            events.append("vector")

        def recover_incomplete(self) -> None:
            events.append("recovery")

        def close(self) -> None:
            events.append("vector.close")

    def build_manifest(supplied_config: AppConfig, *, locks: object) -> object:
        assert supplied_config is config
        assert locks is instances["locks"]
        events.append("manifest")
        value = object()
        instances["manifest"] = value
        return value

    def build_embedding(supplied_config: AppConfig) -> object:
        assert supplied_config is config
        events.append("embedding")
        value = object()
        instances["embedding"] = value
        return value

    def build_ingestion(
        supplied_config: AppConfig,
        embedding: object,
        manifest: object,
        vector: object,
    ) -> object:
        assert supplied_config is config
        assert embedding is instances["embedding"]
        assert manifest is instances["manifest"]
        assert vector is instances["vector"]
        events.append("ingestion")
        value = object()
        instances["ingestion"] = value
        return value

    def build_retrieval(
        supplied_config: AppConfig,
        embedding: object,
        vector: object,
    ) -> object:
        assert supplied_config is config
        assert embedding is instances["embedding"]
        assert vector is instances["vector"]
        events.append("retrieval")
        value = object()
        instances["retrieval"] = value
        return value

    def build_ollama(supplied_config: AppConfig) -> object:
        assert supplied_config is config
        events.append("ollama")
        value = object()
        instances["ollama"] = value
        return value

    def build_rag(
        supplied_config: AppConfig,
        retrieval: object,
        generator: object,
    ) -> object:
        assert supplied_config is config
        assert retrieval is instances["retrieval"]
        assert generator is instances["ollama"]
        events.append("rag")
        value = object()
        instances["rag"] = value
        return value

    def build_locks(path: Path) -> FakeLocks:
        value = FakeLocks(path)
        instances["locks"] = value
        return value

    def build_vector(*args: object, **kwargs: object) -> FakeVectorStore:
        value = FakeVectorStore(*args, **kwargs)
        instances["vector"] = value
        return value

    registered: list[app_module.Services] = []

    def register_routes(flask_app: Flask, services: app_module.Services) -> None:
        assert isinstance(flask_app, Flask)
        assert events[-1] == "rag"
        assert "recovery" in events
        events.append("routes")
        registered.append(services)

    monkeypatch.setattr(app_module, "StorageLocks", build_locks)
    monkeypatch.setattr(app_module, "ManifestStore", build_manifest)
    monkeypatch.setattr(app_module, "ChromaVectorStore", build_vector)
    monkeypatch.setattr(app_module, "LocalEmbeddingService", build_embedding)
    monkeypatch.setattr(app_module, "IngestionService", build_ingestion)
    monkeypatch.setattr(app_module, "RetrievalService", build_retrieval)
    monkeypatch.setattr(app_module, "OllamaClient", build_ollama)
    monkeypatch.setattr(app_module, "RAGService", build_rag)
    monkeypatch.setattr(app_module, "register_routes", register_routes)

    flask_app = app_module.create_app(config)
    runtime = flask_app.extensions["erp_ai_support"]

    assert events == [
        "locks",
        "lifecycle.acquire",
        "manifest",
        "vector",
        "recovery",
        "embedding",
        "ingestion",
        "retrieval",
        "ollama",
        "rag",
        "routes",
    ]
    assert len(registered) == 1
    services = registered[0]
    assert services.embedding_service is instances["embedding"]
    assert services.manifest_store is instances["manifest"]
    assert services.vector_store is instances["vector"]
    assert services.ingestion_service is instances["ingestion"]
    assert services.retrieval_service is instances["retrieval"]
    assert services.ollama_client is instances["ollama"]
    assert services.rag_service is instances["rag"]
    assert runtime.services is services
    assert flask_app.config["MAX_CONTENT_LENGTH"] == 1_024 + 64 * 1_024

    app_module.close_app(flask_app)
    app_module.close_app(flask_app)

    assert events[-2:] == ["vector.close", "lifecycle.release"]
    assert events.count("vector.close") == 1
    assert events.count("lifecycle.release") == 1
    assert runtime.closed is True


def test_partial_composition_failure_releases_lifecycle_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    events: list[str] = []

    class FakeLifecycleLock:
        def acquire(self) -> None:
            events.append("acquire")

        def release(self) -> None:
            events.append("release")

    class FakeLocks:
        def __init__(self, path: Path) -> None:
            assert path == config.chroma_path
            self.lifecycle_lock = FakeLifecycleLock()

    def fail_manifest(*args: object, **kwargs: object) -> object:
        del args, kwargs
        events.append("manifest.failure")
        raise OSError("internal startup canary")

    monkeypatch.setattr(app_module, "StorageLocks", FakeLocks)
    monkeypatch.setattr(app_module, "ManifestStore", fail_manifest)

    with pytest.raises(OSError, match="internal startup canary"):
        app_module.create_app(config)

    assert events == ["acquire", "manifest.failure", "release"]


def test_error_handlers_return_exact_public_envelopes_and_log_only_locally(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(tmp_path)
    flask_app = app_module.create_app(config, _injected_services())
    flask_app.config["TESTING"] = True
    internal_canary = "/private/secret/chunk-and-prompt-CANARY"

    @flask_app.get("/expected")
    def expected_failure() -> object:
        try:
            raise RuntimeError(internal_canary)
        except RuntimeError as internal:
            raise PublicError(
                code="vector_store_unavailable",
                message="A base vetorial local está indisponível.",
                http_status=503,
            ) from internal

    @flask_app.get("/too-large")
    def too_large() -> object:
        raise RequestEntityTooLarge(internal_canary)

    @flask_app.get("/unexpected")
    def unexpected_failure() -> object:
        raise RuntimeError(internal_canary)

    try:
        client = flask_app.test_client()

        expected = client.get("/expected")
        assert expected.status_code == 503
        assert expected.get_json() == {
            "success": False,
            "code": "vector_store_unavailable",
            "message": "A base vetorial local está indisponível.",
        }
        assert internal_canary not in expected.get_data(as_text=True)

        too_large_response = client.get("/too-large")
        assert too_large_response.status_code == 413
        assert too_large_response.get_json() == {
            "success": False,
            "code": "upload_too_large",
            "message": (
                "O arquivo excede o limite de bytes compactados. Reduza o "
                "arquivo e tente novamente."
            ),
        }
        assert internal_canary not in too_large_response.get_data(as_text=True)

        with caplog.at_level(logging.ERROR, logger=flask_app.logger.name):
            unexpected = client.get("/unexpected")
        assert unexpected.status_code == 500
        assert unexpected.get_json() == {
            "success": False,
            "code": "internal_error",
            "message": (
                "Não foi possível concluir a operação devido a uma falha "
                "interna."
            ),
        }
        response_text = unexpected.get_data(as_text=True)
        assert internal_canary not in response_text
        assert "Traceback" not in response_text
        assert "Traceback" in caplog.text
        assert internal_canary in caplog.text
    finally:
        app_module.close_app(flask_app)


def test_main_uses_validated_server_settings_without_reloader_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    flask_app = app_module.create_app(config, _injected_services())
    runtime = flask_app.extensions["erp_ai_support"]
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(flask_app, "run", fake_run)
    monkeypatch.setattr(app_module, "create_app", lambda: flask_app)

    app_module.main()

    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 5_432,
            "debug": False,
            "use_reloader": False,
        }
    ]
    assert runtime.closed is True
