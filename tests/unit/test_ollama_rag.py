"""Focused tests for the local Ollama client and RAG composition."""

from __future__ import annotations

import json
import socket
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator, Sequence

import pytest

from domain import AppConfig, PublicError, RetrievedChunk, Source
from rag import INSUFFICIENT_ANSWER, OllamaClient, RAGService


class _ServerState:
    def __init__(
        self,
        *,
        models: Sequence[str] = ("local-test-model",),
        tags_status: int = 200,
        generation_status: int = 200,
        generation_body: bytes | None = None,
        tags_location: str | None = None,
    ) -> None:
        self.models = tuple(models)
        self.tags_status = tags_status
        self.generation_status = generation_status
        self.generation_body = generation_body or json.dumps(
            {
                "response": "o procedimento salva o cadastro.",
                "done": True,
                "context": [99, 100],
            }
        ).encode("utf-8")
        self.tags_location = tags_location
        self.requests: list[tuple[str, str, object | None]] = []


def _handler_for(state: _ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            state.requests.append(("GET", self.path, None))
            if self.path != "/api/tags":
                self._send(404, b"{}")
                return
            body = json.dumps(
                {"models": [{"name": model, "model": model} for model in state.models]}
            ).encode("utf-8")
            headers = (
                {"Location": state.tags_location}
                if state.tags_location is not None
                else None
            )
            self._send(state.tags_status, body, headers)

        def do_POST(self) -> None:
            raw_length = self.headers.get("Content-Length", "0")
            body = self.rfile.read(int(raw_length))
            payload = json.loads(body.decode("utf-8"))
            state.requests.append(("POST", self.path, payload))
            self._send(state.generation_status, state.generation_body)

        def _send(
            self,
            status: int,
            body: bytes,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@contextmanager
def _local_server(state: _ServerState) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _config(root: Path, ollama_url: str) -> AppConfig:
    return AppConfig(
        ollama_url=ollama_url,
        ollama_model="local-test-model",
        chroma_path=root / "chroma",
        chroma_collection="test_collection",
        upload_folder=root / "uploads",
        embedding_model="local-test-embedding",
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
        ollama_timeout_seconds=2,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _chunk(
    chunk_id: str,
    text: str,
    document: str,
    page: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=f"doc-{chunk_id}",
        display_name=document,
        human_page=page,
        score=0.9,
    )


def test_ollama_client_uses_direct_stateless_requests_and_generation_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _ServerState()
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.10:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://203.0.113.10:9999")

    with _local_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        answer = client.generate("prompt independente", 123, 2)

    assert answer == "o procedimento salva o cadastro."
    assert [request[:2] for request in state.requests] == [
        ("GET", "/api/tags"),
        ("POST", "/api/generate"),
    ]
    payload = state.requests[1][2]
    assert payload == {
        "model": "local-test-model",
        "prompt": "prompt independente",
        "stream": False,
        "think": False,
        "options": {"num_predict": 123, "temperature": 0.1},
    }
    assert "context" not in payload


def test_ollama_client_maps_missing_model_before_generation(tmp_path: Path) -> None:
    state = _ServerState(models=("another-model",))

    with _local_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    assert captured.value.code == "ollama_model_missing"
    assert captured.value.http_status == 503
    assert "ollama pull local-test-model" in captured.value.message
    assert [request[0] for request in state.requests] == ["GET"]


@pytest.mark.parametrize(
    "generation_body",
    [
        b"{invalid-json",
        b'{"response":"   ","done":true}',
        b'{"response":"conteudo parcial","done":false}',
    ],
)
def test_ollama_client_discards_invalid_empty_or_incomplete_generation(
    tmp_path: Path,
    generation_body: bytes,
) -> None:
    state = _ServerState(generation_body=generation_body)

    with _local_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    assert captured.value.code == "generation_failed"
    assert captured.value.http_status == 503


def test_ollama_client_rejects_non_loopback_resolution_before_connecting(
    tmp_path: Path,
) -> None:
    connection_calls = 0

    def resolver(host: str, port: int, **kwargs: object) -> list[tuple[object, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("203.0.113.20", port),
            )
        ]

    def connection_factory(*args: object) -> object:
        nonlocal connection_calls
        connection_calls += 1
        raise AssertionError("an external address must never be connected")

    client = OllamaClient(
        _config(tmp_path, "http://localhost:11434"),
        resolver=resolver,
        connection_factory=connection_factory,  # type: ignore[arg-type]
    )

    with pytest.raises(PublicError) as captured:
        client.list_models(time.monotonic() + 1)

    assert captured.value.code == "ollama_unavailable"
    assert connection_calls == 0


def test_ollama_client_does_not_follow_redirects(tmp_path: Path) -> None:
    state = _ServerState(
        tags_status=302,
        tags_location="http://203.0.113.30/should-not-be-requested",
    )

    with _local_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    assert captured.value.code == "generation_failed"
    assert [request[0] for request in state.requests] == ["GET"]


class _RetrievalFake:
    def __init__(self, context: tuple[RetrievedChunk, ...]) -> None:
        self.context = context
        self.calls: list[str] = []

    def retrieve(self, question: str) -> tuple[RetrievedChunk, ...]:
        self.calls.append(question)
        return self.context


class _GeneratorFake:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, int, int]] = []

    def generate(self, prompt: str, max_tokens: int, timeout_seconds: int) -> str:
        self.calls.append((prompt, max_tokens, timeout_seconds))
        return self.answer


def test_rag_service_short_circuits_empty_context_without_generation(
    tmp_path: Path,
) -> None:
    retrieval = _RetrievalFake(())
    generator = _GeneratorFake("não deve ser usado")
    service = RAGService(
        _config(tmp_path, "http://localhost:11434"),
        retrieval,
        generator,
    )

    result = service.answer("pergunta sem cobertura")

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == ()
    assert retrieval.calls == ["pergunta sem cobertura"]
    assert generator.calls == []


def test_rag_service_generates_validates_and_derives_sources_without_history(
    tmp_path: Path,
) -> None:
    context = (
        _chunk(
            "chunk-1",
            "o procedimento salva o cadastro.",
            "manual.pdf",
            2,
        ),
        _chunk(
            "chunk-2",
            "o campo status fica ativo.",
            "guias/cadastro.pdf",
            4,
        ),
    )
    retrieval = _RetrievalFake(context)
    generator = _GeneratorFake("o procedimento salva o cadastro.")
    config = _config(tmp_path, "http://localhost:11434")
    service = RAGService(config, retrieval, generator)

    first = service.answer("primeira pergunta")
    second = service.answer("segunda pergunta")

    expected_sources = (
        Source(document="manual.pdf", page=2),
        Source(document="guias/cadastro.pdf", page=4),
    )
    assert first.answer == "o procedimento salva o cadastro."
    assert second.answer == first.answer
    assert first.sources == expected_sources
    assert second.sources == expected_sources
    assert retrieval.calls == ["primeira pergunta", "segunda pergunta"]
    assert len(generator.calls) == 2
    assert all(
        (max_tokens, timeout) == (
            config.max_answer_tokens,
            config.ollama_timeout_seconds,
        )
        for _, max_tokens, timeout in generator.calls
    )
    assert "primeira pergunta" in generator.calls[0][0]
    assert "primeira pergunta" not in generator.calls[1][0]
    assert "segunda pergunta" in generator.calls[1][0]


def test_rag_service_replaces_unsupported_output_and_removes_sources(
    tmp_path: Path,
) -> None:
    context = (
        _chunk("chunk-1", "o prazo é de 30 dias.", "manual.pdf", 1),
    )
    service = RAGService(
        _config(tmp_path, "http://localhost:11434"),
        _RetrievalFake(context),
        _GeneratorFake("o prazo é de 45 dias."),
    )

    result = service.answer("qual é o prazo?")

    assert result.answer == INSUFFICIENT_ANSWER
    assert result.sources == ()
