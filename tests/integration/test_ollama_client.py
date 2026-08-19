"""Integration coverage for the strictly local, stateless Ollama client.

The fake servers in this module bind exclusively to ``127.0.0.1`` and expose
only the two Ollama endpoints used by production.  No test requires Ollama, a
model, internet access, or a remote fallback.

Validates Requirements 1.6, 13.1-13.3, 13.13, 13.15, 13.16, 15.13, 15.14,
18.4, 19.1, 19.2, and 19.10.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Iterator, Sequence

import pytest

from domain import AppConfig, PublicError
from rag import OllamaClient


_MODEL = "local-test-model"
_UNAVAILABLE_MESSAGE = (
    "O Ollama local está indisponível. Inicie o serviço e tente novamente."
)
_GENERATION_FAILED_MESSAGE = (
    "A resposta não pôde ser gerada. Verifique o Ollama local e o modelo "
    "configurado e tente novamente."
)


@dataclass(frozen=True, slots=True)
class _Reply:
    status: int = 200
    body: bytes = b"{}"
    location: str | None = None
    advertised_length: int | None = None
    wait_for_release: bool = False


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    method: str
    path: str
    host: str
    body: bytes


class _FakeOllamaState:
    def __init__(
        self,
        *,
        models: Sequence[str] = (_MODEL,),
        tags_reply: _Reply | None = None,
        generation_reply: _Reply | None = None,
    ) -> None:
        models_body = json.dumps(
            {"models": [{"name": model, "model": model} for model in models]},
            separators=(",", ":"),
        ).encode("utf-8")
        generation_body = json.dumps(
            {
                "response": "Resposta produzida somente pelo servidor local.",
                "done": True,
                "context": [91, 92, 93],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.tags_reply = tags_reply or _Reply(body=models_body)
        self.generation_reply = generation_reply or _Reply(body=generation_body)
        self.release = Event()
        self.blocked = Event()
        self._requests: list[_RecordedRequest] = []
        self._request_lock = Lock()

    @property
    def requests(self) -> tuple[_RecordedRequest, ...]:
        with self._request_lock:
            return tuple(self._requests)

    def record(self, request: _RecordedRequest) -> None:
        with self._request_lock:
            self._requests.append(request)


class _LoopbackThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False


def _handler_for(state: _FakeOllamaState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            state.record(
                _RecordedRequest(
                    method="GET",
                    path=self.path,
                    host=self.headers.get("Host", ""),
                    body=b"",
                )
            )
            if self.path != "/api/tags":
                self._send(_Reply(status=404, body=b"{}"))
                return
            self._send(state.tags_reply)

        def do_POST(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"), 10)
            except ValueError:
                content_length = 0
            body = self.rfile.read(max(0, content_length))
            state.record(
                _RecordedRequest(
                    method="POST",
                    path=self.path,
                    host=self.headers.get("Host", ""),
                    body=body,
                )
            )
            if self.path != "/api/generate":
                self._send(_Reply(status=404, body=b"{}"))
                return
            self._send(state.generation_reply)

        def _send(self, reply: _Reply) -> None:
            if reply.wait_for_release:
                state.blocked.set()
                state.release.wait(timeout=5.0)

            advertised_length = (
                len(reply.body)
                if reply.advertised_length is None
                else reply.advertised_length
            )
            try:
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(advertised_length))
                self.send_header("Connection", "close")
                if reply.location is not None:
                    self.send_header("Location", reply.location)
                self.end_headers()
                if reply.body:
                    self.wfile.write(reply.body)
                    self.wfile.flush()
            except OSError:
                # Expected when a deadline closes the client socket while this
                # fake endpoint is intentionally blocked.
                return
            finally:
                self.close_connection = True

            if advertised_length > len(reply.body):
                # Produce EOF before the declared body is complete.
                try:
                    self.connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


@contextmanager
def _fake_ollama_server(state: _FakeOllamaState) -> Iterator[str]:
    server = _LoopbackThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler_for(state),
    )
    host, port = server.server_address[:2]
    assert ipaddress.ip_address(host).is_loopback

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _config(
    root: Path,
    ollama_url: str,
    *,
    timeout_seconds: int = 2,
) -> AppConfig:
    return AppConfig(
        ollama_url=ollama_url,
        ollama_model=_MODEL,
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
        ollama_timeout_seconds=timeout_seconds,
        max_answer_tokens=500,
        flask_host="127.0.0.1",
        flask_port=5_000,
        flask_debug=False,
    )


def _generation_body(response: str, *, done: bool = True) -> bytes:
    return json.dumps(
        {"response": response, "done": done},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _guarded_loopback_resolver(
    calls: list[tuple[str, int]],
):
    def resolver(
        host: str,
        port: int,
        **kwargs: object,
    ) -> list[tuple[object, ...]]:
        calls.append((host, port))
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise AssertionError("an external hostname must never be resolved")
        results = socket.getaddrinfo(host, port, **kwargs)
        assert results
        for result in results:
            raw_address = result[4]
            assert isinstance(raw_address, tuple)
            address = str(raw_address[0]).split("%", 1)[0]
            assert ipaddress.ip_address(address).is_loopback
        return results

    return resolver


def _assert_generation_failed(error: PublicError) -> None:
    assert error.code == "generation_failed"
    assert error.http_status == 503
    assert error.message == _GENERATION_FAILED_MESSAGE


def test_generate_is_local_direct_limited_and_stateless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _FakeOllamaState()
    resolver_calls: list[tuple[str, int]] = []
    proxy = "http://198.51.100.40:65535"
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(variable, proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")

    with _fake_ollama_server(state) as url:
        client = OllamaClient(
            _config(tmp_path, url),
            resolver=_guarded_loopback_resolver(resolver_calls),
        )
        first = client.generate("primeiro prompt independente", 123, 2)
        second = client.generate("segundo prompt independente", 321, 2)

    assert first == "Resposta produzida somente pelo servidor local."
    assert second == first
    requests = state.requests
    assert [(request.method, request.path) for request in requests] == [
        ("GET", "/api/tags"),
        ("POST", "/api/generate"),
        ("GET", "/api/tags"),
        ("POST", "/api/generate"),
    ]
    assert all(request.host.startswith("127.0.0.1:") for request in requests)
    assert len(resolver_calls) == 4
    assert all(host == "127.0.0.1" for host, _ in resolver_calls)

    payloads = [
        json.loads(request.body.decode("utf-8"))
        for request in requests
        if request.method == "POST"
    ]
    assert payloads == [
        {
            "model": _MODEL,
            "prompt": "primeiro prompt independente",
            "stream": False,
            "think": False,
            "options": {"num_predict": 123, "temperature": 0.1},
        },
        {
            "model": _MODEL,
            "prompt": "segundo prompt independente",
            "stream": False,
            "think": False,
            "options": {"num_predict": 321, "temperature": 0.1},
        },
    ]
    assert all(
        forbidden not in payload
        for payload in payloads
        for forbidden in ("context", "history", "messages")
    )


def test_missing_configured_model_stops_before_generation(tmp_path: Path) -> None:
    state = _FakeOllamaState(models=("another-local-model",))

    with _fake_ollama_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    error = captured.value
    assert error.code == "ollama_model_missing"
    assert error.http_status == 503
    assert error.message == (
        f"O modelo {_MODEL} não está instalado no Ollama. "
        f"Execute ollama pull {_MODEL} e tente novamente."
    )
    assert [(request.method, request.path) for request in state.requests] == [
        ("GET", "/api/tags")
    ]


def test_connection_refused_maps_to_unavailable_without_remote_fallback(
    tmp_path: Path,
) -> None:
    resolver_calls: list[tuple[str, int]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as unserved_socket:
        unserved_socket.bind(("127.0.0.1", 0))
        host, port = unserved_socket.getsockname()[:2]
        client = OllamaClient(
            _config(
                tmp_path,
                f"http://{host}:{port}",
                timeout_seconds=1,
            ),
            resolver=_guarded_loopback_resolver(resolver_calls),
        )
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 1)

    error = captured.value
    assert error.code == "ollama_unavailable"
    assert error.http_status == 503
    assert error.message == _UNAVAILABLE_MESSAGE
    assert resolver_calls
    assert all(host == "127.0.0.1" for host, _ in resolver_calls)


@pytest.mark.parametrize(
    ("phase", "expected_code", "expected_requests"),
    [
        pytest.param(
            "tags",
            "ollama_unavailable",
            [("GET", "/api/tags")],
            id="connection-preflight",
        ),
        pytest.param(
            "generation",
            "generation_failed",
            [("GET", "/api/tags"), ("POST", "/api/generate")],
            id="generation",
        ),
    ],
)
def test_timeout_is_mapped_by_phase_and_returns_no_partial_answer(
    tmp_path: Path,
    phase: str,
    expected_code: str,
    expected_requests: list[tuple[str, str]],
) -> None:
    blocked_reply = _Reply(
        body=_generation_body("RESPOSTA_PARCIAL_NAO_RETORNAR"),
        wait_for_release=True,
    )
    state = _FakeOllamaState(
        tags_reply=blocked_reply if phase == "tags" else None,
        generation_reply=blocked_reply if phase == "generation" else None,
    )

    with _fake_ollama_server(state) as url:
        client = OllamaClient(_config(tmp_path, url, timeout_seconds=1))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 1)

    error = captured.value
    assert state.blocked.is_set()
    assert error.code == expected_code
    assert error.http_status == 503
    expected_message = (
        _UNAVAILABLE_MESSAGE
        if expected_code == "ollama_unavailable"
        else _GENERATION_FAILED_MESSAGE
    )
    assert error.message == expected_message
    assert "RESPOSTA_PARCIAL_NAO_RETORNAR" not in error.message
    assert [
        (request.method, request.path) for request in state.requests
    ] == expected_requests


@pytest.mark.parametrize(
    "generation_body",
    [
        pytest.param(b"{invalid-json", id="invalid-json"),
        pytest.param(b"", id="empty-http-body"),
        pytest.param(_generation_body(""), id="empty-response"),
        pytest.param(_generation_body(" \n\t "), id="whitespace-response"),
    ],
)
def test_invalid_json_and_empty_generation_are_discarded(
    tmp_path: Path,
    generation_body: bytes,
) -> None:
    state = _FakeOllamaState(
        generation_reply=_Reply(body=generation_body),
    )

    with _fake_ollama_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    _assert_generation_failed(captured.value)
    assert [(request.method, request.path) for request in state.requests] == [
        ("GET", "/api/tags"),
        ("POST", "/api/generate"),
    ]


def test_invalid_tags_json_stops_before_generation(tmp_path: Path) -> None:
    state = _FakeOllamaState(tags_reply=_Reply(body=b"{invalid-json"))

    with _fake_ollama_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    _assert_generation_failed(captured.value)
    assert [(request.method, request.path) for request in state.requests] == [
        ("GET", "/api/tags")
    ]


@pytest.mark.parametrize("failure_mode", ["done-false", "truncated-http"])
def test_interrupted_or_incomplete_generation_discards_partial_content(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    sentinel = "RESPOSTA_PARCIAL_NAO_RETORNAR"
    body = _generation_body(sentinel, done=failure_mode != "done-false")
    reply = _Reply(
        body=body,
        advertised_length=(
            len(body) + 32 if failure_mode == "truncated-http" else None
        ),
    )
    state = _FakeOllamaState(generation_reply=reply)

    with _fake_ollama_server(state) as url:
        client = OllamaClient(_config(tmp_path, url))
        with pytest.raises(PublicError) as captured:
            client.generate("prompt", 100, 2)

    _assert_generation_failed(captured.value)
    assert sentinel not in captured.value.message
    assert [(request.method, request.path) for request in state.requests] == [
        ("GET", "/api/tags"),
        ("POST", "/api/generate"),
    ]


@pytest.mark.parametrize("redirect_phase", ["tags", "generation"])
def test_redirect_is_refused_and_target_is_never_contacted(
    tmp_path: Path,
    redirect_phase: str,
) -> None:
    trap_state = _FakeOllamaState()
    with _fake_ollama_server(trap_state) as trap_url:
        location = f"{trap_url}/redirect-target"
        redirect = _Reply(status=307, body=b"{}", location=location)
        state = _FakeOllamaState(
            tags_reply=redirect if redirect_phase == "tags" else None,
            generation_reply=(
                redirect if redirect_phase == "generation" else None
            ),
        )
        with _fake_ollama_server(state) as url:
            client = OllamaClient(_config(tmp_path, url))
            with pytest.raises(PublicError) as captured:
                client.generate("prompt", 100, 2)

    _assert_generation_failed(captured.value)
    assert trap_state.requests == ()
    expected = [("GET", "/api/tags")]
    if redirect_phase == "generation":
        expected.append(("POST", "/api/generate"))
    assert [(request.method, request.path) for request in state.requests] == expected


def test_external_configured_host_is_rejected_before_resolution(
    tmp_path: Path,
) -> None:
    resolver_called = False

    def forbidden_resolver(*args: object, **kwargs: object) -> list[object]:
        nonlocal resolver_called
        resolver_called = True
        raise AssertionError("an external hostname must not be resolved")

    with pytest.raises(ValueError, match="loopback"):
        OllamaClient(
            _config(tmp_path, "http://example.invalid:11434"),
            resolver=forbidden_resolver,  # type: ignore[arg-type]
        )

    assert resolver_called is False


def test_non_loopback_dns_result_is_rejected_before_any_connection(
    tmp_path: Path,
) -> None:
    connection_calls = 0

    def mixed_resolver(
        host: str,
        port: int,
        **kwargs: object,
    ) -> list[tuple[object, ...]]:
        assert host == "localhost"
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.1", port),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("203.0.113.80", port),
            ),
        ]

    def forbidden_connection(*args: object) -> object:
        nonlocal connection_calls
        connection_calls += 1
        raise AssertionError("no address may be contacted after unsafe resolution")

    client = OllamaClient(
        _config(tmp_path, "http://localhost:11434"),
        resolver=mixed_resolver,
        connection_factory=forbidden_connection,  # type: ignore[arg-type]
    )
    with pytest.raises(PublicError) as captured:
        client.generate("prompt", 100, 2)

    error = captured.value
    assert error.code == "ollama_unavailable"
    assert error.http_status == 503
    assert error.message == _UNAVAILABLE_MESSAGE
    assert connection_calls == 0
