"""Flask composition root and HTTP error boundary for ERP AI Support.

Importing this module has no runtime side effects.  The concrete local services
are constructed only by :func:`create_app`, and the development server is
started only by :func:`main` under the ``python app.py`` entry point.
"""

from __future__ import annotations

import logging
import os
import shutil
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final
from uuid import uuid4

from flask import Flask, Request, Response, jsonify, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge

from config import load_config
from domain import AppConfig, PublicError
from ingest import IngestionService
from rag import (
    ChromaVectorStore,
    LocalEmbeddingService,
    ManifestStore,
    OllamaClient,
    RAGService,
    RetrievalService,
    StorageLocks,
)


_LOGGER = logging.getLogger(__name__)
_RUNTIME_EXTENSION_KEY: Final = "erp_ai_support"
_FINALIZER_EXTENSION_KEY: Final = "erp_ai_support.finalizer"
# Flask's request-wide limit is only an early defense for multipart framing.
# Task 8.3's bounded copy remains the authoritative uploaded-file byte limit.
_MULTIPART_DEFENSE_OVERHEAD_BYTES: Final = 64 * 1024

_UPLOAD_TOO_LARGE_MESSAGE: Final = (
    "O arquivo excede o limite de bytes compactados. Reduza o arquivo e "
    "tente novamente."
)
_INTERNAL_ERROR_MESSAGE: Final = (
    "Não foi possível concluir a operação devido a uma falha interna."
)

_ALLOWED_ZIP_MIMES: Final = frozenset(
    {"application/zip", "application/x-zip-compressed"}
)
_COPY_BLOCK_SIZE: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class Services:
    """The single application-owned instance of each concrete MVP service."""

    embedding_service: LocalEmbeddingService
    manifest_store: ManifestStore
    vector_store: ChromaVectorStore
    ingestion_service: IngestionService
    retrieval_service: RetrievalService
    ollama_client: OllamaClient
    rag_service: RAGService


class ApplicationRuntime:
    """Own the composed services and release process resources exactly once."""

    def __init__(
        self,
        config: AppConfig,
        services: Services,
        *,
        lifecycle_lock: object | None,
        owns_vector_store: bool,
    ) -> None:
        self.config = config
        self.services = services
        self._lifecycle_lock = lifecycle_lock
        self._owns_vector_store = owns_vector_store
        self._state_lock = Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether resource shutdown has already run."""

        with self._state_lock:
            return self._closed

    def close(self) -> None:
        """Close the embedded store, then always release the lifecycle lock.

        Shutdown is idempotent because it can be requested by ``main()``, an
        explicit test teardown, and the weak-reference finalizer.  Cleanup
        failures remain local log events and never become HTTP responses.
        """

        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        try:
            if self._owns_vector_store:
                close = getattr(self.services.vector_store, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        _LOGGER.exception(
                            "Falha local ao encerrar a base vetorial."
                        )
        finally:
            release = getattr(self._lifecycle_lock, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    _LOGGER.exception(
                        "Falha local ao liberar o lock de ciclo de vida."
                    )


def _release_partial_resources(
    vector_store: object | None,
    lifecycle_lock: object | None,
) -> None:
    """Best-effort cleanup for a composition failure before runtime publish."""

    try:
        close = getattr(vector_store, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                _LOGGER.exception(
                    "Falha local ao encerrar composição parcial da base vetorial."
                )
    finally:
        release = getattr(lifecycle_lock, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                _LOGGER.exception(
                    "Falha local ao liberar lock após composição parcial."
                )


def _compose_runtime(config: AppConfig) -> ApplicationRuntime:
    """Construct each concrete service once under the process lifecycle lock."""

    locks = StorageLocks(config.chroma_path)
    lifecycle_lock = locks.lifecycle_lock
    vector_store: ChromaVectorStore | None = None

    lifecycle_lock.acquire()
    try:
        manifest_store = ManifestStore(config, locks=locks)
        vector_store = ChromaVectorStore(
            config,
            manifest_store,
            recover_on_startup=False,
        )
        # Recovery is explicit and complete before any route is registered.
        # It is safe and idempotent when no incomplete journal exists.
        vector_store.recover_incomplete()

        embedding_service = LocalEmbeddingService(config)
        ingestion_service = IngestionService(
            config,
            embedding_service,
            manifest_store,
            vector_store,
        )
        retrieval_service = RetrievalService(
            config,
            embedding_service,
            vector_store,
        )
        ollama_client = OllamaClient(config)
        rag_service = RAGService(
            config,
            retrieval_service,
            ollama_client,
        )
        services = Services(
            embedding_service=embedding_service,
            manifest_store=manifest_store,
            vector_store=vector_store,
            ingestion_service=ingestion_service,
            retrieval_service=retrieval_service,
            ollama_client=ollama_client,
            rag_service=rag_service,
        )
        return ApplicationRuntime(
            config,
            services,
            lifecycle_lock=lifecycle_lock,
            owns_vector_store=True,
        )
    except BaseException:
        _release_partial_resources(vector_store, lifecycle_lock)
        raise


def error_response(error: PublicError) -> tuple[Response, int]:
    """Serialize only the allowlisted public error contract."""

    if not isinstance(error, PublicError):
        raise TypeError("error must be PublicError")
    response = jsonify(
        success=False,
        code=error.code,
        message=error.message,
    )
    return response, error.http_status


def _register_error_handlers(app: Flask) -> None:
    """Install the uniform JSON boundary for expected and unexpected errors."""

    @app.errorhandler(PublicError)
    def handle_public_error(error: PublicError) -> tuple[Response, int]:
        # Log only stable public metadata for expected failures.  Internal
        # exceptions chained to PublicError never enter the response body.
        app.logger.warning(
            "Falha pública tratada (code=%s, status=%s).",
            error.code,
            error.http_status,
        )
        return error_response(error)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(
        error: RequestEntityTooLarge,
    ) -> tuple[Response, int]:
        del error
        app.logger.warning("Requisição rejeitada pelo limite global do Flask.")
        return error_response(
            PublicError(
                code="upload_too_large",
                message=_UPLOAD_TOO_LARGE_MESSAGE,
                http_status=413,
            )
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception) -> tuple[Response, int]:
        # Explicit exception metadata keeps the traceback in the local server
        # log even if Flask invokes this handler outside the original ``except``
        # frame. Neither the exception nor its message is interpolated into HTTP.
        app.logger.error(
            "Falha interna não tratada durante uma requisição.",
            exc_info=(type(error), error, error.__traceback__),
        )
        return error_response(
            PublicError(
                code="internal_error",
                message=_INTERNAL_ERROR_MESSAGE,
                http_status=500,
            )
        )


def validate_upload_contract(incoming_request: Request) -> FileStorage:
    """Validate a multipart/form-data upload and return the FileStorage object.

    Checks:
    - Content-Type is multipart/form-data
    - Exactly one ``file`` field is present
    - The filename is non-empty and ends with ``.zip`` (case-insensitive)
    - The MIME type is ``application/zip`` or ``application/x-zip-compressed``
    """

    if not incoming_request.content_type or not incoming_request.content_type.startswith(
        "multipart/form-data"
    ):
        raise PublicError(
            code="unsupported_media_type",
            message=(
                "Envie o arquivo com Content-Type multipart/form-data."
            ),
            http_status=415,
        )

    all_files = incoming_request.files.getlist("file")
    if len(all_files) == 0:
        raise PublicError(
            code="file_missing",
            message="O campo file é obrigatório.",
            http_status=400,
        )
    if len(all_files) > 1:
        raise PublicError(
            code="file_ambiguous",
            message="Envie exatamente um arquivo por requisição.",
            http_status=400,
        )

    file_storage = all_files[0]
    filename = file_storage.filename or ""
    if not filename:
        raise PublicError(
            code="file_name_missing",
            message="O arquivo enviado não possui nome.",
            http_status=400,
        )

    if not filename.lower().endswith(".zip"):
        raise PublicError(
            code="unsupported_file_type",
            message="Somente arquivos .zip são aceitos.",
            http_status=415,
        )

    mime = (file_storage.mimetype or "").lower().strip()
    if mime not in _ALLOWED_ZIP_MIMES:
        raise PublicError(
            code="unsupported_media_type",
            message=(
                "O MIME type do arquivo deve ser application/zip ou "
                "application/x-zip-compressed."
            ),
            http_status=415,
        )

    return file_storage


def copy_upload_bounded(
    file_storage: FileStorage,
    dest_path: Path,
    max_bytes: int,
) -> None:
    """Copy a FileStorage stream to dest_path in 64 KB blocks.

    Raises PublicError ``upload_too_large`` (413) if the stream would exceed
    ``max_bytes`` without writing the offending byte to disk.
    """

    written = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)

    descriptor: int | None = None
    try:
        descriptor = os.open(dest_path, flags, 0o600)
        stream = file_storage.stream
        while True:
            block = stream.read(_COPY_BLOCK_SIZE)
            if not block:
                break
            if not isinstance(block, bytes):
                block = block.encode("utf-8") if isinstance(block, str) else bytes(block)
            if written + len(block) > max_bytes:
                raise PublicError(
                    code="upload_too_large",
                    message=_UPLOAD_TOO_LARGE_MESSAGE,
                    http_status=413,
                )
            view = memoryview(block)
            while view:
                amount = os.write(descriptor, view)
                if amount <= 0:
                    raise OSError("short write during upload copy")
                written += amount
                view = view[amount:]
        os.fchmod(descriptor, 0o600)
        os.close(descriptor)
        descriptor = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def validate_chat_request(
    incoming_request: Request,
    max_chars: int,
) -> str:
    """Validate one chat request in the normative decision order.

    The Unicode code-point limit is applied to the original string before
    ``str.strip()``.  Every rejection raises a public allowlisted error before
    any RAG dependency can be called.
    """

    if incoming_request.mimetype != "application/json":
        raise PublicError(
            code="unsupported_media_type",
            message="Envie a pergunta com Content-Type application/json.",
            http_status=400,
        )

    payload = incoming_request.get_json(silent=True)
    if type(payload) is not dict:
        raise PublicError(
            code="invalid_json",
            message="O corpo da requisição deve ser um objeto JSON válido.",
            http_status=400,
        )

    question = payload.get("question")
    if type(question) is not str:
        raise PublicError(
            code="question_missing",
            message="O campo question é obrigatório e deve ser uma string.",
            http_status=400,
        )

    if len(question) > max_chars:
        raise PublicError(
            code="question_too_large",
            message="A pergunta excede o limite configurado de caracteres.",
            http_status=413,
        )

    normalized_question = question.strip()
    if not normalized_question:
        raise PublicError(
            code="question_empty",
            message=(
                "Forneça pelo menos um caractere não branco antes de enviar "
                "a pergunta."
            ),
            http_status=400,
        )
    return normalized_question


def register_routes(app: Flask, services: Services) -> None:
    """Register the prescribed index and independent chat endpoints."""

    if not isinstance(app, Flask):
        raise TypeError("app must be a Flask application")
    if not isinstance(services, Services):
        raise TypeError("services must be Services")

    runtime = app.extensions.get(_RUNTIME_EXTENSION_KEY)
    if not isinstance(runtime, ApplicationRuntime) or runtime.services is not services:
        raise RuntimeError("application runtime was not initialized")
    max_question_chars = runtime.config.max_question_chars

    @app.get("/")
    def index() -> tuple[str, int]:
        return render_template("index.html"), 200

    @app.post("/upload")
    def upload() -> tuple[Response, int]:
        file_storage = validate_upload_contract(request)
        upload_id = uuid4()
        staging_dir = runtime.config.upload_folder / str(upload_id)
        staging_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        try:
            dest_path = staging_dir / "upload.zip"
            copy_upload_bounded(
                file_storage,
                dest_path,
                runtime.config.max_upload_bytes,
            )
            result = services.ingestion_service.ingest(dest_path, upload_id)
        finally:
            try:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
            except OSError:
                _LOGGER.exception(
                    "Falha ao remover diretório de staging do upload."
                )
        return (
            jsonify(
                {
                    "success": True,
                    "documents": result.documents,
                    "pages": result.pages,
                    "chunks": result.chunks,
                    "warnings": list(result.warnings),
                }
            ),
            200,
        )

    @app.post("/chat")
    def chat() -> tuple[Response, int]:
        question = validate_chat_request(request, max_question_chars)
        result = services.rag_service.answer(question)
        sources = [
            {"document": source.document, "page": source.page}
            for source in result.sources
        ]
        return jsonify({"answer": result.answer, "sources": sources}), 200


def create_app(
    config: AppConfig | None = None,
    services: Services | None = None,
) -> Flask:
    """Create one configured Flask application and its local service graph.

    Passing ``services`` is the test seam: the supplied complete graph is used
    as-is, without constructing duplicates or taking ownership of fake/external
    resources.  The normal production path owns its vector store and lifecycle
    lock, performs startup recovery, and releases both on shutdown or failure.
    """

    effective_config = load_config() if config is None else config
    if not isinstance(effective_config, AppConfig):
        raise TypeError("config must be AppConfig or None")
    if services is not None and not isinstance(services, Services):
        raise TypeError("services must be Services or None")

    runtime = (
        _compose_runtime(effective_config)
        if services is None
        else ApplicationRuntime(
            effective_config,
            services,
            lifecycle_lock=None,
            owns_vector_store=False,
        )
    )

    try:
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = (
            effective_config.max_upload_bytes
            + _MULTIPART_DEFENSE_OVERHEAD_BYTES
        )
        app.extensions[_RUNTIME_EXTENSION_KEY] = runtime
        _register_error_handlers(app)
        register_routes(app, runtime.services)

        # Flask teardown_appcontext runs after every request and therefore must
        # not release a process-lifetime lock.  This finalizer covers abandoned
        # app factories; ``main`` and tests can call ``close_app`` explicitly.
        app.extensions[_FINALIZER_EXTENSION_KEY] = weakref.finalize(
            app,
            runtime.close,
        )
        return app
    except BaseException:
        runtime.close()
        raise


def close_app(app: Flask) -> None:
    """Deterministically release resources owned by a created application."""

    if not isinstance(app, Flask):
        raise TypeError("app must be a Flask application")
    finalizer = app.extensions.get(_FINALIZER_EXTENSION_KEY)
    if callable(finalizer):
        finalizer()
        return
    runtime = app.extensions.get(_RUNTIME_EXTENSION_KEY)
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def main() -> None:
    """Run the single-process local Flask server from validated configuration."""

    try:
        app = create_app()
    except PublicError as error:
        # Startup failures are local console events.  Print only the public
        # code/message and exit before binding an HTTP socket.
        _LOGGER.error("%s: %s", error.code, error.message)
        raise SystemExit(1) from None

    runtime = app.extensions[_RUNTIME_EXTENSION_KEY]
    if not isinstance(runtime, ApplicationRuntime):
        close_app(app)
        raise RuntimeError("application runtime was not initialized")

    try:
        app.run(
            host=runtime.config.flask_host,
            port=runtime.config.flask_port,
            debug=runtime.config.flask_debug,
            # The debug reloader would start a second process and violate the
            # lifecycle-lock/single-process guarantee.
            use_reloader=False,
        )
    finally:
        close_app(app)


if __name__ == "__main__":
    main()


__all__ = [
    "ApplicationRuntime",
    "Services",
    "close_app",
    "copy_upload_bounded",
    "create_app",
    "error_response",
    "main",
    "register_routes",
    "validate_chat_request",
    "validate_upload_contract",
]
