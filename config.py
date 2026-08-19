"""Centralized, validated configuration for ERP AI Support.

Only this module reads environment variables or ``.env``.  It composes all
known values, validates the complete configuration, safely prepares local data
directories, and returns one immutable :class:`domain.AppConfig` instance.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from dotenv import dotenv_values
from dotenv.parser import parse_stream

from domain import AppConfig, ChunkingProfile, PublicError, VectorSpace


_MEBIBYTE = Decimal(1_048_576)
_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+\Z")
_ALLOWED_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_CHUNK_SCHEMA_VERSION = "char-v1"

DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen3:8b",
        "CHROMA_PATH": "./data/chroma",
        "CHROMA_COLLECTION": "erp_ai_support",
        "UPLOAD_FOLDER": "./documents/uploads",
        "EMBEDDING_MODEL": (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        "TOP_K": "6",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "150",
        "RELEVANCE_THRESHOLD": "0.30",
        "MAX_UPLOAD_MB": "100",
        "MAX_ZIP_ENTRIES": "1000",
        "MAX_ZIP_ENTRY_MB": "100",
        "MAX_UNCOMPRESSED_MB": "500",
        "MAX_COMPRESSION_RATIO": "100",
        "MAX_QUESTION_CHARS": "2000",
        "OLLAMA_TIMEOUT_SECONDS": "120",
        "MAX_ANSWER_TOKENS": "500",
        "FLASK_HOST": "127.0.0.1",
        "FLASK_PORT": "5000",
        "FLASK_DEBUG": "false",
    }
)

CONFIG_VARIABLES = tuple(DEFAULTS)


def _invalid_config(variable: str) -> PublicError:
    return PublicError(
        code="invalid_config",
        message=(
            f"A configuração {variable} é inválida. "
            "Corrija o valor e inicie a aplicação novamente."
        ),
        http_status=500,
    )


def _dotenv_load_failed() -> PublicError:
    return PublicError(
        code="dotenv_load_failed",
        message=(
            "Não foi possível carregar o arquivo .env. "
            "Verifique o arquivo e tente novamente."
        ),
        http_status=500,
    )


def _data_path_invalid(variable: str) -> PublicError:
    return PublicError(
        code="data_path_invalid",
        message=(
            f"O diretório configurado em {variable} não pôde ser preparado "
            "para leitura e escrita."
        ),
        http_status=500,
    )


def _vector_space_mismatch() -> PublicError:
    return PublicError(
        code="vector_space_mismatch",
        message=(
            "A coleção usa um espaço vetorial incompatível. "
            "Recrie o índice com a configuração atual."
        ),
        http_status=503,
    )


def _chunk_profile_mismatch() -> PublicError:
    return PublicError(
        code="chunk_profile_mismatch",
        message=(
            "A coleção usa um perfil de chunking incompatível. "
            "Recrie o índice com a configuração atual."
        ),
        http_status=503,
    )


def _trim_text(raw: object, variable: str) -> str:
    if not isinstance(raw, str):
        raise _invalid_config(variable)
    return raw.strip()


def _parse_non_empty(raw: object, variable: str) -> str:
    value = _trim_text(raw, variable)
    if not value:
        raise _invalid_config(variable)
    return value


def parse_bool(raw: str, variable: str) -> bool:
    """Parse only the case-insensitive literals ``true`` and ``false``."""

    value = _trim_text(raw, variable).casefold()
    if value == "true":
        return True
    if value == "false":
        return False
    raise _invalid_config(variable)


def parse_int(
    raw: str,
    variable: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Parse a base-10 integer and enforce inclusive bounds."""

    value = _trim_text(raw, variable)
    if _INTEGER_PATTERN.fullmatch(value) is None:
        raise _invalid_config(variable)
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_config(variable) from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise _invalid_config(variable)
    return parsed


def parse_float(
    raw: str,
    variable: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Parse a finite floating-point number and enforce inclusive bounds."""

    value = _trim_text(raw, variable)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_config(variable) from exc
    if (
        not math.isfinite(parsed)
        or parsed < minimum
        or (maximum is not None and parsed > maximum)
    ):
        raise _invalid_config(variable)
    return parsed


def parse_megabytes(raw: str, variable: str) -> tuple[Decimal, int]:
    """Parse positive MiB exactly and return its decimal value and byte floor."""

    value = _trim_text(raw, variable)
    try:
        megabytes = Decimal(value)
        if not megabytes.is_finite() or megabytes <= 0:
            raise InvalidOperation
        byte_limit = int(
            (megabytes * _MEBIBYTE).to_integral_value(rounding=ROUND_FLOOR)
        )
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise _invalid_config(variable) from exc
    return megabytes, byte_limit


def validate_local_ollama_url(raw: str) -> str:
    """Validate and normalize an HTTP(S) Ollama URL on an explicit loopback host."""

    variable = "OLLAMA_URL"
    value = _parse_non_empty(raw, variable)

    # ``urlsplit`` removes a few ASCII controls on some Python versions.  Reject
    # them before parsing so validation never silently changes the authority.
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value):
        raise _invalid_config(variable)

    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise _invalid_config(variable) from exc

    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold() if parsed.hostname is not None else None
    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or hostname not in _ALLOWED_LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise _invalid_config(variable)

    # Reject an explicitly empty port, which ``SplitResult.port`` treats as
    # absent.  Bracketed IPv6 needs a separate authority-tail check.
    authority = parsed.netloc
    if hostname == "::1":
        closing_bracket = authority.find("]")
        if closing_bracket < 0 or authority[closing_bracket + 1 :] == ":":
            raise _invalid_config(variable)
    elif authority.endswith(":"):
        raise _invalid_config(variable)

    normalized_authority = f"[{hostname}]" if hostname == "::1" else hostname
    if port is not None:
        normalized_authority = f"{normalized_authority}:{port}"

    path = parsed.path
    if path == "/":
        path = ""
    elif path:
        path = path.rstrip("/")

    return urlunsplit((scheme, normalized_authority, path, parsed.query, ""))


def _validate_local_host(raw: object) -> str:
    variable = "FLASK_HOST"
    host = _parse_non_empty(raw, variable).casefold()
    if host not in _ALLOWED_LOOPBACK_HOSTS:
        raise _invalid_config(variable)
    return host


def _mode_has_any(mode: int, bits: int) -> bool:
    return bool(stat.S_IMODE(mode) & bits)


def _nearest_existing_parent(path: Path) -> Path:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent


def ensure_data_directory(raw: str, variable: str, cwd: Path) -> Path:
    """Resolve, create, and probe one configured local data directory.

    Internal filesystem exceptions are deliberately replaced with a public
    message that names only the affected variable, never the resolved path.
    """

    value = _parse_non_empty(raw, variable)
    try:
        base = Path(cwd).resolve(strict=False)
        configured = Path(value)
        candidate = (
            configured if configured.is_absolute() else base / configured
        ).resolve(strict=False)

        if candidate.exists():
            if not candidate.is_dir():
                raise OSError("configured data path is not a directory")
        else:
            parent = _nearest_existing_parent(candidate)
            if not parent.is_dir():
                raise OSError("data directory parent is not a directory")
            parent_mode = parent.stat().st_mode
            if (
                not _mode_has_any(parent_mode, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                or not _mode_has_any(parent_mode, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                or not os.access(parent, os.W_OK | os.X_OK)
            ):
                raise OSError("data directory parent is not writable")
            candidate.mkdir(mode=0o700, parents=True, exist_ok=False)
            candidate.chmod(0o700)

        mode = candidate.stat().st_mode
        if (
            not _mode_has_any(mode, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            or not _mode_has_any(mode, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            or not _mode_has_any(mode, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            or not os.access(candidate, os.R_OK | os.W_OK | os.X_OK)
        ):
            raise OSError("configured data directory is not accessible")

        # Force an actual directory read rather than relying only on access(2).
        with os.scandir(candidate) as entries:
            next(entries, None)

        probe = candidate / f".erp-ai-support-probe-{uuid4().hex}"
        try:
            descriptor = os.open(
                probe,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            probe.unlink()
        except (OSError, ValueError):
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise _data_path_invalid(variable) from exc

    return candidate


def _resolve_dotenv_path(cwd: Path, dotenv_path: Path | None) -> Path:
    path = Path(".env") if dotenv_path is None else Path(dotenv_path)
    if not path.is_absolute():
        path = cwd / path
    return path


def _load_dotenv_file(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}

    try:
        if not path.is_file():
            raise OSError(".env is not a regular file")
        mode = path.stat().st_mode
        if (
            not _mode_has_any(mode, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            or not os.access(path, os.R_OK)
        ):
            raise OSError(".env is not readable")

        # python-dotenv exposes parser errors on each binding; ``dotenv_values``
        # otherwise logs and skips malformed lines, which would hide an invalid
        # file and incorrectly fall back to defaults.
        with path.open("r", encoding="utf-8") as stream:
            if any(binding.error for binding in parse_stream(stream)):
                raise ValueError("python-dotenv reported a malformed binding")

        values = dotenv_values(
            dotenv_path=path,
            encoding="utf-8",
            interpolate=False,
        )
        if values is None:
            raise ValueError("python-dotenv returned no mapping")
        return values
    except Exception as exc:
        # Do not put ``path`` or the parser's internal message in PublicError.
        raise _dotenv_load_failed() from exc


def _compose_raw_values(
    environment: Mapping[str, str], dotenv_data: Mapping[str, object]
) -> dict[str, object]:
    values: dict[str, object] = dict(DEFAULTS)
    for variable in CONFIG_VARIABLES:
        if variable in dotenv_data:
            values[variable] = dotenv_data[variable]
        if variable in environment:
            values[variable] = environment[variable]
    return values


def load_config(
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    dotenv_path: Path | None = None,
) -> AppConfig:
    """Load and validate the complete MVP configuration.

    Precedence is ``defaults < .env < environ``.  No ``AppConfig`` is created
    until every scalar value, cross-field relation, URL, host, and data path has
    passed validation.
    """

    working_directory = Path.cwd() if cwd is None else Path(cwd)
    try:
        working_directory = working_directory.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _data_path_invalid("CHROMA_PATH") from exc

    environment = os.environ if environ is None else environ
    if not isinstance(environment, Mapping):
        raise _invalid_config("ambiente")

    env_file = _resolve_dotenv_path(working_directory, dotenv_path)
    dotenv_data = _load_dotenv_file(env_file)
    raw = _compose_raw_values(environment, dotenv_data)

    # Parse all non-path values first so malformed scalar configuration cannot
    # create directories as a side effect.
    ollama_url = validate_local_ollama_url(raw["OLLAMA_URL"])
    ollama_model = _parse_non_empty(raw["OLLAMA_MODEL"], "OLLAMA_MODEL")
    chroma_collection = _parse_non_empty(
        raw["CHROMA_COLLECTION"], "CHROMA_COLLECTION"
    )
    embedding_model = _parse_non_empty(raw["EMBEDDING_MODEL"], "EMBEDDING_MODEL")
    top_k = parse_int(raw["TOP_K"], "TOP_K", 5, 8)
    chunk_size = parse_int(raw["CHUNK_SIZE"], "CHUNK_SIZE", 500, 1_000)
    chunk_overlap = parse_int(raw["CHUNK_OVERLAP"], "CHUNK_OVERLAP", 0)
    relevance_threshold = parse_float(
        raw["RELEVANCE_THRESHOLD"], "RELEVANCE_THRESHOLD", -1.0, 1.0
    )
    _, max_upload_bytes = parse_megabytes(raw["MAX_UPLOAD_MB"], "MAX_UPLOAD_MB")
    max_zip_entries = parse_int(raw["MAX_ZIP_ENTRIES"], "MAX_ZIP_ENTRIES", 1)
    max_zip_entry_mb, max_zip_entry_bytes = parse_megabytes(
        raw["MAX_ZIP_ENTRY_MB"], "MAX_ZIP_ENTRY_MB"
    )
    max_uncompressed_mb, max_uncompressed_bytes = parse_megabytes(
        raw["MAX_UNCOMPRESSED_MB"], "MAX_UNCOMPRESSED_MB"
    )
    max_compression_ratio = parse_float(
        raw["MAX_COMPRESSION_RATIO"], "MAX_COMPRESSION_RATIO", 1.0
    )
    max_question_chars = parse_int(
        raw["MAX_QUESTION_CHARS"], "MAX_QUESTION_CHARS", 1
    )
    ollama_timeout_seconds = parse_int(
        raw["OLLAMA_TIMEOUT_SECONDS"], "OLLAMA_TIMEOUT_SECONDS", 1, 600
    )
    max_answer_tokens = parse_int(
        raw["MAX_ANSWER_TOKENS"], "MAX_ANSWER_TOKENS", 64, 2_048
    )
    flask_host = _validate_local_host(raw["FLASK_HOST"])
    flask_port = parse_int(raw["FLASK_PORT"], "FLASK_PORT", 1, 65_535)
    flask_debug = parse_bool(raw["FLASK_DEBUG"], "FLASK_DEBUG")

    if chunk_overlap >= chunk_size:
        raise _invalid_config("CHUNK_OVERLAP")
    if max_uncompressed_mb < max_zip_entry_mb:
        raise _invalid_config("MAX_UNCOMPRESSED_MB")

    chroma_path = ensure_data_directory(
        raw["CHROMA_PATH"], "CHROMA_PATH", working_directory
    )
    upload_folder = ensure_data_directory(
        raw["UPLOAD_FOLDER"], "UPLOAD_FOLDER", working_directory
    )

    return AppConfig(
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        chroma_path=chroma_path,
        chroma_collection=chroma_collection,
        upload_folder=upload_folder,
        embedding_model=embedding_model,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        relevance_threshold=relevance_threshold,
        max_upload_bytes=max_upload_bytes,
        max_zip_entries=max_zip_entries,
        max_zip_entry_bytes=max_zip_entry_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_compression_ratio=max_compression_ratio,
        max_question_chars=max_question_chars,
        ollama_timeout_seconds=ollama_timeout_seconds,
        max_answer_tokens=max_answer_tokens,
        flask_host=flask_host,
        flask_port=flask_port,
        flask_debug=flask_debug,
    )


def _metadata_matches(
    metadata: Mapping[str, object], key: str, expected: object
) -> bool:
    if key not in metadata:
        return False
    actual = metadata[key]
    # Prevent Python's ``True == 1`` coercion from accepting the wrong persisted
    # scalar type.
    return type(actual) is type(expected) and actual == expected


def validate_vector_space_compatibility(
    expected_space: VectorSpace,
    recorded_metadata: Mapping[str, object],
) -> None:
    """Reject persisted metadata outside the expected embedding space."""

    valid_space = (
        type(expected_space.model) is str
        and bool(expected_space.model)
        and type(expected_space.dimension) is int
        and expected_space.dimension > 0
        and expected_space.normalized is True
        and expected_space.metric == "cosine"
    )
    expected_metadata: Mapping[str, object] = {
        "embedding_model": expected_space.model,
        "embedding_dimension": expected_space.dimension,
        "embedding_normalized": expected_space.normalized,
        "distance_metric": expected_space.metric,
    }
    if not valid_space or not isinstance(recorded_metadata, Mapping):
        raise _vector_space_mismatch()
    if any(
        not _metadata_matches(recorded_metadata, key, expected)
        for key, expected in expected_metadata.items()
    ):
        raise _vector_space_mismatch()


def validate_chunking_profile_compatibility(
    expected_profile: ChunkingProfile,
    recorded_metadata: Mapping[str, object],
) -> None:
    """Reject persisted metadata outside the expected character profile."""

    valid_profile = (
        type(expected_profile.size) is int
        and expected_profile.size > 0
        and type(expected_profile.overlap) is int
        and 0 <= expected_profile.overlap < expected_profile.size
        and expected_profile.schema_version == _CHUNK_SCHEMA_VERSION
    )
    expected_metadata: Mapping[str, object] = {
        "chunk_size": expected_profile.size,
        "chunk_overlap": expected_profile.overlap,
        "chunk_schema_version": expected_profile.schema_version,
    }
    if not valid_profile or not isinstance(recorded_metadata, Mapping):
        raise _chunk_profile_mismatch()
    if any(
        not _metadata_matches(recorded_metadata, key, expected)
        for key, expected in expected_metadata.items()
    ):
        raise _chunk_profile_mismatch()


def validate_collection_compatibility(
    config: AppConfig,
    actual_space: VectorSpace,
    recorded_metadata: Mapping[str, object],
) -> None:
    """Pure guard for an existing collection's vector and chunk contracts."""

    if actual_space.model != config.embedding_model:
        raise _vector_space_mismatch()
    validate_vector_space_compatibility(actual_space, recorded_metadata)
    validate_chunking_profile_compatibility(
        ChunkingProfile(
            size=config.chunk_size,
            overlap=config.chunk_overlap,
            schema_version=_CHUNK_SCHEMA_VERSION,
        ),
        recorded_metadata,
    )


__all__ = [
    "CONFIG_VARIABLES",
    "DEFAULTS",
    "ensure_data_directory",
    "load_config",
    "parse_bool",
    "parse_float",
    "parse_int",
    "parse_megabytes",
    "validate_chunking_profile_compatibility",
    "validate_collection_compatibility",
    "validate_local_ollama_url",
    "validate_vector_space_compatibility",
]
