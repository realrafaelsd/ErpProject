"""Secure ZIP handling and ordered PDF discovery for ERP AI Support.

Archive members are fully inspected before extraction, extraction uses only
bounded application-controlled streaming writes, and the published regular
files are discovered without scanning or exposing the staging filesystem.
"""

from __future__ import annotations

import math
import ntpath
import os
import shutil
import stat
import zipfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Final, Iterable, Mapping, Protocol, Sequence
from uuid import UUID

from domain import (
    AppConfig,
    ArchiveEntryPlan,
    ArchivePlan,
    Chunk,
    ChunkingProfile,
    CommitPlan,
    DocumentManifest,
    DocumentManifestKey,
    EmbeddingProvider,
    ExtractedDocument,
    ExtractedEntry,
    PdfPage,
    PublicError,
    StoredChunk,
    UploadResult,
    VectorSpace,
    VectorStore,
)


_COPY_BUFFER_SIZE: Final = 64 * 1024

_ERROR_SPECS: Final[dict[str, tuple[str, int]]] = {
    "invalid_zip": (
        "O arquivo não é um ZIP válido. Envie um arquivo ZIP válido.",
        422,
    ),
    "unsafe_zip_entry": (
        "O arquivo ZIP contém uma entrada insegura e foi rejeitado.",
        422,
    ),
    "zip_entry_count_exceeded": (
        "O ZIP excede o limite de quantidade de entradas. "
        "Reduza essa quantidade e tente novamente.",
        422,
    ),
    "zip_entry_size_exceeded": (
        "O ZIP excede o limite de bytes por entrada. "
        "Reduza o tamanho das entradas e tente novamente.",
        422,
    ),
    "zip_total_size_exceeded": (
        "O ZIP excede o limite de bytes descompactados totais. "
        "Reduza o conteúdo e tente novamente.",
        422,
    ),
    "zip_compression_ratio_exceeded": (
        "O ZIP excede o limite de razão de compressão. "
        "Recompacte ou reduza o conteúdo e tente novamente.",
        422,
    ),
    "zip_extraction_failed": (
        "Não foi possível extrair o arquivo ZIP com segurança.",
        422,
    ),
}


def _public_error(code: str) -> PublicError:
    message, status = _ERROR_SPECS[code]
    return PublicError(code=code, message=message, http_status=status)


def _contains_control(value: str) -> bool:
    """Return whether *value* contains a C0/C1 or DEL control character."""

    return any(ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value)


def _raw_member_name(info: zipfile.ZipInfo) -> str:
    """Return the member name before ``zipfile``'s NUL truncation.

    ``ZipInfo.filename`` is truncated at the first NUL by the standard library,
    while ``orig_filename`` retains the decoded original name. Inspecting the
    latter prevents a malicious suffix from disappearing before validation.
    """

    name = getattr(info, "orig_filename", info.filename)
    if not isinstance(name, str) or not name:
        raise _public_error("unsafe_zip_entry")
    return name


def _normalize_member_path(name: str) -> tuple[tuple[str, ...], bool]:
    """Validate a ZIP member name and return safe relative path components.

    Both slash styles are interpreted as separators for security decisions.
    Empty and ``.`` components are normalized away so aliases are caught by the
    duplicate-destination check; ``..`` is always rejected.
    """

    if _contains_control(name):
        raise _public_error("unsafe_zip_entry")

    drive, _ = ntpath.splitdrive(name)
    if drive:
        # Covers drive prefixes (including drive-relative paths), UNC shares,
        # and extended Windows device/UNC prefixes.
        raise _public_error("unsafe_zip_entry")

    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise _public_error("unsafe_zip_entry")

    components: list[str] = []
    for component in normalized.split("/"):
        if component in {"", "."}:
            continue
        if component == ".." or _contains_control(component):
            raise _public_error("unsafe_zip_entry")
        if ntpath.splitdrive(component)[0]:
            # A drive-relative component must remain unsafe even when hidden
            # behind leading ``.`` segments or a parent directory.
            raise _public_error("unsafe_zip_entry")
        components.append(component)

    if not components:
        raise _public_error("unsafe_zip_entry")

    return tuple(components), normalized.endswith("/")


def _classify_member(
    info: zipfile.ZipInfo, name_declares_directory: bool
) -> bool:
    """Return whether a member is a directory, rejecting special file types."""

    if type(info.external_attr) is not int:
        raise _public_error("invalid_zip")

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    allowed_types = {0, stat.S_IFREG, stat.S_IFDIR}
    if file_type not in allowed_types:
        # Includes symlinks, sockets, FIFOs, block devices and char devices.
        raise _public_error("unsafe_zip_entry")

    dos_directory = bool(info.external_attr & 0x10)
    if file_type == stat.S_IFREG:
        if name_declares_directory or dos_directory:
            raise _public_error("unsafe_zip_entry")
        is_directory = False
    elif file_type == stat.S_IFDIR:
        is_directory = True
    else:
        # Many normal ZIP writers store Unix permission bits without an
        # explicit S_IFREG type. DOS archives likewise rely on name/attributes.
        is_directory = name_declares_directory or info.is_dir() or dos_directory

    # Directory payloads are ambiguous and are never needed by this pipeline.
    if is_directory and (info.file_size != 0 or info.compress_size != 0):
        raise _public_error("unsafe_zip_entry")

    return is_directory


def _is_within(root: Path, target: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(target))) == os.fspath(root)
    except (TypeError, ValueError, OSError):
        return False


def _reject_existing_symlink(root: Path, components: Sequence[str]) -> None:
    """Reject symlinks already present below the extraction root."""

    current = root
    for component in components:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # Descendants cannot exist once one component is absent.
            return
        except OSError as exc:
            raise _public_error("zip_extraction_failed") from exc
        if stat.S_ISLNK(mode):
            raise _public_error("unsafe_zip_entry")


def _resolved_target(root: Path, components: Sequence[str]) -> Path:
    try:
        target = root.joinpath(*components).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _public_error("zip_extraction_failed") from exc

    if target == root or not _is_within(root, target):
        raise _public_error("unsafe_zip_entry")
    _reject_existing_symlink(root, components)
    return target


def _destination_key(path: Path) -> str:
    """Use the host filesystem's path normalization for duplicate detection."""

    return os.path.normcase(os.path.normpath(os.fspath(path)))


class ZipValidator:
    """Inspect untrusted ZIP metadata and extract approved regular files safely."""

    def __init__(self, config: AppConfig) -> None:
        self._max_entries = config.max_zip_entries
        self._max_entry_bytes = config.max_zip_entry_bytes
        self._max_total_bytes = config.max_uncompressed_bytes
        self._max_compression_ratio = Decimal(str(config.max_compression_ratio))

        if type(self._max_entries) is not int or self._max_entries <= 0:
            raise ValueError("max_zip_entries must be a positive integer")
        if type(self._max_entry_bytes) is not int or self._max_entry_bytes < 0:
            raise ValueError("max_zip_entry_bytes must be a non-negative integer")
        if type(self._max_total_bytes) is not int or self._max_total_bytes < 0:
            raise ValueError("max_uncompressed_bytes must be a non-negative integer")
        if (
            not self._max_compression_ratio.is_finite()
            or self._max_compression_ratio < 1
        ):
            raise ValueError("max_compression_ratio must be finite and at least one")

    def inspect(self, archive_path: Path, extraction_root: Path) -> ArchivePlan:
        """Materialize and validate every member without writing any output.

        ZIP structure/enumeration errors are mapped to ``invalid_zip``. Path,
        type, duplicate and declared-limit checks all finish before this method
        returns a plan that can be passed to :meth:`extract`.
        """

        try:
            root = Path(extraction_root).resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _public_error("zip_extraction_failed") from exc

        try:
            with zipfile.ZipFile(Path(archive_path), mode="r") as archive:
                entries = tuple(archive.infolist())
        except PublicError:
            raise
        except Exception as exc:
            raise _public_error("invalid_zip") from exc

        return self._build_plan(entries, root)

    def extract(
        self, archive_path: Path, plan: ArchivePlan
    ) -> list[ExtractedEntry]:
        """Extract an inspected plan with actual byte limits and no overwrite.

        The archive is fully re-enumerated and compared with the supplied plan
        before the first directory or file is created. Only a wholly successful
        extraction returns its list of regular files. Artifacts created by this
        call are removed on failure as an additional defense; the ingestion
        service remains responsible for deleting its complete staging area.
        """

        if not isinstance(plan, ArchivePlan):
            raise _public_error("zip_extraction_failed")

        if not plan.entries:
            # An empty archive has no extraction root encoded in its plan. It is
            # still re-enumerated so a replaced archive cannot be accepted.
            try:
                with zipfile.ZipFile(Path(archive_path), mode="r") as archive:
                    entries = tuple(archive.infolist())
            except Exception as exc:
                raise _public_error("invalid_zip") from exc
            if entries:
                raise _public_error("invalid_zip")
            return []

        root = self._infer_root(plan)
        created_files: list[Path] = []
        created_directories: list[Path] = []
        archive: zipfile.ZipFile | None = None
        phase = "validation"

        try:
            archive = zipfile.ZipFile(Path(archive_path), mode="r")
            infos = tuple(archive.infolist())
            current_plan = self._build_plan(infos, root)
            if current_plan != plan:
                raise _public_error("invalid_zip")

            # Everything above is read-only. Filesystem mutation starts here.
            phase = "extraction"
            self._ensure_directory(root, created_directories)

            extracted: list[ExtractedEntry] = []
            total_written = 0
            for info, entry in zip(infos, plan.entries, strict=True):
                components = tuple(entry.relative_path.split("/"))
                self._revalidate_target(root, entry, components)

                if entry.is_directory:
                    self._ensure_path_directories(
                        root, components, created_directories
                    )
                    continue

                self._ensure_path_directories(
                    root, components[:-1], created_directories
                )
                written, total_written = self._extract_regular_file(
                    archive=archive,
                    info=info,
                    entry=entry,
                    total_written=total_written,
                    created_files=created_files,
                )
                extracted.append(
                    ExtractedEntry(
                        ordinal=entry.ordinal,
                        path=entry.resolved_target,
                        display_name=entry.relative_path,
                        size=written,
                    )
                )

            archive.close()
            archive = None
            return extracted
        except PublicError:
            self._rollback_created(created_files, created_directories)
            raise
        except Exception as exc:
            self._rollback_created(created_files, created_directories)
            code = "invalid_zip" if phase == "validation" else "zip_extraction_failed"
            raise _public_error(code) from exc
        finally:
            if archive is not None:
                try:
                    archive.close()
                except Exception:
                    pass

    def _build_plan(
        self, entries: Sequence[zipfile.ZipInfo], root: Path
    ) -> ArchivePlan:
        if len(entries) > self._max_entries:
            raise _public_error("zip_entry_count_exceeded")

        planned: list[ArchiveEntryPlan] = []
        destinations: dict[str, tuple[Path, bool]] = {}
        regular_targets: set[Path] = set()
        declared_total = 0

        for ordinal, info in enumerate(entries):
            if not isinstance(info, zipfile.ZipInfo):
                raise _public_error("invalid_zip")
            if (
                type(info.file_size) is not int
                or type(info.compress_size) is not int
                or info.file_size < 0
                or info.compress_size < 0
            ):
                raise _public_error("invalid_zip")

            raw_name = _raw_member_name(info)
            components, name_declares_directory = _normalize_member_path(raw_name)
            is_directory = _classify_member(info, name_declares_directory)
            target = _resolved_target(root, components)
            key = _destination_key(target)

            if key in destinations:
                raise _public_error("unsafe_zip_entry")
            if any(parent in regular_targets for parent in target.parents):
                raise _public_error("unsafe_zip_entry")
            if not is_directory and any(
                target in existing_target.parents
                for existing_target, _ in destinations.values()
            ):
                raise _public_error("unsafe_zip_entry")

            if info.file_size > self._max_entry_bytes:
                raise _public_error("zip_entry_size_exceeded")

            declared_total += info.file_size
            if declared_total > self._max_total_bytes:
                raise _public_error("zip_total_size_exceeded")

            if info.file_size > 0 and info.compress_size == 0:
                raise _public_error("zip_compression_ratio_exceeded")
            if info.compress_size > 0 and (
                Decimal(info.file_size)
                > self._max_compression_ratio * Decimal(info.compress_size)
            ):
                raise _public_error("zip_compression_ratio_exceeded")

            # The application does not accept passwords. Detecting encryption
            # here avoids creating outputs before ``ZipFile.open`` would fail.
            if info.flag_bits & 0x1:
                raise _public_error("zip_extraction_failed")

            relative_path = "/".join(components)
            planned_entry = ArchiveEntryPlan(
                ordinal=ordinal,
                archive_name=raw_name,
                relative_path=relative_path,
                resolved_target=target,
                is_directory=is_directory,
                declared_size=info.file_size,
                compressed_size=info.compress_size,
            )
            planned.append(planned_entry)
            destinations[key] = (target, is_directory)
            if not is_directory:
                regular_targets.add(target)

        return ArchivePlan(
            entries=tuple(planned), declared_total_bytes=declared_total
        )

    @staticmethod
    def _infer_root(plan: ArchivePlan) -> Path:
        inferred_root: Path | None = None

        for entry in plan.entries:
            try:
                components, _ = _normalize_member_path(entry.relative_path)
            except PublicError as exc:
                raise _public_error("zip_extraction_failed") from exc
            if "/".join(components) != entry.relative_path:
                raise _public_error("zip_extraction_failed")

            target = Path(entry.resolved_target)
            if not target.is_absolute():
                raise _public_error("zip_extraction_failed")
            candidate = target
            for _ in components:
                candidate = candidate.parent

            if inferred_root is None:
                inferred_root = candidate
            elif candidate != inferred_root:
                raise _public_error("zip_extraction_failed")

            if candidate.joinpath(*components) != target:
                raise _public_error("zip_extraction_failed")

        if inferred_root is None:
            raise _public_error("zip_extraction_failed")
        return inferred_root

    @staticmethod
    def _revalidate_target(
        root: Path, entry: ArchiveEntryPlan, components: Sequence[str]
    ) -> None:
        try:
            current = root.joinpath(*components).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _public_error("zip_extraction_failed") from exc
        if current != entry.resolved_target or not _is_within(root, current):
            raise _public_error("unsafe_zip_entry")
        _reject_existing_symlink(root, components)

    @staticmethod
    def _ensure_directory(path: Path, created: list[Path]) -> None:
        """Create one directory exclusively or validate an existing directory."""

        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            try:
                os.mkdir(path, 0o700)
                created.append(path)
                mode = os.lstat(path).st_mode
            except FileExistsError:
                mode = os.lstat(path).st_mode
            except OSError as exc:
                raise _public_error("zip_extraction_failed") from exc
        except OSError as exc:
            raise _public_error("zip_extraction_failed") from exc

        if stat.S_ISLNK(mode):
            raise _public_error("unsafe_zip_entry")
        if not stat.S_ISDIR(mode):
            raise _public_error("zip_extraction_failed")
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY
            flags |= getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            os.fchmod(descriptor, 0o700)
        except OSError as exc:
            raise _public_error("zip_extraction_failed") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _ensure_path_directories(
        self,
        root: Path,
        components: Iterable[str],
        created: list[Path],
    ) -> None:
        current = root
        for component in components:
            current = current / component
            if not _is_within(root, current):
                raise _public_error("unsafe_zip_entry")
            self._ensure_directory(current, created)

    def _extract_regular_file(
        self,
        *,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        entry: ArchiveEntryPlan,
        total_written: int,
        created_files: list[Path],
    ) -> tuple[int, int]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)

        descriptor: int | None = None
        entry_written = 0
        try:
            descriptor = os.open(entry.resolved_target, flags, 0o600)
            created_files.append(entry.resolved_target)

            with archive.open(info, mode="r") as source:
                while True:
                    entry_remaining = self._max_entry_bytes - entry_written
                    total_remaining = self._max_total_bytes - total_written
                    read_size = min(
                        _COPY_BUFFER_SIZE,
                        entry_remaining + 1,
                        total_remaining + 1,
                    )
                    block = source.read(read_size)
                    if block == b"":
                        break
                    if not isinstance(block, bytes):
                        raise OSError("ZIP member stream returned non-bytes data")

                    block_size = len(block)
                    if entry_written + block_size > self._max_entry_bytes:
                        raise _public_error("zip_entry_size_exceeded")
                    if total_written + block_size > self._max_total_bytes:
                        raise _public_error("zip_total_size_exceeded")

                    view = memoryview(block)
                    while view:
                        amount = os.write(descriptor, view)
                        if amount <= 0:
                            raise OSError("short write while extracting ZIP member")
                        entry_written += amount
                        total_written += amount
                        view = view[amount:]

            if entry_written != entry.declared_size:
                raise _public_error("zip_extraction_failed")
            os.fchmod(descriptor, 0o600)
            os.close(descriptor)
            descriptor = None
            return entry_written, total_written
        except PublicError:
            raise
        except Exception as exc:
            raise _public_error("zip_extraction_failed") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    @staticmethod
    def _rollback_created(files: Sequence[Path], directories: Sequence[Path]) -> None:
        for path in reversed(files):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        for path in reversed(directories):
            try:
                path.rmdir()
            except OSError:
                pass


# Kept close to the discovery helpers so the security-critical ZipValidator
# implementation above remains unchanged.
from unicodedata import normalize as _unicode_normalize


_NO_PDFS_MESSAGE: Final = "O arquivo ZIP não contém documentos PDF."
_NON_PDF_WARNING_TEMPLATE: Final = (
    "{document}: arquivo ignorado porque não possui a extensão .pdf."
)


def normalize_display_name(value: str) -> str:
    """Return a safe, NFC-normalized POSIX path for user-facing output.

    ``value`` must describe a relative archive path, never an extracted system
    path. Both slash styles are understood so a Windows-looking member cannot
    smuggle an absolute path or a drive prefix into output. C0/C1 and DEL
    controls are removed from each segment; traversal and drive segments are
    rejected rather than rewritten into a misleading display name.
    """

    if not isinstance(value, str) or not value:
        raise _public_error("unsafe_zip_entry")

    drive, _ = ntpath.splitdrive(value)
    normalized_separators = value.replace("\\", "/")
    if drive or normalized_separators.startswith("/"):
        raise _public_error("unsafe_zip_entry")

    safe_segments: list[str] = []
    for raw_segment in normalized_separators.split("/"):
        if raw_segment in {"", "."}:
            continue
        if raw_segment == ".." or ntpath.splitdrive(raw_segment)[0]:
            raise _public_error("unsafe_zip_entry")

        without_controls = "".join(
            character
            for character in raw_segment
            if not (
                ord(character) < 0x20
                or 0x7F <= ord(character) <= 0x9F
            )
        )
        safe_segment = _unicode_normalize("NFC", without_controls)
        if (
            not safe_segment
            or safe_segment in {".", ".."}
            or ntpath.splitdrive(safe_segment)[0]
            or "/" in safe_segment
            or "\\" in safe_segment
        ):
            raise _public_error("unsafe_zip_entry")
        safe_segments.append(safe_segment)

    if not safe_segments:
        raise _public_error("unsafe_zip_entry")
    return "/".join(safe_segments)


class WarningCollector:
    """Collect one warning per structural event while preserving first order.

    A key is the tuple ``(code, entry_ordinal, page)``. Re-adding the same key
    never adds a second message, even if a later caller supplies different
    text. Distinct archive occurrences remain distinct because their ordinals
    differ.
    """

    def __init__(self) -> None:
        self._messages: dict[tuple[str, int, int | None], str] = {}

    def add(
        self,
        code: str,
        entry_ordinal: int,
        page: int | None,
        message: str,
    ) -> bool:
        """Add an event and return ``True`` only when its key was new."""

        if not isinstance(code, str) or not code:
            raise ValueError("warning code must be a non-empty string")
        if type(entry_ordinal) is not int or entry_ordinal < 0:
            raise ValueError("warning entry ordinal must be a non-negative integer")
        if page is not None and (type(page) is not int or page < 1):
            raise ValueError("warning page must be a positive integer or None")
        if not isinstance(message, str) or not message:
            raise ValueError("warning message must be a non-empty string")

        key = (code, entry_ordinal, page)
        if key in self._messages:
            return False
        self._messages[key] = message
        return True

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return an immutable snapshot in first-occurrence order."""

        return tuple(self._messages.values())

    def as_tuple(self) -> tuple[str, ...]:
        """Return the current warning snapshot."""

        return self.warnings

    def __iter__(self):
        return iter(self.warnings)

    def __len__(self) -> int:
        return len(self._messages)


def discover_pdf_entries(
    entries: Iterable[ExtractedEntry],
    warnings: WarningCollector | None = None,
) -> tuple[ExtractedEntry, ...]:
    """Select PDF entries in ZIP declaration order and collect non-PDF warnings.

    ``ExtractedEntry`` is the publication boundary of :class:`ZipValidator` and
    therefore represents only regular files from a completely extracted ZIP.
    Discovery deliberately walks this finite list instead of the staging
    filesystem, so unplanned files cannot be included and nested archive paths
    remain represented by their safe relative display names.
    """

    collector = warnings if warnings is not None else WarningCollector()
    if not isinstance(collector, WarningCollector):
        raise TypeError("warnings must be a WarningCollector")

    materialized = tuple(entries)
    if any(not isinstance(entry, ExtractedEntry) for entry in materialized):
        raise TypeError("entries must contain only ExtractedEntry values")

    ordered_entries = sorted(materialized, key=lambda entry: entry.ordinal)
    candidates: list[ExtractedEntry] = []
    for entry in ordered_entries:
        display_name = normalize_display_name(entry.display_name)
        normalized_entry = ExtractedEntry(
            ordinal=entry.ordinal,
            path=entry.path,
            display_name=display_name,
            size=entry.size,
        )
        if display_name.casefold().endswith(".pdf"):
            candidates.append(normalized_entry)
            continue

        collector.add(
            "non_pdf",
            entry.ordinal,
            None,
            _NON_PDF_WARNING_TEMPLATE.format(document=display_name),
        )

    if not candidates:
        raise PublicError(
            code="no_pdfs",
            message=_NO_PDFS_MESSAGE,
            http_status=422,
        )
    return tuple(candidates)


# A concise alias keeps call sites readable without introducing another code
# path; both names obey the same immutable return contract.
discover_pdfs = discover_pdf_entries


_PDF_READ_WARNING_TEMPLATE: Final = (
    "{document}: arquivo ignorado por falha de leitura; substitua-o ou "
    "remova-o antes de reenviar."
)
_EMPTY_PDF_PAGE_WARNING_TEMPLATE: Final = (
    "{document}, página {page}: nenhum caractere não branco foi extraído. "
    "O MVP não executa OCR; aplique OCR antes de uma nova importação se a "
    "página contiver texto em imagem."
)
_NO_READABLE_PDFS_MESSAGE: Final = (
    "Nenhum PDF do arquivo pôde ser lido. Substitua ou remova os arquivos "
    "com problema."
)


def _write_utf8_spool(path: Path, text: str) -> None:
    """Create one private spool file without transforming its text."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)

    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        stream = os.fdopen(
            descriptor,
            mode="w",
            encoding="utf-8",
            errors="strict",
            newline="",
        )
        descriptor = None  # ``stream`` owns the descriptor from this point.
        with stream:
            written = stream.write(text)
            if written != len(text):
                raise OSError("short write while spooling PDF page text")
            stream.flush()
            os.fchmod(stream.fileno(), 0o600)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_utf8_spool(path: Path) -> str:
    """Read one private spool file with newline translation disabled."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)

    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("PDF text spool is not a regular file")
        stream = os.fdopen(
            descriptor,
            mode="r",
            encoding="utf-8",
            errors="strict",
            newline="",
        )
        descriptor = None  # ``stream`` owns the descriptor from this point.
        with stream:
            return stream.read()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of a regular file using bounded reads.

    The digest is calculated directly from the original bytes and never from
    extracted PDF text, a filename, or upload metadata. ``O_NOFOLLOW`` is used
    when available so the identity stage cannot silently follow a replacement
    symlink between ZIP extraction and PDF processing.
    """

    try:
        source_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise TypeError("path must be path-like") from exc

    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)

    descriptor: int | None = None
    digest = sha256()
    try:
        descriptor = os.open(source_path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("document identity source is not a regular file")

        while True:
            block = os.read(descriptor, _COPY_BUFFER_SIZE)
            if block == b"":
                break
            digest.update(block)
        return digest.hexdigest()
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def make_chunk_id(
    document_id: str,
    page: int,
    start: int,
    schema: str = "char-v1",
) -> str:
    """Build the deterministic ``char-v1`` identifier for one chunk.

    The byte payload is exactly ``schema NUL document NUL page NUL start``.
    Schema is UTF-8 and the remaining fields are ASCII, matching the persisted
    identity contract without incorporating text, filename, or transaction ID.
    """

    if schema != "char-v1":
        raise ValueError("unsupported chunk schema version")
    if not isinstance(document_id, str) or not document_id or "\0" in document_id:
        raise ValueError("document_id must be a non-empty NUL-free string")
    if type(page) is not int or page < 1:
        raise ValueError("page must be a positive integer")
    if type(start) is not int or start < 0:
        raise ValueError("start must be a non-negative integer")

    try:
        document_bytes = document_id.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("document_id must contain only ASCII characters") from exc

    payload = b"\0".join(
        (
            schema.encode("utf-8"),
            document_bytes,
            str(page).encode("ascii"),
            str(start).encode("ascii"),
        )
    )
    return "chk_" + sha256(payload).hexdigest()


class ChunkingService:
    """Split extracted page text into deterministic, traceable character chunks."""

    def __init__(
        self,
        chunk_size: AppConfig | ChunkingProfile | int,
        chunk_overlap: int | None = None,
        schema_version: str = "char-v1",
    ) -> None:
        """Create a chunker from application config, a profile, or raw values."""

        if isinstance(chunk_size, AppConfig):
            if chunk_overlap is not None or schema_version != "char-v1":
                raise ValueError(
                    "chunk_overlap/schema_version must not override AppConfig"
                )
            size = chunk_size.chunk_size
            overlap = chunk_size.chunk_overlap
            schema = "char-v1"
        elif isinstance(chunk_size, ChunkingProfile):
            if chunk_overlap is not None or schema_version != "char-v1":
                raise ValueError(
                    "chunk_overlap/schema_version must not override ChunkingProfile"
                )
            size = chunk_size.size
            overlap = chunk_size.overlap
            schema = chunk_size.schema_version
        else:
            size = chunk_size
            overlap = chunk_overlap
            schema = schema_version

        if type(size) is not int or size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(overlap) is not int or overlap < 0 or overlap >= size:
            raise ValueError(
                "chunk_overlap must be an integer from zero to chunk_size - 1"
            )
        if schema != "char-v1":
            raise ValueError("unsupported chunk schema version")

        self._profile = ChunkingProfile(
            size=size,
            overlap=overlap,
            schema_version="char-v1",
        )
        self._stride = size - overlap

    @property
    def profile(self) -> ChunkingProfile:
        """Return the immutable profile used for every emitted chunk."""

        return self._profile

    def split_page(
        self,
        page: PdfPage,
        document_id: str,
        display_name: str,
        transaction_id: str,
    ) -> list[Chunk]:
        """Split one page without modifying text or crossing page boundaries."""

        if not isinstance(page, PdfPage):
            raise TypeError("page must be a PdfPage")
        if type(page.human_page) is not int or page.human_page < 1:
            raise ValueError("page number must be a positive integer")
        if not isinstance(page.text, str):
            raise TypeError("page text must be a string")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError("display_name must be a non-empty string")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("transaction_id must be a non-empty string")

        # Whitespace-only pages intentionally produce no chunks, but mixed
        # content is sliced from the unmodified original string below.
        if page.text.strip() == "":
            return []

        chunks: list[Chunk] = []
        start = 0
        text_length = len(page.text)
        while start < text_length:
            end = min(start + self._profile.size, text_length)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        document_id,
                        page.human_page,
                        start,
                        self._profile.schema_version,
                    ),
                    document_id=document_id,
                    display_name=display_name,
                    human_page=page.human_page,
                    start_offset=start,
                    text=page.text[start:end],
                    transaction_id=transaction_id,
                )
            )
            if end == text_length:
                break
            start += self._stride

        return chunks

    def split_document(
        self,
        document: ExtractedDocument,
        transaction_id: str,
    ) -> list[Chunk]:
        """Split all pages in extractor order while preserving page provenance."""

        if not isinstance(document, ExtractedDocument):
            raise TypeError("document must be an ExtractedDocument")

        chunks: list[Chunk] = []
        for page in document.pages:
            chunks.extend(
                self.split_page(
                    page,
                    document.document_id,
                    document.display_name,
                    transaction_id,
                )
            )
        return chunks


class PdfExtractor:
    """Extract complete PDFs page by page through a faithful UTF-8 spool.

    A document is returned only after every page has been enumerated,
    extracted with ``get_text("text")``, round-tripped through its private
    spool, and the spool has been removed. Any failure discards all pages from
    that document and produces one sanitized read warning. Empty pages remain
    in the returned document so callers count them, while their exact text is
    preserved for the chunking service to recognize and skip.
    """

    def __init__(self, warnings: WarningCollector | None = None) -> None:
        if warnings is not None and not isinstance(warnings, WarningCollector):
            raise TypeError("warnings must be a WarningCollector")
        self._warnings = warnings if warnings is not None else WarningCollector()

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return warnings emitted by successful and discarded documents."""

        return self._warnings.warnings

    def extract(
        self,
        entry: ExtractedEntry,
        spool_root: Path,
        document_id: str | None = None,
        warnings: WarningCollector | None = None,
    ) -> "ExtractedDocument | None":
        """Return a fully readable document or ``None`` after a read failure.

        Callers that already calculated an identity for deduplication may pass
        it explicitly. Otherwise the original regular file is hashed by
        :func:`sha256_file` before PyMuPDF is opened, keeping identity based
        exclusively on source bytes. No partial pages or empty-page warnings
        escape a failed document.
        """

        if not isinstance(entry, ExtractedEntry):
            raise TypeError("entry must be an ExtractedEntry")
        if type(entry.ordinal) is not int or entry.ordinal < 0:
            raise ValueError("entry ordinal must be a non-negative integer")
        if document_id is not None:
            if not isinstance(document_id, str):
                raise TypeError("document_id must be a string or None")
            if not document_id or "\0" in document_id:
                raise ValueError("document_id must be non-empty and NUL-free")
            try:
                document_id.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "document_id must contain only ASCII characters"
                ) from exc
        if warnings is not None and not isinstance(warnings, WarningCollector):
            raise TypeError("warnings must be a WarningCollector")

        collector = warnings if warnings is not None else self._warnings
        display_name = normalize_display_name(entry.display_name)
        try:
            root = Path(spool_root)
        except (TypeError, ValueError) as exc:
            raise TypeError("spool_root must be path-like") from exc

        spool_directory: Path | None = None
        try:
            effective_document_id = (
                sha256_file(entry.path) if document_id is None else document_id
            )
            spool_directory = self._create_spool_directory(root, entry.ordinal)
            spool_pages = self._extract_to_spool(entry, spool_directory)
            pages: list[PdfPage] = []
            empty_pages: list[int] = []
            for human_page, spool_path in spool_pages:
                text = _read_utf8_spool(spool_path)
                pages.append(PdfPage(human_page=human_page, text=text))
                if text.strip() == "":
                    empty_pages.append(human_page)

            # Publication happens only after cleanup succeeds. If cleanup
            # fails, the complete document is treated as unreadable and none
            # of its pages or page-level warnings are returned.
            self._remove_spool_directory(spool_directory)
            spool_directory = None
        except Exception:
            if spool_directory is not None:
                self._discard_spool_directory(spool_directory)
            collector.add(
                "pdf_read_failed",
                entry.ordinal,
                None,
                _PDF_READ_WARNING_TEMPLATE.format(document=display_name),
            )
            return None

        for human_page in empty_pages:
            collector.add(
                "pdf_page_empty",
                entry.ordinal,
                human_page,
                _EMPTY_PDF_PAGE_WARNING_TEMPLATE.format(
                    document=display_name,
                    page=human_page,
                ),
            )

        return ExtractedDocument(
            document_id=effective_document_id,
            display_name=display_name,
            pages=tuple(pages),
        )

    def extract_all(
        self,
        entries: Iterable[ExtractedEntry],
        spool_root: Path,
        document_ids: dict[int, str] | None = None,
        warnings: WarningCollector | None = None,
    ) -> "tuple[ExtractedDocument, ...]":
        """Extract all candidates, continuing past unreadable PDFs.

        The returned tuple contains every and only completely readable PDF in
        input order. If no candidate is readable, a stable 422 error is raised
        after each candidate has emitted its single read warning. Documents
        with zero pages or only empty pages are readable and therefore prevent
        this rejection; retaining all their pages makes upload counts include
        empty pages naturally.
        """

        if document_ids is not None and not isinstance(document_ids, dict):
            raise TypeError("document_ids must be a dict keyed by entry ordinal")
        if warnings is not None and not isinstance(warnings, WarningCollector):
            raise TypeError("warnings must be a WarningCollector")

        readable: list[ExtractedDocument] = []
        for entry in entries:
            if not isinstance(entry, ExtractedEntry):
                raise TypeError("entries must contain only ExtractedEntry values")
            document_id: str | None = None
            if document_ids is not None:
                document_id = document_ids.get(entry.ordinal)
            document = self.extract(
                entry,
                spool_root,
                document_id=document_id,
                warnings=warnings,
            )
            if document is not None:
                readable.append(document)

        if not readable:
            raise PublicError(
                code="no_readable_pdfs",
                message=_NO_READABLE_PDFS_MESSAGE,
                http_status=422,
            )
        return tuple(readable)

    @staticmethod
    def _create_spool_directory(root: Path, entry_ordinal: int) -> Path:
        import tempfile

        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_mode = os.lstat(root).st_mode
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise OSError("PDF spool root is not a real directory")
        os.chmod(root, 0o700)

        directory = Path(
            tempfile.mkdtemp(prefix=f"pdf-{entry_ordinal}-", dir=root)
        )
        os.chmod(directory, 0o700)
        return directory

    @staticmethod
    def _extract_to_spool(
        entry: ExtractedEntry,
        spool_directory: Path,
    ) -> tuple[tuple[int, Path], ...]:
        import pymupdf

        source_mode = os.lstat(entry.path).st_mode
        if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
            raise OSError("PDF candidate is not a regular file")

        spooled: list[tuple[int, Path]] = []
        with pymupdf.open(filename=entry.path) as document:
            if bool(document.needs_pass):
                raise ValueError("encrypted PDF requires authentication")
            page_count = document.page_count
            if type(page_count) is not int or page_count < 0:
                raise ValueError("invalid PDF page count")

            for page_index in range(page_count):
                page = document.load_page(page_index)
                text = page.get_text("text")
                if not isinstance(text, str):
                    raise TypeError("PyMuPDF returned non-text page content")

                human_page = page_index + 1
                spool_path = spool_directory / f"page-{human_page:08d}.txt"
                _write_utf8_spool(spool_path, text)
                spooled.append((human_page, spool_path))

        return tuple(spooled)

    @staticmethod
    def _remove_spool_directory(path: Path) -> None:
        import shutil

        shutil.rmtree(path)

    @staticmethod
    def _discard_spool_directory(path: Path) -> None:
        import shutil

        try:
            shutil.rmtree(path)
        except OSError:
            # The upload-level ``finally`` remains the final cleanup boundary.
            pass


class _ManifestLookup(Protocol):
    """Read-only manifest surface used during ingestion deduplication."""

    def is_duplicate(self, key: DocumentManifestKey) -> bool:
        """Return whether *key* belongs to a committed transaction."""


class _TransactionalVectorStore(VectorStore, Protocol):
    """Vector-store surface that shares the non-blocking ingestion mutex."""

    def ingestion_guard(self) -> AbstractContextManager[None]:
        """Return the guard held from definitive checks through commit."""


@dataclass(frozen=True, slots=True)
class _StagingLayout:
    archive_path: Path
    extraction_root: Path
    spool_root: Path
    cleanup_root: Path
    archive_outside_cleanup_root: bool


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    entry_ordinal: int
    manifest: DocumentManifest
    chunk_ids: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedIngestion:
    transaction_id: str
    space: VectorSpace
    profile: ChunkingProfile
    documents: tuple[_PreparedDocument, ...]
    chunks: tuple[StoredChunk, ...]
    base_warnings: tuple[str, ...]


_DUPLICATE_WARNING_TEMPLATE: Final = (
    "{document}: documento duplicado; nenhuma nova indexação foi realizada."
)
_INTERNAL_ERROR_MESSAGE: Final = (
    "Não foi possível concluir a operação devido a uma falha interna."
)
_VECTOR_STORE_UNAVAILABLE_MESSAGE: Final = (
    "A base vetorial local está indisponível. Verifique o armazenamento "
    "configurado e suas permissões."
)
_CHUNK_IDENTITY_CONFLICT_MESSAGE: Final = (
    "Foi detectado um conflito de identidade de chunk. A importação foi "
    "cancelada sem alterar a base."
)


def _embedding_failed_for_model(model: str) -> PublicError:
    return PublicError(
        code="embedding_failed",
        message=(
            f"A geração local de embeddings com {model} falhou. Verifique a "
            "disponibilidade local do modelo e tente novamente."
        ),
        http_status=503,
    )


def _vector_space_mismatch_for_ingestion() -> PublicError:
    return PublicError(
        code="vector_space_mismatch",
        message=(
            "A coleção usa um espaço vetorial incompatível. Recrie o índice "
            "com a configuração atual."
        ),
        http_status=503,
    )


def _vector_store_unavailable_for_ingestion() -> PublicError:
    return PublicError(
        code="vector_store_unavailable",
        message=_VECTOR_STORE_UNAVAILABLE_MESSAGE,
        http_status=503,
    )


def _chunk_identity_conflict_for_ingestion() -> PublicError:
    return PublicError(
        code="chunk_identity_conflict",
        message=_CHUNK_IDENTITY_CONFLICT_MESSAGE,
        http_status=409,
    )


def _internal_ingestion_error() -> PublicError:
    return PublicError(
        code="internal_error",
        message=_INTERNAL_ERROR_MESSAGE,
        http_status=500,
    )


class IngestionService:
    """Coordinate one ZIP upload through one confirmed atomic commit.

    ZIP/PDF work, chunking, and all embedding inference finish before the
    service enters the persistent phase. The complete staging area is then
    removed in ``finally``. Only after cleanup succeeds does the service take
    the vector store's non-blocking ingestion guard, recheck committed
    document identities and chunk collisions, and submit one ``CommitPlan``.
    """

    def __init__(
        self,
        config: AppConfig,
        embedding_provider: EmbeddingProvider,
        manifest_store: _ManifestLookup,
        vector_store: _TransactionalVectorStore,
        *,
        zip_validator: ZipValidator | None = None,
        pdf_extractor: PdfExtractor | None = None,
        chunking_service: ChunkingService | None = None,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be AppConfig")
        self._config = config
        self._embedding_provider = embedding_provider
        self._manifest_store = manifest_store
        self._vector_store = vector_store
        self._zip_validator = zip_validator or ZipValidator(config)
        self._pdf_extractor = pdf_extractor or PdfExtractor()
        self._chunking_service = chunking_service or ChunkingService(config)

        expected_profile = ChunkingProfile(
            size=config.chunk_size,
            overlap=config.chunk_overlap,
            schema_version="char-v1",
        )
        if self._chunking_service.profile != expected_profile:
            raise ValueError("chunking service does not match AppConfig")
        for dependency, method in (
            (embedding_provider, "ensure_ready"),
            (embedding_provider, "embed_documents"),
            (manifest_store, "is_duplicate"),
            (vector_store, "ingestion_guard"),
            (vector_store, "ensure_compatible"),
            (vector_store, "existing_chunks"),
            (vector_store, "commit_chunks"),
        ):
            if not callable(getattr(dependency, method, None)):
                raise TypeError(f"dependency must provide {method}()")

    def ingest(self, archive_path: Path, upload_id: UUID | str) -> UploadResult:
        """Process, clean, revalidate, and atomically confirm one upload."""

        transaction_id = self._normalize_transaction_id(upload_id)
        layout = self._staging_layout(archive_path, transaction_id)
        try:
            prepared = self._prepare(layout, transaction_id)
        finally:
            self._cleanup_staging(layout)
        return self._confirm(prepared)

    def _prepare(
        self,
        layout: _StagingLayout,
        transaction_id: str,
    ) -> _PreparedIngestion:
        warnings = WarningCollector()
        archive_plan = self._zip_validator.inspect(
            layout.archive_path,
            layout.extraction_root,
        )
        extracted_entries = self._zip_validator.extract(
            layout.archive_path,
            archive_plan,
        )
        candidates = discover_pdf_entries(extracted_entries, warnings)

        space = self._validated_space(self._embedding_provider.ensure_ready())
        profile = self._chunking_service.profile
        drafts: list[tuple[_PreparedDocument, tuple[Chunk, ...]]] = []
        seen_readable_documents: set[str] = set()
        recognized_document = False

        for entry in candidates:
            try:
                document_id = sha256_file(entry.path)
            except (OSError, RuntimeError, TypeError, ValueError):
                warnings.add(
                    "pdf_read_failed",
                    entry.ordinal,
                    None,
                    _PDF_READ_WARNING_TEMPLATE.format(
                        document=entry.display_name
                    ),
                )
                continue

            key = DocumentManifestKey(
                document_id=document_id,
                vector_fingerprint=space.fingerprint,
                chunk_size=profile.size,
                chunk_overlap=profile.overlap,
                chunk_schema_version=profile.schema_version,
            )
            if self._manifest_store.is_duplicate(key):
                recognized_document = True
                self._add_duplicate_warning(warnings, entry)
                continue
            if document_id in seen_readable_documents:
                recognized_document = True
                self._add_duplicate_warning(warnings, entry)
                continue

            document_warnings = WarningCollector()
            document = self._pdf_extractor.extract(
                entry,
                layout.spool_root,
                document_id=document_id,
                warnings=document_warnings,
            )
            if document is None:
                # Concrete PdfExtractor already emits this event. Keeping the
                # orchestration-level fallback makes the contract robust for a
                # narrow injected extractor that simply returns ``None``.
                warnings.add(
                    "pdf_read_failed",
                    entry.ordinal,
                    None,
                    _PDF_READ_WARNING_TEMPLATE.format(
                        document=entry.display_name
                    ),
                )
                continue

            self._validate_extracted_document(document, entry, document_id)
            recognized_document = True
            seen_readable_documents.add(document_id)
            chunks = tuple(
                self._chunking_service.split_document(
                    document,
                    transaction_id,
                )
            )
            self._validate_document_chunks(
                chunks,
                document,
                transaction_id,
            )
            manifest = DocumentManifest(
                key=key,
                first_display_name=document.display_name,
                page_count=len(document.pages),
                chunk_count=len(chunks),
                transaction_id=transaction_id,
            )
            drafts.append(
                (
                    _PreparedDocument(
                        entry_ordinal=entry.ordinal,
                        manifest=manifest,
                        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                        warnings=document_warnings.warnings,
                    ),
                    chunks,
                )
            )

        if not recognized_document:
            raise PublicError(
                code="no_readable_pdfs",
                message=_NO_READABLE_PDFS_MESSAGE,
                http_status=422,
            )

        plain_chunks = tuple(
            chunk
            for _, document_chunks in drafts
            for chunk in document_chunks
        )
        chunk_ids = [chunk.chunk_id for chunk in plain_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise _chunk_identity_conflict_for_ingestion()

        vectors = self._embed_all_chunks(plain_chunks, space)
        stored_chunks = tuple(
            StoredChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(plain_chunks, vectors, strict=True)
        )
        return _PreparedIngestion(
            transaction_id=transaction_id,
            space=space,
            profile=profile,
            documents=tuple(draft for draft, _ in drafts),
            chunks=stored_chunks,
            base_warnings=warnings.warnings,
        )

    def _confirm(self, prepared: _PreparedIngestion) -> UploadResult:
        warnings = list(prepared.base_warnings)
        documents: list[_PreparedDocument] = []
        new_chunk_ids: tuple[str, ...] = ()

        # ChromaVectorStore.commit_chunks() detects ownership and reuses this
        # same guard, so duplicate/collision checks and journaled commit share
        # one uninterrupted non-blocking mutation window.
        with self._vector_store.ingestion_guard():
            self._vector_store.ensure_compatible(
                prepared.space,
                prepared.profile,
            )
            for document in prepared.documents:
                if self._manifest_store.is_duplicate(document.manifest.key):
                    warnings.append(
                        _DUPLICATE_WARNING_TEMPLATE.format(
                            document=document.manifest.first_display_name
                        )
                    )
                    continue
                documents.append(document)
                warnings.extend(document.warnings)

            eligible_ids = {
                chunk_id
                for document in documents
                for chunk_id in document.chunk_ids
            }
            chunks = tuple(
                stored
                for stored in prepared.chunks
                if stored.chunk.chunk_id in eligible_ids
            )
            requested_ids = tuple(
                stored.chunk.chunk_id for stored in chunks
            )
            existing = self._read_existing_chunks(requested_ids)
            for stored in chunks:
                current = existing.get(stored.chunk.chunk_id)
                if current is not None and not self._same_logical_chunk(
                    current,
                    stored,
                ):
                    raise _chunk_identity_conflict_for_ingestion()

            new_chunk_ids = tuple(
                chunk_id
                for chunk_id in requested_ids
                if chunk_id not in existing
            )
            if documents:
                plan = CommitPlan(
                    transaction_id=prepared.transaction_id,
                    chunks=chunks,
                    manifests=tuple(
                        document.manifest for document in documents
                    ),
                    new_chunk_ids=new_chunk_ids,
                    documents=len(documents),
                    pages=sum(
                        document.manifest.page_count
                        for document in documents
                    ),
                    warnings=tuple(warnings),
                )
                self._vector_store.commit_chunks(plan)

        return UploadResult(
            success=True,
            documents=len(documents),
            pages=sum(
                document.manifest.page_count for document in documents
            ),
            chunks=len(new_chunk_ids),
            warnings=tuple(warnings),
        )

    def _read_existing_chunks(
        self,
        requested_ids: tuple[str, ...],
    ) -> Mapping[str, StoredChunk]:
        if not requested_ids:
            return {}
        try:
            existing = self._vector_store.existing_chunks(requested_ids)
        except PublicError:
            raise
        except Exception as exc:
            raise _vector_store_unavailable_for_ingestion() from exc
        if not isinstance(existing, Mapping):
            raise _vector_store_unavailable_for_ingestion()
        requested = set(requested_ids)
        if any(
            type(chunk_id) is not str
            or chunk_id not in requested
            or not isinstance(stored, StoredChunk)
            or stored.chunk.chunk_id != chunk_id
            for chunk_id, stored in existing.items()
        ):
            raise _vector_store_unavailable_for_ingestion()
        return existing

    def _embed_all_chunks(
        self,
        chunks: tuple[Chunk, ...],
        space: VectorSpace,
    ) -> tuple[tuple[float, ...], ...]:
        if not chunks:
            return ()
        try:
            raw_vectors = self._embedding_provider.embed_documents(
                [chunk.text for chunk in chunks]
            )
        except PublicError:
            raise
        except Exception as exc:
            raise _embedding_failed_for_model(self._config.embedding_model) from exc

        try:
            rows = list(raw_vectors)
        except Exception as exc:
            raise _embedding_failed_for_model(self._config.embedding_model) from exc
        if len(rows) != len(chunks):
            raise _embedding_failed_for_model(self._config.embedding_model)

        vectors: list[tuple[float, ...]] = []
        for row in rows:
            if isinstance(row, (str, bytes, bytearray, Mapping)):
                raise _embedding_failed_for_model(self._config.embedding_model)
            try:
                components = list(row)
            except Exception as exc:
                raise _embedding_failed_for_model(
                    self._config.embedding_model
                ) from exc
            if len(components) != space.dimension:
                raise _embedding_failed_for_model(self._config.embedding_model)

            vector: list[float] = []
            for component in components:
                if isinstance(component, bool) or not isinstance(component, Real):
                    raise _embedding_failed_for_model(
                        self._config.embedding_model
                    )
                value = float(component)
                if not math.isfinite(value):
                    raise _embedding_failed_for_model(
                        self._config.embedding_model
                    )
                vector.append(value)
            vectors.append(tuple(vector))
        return tuple(vectors)

    def _validated_space(self, value: object) -> VectorSpace:
        if (
            not isinstance(value, VectorSpace)
            or value.model != self._config.embedding_model
            or type(value.dimension) is not int
            or value.dimension <= 0
            or value.normalized is not True
            or value.metric != "cosine"
        ):
            raise _vector_space_mismatch_for_ingestion()
        return value

    @staticmethod
    def _validate_extracted_document(
        document: ExtractedDocument,
        entry: ExtractedEntry,
        expected_document_id: str,
    ) -> None:
        if (
            not isinstance(document, ExtractedDocument)
            or document.document_id != expected_document_id
            or document.display_name != entry.display_name
            or any(
                not isinstance(page, PdfPage)
                or type(page.human_page) is not int
                or page.human_page != index
                or not isinstance(page.text, str)
                for index, page in enumerate(document.pages, start=1)
            )
        ):
            raise _internal_ingestion_error()

    @staticmethod
    def _validate_document_chunks(
        chunks: tuple[Chunk, ...],
        document: ExtractedDocument,
        transaction_id: str,
    ) -> None:
        page_texts = {
            page.human_page: page.text for page in document.pages
        }
        if any(
            not isinstance(chunk, Chunk)
            or chunk.document_id != document.document_id
            or chunk.display_name != document.display_name
            or chunk.transaction_id != transaction_id
            or chunk.human_page not in page_texts
            or type(chunk.start_offset) is not int
            or chunk.start_offset < 0
            or not chunk.text
            or page_texts[chunk.human_page][
                chunk.start_offset : chunk.start_offset + len(chunk.text)
            ]
            != chunk.text
            for chunk in chunks
        ):
            raise _internal_ingestion_error()

    @staticmethod
    def _same_logical_chunk(
        existing: StoredChunk,
        candidate: StoredChunk,
    ) -> bool:
        old = existing.chunk
        new = candidate.chunk
        return (
            old.chunk_id == new.chunk_id
            and old.text == new.text
            and old.document_id == new.document_id
            and old.display_name == new.display_name
            and old.human_page == new.human_page
            and old.start_offset == new.start_offset
        )

    @staticmethod
    def _add_duplicate_warning(
        warnings: WarningCollector,
        entry: ExtractedEntry,
    ) -> None:
        warnings.add(
            "duplicate_document",
            entry.ordinal,
            None,
            _DUPLICATE_WARNING_TEMPLATE.format(
                document=entry.display_name
            ),
        )

    def _staging_layout(
        self,
        archive_path: Path,
        transaction_id: str,
    ) -> _StagingLayout:
        try:
            upload_root = self._config.upload_folder.resolve(strict=False)
            archive = Path(archive_path).resolve(strict=False)
            if archive == upload_root or not _is_within(upload_root, archive):
                raise ValueError("archive is outside UPLOAD_FOLDER")
            relative = archive.relative_to(upload_root)
            digest = sha256(transaction_id.encode("utf-8")).hexdigest()[:24]

            if len(relative.parts) == 1:
                # Supporting a direct child keeps the service independently
                # testable. Production composition uses an exclusive upload
                # directory, handled by the branch below.
                cleanup_root = upload_root / f".ingestion-{digest}"
                if cleanup_root.exists() or cleanup_root.is_symlink():
                    raise FileExistsError("ingestion work area already exists")
                archive_outside_cleanup_root = True
            else:
                cleanup_root = upload_root / relative.parts[0]
                mode = os.lstat(cleanup_root).st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    raise OSError("upload staging root is not a real directory")
                archive_outside_cleanup_root = False

            extraction_root = cleanup_root / f".extracted-{digest}"
            spool_root = cleanup_root / f".spool-{digest}"
            if archive in {extraction_root, spool_root}:
                raise ValueError("archive conflicts with ingestion work area")
            return _StagingLayout(
                archive_path=archive,
                extraction_root=extraction_root,
                spool_root=spool_root,
                cleanup_root=cleanup_root,
                archive_outside_cleanup_root=archive_outside_cleanup_root,
            )
        except PublicError:
            raise
        except Exception as exc:
            raise _public_error("zip_extraction_failed") from exc

    @staticmethod
    def _cleanup_staging(layout: _StagingLayout) -> None:
        try:
            if layout.cleanup_root.is_symlink():
                layout.cleanup_root.unlink()
            elif layout.cleanup_root.exists():
                shutil.rmtree(layout.cleanup_root)
            if layout.archive_outside_cleanup_root:
                layout.archive_path.unlink(missing_ok=True)
        except (OSError, RuntimeError) as exc:
            raise _public_error("zip_extraction_failed") from exc

    @staticmethod
    def _normalize_transaction_id(upload_id: UUID | str) -> str:
        if isinstance(upload_id, UUID):
            return str(upload_id)
        if type(upload_id) is str and upload_id and "\0" not in upload_id:
            return upload_id
        raise TypeError("upload_id must be a UUID or non-empty string")


__all__ = [
    "ChunkingService",
    "IngestionService",
    "PdfExtractor",
    "WarningCollector",
    "ZipValidator",
    "discover_pdf_entries",
    "discover_pdfs",
    "make_chunk_id",
    "normalize_display_name",
    "sha256_file",
]
