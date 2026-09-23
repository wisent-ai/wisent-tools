"""Validated Stado object addresses, errors, and transport constants."""
import mimetypes
import posixpath
from pathlib import Path
from urllib.parse import urlsplit

ERROR_DETAIL_LIMIT = int("500")
HTTP_OK = int("200")
HTTP_REDIRECT = int("300")
HTTP_NOT_FOUND = int("404")
HTTP_CONFLICT = int("409")
CHUNK_BYTES = int("1048576")
EXIT_OK = int("0")
EXIT_MISSING = int("1")


class StadoError(RuntimeError):
    """A Stado object request failed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class StadoNotFound(StadoError):
    """The requested Stado object does not exist."""


class StadoConflict(StadoError):
    """A create-only Stado object already exists."""


def has_control_characters(value: str) -> bool:
    return any(ord(character) < int("32") or ord(character) == int("127") for character in value)


def split_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri or has_control_characters(uri):
        raise ValueError(f"invalid Stado object URI: {uri}")
    if "\\" in uri or "%" in uri:
        raise ValueError(f"unsafe Stado object URI: {uri}")
    parsed = urlsplit(uri)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid Stado object URI: {uri}") from exc
    if (
        parsed.scheme != "stado"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc != parsed.hostname
    ):
        raise ValueError(f"invalid Stado object URI: {uri}")
    if parsed.path and (not parsed.path.startswith("/") or parsed.path.startswith("//")):
        raise ValueError(f"unsafe Stado object URI: {uri}")
    key = parsed.path[int("1"):] if parsed.path else ""
    segments = key.split("/") if key else []
    if any(not part or part in {".", ".."} for part in segments):
        raise ValueError(f"unsafe Stado object URI: {uri}")
    return parsed.hostname, key


def join_uri(base: str, *parts: str) -> str:
    namespace, key = split_uri(base)
    clean = [key] if key else []
    for part in parts:
        value = str(part)
        if (
            not value
            or value.startswith("/")
            or value.endswith("/")
            or "\\" in value
            or "%" in value
            or "?" in value
            or "#" in value
            or has_control_characters(value)
        ):
            raise ValueError("unsafe Stado object key")
        segments = value.split("/")
        if any(not segment or segment in {".", ".."} for segment in segments):
            raise ValueError("unsafe Stado object key")
        clean.extend(segments)
    joined = posixpath.join(*clean) if clean else ""
    result = f"stado://{namespace}/{joined}" if joined else f"stado://{namespace}"
    split_uri(result)
    return result


def content_type(path: Path) -> str:
    guessed, encoding = mimetypes.guess_type(path.name)
    if encoding == "gzip":
        return "application/gzip"
    return guessed or "application/octet-stream"
