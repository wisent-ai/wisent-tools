"""Provider-neutral HTTP client for Stado object persistence."""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from .core import (
    CHUNK_BYTES as _CHUNK_BYTES,
    ERROR_DETAIL_LIMIT as _ERROR_DETAIL_LIMIT,
    EXIT_MISSING as _EXIT_MISSING,
    EXIT_OK as _EXIT_OK,
    HTTP_CONFLICT as _HTTP_CONFLICT,
    HTTP_NOT_FOUND as _HTTP_NOT_FOUND,
    HTTP_OK as _HTTP_OK,
    HTTP_REDIRECT as _HTTP_REDIRECT,
    StadoConflict,
    StadoError,
    StadoNotFound,
    content_type as _content_type,
    has_control_characters as _has_control_characters,
    join_uri,
    split_uri,
)

class StadoClient:
    def __init__(self, api_url: str | None = None, token: str | None = None):
        if os.environ.get("WC_JOB_ID", "").strip():
            raise StadoError(
                "machine jobs must use pre-staged inputs and write artifacts under output/"
            )
        raw_url = (api_url or os.environ.get("STADO_API_URL", "")).strip()
        self.token = (token or os.environ.get("STADO_API_TOKEN", "")).strip()
        if not raw_url:
            raise StadoError("STADO_API_URL is required for remote object storage")
        if not self.token:
            raise StadoError("STADO_API_TOKEN is required for remote object storage")
        if _has_control_characters(raw_url) or "\\" in raw_url or "%" in raw_url:
            raise StadoError("STADO_API_URL contains unsafe URL syntax")
        if _has_control_characters(self.token):
            raise StadoError("STADO_API_TOKEN contains control characters")
        parsed = urlsplit(raw_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise StadoError("STADO_API_URL has an invalid port") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise StadoError(
                "STADO_API_URL must be an absolute URL without credentials, query, or fragment"
            )
        if parsed.path not in {"", "/"} and (
            not parsed.path.startswith("/")
            or parsed.path.startswith("//")
            or any(
                segment in {"", ".", ".."}
                for segment in parsed.path[int("1"):].split("/")
            )
        ):
            raise StadoError("STADO_API_URL contains an unsafe base path")
        if parsed.scheme == "http":
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname == "localhost"
            if not loopback:
                raise StadoError(
                    "STADO_API_URL must use HTTPS except for authenticated loopback"
                )
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = port
        self._base_path = "" if parsed.path == "/" else parsed.path

    def _connection(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        return cls(self._host, self._port)

    def _target(self, route: str, **query: str) -> str:
        suffix = f"?{urlencode(query)}" if query else ""
        return f"{self._base_path}{route}{suffix}"

    def _headers(self, content_type: str | None = None, length: int | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        if length is not None:
            headers["Content-Length"] = str(length)
        return headers

    @staticmethod
    def _raise(status: int, reason: str, body: bytes) -> None:
        detail = body.decode("utf-8", errors="replace")[:_ERROR_DETAIL_LIMIT]
        message = f"Stado object request failed ({status} {reason})"
        if detail:
            message = f"{message}: {detail}"
        if status == _HTTP_NOT_FOUND:
            raise StadoNotFound(message, status=status)
        if status == _HTTP_CONFLICT:
            raise StadoConflict(message, status=status)
        raise StadoError(message, status=status)

    def _request(self, method: str, route: str, *, query: dict[str, str] | None = None,
                 body: bytes | None = None, content_type: str | None = None) -> bytes:
        conn = self._connection()
        try:
            conn.request(method, self._target(route, **(query or {})), body=body,
                         headers=self._headers(content_type, len(body) if body is not None else None))
            response = conn.getresponse()
            payload = response.read()
            if response.status < _HTTP_OK or response.status >= _HTTP_REDIRECT:
                self._raise(response.status, response.reason, payload)
            return payload
        finally:
            conn.close()

    def get_bytes(self, uri: str) -> bytes:
        split_uri(uri)
        return self._request("GET", "/api/object", query={"uri": uri})

    def get_text(self, uri: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(uri).decode(encoding)

    def get_json(self, uri: str) -> Any:
        return json.loads(self.get_bytes(uri))

    def get_file(self, uri: str, destination: str | Path) -> Path:
        split_uri(uri)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        conn = self._connection()
        try:
            conn.request("GET", self._target("/api/object", uri=uri), headers=self._headers())
            response = conn.getresponse()
            if response.status < _HTTP_OK or response.status >= _HTTP_REDIRECT:
                self._raise(response.status, response.reason, response.read())
            with open(temporary, "wb") as stream:
                while chunk := response.read(_CHUNK_BYTES):
                    stream.write(chunk)
            os.replace(temporary, destination)
            return destination
        finally:
            conn.close()
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def put_bytes(self, uri: str, payload: bytes, content_type: str = "application/octet-stream",
                  *, if_absent: bool = False) -> None:
        split_uri(uri)
        query = {"uri": uri}
        if if_absent:
            query["if_absent"] = "true"
        self._request("PUT", "/api/object", query=query, body=payload, content_type=content_type)

    def put_json(self, uri: str, value: Any, *, if_absent: bool = False) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.put_bytes(uri, payload, "application/json", if_absent=if_absent)

    def put_file(self, uri: str, source: str | Path, *, content_type: str | None = None,
                 if_absent: bool = False) -> None:
        split_uri(uri)
        source = Path(source)
        size = source.stat().st_size
        query = {"uri": uri}
        if if_absent:
            query["if_absent"] = "true"
        conn = self._connection()
        try:
            conn.putrequest("PUT", self._target("/api/object", **query))
            for name, value in self._headers(content_type or _content_type(source), size).items():
                conn.putheader(name, value)
            conn.endheaders()
            with source.open("rb") as stream:
                while chunk := stream.read(_CHUNK_BYTES):
                    conn.send(chunk)
            response = conn.getresponse()
            payload = response.read()
            if response.status < _HTTP_OK or response.status >= _HTTP_REDIRECT:
                self._raise(response.status, response.reason, payload)
        finally:
            conn.close()

    def delete(self, uri: str) -> None:
        split_uri(uri)
        self._request("DELETE", "/api/object", query={"uri": uri})

    def stat(self, uri: str) -> dict[str, Any] | None:
        split_uri(uri)
        try:
            return json.loads(self._request("GET", "/api/object/stat", query={"uri": uri}))
        except StadoNotFound:
            return None

    def list(self, namespace: str, prefix: str = "") -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/object/list",
                                query={"namespace": namespace, "prefix": prefix})
        value = json.loads(payload)
        objects = value.get("objects") if isinstance(value, dict) else None
        if not isinstance(objects, list):
            raise StadoError("Stado list response has no objects array")
        return objects

    def list_uri(self, root_uri: str) -> list[dict[str, Any]]:
        namespace, prefix = split_uri(root_uri)
        return self.list(namespace, prefix)

    def put_tree(self, source: str | Path, root_uri: str, *, delete_missing: bool = False) -> int:
        source = Path(source)
        if not source.is_dir():
            raise ValueError(f"tree source is not a directory: {source}")
        local: dict[str, Path] = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"refusing to upload symlink: {path}")
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                local[join_uri(root_uri, relative)] = path
        for uri, path in local.items():
            self.put_file(uri, path)
        if delete_missing:
            for item in self.list_uri(root_uri):
                uri = item.get("uri")
                if isinstance(uri, str) and uri not in local:
                    self.delete(uri)
        return len(local)

    def get_prefix(self, root_uri: str, destination: str | Path) -> int:
        namespace, prefix = split_uri(root_uri)
        destination = Path(destination)
        count = _EXIT_OK
        for item in self.list(namespace, prefix):
            key = item.get("key")
            uri = item.get("uri")
            if not isinstance(key, str) or not isinstance(uri, str):
                continue
            if prefix and key != prefix and not key.startswith(prefix.rstrip("/") + "/"):
                continue
            relative = key[len(prefix):].lstrip("/") if prefix else key
            if not relative or any(part in {".", ".."} for part in relative.split("/")):
                continue
            self.get_file(uri, destination / Path(relative))
            count += _EXIT_MISSING
        return count
