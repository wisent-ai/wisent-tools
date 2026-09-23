"""Materialize immutable wisent-tools inputs through the private Stado boundary."""
from __future__ import annotations

import atexit
import argparse
import hashlib
import hmac
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from wisent.stado import StadoClient, StadoConflict, StadoError, join_uri, split_uri
from wisent.input_support.archive import extract_archive, model_root
from wisent.stado_support.core import CHUNK_BYTES as _COPY_CHUNK_BYTES


PRODUCT_NAMESPACE = "wisent-tools"
MODEL_PREFIX = "models/"
DATASET_PREFIX = "datasets/"
EVALUATION_ROOT = "stado://wisent-tools/evaluations"
_HEX_DIGEST_LENGTH = len(hashlib.sha256().hexdigest())


@dataclass(frozen=True)
class MaterializedInput:
    """Verified local representation of one immutable Stado input."""

    uri: str
    sha256: str
    path: Path


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise StadoError(f"{name} is required")
    return value


def _require_digest(value: str, name: str) -> str:
    if (len(value) != _HEX_DIGEST_LENGTH
            or any(character not in "0123456789abcdef" for character in value)):
        raise StadoError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_private_uri(uri: str, prefix: str) -> str:
    """Require an exact product-owned Stado namespace and object category."""
    namespace, key = split_uri(uri)
    normalized = prefix.strip("/") + "/"
    if namespace != PRODUCT_NAMESPACE or not key.startswith(normalized) or key == normalized:
        raise StadoError(
            f"input must be under stado://{PRODUCT_NAMESPACE}/{normalized}"
        )
    return uri


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise StadoError(f"materialized input is not a regular file: {path}")
    actual = _file_sha256(path)
    if not hmac.compare_digest(actual, expected):
        raise StadoError(
            f"materialized input digest mismatch: expected {expected}, got {actual}"
        )


@contextmanager
def _object_file(
    *,
    uri: str,
    digest: str,
    prefix: str,
    staged_path: Path,
    suffix: str,
) -> Iterator[Path]:
    require_private_uri(uri, prefix)
    _require_digest(digest, "input digest")
    machine_job = bool(os.environ.get("WC_JOB_ID", "").strip())
    if machine_job:
        input_root = (Path.cwd() / "inputs").resolve()
        resolved_staged = staged_path.resolve()
        if resolved_staged != input_root and input_root not in resolved_staged.parents:
            raise StadoError("machine input path must stay under inputs/")
    if staged_path.exists():
        _verify(staged_path, digest)
        yield staged_path
        return
    if machine_job:
        raise StadoError(
            f"Stado machine input was not pre-staged at {staged_path}; "
            "jobs never receive object-storage credentials"
        )
    with tempfile.TemporaryDirectory(prefix="wisent-tools-input-") as temporary:
        destination = Path(temporary) / f"object{suffix}"
        StadoClient().get_file(uri, destination)
        _verify(destination, digest)
        yield destination


@contextmanager
def private_object_input(
    uri: str,
    digest: str,
    *,
    prefix: str,
    staged_path: str | Path,
    suffix: str = "",
) -> Iterator[MaterializedInput]:
    """Yield one verified private object, downloaded only outside machine jobs."""
    with _object_file(
        uri=uri,
        digest=digest,
        prefix=prefix,
        staged_path=Path(staged_path),
        suffix=suffix,
    ) as path:
        yield MaterializedInput(uri=uri, sha256=digest, path=path)




def materialize_model_at(destination: str | Path) -> MaterializedInput:
    """Verify and extract the configured immutable model archive."""
    uri = require_private_uri(_required_env("STADO_MODEL_URI"), MODEL_PREFIX)
    digest = _require_digest(_required_env("STADO_MODEL_SHA256"), "STADO_MODEL_SHA256")
    staged = Path(os.environ.get("STADO_MODEL_ARCHIVE_PATH", "inputs/model.tar.gz"))
    destination = Path(destination)
    with _object_file(
        uri=uri,
        digest=digest,
        prefix=MODEL_PREFIX,
        staged_path=staged,
        suffix=".tar.gz",
    ) as archive:
        extract_archive(archive, destination)
    return MaterializedInput(uri=uri, sha256=digest, path=model_root(destination))


@contextmanager
def private_model_input() -> Iterator[MaterializedInput]:
    """Yield a temporary verified local model directory."""
    with tempfile.TemporaryDirectory(prefix="wisent-tools-model-") as temporary:
        yield materialize_model_at(Path(temporary) / "model")


def process_model_input() -> MaterializedInput:
    """Keep a verified temporary model available until this process exits."""
    os.environ["HF_HUB_OFFLINE"] = "true"
    os.environ["HF_DATASETS_OFFLINE"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "true"
    context = private_model_input()
    materialized = context.__enter__()
    atexit.register(context.__exit__, None, None, None)
    return materialized


@contextmanager
def private_dataset_input() -> Iterator[MaterializedInput]:
    """Yield the configured verified private JSONL dataset object."""
    uri = require_private_uri(_required_env("STADO_DATASET_URI"), DATASET_PREFIX)
    digest = _require_digest(_required_env("STADO_DATASET_SHA256"), "STADO_DATASET_SHA256")
    staged = Path(os.environ.get("STADO_DATASET_PATH", "inputs/dataset.jsonl"))
    with _object_file(
        uri=uri,
        digest=digest,
        prefix=DATASET_PREFIX,
        staged_path=staged,
        suffix=".jsonl",
    ) as dataset:
        yield MaterializedInput(uri=uri, sha256=digest, path=dataset)


def read_jsonl_records(path: str | Path) -> list[dict]:
    """Read a canonical object-per-line dataset without provider loaders."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, int("1")):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StadoError(f"invalid dataset JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise StadoError(f"dataset line {line_number} must be a JSON object")
            records.append(value)
    if not records:
        raise StadoError("materialized dataset is empty")
    return records


def load_private_dataset_records() -> tuple[str, list[dict]]:
    """Materialize, verify, and parse the configured private JSONL dataset."""
    with private_dataset_input() as materialized:
        return materialized.uri, read_jsonl_records(materialized.path)


def materialize_dataset_cache_at(destination: str | Path) -> MaterializedInput:
    """Verify and extract the configured offline dataset-cache archive."""
    uri = require_private_uri(_required_env("STADO_DATASET_URI"), DATASET_PREFIX)
    digest = _require_digest(_required_env("STADO_DATASET_SHA256"), "STADO_DATASET_SHA256")
    staged = Path(os.environ.get("STADO_DATASET_ARCHIVE_PATH", "inputs/dataset.tar.gz"))
    destination = Path(destination)
    with _object_file(
        uri=uri,
        digest=digest,
        prefix=DATASET_PREFIX,
        staged_path=staged,
        suffix=".tar.gz",
    ) as archive:
        extract_archive(archive, destination)
    return MaterializedInput(uri=uri, sha256=digest, path=destination)


def publish_private_result(filename: str, value: dict) -> str:
    """Create one immutable evaluation result at the private Stado boundary."""
    evaluation_id = _required_env("STADO_EVALUATION_ID")
    if (
        any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in evaluation_id)
        or evaluation_id in {".", ".."}
    ):
        raise StadoError("STADO_EVALUATION_ID is unsafe")
    if "\\" in filename or Path(filename).name != filename or not filename.endswith(".json"):
        raise StadoError("evaluation result name must be one JSON filename")
    uri = join_uri(EVALUATION_ROOT, evaluation_id, filename)
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if os.environ.get("WC_JOB_ID", "").strip():
        output_root = Path(
            os.environ.get("STADO_JOB_OUTPUT_DIR", str(Path.cwd() / "output"))
        )
        destination = output_root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_bytes()
            if hmac.compare_digest(hashlib.sha256(existing).digest(),
                                   hashlib.sha256(payload).digest()):
                return uri
            raise StadoError(f"immutable machine result conflicts: {destination}")
        with destination.open("xb") as stream:
            stream.write(payload)
        return uri
    client = StadoClient()
    try:
        client.put_bytes(uri, payload, "application/json", if_absent=True)
    except StadoConflict:
        existing = client.get_bytes(uri)
        if not hmac.compare_digest(hashlib.sha256(existing).digest(),
                                   hashlib.sha256(payload).digest()):
            raise StadoError(f"immutable Stado result conflicts: {uri}")
    return uri


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("destination")
    dataset_parser = subparsers.add_parser("dataset-cache")
    dataset_parser.add_argument("destination")
    args = parser.parse_args()
    if args.command == "model":
        print(materialize_model_at(args.destination).path)
        return int("0")
    if args.command == "dataset-cache":
        print(materialize_dataset_cache_at(args.destination).path)
        return int("0")
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_main())
