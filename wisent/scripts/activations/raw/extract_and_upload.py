"""Build immutable packed raw-activation v2 artifacts.

The writer deliberately has no upload side effects.  It stages every tensor and
route manifest, validates their byte hashes, and publishes each completion
marker with a separate final rename.  ``upload_worker`` is the only publisher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

class ArtifactScheme(str, Enum):
    RELATIVE = ""
    LOCAL = "local"
    HF = "hf"
    GCS = "gs"


RAW_PREFIX = "raw_activations_v2"
RAW_STRATEGIES = ("chat_last", "chat_mean", "chat_first", "chat_max_norm", "chat_weighted", "mc_balanced", "role_play")
REF_KEYS = frozenset(("uri", "generation", "size", "sha256"))
METADATA_KEYS = (
    "pair_ids", "stable_ids", "positive_lengths", "negative_lengths",
    "positive_prompt_lengths", "negative_prompt_lengths",
    "positive_answer_onsets", "negative_answer_onsets",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_decimal(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{field} must be a canonical positive decimal string")
    if value.startswith("0") or int(value) <= 0:
        raise ValueError(f"{field} must be a canonical positive decimal string")
    return value


def _sha(value: object, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _revision(value: object, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{field} must be an exact 40-character revision")
    return value


def _validate_ref(ref: object, field: str, *, generation: bool = True) -> dict:
    if not isinstance(ref, dict):
        raise ValueError(f"{field} must be an ArtifactRef object")
    expected = REF_KEYS if generation else REF_KEYS - {"generation"}
    if set(ref) != expected:
        raise ValueError(f"{field} must have exactly {sorted(expected)}")
    if not isinstance(ref.get("uri"), str) or not ref["uri"]:
        raise ValueError(f"{field}.uri must be non-empty")
    if generation and (not isinstance(ref.get("generation"), str) or not ref["generation"]):
        raise ValueError(f"{field}.generation must be non-empty")
    _positive_decimal(ref.get("size"), f"{field}.size")
    _sha(ref.get("sha256"), f"{field}.sha256")
    return dict(ref)


def validate_target_manifest(manifest: object) -> dict:
    """Validate the v2 fields which bind raw data; reject partial metadata."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("target manifest schema_version must be 2")
    for name in ("protocol", "target", "revisions", "activation", "support",
                 "evaluation", "calibration", "execution", "manifest_sha256"):
        if name not in manifest:
            raise ValueError(f"target manifest is missing {name}")
    claimed = _sha(manifest["manifest_sha256"], "manifest_sha256")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if _canonical_sha256(unsigned) != claimed:
        raise ValueError("target manifest canonical hash mismatch")
    target = manifest["target"]
    required_target = ("target_id", "result_id", "model_name", "model_slug",
                       "benchmark", "expected_pairs", "result_prefix")
    if not isinstance(target, dict) or any(not target.get(k) for k in required_target):
        raise ValueError("target manifest has incomplete target identity")
    if not isinstance(target["benchmark"], str):
        raise ValueError("target.benchmark must be a string")
    model_slug = target["model_slug"]
    if (not isinstance(model_slug, str) or model_slug in {".", ".."}
            or any(not (c.isalnum() or c in "._-") for c in model_slug)):
        raise ValueError("target.model_slug must be one safe path segment")
    revisions = manifest["revisions"]
    for key in ("activation_revision", "model_revision", "tokenizer_revision"):
        _revision(revisions.get(key), f"revisions.{key}")
    support = manifest["support"]
    splits = support.get("splits") if isinstance(support, dict) else None
    if not isinstance(splits, dict) or not splits:
        raise ValueError("target manifest support.splits must be non-empty")
    rows: list[dict] = []
    for split, split_rows in splits.items():
        if not isinstance(split, str) or not split or not isinstance(split_rows, list):
            raise ValueError("support splits must map names to row arrays")
        for row in split_rows:
            if not isinstance(row, dict) or set(row) != {"pair_id", "stable_id"}:
                raise ValueError("support rows must contain exactly pair_id and stable_id")
            if isinstance(row["pair_id"], bool) or not isinstance(row["pair_id"], (str, int)):
                raise ValueError("support pair_id must be an explicit string or integer")
            if not isinstance(row["stable_id"], str) or not row["stable_id"]:
                raise ValueError("support stable_id must be non-empty")
            rows.append({"pair_id": row["pair_id"], "stable_id": row["stable_id"], "split": split})
    if len(rows) != support.get("pair_count") or len(rows) != target["expected_pairs"]:
        raise ValueError("target support count does not match expected_pairs")
    identities = [(type(r["pair_id"]).__name__, str(r["pair_id"]), r["stable_id"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("target support contains duplicate identities")
    return manifest


def _get(pair: object, name: str) -> Any:
    if isinstance(pair, Mapping):
        if name in pair:
            return pair[name]
    elif hasattr(pair, name):
        return getattr(pair, name)
    raise ValueError(f"RawPairData v2 is missing {name}")


def _layer_tensor(pair: object, polarity: str, layer: int):
    states = _get(pair, f"{polarity}_hidden_states")
    for key in (str(layer), layer):
        if key in states:
            tensor = states[key]
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                raise ValueError(f"{polarity} layer {layer} must be [tokens, hidden]")
            return tensor.detach().cpu().contiguous()
    raise ValueError(f"RawPairData v2 is missing {polarity} layer {layer}")


def _token_ids(pair: object, polarity: str):
    import torch
    ids = _get(pair, f"{polarity}_input_ids")
    attention = _get(pair, f"{polarity}_attention_mask")
    if getattr(ids, "ndim", None) == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if getattr(attention, "ndim", None) == 2 and attention.shape[0] == 1:
        attention = attention[0]
    if getattr(ids, "ndim", None) != 1 or getattr(attention, "ndim", None) != 1:
        raise ValueError(f"{polarity} input ids and attention mask must be one-dimensional")
    effective = _get(pair, f"{polarity}_effective_length")
    if isinstance(effective, bool) or not isinstance(effective, int) or effective <= 0:
        raise ValueError(f"{polarity}_effective_length must be positive")
    if ids.shape[0] < effective or attention.shape[0] < effective:
        raise ValueError(f"{polarity}_effective_length exceeds final encoding")
    attention = attention[:effective].detach().cpu().to(dtype=torch.int64)
    if not torch.all(attention == 1):
        raise ValueError(f"{polarity} effective attention mask must be all-one")
    return ids[:effective].detach().cpu().to(dtype=torch.int64).contiguous()

def _effective_hidden(pair: object, polarity: str, layer: int):
    tensor = _layer_tensor(pair, polarity, layer)
    effective = _get(pair, f"{polarity}_effective_length")
    if tensor.shape[0] < effective:
        raise ValueError(f"{polarity} hidden states are shorter than effective_length")
    return tensor[:effective].contiguous()


def _support_rows(manifest: dict) -> list[dict]:
    return [
        {"pair_id": row["pair_id"], "stable_id": row["stable_id"], "split": split}
        for split, rows in manifest["support"]["splits"].items() for row in rows
    ]


def _pair_index(raw_pairs: Iterable[object]) -> dict[tuple[str, str], object]:
    result = {}
    for pair in raw_pairs:
        pair_id, stable_id = _get(pair, "pair_id"), _get(pair, "stable_id")
        key = (f"{type(pair_id).__name__}:{pair_id}", stable_id)
        if key in result:
            raise ValueError(f"duplicate RawPairData identity {key}")
        result[key] = pair
    return result


def _concat_sequences(sequences: Sequence[Any]):
    import torch
    trailing = tuple(sequences[0].shape[1:])
    if any(tuple(t.shape[1:]) != trailing for t in sequences):
        raise ValueError("packed sequence trailing dimensions differ")
    packed = torch.cat(tuple(sequences), dim=0).contiguous()
    return packed, torch.ones((packed.shape[0],), dtype=torch.int64)


def _save_safetensors_canonical(path: Path, tensors: dict, metadata: dict) -> None:
    from safetensors.torch import save
    encoded = save({key: tensors[key] for key in sorted(tensors)},
                   metadata={key: metadata[key] for key in sorted(metadata)})
    header_length = int.from_bytes(encoded[:8], "little")
    header = json.loads(encoded[8:8 + header_length])
    canonical = json.dumps(header, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    canonical += b" " * ((8 - len(canonical) % 8) % 8)
    path.write_bytes(len(canonical).to_bytes(8, "little") + canonical
                     + encoded[8 + header_length:])


def _validate_artifact_base_uri(value: str) -> str:
    from urllib.parse import unquote, urlsplit
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact_base_uri must be a safe URI or relative prefix")
    parsed = urlsplit(value)
    try:
        scheme = ArtifactScheme(parsed.scheme)
    except ValueError as exc:
        raise ValueError("artifact_base_uri has an unsupported URI form") from exc
    if len(parsed[3]) != 0 or len(parsed[4]) != 0:
        raise ValueError("artifact_base_uri has an unsupported URI form")
    if scheme is ArtifactScheme.RELATIVE and (value.startswith("/") or value.startswith("~")):
        raise ValueError("artifact_base_uri must not be absolute")
    if any(part in {".", ".."} for part in unquote(parsed.path).split("/")):
        raise ValueError("artifact_base_uri contains path traversal")
    if scheme is not ArtifactScheme.RELATIVE and not parsed.netloc:
        raise ValueError("artifact_base_uri URI must have an authority")
    return value.rstrip("/")


def _artifact_ref(path: Path, uri: str) -> dict:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"empty artifact: {path}")
    return {"uri": uri, "sha256": _file_sha256(path), "size": str(size)}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_bytes(payload)
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise FileExistsError(f"immutable staged object conflict: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(content)
    try:
        os.link(temp, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable staged object conflict: {path}")
    finally:
        temp.unlink(missing_ok=True)

def _install_file_create_only(temp: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == temp.read_bytes():
            temp.unlink()
            return
        temp.unlink(missing_ok=True)
        raise FileExistsError(f"immutable staged object conflict: {destination}")
    try:
        os.link(temp, destination)
    except FileExistsError:
        if destination.read_bytes() != temp.read_bytes():
            raise FileExistsError(f"immutable staged object conflict: {destination}")
    finally:
        temp.unlink(missing_ok=True)


def _with_manifest_hash(payload: dict) -> dict:
    result = dict(payload)
    result["manifest_sha256"] = _canonical_sha256(result)
    return result


def write_raw_activations_v2(
    raw_pairs: Iterable[object], target_manifest: dict, target_manifest_ref: dict,
    staging_dir: str | Path, *, strategy: str, layer_count: int,
    activation_revision: str, model_revision: str, tokenizer_revision: str,
    prefix: str = RAW_PREFIX, artifact_base_uri: str = "local://raw_activations_v2",
) -> list[Path]:
    """Pack one route per layer and return completion-marker paths.

    The ordered join is solely ``(pair_id, stable_id)`` from the supplied target
    support.  No positional or benchmark-derived identity is accepted.
    """
    manifest = validate_target_manifest(target_manifest)
    target_ref = _validate_ref(target_manifest_ref, "target_manifest_ref")
    manifest_bytes = _canonical_bytes(manifest)
    if target_ref["size"] != str(len(manifest_bytes)) or target_ref["sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("target_manifest_ref does not bind the canonical TargetManifest bytes")
    if strategy not in RAW_STRATEGIES:
        raise ValueError(f"unsupported raw strategy: {strategy}")
    if prefix != RAW_PREFIX:
        raise ValueError(f"v2 writes require the clean {RAW_PREFIX!r} prefix")
    artifact_base_uri = _validate_artifact_base_uri(artifact_base_uri)
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("layer_count must be positive")
    supplied_revisions = {
        "activation": _revision(activation_revision, "activation_revision"),
        "model": _revision(model_revision, "model_revision"),
        "tokenizer": _revision(tokenizer_revision, "tokenizer_revision"),
    }
    manifest_revision_keys = {"activation": "activation_revision", "model": "model_revision", "tokenizer": "tokenizer_revision"}
    if any(manifest["revisions"].get(manifest_revision_keys[k]) != v for k, v in supplied_revisions.items()):
        raise ValueError("explicit revisions do not exactly match TargetManifest v2")
    rows = _support_rows(manifest)
    indexed = _pair_index(raw_pairs)
    ordered = []
    for row in rows:
        key = (f"{type(row['pair_id']).__name__}:{row['pair_id']}", row["stable_id"])
        if key not in indexed:
            raise ValueError(f"RawPairData support is missing explicit identity {key}")
        ordered.append(indexed.pop(key))
    if indexed:
        raise ValueError("RawPairData contains identities outside target support")

    target = manifest["target"]
    target_hash = hashlib.sha256(target["target_id"].encode()).hexdigest()
    relative_base = Path(prefix) / target["model_slug"] / target_hash / strategy
    root = Path(staging_dir)
    prefix_root = (root / prefix).resolve()
    resolved_base = (root / relative_base).resolve()
    if resolved_base != prefix_root and prefix_root not in resolved_base.parents:
        raise ValueError("resolved raw activation route escapes the v2 staging prefix")
    completion_paths: list[Path] = []
    pending_markers: list[tuple[Path, dict]] = []
    routes = manifest["activation"].get("routes")
    if not isinstance(routes, list):
        raise ValueError("TargetManifest activation.routes must be an array")
    route_by_layer = {}
    for source_route in routes:
        if isinstance(source_route, dict) and source_route.get("strategy") == strategy:
            source_layer = source_route.get("layer")
            if isinstance(source_layer, bool) or not isinstance(source_layer, int) or source_layer < 1:
                raise ValueError("TargetManifest route layers must be 1-based integers")
            if source_layer in route_by_layer:
                raise ValueError(f"duplicate TargetManifest route for {strategy} layer {source_layer}")
            route_by_layer[source_layer] = source_route
    if set(route_by_layer) != set(range(1, layer_count + 1)):
        raise ValueError("TargetManifest does not contain the exact requested strategy/layer routes")
    for layer in range(1, layer_count + 1):
        positive = [_effective_hidden(pair, "pos", layer) for pair in ordered]
        negative = [_effective_hidden(pair, "neg", layer) for pair in ordered]
        pos_ids = [_token_ids(pair, "pos") for pair in ordered]
        neg_ids = [_token_ids(pair, "neg") for pair in ordered]
        if any(t.shape[0] != ids.shape[0] for t, ids in zip(positive, pos_ids)):
            raise ValueError("positive token ids and hidden states have different lengths")
        if any(t.shape[0] != ids.shape[0] for t, ids in zip(negative, neg_ids)):
            raise ValueError("negative token ids and hidden states have different lengths")
        pos_packed, pos_mask = _concat_sequences(positive)
        neg_packed, neg_mask = _concat_sequences(negative)
        pos_id_packed, pos_id_mask = _concat_sequences(pos_ids)
        neg_id_packed, neg_id_mask = _concat_sequences(neg_ids)
        if not __import__("torch").equal(pos_mask, pos_id_mask) or not __import__("torch").equal(neg_mask, neg_id_mask):
            raise ValueError("token validity masks disagree")
        positive_lengths = [int(t.shape[0]) for t in positive]
        negative_lengths = [int(t.shape[0]) for t in negative]
        pos_prompt = [int(_get(p, "pos_answer_onset")) for p in ordered]
        neg_prompt = [int(_get(p, "neg_answer_onset")) for p in ordered]
        pos_onsets = [int(_get(p, "pos_answer_onset")) for p in ordered]
        neg_onsets = [int(_get(p, "neg_answer_onset")) for p in ordered]
        for name, values, lengths in (
            ("positive_prompt_lengths", pos_prompt, positive_lengths),
            ("negative_prompt_lengths", neg_prompt, negative_lengths),
            ("positive_answer_onsets", pos_onsets, positive_lengths),
            ("negative_answer_onsets", neg_onsets, negative_lengths),
        ):
            if any(v < 0 or v > n for v, n in zip(values, lengths)):
                raise ValueError(f"{name} must lie inside its final encoded sequence")
        metadata_values = {
            "pair_ids": [r["pair_id"] for r in rows],
            "stable_ids": [r["stable_id"] for r in rows],
            "positive_lengths": positive_lengths,
            "negative_lengths": negative_lengths,
            "positive_prompt_lengths": pos_prompt,
            "negative_prompt_lengths": neg_prompt,
            "positive_answer_onsets": pos_onsets,
            "negative_answer_onsets": neg_onsets,
        }
        metadata = {key: json.dumps(metadata_values[key], separators=(",", ":"),
                                    ensure_ascii=False) for key in METADATA_KEYS}
        layer_dir = root / relative_base / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = layer_dir / "activations.safetensors"
        artifact_temp = layer_dir / f".activations.{os.getpid()}.tmp"
        tensors = {
            "positive_activations": pos_packed,
            "negative_activations": neg_packed,
            "positive_token_ids": pos_id_packed,
            "negative_token_ids": neg_id_packed,
            "positive_attention_mask": pos_mask,
            "negative_attention_mask": neg_mask,
        }
        _save_safetensors_canonical(artifact_temp, tensors, metadata)
        _install_file_create_only(artifact_temp, artifact_path)
        base_uri = artifact_base_uri
        rel_layer = (relative_base / f"layer_{layer}").as_posix()
        artifact = _artifact_ref(artifact_path, f"{base_uri}/{rel_layer}/activations.safetensors")
        route = _with_manifest_hash({
            "schema_version": 2, "kind": "raw_activation_route", "complete": True,
            "target": {
                "target_id": target["target_id"], "model": target["model_name"],
                "model_slug": target["model_slug"], "benchmark": target["benchmark"],
                "strategy": strategy, "layer": layer, "layer_count": layer_count,
            },
            "revisions": supplied_revisions, "support": rows, "artifact": artifact,
        })
        route_path = layer_dir / "manifest.json"
        _write_json_atomic(route_path, route)
        route_artifact = _artifact_ref(route_path, f"{base_uri}/{rel_layer}/manifest.json")
        source_route_ref = _validate_ref(route_by_layer[layer].get("completion_ref"), "source_route_ref")
        _validate_ref(route_by_layer[layer].get("proof_ref"), "source_route_proof_ref")
        completion = _with_manifest_hash({
            "schema_version": 2, "complete": True, "kind": "raw_activation_trajectory",
            "target": route["target"], "revisions": supplied_revisions, "support": rows,
            "target_manifest_ref": target_ref, "source_route_ref": source_route_ref,
            "artifact": artifact,
        })
        marker_path = layer_dir / "_complete.json"
        if _file_sha256(artifact_path) != artifact["sha256"] or artifact_path.stat().st_size != int(artifact["size"]):
            raise RuntimeError("staged artifact changed during validation")
        if _file_sha256(route_path) != route_artifact["sha256"] or route_path.stat().st_size != int(route_artifact["size"]):
            raise RuntimeError("staged raw route manifest changed during validation")
        pending_markers.append((marker_path, completion))
    for marker_path, completion in pending_markers:
        _write_json_atomic(marker_path, completion)
        completion_paths.append(marker_path)
    return completion_paths


def _load_json(path: str) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--target-manifest-ref", required=True)
    parser.add_argument("--raw-pairs", required=True,
                        help="torch file containing a list of core RawPairData v2 values")
    parser.add_argument("--strategy", required=True, choices=RAW_STRATEGIES)
    parser.add_argument("--layer-count", required=True, type=int)
    parser.add_argument("--activation-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--prefix", required=True, choices=(RAW_PREFIX,))
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--artifact-base-uri", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", required=True, choices=("dataset", "model", "gcs"))
    args = parser.parse_args()
    import torch
    raw_pairs = torch.load(args.raw_pairs, map_location="cpu", weights_only=False)
    if not isinstance(raw_pairs, (list, tuple)):
        raise ValueError("--raw-pairs must contain a list of RawPairData v2 values")
    markers = write_raw_activations_v2(
        raw_pairs, _load_json(args.target_manifest), _load_json(args.target_manifest_ref),
        Path(args.staging_dir) / "data", strategy=args.strategy, layer_count=args.layer_count,
        activation_revision=args.activation_revision, model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision, prefix=args.prefix,
        artifact_base_uri=args.artifact_base_uri,
    )
    from .upload_worker import handoff
    job_dir = Path(args.staging_dir)
    handoff(job_dir, args.repo_id, RAW_PREFIX, args.repo_type,
            os.environ.get("WC_JOB_ID", ""), v2=True)
    print(f"staged {len(markers)} immutable raw activation routes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
