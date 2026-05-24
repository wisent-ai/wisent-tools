#!/usr/bin/env python3
"""Backfill stable_ids metadata into older HF activation shards.

Pre-wisent-0.11.30 shards have pair_ids but no stable_ids. The
coverage-aware extraction in wisent-tools 0.1.26 cannot grow such
shards because it has no way to know which pairs the shard already
covers. Without this migration, those shards stay frozen.

Strategy:
  1. List every (model, task, strategy) under activations/ on HF.
  2. For each, read layer_1.safetensors metadata. Skip if stable_ids
     already present.
  3. Pull pair_texts/{task}.json (the source of truth used at
     extraction time) and compute stable_id for each row from the
     canonical sha256(prompt + pos + neg)[:16] formula.
  4. For every layer's shard under that (model, task, strategy),
     re-upload with stable_ids added to metadata. Tensors are
     unchanged so the data on disk is identical except for the
     metadata header.

Run on a host that has the wisent-ai/activations write token and
gcloud ADC configured. Safe to interrupt: a shard that already has
stable_ids is left alone, so re-running is idempotent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

HF_REPO_ID = "wisent-ai/activations"
HF_REPO_TYPE = "dataset"


def _stable_id_for_pair(pair: dict) -> str:
    """Canonical stable_id formula matching wisent/activation_cache.py."""
    prompt = pair.get("prompt", "") or ""
    pos_text = (pair.get("positive_response", {}).get("text") or
                pair.get("positive_response", {}).get("model_response") or "")
    neg_text = (pair.get("negative_response", {}).get("text") or
                pair.get("negative_response", {}).get("model_response") or "")
    return hashlib.sha256(
        (prompt + "\x1f" + pos_text + "\x1f" + neg_text).encode("utf-8")
    ).hexdigest()[:16]


def _load_pair_texts(api, task: str, work_dir: Path) -> dict | None:
    """Download pair_texts/{task}.json and return {pair_id: pair_dict}.

    Returns None when the pair_texts file is absent — without it we
    cannot recompute stable_ids and the shard must be left opaque."""
    from huggingface_hub import hf_hub_download
    target = f"pair_texts/{task}.json"
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=target,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        print(f"  [{task}] pair_texts download failed: {exc}", flush=True)
        return None
    try:
        with open(local) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [{task}] pair_texts parse failed: {exc}", flush=True)
        return None
    if isinstance(data, dict) and "pairs" in data:
        pairs = data["pairs"]
        if isinstance(pairs, list):
            return {idx: p for idx, p in enumerate(pairs)}
        if isinstance(pairs, dict):
            return {int(k): v for k, v in pairs.items()}
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    if isinstance(data, list):
        return {idx: p for idx, p in enumerate(data)}
    print(f"  [{task}] pair_texts has unexpected shape", flush=True)
    return None


def _list_shards(api, model_dir: str) -> Iterable[tuple[str, str, str]]:
    """Yield (task, strategy_path, layer_filename) for each shard under
    activations/{model_dir}/..."""
    from huggingface_hub import HfApi
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    prefix = f"activations/{model_dir}/"
    for fpath in files:
        if not fpath.startswith(prefix):
            continue
        if not fpath.endswith(".safetensors"):
            continue
        rest = fpath[len(prefix):]
        parts = rest.split("/")
        if len(parts) < 3:
            continue
        task = parts[0]
        layer_fname = parts[-1]
        strategy_path = "/".join(parts[1:-1])
        yield task, strategy_path, layer_fname


def _stage_shard(api, model_dir: str, task: str, strategy_path: str,
                 layer_fname: str, pair_texts_cache: dict,
                 staging_root: Path, dry_run: bool) -> str:
    """Stage a single shard for batched flush. Returns one of:
      'skipped_has_stable_ids'
      'skipped_no_pair_texts'
      'skipped_no_pair_ids'
      'staged'
      'error'
    The shard's rewritten bytes land at staging_root/<hf_path>; caller
    flushes the entire staging dir as ONE HF commit to stay under the
    128 commits/hour cap.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    hf_path = f"activations/{model_dir}/{task}/{strategy_path}/{layer_fname}"
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=hf_path,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        print(f"  {hf_path}: download failed: {exc}", flush=True)
        return "error"

    try:
        with safe_open(local, framework="pt") as so:
            meta = so.metadata() or {}
    except Exception as exc:
        print(f"  {hf_path}: metadata read failed: {exc}", flush=True)
        return "error"

    if meta.get("stable_ids"):
        return "skipped_has_stable_ids"

    pair_ids_raw = meta.get("pair_ids", "")
    if not pair_ids_raw:
        print(f"  {hf_path}: no pair_ids in metadata; cannot migrate", flush=True)
        return "skipped_no_pair_ids"
    try:
        pair_ids = json.loads(pair_ids_raw)
    except Exception:
        print(f"  {hf_path}: pair_ids parse failed", flush=True)
        return "error"

    if task not in pair_texts_cache:
        pair_texts_cache[task] = _load_pair_texts(api, task,
                                                  Path(tempfile.gettempdir()))
    pair_texts = pair_texts_cache.get(task)
    if pair_texts is None:
        return "skipped_no_pair_texts"

    stable_ids = []
    for pid in pair_ids:
        pair = pair_texts.get(int(pid)) if not isinstance(pid, int) else pair_texts.get(pid)
        if pair is None:
            print(f"  {hf_path}: pair_id {pid} missing from pair_texts; abort",
                  flush=True)
            return "error"
        stable_ids.append(_stable_id_for_pair(pair))

    new_meta = dict(meta)
    new_meta["stable_ids"] = json.dumps(stable_ids)

    if dry_run:
        print(f"  [DRY] {hf_path}: would add {len(stable_ids)} stable_ids",
              flush=True)
        return "staged"

    try:
        tensors = load_file(local)
    except Exception as exc:
        print(f"  {hf_path}: tensor load failed: {exc}", flush=True)
        return "error"

    out_path = staging_root / hf_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(tensors, str(out_path), metadata=new_meta)
    except Exception as exc:
        print(f"  {hf_path}: stage save failed: {exc}", flush=True)
        return "error"
    return "staged"


def _flush_batch(api, staging_root: Path, batch_label: str) -> bool:
    """Upload the entire staging directory as ONE HF commit."""
    import shutil
    if not staging_root.is_dir() or not any(staging_root.rglob("*.safetensors")):
        return True
    print(f"  flushing batch {batch_label} from {staging_root}...", flush=True)
    try:
        api.upload_folder(
            folder_path=str(staging_root),
            path_in_repo=".",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"  batch {batch_label}: flushed", flush=True)
        shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as exc:
        print(f"  batch {batch_label}: upload_folder failed: {exc}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True,
        help='HF directory under activations/, e.g. "meta-llama__Llama-3.2-1B-Instruct"',
    )
    parser.add_argument("--task-filter", default="",
                        help="Only process tasks containing this substring.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without uploading.")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN required to upload migrated shards.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    counters = {
        "staged": 0,
        "skipped_has_stable_ids": 0,
        "skipped_no_pair_texts": 0,
        "skipped_no_pair_ids": 0,
        "error": 0,
    }
    pair_texts_cache: dict = {}

    BATCH_FLUSH_EVERY = 500
    staging_root = Path(tempfile.mkdtemp(prefix="wisent_migrate_"))
    staged_since_flush = 0
    batch_idx = 0

    for task, strategy_path, layer_fname in _list_shards(api, args.model_dir):
        if args.task_filter and args.task_filter not in task:
            continue
        result = _stage_shard(
            api, args.model_dir, task, strategy_path, layer_fname,
            pair_texts_cache, staging_root, dry_run=args.dry_run,
        )
        counters[result] = counters.get(result, 0) + 1
        if result == "staged" and not args.dry_run:
            staged_since_flush += 1
            if staged_since_flush >= BATCH_FLUSH_EVERY:
                batch_idx += 1
                ok = _flush_batch(api, staging_root, f"#{batch_idx}")
                if not ok:
                    counters["error"] += 1
                staged_since_flush = 0

    if staged_since_flush > 0 and not args.dry_run:
        batch_idx += 1
        _flush_batch(api, staging_root, f"#{batch_idx} (final)")

    import shutil
    shutil.rmtree(staging_root, ignore_errors=True)

    print("\n=== migration summary ===", flush=True)
    for k, v in counters.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  batches uploaded: {batch_idx}", flush=True)
    return 0 if counters["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
