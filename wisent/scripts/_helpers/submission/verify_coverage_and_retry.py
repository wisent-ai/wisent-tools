"""HF activation coverage verifier + retry orchestrator.

Walks the expected universe per requested model
    benchmark_tags.json keys (top-level tasks)
      x VALIDATED_STRATEGIES (7 canonical strategies)
      x N layers (probed from HF for the first verifiable strategy dir
                  of that model so nothing is hardcoded)

and classifies every (model, task, strategy) tuple against HF
`wisent-ai/activations` as MISSING / OPAQUE (pre-wisent-0.11.30, no
stable_ids metadata) / VERIFIABLE. VERIFIABLE tuples additionally have
their full layer set listed so partial-layer extractions are caught.

Re-submits the targeted wisent-compute jobs for the MISSING + partial
VERIFIABLE subset. OPAQUE strategies are reported but NOT re-dispatched:
extract_and_upload._strategy_shard_state SKIPS them at line 161-169
to avoid _merge_existing_shard clobbering existing rows; they require
migrate_stable_ids.py first.

State on gs://wisent-compute/coverage/verify_state.json carries
per-(model, task) attempt count + last batch_id. After ATTEMPT_CAP
attempts a tuple is marked UNFIXABLE and surfaced in the report bucket
but not re-submitted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    GSM8K_DEFAULT_LIMIT,
    HF_RETRY_BASE_WAIT,
    HF_RETRY_MAX_RETRIES,
    PARALLEL_CLEANUP_WORKERS,
    PRIORITY_LOW,
)
from wisent.scripts.activations.extract_and_upload import VALIDATED_STRATEGIES
from wisent.scripts._helpers.submission.submit_top_level_benchmarks import (
    load_benchmark_names,
)
from wisent_compute.config import BUCKET
from wisent_compute.queue.storage import JobStorage
from wisent_compute.queue.submit import submit_batch

HF_REPO = "wisent-ai/activations"
HF_BASE = "https://huggingface.co"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
STATE_BLOB = "coverage/verify_state.json"
ATTEMPT_CAP = HF_RETRY_MAX_RETRIES
RETRY_BASE = HF_RETRY_BASE_WAIT
DEFAULT_THREADS = PARALLEL_CLEANUP_WORKERS // 2
PROGRESS_LOG_EVERY = PARALLEL_CLEANUP_WORKERS * 25
SAFETENSORS_HDR_BYTES = 8
SAFETENSORS_HDR_MAX = HF_RETRY_BASE_WAIT ** 20  # ~1M sanity bound
MISSING, OPAQUE, VERIFIABLE = "missing", "opaque", "verifiable"
ERR_PREVIEW_CHARS = 200
DRYRUN_GAPS_PREVIEW = PARALLEL_CLEANUP_WORKERS * 6
DRYRUN_OPAQUE_PREVIEW = PARALLEL_CLEANUP_WORKERS * 2 + 4


def _model_safe(model: str) -> str:
    return model.replace("/", "__")


def _hf_request(url: str, range_header: str | None = None) -> bytes:
    """Authenticated GET with 429 retry-with-backoff. Raises on 4xx/5xx."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(ATTEMPT_CAP + 3):
        try:
            with urllib.request.urlopen(req) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code == 429:
                time.sleep(RETRY_BASE ** attempt)
                continue
            raise
    raise RuntimeError(f"HF 429 retry-cap exceeded for {url}")


def _fetch_metadata(model: str, task: str, strategy: str) -> dict | None:
    safe = _model_safe(model)
    url = (f"{HF_BASE}/datasets/{HF_REPO}/resolve/main/"
           f"activations/{safe}/{task}/{strategy}/layer_1.safetensors")
    try:
        buf = _hf_request(url, range_header=f"bytes=0-{SAFETENSORS_HDR_BYTES - 1}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if len(buf) < SAFETENSORS_HDR_BYTES:
        return None
    n = int.from_bytes(buf, "little")
    if n <= 0 or n > SAFETENSORS_HDR_MAX:
        return None
    hdr = _hf_request(url, range_header=f"bytes={SAFETENSORS_HDR_BYTES}-{SAFETENSORS_HDR_BYTES - 1 + n}")
    return json.loads(hdr).get("__metadata__") or {}


def _hf_tree(url: str) -> tuple[list, str | None]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(ATTEMPT_CAP + 3):
        try:
            with urllib.request.urlopen(req) as r:
                entries = json.loads(r.read())
                link_hdr = r.headers.get("Link") or r.headers.get("link") or ""
                break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return [], None
            if e.code == 429:
                time.sleep(RETRY_BASE ** attempt); continue
            raise
    else:
        raise RuntimeError(f"HF tree 429 retry-cap exceeded for {url}")
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_hdr)
    nxt = (m.group(1) if m and m.group(1).startswith("http")
           else HF_BASE + m.group(1) if m else None)
    return entries, nxt


def _list_strategy_layers(model: str, task: str, strategy: str) -> set[int]:
    safe = _model_safe(model)
    url = (f"{HF_BASE}/api/datasets/{HF_REPO}/tree/main/"
           f"activations/{safe}/{task}/{strategy}?recursive=false&expand=false")
    out: set[int] = set()
    while url:
        entries, url = _hf_tree(url)
        for e in entries:
            m = re.match(r"layer_(\d+)\.safetensors$", Path(e["path"]).name)
            if m:
                out.add(int(m.group(1)))
    return out


def _expected_layer_count(model: str, tasks: list[str]) -> int:
    """First verifiable strategy dir of this model -> max layer index."""
    for task in tasks:
        for strategy in VALIDATED_STRATEGIES:
            meta = _fetch_metadata(model, task, strategy)
            if meta is None or "stable_ids" not in meta:
                continue
            layers = _list_strategy_layers(model, task, strategy)
            if layers:
                return max(layers)
    raise RuntimeError(
        f"No verifiable shard for {model} -- run one extraction first."
    )


def _classify_shard(model: str, task: str, strategy: str, n_layers: int) -> dict:
    meta = _fetch_metadata(model, task, strategy)
    if meta is None:
        return {"state": MISSING, "n_pairs": 0,
                "layers_present": 0, "layers_expected": n_layers}
    raw = meta.get("stable_ids", "")
    if not raw:
        return {"state": OPAQUE, "n_pairs": 0,
                "layers_present": 1, "layers_expected": n_layers}
    try:
        sids = json.loads(raw)
    except Exception:
        return {"state": OPAQUE, "n_pairs": 0,
                "layers_present": 1, "layers_expected": n_layers}
    layers = _list_strategy_layers(model, task, strategy)
    return {"state": VERIFIABLE, "n_pairs": len(sids),
            "layers_present": len(layers), "layers_expected": n_layers}


def walk_coverage(model: str, tasks: list[str], threads: int) -> dict:
    n_layers = _expected_layer_count(model, tasks)
    out: dict[tuple[str, str], dict] = {}
    units = [(t, s) for t in tasks for s in VALIDATED_STRATEGIES]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(_classify_shard, model, t, s, n_layers): (t, s) for t, s in units}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % PROGRESS_LOG_EVERY == 0:
                sys.stderr.write(f"  [{model}] {done}/{len(units)}\n"); sys.stderr.flush()
            out[futs[f]] = f.result()
    return {"n_layers": n_layers,
            "shards": {f"{t}::{s}": r for (t, s), r in out.items()}}


def compute_gaps(model: str, coverage: dict, state: dict) -> tuple[list, list, list]:
    gaps: dict[str, list[str]] = {}
    opaque: list[tuple[str, str]] = []
    unfix: list[tuple[str, str, str]] = []
    for key, rec in coverage["shards"].items():
        task, strategy = key.split("::", 1)
        if rec["state"] == OPAQUE:
            opaque.append((task, strategy)); continue
        partial = (rec["state"] == VERIFIABLE
                   and rec["layers_present"] < rec["layers_expected"])
        if rec["state"] == MISSING or partial:
            attempts = state.get(model, {}).get(task, {}).get("attempts", 0)
            if attempts >= ATTEMPT_CAP:
                last_err = state[model][task].get("last_error", "")
                unfix.append((task, strategy, last_err[:ERR_PREVIEW_CHARS])); continue
            gaps.setdefault(task, []).append(strategy)
    return ([(t, sorted(ss)) for t, ss in sorted(gaps.items())],
            sorted(opaque), sorted(unfix))


def _state_load(store: JobStorage) -> dict:
    txt = store._download_text(STATE_BLOB)
    return json.loads(txt) if txt else {}


def _state_save(store: JobStorage, state: dict) -> None:
    store._upload_text(STATE_BLOB, json.dumps(state, indent=2, sort_keys=True))


def submit_gaps(model: str, gaps: list, args, state: dict, store: JobStorage) -> int:
    if not gaps:
        return 0
    commands: list[str] = []
    for task, strategies in gaps:
        commands.append(
            f"python3 -m wisent.scripts.activations.extract_and_upload "
            f"--task {task} --model '{model}' --device cuda "
            f"--layers all --strategies {' '.join(strategies)} "
            f"--component {args.component} --limit {args.limit}"
        )
    batch_id = f"verify-retry-{int(time.time())}-{_model_safe(model)}"
    submitted = submit_batch(
        commands, provider="gcp", batch_id=batch_id, bucket=BUCKET,
        preemptible=False, priority=args.priority,
    )
    for task, strategies in gaps:
        slot = state.setdefault(model, {}).setdefault(task, {})
        slot["attempts"] = slot.get("attempts", 0) + 1
        slot["last_batch_id"] = batch_id
        slot["last_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        slot["last_strategies"] = strategies
    _state_save(store, state)
    sys.stderr.write(f"[{model}] submitted {submitted}/{len(commands)} jobs in batch {batch_id}\n")
    return submitted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True, help="comma-separated HF model ids")
    p.add_argument("--limit", type=int, default=GSM8K_DEFAULT_LIMIT)
    p.add_argument("--component", default="residual_stream")
    p.add_argument("--priority", type=int, default=PRIORITY_LOW)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--execute", action="store_true",
                   help="actually submit gap jobs; default is dry-run")
    args = p.parse_args()
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN env var is required")
    tasks = load_benchmark_names()
    store = JobStorage(BUCKET)
    state = _state_load(store)
    total_submitted = 0
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        sys.stderr.write(f"[{model}] walking coverage over "
                         f"{len(tasks)} tasks x {len(VALIDATED_STRATEGIES)} strategies\n")
        coverage = walk_coverage(model, tasks, args.threads)
        gaps, opaque, unfix = compute_gaps(model, coverage, state)
        print(json.dumps({
            "model": model, "n_layers": coverage["n_layers"],
            "missing_or_partial_strategy_count": sum(len(s) for _, s in gaps),
            "opaque_strategy_count": len(opaque),
            "unfixable_strategy_count": len(unfix),
            "gaps_sample": gaps[:DRYRUN_GAPS_PREVIEW],
            "opaque_sample": opaque[:DRYRUN_OPAQUE_PREVIEW],
            "unfixable_sample": unfix[:DRYRUN_OPAQUE_PREVIEW],
        }, indent=2))
        if args.execute:
            total_submitted += submit_gaps(model, gaps, args, state, store)
    sys.stderr.write(f"total submitted: {total_submitted}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
