#!/usr/bin/env python3
"""Extract activations for ONE benchmark + model across 7 strategies; upload to HF.

Per-task pipeline: (1) `wisent generate-pairs-from-task` (HF-cached), (2) for each
strategy, `wisent get-activations` on residual_stream / --layers all, (3) upload
every layer shard to wisent-ai/activations, (4) delete local JSON.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .profile_gpu import GPUProfiler, render_png


VALIDATED_STRATEGIES = [
    "chat_last",
    "chat_mean",
    "chat_first",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
]
# Strategies in the same family share an identical (full_text, answer_text,
# prompt_only) tuple from build_extraction_texts, so they can share a forward
# pass and only differ in the post-hoc per-layer aggregation.
CHAT_FAMILY = {"chat_last", "chat_mean", "chat_first", "chat_max_norm", "chat_weighted"}
DEFAULT_BATCH_FLOOR = 8


def _group_strategies_by_family(strategies: list[str]) -> list[list[str]]:
    """Group strategies that can share a forward pass.

    chat_* are bundled together (one forward pass per pair → 5 strategies).
    mc_balanced and role_play stay solo (their input texts differ).
    """
    chat = [s for s in strategies if s in CHAT_FAMILY]
    other = [s for s in strategies if s not in CHAT_FAMILY]
    groups: list[list[str]] = []
    if chat:
        groups.append(chat)
    for s in other:
        groups.append([s])
    return groups


def _wisent_bin() -> str:
    found = shutil.which("wisent")
    if not found:
        raise SystemExit("wisent CLI not found on PATH")
    return found


def _pair_stable_id(p):
    """Deterministic stable_id for a pair. Use explicit field
    when present, else compute sha256(prompt + pos + neg)[:16]
    matching the writer convention in activation_cache.py."""
    import hashlib
    s = p.get("stable_id", "")
    if s:
        return s
    prompt = p.get("prompt", "") or ""
    pos_text = (p.get("positive_response", {}).get("text") or
                p.get("positive_response", {}).get("model_response") or "")
    neg_text = (p.get("negative_response", {}).get("text") or
                p.get("negative_response", {}).get("model_response") or "")
    return hashlib.sha256(
        (prompt + "\x1f" + pos_text + "\x1f" + neg_text).encode("utf-8")
    ).hexdigest()[:16]


_SHARD_MISSING = "missing"
_SHARD_OPAQUE = "opaque"
_SHARD_VERIFIABLE = "verifiable"


def _strategy_shard_state(model, task, strategy, component):
    """Return (state, stable_ids) for HF layer_1 shard.

    state values:
      _SHARD_MISSING    : no shard at this path. Caller should
                          re-extract everything from scratch.
      _SHARD_OPAQUE     : shard exists but lacks stable_ids
                          metadata (pre-wisent-0.11.30 format).
                          Caller MUST skip the strategy. Re-
                          extracting and merging would clobber
                          the existing rows because
                          _merge_existing_shard drops rows with
                          no matching stable_id from the new
                          batch.
      _SHARD_VERIFIABLE : stable_ids present; coverage check OK.
    """
    import json
    from huggingface_hub import hf_hub_download
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name,
    )
    from safetensors import safe_open
    safe = model_to_safe_name(model)
    hf_strat = f"{strategy}/{component}" if component != "residual_stream" else strategy
    hp = f"activations/{safe}/{task}/{hf_strat}/layer_1.safetensors"
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            filename=hp, token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception:
        return _SHARD_MISSING, set()
    try:
        with safe_open(local, framework="pt") as so:
            meta = so.metadata() or {}
    except Exception:
        return _SHARD_OPAQUE, set()
    raw = meta.get("stable_ids", "")
    if not raw:
        return _SHARD_OPAQUE, set()
    try:
        sids = json.loads(raw)
    except Exception:
        return _SHARD_OPAQUE, set()
    if not sids:
        return _SHARD_OPAQUE, set()
    return _SHARD_VERIFIABLE, set(sids)


def _strategies_pending(pairs_file, model, task, strategies, component):
    """Return (pending, skipped, requested_count, union_missing_sids).

    Decision matrix per strategy:
      shard MISSING    -> pending; union_missing |= requested_sids
      shard OPAQUE     -> SKIPPED (pre-stable_ids format; re-extract
                          would clobber via merge wiping rows that
                          have no matching stable_id in the new
                          batch). Operator must migrate the shard
                          before re-running.
      shard VERIFIABLE -> coverage check via requested - done.
                          empty missing -> skipped.
                          nonempty       -> pending; union_missing |= missing."""
    import json
    with open(pairs_file) as f:
        doc = json.load(f)
    pairs = doc.get("pairs", [])
    requested_sids = {_pair_stable_id(p) for p in pairs}
    pending = []
    skipped = []
    union_missing = set()
    for s in strategies:
        state, done = _strategy_shard_state(model, task, s, component)
        if state == _SHARD_MISSING:
            pending.append(s)
            union_missing |= requested_sids
            continue
        if state == _SHARD_OPAQUE:
            skipped.append(s)
            print(
                f"[{task}] strategy={s} HF shard exists but lacks "
                f"stable_ids metadata (pre-0.11.30 format); SKIPPING "
                f"to preserve existing rows. Migrate the shard to "
                f"add stable_ids before re-running.",
                flush=True,
            )
            continue
        missing = requested_sids - done
        if not missing:
            skipped.append(s)
            print(
                f"[{task}] strategy={s} already covers all "
                f"{len(requested_sids)} requested pairs on HF; skipping",
                flush=True,
            )
        else:
            pending.append(s)
            union_missing |= missing
            print(
                f"[{task}] strategy={s} HF shard has {len(done)} of "
                f"{len(requested_sids)} requested pairs ({len(missing)} "
                f"missing); will re-extract and merge",
                flush=True,
            )
    return pending, skipped, len(requested_sids), union_missing
def _prune_pairs_file_to_sids(pairs_file, keep_sids):
    """Rewrite pairs_file in place keeping only pairs whose
    stable_id is in keep_sids. Returns (kept_count, original_count)."""
    import json
    with open(pairs_file) as f:
        doc = json.load(f)
    pairs = doc.get("pairs", [])
    kept = [p for p in pairs if _pair_stable_id(p) in keep_sids]
    if len(kept) == len(pairs):
        return len(kept), len(pairs)
    doc["pairs"] = kept
    doc["num_pairs"] = len(kept)
    with open(pairs_file, "w") as f:
        json.dump(doc, f)
    return len(kept), len(pairs)
def _count_pairs_by_subtask(pairs_file: Path) -> tuple[int, dict[str, int]]:
    """Read the pairs JSON and count pairs grouped by subtask metadata.

    Returns (total_count, {subtask_name: count}). Best-effort: if the
    pairs file is malformed or missing, returns (0, {}). Subtask name is
    pulled from pair["metadata"]["task"|"subtask"|"dataset"], with a
    "(default)" bucket when none is set (leaf tasks tend not to tag).
    """
    try:
        if not pairs_file.is_file():
            return 0, {}
        data = json.loads(pairs_file.read_text())
        pairs = data if isinstance(data, list) else (data.get("pairs") or [])
        total = len(pairs)
        sub: dict[str, int] = {}
        for p in pairs:
            md = p.get("metadata") if isinstance(p, dict) else None
            md = md if isinstance(md, dict) else {}
            key = md.get("task") or md.get("subtask") or md.get("dataset") or "(default)"
            sub[key] = sub.get(key, 0) + 1
        return total, sub
    except Exception:
        return 0, {}


def _write_manifest(
    *,
    args,
    pairs_initial_count: int,
    pairs_initial_subtask_counts: dict[str, int],
    pairs_post_filter_count: int,
    pairs_post_filter_subtask_counts: dict[str, int],
    pending: list[str],
    failed_strategies: list[tuple[str, str]],
) -> None:
    """Drop a JSON manifest into /home/ubuntu/output where the runner's
    startup script picks it up and uploads to
    gs://<bucket>/status/<JOB_ID>/output/manifest.json. Lets the
    wisent-enterprise dashboard show pairs-extracted-per-subtask without
    parsing command_output.log."""
    out_dir = Path(os.environ.get("WC_OUTPUT_DIR", "/home/ubuntu/output"))
    if not out_dir.is_dir():
        return
    failed_set = {s for s, _ in failed_strategies}
    uploaded = [s for s in pending if s not in failed_set]
    manifest = {
        "job_id": os.environ.get("JOB_ID")
                  or os.environ.get("WC_JOB_ID")
                  or "",
        "model": args.model,
        "task": args.task,
        "limit": args.limit,
        "component": args.component,
        "layers": args.layers,
        "pairs_initial_count": pairs_initial_count,
        "pairs_initial_subtask_counts": pairs_initial_subtask_counts,
        "pairs_post_filter_count": pairs_post_filter_count,
        "pairs_post_filter_subtask_counts": pairs_post_filter_subtask_counts,
        "strategies_pending": list(pending),
        "strategies_uploaded": uploaded,
        "strategies_failed": [{"strategy": s, "error": e[:240]}
                              for s, e in failed_strategies],
        "completed_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "manifest_schema_version": 1,
    }
    target = out_dir / "manifest.json"
    try:
        target.write_text(json.dumps(manifest, indent=2))
        print(f"[{args.task}] manifest written to {target}", flush=True)
    except Exception as exc:
        print(f"[{args.task}] manifest write failed: {exc!r}", flush=True)


def generate_pairs(task: str, out_path: Path, limit: int | None = None) -> None:
    cmd = [_wisent_bin(), "generate-pairs-from-task", task, "--output", str(out_path)]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    print(f"[pairs] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0 or not out_path.is_file():
        raise SystemExit(f"pair_texts generation failed for {task} (rc={result.returncode})")


def run_get_activations(
    *,
    pairs_file: Path,
    output_file: Path,
    model: str,
    strategy: str,
    component: str,
    device: str,
    batch_size: int,
    layers: str,
    cached_model=None,
):
    """Run wisent get-activations for one strategy.

    Prefers the in-process path so all 7 strategies of a task reuse one
    WisentModel load (~7x faster on large models). When the in-process
    import path fails (e.g. older wisent without execute_get_activations),
    runs the wisent CLI as a subprocess instead. Returns cached_model so
    the caller threads the same model through subsequent strategies.
    """
    try:
        from types import SimpleNamespace
        from wisent.core.utils.cli.analysis.analysis.geometry.get_activations import (
            execute_get_activations,
        )
        ns = SimpleNamespace(
            pairs_file=str(pairs_file),
            output=str(output_file),
            model=model,
            device=device,
            layers=layers,
            extraction_strategy=strategy,
            extraction_component=component,
            batch_size=batch_size,
            verbose=False,
            timing=False,
            raw=False,
            cached_model=cached_model,
        )
        execute_get_activations(ns)
        return cached_model
    except Exception as exc:
        print(f"[acts ] in-process call failed ({exc}); using subprocess path", flush=True)
        cmd = [
            _wisent_bin(), "get-activations", str(pairs_file),
            "--output", str(output_file),
            "--model", model,
            "--device", device,
            "--layers", layers,
            "--extraction-strategy", strategy,
            "--extraction-component", component,
            "--batch-size", str(batch_size),
        ]
        print(f"[acts ] {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"get-activations failed (rc={result.returncode}) for {strategy}"
            )
        return None


def run_get_activations_group(
    *,
    pairs_file: Path,
    output_dir: Path,
    model: str,
    strategies: list[str],
    component: str,
    device: str,
    layers: str,
    cached_model=None,
) -> tuple[dict[str, Path], object]:
    """Run a group of strategies sharing one forward pass per pair.

    Falls back to per-strategy run_get_activations if the in-process
    multi-strategy API is not available (older wisent).

    Returns ({strategy: output_path}, cached_model).
    """
    if len(strategies) == 1 or cached_model is None:
        outs: dict[str, Path] = {}
        for s in strategies:
            out_file = output_dir / f"{Path(pairs_file).stem.replace('__pairs','')}__{s}.json"
            cached_model = run_get_activations(
                pairs_file=pairs_file, output_file=out_file,
                model=model, strategy=s, component=component,
                device=device, batch_size=DEFAULT_BATCH_FLOOR,
                layers=layers, cached_model=cached_model,
            )
            outs[s] = out_file
        return outs, cached_model

    try:
        from wisent.core.utils.cli.analysis.analysis.geometry.get_activations import (
            execute_get_activations_multi,
        )
    except Exception as exc:
        print(f"[multi] in-process API unavailable ({exc}); falling back per-strategy", flush=True)
        outs = {}
        for s in strategies:
            out_file = output_dir / f"{Path(pairs_file).stem.replace('__pairs','')}__{s}.json"
            cached_model = run_get_activations(
                pairs_file=pairs_file, output_file=out_file,
                model=model, strategy=s, component=component,
                device=device, batch_size=DEFAULT_BATCH_FLOOR,
                layers=layers, cached_model=cached_model,
            )
            outs[s] = out_file
        return outs, cached_model

    print(f"[multi] running {len(strategies)} strategies in one forward pass per pair: {strategies}", flush=True)
    written = execute_get_activations_multi(
        pairs_file=str(pairs_file),
        output_dir=str(output_dir),
        model=cached_model,
        model_name=model,
        device=device,
        layers=layers,
        strategies=strategies,
        component=component,
        capture_qk=True,
        verbose=False,
    )
    return {s: Path(p) for s, p in written.items()}, cached_model


def _try_preload_model(model_id: str, device: str):
    """Load the WisentModel once so all 7 strategies reuse it. Returns model or None."""
    try:
        from wisent.core.primitives.models.wisent_model import WisentModel
    except Exception as exc:
        print(f"[preload] WisentModel import failed ({exc}); strategies will each load their own", flush=True)
        return None
    try:
        print(f"[preload] loading {model_id} on {device} (one-shot for all strategies)", flush=True)
        return WisentModel(model_id, device=device)
    except Exception as exc:
        print(f"[preload] model load failed ({exc}); per-strategy load will be used", flush=True)
        return None


def upload_to_hf(activations_file: Path, model: str, task: str) -> None:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.find_best.activation_cache import (
        upload_extracted_activations,
    )
    upload_extracted_activations(str(activations_file), model, task)


from .auto_batch import auto_batch_size  # noqa: E402


def _list_hf_repo_files_once() -> list[str] | None:
    """One-shot listing of every file in wisent-ai/activations.

    The repo has thousands of files, so a single listing call hits HF
    pagination + rate limits even with the SDK's 5x retry. Calling it
    seven times (once per strategy) compounds the rate-limit risk;
    sharing the result across all strategies drops it to one call.
    Returns None on failure so callers treat each strategy as not-yet-done.
    """
    try:
        from huggingface_hub import HfApi
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            HF_REPO_ID, HF_REPO_TYPE,
        )
        api = HfApi()
        return list(api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE))
    except Exception as exc:
        print(f"[skip-check] HF listing unavailable: {exc}", flush=True)
        return None


def hf_already_has_strategy(
    model: str,
    task: str,
    strategy: str,
    component: str,
    cached_files: list[str] | None,
) -> bool:
    """True if every layer for (model, task, strategy) already exists on HF.

    Used as an idempotency check so a Spot-preempted job that gets requeued
    doesn't re-extract strategies it had already finished + uploaded before
    the preemption. Returns False when cached_files is None so the caller
    proceeds with extraction (worst case: redundant re-upload of an
    already-present shard, which the writer overwrites deterministically).
    """
    if cached_files is None:
        return False
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            model_to_safe_name,
        )
    except Exception:
        return False
    safe = model_to_safe_name(model)
    hf_strategy = strategy if component == "residual_stream" else f"{strategy}/{component}"
    prefix = f"activations/{safe}/{task}/{hf_strategy}/"
    return any(f.startswith(prefix) for f in cached_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategies", nargs="+", default=VALIDATED_STRATEGIES)
    parser.add_argument("--component", default="residual_stream")
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_FLOOR,
                        help="Floor batch size; auto-tune may pick larger.")
    parser.add_argument("--layers", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on contrastive pairs (forwarded to generate-pairs-from-task).")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Per-job working dir. Defaults to a unique tempfile.mkdtemp() per "
             "invocation so concurrent agent slots can't collide on shared "
             "files (the previous shared default /tmp/wisent_activations_work "
             "caused 'No such file or directory' failures when two jobs ran "
             "the same task concurrently and one's cleanup wiped the other's "
             "pairs file).",
    )
    args = parser.parse_args()
    # Always print a startup line so command_output.log is never empty.
    # If the script exits before any other output (e.g. an early-import
    # crash or a shell wrapper failure), this line confirms the script
    # at least started.
    import datetime as _dt
    print(f"[{args.task}] extract_and_upload start at {_dt.datetime.utcnow().isoformat()}Z model={args.model} strategies={args.strategies} layers={args.layers}", flush=True)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        owns_work_dir = False
    else:
        # Unique-per-invocation. Task name is in the prefix so logs/profile
        # PNGs are still self-describing on disk before the rmtree at exit.
        safe_task = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.task)[:64]
        work_dir = Path(tempfile.mkdtemp(prefix=f"wisent_act_{safe_task}_pid{os.getpid()}_"))
        owns_work_dir = True
    pairs_file = work_dir / f"{args.task}__pairs.json"

    generate_pairs(args.task, pairs_file, limit=args.limit)
    print(f"[{args.task}] pairs_file={pairs_file}", flush=True)
    pairs_initial_count, pairs_initial_subtask_counts = _count_pairs_by_subtask(pairs_file)
    print(f"[{args.task}] pairs_initial_count={pairs_initial_count}", flush=True)
    if pairs_initial_count == 0:
        raise SystemExit(
            f"[{args.task}] FAIL: generate-pairs-from-task produced 0 contrastive pairs "
            "for this task. Aborting before any extraction — uploading empty payloads "
            "would mark the job COMPLETED while polluting HF with zero-activation shards."
        )

    # Per-pair tracking: drop pairs whose stable_id already has activations
    # on HF for THIS (model, task) — strategy-agnostic: if any strategy's
    # layer-1 shard contains the stable_id, the pair has been processed.
    # This prevents re-extraction across re-runs and lets follow-up jobs
    # incrementally extend coverage. Falls through silently if HF is
    # unavailable or no existing shards (first run).
    try:
        pending, skipped_strategies, requested_count, union_missing = _strategies_pending(
            pairs_file, args.model, args.task, args.strategies, args.component,
        )
    except Exception as _exc:
        print(f"[{args.task}] coverage probe failed ({_exc}); processing all strategies", flush=True)
        pending = list(args.strategies)
        skipped_strategies = []
        requested_count = pairs_initial_count
        union_missing = None
    pairs_post_filter_count = requested_count
    pairs_post_filter_subtask_counts = pairs_initial_subtask_counts
    print(f"[{args.task}] pending strategies={pending} skipped={skipped_strategies} requested_count={requested_count}", flush=True)
    if not pending:
        # Every requested strategy already coverage-complete on HF.
        print(f"[{args.task}] all {len(args.strategies)} strategies already coverage-complete on HF for {requested_count} pairs; nothing to do", flush=True)
        return 0
    if union_missing is not None and len(union_missing) < requested_count:
        # Prune pairs_file to just the pairs at least one pending
        # strategy still needs. The GPU forward pass skips pairs
        # every pending strategy already has. _merge_existing_shard
        # on upload preserves the existing rows for strategies that
        # had a strict superset of the union.
        kept, orig = _prune_pairs_file_to_sids(pairs_file, union_missing)
        print(f"[{args.task}] pruned pairs_file to {kept}/{orig} pairs (union of missing across pending strategies)", flush=True)
        pairs_post_filter_count = kept

    profiler = GPUProfiler(
        csv_path=work_dir / f"{args.task}__profile.csv",
        phases_path=work_dir / f"{args.task}__phases.csv",
        interval_sec=3.0,
    )
    profiler.start()
    profiler.mark_phase("start")

    profiler.mark_phase("model_load")
    cached = _try_preload_model(args.model, args.device) if pending else None
    profiler.mark_phase("model_loaded")
    profiler.mark_phase("auto_batch_probe")
    effective_bs = (
        auto_batch_size(cached, args.device, args.batch_size, pairs_file=pairs_file)
        if cached else args.batch_size
    )
    profiler.mark_phase(f"bs={effective_bs}")
    if effective_bs != args.batch_size:
        print(f"[{args.task}] auto-tuned batch_size {args.batch_size} -> {effective_bs}", flush=True)

    failed_strategies: list[tuple[str, str]] = []
    uploaded_strategies: set[str] = set()
    groups = _group_strategies_by_family(pending)
    for group in groups:
        group_label = "+".join(group)
        profiler.mark_phase(f"extract_{group_label}")
        try:
            outs, cached = run_get_activations_group(
                pairs_file=pairs_file,
                output_dir=work_dir,
                model=args.model,
                strategies=group,
                component=args.component,
                device=args.device,
                layers=args.layers,
                cached_model=cached,
            )
        except Exception as exc:
            for s in group:
                failed_strategies.append((s, str(exc)))
            profiler.mark_phase(f"FAILED_{group_label}")
            print(f"[{args.task}] group={group_label} extraction failed: {exc}; continuing", flush=True)
            continue
        for strategy, out_file in outs.items():
            try:
                with open(out_file) as _f:
                    _doc = json.load(_f)
                _num_pairs_out = int(_doc.get("num_pairs", 0))
                _activations_present = sum(
                    1 for _p in _doc.get("pairs", [])
                    if _p.get("positive_response", {}).get("layers_activations")
                )
            except Exception as _exc:
                failed_strategies.append((strategy, f"output unreadable: {_exc}"))
                print(f"[{args.task}] strategy={strategy} FAIL: output unreadable ({_exc})", flush=True)
                continue
            if _num_pairs_out == 0 or _activations_present == 0:
                failed_strategies.append(
                    (strategy, f"0 activations (num_pairs={_num_pairs_out}, with_acts={_activations_present})")
                )
                print(
                    f"[{args.task}] strategy={strategy} FAIL: 0 activations extracted "
                    f"(num_pairs={_num_pairs_out}, with_acts={_activations_present}); not uploading",
                    flush=True,
                )
                continue
            profiler.mark_phase(f"upload_{strategy}")
            try:
                upload_to_hf(out_file, args.model, args.task)
                print(f"[{args.task}] uploaded strategy={strategy}", flush=True)
                uploaded_strategies.add(strategy)
            except Exception as exc:
                failed_strategies.append((strategy, str(exc)))
                print(f"[{args.task}] strategy={strategy} upload failed: {exc}; continuing", flush=True)
                continue
            try:
                out_file.unlink()
            except OSError:
                pass

    # Catch silently-dropped strategies: any strategy in pending that
    # neither uploaded nor recorded as failed has no output on HF.
    # Without this check the script could exit 0 with partial output.
    _failed_set = {s for s, _ in failed_strategies}
    for _s in pending:
        if _s in uploaded_strategies or _s in _failed_set:
            continue
        failed_strategies.append((_s, "no output produced (silent skip in group call)"))
        print(f"[{args.task}] strategy={_s} FAIL: no output produced (silent skip)", flush=True)

    profiler.mark_phase("done")
    profiler.stop()

    _write_manifest(
        args=args,
        pairs_initial_count=pairs_initial_count,
        pairs_initial_subtask_counts=pairs_initial_subtask_counts,
        pairs_post_filter_count=pairs_post_filter_count,
        pairs_post_filter_subtask_counts=pairs_post_filter_subtask_counts,
        pending=list(pending),
        failed_strategies=list(failed_strategies),
    )
    png_path = work_dir / f"{args.task}__profile.png"
    if render_png(profiler.csv_path, profiler.phases_path, png_path,
                  title=f"{args.task} on {args.model}"):
        print(f"[profile] saved {png_path}", flush=True)

    try:
        pairs_file.unlink()
    except OSError:
        pass
    if owns_work_dir:
        # We mkdtemp'd it; rmtree the whole thing so the VM's /tmp doesn't
        # accumulate one dir per task. Best-effort: profile PNG is the only
        # artifact the operator might want to keep, and it's already been
        # printed to stdout by render_png; the GCS uploader doesn't pick it
        # up so it would be discarded on VM shutdown anyway.
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
    if failed_strategies:
        print(f"[{args.task}] {len(failed_strategies)}/{len(pending)} strategies failed:", flush=True)
        for s, e in failed_strategies:
            print(f"  {s}: {e}", flush=True)
        # Any strategy failure -> non-zero. Earlier behavior masked HF 429
        # rate-limit failures (5/7 strategies dying but script returning 0,
        # marking the wisent-compute job COMPLETED despite missing uploads).
        # Confirmed live 2026-05-06: 20/20 sampled completions had this
        # exact pattern. Runner restart budget retries on a fresh VM.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
