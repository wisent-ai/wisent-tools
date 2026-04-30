#!/usr/bin/env python3
"""Extract activations for ONE benchmark + model across 7 strategies; upload to HF.

Per-task pipeline: (1) `wisent generate-pairs-from-task` (HF-cached), (2) for each
strategy, `wisent get-activations` on residual_stream / --layers all, (3) upload
every layer shard to wisent-ai/activations, (4) delete local JSON.
"""
from __future__ import annotations

import argparse
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
        default=str(Path(tempfile.gettempdir()) / "wisent_activations_work"),
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = work_dir / f"{args.task}__pairs.json"

    generate_pairs(args.task, pairs_file, limit=args.limit)
    print(f"[{args.task}] pairs_file={pairs_file}", flush=True)

    # Decide which strategies actually need extraction (skipping ones already
    # on HF), then load the model once if any remain. For large models this
    # cuts ~6x of weight-loading per task.
    cached_hf_files = _list_hf_repo_files_once()
    pending = [
        s for s in args.strategies
        if not hf_already_has_strategy(
            args.model, args.task, s, args.component, cached_hf_files,
        )
    ]
    skipped = [s for s in args.strategies if s not in pending]
    for s in skipped:
        print(f"[{args.task}] strategy={s} already on HF, skipping", flush=True)

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
            profiler.mark_phase(f"upload_{strategy}")
            try:
                upload_to_hf(out_file, args.model, args.task)
                print(f"[{args.task}] uploaded strategy={strategy}", flush=True)
            except Exception as exc:
                failed_strategies.append((strategy, str(exc)))
                print(f"[{args.task}] strategy={strategy} upload failed: {exc}; continuing", flush=True)
                continue
            try:
                out_file.unlink()
            except OSError:
                pass

    profiler.mark_phase("done")
    profiler.stop()
    png_path = work_dir / f"{args.task}__profile.png"
    if render_png(profiler.csv_path, profiler.phases_path, png_path,
                  title=f"{args.task} on {args.model}"):
        print(f"[profile] saved {png_path}", flush=True)

    try:
        pairs_file.unlink()
    except OSError:
        pass
    if failed_strategies:
        print(f"[{args.task}] {len(failed_strategies)}/{len(pending)} strategies failed:", flush=True)
        for s, e in failed_strategies:
            print(f"  {s}: {e}", flush=True)
        return 2 if len(failed_strategies) == len(pending) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
