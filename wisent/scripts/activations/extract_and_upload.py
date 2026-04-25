#!/usr/bin/env python3
"""Extract activations for ONE benchmark + model across all 7 strategies and upload to HF.

Pipeline per task:
1. Generate (or HF-fetch) pair_texts via `wisent generate-pairs-from-task`.
2. For each of the 7 validated strategies, run `wisent get-activations`
   with extraction_component=residual_stream, --layers all.
3. Upload every layer shard to wisent-ai/activations via
   wisent.core.utils.cli.commands.optimize_steering.pipeline.find_best.activation_cache
   .upload_extracted_activations.
4. Delete the local activations JSON (HF shards are canonical).

Designed to be invoked as:
    python3 -m wisent.scripts.activations.extract_and_upload \
        --task <task> --model <model_id> [--strategies S1 S2 ...] \
        --device cuda --batch-size 8 --layers all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VALIDATED_STRATEGIES = [
    "chat_last",
    "chat_mean",
    "chat_first",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
]


def _wisent_bin() -> str:
    found = shutil.which("wisent")
    if not found:
        raise SystemExit("wisent CLI not found on PATH")
    return found


def generate_pairs(task: str, out_path: Path) -> None:
    cmd = [_wisent_bin(), "generate-pairs-from-task", task, "--output", str(out_path)]
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
            raise SystemExit(f"get-activations failed (rc={result.returncode}) for {strategy}")
        return None


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


def hf_already_has_strategy(model: str, task: str, strategy: str, component: str) -> bool:
    """True if every layer for (model, task, strategy) already exists on HF.

    Used as an idempotency check so a Spot-preempted job that gets requeued
    doesn't re-extract strategies it had already finished + uploaded before
    the preemption. Returns False on any HF query error so the caller treats
    it as not-yet-done and proceeds with extraction (worst case: redundant
    re-upload of an already-present shard, which the writer overwrites
    deterministically).
    """
    try:
        from huggingface_hub import HfApi
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name,
        )
    except Exception:
        return False
    try:
        safe = model_to_safe_name(model)
        hf_strategy = strategy if component == "residual_stream" else f"{strategy}/{component}"
        prefix = f"activations/{safe}/{task}/{hf_strategy}/"
        api = HfApi()
        files = list(api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE))
    except Exception as exc:
        print(f"[skip-check] HF list failed for {task}/{strategy}: {exc}", flush=True)
        return False
    matching = [f for f in files if f.startswith(prefix)]
    return len(matching) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategies", nargs="+", default=VALIDATED_STRATEGIES)
    parser.add_argument("--component", default="residual_stream")
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument(
        "--work-dir",
        default=str(Path(tempfile.gettempdir()) / "wisent_activations_work"),
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pairs_file = work_dir / f"{args.task}__pairs.json"

    generate_pairs(args.task, pairs_file)
    print(f"[{args.task}] pairs_file={pairs_file}", flush=True)

    # Decide which strategies actually need extraction (skipping ones already
    # on HF), then load the model once if any remain. For large models this
    # cuts ~6x of weight-loading per task.
    pending = [
        s for s in args.strategies
        if not hf_already_has_strategy(args.model, args.task, s, args.component)
    ]
    skipped = [s for s in args.strategies if s not in pending]
    for s in skipped:
        print(f"[{args.task}] strategy={s} already on HF, skipping", flush=True)

    cached = _try_preload_model(args.model, args.device) if pending else None
    effective_bs = (
        auto_batch_size(cached, args.device, args.batch_size, pairs_file=pairs_file)
        if cached else args.batch_size
    )
    if effective_bs != args.batch_size:
        print(f"[{args.task}] auto-tuned batch_size {args.batch_size} -> {effective_bs}", flush=True)

    for strategy in pending:
        out_file = work_dir / f"{args.task}__{strategy}.json"
        cached = run_get_activations(
            pairs_file=pairs_file,
            output_file=out_file,
            model=args.model,
            strategy=strategy,
            component=args.component,
            device=args.device,
            batch_size=effective_bs,
            layers=args.layers,
            cached_model=cached,
        )
        upload_to_hf(out_file, args.model, args.task)
        print(f"[{args.task}] uploaded strategy={strategy}", flush=True)
        try:
            out_file.unlink()
        except OSError:
            pass

    try:
        pairs_file.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
