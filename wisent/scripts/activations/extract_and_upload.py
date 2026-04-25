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
) -> None:
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


def upload_to_hf(activations_file: Path, model: str, task: str) -> None:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.find_best.activation_cache import (
        upload_extracted_activations,
    )
    upload_extracted_activations(str(activations_file), model, task)


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

    for strategy in args.strategies:
        # Idempotency: if a previous (possibly preempted) attempt already
        # uploaded this strategy's shards to HF, skip the re-extraction.
        if hf_already_has_strategy(args.model, args.task, strategy, args.component):
            print(f"[{args.task}] strategy={strategy} already on HF, skipping", flush=True)
            continue
        out_file = work_dir / f"{args.task}__{strategy}.json"
        run_get_activations(
            pairs_file=pairs_file,
            output_file=out_file,
            model=args.model,
            strategy=strategy,
            component=args.component,
            device=args.device,
            batch_size=args.batch_size,
            layers=args.layers,
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
