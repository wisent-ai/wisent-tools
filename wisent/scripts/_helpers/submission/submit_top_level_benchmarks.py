#!/usr/bin/env python3
"""Submit raw activation jobs for top-level benchmarks.

Creates one queue job per (model, top-level benchmark, prompt format) using
the canonical wisent benchmark list — NOT leaf-expanded subtask names.

Source of truth: benchmark_tags.json (380 entries). The pair-generation
pipeline already handles group→subtask expansion via build_contrastive_pairs
in lm_task_pairs_generation.py, so submitting the top-level name is
sufficient. Combined with the per-pair stable_id tracking shipped in wisent
0.11.30/31, re-runs incrementally extend coverage across the same shards
instead of redoing the same first-N pairs.

Usage:
    python -m wisent.scripts._helpers.submission.submit_top_level_benchmarks \\
        --model meta-llama/Llama-3.2-1B-Instruct --priority 1000
    python -m wisent.scripts._helpers.submission.submit_top_level_benchmarks \\
        --model X --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from wisent.core.utils.config_tools.constants import GSM8K_DEFAULT_LIMIT


def load_benchmark_names() -> list[str]:
    """380 canonical wisent benchmark names from benchmark_tags.json."""
    candidates = []
    try:
        import wisent
        candidates.append(Path(wisent.__file__).parent / "support"
                          / "examples" / "scripts" / "benchmark_tags.json")
    except Exception:
        pass
    candidates += [
        Path("/opt/wisent-agent/.venv/lib/python3.10/site-packages/wisent"
             "/support/examples/scripts/benchmark_tags.json"),
        Path("/usr/local/lib/python3.13/site-packages/wisent/support/"
             "examples/scripts/benchmark_tags.json"),
    ]
    for p in candidates:
        if p.is_file():
            return sorted(json.loads(p.read_text()).keys())
    raise SystemExit("benchmark_tags.json not found")


def _wc_bin() -> str:
    """`wc` (wisent-compute) CLI; prefers user pip-install."""
    for cand in (
        os.environ.get("WC_BIN"),
        os.path.expanduser("~/Library/Python/3.12/bin/wc"),
        os.path.expanduser("~/.local/bin/wc"),
        "/opt/wisent-agent/.venv/bin/wc",
    ):
        if cand and os.path.isfile(cand):
            return cand
    found = shutil.which("wc")
    if found:
        try:
            r = subprocess.run([found, "--help"], capture_output=True, text=True)
            if "wisent" in (r.stdout or "").lower():
                return found
        except Exception:
            pass
    raise SystemExit("wc binary not found")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=GSM8K_DEFAULT_LIMIT,
                   help="per-benchmark pair cap (default from constants)")
    p.add_argument("--prompt-formats", default="chat,mc_balanced,role_play")
    p.add_argument("--component", default="residual_stream")
    p.add_argument("--layers", default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--provider", default="local", choices=("local",))
    p.add_argument("--gpu-type", default="")
    p.add_argument("--vram-gb", type=int, default=0)
    p.add_argument("--exclusive", action="store_true")
    p.add_argument("--exclude", default="")
    p.add_argument("--only", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    names = load_benchmark_names()
    excludes = {x.strip() for x in args.exclude.split(",") if x.strip()}
    onlys = {x.strip() for x in args.only.split(",") if x.strip()}
    if onlys:
        names = [n for n in names if n in onlys]
    names = [n for n in names if n not in excludes]
    print(f"Submitting {len(names)} top-level jobs for {args.model}", flush=True)

    prompt_formats = [x.strip() for x in args.prompt_formats.split(",") if x.strip()]
    pairs: list[str] = []
    for name in names:
        for prompt_format in prompt_formats:
            cmd = (
                "python3 -m wisent.scripts.activations.raw.extract_and_upload "
                f"--task {name} --model '{args.model}' --device {args.device} "
                f"--layers {args.layers} --prompt-format {prompt_format} "
                f"--component {args.component} --limit {args.limit}"
            )
            pairs.append(cmd)
    if args.dry_run:
        for c in pairs:
            print(c)
        return 0

    from wisent_compute.config import BUCKET as _BUCKET
    from wisent_compute.queue.submit import submit_job as _submit_job
    import time as _time
    batch_id = f"batch-{int(_time.time())}-{args.model.replace('/', '_')}"
    n = 0
    for cmd in pairs:
        _submit_job(
            cmd, provider=args.provider, batch_id=batch_id, bucket=_BUCKET,
            preemptible=False, pin_to_provider=True, priority=args.priority,
            gpu_type=args.gpu_type, vram_gb=args.vram_gb,
            exclusive=args.exclusive,
        )
        n += 1
    print(f"\nSubmitted: {n}/{len(pairs)}  batch_id={batch_id}")
    return 0 if n == len(pairs) else 1


if __name__ == "__main__":
    sys.exit(main())
