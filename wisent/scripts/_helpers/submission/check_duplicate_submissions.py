"""Detect duplicate (model, task) submissions in the wisent-compute queue.

wc submit generates a fresh random job_id per call (queue/submit.py
_generate_job_id = os.urandom(4).hex()) with no command-based dedup, so
re-running submit_top_level_benchmarks for the same model creates
redundant jobs that all extract the same activations.

Output: prints (model, task) tuples that appear in multiple jobs submitted
within --within-hours hours, plus the per-state breakdown.

Usage:
    python -m wisent.scripts._helpers.submission.check_duplicate_submissions \\
        --within-hours 4
    python -m wisent.scripts._helpers.submission.check_duplicate_submissions \\
        --within-hours 6 --model meta-llama/Llama-2-7b-chat-hf
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from wisent_compute.config import BUCKET
from wisent_compute.queue.storage import JobStorage


_MODEL_RE = re.compile(r"--model\s+['\"]?([^'\"\s]+)")
_TASK_RE = re.compile(r"--task\s+([^\s]+)")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract(cmd: str) -> tuple[str, str]:
    m = _MODEL_RE.search(cmd or "")
    t = _TASK_RE.search(cmd or "")
    return (m.group(1) if m else "", t.group(1) if t else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--within-hours", type=float, required=True,
                   help="Only consider jobs with submitted_at within this window.")
    p.add_argument("--model", default="",
                   help="Filter to one model (e.g. meta-llama/Llama-2-7b-chat-hf).")
    p.add_argument("--show-job-ids", action="store_true")
    args = p.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.within_hours)

    store = JobStorage(BUCKET)
    all_jobs = store.list_all_jobs()

    by_key: dict[tuple[str, str], list] = defaultdict(list)
    state_counts: dict[str, int] = defaultdict(int)
    skipped_no_ts = 0
    skipped_old = 0
    most_recent: datetime | None = None

    for state, jobs in all_jobs.items():
        for job in jobs:
            ts_raw = getattr(job, "created_at", None) or getattr(job, "submitted_at", None)
            sub = _parse_iso(ts_raw)
            if sub is not None and (most_recent is None or sub > most_recent):
                most_recent = sub
            if sub is None:
                skipped_no_ts += 1
                continue
            if sub < cutoff:
                skipped_old += 1
                continue
            model, task = _extract(job.command or "")
            if args.model and model != args.model:
                continue
            by_key[(model, task)].append((state, job))
            state_counts[state] += 1

    print(f"Most recent created_at across all jobs: {most_recent}")
    print(f"Cutoff: {cutoff}")
    print(f"Skipped {skipped_no_ts} jobs (no timestamp), {skipped_old} jobs (older than cutoff)")

    hour_buckets: dict[str, int] = defaultdict(int)
    for state, jobs in all_jobs.items():
        for job in jobs:
            ts_raw = getattr(job, "created_at", None) or getattr(job, "submitted_at", None)
            sub = _parse_iso(ts_raw)
            if sub is None:
                continue
            key = sub.strftime("%Y-%m-%d %H")
            hour_buckets[key] += 1
    print("\nJobs by hour (most recent 12 hours of activity):")
    for h in sorted(hour_buckets.keys(), reverse=True)[:12]:
        print(f"  {h}: {hour_buckets[h]}")
    print(f"Jobs submitted within last {args.within_hours} h: "
          f"{sum(state_counts.values())}")
    for st in ("queue", "running", "completed", "failed"):
        print(f"  {st:>10}: {state_counts.get(st, 0)}")

    duplicates = [(k, v) for k, v in by_key.items() if len(v) > 1]
    duplicates.sort(key=lambda kv: -len(kv[1]))

    print(f"\nDistinct (model, task) groups: {len(by_key)}")
    print(f"Groups with duplicates (count > 1): {len(duplicates)}")

    per_model_dup_count: dict[str, int] = defaultdict(int)
    per_model_extra: dict[str, int] = defaultdict(int)
    for (model, task), entries in duplicates:
        per_model_dup_count[model] += 1
        per_model_extra[model] += len(entries) - 1

    print("\nDuplicate-(model, task) breakdown by model:")
    for model in sorted(per_model_dup_count):
        print(f"  {model:<46} groups_with_dup={per_model_dup_count[model]:<5} "
              f"redundant_jobs={per_model_extra[model]}")

    if args.show_job_ids and duplicates:
        print("\nFirst 20 duplicate groups:")
        for (model, task), entries in duplicates[:20]:
            print(f"  ({model}, {task}) -> {len(entries)} jobs")
            for state, job in entries:
                print(f"    {job.job_id} {state} submitted_at="
                      f"{getattr(job, 'submitted_at', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
