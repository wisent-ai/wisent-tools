"""Decoupled HF upload workers for raw-activation extraction.

The extractor writes a job's activations into a PERSISTENT pending dir
(<TMPDIR>/wisent_raw_pending/<id>/data) and hands the upload off to a
detached, torch-free worker (this module), then exits — so the agent's
GPU slot and the model's RAM are released at extraction-end instead of
being pinned through the slow, bandwidth-bound upload. Workers carry no
model, so many drain the pending pool concurrently at the uplink limit
without the per-process RAM that drove the agent OOM under the coupled
design.

Durability: a pending dir survives until its upload succeeds (then it's
removed), so a worker death or box restart loses no work — sweep()
re-spawns a worker for any dir whose extraction finished (.upload_meta
written) but has no live worker. Concurrency is bounded by
MAX_UPLOAD_WORKERS: uploads beyond what a fixed uplink can carry only add
overhead. That is the network pool size, distinct from the GPU/job
concurrency the agent's RAM admission gate governs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_UPLOAD_WORKERS = 12
_MODULE = "wisent.scripts.activations.raw.upload_worker"


def pending_root() -> Path:
    root = Path(os.environ.get("TMPDIR", "/tmp")) / "wisent_raw_pending"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job_dir(task: str, prompt_format: str) -> Path:
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in f"{task}__{prompt_format}"
    )[:80]
    d = Path(tempfile.mkdtemp(prefix=f"{safe}__", dir=str(pending_root())))
    (d / "data").mkdir(parents=True, exist_ok=True)
    return d


def _live_worker_count() -> int:
    try:
        r = subprocess.run(["pgrep", "-fc", _MODULE],
                           capture_output=True, text=True)
        return int((r.stdout or "0").strip() or "0")
    except Exception:
        return 0


def _has_live_worker(job_dir: Path) -> bool:
    pid_f = job_dir / ".uploading"
    if not pid_f.exists():
        return False
    try:
        os.kill(int(pid_f.read_text().strip()), 0)
        return True
    except (ValueError, OSError):
        return False


def spawn_worker(job_dir: Path) -> bool:
    """Spawn a detached worker if under the pool cap. True if spawned."""
    if _live_worker_count() >= MAX_UPLOAD_WORKERS:
        return False
    subprocess.Popen(
        [sys.executable, "-m", _MODULE, str(job_dir)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return True


def handoff(job_dir: Path, repo_id: str, base_in_repo: str,
            repo_type: str) -> None:
    """Record upload metadata, then hand the dir to the worker pool."""
    (job_dir / ".upload_meta").write_text(
        f"{repo_id}\n{base_in_repo}\n{repo_type}\n"
    )
    spawn_worker(job_dir)
    sweep()


def sweep() -> int:
    """Re-spawn workers for extraction-complete pending dirs lacking a live
    worker, up to the pool cap. Returns count spawned."""
    spawned = 0
    try:
        dirs = sorted(pending_root().iterdir())
    except OSError:
        return 0
    for d in dirs:
        if not d.is_dir() or not (d / ".upload_meta").exists():
            continue
        if _has_live_worker(d):
            continue
        if not spawn_worker(d):
            break
        spawned += 1
    return spawned


def run_worker(job_dir_str: str) -> int:
    job_dir = Path(job_dir_str)
    if not job_dir.is_dir() or not (job_dir / ".upload_meta").exists():
        return 0
    (job_dir / ".uploading").write_text(str(os.getpid()))
    meta = (job_dir / ".upload_meta").read_text().splitlines()
    repo_id, base_in_repo, repo_type = meta[0], meta[1], meta[2]
    from .commit_rate import acquire_commit_slot
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    env.pop("HF_HUB_DISABLE_XET", None)
    data = job_dir / "data"
    rc = 1
    for attempt in range(20):
        acquire_commit_slot()
        r = subprocess.run(
            ["hf", "upload", repo_id, str(data), base_in_repo,
             "--repo-type", repo_type],
            env=env, capture_output=True, text=True,
        )
        if r.returncode == 0:
            shutil.rmtree(job_dir, ignore_errors=True)
            rc = 0
            break
        time.sleep(min(30.0, 2.0 ** attempt))
    if rc != 0:
        try:
            (job_dir / ".uploading").unlink()
        except OSError:
            pass
    # chain-sweep so the pool stays self-sustaining as workers exit
    sweep()
    return rc


if __name__ == "__main__":
    raise SystemExit(run_worker(sys.argv[1]))
