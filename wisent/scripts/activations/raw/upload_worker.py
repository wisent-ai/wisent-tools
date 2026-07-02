"""Decoupled raw-activation upload workers with restart-safe pending dirs."""
from __future__ import annotations

import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

DEFAULT_MAX_UPLOAD_WORKERS = 2
DEFAULT_GCS_UPLOAD_WORKERS = 1
DEFAULT_MIN_UPLOAD_MEM_GB = 24.0
_MODULE = "wisent.scripts.activations.raw.upload_worker"
_EXTRACTOR = "wisent.scripts.activations.raw.extract_and_upload"
_ACTIVE_UPLOAD_PGID: int | None = None


def _pause_during_extract() -> bool:
    return False


def pending_root() -> Path:
    root = Path(os.environ.get("TMPDIR", "/tmp")) / "wisent_raw_pending"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job_dir(task: str, prompt_format: str) -> Path:
    from .storage.cold_tier import active_pending_root
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in f"{task}__{prompt_format}"
    )[:80]
    d = Path(tempfile.mkdtemp(prefix=f"{safe}__", dir=str(active_pending_root(pending_root()))))
    (d / "data").mkdir(parents=True, exist_ok=True)
    return d


def _live_worker_count() -> int:
    try:
        r = subprocess.run(["pgrep", "-fc", _MODULE],
                           capture_output=True, text=True)
        return int((r.stdout or "0").strip() or "0")
    except Exception:
        return 0


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return -1.0


def _upload_memory_ok() -> bool:
    avail = _mem_available_gb()
    floor = _env_float("WISENT_RAW_UPLOAD_MIN_MEM_GB", DEFAULT_MIN_UPLOAD_MEM_GB)
    return avail < 0 or avail >= floor


def _has_live_worker(job_dir: Path) -> bool:
    pid_f = job_dir / ".uploading"
    if not pid_f.exists():
        return False
    try:
        os.kill(int(pid_f.read_text().strip()), 0)
        return True
    except (ValueError, OSError):
        return False


def _is_cold_moving(job_dir: Path) -> bool:
    pid_f = job_dir / ".cold_moving"
    if not pid_f.exists():
        return False
    try:
        os.kill(int(pid_f.read_text().strip()), 0)
        return True
    except (ValueError, OSError):
        return False


def spawn_worker(job_dir: Path) -> bool:
    if _live_worker_count() >= _env_int("WISENT_RAW_MAX_UPLOAD_WORKERS", DEFAULT_MAX_UPLOAD_WORKERS):
        return False
    if not _upload_memory_ok():
        return False
    proc = subprocess.Popen(
        [sys.executable, "-m", _MODULE, str(job_dir)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    (job_dir / ".uploading").write_text(str(proc.pid))
    return True


def handoff(job_dir: Path, repo_id: str, base_in_repo: str,
            repo_type: str, job_id: str = "") -> None:
    (job_dir / ".upload_meta").write_text(
        f"{repo_id}\n{base_in_repo}\n{repo_type}\n{job_id}\n"
    )
    spawn_worker(job_dir)
    sweep()


def _mark_uploaded(job_id: str) -> None:
    if not job_id:
        return
    try:
        from google.cloud import storage
        bucket = storage.Client().bucket(os.environ.get("WC_BUCKET", "wisent-compute"))
        src = bucket.blob(f"completed/{job_id}.json")
        if not src.exists():
            return
        rec = json.loads(src.download_as_text())
        rec["state"] = "uploaded"
        bucket.blob(f"uploaded/{job_id}.json").upload_from_string(json.dumps(rec))
        src.delete()
    except Exception:
        pass


def _is_gcs_target(repo_id: str, repo_type: str) -> bool:
    return repo_type == "gcs" or repo_id.startswith("gs://")


def _gcs_dest(repo_id: str, base_in_repo: str) -> str:
    return f"{repo_id.rstrip('/')}/{base_in_repo.strip('/')}"


def _parse_gcs_dest(repo_id: str, base_in_repo: str) -> tuple[str, str]:
    dest = _gcs_dest(repo_id, base_in_repo)
    if not dest.startswith("gs://"):
        raise ValueError(f"not a GCS destination: {dest}")
    bucket, _, prefix = dest[5:].partition("/")
    return bucket, prefix.strip("/")


def sweep() -> int:
    from .storage.cold_tier import pending_roots, spill_hot_to_cold
    spawned = 0
    hot = pending_root()
    spill_hot_to_cold(hot, _has_live_worker)
    for root in pending_roots(hot):
        try:
            dirs = sorted(root.iterdir())
        except OSError:
            continue
        for d in dirs:
            if not d.is_dir() or not (d / ".upload_meta").exists():
                continue
            if _is_cold_moving(d):
                continue
            if _has_live_worker(d):
                continue
            if not spawn_worker(d):
                return spawned
            spawned += 1
    return spawned


def _io_bytes(pid: int) -> int:
    try:
        tot = 0
        with open(f"/proc/{pid}/io") as fh:
            for line in fh:
                if line.startswith(("rchar:", "wchar:")):
                    tot += int(line.split()[1])
        return tot
    except OSError:
        return -1


def _append_log(log_path: Path, message: str) -> None:
    try:
        with log_path.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")
    except OSError:
        pass


def _extractor_pids() -> list[int]:
    try:
        r = subprocess.run(["pgrep", "-f", _EXTRACTOR],
                           capture_output=True, text=True)
        return [int(s) for s in (r.stdout or "").split() if int(s) != os.getpid()]
    except Exception:
        return []


def _wait_for_extractors_to_clear(log_path: Path, label: str = "before_start") -> None:
    logged = False
    while _extractor_pids():
        if not logged:
            _append_log(log_path, f"pause {label} active_extractor")
            logged = True
        time.sleep(5.0)
    if logged:
        _append_log(log_path, f"resume {label} active_extractor")


def _pause_child_during_extract(proc: subprocess.Popen, log_path: Path) -> bool:
    return False


def _kill_active_upload() -> None:
    pgid = _ACTIVE_UPLOAD_PGID
    if not pgid:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        time.sleep(1.0)


def _stop_with_child(signum, _frame) -> None:
    _kill_active_upload()
    raise SystemExit(128 + int(signum))


def _run_upload(cmd: list[str], env: dict, stall_s: float, log_path: Path):
    _append_log(log_path, "start " + " ".join(shlex.quote(x) for x in cmd))
    global _ACTIVE_UPLOAD_PGID
    with log_path.open("ab") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                stdin=subprocess.DEVNULL)
    try:
        _ACTIVE_UPLOAD_PGID = os.getpgid(proc.pid)
        last_io = _io_bytes(proc.pid)
        last_progress = time.time()
        while True:
            try:
                rc = proc.wait(timeout=min(stall_s, 5.0))
                _append_log(log_path, f"exit rc={rc} io={last_io}")
                return rc
            except subprocess.TimeoutExpired:
                if _pause_child_during_extract(proc, log_path):
                    last_io = _io_bytes(proc.pid)
                    last_progress = time.time()
                    continue
                io = _io_bytes(proc.pid)
                if io > last_io:
                    last_io, last_progress = io, time.time()
                elif time.time() - last_progress >= stall_s:
                    _kill_active_upload()
                    proc.wait()
                    _append_log(log_path, f"killed no_io_progress pid={proc.pid} io={io}")
                    return None
    finally:
        _ACTIVE_UPLOAD_PGID = None


def _upload_command(repo_id: str, data: Path, base_in_repo: str,
                    repo_type: str) -> list[str]:
    args = [
        "upload", repo_id, str(data), base_in_repo,
        "--repo-type", repo_type,
    ]
    hf = shutil.which("hf")
    if hf:
        return [hf, *args]
    hfc = shutil.which("huggingface-cli")
    if hfc:
        return [hfc, *args]
    return [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", *args]


def _run_gcs_upload_child(repo_id: str, data: Path, base_in_repo: str,
                          log_path: Path) -> int:
    try:
        from google.cloud import storage
        bucket_name, prefix = _parse_gcs_dest(repo_id, base_in_repo)
        workers = _env_int("WISENT_GCS_UPLOAD_WORKERS", DEFAULT_GCS_UPLOAD_WORKERS)
        _append_log(log_path, f"gcs_start workers={workers} dest=gs://{bucket_name}/{prefix}")
        bucket = storage.Client().bucket(bucket_name)

        def _one(path: Path) -> None:
            rel = path.relative_to(data).as_posix()
            name = f"{prefix}/{rel}" if prefix else rel
            bucket.blob(name).upload_from_filename(str(path))

        done_count = 0
        if workers <= 1:
            for path in data.rglob("*"):
                if not path.is_file():
                    continue
                _one(path)
                done_count += 1
                if done_count % 25 == 0:
                    _append_log(log_path, f"gcs_progress files={done_count}")
        else:
            pending = set()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for path in data.rglob("*"):
                    if not path.is_file():
                        continue
                    pending.add(pool.submit(_one, path))
                    if len(pending) >= workers * 2:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in finished:
                            fut.result()
                            done_count += 1
                            if done_count % 25 == 0:
                                _append_log(log_path, f"gcs_progress files={done_count}")
                for fut in as_completed(pending):
                    fut.result()
                    done_count += 1
                    if done_count % 25 == 0:
                        _append_log(log_path, f"gcs_progress files={done_count}")
        _append_log(log_path, f"gcs_done files={done_count}")
        return 0
    except Exception as exc:
        _append_log(log_path, f"gcs_error {type(exc).__name__}: {exc}")
        return 1


def _run_gcs_upload(repo_id: str, data: Path, base_in_repo: str,
                    log_path: Path, stall_s: float) -> int | None:
    cmd = [
        sys.executable, "-m", _MODULE, "--gcs-upload",
        repo_id, str(data), base_in_repo, str(log_path),
    ]
    return _run_upload(cmd, os.environ.copy(), stall_s, log_path)


def run_worker(job_dir_str: str) -> int:
    signal.signal(signal.SIGTERM, _stop_with_child)
    signal.signal(signal.SIGINT, _stop_with_child)
    for sig in ("SIGTSTP", "SIGTTIN", "SIGTTOU"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), signal.SIG_IGN)
    job_dir = Path(job_dir_str)
    if not job_dir.is_dir() or not (job_dir / ".upload_meta").exists():
        return 0
    if _is_cold_moving(job_dir):
        try:
            (job_dir / ".uploading").unlink()
        except OSError:
            pass
        return 0
    (job_dir / ".uploading").write_text(str(os.getpid()))
    meta = (job_dir / ".upload_meta").read_text().splitlines()
    repo_id, base_in_repo, repo_type = meta[0], meta[1], meta[2]
    job_id = meta[3] if len(meta) > 3 else ""
    from .commit_rate import acquire_commit_slot
    # Use standard LFS multipart upload; xet has wedged on this uplink.
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    stall_s = float(os.environ.get("WISENT_UPLOAD_STALL_S", str(900.0)))
    data = job_dir / "data"
    log_path = job_dir / ".upload_log"
    _append_log(log_path, f"worker pid={os.getpid()} mode=concurrent_no_pause job_id={job_id} dest={base_in_repo}")
    rc = 1
    for attempt in range(20):
        if _pause_during_extract():
            _wait_for_extractors_to_clear(log_path)
        if not _is_gcs_target(repo_id, repo_type):
            acquire_commit_slot()
        _append_log(log_path, f"attempt={attempt + 1}")
        if _is_gcs_target(repo_id, repo_type):
            ret = _run_gcs_upload(repo_id, data, base_in_repo, log_path, stall_s)
        else:
            ret = _run_upload(
                _upload_command(repo_id, data, base_in_repo, repo_type),
                env, stall_s, log_path,
            )
        if ret == 0:
            _mark_uploaded(job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            rc = 0
            break
        _append_log(log_path, f"retry ret={ret}")
        time.sleep(min(30.0, 2.0 ** attempt))
    if rc != 0:
        try:
            (job_dir / ".uploading").unlink()
        except OSError:
            pass
    sweep()
    return rc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gcs-upload":
        if len(sys.argv) != 6:
            raise SystemExit("usage: upload_worker --gcs-upload REPO_ID DATA BASE_IN_REPO LOG_PATH")
        raise SystemExit(_run_gcs_upload_child(
            sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
        ))
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        print(f"swept; spawned {sweep()} worker(s)")
        raise SystemExit(0)
    raise SystemExit(run_worker(sys.argv[1]))
