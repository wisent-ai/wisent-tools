"""Cold tier for raw activation pending uploads.

Hot root: fast NVMe TMPDIR for active extractor writes.
Cold root: large HDD for upload-ready backlog while bandwidth drains.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def _gb_free(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def cold_root() -> Path | None:
    raw = os.environ.get("WISENT_RAW_COLD_PENDING_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw)
    root.mkdir(parents=True, exist_ok=True)
    return root


def pending_roots(hot_root: Path) -> list[Path]:
    cold = cold_root()
    return [hot_root] if cold is None or cold == hot_root else [hot_root, cold]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _hot_free_target_gb() -> float:
    target = _env_float("WISENT_RAW_HOT_FREE_TARGET_GB", 768.0)
    claim_min = _env_float("WISENT_RAW_CLAIM_MIN_FREE_GB", target)
    claim_reserve = _env_float("WISENT_RAW_CLAIM_RESERVE_GB", 180.0)
    return max(target, claim_min + claim_reserve)


def active_pending_root(hot_root: Path) -> Path:
    """Root for a new extraction dir: hot NVMe unless it is pressure-gated."""
    raw = os.environ.get("WISENT_RAW_ACTIVE_FALLBACK_ROOT", "").strip()
    if not raw:
        return hot_root
    hot_min = _env_float(
        "WISENT_RAW_ACTIVE_HOT_MIN_FREE_GB",
        150.0,
    )
    cap = _env_float("WISENT_RAW_ACTIVE_HOT_MIN_FREE_GB_CAP", 150.0)
    if cap > 0:
        hot_min = min(hot_min, cap)
    if hot_min <= 0 or _gb_free(hot_root) >= hot_min:
        return hot_root
    fallback = Path(raw)
    fallback.mkdir(parents=True, exist_ok=True)
    min_free = _env_float("WISENT_RAW_ACTIVE_FALLBACK_MIN_FREE_GB", 512.0)
    return fallback if _gb_free(fallback) >= min_free else hot_root


def _unique_dst(cold: Path, name: str) -> Path:
    dst = cold / name
    if not dst.exists():
        return dst
    base = f"{name}.{int(time.time())}"
    dst = cold / base
    i = 1
    while dst.exists():
        dst = cold / f"{base}.{i}"
        i += 1
    return dst


def _publish_to_cold(src: Path, cold: Path) -> bool:
    """Start a background copy through a non-scanned staging dir."""
    dst = _unique_dst(cold, src.name)
    staging_parent = cold.parent / f"{cold.name}._moving"
    staging_parent.mkdir(parents=True, exist_ok=True)
    tmp = staging_parent / f"{dst.name}.partial"
    lock = src / ".cold_moving"
    if _lock_active(lock):
        return False
    bwlimit = os.environ.get("WISENT_RAW_COLD_COPY_BWLIMIT_KB", "32768")
    try:
        lock.write_text("starting\n")
        log = staging_parent / "cold_spill.log"
        script = r'''
set -u
bwlimit="$1"; src="$2"; tmp="$3"; dst="$4"; lock="$5"; log="$6"
cleanup() {
  rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$lock"
    printf '%s interrupted rc=%s src=%s tmp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$src" "$tmp" >> "$log"
  fi
  exit "$rc"
}
trap cleanup EXIT
{
  printf '%s start src=%s dst=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$src" "$dst"
  ionice -c3 nice -n 19 rsync -a --delete --partial --append-verify --bwlimit="$bwlimit" "$src/" "$tmp/"
  mv "$tmp" "$dst"
  rm -rf "$src"
  printf '%s done src=%s dst=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$src" "$dst"
} >> "$log" 2>&1
'''
        proc = subprocess.Popen(
            ["/bin/bash", "-c", script, "cold-spill", bwlimit,
             str(src), str(tmp), str(dst), str(lock), str(log)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        lock.write_text(f"{proc.pid}\n")
        return True
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            lock.unlink()
        except OSError:
            pass
        return False


def _lock_active(lock: Path) -> bool:
    if not lock.exists():
        return False
    try:
        raw = lock.read_text().strip()
        pid = int(raw)
    except ValueError:
        try:
            if time.time() - lock.stat().st_mtime < 60:
                return True
        except OSError:
            return False
        try:
            lock.unlink()
        except OSError:
            pass
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        pass
    try:
        lock.unlink()
    except OSError:
        pass
    return False


def _dir_bytes(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for ent in it:
                    try:
                        st = ent.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    total += st.st_size
                    if ent.is_dir(follow_symlinks=False):
                        stack.append(Path(ent.path))
        except OSError:
            continue
    return total


def _pick_candidate(candidates: list[tuple[int, Path]], need_bytes: int) -> tuple[int, Path]:
    enough = [c for c in candidates if c[0] >= need_bytes]
    if enough:
        return min(enough, key=lambda c: c[0])
    return max(candidates, key=lambda c: c[0])


def _active_cold_movers(hot_root: Path) -> int:
    active = 0
    try:
        dirs = [p for p in hot_root.iterdir() if p.is_dir()]
    except OSError:
        return 0
    for d in dirs:
        if _lock_active(d / ".cold_moving"):
            active += 1
    return active


def spill_hot_to_cold(hot_root: Path, has_live_worker) -> int:
    cold = cold_root()
    if cold is None:
        return 0
    max_movers = max(1, _env_int("WISENT_RAW_COLD_MAX_MOVERS", 1))
    if _active_cold_movers(hot_root) >= max_movers:
        return 0
    target = _hot_free_target_gb()
    try:
        free = _gb_free(hot_root)
        dirs = [p for p in hot_root.iterdir() if p.is_dir()]
    except OSError:
        return 0
    candidates = []
    for d in dirs:
        if (
            (d / ".upload_meta").exists()
            and not has_live_worker(d)
            and not _lock_active(d / ".cold_moving")
        ):
            candidates.append((_dir_bytes(d), d))
    moved = 0
    while free < target and candidates:
        need_bytes = int((target - free) * (1024 ** 3))
        size, d = _pick_candidate(candidates, need_bytes)
        candidates = [c for c in candidates if c[1] != d]
        if _publish_to_cold(d, cold):
            moved += 1
            break
        else:
            break
    return moved
