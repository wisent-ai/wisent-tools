"""Decoupled raw-activation upload workers with restart-safe pending dirs."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from enum import Enum
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

class PublishScheme(str, Enum):
    RELATIVE = ""
    LOCAL = "local"
    HF = "hf"
    GCS = "gs"
    BUNDLE = "bundle"


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
            repo_type: str, job_id: str = "", *, v2: bool = False) -> None:
    normalized_prefix = base_in_repo.strip("/")
    if v2 and normalized_prefix != "raw_activations_v2":
        raise ValueError("v2 handoff requires the raw_activations_v2 prefix")
    if not v2 and (normalized_prefix == "raw_activations" or normalized_prefix.startswith("raw_activations/")):
        raise ValueError("legacy raw_activations is immutable; use an explicit v2 handoff")
    meta = {
        "schema_version": 2 if v2 else 1,
        "repo_id": repo_id,
        "base_in_repo": base_in_repo,
        "repo_type": repo_type,
        "job_id": job_id,
        "publish_mode": "create_only_two_phase" if v2 else "legacy_large_folder",
    }
    (job_dir / ".upload_meta").write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":"))
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


def _build_upload_root(job_dir: Path, data: Path, base_in_repo: str) -> Path:
    """Mirror data/ under <job_dir>/upload_root/<base_in_repo>/ via hardlinks.

    `hf upload-large-folder` uploads a folder to the repo ROOT (no path-in-repo
    arg), so to land shards at raw_activations/<model>/<task>/<format>/ the
    local tree must already carry that prefix. Hardlinks are zero-copy on the
    same filesystem and idempotent, so retries reuse the tree (and the tool's
    own .cache resume state under upload_root) instead of rebuilding.
    """
    root = job_dir / "upload_root"
    dest = root / base_in_repo.strip("/")
    dest.mkdir(parents=True, exist_ok=True)
    for src in data.rglob("*"):
        if not src.is_file():
            continue
        link = dest / src.relative_to(data)
        if link.exists():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, link)
        except OSError:
            shutil.copy2(src, link)
    return root


def _upload_command(repo_id: str, upload_root: Path, repo_type: str) -> list[str]:
    # upload-large-folder: resumable, chunked multi-commit upload built for
    # large folders (the single-shot `upload` warns it "might fail" and has no
    # resume). Uploads upload_root/* to the repo, preserving the mirrored
    # raw_activations/<...> prefix.
    args = [
        "upload-large-folder", repo_id, str(upload_root),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_publish_ref(ref: object, field: str, *, expected_suffix: str | None = None) -> None:
    from urllib.parse import unquote, urlsplit
    expected = {"uri", "sha256", "size"} | ({"generation"} if field != "artifact" else set())
    if not isinstance(ref, dict) or set(ref) != expected:
        raise ValueError(f"{field} has an invalid ArtifactRef shape")
    uri, size, digest = ref.get("uri"), ref.get("size"), ref.get("sha256")
    if not isinstance(uri, str) or not uri or "\\" in uri:
        raise ValueError(f"{field}.uri is unsafe")
    parsed = urlsplit(uri)
    try:
        scheme = PublishScheme(parsed.scheme)
    except ValueError as exc:
        raise ValueError(f"{field}.uri is unsafe") from exc
    if scheme is PublishScheme.BUNDLE and field != "source_route_ref":
        raise ValueError(f"{field}.uri is unsafe")
    if scheme is PublishScheme.BUNDLE and (parsed.netloc or not parsed.path.startswith("/")):
        raise ValueError(f"{field}.uri is unsafe")
    if len(parsed[3]) != 0 or len(parsed[4]) != 0:
        raise ValueError(f"{field}.uri is unsafe")
    decoded_path = unquote(parsed.path)
    parts = ([unquote(parsed.netloc)] if parsed.netloc else []) + decoded_path.split("/")
    if any(part in {".", ".."} for part in parts) or (scheme is PublishScheme.RELATIVE and uri.startswith("/")):
        raise ValueError(f"{field}.uri contains traversal or an absolute path")
    normalized = "/".join(part for part in parts if part)
    if expected_suffix is not None and not normalized.endswith(expected_suffix):
        raise ValueError(f"{field}.uri is not bound to its staged path")
    if not isinstance(size, str) or not size.isascii() or not size.isdecimal() or size.startswith("0"):
        raise ValueError(f"{field}.size is not canonical positive decimal")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field}.sha256 is invalid")
    if field != "artifact" and (not isinstance(ref.get("generation"), str) or not ref["generation"]):
        raise ValueError(f"{field}.generation is invalid")


def _v2_inventory(job_dir: Path) -> tuple[Path, list[Path], list[Path]]:
    data = job_dir / "data"
    if not data.is_dir():
        raise ValueError("v2 job is missing data directory")
    data_resolved = data.resolve()
    files = sorted(path for path in data.rglob("*") if path.is_file())
    if not files:
        raise ValueError("v2 job is empty")
    for path in files:
        relative = path.relative_to(data)
        if path.is_symlink() or not relative.parts or relative.parts[0] != "raw_activations_v2":
            raise ValueError(f"v2 inventory path is outside raw_activations_v2: {relative}")
        resolved = path.resolve()
        if data_resolved not in resolved.parents:
            raise ValueError(f"v2 inventory path escapes staging: {relative}")
    markers = sorted(path for path in files if path.name == "_complete.json")
    if not markers:
        raise ValueError("v2 job has no completion markers")
    expected_files: set[Path] = set()
    for marker in markers:
        expected_files.update({marker, marker.parent / "activations.safetensors",
                               marker.parent / "manifest.json"})
        payload = json.loads(marker.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid v2 completion marker: {marker}")
        expected = {"schema_version", "complete", "kind", "target", "revisions",
                    "support", "target_manifest_ref", "source_route_ref", "artifact",
                    "manifest_sha256"}
        if (set(payload) != expected or payload["schema_version"] != 2
                or payload["complete"] is not True
                or payload["kind"] != "raw_activation_trajectory"):
            raise ValueError(f"invalid v2 completion marker: {marker}")
        target = payload.get("target")
        if not isinstance(target, dict):
            raise ValueError(f"invalid v2 target identity: {marker}")
        required_target = {"target_id", "model", "model_slug", "benchmark", "strategy", "layer", "layer_count"}
        if set(target) != required_target:
            raise ValueError(f"invalid v2 target identity: {marker}")
        slug = target["model_slug"]
        layer = target["layer"]
        if (not isinstance(slug, str) or slug in {".", ".."}
                or any(not (c.isalnum() or c in "._-") for c in slug)
                or isinstance(layer, bool) or not isinstance(layer, int) or layer < 1):
            raise ValueError(f"unsafe v2 target route identity: {marker}")
        target_hash = hashlib.sha256(str(target["target_id"]).encode()).hexdigest()
        expected_parent = Path("raw_activations_v2") / slug / target_hash / str(target["strategy"]) / f"layer_{layer}"
        if marker.parent.relative_to(data) != expected_parent:
            raise ValueError(f"v2 route path does not match marker target: {marker}")
        unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != payload["manifest_sha256"]:
            raise ValueError(f"completion manifest hash mismatch: {marker}")
        artifact_path = marker.parent / "activations.safetensors"
        artifact = payload["artifact"]
        expected_suffix = artifact_path.relative_to(data).as_posix()
        _validate_publish_ref(artifact, "artifact", expected_suffix=expected_suffix)
        _validate_publish_ref(payload["target_manifest_ref"], "target_manifest_ref")
        _validate_publish_ref(payload["source_route_ref"], "source_route_ref")
        if (not artifact_path.is_file() or artifact.get("size") != str(artifact_path.stat().st_size)
                or artifact.get("sha256") != _sha256_file(artifact_path)):
            raise ValueError(f"completion artifact ref does not verify: {marker}")
        route_path = marker.parent / "manifest.json"
        if not route_path.is_file():
            raise ValueError(f"route manifest is missing: {marker}")
        try:
            route = json.loads(route_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"route manifest is invalid JSON: {route_path}") from exc
        route_expected = {"schema_version", "kind", "complete", "target", "revisions",
                          "support", "artifact", "manifest_sha256"}
        if (not isinstance(route, dict) or set(route) != route_expected
                or route.get("schema_version") != 2 or route.get("complete") is not True
                or route.get("kind") != "raw_activation_route"):
            raise ValueError(f"route manifest schema is invalid: {route_path}")
        route_unsigned = {key: value for key, value in route.items() if key != "manifest_sha256"}
        route_canonical = json.dumps(route_unsigned, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(route_canonical).hexdigest() != route["manifest_sha256"]:
            raise ValueError(f"route manifest hash mismatch: {route_path}")
        for field in ("target", "revisions", "support", "artifact"):
            if route[field] != payload[field]:
                raise ValueError(f"route manifest {field} does not match completion marker")
    actual_files = set(files)
    if actual_files != expected_files:
        extras = sorted(path.relative_to(data).as_posix() for path in actual_files - expected_files)
        missing = sorted(path.relative_to(data).as_posix() for path in expected_files - actual_files)
        raise ValueError(f"v2 inventory does not match per-route whitelist extras={extras} missing={missing}")
    nonmarkers = sorted(expected_files - set(markers))
    return data, nonmarkers, markers


def _remote_bytes_hf(repo_id: str, repo_type: str, path: str, revision: str):
    from huggingface_hub import hf_hub_download
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=path, repo_type=repo_type,
                                 revision=revision, force_download=True)
    except Exception as exc:
        response = getattr(exc, "response", None)
        if type(exc).__name__ in {"EntryNotFoundError", "RemoteEntryNotFoundError"} or getattr(response, "status_code", None) == 404:
            return None
        raise
    return Path(cached).read_bytes()


def publish_v2_hf(job_dir: str | Path, repo_id: str, repo_type: str = "dataset", *, api=None) -> None:
    """Create-only two-commit HF publication with byte verification."""
    from huggingface_hub import CommitOperationAdd, HfApi
    job = Path(job_dir)
    data, nonmarkers, markers = _v2_inventory(job)
    api = api or HfApi()

    def publish_phase(paths: list[Path], message: str) -> None:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        pending = []
        for local in paths:
            remote_path = local.relative_to(data).as_posix()
            remote = _remote_bytes_hf(repo_id, repo_type, remote_path, info.sha)
            local_bytes = local.read_bytes()
            if remote is not None:
                if remote != local_bytes:
                    raise FileExistsError(f"immutable remote conflict: {remote_path}")
                continue
            pending.append(CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local)))
        if pending:
            api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=pending,
                              commit_message=message, parent_commit=info.sha)
        verified_revision = api.repo_info(repo_id=repo_id, repo_type=repo_type).sha
        for local in paths:
            remote_path = local.relative_to(data).as_posix()
            remote = _remote_bytes_hf(repo_id, repo_type, remote_path, verified_revision)
            if remote is None or len(remote) != local.stat().st_size or hashlib.sha256(remote).hexdigest() != _sha256_file(local):
                raise RuntimeError(f"remote verification failed: {remote_path}")

    publish_phase(nonmarkers, "Publish immutable raw activations v2 data")
    publish_phase(markers, "Finalize immutable raw activations v2 routes")


def publish_v2_gcs(job_dir: str | Path, repo_id: str, *, bucket=None) -> None:
    """Create-only GCS publication; completion markers are a second phase."""
    from google.cloud import storage
    data, nonmarkers, markers = _v2_inventory(Path(job_dir))
    bucket_name, prefix = _parse_gcs_dest(repo_id, "")
    bucket = bucket or storage.Client().bucket(bucket_name)
    for phase in (nonmarkers, markers):
        for local in phase:
            relative = local.relative_to(data).as_posix()
            name = f"{prefix}/{relative}" if prefix else relative
            blob = bucket.blob(name)
            if blob.exists():
                remote = blob.download_as_bytes()
                if remote != local.read_bytes():
                    raise FileExistsError(f"immutable remote conflict: {name}")
            else:
                blob.metadata = {"sha256": _sha256_file(local), "size": str(local.stat().st_size)}
                blob.upload_from_filename(str(local), if_generation_match=0)
            remote = blob.download_as_bytes()
            if len(remote) != local.stat().st_size or hashlib.sha256(remote).hexdigest() != _sha256_file(local):
                raise RuntimeError(f"remote verification failed: {name}")


def _read_upload_meta(path: Path) -> dict:
    text = path.read_text()
    try:
        meta = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        return {"schema_version": 1, "repo_id": lines[0], "base_in_repo": lines[1],
                "repo_type": lines[2], "job_id": lines[3] if len(lines) > 3 else "",
                "publish_mode": "legacy_large_folder"}
    if not isinstance(meta, dict):
        raise ValueError("upload metadata must be an object")
    return meta


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
    meta = _read_upload_meta(job_dir / ".upload_meta")
    repo_id = meta["repo_id"]
    base_in_repo = meta.get("base_in_repo", "")
    repo_type = meta["repo_type"]
    job_id = meta.get("job_id", "")
    from .commit_rate import acquire_commit_slot
    # Use standard LFS multipart upload; xet has wedged on this uplink.
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    stall_s = float(os.environ.get("WISENT_UPLOAD_STALL_S", str(900.0)))
    data = job_dir / "data"
    log_path = job_dir / ".upload_log"
    is_gcs = _is_gcs_target(repo_id, repo_type)
    if meta.get("publish_mode") == "create_only_two_phase":
        max_attempts = _env_int("WISENT_RAW_V2_UPLOAD_ATTEMPTS", 20)
        terminal_failure = False
        for attempt in range(max_attempts):
            try:
                _append_log(log_path, f"v2_attempt={attempt + 1}")
                if is_gcs:
                    publish_v2_gcs(job_dir, repo_id)
                else:
                    acquire_commit_slot()
                    publish_v2_hf(job_dir, repo_id, repo_type)
            except (ValueError, FileExistsError) as exc:
                terminal_failure = True
                _append_log(log_path, f"v2_publish_terminal {type(exc).__name__}: {exc}")
                break
            except Exception as exc:
                _append_log(log_path, f"v2_publish_retry {type(exc).__name__}: {exc}")
                if attempt + 1 < max_attempts:
                    time.sleep(min(30.0, 2.0 ** attempt))
                    continue
                break
            _mark_uploaded(job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            sweep()
            return 0
        try:
            (job_dir / ".uploading").unlink()
        except OSError:
            pass
        if not terminal_failure:
            sweep()
        return 1
    # Mirror data/ under the repo-path prefix once (idempotent hardlinks) so
    # `hf upload-large-folder` (folder->repo-root, no path-in-repo) lands shards
    # at raw_activations/<...>. GCS path uploads data/ directly, no mirror.
    upload_root = None if is_gcs else _build_upload_root(job_dir, data, base_in_repo)
    _append_log(log_path, f"worker pid={os.getpid()} mode=concurrent_no_pause job_id={job_id} dest={base_in_repo}")
    rc = 1
    for attempt in range(20):
        if _pause_during_extract():
            _wait_for_extractors_to_clear(log_path)
        if not is_gcs:
            # One slot per job: upload-large-folder makes multiple internal
            # commits, so this under-counts the fleet gate. Acceptable on the
            # single-box HF path (throughput << 120/hr cap; HF client backs off
            # on 429). Documented trade-off, not an oversight.
            acquire_commit_slot()
        _append_log(log_path, f"attempt={attempt + 1}")
        if is_gcs:
            ret = _run_gcs_upload(repo_id, data, base_in_repo, log_path, stall_s)
        else:
            ret = _run_upload(
                _upload_command(repo_id, upload_root, repo_type),
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
