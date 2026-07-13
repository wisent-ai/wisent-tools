import json
from pathlib import Path

import pytest

from wisent.scripts.activations.raw import upload_worker
from test_raw_writer_v2 import write_bundle


class FakeInfo:
    def __init__(self, sha):
        self.sha = sha


class FakeHfApi:
    def __init__(self, fail_after_apply_on=None):
        self.remote = {}
        self.commits = []
        self.calls = 0
        self.fail_after_apply_on = fail_after_apply_on

    def repo_info(self, *, repo_id, repo_type):
        return FakeInfo(f"fake-revision-{len(self.commits)}")

    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit):
        self.calls += 1
        paths = []
        for operation in operations:
            remote_path = operation.path_in_repo
            self.remote[remote_path] = Path(operation.path_or_fileobj).read_bytes()
            paths.append(remote_path)
        self.commits.append({"message": commit_message, "paths": paths, "parent": parent_commit})
        if self.calls == self.fail_after_apply_on:
            raise RuntimeError(f"simulated interruption after phase {self.calls}")


def staged_job(tmp_path):
    job = tmp_path / "job"
    write_bundle(job / "data")
    return job


def bind_fake_remote(monkeypatch, api):
    monkeypatch.setattr(
        upload_worker,
        "_remote_bytes_hf",
        lambda repo_id, repo_type, path, revision: api.remote.get(path),
    )


def test_hf_publish_is_two_phase_complete_last_and_idempotent(tmp_path, monkeypatch):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)

    upload_worker.publish_v2_hf(job, "fake/repo", api=api)

    assert len(api.commits) == 2
    first, second = api.commits
    assert first["message"] == "Publish immutable raw activations v2 data"
    assert second["message"] == "Finalize immutable raw activations v2 routes"
    assert first["paths"] and all(not path.endswith("/_complete.json") for path in first["paths"])
    assert second["paths"] and all(path.endswith("/_complete.json") for path in second["paths"])
    assert not set(first["paths"]) & set(second["paths"])
    assert all(path.startswith("raw_activations_v2/") for path in first["paths"] + second["paths"])

    before = list(api.commits)
    upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert api.commits == before


@pytest.mark.parametrize("failed_phase", [1, 2])
def test_hf_publish_resumes_after_interrupted_phase_without_overwrite(tmp_path, monkeypatch, failed_phase):
    job = staged_job(tmp_path)
    api = FakeHfApi(fail_after_apply_on=failed_phase)
    bind_fake_remote(monkeypatch, api)

    with pytest.raises(RuntimeError, match=f"phase {failed_phase}"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    uploaded_before_resume = dict(api.remote)
    api.fail_after_apply_on = None
    upload_worker.publish_v2_hf(job, "fake/repo", api=api)

    for path, content in uploaded_before_resume.items():
        assert api.remote[path] == content
    assert any(path.endswith("/_complete.json") for path in api.remote)
    assert all(not path.endswith("/_complete.json") for path in api.commits[0]["paths"])
    assert all(path.endswith("/_complete.json") for path in api.commits[-1]["paths"])


def test_hf_publish_rejects_existing_conflicting_byte(tmp_path, monkeypatch):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    data_file = next(path for path in (job / "data").rglob("activations.safetensors"))
    relative = data_file.relative_to(job / "data").as_posix()
    api.remote[relative] = b"different immutable content"

    with pytest.raises(FileExistsError, match="immutable remote conflict"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits
    assert not any(path.endswith("/_complete.json") for path in api.remote)


def test_hf_publish_verifies_every_ref_before_first_upload(tmp_path, monkeypatch):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    completion_path = next((job / "data").rglob("_complete.json"))
    completion_payload = json.loads(completion_path.read_text())
    completion_payload["artifact"]["sha256"] = "0" * 64
    unhashed_payload = {key: value for key, value in completion_payload.items() if key != "manifest_sha256"}
    completion_payload.update(manifest_sha256=upload_worker.hashlib.sha256(
        json.dumps(unhashed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest())
    completion_path.write_text(json.dumps(completion_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    with pytest.raises(ValueError, match="artifact ref does not verify"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits and not api.remote


def test_hf_publish_rejects_completion_path_traversal_and_partial_schema(tmp_path, monkeypatch):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    completion_path = next((job / "data").rglob("_complete.json"))
    completion_payload = json.loads(completion_path.read_text())
    completion_payload["artifact"]["uri"] = "../escape.safetensors"
    unhashed_payload = {key: value for key, value in completion_payload.items() if key != "manifest_sha256"}
    completion_payload.update(manifest_sha256=upload_worker.hashlib.sha256(
        json.dumps(unhashed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest())
    completion_path.write_text(json.dumps(completion_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    with pytest.raises(ValueError, match="traversal|URI|relative|artifact"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits

    job = staged_job(tmp_path / "partial")
    completion_path = next((job / "data").rglob("_complete.json"))
    completion_payload = json.loads(completion_path.read_text())
    completion_payload.pop("source_route_ref")
    completion_path.write_text(json.dumps(completion_payload))
    with pytest.raises(ValueError, match="invalid v2 completion"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits


@pytest.mark.parametrize("ref_name", ["artifact", "target_manifest_ref", "source_route_ref"])
@pytest.mark.parametrize("suffix", ["?x", "#x"])
def test_hf_publish_rejects_query_or_fragment_in_every_canonical_ref(tmp_path, monkeypatch, ref_name, suffix):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    completion_path = next((job / "data").rglob("_complete.json"))
    completion_payload = json.loads(completion_path.read_text())
    completion_payload[ref_name]["uri"] += suffix
    unhashed_payload = {key: value for key, value in completion_payload.items() if key != "manifest_sha256"}
    completion_payload.update(manifest_sha256=upload_worker.hashlib.sha256(
        json.dumps(unhashed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest())
    rewrite_canonical(completion_path, completion_payload)
    with pytest.raises(ValueError, match=f"{ref_name}.*unsafe"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits and not api.remote


def rewrite_canonical(path, value):
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


@pytest.mark.parametrize("corruption", ["malformed", "self_hash", "target", "revision", "support", "artifact"])
def test_inventory_rejects_corrupt_or_cross_bound_route_before_any_completion_upload(tmp_path, monkeypatch, corruption):
    job = staged_job(tmp_path)
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    route_path = next((job / "data").rglob("manifest.json"))
    if corruption == "malformed":
        route_path.write_text("{not-json")
    else:
        route = json.loads(route_path.read_text())
        if corruption == "self_hash":
            route["manifest_sha256"] = "0" * 64
        elif corruption == "target":
            route["target"]["target_id"] = "other-target"
        elif corruption == "revision":
            route["revisions"]["model"] = "9" * 40
        elif corruption == "support":
            route["support"][0]["stable_id"] = "other-stable"
        elif corruption == "artifact":
            route["artifact"]["sha256"] = "0" * 64
        if corruption != "self_hash":
            unsigned = {key: value for key, value in route.items() if key != "manifest_sha256"}
            route["manifest_sha256"] = upload_worker.hashlib.sha256(
                json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
        rewrite_canonical(route_path, route)
    with pytest.raises((ValueError, json.JSONDecodeError), match="route|manifest|target|revision|support|artifact|JSON|Expecting"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits and not api.remote


@pytest.mark.parametrize("leftover", ["legacy.safetensors", "unrelated.txt", "raw_activations/old/_complete.json"])
def test_inventory_rejects_non_v2_prefix_and_unwhitelisted_leftovers(tmp_path, monkeypatch, leftover):
    job = staged_job(tmp_path)
    path = job / "data" / leftover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"must never upload")
    api = FakeHfApi()
    bind_fake_remote(monkeypatch, api)
    with pytest.raises(ValueError, match="prefix|whitelist|unexpected|unrelated|raw_activations_v2"):
        upload_worker.publish_v2_hf(job, "fake/repo", api=api)
    assert not api.commits and not api.remote


def make_worker_meta(job):
    (job / ".upload_meta").write_text(json.dumps({
        "schema_version": 2, "repo_id": "fake/repo", "base_in_repo": "raw_activations_v2",
        "repo_type": "dataset", "job_id": "job-1", "publish_mode": "create_only_two_phase",
    }))


def patch_worker_runtime(monkeypatch):
    from wisent.scripts.activations.raw import commit_rate
    monkeypatch.setattr(upload_worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(commit_rate, "acquire_commit_slot", lambda *args: None)
    monkeypatch.setattr(upload_worker, "_mark_uploaded", lambda job_id: None)
    monkeypatch.setattr(upload_worker, "sweep", lambda: 0)
    monkeypatch.setattr(upload_worker.time, "sleep", lambda seconds: None)


@pytest.mark.parametrize("failures", [1, 2])
def test_worker_retries_transient_phase_or_verification_failure_and_resumes(tmp_path, monkeypatch, failures):
    job = staged_job(tmp_path)
    make_worker_meta(job)
    patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("WISENT_RAW_V2_UPLOAD_ATTEMPTS", "3")
    calls = []
    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= failures:
            raise RuntimeError("transient phase/verification failure")
    monkeypatch.setattr(upload_worker, "publish_v2_hf", flaky)
    assert upload_worker.run_worker(str(job)) == 0
    assert len(calls) == failures + 1
    assert not job.exists()


def test_worker_bounded_retry_exhaustion_is_terminal_and_preserves_job(tmp_path, monkeypatch):
    job = staged_job(tmp_path)
    make_worker_meta(job)
    patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("WISENT_RAW_V2_UPLOAD_ATTEMPTS", "3")
    calls = []
    def always_fails(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("persistent failure")
    monkeypatch.setattr(upload_worker, "publish_v2_hf", always_fails)
    assert upload_worker.run_worker(str(job)) == 1
    assert len(calls) == 3
    assert job.is_dir()
    assert not (job / ".uploading").exists()
    assert "persistent failure" in (job / ".upload_log").read_text()
