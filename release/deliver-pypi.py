#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import tempfile


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing {name}")
    return value

archive = pathlib.Path(required("WISENT_RELEASE_ARCHIVE"))
digest = required("WISENT_RELEASE_SHA256")
if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
    raise RuntimeError("canonical Stado archive digest mismatch")
with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [m for m in bundle.getmembers() if m.isfile() and m.name.startswith("python-distributions/")]
        bundle.extractall(root, members=members, filter="data")
    artifacts = sorted(str(path) for path in (root / "python-distributions").iterdir())
    if not artifacts:
        raise RuntimeError("canonical archive contains no Python distributions")
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = required("PYPI_TOKEN")
    completed = subprocess.run(["python3", "-m", "twine", "upload", "--non-interactive", *artifacts], check=True, capture_output=True, text=True, env=env)
receipt = {
    "schema_version": 1,
    "channel": "pypi",
    "product": required("WISENT_PRODUCT"),
    "version": required("WISENT_VERSION"),
    "release_uri": required("WISENT_RELEASE_URI"),
    "release_sha256": digest,
    "provider_output": completed.stdout.strip(),
}
out = pathlib.Path(required("WISENT_OUTPUT_DIR"))
out.mkdir(parents=True, exist_ok=True)
(out / "pypi-receipt.json").write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
