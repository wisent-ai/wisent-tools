#!/usr/bin/env python3
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
out = root / "dist"
shutil.rmtree(out, ignore_errors=True)
subprocess.run(["python3", "-m", "build", "--sdist", "--wheel", "--outdir", str(out)], cwd=root, check=True)
artifacts = sorted(path for path in out.iterdir() if path.suffix in {".whl", ".gz"})
if len(artifacts) != 2:
    raise RuntimeError("Python build must produce exactly one wheel and one source distribution")
