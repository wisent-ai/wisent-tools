"""Accurate fleet-wide HF commit-rate counter.

A true rolling-1-hour window of commit timestamps in one GCS object,
CAS-guarded so every agent shares the same count. Unlike a token bucket
(which only approximates a rate and only sees commits routed through it),
this stores the actual timestamps of commits and counts how many fall in
the trailing WINDOW_S — an exact "commits in the last hour" number.

Lives in wisent-tools, next to the extractor, so it ships WITH the job
code: every job running this extractor version routes its commits through
this same counter regardless of the agent's wisent-compute version. That
removes the ungated-bypass that desynced the old client bucket.

Mandatory gate: a commit only proceeds after acquire_commit_slot()
reserves a slot. If no slot frees within the timeout it raises, so the
job fails-and-retries rather than committing ungated (which would
429-storm the repo and desync everyone).
"""
from __future__ import annotations

import json
import os
import random
import time

_BUCKET = os.environ.get("WC_BUCKET", "wisent-compute")
_OBJECT = "hf_rate/commit_log.json"
_WINDOW_S = 3600
# Margin below HF's documented 128/hour so window-boundary fuzz and any
# out-of-band commit don't tip us over the real server-side limit.
_CAP = 120
_MAX_SLEEP_S = 60.0


def _blob():
    from google.cloud import storage
    return storage.Client().bucket(_BUCKET).blob(_OBJECT)


def acquire_commit_slot(timeout_s: float = 7200.0) -> None:
    """Reserve one commit slot in the rolling window, blocking until the
    fleet is under the cap. Raises TimeoutError if none frees in time
    (caller must NOT commit on raise)."""
    blob = _blob()
    deadline = time.time() + timeout_s
    while True:
        now = time.time()
        try:
            if blob.exists():
                blob.reload()
                generation = blob.generation
                stamps = json.loads(blob.download_as_text())
            else:
                generation = 0
                stamps = []
            stamps = [t for t in stamps if now - float(t) < _WINDOW_S]
            if len(stamps) < _CAP:
                stamps.append(now)
                blob.upload_from_string(
                    json.dumps(stamps),
                    if_generation_match=generation,
                )
                return
            # At the cap: wait until the oldest stamp leaves the window.
            sleep = max(1.0, _WINDOW_S - (now - min(float(t) for t in stamps)))
        except Exception:
            # CAS race (another agent wrote first) or transient GCS error:
            # short jittered retry. Never proceed to commit on error.
            sleep = random.uniform(1.0, 5.0)
        if time.time() >= deadline:
            raise TimeoutError(
                f"commit slot not acquired within {timeout_s:.0f}s "
                f"(fleet at HF commit cap)"
            )
        time.sleep(min(_MAX_SLEEP_S, sleep) + random.uniform(0.0, 1.0))
