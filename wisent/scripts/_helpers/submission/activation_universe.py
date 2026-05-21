"""Activation-extraction Universe adapter for wisent_compute.coverage.

Per (model, task, strategy) tuple yields a UniverseEntry whose:
  - group_key    is safe(model)/task/strategy (state-file key)
  - command      is the wisent.scripts.activations.extract_and_upload
                 invocation that would produce the expected_uri
  - expected_uri is the HF wisent-ai/activations layer_1.safetensors
                 shard URL for that triple

Verifier is URIExistsVerifier seeded with HF_TOKEN.

The class contains no orchestration: walking, retrying, state tracking,
and submit_batch live in wisent_compute.coverage. This file is purely
the consumer-side answer to "what is the expected universe for
activation extraction" and is registered via setup.py entry_points
group wisent_compute.coverage_universes so
wisent_compute.coverage.discover_universes() finds it without
wisent-compute depending on wisent or wisent-tools.
"""
from __future__ import annotations

import os
from typing import Iterator

from wisent_compute.coverage import (
    Universe,
    UniverseEntry,
    URIExistsVerifier,
    Verifier,
)

import json
from pathlib import Path

# Canonical 7-strategy list. Duplicates VALIDATED_STRATEGIES in
# wisent.scripts.activations.extract_and_upload to avoid importing that
# module: as of wisent-tools 0.1.20 (commit 98a483e) extract_and_upload
# has a SyntaxError at line 460 (try-block indentation), so any import
# of it raises and would break entry-point discovery here. Same logic
# inlines load_benchmark_names to avoid pulling in
# wisent.scripts._helpers.submission.submit_top_level_benchmarks which
# transitively imports wisent.core...constants, pulling tensorflow,
# pulling a broken protobuf version. Update both lists together.
VALIDATED_STRATEGIES = [
    "chat_last",
    "chat_mean",
    "chat_first",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
]


def load_benchmark_names() -> list[str]:
    """Return sorted benchmark_tags.json keys.

    Inlined here (instead of imported from submit_top_level_benchmarks)
    so this module has no transitive import chain that touches
    wisent.core.* — the entry-point loader needs to import this at
    `wc-coverage` startup without pulling in the full wisent dep graph.
    """
    candidates: list[Path] = []
    try:
        import wisent as _w
        candidates.append(Path(_w.__file__).parent / "support" / "examples"
                          / "scripts" / "benchmark_tags.json")
    except Exception:
        pass
    candidates += [
        Path("/opt/wisent-agent/.venv/lib/python3.10/site-packages/wisent"
             "/support/examples/scripts/benchmark_tags.json"),
        Path("/usr/local/lib/python3.13/site-packages/wisent/support"
             "/examples/scripts/benchmark_tags.json"),
    ]
    for p in candidates:
        if p.is_file():
            return sorted(json.loads(p.read_text()).keys())
    raise RuntimeError("benchmark_tags.json not found")

HF_REPO = "wisent-ai/activations"
HF_BASE = "https://huggingface.co"


def _safe(model: str) -> str:
    return model.replace("/", "__")


def _shard_uri(model: str, task: str, strategy: str) -> str:
    return (
        f"{HF_BASE}/datasets/{HF_REPO}/resolve/main/"
        f"activations/{_safe(model)}/{task}/{strategy}/layer_1.safetensors"
    )


def _command(model: str, task: str, strategy: str, limit: int, component: str) -> str:
    return (
        f"python3 -m wisent.scripts.activations.extract_and_upload "
        f"--task {task} --model '{model}' --device cuda "
        f"--layers all --strategies {strategy} "
        f"--component {component} --limit {limit}"
    )


class ActivationExtractionUniverse(Universe):
    """One UniverseEntry per (model, task, strategy) of activation extraction.

    Construct with the explicit model list + per-job knobs you want this
    coverage cycle to target. discover_universes() returns this class
    (not an instance); the CLI / caller instantiates with the right
    models and limit for the current cycle.
    """

    def __init__(
        self,
        models: list[str],
        limit: int,
        component: str = "residual_stream",
        priority: int = 0,
    ):
        if not models:
            raise ValueError("ActivationExtractionUniverse needs at least one model")
        self._models = list(models)
        self._limit = int(limit)
        self._component = component
        self._priority = int(priority)

    @property
    def id(self) -> str:
        return "activation-extraction"

    def iter_entries(self) -> Iterator[UniverseEntry]:
        tasks = load_benchmark_names()
        for model in self._models:
            for task in tasks:
                for strategy in VALIDATED_STRATEGIES:
                    yield UniverseEntry(
                        group_key=f"{_safe(model)}/{task}/{strategy}",
                        command=_command(
                            model, task, strategy, self._limit, self._component
                        ),
                        expected_uri=_shard_uri(model, task, strategy),
                        extra={"model": model, "task": task, "strategy": strategy},
                    )

    def verifier(self) -> Verifier:
        return URIExistsVerifier(bearer_token=os.environ.get("HF_TOKEN", ""))

    def submit_kwargs(self) -> dict:
        return {
            "provider": "gcp",
            "preemptible": False,
            "priority": self._priority,
        }
