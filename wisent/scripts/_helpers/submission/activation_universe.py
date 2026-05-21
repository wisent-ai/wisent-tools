"""Activation-extraction Universe adapter for wisent_compute.coverage.

CANONICAL DESIGN — RAW THEN TRANSFORM:

Per (model, task) we extract exactly **3 raw activation forward passes**:
  - chat         (raw activations for the chat-template prompt format)
  - mc_balanced  (raw activations for the multiple-choice prompt format)
  - role_play    (raw activations for the role-play prompt format)

The 7 "strategies" (chat_last / chat_mean / chat_first / chat_max_norm /
chat_weighted / mc_balanced / role_play) are post-hoc aggregations
derived from those 3 raw shards. Specifically chat_{last, mean, first,
max_norm, weighted} are 5 different per-token aggregations of the SAME
raw chat forward pass; mc_balanced and role_play map 1:1 to their raw
forward passes.

Per (model, task, prompt_format) yields a UniverseEntry whose:
  - group_key    is safe(model)/task/prompt_format (state-file key)
  - command      is the raw extraction invocation that would produce
                 the expected_uri
  - expected_uri is gs/HF raw_activations layer_1_chunk_0.safetensors
  - extra        carries the verify_command HEADing expected_uri

The raw_activations/<safe>/<task>/<prompt_format>/layer_<L>_chunk_<C>
layout is the one migrate_raw.py populates from the Supabase→HF
migration; new extractions should target the same layout instead of
the redundant 7-strategy activations/<safe>/<task>/<strategy>/ tree
that extract_and_upload currently writes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from wisent_compute.coverage import (
    Universe,
    UniverseEntry,
    URIExistsVerifier,
    Verifier,
)

# Canonical 3 raw forward-pass prompt formats. The 7 VALIDATED_STRATEGIES
# are downstream aggregations of these — chat_{last,mean,first,max_norm,
# weighted} all derive from raw `chat`, mc_balanced is raw `mc_balanced`,
# role_play is raw `role_play`. Stored at raw_activations/<safe>/<task>/
# <prompt_format>/layer_<L>_chunk_<C>.safetensors per migrate_raw.py
# convention.
RAW_PROMPT_FORMATS = ["chat", "mc_balanced", "role_play"]

HF_REPO = "wisent-ai/activations"
HF_BASE = "https://huggingface.co"
RAW_LAYER = 1
RAW_CHUNK = 0


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


def _safe(model: str) -> str:
    return model.replace("/", "__")


def _raw_shard_uri(model: str, task: str, prompt_format: str) -> str:
    """raw_activations/<safe>/<task>/<prompt_format>/layer_1_chunk_0.safetensors
    URL on HF wisent-ai/activations — the canonical raw-forward-pass shard.
    """
    return (
        f"{HF_BASE}/datasets/{HF_REPO}/resolve/main/"
        f"raw_activations/{_safe(model)}/{task}/{prompt_format}/"
        f"layer_{RAW_LAYER}_chunk_{RAW_CHUNK}.safetensors"
    )


def _command(model: str, task: str, prompt_format: str, limit: int, component: str) -> str:
    """Raw-extraction command. Targets the raw forward pass for one
    prompt_format and writes raw_activations/... (not the legacy 7-strategy
    activations/... tree). extract_and_upload accepts --strategies; the
    raw-mode invocation passes the prompt_format as the strategy name
    (chat / mc_balanced / role_play) and relies on the script's
    raw-output mode. If the installed extract_and_upload still writes the
    aggregated activations/<strategy>/ layout, this command produces no
    expected_uri match and the verifier reports the entry MISSING — that
    is the truth the agreed design expects to surface."""
    return (
        f"python3 -m wisent.scripts.activations.extract_and_upload "
        f"--task {task} --model '{model}' --device cuda "
        f"--layers all --strategies {prompt_format} --raw "
        f"--component {component} --limit {limit}"
    )


def _verify_command(model: str, task: str, prompt_format: str) -> str:
    """Post-run HEAD against the expected raw_activations URI. A 404
    routes the job to failed/ instead of completed/, closing the bug
    that let exit=0 mean 'I uploaded nothing' (job 00f34aa3, 2026-05-20)."""
    uri = _raw_shard_uri(model, task, prompt_format)
    return (
        f'curl --fail --silent --show-error --head -o /dev/null '
        f'-H "Authorization: Bearer ${{HF_TOKEN}}" "{uri}"'
    )


class ActivationExtractionUniverse(Universe):
    """Yields one UniverseEntry per (model, task, prompt_format) raw
    forward pass. 3 entries per (model, task), not 7.

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
                for prompt_format in RAW_PROMPT_FORMATS:
                    yield UniverseEntry(
                        group_key=f"{_safe(model)}/{task}/{prompt_format}",
                        command=_command(
                            model, task, prompt_format, self._limit, self._component
                        ),
                        expected_uri=_raw_shard_uri(model, task, prompt_format),
                        extra={
                            "model": model,
                            "task": task,
                            "prompt_format": prompt_format,
                            "verify_command": _verify_command(model, task, prompt_format),
                        },
                    )

    def verifier(self) -> Verifier:
        return URIExistsVerifier(bearer_token=os.environ.get("HF_TOKEN", ""))

    def submit_kwargs(self) -> dict:
        return {
            "provider": "gcp",
            "preemptible": False,
            "priority": self._priority,
        }
