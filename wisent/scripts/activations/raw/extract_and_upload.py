"""Raw-mode activation extractor: 1 raw forward pass per (model, task, prompt_format).

Writes the canonical raw_activations/<safe>/<task>/<prompt_format>/
layer_<L>_chunk_<C>.safetensors layout that migrate_raw.py uses,
NOT the legacy 7-strategy activations/<safe>/<task>/<strategy>/ tree
that wisent.scripts.activations.extract_and_upload writes.

The agreed pipeline: extract 3 raw forward passes (chat / mc_balanced /
role_play) per (model, task), and derive the 7 chat_*/mc_balanced/role_play
aggregations on the fly at consumer time. The 5 chat_* aggregations
(last, mean, first, max_norm, weighted) all come from the same raw
chat forward pass, so storing them as separate shards multiplies HF
storage and HF API surface 5x for no information gain.

Usage (the activation-extraction Universe in wisent-tools emits this exact
command shape):
    python3 -m wisent.scripts.activations.raw.extract_and_upload \\
        --task <task> --model '<model>' --device cuda --layers all \\
        --prompt-format chat --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Helpers inlined to avoid `from wisent.scripts.activations.extract_and_upload`
# — that module has a SyntaxError at line 460 in the published wheel (try/except
# block indentation bug) and any import of it raises before this module can
# load. Confirmed live 2026-05-21 on job 4394727f: the agent attempted to
# import raw.extract_and_upload, the chain hit the broken parent, and the job
# failed at runpy with the line-460 SyntaxError. Inline the three names we
# need (DEFAULT_BATCH_FLOOR + _preload_model + generate_pairs) so this
# module has zero dependency on the broken parent.
import shutil as _shutil
import subprocess as _subprocess

DEFAULT_BATCH_FLOOR = 8


def _wisent_bin() -> str:
    found = _shutil.which("wisent")
    if not found:
        raise SystemExit("wisent CLI not found on PATH")
    return found


def _strip_broken_torchvision() -> None:
    """Uninstall torchvision when present + incompatible with installed
    torch. wisent CLI -> transformers -> torchvision top-level import
    fails at `torch.library.register_fake("torchvision::nms")` with
    `RuntimeError: operator torchvision::nms does not exist` on agents
    where torchvision was built against a different torch ABI
    (confirmed live 2026-05-21 on 18+ raw jobs). Text/audio/robotics
    extraction does not need torchvision."""
    try:
        import torchvision  # noqa: F401
    except Exception:
        return
    try:
        import torch
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except Exception:
        pass
    print("[raw] torchvision/torch ABI mismatch; uninstalling torchvision", flush=True)
    _subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision",
         "--break-system-packages"],
        capture_output=True,
    )


def generate_pairs(task: str, out_path: Path, limit=None) -> None:
    cmd = [_wisent_bin(), "generate-pairs-from-task", task, "--output", str(out_path)]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    print(f"[pairs] {' '.join(cmd)}", flush=True)
    result = _subprocess.run(cmd)
    if result.returncode != 0 or not out_path.is_file():
        raise SystemExit(f"pair_texts generation failed for {task} (rc={result.returncode})")


def _preload_model(model_id: str, device: str):
    """Load the WisentModel once so all layers share it. Fail fast: the
    per-token path REQUIRES a live model (collect_raw reads
    collector.model.tokenizer), so a None here would surface only as a
    cryptic 'NoneType has no attribute tokenizer' mid-extraction (live
    failure 2026-05-24, job 65743d96). Raise with the real cause instead."""
    from wisent.core.primitives.models.wisent_model import WisentModel
    print(f"[preload] loading {model_id} on {device} (one-shot)", flush=True)
    return WisentModel(model_id, device=device)

HF_REPO_ID = "wisent-ai/activations"
HF_REPO_TYPE = "dataset"
RAW_PROMPT_FORMATS = ("chat", "mc_balanced", "role_play")
PAIR_CHUNK_SIZE = 25
ARCH_MODULE_LIMIT = 500
_PF2STRAT = {"chat": "chat_last", "mc_balanced": "mc_balanced", "role_play": "role_play"}


def _safe(model: str) -> str:
    return model.replace("/", "__")


def _build_collector(cached_model):
    from wisent.core.primitives.model_interface.core.activations.hooks.activations_collector import (
        ActivationCollector,
    )
    return ActivationCollector(
        model=cached_model,
        architecture_module_limit=ARCH_MODULE_LIMIT,
        store_device="cpu",
    )


def _build_pair_from_dict(d: dict):
    from wisent.core.primitives.contrastive_pairs import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import (
        PositiveResponse, NegativeResponse,
    )
    pos = d["positive"] if "positive" in d else d.get(
        "positive_response", {}).get("model_response", "")
    neg = d["negative"] if "negative" in d else d.get(
        "negative_response", {}).get("model_response", "")
    return ContrastivePair(
        prompt=d["prompt"],
        positive_response=PositiveResponse(model_response=pos),
        negative_response=NegativeResponse(model_response=neg),
    )


def _save_chunk_per_token(out_path: Path, pos_list, neg_list, pids) -> None:
    """Per-token chunk writer: tensors are [seq_len, hidden_dim]."""
    import torch
    from safetensors.torch import save_file
    tensors: dict = {}
    for i, pid in enumerate(pids):
        tensors[f"pos_{pid}"] = pos_list[i].to(torch.float32).contiguous()
        tensors[f"neg_{pid}"] = neg_list[i].to(torch.float32).contiguous()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"pair_ids": json.dumps(pids)})


def _stream_extract_to_safetensors(
    pairs_file: Path, staging: Path, model: str, task: str,
    prompt_format: str, component: str, device: str, layers: str,
    cached_model,
) -> int:
    """Per-token streaming extraction via collect_raw — no JSON intermediate.
    Each layer chunk file holds [seq_len, hidden_dim] tensors per pair so
    consumers can derive any aggregation (chat_last/mean/first/max_norm/
    weighted) without re-running the model. Memory bounded by
    PAIR_CHUNK_SIZE × 2 × n_layers × seq_len × hidden_dim × 4 bytes."""
    from wisent.core.primitives.model_interface.core.activations import (
        ExtractionStrategy, ExtractionComponent,
    )
    from wisent.core.primitives.model_interface.core.activations.pipeline.raw_collector import (
        collect_raw,
    )
    strategy = ExtractionStrategy(_PF2STRAT[prompt_format])
    comp = (ExtractionComponent.RESIDUAL_STREAM
            if component == "residual_stream"
            else ExtractionComponent.default())

    with open(pairs_file, "r") as f:
        doc = json.load(f)
    pairs_list = doc.get("pairs", doc) if isinstance(doc, dict) else doc
    if not pairs_list:
        raise SystemExit(f"[{task}/{prompt_format}] empty pairs file")

    collector = _build_collector(cached_model)
    base = staging / "raw_activations" / _safe(model) / task / prompt_format
    n_layers_seen = 0

    for chunk_idx, start in enumerate(range(0, len(pairs_list), PAIR_CHUNK_SIZE)):
        chunk_pairs = pairs_list[start: start + PAIR_CHUNK_SIZE]
        per_layer_pos: dict = {}
        per_layer_neg: dict = {}
        pids: list = []
        for idx, p in enumerate(chunk_pairs):
            pid = int(p.get("contrastive_pair_id") or p.get("pair_id") or (start + idx))
            pids.append(pid)
            raw = collect_raw(collector, _build_pair_from_dict(p),
                              strategy=strategy, layers=None, component=comp)
            for layer_name, t in raw["pos_hidden_states"].items():
                per_layer_pos.setdefault(layer_name, []).append(t)
            for layer_name, t in raw["neg_hidden_states"].items():
                per_layer_neg.setdefault(layer_name, []).append(t)

        written: list = []
        for layer_name in per_layer_pos:
            try:
                layer_int = int(layer_name)
            except ValueError:
                continue
            out_path = base / f"layer_{layer_int}_chunk_{chunk_idx}.safetensors"
            _save_chunk_per_token(
                out_path, per_layer_pos[layer_name], per_layer_neg[layer_name], pids,
            )
            written.append(out_path)
        n_layers_seen = max(n_layers_seen, len(per_layer_pos))
        del per_layer_pos, per_layer_neg
        # Durable per-chunk commit, gated on the fleet-wide 128/hr bucket.
        _commit_files(written, model, task, prompt_format, f"chunk_{chunk_idx}")
        for p in written:
            p.unlink(missing_ok=True)
        print(f"[{task}/{prompt_format}] chunk {chunk_idx} uploaded", flush=True)
    n_chunks = (len(pairs_list) + PAIR_CHUNK_SIZE - 1) // PAIR_CHUNK_SIZE
    marker = base / "_complete.json"
    marker.write_text(json.dumps(
        {"n_chunks": n_chunks, "n_layers": n_layers_seen, "n_pairs": len(pairs_list)}
    ))
    _commit_files([marker], model, task, prompt_format, "complete")
    marker.unlink(missing_ok=True)
    return n_layers_seen


def _commit_files(files, model: str, task: str, prompt_format: str,
                  label: str) -> None:
    """One HF commit, gated by the in-package rolling-window commit
    counter (mandatory: reserves a slot in the shared 1-hr window before
    committing, blocks when the fleet is at the cap, raises on timeout so
    the job retries rather than committing ungated)."""
    if not files:
        return
    from .commit_rate import acquire_commit_slot
    acquire_commit_slot()
    # Upload via the `hf` CLI in a clean subprocess: in-process create_commit
    # segfaults CPython 3.13 in this CUDA/torch process (dmesg-confirmed).
    import subprocess
    base_in_repo = f"raw_activations/{_safe(model)}/{task}/{prompt_format}"
    local_dir = str(files[0].parent)
    r = subprocess.run(
        ["hf", "upload", HF_REPO_ID, local_dir, base_in_repo,
         "--repo-type", HF_REPO_TYPE],
        env={**os.environ, "HF_HUB_DISABLE_XET": "1"},
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"hf upload failed rc={r.returncode}: {r.stderr[-400:]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-format", required=True, choices=RAW_PROMPT_FORMATS,
                   dest="prompt_format")
    p.add_argument("--component", default="residual_stream")
    p.add_argument("--device", required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--work-dir", default=None)
    args = p.parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        owns = False
    else:
        safe_task = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.task)[:64]
        work_dir = Path(tempfile.mkdtemp(
            prefix=f"wisent_raw_{safe_task}_pid{os.getpid()}_"
        ))
        owns = True

    pairs_file = work_dir / f"{args.task}__pairs.json"
    print(f"[{args.task}/{args.prompt_format}] start model={args.model}", flush=True)
    _strip_broken_torchvision()
    staging = Path(tempfile.mkdtemp(prefix="wisent_raw_stage_"))
    try:
        generate_pairs(args.task, pairs_file, limit=args.limit)
        cached = _preload_model(args.model, args.device)
        n_layers = _stream_extract_to_safetensors(
            pairs_file=pairs_file, staging=staging,
            model=args.model, task=args.task,
            prompt_format=args.prompt_format, component=args.component,
            device=args.device, layers=args.layers, cached_model=cached,
        )
        print(
            f"[{args.task}/{args.prompt_format}] uploaded {n_layers} layers",
            flush=True,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if owns:
            shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
