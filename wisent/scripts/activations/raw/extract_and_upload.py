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
# need (DEFAULT_BATCH_FLOOR + _try_preload_model + generate_pairs) so this
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


def _try_preload_model(model_id: str, device: str):
    """Load the WisentModel once so all layers share it. Returns model or None."""
    try:
        from wisent.core.primitives.models.wisent_model import WisentModel
    except Exception as exc:
        print(f"[preload] WisentModel import failed ({exc}); per-strategy load will be used", flush=True)
        return None
    try:
        print(f"[preload] loading {model_id} on {device} (one-shot)", flush=True)
        return WisentModel(model_id, device=device)
    except Exception as exc:
        print(f"[preload] model load failed ({exc}); per-strategy load will be used", flush=True)
        return None

HF_REPO_ID = "wisent-ai/activations"
HF_REPO_TYPE = "dataset"
RAW_PROMPT_FORMATS = ("chat", "mc_balanced", "role_play")
CHUNK_SIZE = 10000
# Map prompt_format -> upstream ExtractionStrategy enum value. The 5
# chat_* strategies share the same forward pass (raw=True preserves
# tokens), so any chat_* value gives the chat pass. Live failure
# 2026-05-22 on job 0fbc8615: 'chat' is not a valid ExtractionStrategy.
_PF2STRAT = {"chat": "chat_last", "mc_balanced": "mc_balanced", "role_play": "role_play"}


def _safe(model: str) -> str:
    return model.replace("/", "__")


def _raw_hf_path(model: str, task: str, prompt_format: str, layer: int, chunk: int) -> str:
    return (
        f"raw_activations/{_safe(model)}/{task}/{prompt_format}/"
        f"layer_{layer}_chunk_{chunk}.safetensors"
    )


def _run_raw_extraction(
    pairs_file: Path, output_file: Path, model: str, prompt_format: str,
    component: str, device: str, layers: str, cached_model,
) -> None:
    """Call wisent's in-process get-activations API with raw=True so the
    output JSON carries per-token hidden states (not aggregated)."""
    from wisent.core.utils.cli.analysis.analysis.geometry.get_activations import (
        execute_get_activations,
    )
    ns = SimpleNamespace(
        pairs_file=str(pairs_file),
        output=str(output_file),
        model=model,
        device=device,
        layers=layers,
        extraction_strategy=_PF2STRAT[prompt_format],
        extraction_component=component,
        batch_size=DEFAULT_BATCH_FLOOR,
        verbose=False,
        timing=False,
        raw=True,
        cached_model=cached_model,
    )
    execute_get_activations(ns)


def _collect_per_layer(out_doc: dict) -> dict:
    """Group activations by layer index. Each list element is
    (pair_id, pos_tensor_like, neg_tensor_like). pair_id is the
    explicit field if present, else the positional index in pairs."""
    pairs = out_doc.get("pairs", [])
    by_layer: dict = {}
    for idx, p in enumerate(pairs):
        pid = int(p.get("contrastive_pair_id") or p.get("pair_id") or idx)
        pos = p.get("positive_response", {}).get("layers_activations", {}) or {}
        neg = p.get("negative_response", {}).get("layers_activations", {}) or {}
        for layer_str, pos_vec in pos.items():
            try:
                layer = int(layer_str)
            except (TypeError, ValueError):
                continue
            neg_vec = neg.get(layer_str)
            if neg_vec is None:
                continue
            by_layer.setdefault(layer, []).append((pid, pos_vec, neg_vec))
    return by_layer


def _save_layer_chunk(out_path: Path, chunk: list) -> None:
    """Write one safetensors file with {pos_<pid>, neg_<pid>} keys +
    pair_ids metadata, matching migrate_raw.py's writer."""
    import torch
    from safetensors.torch import save_file
    tensors: dict = {}
    pids: list = []
    for pid, pos_vec, neg_vec in chunk:
        tensors[f"pos_{pid}"] = torch.as_tensor(pos_vec, dtype=torch.float32)
        tensors[f"neg_{pid}"] = torch.as_tensor(neg_vec, dtype=torch.float32)
        pids.append(pid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"pair_ids": json.dumps(pids)})


def _upload_staging(
    staging: Path, model: str, task: str, prompt_format: str, n_layers: int,
) -> None:
    """Upload the raw_activations/<safe>/<task>/<prompt_format>/ subtree
    in one commit. HfApi.upload_folder is the same call migrate_raw.py
    uses; on 429 the script raises and the verify_command's curl HEAD
    will surface the missing shard on the next coverage cycle."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.upload_folder(
        folder_path=str(staging),
        path_in_repo=".",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message=(
            f"raw activations: {model}/{task}/{prompt_format} "
            f"({n_layers} layers)"
        ),
    )


def _shard_raw_output(out_file: Path, model: str, task: str, prompt_format: str) -> int:
    """Convert the extractor's JSON output to the migrate_raw.py
    safetensors layout, upload to HF, return n_layers uploaded."""
    with open(out_file) as f:
        doc = json.load(f)
    by_layer = _collect_per_layer(doc)
    if not by_layer:
        raise RuntimeError(
            f"raw extraction produced no per-layer activations for "
            f"({model}, {task}, {prompt_format}). Check that "
            f"execute_get_activations(raw=True) returns "
            f"layers_activations in the output JSON."
        )
    staging = Path(tempfile.mkdtemp(prefix="wisent_raw_stage_"))
    try:
        for layer, entries in sorted(by_layer.items()):
            for ci in range(0, len(entries), CHUNK_SIZE):
                chunk = entries[ci:ci + CHUNK_SIZE]
                out_path = staging / _raw_hf_path(
                    model, task, prompt_format, layer, ci // CHUNK_SIZE,
                )
                _save_layer_chunk(out_path, chunk)
        _upload_staging(staging, model, task, prompt_format, len(by_layer))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(by_layer)


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
    out_file = work_dir / f"{args.task}__{args.prompt_format}__raw.json"
    print(
        f"[{args.task}/{args.prompt_format}] raw_extract_and_upload start "
        f"model={args.model} layers={args.layers}",
        flush=True,
    )
    _strip_broken_torchvision()
    try:
        generate_pairs(args.task, pairs_file, limit=args.limit)
        cached = _try_preload_model(args.model, args.device)
        _run_raw_extraction(
            pairs_file=pairs_file, output_file=out_file,
            model=args.model, prompt_format=args.prompt_format,
            component=args.component, device=args.device,
            layers=args.layers, cached_model=cached,
        )
        n_layers = _shard_raw_output(
            out_file, args.model, args.task, args.prompt_format,
        )
        print(
            f"[{args.task}/{args.prompt_format}] uploaded {n_layers} layers "
            f"to raw_activations/{_safe(args.model)}/{args.task}/{args.prompt_format}/",
            flush=True,
        )
    finally:
        if owns:
            shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
