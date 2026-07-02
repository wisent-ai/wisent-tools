"""Raw-mode activation extractor: 1 raw forward pass per (model, task, prompt_format).

Writes the canonical raw_activations/<safe>/<task>/<prompt_format>/
layer_<L>_chunk_<C>.safetensors layout that migrate_raw.py uses,
NOT the removed foreground-upload activation tree.

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

# Helpers inlined to keep raw extraction independent of the removed
# foreground uploader compatibility stub. Inline the three names we need
# (DEFAULT_BATCH_FLOOR + _preload_model + generate_pairs) so this module
# has zero dependency on that old entrypoint.
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


def _upload_target() -> tuple[str, str]:
    backend = os.environ.get("WISENT_RAW_UPLOAD_BACKEND", "hf").strip().lower()
    if backend == "gcs":
        uri = os.environ.get("WISENT_RAW_GCS_URI", "").strip()
        if not uri.startswith("gs://"):
            raise SystemExit("WISENT_RAW_UPLOAD_BACKEND=gcs requires WISENT_RAW_GCS_URI=gs://...")
        return uri, "gcs"
    return HF_REPO_ID, HF_REPO_TYPE


def _configure_hf_cache() -> None:
    """Use the large persistent disk for HF cache on the local RTX box."""
    if os.environ.get("HF_HOME"):
        return
    base = Path("/mnt/wd16tb/hf_cache")
    if not base.parent.is_dir():
        return
    base.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(base)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(base / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(base / "transformers"))


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
    pairs_file: Path, out_dir: Path, model: str, task: str,
    prompt_format: str, component: str, device: str, layers: str,
    cached_model,
) -> int:
    """Per-token streaming extraction via collect_raw — no JSON intermediate.
    Writes every layer chunk + the _complete.json marker into `out_dir`
    (a persistent pending dir). Extraction ONLY: the upload is handed off
    to a detached worker after the model is released, so the GPU slot and
    model RAM free at extraction-end, not upload-end. Returns n_layers."""
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
    out_dir.mkdir(parents=True, exist_ok=True)
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
        for layer_name in per_layer_pos:
            try:
                layer_int = int(layer_name)
            except ValueError:
                continue
            out_path = out_dir / f"layer_{layer_int}_chunk_{chunk_idx}.safetensors"
            _save_chunk_per_token(
                out_path, per_layer_pos[layer_name], per_layer_neg[layer_name], pids,
            )
        n_layers_seen = max(n_layers_seen, len(per_layer_pos))
        del per_layer_pos, per_layer_neg
        print(f"[{task}/{prompt_format}] chunk {chunk_idx} extracted", flush=True)
    n_chunks = (len(pairs_list) + PAIR_CHUNK_SIZE - 1) // PAIR_CHUNK_SIZE
    (out_dir / "_complete.json").write_text(json.dumps(
        {"n_chunks": n_chunks, "n_layers": n_layers_seen, "n_pairs": len(pairs_list)}
    ))
    return n_layers_seen


def main() -> int:
    _configure_hf_cache()
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
    from .upload_worker import new_job_dir, handoff, sweep
    sweep()  # re-spawn uploaders orphaned by a prior worker death / box restart
    job_dir = new_job_dir(args.task, args.prompt_format)
    try:
        generate_pairs(args.task, pairs_file, limit=args.limit)
        cached = _preload_model(args.model, args.device)
        n_layers = _stream_extract_to_safetensors(
            pairs_file=pairs_file, out_dir=job_dir / "data",
            model=args.model, task=args.task,
            prompt_format=args.prompt_format, component=args.component,
            device=args.device, layers=args.layers, cached_model=cached,
        )
        del cached  # free the model+VRAM at extraction-end, before any upload
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)  # don't upload a partial job
        raise
    finally:
        if owns:
            shutil.rmtree(work_dir, ignore_errors=True)
    # Hand the upload to a detached, torch-free worker and exit immediately:
    # the agent slot + model RAM are already free, so the GPU isn't pinned
    # by the bandwidth-bound upload. The worker pool drains pending to the
    # configured raw upload backend.
    base_in_repo = f"raw_activations/{_safe(args.model)}/{args.task}/{args.prompt_format}"
    repo_id, repo_type = _upload_target()
    handoff(job_dir, repo_id, base_in_repo, repo_type, os.environ.get("WC_JOB_ID", ""))
    print(
        f"[{args.task}/{args.prompt_format}] extracted {n_layers} layers; "
        f"upload handed off (slot freed)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
