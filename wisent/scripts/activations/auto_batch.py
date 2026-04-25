"""Auto-batch-size: probe largest safe forward-pass batch on the loaded GPU.

Two-stage:
  1. Heuristic candidate from model config + free VRAM.
  2. Real forward-pass probe at that candidate; halve on torch.cuda.OOM
     until either a batch passes or we reach the requested floor.

Right-sizes seq_len from real pair lengths in the pairs JSON when available,
rather than assuming the model's full context window — most activation
extraction pairs are 50-200 tokens, not 2048+. Skipped on CPU/MPS or when
no cached model is in scope.
"""
from __future__ import annotations

from pathlib import Path


def _measure_pairs_seq_len(pairs_file: Path, tokenizer, hard_cap: int = 2048) -> int:
    """Longest tokenized pair text in the file, plus 16-token buffer, capped."""
    try:
        import json as _json
        with pairs_file.open() as f:
            data = _json.load(f)
        pairs = data.get("pairs", data) if isinstance(data, dict) else data
        if not pairs:
            return 256
        lengths = []
        for p in pairs[:32]:
            for key in ("positive_response", "negative_response"):
                resp = p.get(key) if isinstance(p, dict) else None
                if not resp:
                    continue
                text = resp.get("text") if isinstance(resp, dict) else None
                if not text:
                    continue
                try:
                    tok = tokenizer(text, return_tensors=None, add_special_tokens=True)
                    lengths.append(len(tok["input_ids"]))
                except Exception:
                    pass
        return min(hard_cap, max(64, max(lengths) + 16) if lengths else 256)
    except Exception:
        return 256


def auto_batch_size(
    cached_model,
    device: str,
    requested: int,
    pairs_file: Path | None = None,
    ceiling: int = 128,
) -> int:
    """Probe the largest safe batch size on this GPU. Returns >= requested."""
    if cached_model is None or not device.startswith("cuda"):
        return requested
    try:
        import torch
    except ImportError:
        return requested
    try:
        hf = getattr(cached_model, "hf_model", None)
        tok = getattr(cached_model, "tokenizer", None)
        if hf is None:
            return requested
        cfg = hf.config
        hidden = getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0)
        layers = getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0)
        if not hidden or not layers:
            return requested
        if pairs_file and tok:
            seq_len = _measure_pairs_seq_len(pairs_file, tok)
        else:
            seq_len = min(512, getattr(cfg, "max_position_embeddings", 2048) or 2048)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        per_sample = hidden * seq_len * layers * 2 * 3
        candidate = max(requested, min(ceiling, int(0.70 * free_bytes / per_sample)))
        if candidate <= requested:
            return requested
        print(f"[auto-bs] free={free_bytes/1e9:.1f}GB seq={seq_len} candidate={candidate}", flush=True)
        device_obj = next(hf.parameters()).device
        vocab = getattr(cfg, "vocab_size", 32000)
        while candidate > requested:
            try:
                fake = torch.randint(0, vocab, (candidate, seq_len), device=device_obj)
                with torch.no_grad():
                    _ = hf(fake)
                torch.cuda.empty_cache()
                print(f"[auto-bs] OK batch_size={candidate}", flush=True)
                return candidate
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                candidate //= 2
                print(f"[auto-bs] OOM, retry batch_size={candidate}", flush=True)
            except Exception as exc:
                print(f"[auto-bs] probe failed: {exc}", flush=True)
                return requested
        return requested
    except Exception as exc:
        print(f"[auto-bs] error: {exc}", flush=True)
        return requested
