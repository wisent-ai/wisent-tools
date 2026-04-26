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


def _count_pairs(pairs_file: Path) -> int | None:
    """Count of contrastive pairs in the file, or None on read failure."""
    try:
        import json as _json
        with pairs_file.open() as f:
            data = _json.load(f)
        pairs = data.get("pairs", data) if isinstance(data, dict) else data
        return len(pairs) if pairs is not None else None
    except Exception:
        return None


def auto_batch_size(
    cached_model,
    device: str,
    requested: int,
    pairs_file: Path | None = None,
    ceiling: int = 128,
) -> int:
    """Measure-then-size: run one forward pass at batch=1 to learn real
    bytes-per-sample on this GPU + this model + this seq_len, then scale
    the candidate from that measurement. Falls back to a static formula
    only if measurement fails. Probe-and-halve still guards the final pick.
    """
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
        if pairs_file and tok:
            seq_len = _measure_pairs_seq_len(pairs_file, tok)
        else:
            seq_len = min(512, getattr(cfg, "max_position_embeddings", 2048) or 2048)
        device_obj = next(hf.parameters()).device
        vocab = getattr(cfg, "vocab_size", 32000)

        # Measurement pass: run one forward at batch=1 and see what it costs.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_obj)
        baseline = torch.cuda.memory_allocated(device_obj)
        try:
            probe = torch.randint(0, vocab, (1, seq_len), device=device_obj)
            with torch.no_grad():
                _ = hf(probe)
            peak = torch.cuda.max_memory_allocated(device_obj)
            bytes_per_sample = max(1, peak - baseline)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[auto-bs] OOM at batch=1; falling back to requested floor", flush=True)
            return requested

        free_bytes, _ = torch.cuda.mem_get_info(0)
        # Cap by the actual number of pairs — batching beyond len(pairs) wastes
        # VRAM with no throughput gain (the loop only ever has len(pairs)
        # items to process), and a runaway probe at batch=128 with 5 pairs
        # was eating ~80% of a T4 for nothing.
        n_pairs = _count_pairs(pairs_file) if pairs_file else None
        effective_ceiling = min(ceiling, n_pairs) if n_pairs else ceiling
        # 0.85 is the only soft constant left: leaves slack for activation
        # accumulation and CUDA fragmentation. The 3x workspace guess from
        # the v0.1.5 formula is gone — bytes_per_sample is measured.
        candidate = max(requested, min(effective_ceiling, int(0.85 * free_bytes / bytes_per_sample)))
        print(
            f"[auto-bs] measured per_sample={bytes_per_sample/1e6:.1f}MB "
            f"free={free_bytes/1e9:.1f}GB seq={seq_len} candidate={candidate}",
            flush=True,
        )
        if candidate <= requested:
            return requested

        # Confirm the candidate with one real probe; halve on OOM.
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
