"""Count contrastive pairs in raw_activations shards on HF.

Reads only the safetensors JSON header (first 256 KB via Range
request) and reports pos_<pid> / neg_<pid> tensor counts plus the
pair_ids metadata list length for the specified (task, model,
layer, chunk) across the three prompt_format dimensions."""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

HF_BASE = "https://huggingface.co/datasets/wisent-ai/activations/resolve/main"
DEFAULT_LAYER = 1
DEFAULT_CHUNK = 0
DEFAULT_PROMPT_FORMATS = "chat,mc_balanced,role_play"
HEADER_FETCH_BYTES = 262143


def _safe_model(model: str) -> str:
    return model.replace("/", "__")


def _header_only(uri: str, token: str) -> dict:
    with tempfile.NamedTemporaryFile(prefix="shard_hdr_", suffix=".bin") as tmp:
        rc = subprocess.run(
            ["curl", "-sf", "-L",
             "-H", f"Authorization: Bearer {token}",
             "-r", f"0-{HEADER_FETCH_BYTES}",
             "-o", tmp.name, uri],
        ).returncode
        if rc != 0:
            raise SystemExit(f"curl failed for {uri}")
        with open(tmp.name, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(header_len))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--prompt-formats", default=DEFAULT_PROMPT_FORMATS)
    p.add_argument("--dump-pair-ids", action="store_true",
                   help="Print the full pair_ids list (one per line) after the count line.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required")

    safe = _safe_model(args.model)
    for pf in args.prompt_formats.split(","):
        uri = (f"{HF_BASE}/raw_activations/{safe}/{args.task}/{pf}/"
               f"layer_{args.layer}_chunk_{args.chunk}.safetensors")
        header = _header_only(uri, token)
        meta = header.pop("__metadata__", {})
        pair_ids = json.loads(meta.get("pair_ids", "[]"))
        pos = [k for k in header if k.startswith("pos_")]
        neg = [k for k in header if k.startswith("neg_")]
        print(f"{pf}: pos={len(pos)}  neg={len(neg)}  "
              f"pair_ids_meta={len(pair_ids)}")
        if args.dump_pair_ids:
            for pid in pair_ids:
                print(pid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
