#!/usr/bin/env python3
"""Disabled compatibility stub for the removed foreground activation uploader.

The foreground activation uploader has been removed. Use
wisent.scripts.activations.raw.extract_and_upload instead; queued jobs must
stage outputs for detached upload workers.
"""
from __future__ import annotations

import sys


MESSAGE = (
    "foreground activation uploader has been removed; use "
    "wisent.scripts.activations.raw.extract_and_upload"
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
