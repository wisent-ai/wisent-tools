#!/usr/bin/env python3
"""Disabled: old aggregated-shard migration was retired with foreground uploads."""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "disabled: aggregated activation shard migration removed. Raw "
        "activation outputs are the supported storage path."
    )


if __name__ == "__main__":
    raise SystemExit(main())
