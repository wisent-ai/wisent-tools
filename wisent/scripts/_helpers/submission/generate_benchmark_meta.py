"""Emit benchmark_meta.json for the wisent-enterprise /jobs Benchmarks tab.

Joins the canonical 380-entry benchmark_tags.json (top-level benchmark
names) with GROUP_TASK_EXPANSIONS (55 group→subtasks entries) to build
a single JSON file the Next.js dashboard can ship as a static asset.

Output schema (one entry per top-level benchmark):
    {
      "<top_level_name>": {
        "subtask_count": int,
        "subtasks": [str, ...],   # at least [<top_level_name>] when leaf
        "tags": [str, ...]        # passthrough from benchmark_tags.json
      },
      ...
    }

Usage:
    python -m wisent.scripts._helpers.submission.generate_benchmark_meta \\
        --output /Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/\\
backends/wisent-enterprise/lib/jobs/benchmark_meta.json
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any


def _load_benchmark_tags() -> dict[str, Any]:
    candidates = [
        Path("/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/"
             "backends/wisent-open-source/wisent/support/examples/scripts/"
             "benchmark_tags.json"),
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text())
    raise SystemExit("benchmark_tags.json not found in any known location")


def _load_group_expansions() -> dict[str, list[str]]:
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping \
        import GROUP_TASK_EXPANSIONS
    return dict(GROUP_TASK_EXPANSIONS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True,
                   help="Path to write the JSON. Overwritten if exists.")
    args = p.parse_args()

    tags = _load_benchmark_tags()
    expansions = _load_group_expansions()

    out: dict[str, dict[str, Any]] = {}
    # Top-level benchmarks from benchmark_tags.json get tags. Mark them
    # is_top_level so the dashboard's default tree-root view filters to
    # exactly the 380 user-facing entries.
    for name, payload in tags.items():
        sub = expansions.get(name)
        if sub is None or len(sub) == 0:
            subtasks = [name]
        else:
            subtasks = list(sub)
        out[name] = {
            "subtask_count": len(subtasks),
            "subtasks": subtasks,
            "tags": list((payload or {}).get("tags") or []),
            "is_top_level": True,
        }
    # Add every GROUP_TASK_EXPANSIONS key not already covered. These are
    # intermediate group nodes (eg tmmluplus_stem expanding to 15 leaves)
    # that the tree UI needs to find when a user clicks to expand a row
    # whose subtask is itself a sub-group.
    for name, sub in expansions.items():
        if name in out:
            continue
        out[name] = {
            "subtask_count": len(sub),
            "subtasks": list(sub),
            "tags": [],
            "is_top_level": False,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {len(out)} entries to {out_path}", file=sys.stderr)
    print(f"  groups: {sum(1 for v in out.values() if v['subtask_count'] > 1)}",
          file=sys.stderr)
    print(f"  leaves: {sum(1 for v in out.values() if v['subtask_count'] == 1)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
