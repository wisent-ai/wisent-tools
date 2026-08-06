"""Compatibility command for the packaged Wisent Tools surface inspector.

The extractor itself lives in `wisent/surface.py`, inside the package, so it travels
in the sdist and can therefore read an unpacked published artifact with exactly the
code that reads the working tree. This file is the spelling the version gate and
scripts/baseline.py use, and it exists to do the one thing the import system will not
do for them: put the repository root on `sys.path`.

That is not decoration. Running `python3 scripts/surface.py` makes `sys.path[0]` the
*scripts* directory, never the repository root, so `import wisent` has nothing to
resolve against on a machine where the project is not installed — which is exactly
the machine this runs on, because .github/workflows/version-check.yml installs the
package deliberately nowhere. Resolving the root from `__file__` rather than trusting
PYTHONPATH or an editable install is what lets the gate run from a bare
`git clone --depth 1 --no-tags`.

Re-exported for scripts/baseline.py, which recovers an already-published surface with
this same reader: `surface` and `setup_string`, which are the only two names it calls.
Nothing more is forwarded on purpose — a shim that re-exported the whole module would
let callers reach an internal of the extractor through a name this file appears to
own, and the next move of that internal would break them from here.
"""

import pathlib
import sys

sys.path.insert(int(False), str(pathlib.Path(__file__).resolve().parent.parent))

from wisent.surface import main, setup_string, surface  # noqa: E402  (path set above)

__all__ = ["main", "setup_string", "surface"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[int(True) :]))
