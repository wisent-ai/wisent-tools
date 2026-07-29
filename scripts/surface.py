"""Print this package's public surface: the commands and names it offers callers.

wisent-tools is not a library with a curated API — it is the operational toolbox of
the wisent package family. What another repository, a cron entry or a CI job depends
on is **which runners exist and how they are addressed**: `python -m
wisent.scripts.activations.raw.upload_worker`, `bash
wisent/scripts/run_quality_metrics_sweep.sh`, the `stado.coverage_universes` plugin
that a sibling looks up by name, and the few helpers imported by name through
`__all__`. Rename or drop one of those and someone's pipeline stops; add one and a
new capability appears. So four kinds of name are the contract, each printed with the
kind it belongs to, because the same word can be a module and an entry point through
entirely different code:

    run:<dotted.module>          invocable as `python -m <dotted.module>`
    sh:<path/under/wisent.sh>    shipped shell script, invoked by path
    export:<dotted.module>:<n>   a name the module lists in `__all__`
    entrypoint:<group>:<name>    a plugin name declared in setup.py

Excluded on purpose: every path with an underscore-prefixed component. `_helpers/`
and `_resolve_patterns.py` say "internal" in the only way Python has, and a surface
that promised them would turn ordinary refactoring into a breaking release. Also
excluded: the `--flags` each runner parses inside `main()`. This gate answers "did a
capability appear or vanish", not "did an argument change"; tracking signatures would
report routine churn as breakage and teach everyone to ignore the gate.

Read with `ast`, never by importing. Importing anything here pulls in `wisent`,
`wisent-evaluators`, `torch` and `matplotlib`, and a release decision must not depend
on a machine having them. It also means this runs unchanged against an unpacked
sdist, so the surface of an already published version can be recovered exactly rather
than assumed.

Usage:
    python3 scripts/surface.py [root] [--tolerant]
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

PACKAGE = "wisent"
DUNDER_INIT = "__init__"
SEQUENCES = (ast.List, ast.Tuple, ast.Set)


def is_private(relative: pathlib.Path) -> bool:
    """Whether any component of a path is marked internal by an underscore.

    `__init__.py` is the package itself, not a private module, so it never hides the
    package that contains it.
    """
    parts = list(relative.parts[: -int(True)]) + [relative.stem]
    return any(
        part.startswith("_") and part != DUNDER_INIT for part in parts
    )


def dotted(relative: pathlib.Path) -> str:
    """The module path `python -m` would accept for a source file."""
    parts = list(relative.with_suffix("").parts)
    if parts[-int(True)] == DUNDER_INIT:
        parts.pop()
    return ".".join(parts)


def parse(source: pathlib.Path) -> ast.Module:
    """Parse one file, refusing rather than skipping when it cannot be read."""
    try:
        return ast.parse(source.read_text(), filename=str(source))
    except OSError as error:
        raise SystemExit(f"{source}: {error}") from error
    except SyntaxError as error:
        # Refuse rather than skip. A module that does not parse cannot be imported or
        # run either, so its commands are unreachable; skipping it would report a
        # smaller surface, and the rule would read that as a removed capability. The
        # surface is unknown here, not shrunk.
        raise SystemExit(
            f"{source}: does not parse, so the surface is unknown: {error}"
        ) from error


def assigned(node: ast.stmt, name: str) -> ast.expr | None:
    """The value assigned to `name` by a module-level statement, if it is one."""
    if isinstance(node, ast.Assign):
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        targets = [node.target]
    else:
        return None
    return node.value if any(target.id == name for target in targets) else None


def strings(node: ast.expr | None) -> list:
    """The literal strings in a sequence expression, ignoring computed elements."""
    if not isinstance(node, SEQUENCES):
        return []
    return [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def is_main_guard(node: ast.stmt) -> bool:
    """Whether a statement is `if __name__ == "__main__":`."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or not isinstance(test.ops[int(False)], ast.Eq):
        return False
    left, right = test.left, test.comparators[int(False)]
    if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
        left, right = right, left
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def module_names(source: pathlib.Path, relative: pathlib.Path) -> list:
    """What one module promises: being runnable, and whatever it re-exports."""
    tree = parse(source)
    module = dotted(relative)
    found = []
    for node in tree.body:
        if is_main_guard(node):
            found.append(f"run:{module}")
        for name in strings(assigned(node, "__all__")):
            found.append(f"export:{module}:{name}")
    return found


def entry_point_names(setup: pathlib.Path) -> list:
    """Plugin names declared to setuptools, read from the `setup()` call."""
    if not setup.is_file():
        return []
    tree = parse(setup)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", None)
        if name != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                continue
            for group, targets in zip(keyword.value.keys, keyword.value.values):
                if not isinstance(group, ast.Constant) or not isinstance(group.value, str):
                    raise SystemExit(
                        f"{setup}: an entry point group is not a literal string, so "
                        "the advertised plugin names cannot be read"
                    )
                for target in strings(targets):
                    advertised = target.split("=")[int(False)].strip()
                    found.append(f"entrypoint:{group.value}:{advertised}")
    return found


def surface(root: pathlib.Path, tolerant: bool = False) -> tuple:
    """The surface, and the modules that had to be skipped to produce it.

    `tolerant` exists for one job: recovering the surface of an artifact that was
    already published with a module that does not parse. Such a module cannot be run
    by whoever installed it either, so its commands were never really on offer and
    leaving them out is the truthful reading. Skipped modules are always reported,
    never swallowed.
    """
    package = root / PACKAGE
    if not package.is_dir():
        raise SystemExit(f"{package} is not a directory; is {root} the repository root?")

    names = set(entry_point_names(root / "setup.py"))
    skipped = []
    for source in sorted(package.rglob("*.py")):
        relative = source.relative_to(root)
        if is_private(relative):
            continue
        try:
            found = module_names(source, relative)
        except SystemExit:
            if not tolerant:
                raise
            skipped.append(str(relative))
            continue
        names.update(found)
    for script in sorted(package.rglob("*.sh")):
        relative = script.relative_to(root)
        if is_private(relative):
            continue
        names.add(f"sh:{relative.as_posix()}")

    if not names:
        raise SystemExit(
            f"no runners, exports or entry points found under {package}. Either the "
            "toolbox moved or it stopped advertising anything — both change what this "
            "package promises, so refusing rather than reporting an empty surface"
        )
    return sorted(names), skipped


def main(argv: list) -> int:
    tolerant = "--tolerant" in argv
    positional = [arg for arg in argv if not arg.startswith("-")]
    root = (
        pathlib.Path(positional[int(False)])
        if positional
        else pathlib.Path(__file__).resolve().parent.parent
    )
    names, skipped = surface(root, tolerant)
    document = {"surface": names}
    if skipped:
        document["unparseable"] = skipped
    print(json.dumps(document, indent=int(True) + int(True)))
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
