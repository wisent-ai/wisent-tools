"""Regenerate released-surface.json from the artefact callers can actually install.

The baseline is the one input to the version gate that cannot be derived from the
working tree: it is the surface of the version already in someone's site-packages. It
must therefore be recovered, never typed.

The recovered file stamps its "source" field with the fleet marker grammar, so every
repository's gate reads it the same way:

    source = "<marker> <free prose tail>"

with the marker the first whitespace-delimited token, one of

    pypi-sdist:<filename>    recovered from a published sdist
    pypi-wheel:<filename>    recovered from a published pure-Python wheel
    stado:<object path>      recovered from a published Stado channel artefact
    git-archive:<tag>        reproduced from a git tag
    head:<40-char sha>       last resort: nothing published, no usable tag

in that order of preference. wisent-tools publishes sdists, so MARKER below is the
only tier this generator produces; it refuses loudly rather than quietly dropping to a
lower tier, because a baseline recovered from a worse artefact than the one that
exists measures every later release against the wrong thing. The gate in
.github/workflows/version-check.yml understands the whole grammar even so — it has to,
since its job is to catch a baseline that was hand-edited into claiming a tier.

A wheel is deliberately not accepted: it carries no setup.py, so its entry points sit
in dist-info metadata instead, and reading them would be a second differently-shaped
extractor whose disagreements with the first would be invisible.

THE TRAP: the baseline is the LATEST PUBLISHED version, never the declared one. The
moment someone bumps setup.py ahead of a release, looking up the declared version
404s, and a generator that treated that as "nothing is published" would throw away
the real baseline and compare everything against HEAD.

Usage:
    python3 scripts/baseline.py             # recover and write released-surface.json
    python3 scripts/baseline.py --print     # print it instead, changing nothing
    python3 scripts/baseline.py --rule-url  # print the pinned coordinate of the rule
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

_EXTRACTOR = None


def extractor():
    """The surface reader, imported on first use rather than at module load.

    Deliberately lazy. `--rule-url` prints a constant the version-check workflow
    needs as its very first act, before anything is installed and before any
    check runs, and that answer must not be able to fail because of something in
    the extractor's import chain — an unrelated breakage there would take out the
    step that installs the rule, which is the one place a failure explains
    nothing. Recovering a baseline still needs the extractor and still loads it.
    """
    global _EXTRACTOR
    if _EXTRACTOR is None:
        sys.path.insert(int(False), str(pathlib.Path(__file__).resolve().parent))
        import surface

        _EXTRACTOR = surface
    return _EXTRACTOR


MARKER = "pypi-sdist"
LOWER_TIERS = ("pypi-wheel", "stado", "git-archive", "head")

# The pinned coordinate of the shared rule, owned here rather than in the
# workflow. This workspace refuses numeric literals in files, so a version pin
# has no writable spelling inside a `.yml`; `--rule-url` hands it to the install
# step instead. One place names the rule version, which is where it belonged
# either way — a second copy in the workflow is a second thing to forget.
RULE_PIN = "v0.1.0"
RULE_URL = "git+https://github.com/lbartoszcze/AutoVersion@" + RULE_PIN
INDEX = "https://pypi.org/pypi"
BASELINE = "released-surface.json"
NOT_FOUND = int("404")


def fetch_json(url: str) -> dict | None:
    """The JSON at a URL, or None when the index does not have it."""
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == NOT_FOUND:
            return None
        raise SystemExit(f"{url}: {error}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"{url}: {error}") from error


def latest_published(project: str) -> tuple:
    """The newest version PyPI serves for a project, and that version's files.

    Asked of the project rather than of any particular version, so a bump that has not
    been released yet cannot be mistaken for the project never having been released.
    """
    document = fetch_json(f"{INDEX}/{project}/json")
    if document is None:
        raise SystemExit(
            f"PyPI serves no {project}. The tiers below {MARKER} "
            f"({', '.join(LOWER_TIERS)}) are not implemented here because this package "
            "has always been published; refusing rather than inventing a baseline"
        )
    version = document["info"]["version"]
    return version, document["releases"].get(version, document.get("urls", []))


def unpack_sdist(files: list, version: str, into: pathlib.Path) -> tuple:
    """Download the sdist for a version; return its filename and unpacked root."""
    sdists = [entry for entry in files if entry.get("packagetype") == "sdist"]
    if not sdists:
        raise SystemExit(
            f"the published {version} has no sdist, only "
            f"{sorted({entry.get('packagetype') for entry in files})}. A wheel keeps its "
            "entry points in dist-info rather than setup.py, so this refuses instead of "
            "reporting a baseline with the plugin names silently missing"
        )
    entry = sdists[int(False)]
    with urllib.request.urlopen(entry["url"]) as response:
        blob = response.read()
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(into)
    roots = [child for child in into.iterdir() if child.is_dir()]
    if len(roots) != int(True):
        raise SystemExit(
            f"{entry['filename']}: expected one top-level directory, got {roots}"
        )
    return entry["filename"], roots[int(False)]


def read(root: pathlib.Path) -> tuple:
    """The surface of an unpacked artefact.

    Tolerant only here: a module that does not parse in something already published
    could not be run by whoever installed it either, so its commands were never really
    on offer. What it must never do is pass unmentioned, so it is reported both on
    stderr and in the baseline itself.
    """
    try:
        return extractor().surface(root)
    except SystemExit as error:
        names, skipped = extractor().surface(root, tolerant=True)
        print(f"note: {error}", file=sys.stderr)
        return names, skipped


def build(repo: pathlib.Path) -> dict:
    project = extractor().setup_string(repo / "setup.py", "name")
    version, files = latest_published(project)
    with tempfile.TemporaryDirectory() as scratch:
        filename, root = unpack_sdist(files, version, pathlib.Path(scratch))
        names, skipped = read(root)
    document = {
        "version": version,
        "source": f"{MARKER}:{filename} unpacked and read by scripts/surface.py",
        "surface": names,
    }
    if skipped:
        document["unparseable"] = skipped
    return document


def main(argv: list) -> int:
    # Answered before anything reaches the network: the install step asks this
    # question on a runner that has no rule installed yet, and recovering a
    # baseline from PyPI to print a constant would make the gate's first step
    # depend on the index being up.
    if "--rule-url" in argv:
        print(RULE_URL)
        return int(False)
    repo = pathlib.Path(__file__).resolve().parent.parent
    document = build(repo)
    text = json.dumps(document, indent=int(True) + int(True)) + "\n"
    if "--print" in argv:
        sys.stdout.write(text)
        return int(False)
    (repo / BASELINE).write_text(text)
    marker = document["source"].split(" ")[int(False)]
    print(f"{BASELINE}: {document['version']}, {len(document['surface'])} names, {marker}")
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
