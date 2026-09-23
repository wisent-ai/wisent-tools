"""Provider-neutral object persistence through the Stado API."""
from __future__ import annotations

import argparse
import http.client
import json
import sys

from wisent import failure
from wisent.stado_support.client import StadoClient
from wisent.stado_support.core import (
    EXIT_MISSING as _EXIT_MISSING,
    EXIT_OK as _EXIT_OK,
    StadoConflict,
    StadoError,
    StadoNotFound,
    join_uri,
    split_uri,
)






#: What the operator loses when a command fails, per subcommand. `has-prefix`
#: shares `stado.list` because it is the same call with a narrower answer.
_FAILURE_POINTS = {
    "list": "stado.list",
    "has-prefix": "stado.list",
    "put-tree": "stado.write",
    "get-prefix": "stado.read",
}

_PROGRAM = "wisent.stado"


def _dispatch(client: StadoClient, args: argparse.Namespace) -> int:
    if args.command == "list":
        for item in client.list_uri(args.uri):
            print(item.get("uri", ""))
        return _EXIT_OK
    if args.command == "has-prefix":
        return _EXIT_OK if client.list_uri(args.uri) else _EXIT_MISSING
    if args.command == "put-tree":
        client.put_tree(args.source, args.uri, delete_missing=args.sync)
        return _EXIT_OK
    if args.command == "get-prefix":
        client.get_prefix(args.uri, args.destination)
        return _EXIT_OK
    raise AssertionError(args.command)


def _main() -> int:
    """Run one subcommand and answer with an exit code that means something.

    Every failure leaves through `failure.report`: one structured log line with
    the status and the upstream body for whoever is debugging, one sentence on
    stderr for whoever is watching, and exit 69 when — and only when — trying
    again could work. Nothing here ever exits 1 on a failure: `has-prefix`
    spends exit 1 on the answer "this prefix does not exist", and a dead Stado
    must never be mistaken for an empty one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the traceback of a failure to stderr",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("uri")
    has_parser = subparsers.add_parser("has-prefix")
    has_parser.add_argument("uri")
    put_parser = subparsers.add_parser("put-tree")
    put_parser.add_argument("uri")
    put_parser.add_argument("source")
    put_parser.add_argument("--sync", action="store_true")
    get_parser = subparsers.add_parser("get-prefix")
    get_parser.add_argument("uri")
    get_parser.add_argument("destination")
    args = parser.parse_args()

    try:
        client = StadoClient()
    except StadoError as exc:
        # Every error this constructor raises is about this machine's settings,
        # so it is `config` by construction rather than by guesswork.
        return failure.report(
            "stado.config",
            service=failure.SERVICE_STADO,
            error=exc,
            code=failure.CODE_CONFIG,
            program=_PROGRAM,
            debug=args.debug,
        ).exit_code()

    point = _FAILURE_POINTS.get(args.command, "stado.unknown")
    try:
        return _dispatch(client, args)
    except json.JSONDecodeError as exc:
        # A 200 whose body is not JSON is a broken dependency, not a broken
        # argument — and `JSONDecodeError` is a `ValueError`, so it has to be
        # caught above the argument branch.
        classification = failure.report(
            point,
            service=failure.SERVICE_STADO,
            error=exc,
            code=failure.CODE_INFRA_DOWN,
            reason="Stado returned a body that is not JSON",
            program=_PROGRAM,
            debug=args.debug,
        )
        return classification.exit_code()
    except ValueError as exc:
        # A malformed URI or a source that is not a directory. The text is ours,
        # written for a human, and it names the argument that has to change.
        print(f"{_PROGRAM}: {exc}", file=sys.stderr, flush=True)
        return failure.EXIT_ERROR
    except StadoNotFound as exc:
        classification = failure.report(
            point,
            service=failure.SERVICE_STADO,
            error=exc,
            code=failure.CODE_NOT_FOUND,
            program=_PROGRAM,
            debug=args.debug,
        )
    except (StadoError, OSError, http.client.HTTPException) as exc:
        classification = failure.report(
            point,
            service=failure.SERVICE_STADO,
            error=exc,
            status=getattr(exc, "status", None),
            program=_PROGRAM,
            debug=args.debug,
        )
    return classification.exit_code()


if __name__ == "__main__":
    raise SystemExit(_main())
