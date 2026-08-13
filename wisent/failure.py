"""Failure classification for the operator tools in this repository.

The vocabulary is `wisent-errors`, the shared catalogue every Wisent component
derives from, so a failure reads the same whether it is seen in a JSON body or
in a terminal: the same seven codes, the same severities, the same retry
semantics, the same "is this ours or the caller's" split. This module keeps the
CLI's own half — the failure points, the service names, the log line, the one
sentence on stderr — and takes every derived value from the catalogue.

Two deliberate differences, because this runs in a shell and not behind a load
balancer:

* **No collector call.** A CLI that phones home while failing is one more thing
  that hangs when the network is exactly what broke. A failure leaves one
  structured log line and one sentence on stderr, and nothing else.
* **Nothing is hidden from the operator.** The technical detail — upstream
  status, exception text, response body, environment variable names — goes into
  the log line, which is read by a human who already has shell access. What is
  kept deliberately small is the *machine* interface: the exit code and the one
  human sentence.

Exit codes (ratified across every Wisent CLI in this workstream):

* ``EXIT_UNAVAILABLE`` (69, ``EX_UNAVAILABLE``) whenever the classification is
  retryable — ``infra_down``, ``timeout``, ``rate_limit``. One signal, meaning
  "not your fault, try later", so no one has to guess which of two failures may
  be repeated.
* the caller's existing error code otherwise — ``auth``, ``not_found``,
  ``config`` and bad arguments are not fixed by retrying. Existing codes are
  never renumbered, which is why ``error_exit`` is a parameter: ``stado`` already
  spends exit 1 on the *answer* "this prefix does not exist".
"""
from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass

from wisent_errors import (
    CODES as _CATALOGUE,
    RETRY_EXIT,
    exit_code as _exit_remap,
    from_upstream_status,
    outage,
    retryable,
    severity,
)

#: The log line is read in a terminal, and 500 characters is what this CLI has
#: always emitted. `wisent_errors` bounds an envelope's `detail` at 2000, but it
#: does so inside `failure()`, which this module cannot call: the failure points
#: here are two-segment CLI ids and there is no `impact` axis to pass. Raising
#: the cap would change the log line, so the bound stays local until the package
#: exposes it.
_MAX_DETAIL_CHARS = int("500")

#: `EX_UNAVAILABLE` from sysexits(3): the dependency, not the invocation. The
#: catalogue owns it, because the retry signal has to mean one thing fleet-wide.
EXIT_UNAVAILABLE = RETRY_EXIT

#: The invocation itself is wrong; retrying it unchanged is pointless. Callers
#: whose repository already spends a different code on errors pass their own.
EXIT_ERROR = int("2")


#: The catalogue's seven, in the shape this module has always exposed.
CODES = frozenset(_CATALOGUE)


def _named_code(code: str) -> str:
    """A catalogue code, under the name this repository's call sites already use.

    The names are local; the vocabulary is not. If the catalogue ever stops
    defining one of these, importing this module fails here instead of the CLI
    reporting a code that no longer means anything.
    """
    if code not in CODES:
        raise ImportError(f"wisent_errors no longer defines the {code!r} code")
    return code


CODE_CONFIG = _named_code("config")
CODE_AUTH = _named_code("auth")
CODE_NOT_FOUND = _named_code("not_found")
CODE_RATE_LIMIT = _named_code("rate_limit")
CODE_TIMEOUT = _named_code("timeout")
CODE_INFRA_DOWN = _named_code("infra_down")
CODE_UNKNOWN = _named_code("unknown")

#: One sentence, addressed to the person who ran the command. It names the
#: dependency and says who has to act. No exception text, no response body —
#: those are one line above, in the log.
MESSAGE_BY_CODE = {
    CODE_CONFIG: "is not configured here; set the missing settings and re-run",
    CODE_AUTH: "rejected our credentials; refresh the token and re-run",
    CODE_NOT_FOUND: "does not have what was asked for",
    CODE_RATE_LIMIT: "is rate limiting us; wait and re-run",
    CODE_TIMEOUT: "did not answer in time; ours, not your command",
    CODE_INFRA_DOWN: "is unreachable; ours, not your command",
    CODE_UNKNOWN: "failed in a way this tool does not recognise",
}

#: Dependency axis, named the way it is named operationally.
SERVICE_STADO = "stado"
SERVICE_DATABASE = "database"
SERVICE_HUGGINGFACE = "huggingface"
SERVICE_APP = "app"

logger = logging.getLogger("wisent.failure")

#: Credentials never belong in a log file either — the log outlives the shell.
_SENSITIVE_RE = re.compile(
    r"\b(token|signature|secret|password|api[_-]?key)=[^&\s]+",
    re.IGNORECASE,
)

_TIMEOUT_TYPE_NAMES = (
    "TimeoutError",
    "ReadTimeout",
    "ReadTimeoutError",
    "ConnectTimeout",
    "ConnectTimeoutError",
    "ConnectionTimeout",
    "socket.timeout",
)

_CONFIG_MARKERS = (
    "is required",
    "not configured",
    "missing env",
    "must be set",
    "must use",
    "must be an absolute url",
    "unsafe url syntax",
    "control characters",
    "invalid port",
    "no token",
)

_NETWORK_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "network is unreachable",
    "no route to host",
    "cannot connect",
    "server disconnected",
    "remote end closed connection",
    "max retries exceeded",
    "bad status line",
)


@dataclass(frozen=True)
class Classification:
    """The few distinctions an operator and their wrapper script need."""

    code: str
    service: str
    failure_point: str
    severity: str
    retryable: bool
    outage: bool
    #: Everything technical: status, exception type and text, upstream body.
    #: Log only — it is never the tool's return value.
    detail: str | None = None

    def exit_code(self, error_exit: int = EXIT_ERROR) -> int:
        """69 when repeating the command could work, the caller's code if not."""
        return _exit_remap(self.code, error_exit)


def _from_exception(error: BaseException | None) -> str | None:
    """Classify by type first, then by message.

    Type names are compared as strings so that this module stays importable
    with nothing but the standard library: a tool that fails because `requests`
    or `huggingface_hub` is missing must still be able to say so.
    """
    if error is None:
        return None
    names = {base.__name__ for base in type(error).__mro__}
    if names & set(_TIMEOUT_TYPE_NAMES):
        return CODE_TIMEOUT
    # ConnectionError and its socket-level relatives are OSError subclasses, so
    # the timeout check above has to come first.
    if isinstance(error, (ConnectionError, OSError)):
        return CODE_INFRA_DOWN

    message = str(error).lower()
    if any(marker in message for marker in _CONFIG_MARKERS):
        return CODE_CONFIG
    if any(marker in message for marker in _NETWORK_MARKERS):
        return CODE_INFRA_DOWN
    if "remotedisconnected" in message or "badstatusline" in message:
        return CODE_INFRA_DOWN
    return None


def _detail(
    error: BaseException | None,
    status: int | None,
    reason: str | None,
) -> str | None:
    parts: list[str] = []
    if status is not None:
        parts.append(f"http {status}")
    if reason:
        parts.append(reason)
    if error is not None:
        parts.append(f"{type(error).__name__}: {error}")
    if not parts:
        return None
    return _SENSITIVE_RE.sub(r"\1=[redacted]", " — ".join(parts))[:_MAX_DETAIL_CHARS]


def classify(
    failure_point: str,
    *,
    service: str = SERVICE_APP,
    error: BaseException | None = None,
    status: int | None = None,
    code: str | None = None,
    reason: str | None = None,
) -> Classification:
    """Turn whatever a dependency did into the contract's vocabulary.

    An explicit ``code`` wins: some failures are known exactly at the call site,
    and guessing from an exception type would only throw that knowledge away.
    """
    resolved = code or _from_exception(error)
    if resolved is None and status is not None:
        resolved = from_upstream_status(status)
    if resolved not in CODES:
        resolved = CODE_UNKNOWN
    return Classification(
        code=resolved,
        service=service,
        failure_point=failure_point,
        severity=severity(resolved),
        retryable=retryable(resolved),
        outage=outage(resolved),
        detail=_detail(error, status, reason),
    )


def log_line(classification: Classification) -> str:
    """The one structured line, parseable by eye and by `grep`."""
    fields = [
        f"failure_point={classification.failure_point}",
        f"error_code={classification.code}",
        f"service={classification.service}",
        f"severity={classification.severity}",
        f"retryable={'true' if classification.retryable else 'false'}",
        f"outage={'true' if classification.outage else 'false'}",
    ]
    if classification.detail:
        fields.append(f"detail={json.dumps(classification.detail)}")
    return "wisent.failure " + " ".join(fields)


def human_message(classification: Classification, program: str | None = None) -> str:
    """One sentence for the person watching the terminal."""
    prefix = f"{program}: " if program else ""
    verdict = MESSAGE_BY_CODE.get(classification.code, MESSAGE_BY_CODE[CODE_UNKNOWN])
    tail = " Safe to retry." if classification.retryable else ""
    return f"{prefix}{classification.service} {verdict}.{tail}"


def report(
    failure_point: str,
    *,
    service: str = SERVICE_APP,
    error: BaseException | None = None,
    status: int | None = None,
    code: str | None = None,
    reason: str | None = None,
    program: str | None = None,
    debug: bool = False,
    stream=None,
) -> Classification:
    """Classify, log once, say one honest sentence. Never raises, never blocks.

    The traceback is not printed unless ``debug`` is set; it is always available
    from the log at DEBUG level, so `--debug` is a convenience and not the only
    way to get it.
    """
    classification = classify(
        failure_point,
        service=service,
        error=error,
        status=status,
        code=code,
        reason=reason,
    )
    logger.error(log_line(classification))
    if error is not None:
        logger.debug("traceback for %s", failure_point, exc_info=error)
    target = sys.stderr if stream is None else stream
    print(human_message(classification, program), file=target, flush=True)
    if debug and error is not None:
        traceback.print_exception(type(error), error, error.__traceback__, file=target)
    elif error is not None:
        print("(re-run with --debug for the traceback)", file=target, flush=True)
    return classification
