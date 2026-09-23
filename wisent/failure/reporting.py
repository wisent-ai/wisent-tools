"""Operator-facing rendering for classified Wisent Tools failures."""
from __future__ import annotations

import json
import sys
import traceback

from .core import (
    CODE_UNKNOWN,
    MESSAGE_BY_CODE,
    SERVICE_APP,
    Classification,
    classify,
    logger,
)


def log_line(classification: Classification) -> str:
    """Return the structured failure line used by operators and tooling."""
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
    """Return one actionable sentence for the person watching the terminal."""
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
    """Classify and report a failure without raising or blocking."""
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
