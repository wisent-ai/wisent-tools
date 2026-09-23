"""Product-owned first-use journey for a safe Wisent Tools operation.

This thin adapter persists progress and an event outbox before attempting Stado
delivery. It completes only after the documented local surface inspector returns
a validated structured result; help, setup, and successful process exit are not proof.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from wisent.surface import surface as inspect_surface

PRODUCT_ID = "wisent-tools"
CLIENT_ID = PRODUCT_ID
JOURNEY_ID = "first-use"
JOURNEY_VERSION = "2026-08-04.1"
JOURNEY_VERSION_ID = "12000000-0000-4000-8000-000000000013"
SOURCE_REVISION = "wisent-tools-first-use-2026-08-04"
FIRST_SUCCESS_FACT = "tool_result_observed"
SCHEMA_VERSION = 1
BASE_URL_ENV = "STADO_INTEGRATION_API_URL"
TOKEN_ENV = "WISENT_TOOLS_STADO_INTEGRATION_TOKEN"
SUBJECT_ENV = "WISENT_TOOLS_ONBOARDING_SUBJECT"
STATE_PATH_ENV = "WISENT_TOOLS_ONBOARDING_STATE"

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_EVENT_NAMES = frozenset(
    {
        "onboarding_started",
        "onboarding_resumed",
        "onboarding_step_viewed",
        "onboarding_step_completed",
        "onboarding_step_skipped",
        "onboarding_abandoned",
        "onboarding_reset",
        "onboarding_first_success_observed",
        "onboarding_completed",
    }
)
_ACTIONS = frozenset({"run_tool", "inspect_tool_result"})
_SCREEN_KINDS = frozenset({"machine_discovery", "machine_result"})
_FACTS = frozenset({FIRST_SUCCESS_FACT})
_OPERATORS = frozenset(
    {"present", "absent", "eq", "not_eq", "contains", "gt", "gte", "lt", "lte"}
)
_MAX_BUNDLE_BYTES = 262_144
# A journey graph with more screens than this is not one a first-use flow produces.
_MAX_SCREENS = 128


def _definition() -> dict[str, Any]:
    return {
        "analytics_contract": {
            "completion_event": "onboarding_completed",
            "contract_version": "1",
            "exposure_event": "onboarding_step_viewed",
            "first_success_event": "onboarding_first_success_observed",
            "primary_action_event": "onboarding_step_completed",
            "surface": "developer_toolkit_first_use",
        },
        "entry_screen_id": "discover-tool",
        "experiment_contract": None,
        "first_success_fact": FIRST_SUCCESS_FACT,
        "journey_id": JOURNEY_ID,
        "journey_version": JOURNEY_VERSION,
        "product_id": PRODUCT_ID,
        "published_at": "2026-08-04T00:00:00Z",
        "schema_version": SCHEMA_VERSION,
        "screens": [
            {
                "actions": ["run_tool"],
                "body_key": "wisent-tools.onboarding.discover-tool.body",
                "completion_evidence": None,
                "entry_conditions": None,
                "fallback_screen_id": None,
                "presentation": {
                    "body": (
                        "Invoke the documented surface operation and inspect the "
                        "structured list of supported tools; help text alone is not a result."
                    ),
                    "renderer": "machine_discovery",
                    "title": "Inspect the released toolkit surface",
                },
                "required": True,
                "screen_id": "discover-tool",
                "screen_kind": "machine_discovery",
                "title_key": "wisent-tools.onboarding.discover-tool.title",
                "transitions": [
                    {
                        "next_screen_id": "observe-result",
                        "priority": 10,
                        "reason_code": "canonical_progression",
                    }
                ],
            },
            {
                "actions": ["inspect_tool_result"],
                "body_key": "wisent-tools.onboarding.observe-result.body",
                "completion_evidence": {
                    "fact": FIRST_SUCCESS_FACT,
                    "kind": "fact",
                    "operator": "eq",
                    "value": True,
                },
                "entry_conditions": None,
                "fallback_screen_id": None,
                "presentation": {
                    "body": (
                        "Keep the returned result object with its supported surface array. "
                        "A zero exit without the structured result does not complete."
                    ),
                    "renderer": "machine_result",
                    "title": "Confirm a real structured toolkit result",
                },
                "required": True,
                "screen_id": "observe-result",
                "screen_kind": "machine_result",
                "title_key": "wisent-tools.onboarding.observe-result.title",
                "transitions": [],
            },
        ],
        "source_revision": SOURCE_REVISION,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fallback_bundle() -> dict[str, Any]:
    definition = _definition()
    canonical = _canonical(definition)
    return {
        "journey_version_id": JOURNEY_VERSION_ID,
        "definition": definition,
        "canonical_definition": canonical,
        "content_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source_revision": SOURCE_REVISION,
    }


def _validate_condition(condition: Any, *, optional: bool = False) -> None:
    if condition is None and optional:
        return
    if not isinstance(condition, dict):
        raise ValueError("invalid journey condition")
    kind = condition.get("kind")
    if kind in {"all", "any"}:
        conditions = condition.get("conditions")
        if set(condition) != {"kind", "conditions"} or not isinstance(conditions, list) or not conditions:
            raise ValueError("invalid journey condition group")
        for item in conditions:
            _validate_condition(item)
        return
    if kind == "not":
        if set(condition) != {"kind", "condition"}:
            raise ValueError("invalid journey negation")
        _validate_condition(condition.get("condition"))
        return
    if kind != "fact" or condition.get("fact") not in _FACTS:
        raise ValueError("journey requested an unsupported evidence fact")
    operator = condition.get("operator")
    if operator not in _OPERATORS:
        raise ValueError("journey requested an unsupported condition operator")
    expected_keys = {"kind", "fact", "operator"}
    if operator not in {"present", "absent"}:
        expected_keys.add("value")
    if set(condition) != expected_keys:
        raise ValueError("invalid fact condition")


def _validate_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        raise ValueError("invalid journey bundle envelope")
    version_id = str(bundle.get("journey_version_id", ""))
    if not _UUID.match(version_id) or version_id != JOURNEY_VERSION_ID:
        raise ValueError("unexpected journey version id")
    definition = bundle.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("missing journey definition")
    expected_identity = (
        definition.get("schema_version") == SCHEMA_VERSION
        and definition.get("product_id") == PRODUCT_ID
        and definition.get("journey_id") == JOURNEY_ID
        and definition.get("journey_version") == JOURNEY_VERSION
        and definition.get("first_success_fact") == FIRST_SUCCESS_FACT
        and definition.get("source_revision") == SOURCE_REVISION
        and bundle.get("source_revision") == SOURCE_REVISION
    )
    if not expected_identity:
        raise ValueError("invalid journey identity")
    canonical = bundle.get("canonical_definition")
    if not isinstance(canonical, str) or len(canonical.encode("utf-8")) > _MAX_BUNDLE_BYTES:
        raise ValueError("journey definition is oversized")
    if canonical != _canonical(definition):
        raise ValueError("journey definition is not canonical")
    digest = str(bundle.get("content_sha256", ""))
    if not _SHA256.match(digest) or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
        raise ValueError("journey content hash mismatch")
    screens = definition.get("screens")
    if not isinstance(screens, list) or not screens or len(screens) > _MAX_SCREENS:
        raise ValueError("invalid journey graph")
    by_id: dict[str, dict[str, Any]] = {}
    for screen in screens:
        if not isinstance(screen, dict):
            raise ValueError("invalid journey screen")
        screen_id = screen.get("screen_id")
        if not isinstance(screen_id, str) or not _IDENTIFIER.match(screen_id) or screen_id in by_id:
            raise ValueError("invalid journey screen id")
        if screen.get("screen_kind") not in _SCREEN_KINDS:
            raise ValueError("journey requested an unsupported screen kind")
        actions = screen.get("actions")
        if not isinstance(actions, list) or any(action not in _ACTIONS for action in actions):
            raise ValueError("journey requested an unsupported action")
        _validate_condition(screen.get("completion_evidence"), optional=True)
        if not isinstance(screen.get("transitions"), list):
            raise ValueError("invalid journey transitions")
        if not isinstance(screen.get("title_key"), str) or not isinstance(screen.get("body_key"), str):
            raise ValueError("invalid journey content keys")
        by_id[screen_id] = screen
    if definition.get("entry_screen_id") not in by_id:
        raise ValueError("missing journey entry screen")
    for screen in screens:
        for transition in screen["transitions"]:
            if not isinstance(transition, dict) or transition.get("next_screen_id") not in by_id:
                raise ValueError("invalid journey transition target")
            _validate_condition(transition.get("condition"), optional=True)
    return bundle

__all__ = [name for name in globals() if not name.startswith("__")]








