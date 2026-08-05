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
    if not isinstance(screens, list) or not screens or len(screens) > 128:
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


class _Store:
    def __init__(self) -> None:
        override = os.environ.get(STATE_PATH_ENV, "").strip()
        if override:
            self.path = Path(override).expanduser()
        else:
            state_home = os.environ.get("XDG_STATE_HOME")
            root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
            self.path = root / PRODUCT_ID / "onboarding.json"

    def load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        handle, temporary = tempfile.mkstemp(prefix="onboarding-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except OSError:
                pass


class _Transport:
    def __init__(self) -> None:
        self.base_url = os.environ.get(BASE_URL_ENV, "").strip().rstrip("/")
        self.token = os.environ.get(TOKEN_ENV, "").strip()

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.token)

    def _post(self, operation: str, payload: dict[str, Any]) -> Any:
        if not self.available:
            raise OSError("Stado transport is not configured")
        request = urllib.request.Request(
            f"{self.base_url}/api/integration/onboarding/{operation}",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-Onboarding-Client": CLIENT_ID,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
        if isinstance(body, dict) and body.get("ok") is False:
            raise OSError("Stado rejected onboarding operation")
        return body.get("result") if isinstance(body, dict) and "result" in body else body

    def read_bundle(self) -> Any:
        return self._post(
            "bundle.read",
            {
                "client_id": CLIENT_ID,
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version": JOURNEY_VERSION,
                "if_none_match": None,
            },
        )

    def assign_experiment(self, subject_hash: str) -> Any:
        return self._post(
            "experiments.assign",
            {
                "client_id": CLIENT_ID,
                "product_id": PRODUCT_ID,
                "app_id": PRODUCT_ID,
                "platform": platform.system().lower() or sys.platform,
                "surface": "cli",
                "subject": subject_hash,
                "journey_version_id": JOURNEY_VERSION_ID,
            },
        )

    def collect_event(self, event: dict[str, Any]) -> None:
        self._post("events.collect", {"client_id": CLIENT_ID, **event})

    def read_state(self, subject_hash: str, attempt_id: str | None) -> Any:
        payload = {
            "client_id": CLIENT_ID,
            "product_id": PRODUCT_ID,
            "journey_id": JOURNEY_ID,
            "journey_version_id": JOURNEY_VERSION_ID,
            "subject_hash": subject_hash,
        }
        if attempt_id:
            payload["attempt_id"] = attempt_id
        return self._post("state.read", payload)


def _subject_hash() -> str:
    explicit = os.environ.get(SUBJECT_ENV, "").strip()
    if explicit:
        subject = explicit
    else:
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        subject = f"{user}\0{platform.node()}\0{PRODUCT_ID}"
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _evaluate(condition: Any, evidence: dict[str, Any]) -> bool:
    if not condition:
        return True
    if not isinstance(condition, dict):
        return False
    kind = condition.get("kind")
    if kind == "all":
        return all(_evaluate(item, evidence) for item in condition.get("conditions", []))
    if kind == "any":
        return any(_evaluate(item, evidence) for item in condition.get("conditions", []))
    if kind == "not":
        return not _evaluate(condition.get("condition"), evidence)
    if kind != "fact":
        return False
    fact = condition.get("fact")
    actual = evidence.get(fact)
    operator = condition.get("operator")
    expected = condition.get("value")
    if operator == "present":
        return fact in evidence and actual is not None
    if operator == "absent":
        return fact not in evidence or actual is None
    if operator == "eq":
        return actual == expected
    if operator == "not_eq":
        return actual != expected
    if operator == "contains":
        return isinstance(actual, (list, tuple, str, dict)) and expected in actual
    if (
        operator in {"gt", "gte", "lt", "lte"}
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        return actual <= expected
    return False


def _run_surface(inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs:
        raise ValueError("the surface operation accepts no inputs")
    root = Path(__file__).resolve().parent.parent
    names, skipped = inspect_surface(root)
    result: dict[str, Any] = {"surface": names}
    if skipped:
        result["unparseable"] = skipped
    return result


_LOCAL_TOOLS: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "wisent.surface": (
        {
            "tool_id": "wisent.surface",
            "description": (
                "Inspect the released toolkit surface locally without importing "
                "operator modules or starting their workloads."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "safe": True,
            "local": True,
            "documented_command": "python -m wisent.surface",
        },
        _run_surface,
    )
}


def _valid_tool_result(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("tool_id") != "wisent.surface":
        return False
    if value.get("inputs") != {}:
        return False
    result = value.get("result")
    if not isinstance(result, dict) or set(result) not in ({"surface"}, {"surface", "unparseable"}):
        return False
    names = result.get("surface")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or names != sorted(set(names))
    ):
        return False
    if "unparseable" in result and (
        not isinstance(result["unparseable"], list)
        or any(not isinstance(path, str) or not path for path in result["unparseable"])
    ):
        return False
    required = {
        "run:wisent.onboarding",
        "run:wisent.surface",
        "entrypoint:console_scripts:wisent-tools-onboarding",
    }
    return required.issubset(names)


class OnboardingJourney:
    """Durable first-use state machine for SDK and CLI callers."""

    def __init__(self) -> None:
        self.store = _Store()
        self.transport = _Transport()
        self.subject_hash = _subject_hash()
        self.state = self.store.load()
        self.bundle: dict[str, Any] | None = None
        self.progress: dict[str, Any] | None = None

    def _save(self) -> None:
        self.store.save(self.state)

    def _load_bundle(self) -> None:
        bundle = None
        if self.transport.available:
            try:
                bundle = _validate_bundle(self.transport.read_bundle())
            except Exception:
                bundle = None
        if bundle is None:
            try:
                bundle = _validate_bundle(self.state.get("bundle"))
            except Exception:
                bundle = None
        if bundle is None:
            bundle = _validate_bundle(_fallback_bundle())
        self.state["bundle"] = bundle
        self.bundle = bundle
        self._save()

    def _new_progress(self) -> dict[str, Any]:
        assert self.bundle is not None
        return {
            "attempt_id": str(uuid.uuid4()),
            "product_id": PRODUCT_ID,
            "journey_version_id": JOURNEY_VERSION_ID,
            "subject_hash": self.subject_hash,
            "scope_kind": "device",
            "current_screen_id": self.bundle["definition"]["entry_screen_id"],
            "completed_screen_ids": [],
            "status": "in_progress",
            "answers": [],
            "evidence": {},
            "evidence_revision": self._evidence_revision({}),
            "updated_at": _now(),
        }

    def _valid_remote_progress(self, value: Any) -> bool:
        if isinstance(value, dict) and isinstance(value.get("progress"), dict):
            value = value["progress"]
        if not isinstance(value, dict) or self.bundle is None:
            return False
        screen_ids = {screen["screen_id"] for screen in self.bundle["definition"]["screens"]}
        return (
            value.get("product_id") == PRODUCT_ID
            and value.get("journey_version_id") == JOURNEY_VERSION_ID
            and value.get("subject_hash") == self.subject_hash
            and isinstance(value.get("attempt_id"), str)
            and value.get("current_screen_id") in screen_ids
            and value.get("status") in {"in_progress", "completed", "abandoned"}
            and isinstance(value.get("evidence"), dict)
            and isinstance(value.get("completed_screen_ids"), list)
        )

    def start(self) -> "OnboardingJourney":
        self._load_bundle()
        progress_by_subject = self.state.setdefault("progress", {})
        local = progress_by_subject.get(self.subject_hash)
        if not isinstance(local, dict) or local.get("journey_version_id") != JOURNEY_VERSION_ID:
            local = None
        remote = None
        if self.transport.available:
            try:
                candidate = self.transport.read_state(self.subject_hash, local.get("attempt_id") if local else None)
                if isinstance(candidate, dict) and isinstance(candidate.get("progress"), dict):
                    candidate = candidate["progress"]
                if self._valid_remote_progress(candidate):
                    remote = candidate
            except Exception:
                remote = None
        progress = remote or local or self._new_progress()
        resumed = remote is not None or local is not None
        progress["evidence_revision"] = self._evidence_revision(progress.get("evidence", {}))
        self.progress = progress
        progress_by_subject[self.subject_hash] = progress
        self._save()
        if (
            self.transport.available
            and "experiment_id" not in progress
            and "variant_id" not in progress
        ):
            try:
                assignment = self.transport.assign_experiment(self.subject_hash)
                if isinstance(assignment, dict):
                    progress["experiment_id"] = assignment.get("experimentId", assignment.get("experiment_id"))
                    progress["variant_id"] = assignment.get("variant", assignment.get("variant_id"))
                    self._touch()
            except Exception:
                pass
        self.emit("onboarding_resumed" if resumed else "onboarding_started")
        self.flush()
        return self

    @staticmethod
    def _evidence_revision(evidence: dict[str, Any]) -> str:
        return hashlib.sha256(_canonical(evidence).encode("utf-8")).hexdigest()

    def _touch(self) -> None:
        assert self.progress is not None
        self.progress["updated_at"] = _now()
        self._save()

    def screen(self) -> dict[str, Any]:
        assert self.bundle is not None and self.progress is not None
        for screen in self.bundle["definition"]["screens"]:
            if screen["screen_id"] == self.progress["current_screen_id"]:
                return screen
        raise ValueError("journey progress references an unknown screen")

    def snapshot(self) -> dict[str, Any]:
        assert self.progress is not None
        screen = self.screen()
        presentation = screen.get("presentation", {})
        return {
            "product_id": PRODUCT_ID,
            "journey_id": JOURNEY_ID,
            "journey_version": JOURNEY_VERSION,
            "journey_version_id": JOURNEY_VERSION_ID,
            "source_revision": SOURCE_REVISION,
            "attempt_id": self.progress["attempt_id"],
            "status": self.progress["status"],
            "screen": {
                "screen_id": screen["screen_id"],
                "title": presentation.get("title", screen["title_key"]),
                "body": presentation.get("body", screen["body_key"]),
                "actions": list(screen["actions"]),
            },
            "completed_screen_ids": list(self.progress.get("completed_screen_ids", [])),
        }

    def emit(self, name: str, properties: dict[str, Any] | None = None, screen_id: str | None = None) -> None:
        if name not in _EVENT_NAMES:
            raise ValueError("unsupported onboarding event")
        assert self.progress is not None
        event = {
            "event_id": str(uuid.uuid4()),
            "event_name": name,
            "attempt_id": self.progress["attempt_id"],
            "product_id": PRODUCT_ID,
            "journey_id": JOURNEY_ID,
            "journey_version_id": JOURNEY_VERSION_ID,
            "subject_hash": self.subject_hash,
            "scope_kind": "device",
            "screen_id": screen_id or self.progress["current_screen_id"],
            "occurred_at": _now(),
            "evidence_revision": self.progress["evidence_revision"],
            "properties": properties or {},
            "answers": self.progress.get("answers", []),
        }
        if self.progress.get("experiment_id"):
            event["experiment_id"] = self.progress["experiment_id"]
        if self.progress.get("variant_id"):
            event["variant_id"] = self.progress["variant_id"]
        self.state.setdefault("events", []).append(event)
        self._save()

    def flush(self) -> None:
        if not self.transport.available:
            return
        for event in list(self.state.get("events", [])):
            try:
                self.transport.collect_event(event)
            except Exception:
                return
            self.state["events"] = [queued for queued in self.state.get("events", []) if queued.get("event_id") != event["event_id"]]
            self._save()

    def expose(self) -> None:
        self.emit("onboarding_step_viewed")
        self.flush()

    def _record_evidence(self, fact: str, value: Any) -> None:
        assert self.progress is not None
        evidence = dict(self.progress.get("evidence", {}))
        evidence[fact] = value
        self.progress["evidence"] = evidence
        self.progress["evidence_revision"] = self._evidence_revision(evidence)
        self._touch()

    def _advance(self) -> bool:
        assert self.progress is not None
        current = self.screen()
        evidence = self.progress.get("evidence", {})
        if not _evaluate(current.get("completion_evidence"), evidence):
            return False
        transitions = sorted(current.get("transitions", []), key=lambda item: item.get("priority", 0))
        selected = next((item for item in transitions if _evaluate(item.get("condition"), evidence)), None)
        if selected is None:
            return False
        screen_id = current["screen_id"]
        self.progress["completed_screen_ids"] = list(dict.fromkeys(self.progress.get("completed_screen_ids", []) + [screen_id]))
        self.progress["current_screen_id"] = selected["next_screen_id"]
        self.progress.setdefault("decisions", []).append(
            {
                "product_id": PRODUCT_ID,
                "journey_id": JOURNEY_ID,
                "journey_version": JOURNEY_VERSION,
                "attempt_id": self.progress["attempt_id"],
                "current_screen_id": screen_id,
                "selected_next_screen_id": selected["next_screen_id"],
                "reason_code": selected.get("reason_code", "transition_selected"),
                "evidence_revision": self.progress["evidence_revision"],
                "decided_at": _now(),
                "experiment_id": self.progress.get("experiment_id"),
                "variant_id": self.progress.get("variant_id"),
            }
        )
        self._touch()
        self.emit(
            "onboarding_step_completed",
            {"reason_code": selected.get("reason_code", "transition_selected"), "selected_next_screen_id": selected["next_screen_id"]},
            screen_id,
        )
        self.flush()
        return True

    def run_tool(self) -> dict[str, Any]:
        assert self.progress is not None
        entry = _LOCAL_TOOLS["wisent.surface"]
        tool_result = {"tool_id": "wisent.surface", "inputs": {}, "result": entry[1]({})}
        if not _valid_tool_result(tool_result):
            raise RuntimeError("surface tool did not return the required structured result")
        self.progress["pending_tool_result"] = tool_result
        self._touch()
        if self.progress.get("status") != "completed" and self.screen()["screen_id"] == "discover-tool":
            self._advance()
        return tool_result

    def inspect_tool_result(self, tool_result: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.progress is not None
        observed = tool_result or self.progress.get("pending_tool_result")
        if not _valid_tool_result(observed):
            raise RuntimeError("run the structured surface tool before inspecting its result")
        self._observe_tool_result(observed)
        return observed

    def _observe_tool_result(self, tool_result: dict[str, Any]) -> None:
        assert self.progress is not None
        if not _valid_tool_result(tool_result):
            raise ValueError("invalid surface tool result")
        self._record_evidence(FIRST_SUCCESS_FACT, True)
        if self.progress.get("status") == "completed":
            self.flush()
            return
        current = self.screen()
        if (
            current["screen_id"] != "observe-result"
            or current.get("transitions")
            or not _evaluate(current.get("completion_evidence"), self.progress["evidence"])
        ):
            raise RuntimeError("first success was observed outside the completion screen")
        screen_id = current["screen_id"]
        self.progress["completed_screen_ids"] = list(
            dict.fromkeys(self.progress.get("completed_screen_ids", []) + [screen_id])
        )
        self.progress["status"] = "completed"
        self._touch()
        properties = {
            "fact": FIRST_SUCCESS_FACT,
            "tool_id": tool_result["tool_id"],
            "result_keys": sorted(tool_result["result"]),
            "surface_count": len(tool_result["result"]["surface"]),
        }
        self.emit("onboarding_step_completed", properties, screen_id)
        self.emit("onboarding_first_success_observed", properties, screen_id)
        self.emit("onboarding_completed", properties, screen_id)
        self.flush()

    def abandon(self) -> None:
        assert self.progress is not None
        if self.progress.get("status") == "in_progress":
            self.progress["status"] = "abandoned"
            self._touch()
            self.emit("onboarding_abandoned")
            self.flush()

    def reset(self) -> None:
        assert self.bundle is not None
        previous_screen = self.progress.get("current_screen_id") if self.progress else self.bundle["definition"]["entry_screen_id"]
        self.progress = self._new_progress()
        self.state.setdefault("progress", {})[self.subject_hash] = self.progress
        self._save()
        self.emit("onboarding_reset", screen_id=previous_screen)
        self.emit("onboarding_started")
        self.flush()


def start_onboarding() -> OnboardingJourney:
    """Start or resume the canonical first-use journey."""
    return OnboardingJourney().start()


def run_tool() -> dict[str, Any]:
    """Execute the documented safe surface operation and retain its result."""
    journey = start_onboarding()
    journey.expose()
    return journey.run_tool()


def inspect_tool_result() -> dict[str, Any]:
    """Validate the retained structured result and complete first use."""
    journey = start_onboarding()
    journey.expose()
    return journey.inspect_tool_result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Wisent Tools' durable first-use journey.")
    parser.add_argument(
        "operation",
        nargs="?",
        choices=("start", "status", "run-tool", "inspect", "run", "reset", "abandon"),
        default="start",
    )
    args = parser.parse_args(argv)
    try:
        journey = start_onboarding()
        if args.operation in {"start", "status"}:
            journey.expose()
            output = journey.snapshot()
        elif args.operation == "run-tool":
            journey.expose()
            output = {"tool_call": journey.run_tool(), "onboarding": journey.snapshot()}
        elif args.operation == "inspect":
            journey.expose()
            output = {
                "tool_call": journey.inspect_tool_result(),
                "onboarding": journey.snapshot(),
            }
        elif args.operation == "run":
            journey.expose()
            tool_call = journey.run_tool()
            if journey.progress and journey.progress.get("status") != "completed":
                journey.expose()
            journey.inspect_tool_result(tool_call)
            output = {"tool_call": tool_call, "onboarding": journey.snapshot()}
        elif args.operation == "reset":
            journey.reset()
            output = journey.snapshot()
        else:
            journey.abandon()
            output = journey.snapshot()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "result": output}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
