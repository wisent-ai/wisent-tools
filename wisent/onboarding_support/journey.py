"""Durable first-use onboarding state machine."""
from .runtime import *
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
