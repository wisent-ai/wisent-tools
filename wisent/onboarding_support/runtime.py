"""Persistence, transport, and local execution for onboarding."""
from .definition import *
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
    root = Path(__file__).resolve().parent.parent.parent
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

__all__ = [name for name in globals() if not name.startswith("__")]
