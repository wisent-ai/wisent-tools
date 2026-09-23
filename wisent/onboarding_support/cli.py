"""Command-line adapter for the onboarding journey."""
from . import *
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
