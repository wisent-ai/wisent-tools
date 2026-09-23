"""Product-owned first-use journey for a safe Wisent Tools operation."""

from wisent.onboarding_support import (
    OnboardingJourney,
    inspect_tool_result,
    main,
    run_tool,
    start_onboarding,
)

__all__ = [
    "OnboardingJourney",
    "inspect_tool_result",
    "main",
    "run_tool",
    "start_onboarding",
]


if __name__ == "__main__":
    raise SystemExit(main())
