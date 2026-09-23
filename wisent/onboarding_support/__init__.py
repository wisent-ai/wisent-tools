"""Product-owned first-use journey for Wisent Tools."""
from .definition import *
from .journey import OnboardingJourney
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

from .cli import main
