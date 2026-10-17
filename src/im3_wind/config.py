from __future__ import annotations

from dataclasses import dataclass

HISTORICAL_SCENARIO = "historical"
FUTURE_SCENARIOS = (
    "rcp45cooler",
    "rcp45hotter",
    "rcp85cooler",
    "rcp85hotter",
)
SCENARIOS = (HISTORICAL_SCENARIO,) + FUTURE_SCENARIOS

# These match the return periods currently used by CADENCE/CONUS404.
# Add additional design storms here later (for example 5 or 20 years) if the
# CADENCE UI expands its design-storm choices.
DEFAULT_RETURN_PERIODS = (10, 25, 50, 100, 200, 500)

# IM3/HyperFACETS TGW periods.
HISTORICAL_START = 1980
HISTORICAL_END = 2019
NEAR_FUTURE_START = 2020
NEAR_FUTURE_END = 2059
FAR_FUTURE_START = 2060
FAR_FUTURE_END = 2099

# CADENCE V1 only publishes future deltas through 2050.
CADENCE_START = 2025
CADENCE_END = 2050


@dataclass(frozen=True)
class ClimatePeriod:
    name: str
    start_year: int
    end_year: int

    @property
    def midpoint(self) -> float:
        return (self.start_year + self.end_year) / 2.0


PERIODS = (
    ClimatePeriod("historical", HISTORICAL_START, HISTORICAL_END),
    ClimatePeriod("near_future", NEAR_FUTURE_START, NEAR_FUTURE_END),
    ClimatePeriod("far_future", FAR_FUTURE_START, FAR_FUTURE_END),
)


def validate_scenario(scenario: str) -> str:
    value = scenario.strip().lower()
    if value not in SCENARIOS:
        raise ValueError(f"Unknown IM3 scenario {scenario!r}; expected one of {SCENARIOS}")
    return value
