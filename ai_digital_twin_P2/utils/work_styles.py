"""
work_styles.py

Defines the discrete "Work Style" presets that can be applied to any
individual station on the production line, plus helpers to convert a
chosen work style into the capacity/downtime adjustments consumed by
utils.simulation.run_simulation.

A work style is a bundle of trade-offs a planner can pick for a station
instead of manually tuning raw percentages -- e.g. "run this station in
High-Speed mode" instead of "increase capacity by 15% and accept more
downtime". This mirrors how a real operator would think about a machine's
operating mode.

Each style defines:
    - capacity_delta_pct: extra capacity change (%) vs. the whole-line
      setting (positive = faster/shorter cycle time).
    - extra_downtime_minutes: extra unplanned stoppage time (minutes)
      typical of running in that mode.
    - risk_multiplier: multiplies the station's baseline failure rate
      (​>1 = more failure-prone, <1 = safer).
    - energy_multiplier: relative energy/resource cost index
      (1.0 = baseline "Standard" consumption).
    - description: plain-English explanation shown to the user.
"""

from __future__ import annotations

from typing import Dict, List

WORK_STYLES: Dict[str, Dict] = {
    "Standard": {
        "capacity_delta_pct": 0,
        "extra_downtime_minutes": 0,
        "risk_multiplier": 1.0,
        "energy_multiplier": 1.0,
        "description": (
            "Normal operating parameters — the historical baseline for "
            "this station. No trade-offs applied."
        ),
    },
    "High-Speed": {
        "capacity_delta_pct": 15,
        "extra_downtime_minutes": 3,
        "risk_multiplier": 1.35,
        "energy_multiplier": 1.30,
        "description": (
            "Pushes the station to run faster than normal. Cuts cycle "
            "time noticeably, but raises failure risk and energy use, "
            "and adds a little extra downtime from more frequent "
            "micro-stops."
        ),
    },
    "Energy-Saving": {
        "capacity_delta_pct": -8,
        "extra_downtime_minutes": 0,
        "risk_multiplier": 0.85,
        "energy_multiplier": 0.65,
        "description": (
            "Runs the station at a gentler pace to cut energy "
            "consumption. Slightly slower output, but easier on the "
            "equipment and noticeably cheaper to run."
        ),
    },
    "Quality-Focus": {
        "capacity_delta_pct": -10,
        "extra_downtime_minutes": 2,
        "risk_multiplier": 0.55,
        "energy_multiplier": 1.05,
        "description": (
            "Prioritizes precision and consistency over speed. Extra "
            "checks/adjustments slow the station down a little, but "
            "substantially cut failure risk."
        ),
    },
    "Maintenance Mode": {
        "capacity_delta_pct": -45,
        "extra_downtime_minutes": 20,
        "risk_multiplier": 0.30,
        "energy_multiplier": 0.55,
        "description": (
            "Station runs at reduced load for scheduled maintenance or "
            "inspection. Large short-term hit to output, but resets "
            "failure risk down for future runs. Excluded from the "
            "automatic combination search since it deliberately reduces "
            "output rather than being a production alternative."
        ),
    },
}

# Modes considered by the automatic "find best combination" search.
# Maintenance Mode is intentionally excluded (see description above) --
# it can still be picked manually per station.
SEARCHABLE_WORK_STYLES: List[str] = [
    "Standard", "High-Speed", "Energy-Saving", "Quality-Focus"
]

OBJECTIVE_OPTIONS = [
    "Balanced (All Factors)",
    "Maximize Throughput",
    "Minimize Failure Risk",
    "Minimize Energy Use",
]

OBJECTIVE_HELP = {
    "Balanced (All Factors)": (
        "Looks for the combination with the best overall trade-off "
        "between throughput, failure risk, energy use, and delay -- no "
        "single factor dominates."
    ),
    "Maximize Throughput": (
        "Looks for the combination that produces the most units per "
        "hour, largely ignoring the resulting risk or energy cost."
    ),
    "Minimize Failure Risk": (
        "Looks for the combination with the lowest average failure-risk "
        "score across all stations, even if that costs some throughput."
    ),
    "Minimize Energy Use": (
        "Looks for the combination with the lowest relative energy "
        "index across all stations, even if that costs some throughput."
    ),
}


def station_adjustments_from_modes(mode_selections: Dict[str, str]) -> Dict[str, Dict]:
    """
    Convert a {station: work_style_name} mapping into the
    station_adjustments format expected by utils.simulation.run_simulation.

    Args:
        mode_selections: dict mapping station name -> work style name
            (a key of WORK_STYLES).

    Returns:
        dict mapping station name -> {'capacity_delta_pct': ...,
        'extra_downtime_minutes': ..., 'mode': ..., 'risk_multiplier': ...,
        'energy_multiplier': ...}. Extra keys beyond capacity_delta_pct /
        extra_downtime_minutes are ignored by run_simulation but are kept
        here for downstream risk/energy reporting.
    """
    adjustments = {}
    for station, mode_name in mode_selections.items():
        style = WORK_STYLES.get(mode_name, WORK_STYLES["Standard"])
        adjustments[station] = {
            "capacity_delta_pct": style["capacity_delta_pct"],
            "extra_downtime_minutes": style["extra_downtime_minutes"],
            "mode": mode_name,
            "risk_multiplier": style["risk_multiplier"],
            "energy_multiplier": style["energy_multiplier"],
        }
    return adjustments
