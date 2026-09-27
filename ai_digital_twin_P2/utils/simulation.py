"""
simulation.py

Core simulation engine for the AI-Enabled Digital Twin of the Glass
Production Line.

This module contains placeholder (heuristic) simulation logic that
models how changes in capacity, downtime, bottleneck station, layout,
individual per-station adjustments, and discrete **work-style
combinations** affect overall line throughput, per-station utilization,
and delays.

The logic here is intentionally simple and transparent so it can be
validated by process engineers before being replaced or augmented by
a discrete-event simulation engine (e.g. SimPy) or a trained ML model.

Future improvements:
    - Replace with a proper discrete-event simulation (SimPy).
    - Incorporate real-time IoT sensor feeds for live state.
    - Use ML-predicted cycle times (see models/placeholder_model.py)
      instead of static historical cycle times.
    - Model queueing/buffer dynamics between stations explicitly.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.work_styles import SEARCHABLE_WORK_STYLES, WORK_STYLES, station_adjustments_from_modes

STATIONS = [
    "Raw Material Preparation",
    "Melting Furnace",
    "Forming & Molding",
    "Annealing",
    "Inspection",
    "Packaging",
]


def load_sample_data(path: str) -> pd.DataFrame:
    """
    Load the sample/historical station data from a CSV file.

    Args:
        path: Path to the CSV file with columns
            Station, CycleTime, FailureRate, Utilization.

    Returns:
        A pandas DataFrame with the loaded data.
    """
    df = pd.read_csv(path)
    return df


def run_simulation(
    base_df: pd.DataFrame,
    capacity_change_pct: float = 0.0,
    downtime_minutes: float = 0.0,
    bottleneck_station: Optional[str] = None,
    layout_modified: bool = False,
    station_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """
    Run a what-if simulation over the production line.

    This is placeholder logic that approximates how the line would
    respond to the given scenario parameters. It is deterministic
    given the same inputs (no randomness) so that scenario comparisons
    are stable and reproducible.

    Args:
        base_df: DataFrame with baseline station data
            (Station, CycleTime, FailureRate, Utilization).
        capacity_change_pct: Percentage change in capacity applied to
            EVERY station as a baseline (e.g. +10 for +10% faster,
            -15 for -15% slower). Individual station adjustments
            (see station_adjustments) are added on top of this.
        downtime_minutes: Additional unplanned downtime (minutes)
            injected into the selected bottleneck station only.
        bottleneck_station: Name of the station that receives the
            line-wide `downtime_minutes` value above. If None, no
            station receives the line-wide downtime (per-station
            downtime can still be set via station_adjustments).
        layout_modified: If True, applies a modest efficiency gain to
            EVERY station to model the effect of a layout/material-flow
            improvement (e.g. reduced transport time between stations).
        station_adjustments: Optional dict keyed by station name, each
            value a dict that may contain:
                - 'capacity_delta_pct': extra capacity change (%) applied
                  ONLY to this station, on top of the line-wide
                  capacity_change_pct.
                - 'extra_downtime_minutes': extra downtime (minutes)
                  applied ONLY to this station, on top of any line-wide
                  downtime it may already receive as the bottleneck.
            This allows planners to fine-tune individual workstations
            instead of only applying line-wide changes.

    Returns:
        A dictionary containing:
            - 'station_results': DataFrame with per-station simulated
              CycleTime, Utilization, and Delay (minutes).
            - 'throughput_units_per_hour': float, line throughput.
            - 'avg_utilization': float, average utilization across
              stations.
            - 'total_delay_minutes': float, total delay across the line.
            - 'bottleneck_station': str, the station with the highest
              resulting (post-simulation) cycle time.
    """
    df = base_df.copy()
    station_adjustments = station_adjustments or {}

    own_delay = {}
    new_cycle_times = {}
    new_utilizations = {}

    for _, row in df.iterrows():
        station = row["Station"]
        adj = station_adjustments.get(station, {})
        capacity_delta_pct = adj.get("capacity_delta_pct", 0.0)
        extra_downtime = adj.get("extra_downtime_minutes", 0.0)

        effective_capacity_pct = capacity_change_pct + capacity_delta_pct
        effective_downtime = extra_downtime
        if bottleneck_station and station == bottleneck_station:
            effective_downtime += downtime_minutes

        # 1. Capacity change: scales cycle time inversely.
        capacity_factor = 1.0 - (effective_capacity_pct / 100.0)
        capacity_factor = max(capacity_factor, 0.1)  # guard against near-zero
        cycle_time = row["CycleTime"] * capacity_factor
        utilization = row["Utilization"]

        # 2. Layout modification: flat efficiency improvement, applied
        #    line-wide (reduced transport/handling time).
        if layout_modified:
            cycle_time *= 0.92  # ~8% efficiency gain
            utilization = min(utilization * 0.95, 0.98)

        # 3. Station-specific downtime: increases effective cycle time,
        #    utilization, and creates a direct delay at this station.
        cycle_time += effective_downtime
        utilization = min(utilization + (effective_downtime / 480.0), 0.99)

        new_cycle_times[station] = cycle_time
        new_utilizations[station] = utilization
        own_delay[station] = effective_downtime

    # 4. Propagate delay downstream: a station running late starves
    #    every station after it. We carry forward a fraction (30%) of
    #    delay as it cascades down the line.
    station_order = [s for s in STATIONS if s in df["Station"].values]
    final_delay = {}
    running_carry = 0.0
    for station in station_order:
        total = own_delay.get(station, 0.0) + running_carry
        final_delay[station] = total
        running_carry += own_delay.get(station, 0.0) * 0.3
        # propagated portion also nudges utilization up slightly
        propagated_extra = total - own_delay.get(station, 0.0)
        if propagated_extra > 0:
            new_utilizations[station] = min(
                new_utilizations[station] + (propagated_extra / 960.0), 0.99
            )

    df["CycleTime"] = df["Station"].map(new_cycle_times)
    df["Utilization"] = df["Station"].map(new_utilizations)
    df["Delay"] = df["Station"].map(final_delay)

    # 5. Bottleneck is auto-identified post-simulation as whichever
    #    station now has the highest cycle time (the true pacing
    #    constraint of the line), regardless of which station the
    #    user targeted with downtime.
    detected_bottleneck = df.loc[df["CycleTime"].idxmax(), "Station"]

    # 6. Line throughput is paced by the slowest station (units/hour).
    slowest_cycle_time = df["CycleTime"].max()
    throughput_units_per_hour = 60.0 / slowest_cycle_time if slowest_cycle_time > 0 else 0.0

    avg_utilization = float(df["Utilization"].mean())
    total_delay_minutes = float(df["Delay"].sum())

    df["CycleTime"] = df["CycleTime"].round(2)
    df["Utilization"] = df["Utilization"].round(3)
    df["Delay"] = df["Delay"].round(2)

    return {
        "station_results": df,
        "throughput_units_per_hour": round(throughput_units_per_hour, 2),
        "avg_utilization": round(avg_utilization, 3),
        "total_delay_minutes": round(total_delay_minutes, 2),
        "bottleneck_station": detected_bottleneck,
    }


def compare_scenarios(
    baseline_result: Dict,
    scenario_result: Dict,
) -> pd.DataFrame:
    """
    Build a small comparison table between the baseline and a
    what-if scenario, for key line-level metrics.

    Args:
        baseline_result: Output dict from run_simulation() for the
            baseline (no changes) scenario.
        scenario_result: Output dict from run_simulation() for the
            what-if scenario.

    Returns:
        A DataFrame with metric, baseline value, scenario value, and
        the percentage change.
    """
    metrics = [
        ("Throughput (units/hr)", "throughput_units_per_hour"),
        ("Avg. Utilization", "avg_utilization"),
        ("Total Delay (min)", "total_delay_minutes"),
    ]

    rows: List[Dict] = []
    for label, key in metrics:
        base_val = baseline_result[key]
        scen_val = scenario_result[key]
        pct_change = (
            ((scen_val - base_val) / base_val) * 100.0 if base_val != 0 else 0.0
        )
        rows.append({
            "Metric": label,
            "Baseline": base_val,
            "Scenario": scen_val,
            "Change (%)": round(pct_change, 2),
        })

    return pd.DataFrame(rows)


def _score_combo(
    objective: str,
    throughput: float,
    total_risk: float,
    approx_delay: float,
    avg_energy: float,
) -> float:
    """
    Compute a single comparable score for a candidate work-style
    combination, given the planner's chosen optimization objective.

    Higher scores are always better. Weights below are illustrative,
    chosen so each objective clearly favors the metric it names while
    still avoiding absurd outcomes (e.g. "Maximize Throughput" won't
    pick a combination that also creates enormous unnecessary delay).

    Args:
        objective: one of utils.work_styles.OBJECTIVE_OPTIONS.
        throughput: estimated line throughput (units/hour).
        total_risk: sum of per-station failure-risk scores.
        approx_delay: rough estimate of total cascading delay (minutes).
        avg_energy: average relative energy index across stations.

    Returns:
        A float score; higher is better for the chosen objective.
    """
    if objective == "Maximize Throughput":
        return throughput * 2.0 - approx_delay * 0.01
    if objective == "Minimize Failure Risk":
        return -total_risk * 100.0 + throughput * 0.05
    if objective == "Minimize Energy Use":
        return -avg_energy * 20.0 + throughput * 0.1
    # Default: "Balanced (All Factors)"
    return throughput * 2.0 - total_risk * 50.0 - avg_energy * 1.5 - approx_delay * 0.02


def find_best_combinations(
    base_df: pd.DataFrame,
    capacity_change_pct: float = 0.0,
    downtime_minutes: float = 0.0,
    bottleneck_station: Optional[str] = None,
    layout_modified: bool = False,
    objective: str = "Balanced (All Factors)",
    top_n: int = 5,
) -> List[Dict]:
    """
    Search across work-style combinations for every station to find the
    combinations that best satisfy the chosen objective.

    This performs a full brute-force search over
    len(SEARCHABLE_WORK_STYLES) ** len(stations) combinations using a
    cheap, vectorization-friendly approximate estimate (no pandas
    overhead per combination -- fast even for thousands of combos).
    The best `top_n` candidates are then re-simulated with the full,
    accurate `run_simulation` engine (including proper cascading delay)
    so the numbers ultimately shown to the user are trustworthy.

    Args:
        base_df: DataFrame with baseline station data.
        capacity_change_pct: whole-line capacity change (%) to hold
            constant while searching station work styles.
        downtime_minutes: whole-line downtime (minutes) applied to
            `bottleneck_station`, held constant during the search.
        bottleneck_station: station receiving the whole-line downtime.
        layout_modified: whether the whole-line layout improvement is
            active during the search.
        objective: one of utils.work_styles.OBJECTIVE_OPTIONS,
            determining how candidate combinations are ranked.
        top_n: how many top combinations to return, fully re-simulated.

    Returns:
        A list of up to `top_n` dicts, best first, each containing:
            - 'combo': {station: work_style_name}
            - 'simulation': the accurate run_simulation() result dict
            - 'risk_rows': list of {'Station', 'risk_score'} dicts
            - 'avg_risk': float, average risk score across stations
            - 'avg_energy_index': float, average relative energy index
    """
    stations = [s for s in STATIONS if s in base_df["Station"].values]
    base_lookup = {row.Station: row for row in base_df.itertuples()}
    modes = SEARCHABLE_WORK_STYLES

    scored_combos = []
    for combo_indices in itertools.product(range(len(modes)), repeat=len(stations)):
        combo = {stations[i]: modes[combo_indices[i]] for i in range(len(stations))}

        cycle_times, utilizations, downtimes, risks, energies = [], [], [], [], []
        for station in stations:
            style = WORK_STYLES[combo[station]]
            base_row = base_lookup[station]

            effective_capacity_pct = capacity_change_pct + style["capacity_delta_pct"]
            capacity_factor = max(1.0 - (effective_capacity_pct / 100.0), 0.1)
            cycle_time = base_row.CycleTime * capacity_factor
            if layout_modified:
                cycle_time *= 0.92

            downtime = style["extra_downtime_minutes"]
            if bottleneck_station and station == bottleneck_station:
                downtime += downtime_minutes
            cycle_time += downtime

            utilization = base_row.Utilization
            if layout_modified:
                utilization = min(utilization * 0.95, 0.98)
            utilization = min(utilization + downtime / 480.0, 0.99)

            risk = min(base_row.FailureRate * style["risk_multiplier"] * (1 + utilization), 1.0)

            cycle_times.append(cycle_time)
            utilizations.append(utilization)
            downtimes.append(downtime)
            risks.append(risk)
            energies.append(style["energy_multiplier"])

        bottleneck_cycle_time = max(cycle_times)
        throughput = 60.0 / bottleneck_cycle_time if bottleneck_cycle_time > 0 else 0.0
        total_risk = sum(risks)
        approx_delay = sum(downtimes) * 1.25  # rough cascade approximation
        avg_energy = sum(energies) / len(energies)

        score = _score_combo(objective, throughput, total_risk, approx_delay, avg_energy)
        scored_combos.append({"combo": combo, "score": score})

    scored_combos.sort(key=lambda item: item["score"], reverse=True)
    top_candidates = scored_combos[:top_n]

    # Re-simulate the top candidates with the full, accurate engine.
    refined_results = []
    for candidate in top_candidates:
        combo = candidate["combo"]
        adjustments = station_adjustments_from_modes(combo)

        sim_result = run_simulation(
            base_df,
            capacity_change_pct=capacity_change_pct,
            downtime_minutes=downtime_minutes,
            bottleneck_station=bottleneck_station,
            layout_modified=layout_modified,
            station_adjustments=adjustments,
        )

        risk_rows = []
        for station in stations:
            style = WORK_STYLES[combo[station]]
            base_row = base_lookup[station]
            util = sim_result["station_results"].loc[
                sim_result["station_results"]["Station"] == station, "Utilization"
            ].values[0]
            adjusted_risk = min(base_row.FailureRate * style["risk_multiplier"] * (1 + util), 1.0)
            risk_rows.append({"Station": station, "risk_score": round(adjusted_risk, 4)})

        avg_risk = round(sum(r["risk_score"] for r in risk_rows) / len(risk_rows), 4)
        avg_energy_index = round(
            sum(WORK_STYLES[combo[s]]["energy_multiplier"] for s in stations) / len(stations), 3
        )

        refined_results.append({
            "combo": combo,
            "simulation": sim_result,
            "risk_rows": risk_rows,
            "avg_risk": avg_risk,
            "avg_energy_index": avg_energy_index,
        })

    return refined_results
