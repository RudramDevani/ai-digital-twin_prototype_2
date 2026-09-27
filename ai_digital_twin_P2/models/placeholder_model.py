"""
placeholder_model.py

Placeholder ML/AI model(s) for the AI-Enabled Digital Twin of the Glass
Production Line.

This module currently implements simple, explainable heuristics that
stand in for future trained models. The intent is to keep the same
public interface (class names + method signatures) so that these
placeholders can later be swapped out for real, trained ML models
(e.g. scikit-learn, XGBoost, or a deep learning model) or a real LLM
reasoning layer without changing the rest of the application.

Classes:
    - CycleTimePredictor: predicts adjusted station cycle time.
    - FailurePatternModel: estimates failure risk per station.
    - AIAdvisor: turns simulation results into plain-English analysis
      and actionable improvement suggestions (the "AI Analysis" and
      "AI Suggestions" features of the app).

Future improvements:
    - Replace heuristic logic with trained regression models
      (e.g. Gradient Boosting / Random Forest) for cycle time prediction.
    - Replace failure pattern logic with a classifier trained on
      historical sensor + maintenance data.
    - Replace AIAdvisor's rule-based text generation with a real LLM
      call (e.g. via the Anthropic API) grounded in the same simulation
      data, for richer, more nuanced natural-language recommendations.
    - Add model persistence (joblib/pickle) and versioning.
    - Add confidence intervals / uncertainty estimates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class PredictionResult:
    """Simple container for a single prediction and its metadata."""

    station: str
    predicted_cycle_time: float
    confidence: float


class CycleTimePredictor:
    """
    Placeholder model for predicting station cycle time.

    In production, this class would wrap a trained regression model
    (e.g. scikit-learn Pipeline) that takes in features such as:
        - historical cycle time
        - current utilization
        - upstream/downstream buffer levels
        - maintenance history
    and returns a predicted cycle time in seconds/minutes.

    For now, it applies a simple heuristic adjustment based on
    utilization and a random noise term to emulate model variance.
    """

    def __init__(self, random_seed: int = 42):
        self._rng = random.Random(random_seed)

    def predict(self, station: str, base_cycle_time: float,
                utilization: float) -> PredictionResult:
        """
        Predict the adjusted cycle time for a station.

        Args:
            station: Name of the production station.
            base_cycle_time: Historical/nominal cycle time (minutes).
            utilization: Current utilization fraction (0.0 - 1.0).

        Returns:
            PredictionResult with predicted cycle time and a mock
            confidence score.
        """
        congestion_penalty = max(0.0, utilization - 0.75) * 0.4
        noise = self._rng.uniform(-0.03, 0.03)
        predicted = base_cycle_time * (1 + congestion_penalty + noise)

        confidence = round(1.0 - abs(noise) - congestion_penalty * 0.5, 3)
        confidence = max(0.5, min(confidence, 0.99))

        return PredictionResult(
            station=station,
            predicted_cycle_time=round(predicted, 2),
            confidence=confidence,
        )

    def predict_batch(self, stations_data: List[Dict]) -> List[PredictionResult]:
        """
        Run predictions for a list of station records.

        Args:
            stations_data: List of dicts, each with keys
                'Station', 'CycleTime', 'Utilization'.

        Returns:
            List of PredictionResult objects.
        """
        results = []
        for row in stations_data:
            result = self.predict(
                station=row["Station"],
                base_cycle_time=float(row["CycleTime"]),
                utilization=float(row["Utilization"]),
            )
            results.append(result)
        return results


class FailurePatternModel:
    """
    Placeholder model for detecting/predicting failure risk patterns.

    In production, this would be a classifier trained on historical
    failure logs, sensor readings (temperature, vibration, pressure),
    and maintenance records, returning a probability of failure in
    the next N hours/cycles.

    For now, it derives a simple risk score from the station's
    historical failure rate and current utilization.
    """

    def __init__(self, risk_threshold: float = 0.05):
        self.risk_threshold = risk_threshold

    def assess_risk(self, station: str, failure_rate: float,
                     utilization: float) -> Dict:
        """
        Estimate a failure risk score for a station.

        Args:
            station: Name of the production station.
            failure_rate: Historical failure rate (0.0 - 1.0).
            utilization: Current utilization fraction (0.0 - 1.0).

        Returns:
            Dict with station name, risk_score, and risk_level.
        """
        risk_score = failure_rate * (1 + utilization)
        risk_score = round(min(risk_score, 1.0), 4)

        if risk_score >= self.risk_threshold * 2:
            risk_level = "High"
        elif risk_score >= self.risk_threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "Station": station,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

    def assess_batch(self, stations_data: List[Dict]) -> List[Dict]:
        """
        Assess failure risk for a list of station records.

        Args:
            stations_data: List of dicts, each with keys
                'Station', 'FailureRate', 'Utilization'.

        Returns:
            List of risk assessment dicts.
        """
        results = []
        for row in stations_data:
            result = self.assess_risk(
                station=row["Station"],
                failure_rate=float(row["FailureRate"]),
                utilization=float(row["Utilization"]),
            )
            results.append(result)
        return results


class AIAdvisor:
    """
    Placeholder "AI Analysis & Suggestions" engine.

    This class inspects the outputs of a simulation run (and, where
    available, the outputs of CycleTimePredictor / FailurePatternModel)
    and produces:
        1. A plain-English **analysis** of what is happening on the
           line right now (bottlenecks, utilization hot-spots, delay
           sources, biggest changes vs. baseline).
        2. A set of prioritized, actionable **suggestions** a planner
           could act on, each tagged with a severity level so the UI
           can color-code them (critical / warning / info / positive).

    The logic below is rule-based and fully deterministic so it is
    transparent and explainable -- a reasonable placeholder ahead of
    wiring this up to a real LLM or trained recommender model.
    """

    # Thresholds used to decide when something is worth flagging.
    HIGH_UTILIZATION = 0.90
    MODERATE_UTILIZATION = 0.80
    SIGNIFICANT_THROUGHPUT_DROP_PCT = -5.0
    SIGNIFICANT_DELAY_MINUTES = 15.0

    def generate_analysis(
        self,
        scenario_result: Dict,
        baseline_result: Dict,
    ) -> List[str]:
        """
        Produce a list of plain-English analysis statements comparing
        the scenario to the baseline.

        Args:
            scenario_result: Output dict from utils.simulation.run_simulation
                for the what-if scenario.
            baseline_result: Output dict from utils.simulation.run_simulation
                for the baseline (no changes) run.

        Returns:
            List of human-readable analysis strings.
        """
        insights: List[str] = []
        df = scenario_result["station_results"]
        base_df = baseline_result["station_results"]

        # 1. Bottleneck identification.
        bottleneck = scenario_result["bottleneck_station"]
        bottleneck_row = df[df["Station"] == bottleneck].iloc[0]
        insights.append(
            f"**{bottleneck}** is currently the line's pacing constraint "
            f"(bottleneck), with a cycle time of **{bottleneck_row['CycleTime']} min** "
            f"and utilization of **{bottleneck_row['Utilization'] * 100:.1f}%**."
        )

        # 2. Throughput comparison.
        thr_base = baseline_result["throughput_units_per_hour"]
        thr_scen = scenario_result["throughput_units_per_hour"]
        thr_change_pct = ((thr_scen - thr_base) / thr_base * 100.0) if thr_base else 0.0
        direction = "increased" if thr_change_pct >= 0 else "decreased"
        insights.append(
            f"Line throughput has **{direction} by {abs(thr_change_pct):.1f}%** "
            f"({thr_base} → {thr_scen} units/hr) versus the baseline."
        )

        # 3. Utilization hot-spots.
        hot_stations = df[df["Utilization"] >= self.HIGH_UTILIZATION]["Station"].tolist()
        if hot_stations:
            insights.append(
                "The following station(s) are running above **90% utilization**, "
                f"leaving little slack for variability: **{', '.join(hot_stations)}**."
            )

        # 4. Delay source.
        total_delay = scenario_result["total_delay_minutes"]
        if total_delay > 0:
            worst_delay_row = df.loc[df["Delay"].idxmax()]
            insights.append(
                f"Total simulated delay across the line is **{total_delay} min**, "
                f"with **{worst_delay_row['Station']}** contributing the most "
                f"({worst_delay_row['Delay']} min), including any cascading effect "
                "on downstream stations."
            )
        else:
            insights.append("No delay was introduced in this scenario.")

        # 5. Largest cycle-time shift vs. baseline (which station changed most).
        merged = df[["Station", "CycleTime"]].merge(
            base_df[["Station", "CycleTime"]], on="Station", suffixes=("_scenario", "_baseline")
        )
        merged["delta"] = merged["CycleTime_scenario"] - merged["CycleTime_baseline"]
        biggest = merged.loc[merged["delta"].abs().idxmax()]
        if abs(biggest["delta"]) > 0.01:
            direction2 = "slower" if biggest["delta"] > 0 else "faster"
            insights.append(
                f"**{biggest['Station']}** shows the largest change versus baseline: "
                f"**{abs(biggest['delta']):.2f} min {direction2}** "
                f"({biggest['CycleTime_baseline']} → {biggest['CycleTime_scenario']} min)."
            )

        return insights

    def generate_suggestions(
        self,
        scenario_result: Dict,
        risk_df: pd.DataFrame,
    ) -> List[Dict[str, str]]:
        """
        Produce prioritized, actionable suggestions for the planner.

        Args:
            scenario_result: Output dict from utils.simulation.run_simulation.
            risk_df: DataFrame with failure risk assessments
                (Station, risk_score, risk_level), as produced by
                FailurePatternModel.assess_batch().

        Returns:
            List of dicts, each with keys:
                - 'message': the suggestion text.
                - 'severity': one of 'critical', 'warning', 'info', 'success'
                  (intended to map to UI color: red, orange, blue, green).
        """
        suggestions: List[Dict[str, str]] = []
        df = scenario_result["station_results"]

        # 1. High-risk stations -> preventive maintenance.
        high_risk = risk_df[risk_df["risk_level"] == "High"]["Station"].tolist()
        for station in high_risk:
            suggestions.append({
                "message": (
                    f"**{station}** has a HIGH failure-risk score. Schedule "
                    "preventive maintenance or inspection before running this "
                    "scenario on the live line to avoid unplanned stoppages."
                ),
                "severity": "critical",
            })

        medium_risk = risk_df[risk_df["risk_level"] == "Medium"]["Station"].tolist()
        for station in medium_risk:
            suggestions.append({
                "message": (
                    f"**{station}** shows MEDIUM failure risk. Consider adding it "
                    "to the next maintenance cycle and monitoring closely."
                ),
                "severity": "warning",
            })

        # 2. Utilization-based suggestions.
        for _, row in df.iterrows():
            if row["Utilization"] >= self.HIGH_UTILIZATION:
                suggestions.append({
                    "message": (
                        f"**{row['Station']}** is running at "
                        f"**{row['Utilization'] * 100:.0f}%** utilization. "
                        "Consider increasing capacity here, adding a buffer "
                        "upstream, or redistributing load to reduce strain."
                    ),
                    "severity": "warning",
                })

        # 3. Bottleneck-specific suggestion.
        bottleneck = scenario_result["bottleneck_station"]
        bottleneck_row = df[df["Station"] == bottleneck].iloc[0]
        suggestions.append({
            "message": (
                f"Since **{bottleneck}** paces the entire line, a small capacity "
                "improvement here will have the largest impact on overall "
                "throughput -- more so than improving any other single station."
            ),
            "severity": "info",
        })

        # 4. Delay-driven suggestion.
        if scenario_result["total_delay_minutes"] >= self.SIGNIFICANT_DELAY_MINUTES:
            suggestions.append({
                "message": (
                    f"Total delay of **{scenario_result['total_delay_minutes']} min** "
                    "is significant. Investigate root causes of downtime at the "
                    "affected station(s) and evaluate whether a layout "
                    "modification could reduce cascading delay to downstream "
                    "stations."
                ),
                "severity": "critical",
            })

        # 5. Positive reinforcement when things look healthy.
        if not high_risk and not medium_risk and df["Utilization"].max() < self.MODERATE_UTILIZATION:
            suggestions.append({
                "message": (
                    "All stations are operating with healthy utilization and low "
                    "failure risk under this scenario. This configuration looks "
                    "safe to consider for the live line."
                ),
                "severity": "success",
            })

        return suggestions

    def generate_combination_insight(
        self,
        top_result: Dict,
        current_scenario_result: Dict,
        objective: str,
    ) -> str:
        """
        Produce a short, plain-English recommendation paragraph for the
        #1-ranked work-style combination found by
        utils.simulation.find_best_combinations, comparing it against
        the planner's currently configured scenario.

        Args:
            top_result: one entry from find_best_combinations()'s
                returned list (typically the first / best-ranked one).
            current_scenario_result: the run_simulation() output for
                whatever scenario is currently configured in the app,
                used as the comparison baseline for this recommendation.
            objective: the optimization objective string the search was
                run with (e.g. "Balanced (All Factors)").

        Returns:
            A short natural-language recommendation string.
        """
        sim = top_result["simulation"]
        throughput = sim["throughput_units_per_hour"]
        current_throughput = current_scenario_result["throughput_units_per_hour"]
        throughput_diff_pct = (
            (throughput - current_throughput) / current_throughput * 100.0
            if current_throughput else 0.0
        )
        direction = "higher" if throughput_diff_pct >= 0 else "lower"
        bottleneck = sim["bottleneck_station"]

        return (
            f"Optimizing for **{objective}**, the best work-style combination found "
            f"reassigns all 6 stations to reach an estimated **{throughput} units/hr** "
            f"throughput (**{abs(throughput_diff_pct):.1f}% {direction}** than your "
            f"currently configured scenario), an average failure-risk score of "
            f"**{top_result['avg_risk']}**, and a relative energy index of "
            f"**{top_result['avg_energy_index']}** (1.0 = every station on Standard "
            f"mode). **{bottleneck}** remains the pacing bottleneck under this "
            "combination -- so any further improvement should start there."
        )

    def explain_combination_choices(
        self,
        top_result: Dict,
    ) -> List[Dict[str, str]]:
        """
        Build a per-station explanation of the recommended work style,
        for display in a table alongside the recommendation.

        Args:
            top_result: one entry from find_best_combinations()'s
                returned list.

        Returns:
            List of dicts with keys 'Station', 'Recommended Work Style',
            'Cycle Time (min)', 'Utilization', 'Risk Score'.
        """
        combo = top_result["combo"]
        df = top_result["simulation"]["station_results"]
        risk_lookup = {r["Station"]: r["risk_score"] for r in top_result["risk_rows"]}

        rows = []
        for _, row in df.iterrows():
            station = row["Station"]
            rows.append({
                "Station": station,
                "Recommended Work Style": combo.get(station, "Standard"),
                "Cycle Time (min)": float(row["CycleTime"]),
                "Utilization": float(row["Utilization"]),
                "Risk Score": float(risk_lookup.get(station)) if risk_lookup.get(station) is not None else None,
            })
        return rows


if __name__ == "__main__":
    # Simple smoke test when running this file directly.
    sample = [
        {"Station": "Melting Furnace", "CycleTime": 45.0,
         "FailureRate": 0.05, "Utilization": 0.92},
        {"Station": "Inspection", "CycleTime": 8.5,
         "FailureRate": 0.03, "Utilization": 0.65},
    ]

    predictor = CycleTimePredictor()
    for res in predictor.predict_batch(sample):
        print(res)

    risk_model = FailurePatternModel()
    for res in risk_model.assess_batch(sample):
        print(res)
