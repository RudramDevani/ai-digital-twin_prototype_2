"""
streamlit_app.py

Main entry point for the AI-Enabled Digital Twin for Glass Production
Line Streamlit application.

Run locally with:
    streamlit run streamlit_app.py
"""

import os

import pandas as pd
import streamlit as st

from utils.simulation import STATIONS, compare_scenarios, find_best_combinations, load_sample_data, run_simulation
from utils.work_styles import OBJECTIVE_HELP, OBJECTIVE_OPTIONS, WORK_STYLES
from models.placeholder_model import AIAdvisor, CycleTimePredictor, FailurePatternModel

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Enabled Digital Twin | Glass Production Line",
    page_icon="🔥",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_data.csv")

# Severity -> Streamlit alert renderer, used for AI suggestions.
SEVERITY_RENDERERS = {
    "critical": st.error,
    "warning": st.warning,
    "info": st.info,
    "success": st.success,
}
SEVERITY_ICONS = {
    "critical": "🛑",
    "warning": "⚠️",
    "info": "ℹ️",
    "success": "✅",
}


@st.cache_data
def get_baseline_data(path: str) -> pd.DataFrame:
    """Load and cache the baseline station data."""
    return load_sample_data(path)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🔥 AI-Enabled Digital Twin — Glass Production Line")
st.markdown(
    """
    This application is a **virtual replica (digital twin)** of a 6-station
    glass production line. It learns from historical and (eventually) live
    machine data, and lets production planners run **AI-supported what-if
    scenarios** — including adjustments to individual workstations — to
    evaluate capacity changes, downtime events, bottlenecks, and layout
    modifications *before* committing to them on the real line.

    **Production Line Stations:** Raw Material Preparation → Melting Furnace
    → Forming & Molding → Annealing → Inspection → Packaging
    """
)

with st.expander("📘 Glossary — what do these terms mean?", expanded=False):
    st.markdown(
        """
        - **Cycle Time**: the time (in minutes) a station takes to complete
          one unit of work. Lower is faster.
        - **Utilization**: the fraction of available time a station is
          actively busy. Values close to 100% mean the station has very
          little spare capacity.
        - **Delay**: extra time (in minutes) added at a station because of
          downtime or knock-on effects from an upstream station running late.
        - **Throughput**: how many finished units the whole line can produce
          per hour, paced by its slowest station (the bottleneck).
        - **Bottleneck**: the station that currently limits the speed of the
          entire line — improving it has the biggest impact on throughput.
        - **Failure Risk Score**: a placeholder AI estimate of how likely a
          station is to experience an unplanned failure, based on its
          historical failure rate and how hard it is currently being used.
        - **Confidence**: how certain the placeholder AI model is about a
          given prediction (this is illustrative until real ML models are
          trained on historical data).
        """
    )

st.divider()

# --------------------------------------------------------------------------
# Load baseline data
# --------------------------------------------------------------------------
try:
    baseline_df = get_baseline_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find sample data at `{DATA_PATH}`. "
             "Please ensure data/sample_data.csv exists.")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — Global Scenario Controls
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Whole-Line Scenario Controls")
st.sidebar.caption(
    "These settings apply to **every station** as a starting point. "
    "Use the 'Individual Workstation Adjustments' section further down "
    "to fine-tune a specific station on top of these values."
)

capacity_change_pct = st.sidebar.slider(
    "Capacity change — whole line (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=1,
    help=(
        "Simulates speeding up or slowing down every station's cycle time "
        "by this percentage. For example, +10% makes every station finish "
        "its work 10% faster (shorter cycle time); -15% makes every "
        "station 15% slower. Use this to model a line-wide speed change, "
        "such as a new motor drive setting or a line-wide slowdown."
    ),
)

downtime_minutes = st.sidebar.number_input(
    "Downtime — bottleneck station (minutes)",
    min_value=0,
    max_value=240,
    value=0,
    step=5,
    help=(
        "Adds this many minutes of unplanned stoppage time to whichever "
        "station is selected below as the 'Bottleneck station'. This "
        "downtime also partially cascades to every station after it on "
        "the line, since a late upstream station starves the ones "
        "downstream of work."
    ),
)

bottleneck_station = st.sidebar.selectbox(
    "Bottleneck station (receives the downtime above)",
    options=["None"] + STATIONS,
    index=0,
    help=(
        "Choose which station should receive the 'Downtime' value above. "
        "Note: this is where you INJECT downtime — the app separately "
        "AUTO-DETECTS and reports which station ends up being the true "
        "bottleneck (the slowest one) after the simulation runs, since "
        "that may end up being a different station than the one you pick "
        "here."
    ),
)
bottleneck_arg = None if bottleneck_station == "None" else bottleneck_station

layout_modified = st.sidebar.checkbox(
    "Apply layout modification (efficiency improvement)",
    value=False,
    help=(
        "Simulates a physical layout or material-flow improvement across "
        "the WHOLE line — for example, shortening conveyor distances or "
        "removing an awkward handoff between stations. This gives every "
        "station roughly an 8% cycle-time improvement and slightly lowers "
        "utilization pressure."
    ),
)

st.sidebar.divider()

# --------------------------------------------------------------------------
# Sidebar — Individual Workstation Adjustments (Work Styles)
# --------------------------------------------------------------------------
st.sidebar.header("🔧 Individual Workstation Adjustments")
st.sidebar.caption(
    "Pick a **Work Style** for each station instead of raw numbers. Each "
    "style is a preset bundle of trade-offs (speed vs. risk vs. energy). "
    "You can still fine-tune manually underneath if needed."
)

work_style_help = "Available work styles for this station:\n" + "\n".join(
    f"- **{name}**: {info['description']}" for name, info in WORK_STYLES.items()
)

station_adjustments = {}
station_modes = {}
for station in STATIONS:
    with st.sidebar.expander(f"🏭 {station}", expanded=False):
        mode = st.selectbox(
            "Work style",
            options=list(WORK_STYLES.keys()),
            index=0,
            key=f"mode_{station}",
            help=work_style_help,
        )
        st.caption(f"❓ {WORK_STYLES[mode]['description']}")
        station_modes[station] = mode

        with st.expander("Advanced: manual fine-tune (optional)", expanded=False):
            cap_delta = st.slider(
                "Extra capacity change on top of the work style (%)",
                min_value=-30,
                max_value=30,
                value=0,
                step=1,
                key=f"cap_{station}",
                help=(
                    f"Adjusts **only {station}**'s cycle time, ON TOP of "
                    "both the whole-line capacity setting AND the work "
                    "style selected above. Use this for a specific "
                    "machine upgrade or local tweak not covered by the "
                    "preset work styles."
                ),
            )
            extra_downtime = st.number_input(
                "Extra downtime on top of the work style (minutes)",
                min_value=0,
                max_value=180,
                value=0,
                step=5,
                key=f"down_{station}",
                help=(
                    f"Adds unplanned stoppage minutes to **only {station}**, "
                    "ON TOP of any downtime already built into the selected "
                    "work style. Use this to test an unscheduled breakdown "
                    "regardless of which work style or bottleneck is set."
                ),
            )

        style = WORK_STYLES[mode]
        station_adjustments[station] = {
            "capacity_delta_pct": style["capacity_delta_pct"] + cap_delta,
            "extra_downtime_minutes": style["extra_downtime_minutes"] + extra_downtime,
            "mode": mode,
            "risk_multiplier": style["risk_multiplier"],
            "energy_multiplier": style["energy_multiplier"],
        }

st.sidebar.divider()
run_button = st.sidebar.button(
    "▶️ Run Simulation",
    type="primary",
    use_container_width=True,
    help=(
        "Recomputes the what-if scenario using all the settings above and "
        "refreshes every chart, table, and AI insight on the page. "
        "(Note: this app also auto-updates as you move any slider.)"
    ),
)

st.sidebar.caption(
    "Adjust the parameters above — click **Run Simulation** or simply move "
    "any control — to compare the what-if scenario against the current "
    "baseline."
)

# --------------------------------------------------------------------------
# Run baseline and scenario simulations
# --------------------------------------------------------------------------
baseline_result = run_simulation(baseline_df)

scenario_result = run_simulation(
    baseline_df,
    capacity_change_pct=capacity_change_pct,
    downtime_minutes=downtime_minutes,
    bottleneck_station=bottleneck_arg,
    layout_modified=layout_modified,
    station_adjustments=station_adjustments,
)

# --------------------------------------------------------------------------
# Metrics Display
# --------------------------------------------------------------------------
st.subheader("📊 Line Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

throughput_delta = scenario_result["throughput_units_per_hour"] - baseline_result["throughput_units_per_hour"]
utilization_delta = scenario_result["avg_utilization"] - baseline_result["avg_utilization"]
delay_delta = scenario_result["total_delay_minutes"] - baseline_result["total_delay_minutes"]

col1.metric(
    "Throughput (units/hr)",
    f"{scenario_result['throughput_units_per_hour']}",
    delta=f"{throughput_delta:+.2f}",
    help="Finished units the whole line can produce per hour, paced by "
         "its slowest (bottleneck) station. Higher is better.",
)
col2.metric(
    "Avg. Utilization",
    f"{scenario_result['avg_utilization'] * 100:.1f}%",
    delta=f"{utilization_delta * 100:+.1f}%",
    help="Average busy-time fraction across all 6 stations. Very high "
         "values (>90%) mean little slack to absorb variability.",
)
col3.metric(
    "Total Delay (min)",
    f"{scenario_result['total_delay_minutes']}",
    delta=f"{delay_delta:+.2f}",
    delta_color="inverse",
    help="Sum of all downtime and cascading knock-on delay across every "
         "station in this scenario. Lower is better.",
)
col4.metric(
    "Bottleneck Station (auto-detected)",
    scenario_result["bottleneck_station"],
    help="The station with the highest resulting cycle time after this "
         "scenario is applied — i.e. the true pacing constraint of the "
         "line, whether or not you targeted it with downtime.",
)

st.divider()

# --------------------------------------------------------------------------
# Work Style Overview
# --------------------------------------------------------------------------
st.subheader("🎛️ Current Work Style per Station")
st.caption(
    "❓ This shows the work style you've selected for each station in the "
    "sidebar and what it changes relative to Standard: capacity, downtime, "
    "failure-risk multiplier (×), and relative energy index (×, 1.0 = "
    "Standard consumption)."
)
work_style_rows = []
for station in STATIONS:
    adj = station_adjustments.get(station, {
        "mode": "Standard",
        "risk_multiplier": 1.0,
        "energy_multiplier": 1.0,
        "capacity_delta_pct": 0,
        "extra_downtime_minutes": 0,
    })
    work_style_rows.append({
        "Station": station,
        "Work Style": adj.get("mode", "Standard"),
        "Capacity Δ (%)": adj.get("capacity_delta_pct", 0),
        "Extra Downtime (min)": adj.get("extra_downtime_minutes", 0),
        "Risk ×": adj.get("risk_multiplier", 1.0),
        "Energy ×": adj.get("energy_multiplier", 1.0),
    })
st.dataframe(pd.DataFrame(work_style_rows), use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# Per-Station Detail Table
# --------------------------------------------------------------------------
st.subheader("🏭 Per-Station Simulation Results")
st.caption(
    "❓ **CycleTime** = minutes per unit at that station · "
    "**Utilization** = fraction of time busy (0–1) · "
    "**Delay** = minutes lost to downtime and cascading effects."
)

tab1, tab2 = st.tabs(["Scenario Results", "Baseline Results"])

with tab1:
    st.dataframe(scenario_result["station_results"], use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(baseline_result["station_results"], use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# Scenario Comparison Chart
# --------------------------------------------------------------------------
st.subheader("📈 Scenario Comparison: Baseline vs. What-If")
st.caption(
    "❓ This chart compares each station's cycle time under your current "
    "settings (Scenario) against the unmodified historical data (Baseline)."
)

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    chart_df = pd.DataFrame({
        "Station": baseline_result["station_results"]["Station"],
        "Baseline Cycle Time": baseline_result["station_results"]["CycleTime"],
        "Scenario Cycle Time": scenario_result["station_results"]["CycleTime"],
    }).set_index("Station")
    st.bar_chart(chart_df)

with chart_col2:
    comparison_table = compare_scenarios(baseline_result, scenario_result)
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# AI Insights (placeholder models): predictions + risk
# --------------------------------------------------------------------------
st.subheader("🤖 AI Predictions (Placeholder Models)")

ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    st.markdown(
        "**Predicted Cycle Times** "
        ":grey[— an AI estimate of real-world cycle time, factoring in "
        "congestion effects that the simple simulation above doesn't "
        "capture.]"
    )
    predictor = CycleTimePredictor()
    predictions = predictor.predict_batch(
        scenario_result["station_results"][["Station", "CycleTime", "Utilization"]].to_dict("records")
    )
    pred_df = pd.DataFrame([p.__dict__ for p in predictions])
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.caption("❓ **confidence** = how certain the placeholder model is (0–1, higher is more certain).")

with ai_col2:
    st.markdown(
        "**Failure Risk Assessment** "
        ":grey[— an AI estimate of breakdown risk, based on historical "
        "failure rate, current utilization, and each station's chosen "
        "work style.]"
    )
    risk_model = FailurePatternModel()
    risk_input = scenario_result["station_results"][["Station", "Utilization"]].merge(
        baseline_df[["Station", "FailureRate"]], on="Station"
    )
    # Apply each station's work-style risk multiplier to its baseline
    # failure rate before assessing risk, so a High-Speed station is
    # correctly flagged as riskier than the same station on Standard.
    risk_input["FailureRate"] = risk_input.apply(
        lambda r: r["FailureRate"] * station_adjustments.get(
            r["Station"], {}
        ).get("risk_multiplier", 1.0),
        axis=1,
    )
    risks = risk_model.assess_batch(risk_input.to_dict("records"))
    risk_df = pd.DataFrame(risks)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.caption("❓ **risk_level**: Low / Medium / High likelihood of unplanned failure under this scenario (already adjusted for each station's work style).")

st.divider()

# --------------------------------------------------------------------------
# AI Analysis & Suggestions
# --------------------------------------------------------------------------
st.subheader("🧠 AI Analysis & Suggestions")
st.caption(
    "❓ The Analysis section explains, in plain English, what the numbers "
    "above mean for your line. The Suggestions section turns that analysis "
    "into concrete actions, color-coded by urgency "
    "(🛑 critical · ⚠️ warning · ℹ️ info · ✅ all clear)."
)

advisor = AIAdvisor()

analysis_tab, suggestions_tab = st.tabs(["📋 AI Analysis", "💡 AI Suggestions"])

with analysis_tab:
    analysis_points = advisor.generate_analysis(scenario_result, baseline_result)
    for point in analysis_points:
        st.markdown(f"- {point}")

with suggestions_tab:
    suggestions = advisor.generate_suggestions(scenario_result, risk_df)
    if not suggestions:
        st.info("No specific suggestions for this scenario.")
    for s in suggestions:
        renderer = SEVERITY_RENDERERS.get(s["severity"], st.info)
        renderer(s["message"])

st.caption(
    "Note: AI predictions, analysis, and suggestions above are produced by "
    "placeholder heuristic logic (see `models/placeholder_model.py`), "
    "structured so they can be swapped for trained ML models or an LLM "
    "reasoning layer without changing the rest of the app."
)

st.divider()

# --------------------------------------------------------------------------
# AI Combination Optimizer
# --------------------------------------------------------------------------
st.subheader("🧬 AI Combination Optimizer — Which Work-Style Mix Works Best?")
st.markdown(
    "Instead of guessing which station should run in which work style, let "
    "the AI test **every realistic combination** across all 6 stations and "
    "surface the ones that best match what you care about most."
)

opt_col1, opt_col2 = st.columns([2, 1])

with opt_col1:
    objective = st.selectbox(
        "Optimization objective",
        options=OBJECTIVE_OPTIONS,
        index=0,
        key="combo_objective",
        help=(
            "Tells the AI what 'best' means for this search:\n"
            + "\n".join(f"- **{name}**: {desc}" for name, desc in OBJECTIVE_HELP.items())
        ),
    )
    st.caption(f"❓ {OBJECTIVE_HELP[objective]}")

with opt_col2:
    st.caption(
        "❓ The search tries every combination of **Standard / High-Speed / "
        "Energy-Saving / Quality-Focus** across all 6 stations (Maintenance "
        "Mode is excluded — see its description). It uses your current "
        "whole-line capacity, downtime, bottleneck, and layout settings as "
        "a fixed backdrop."
    )
    find_combo_clicked = st.button(
        "🔍 Find Best Combination",
        type="primary",
        use_container_width=True,
        help=(
            "Runs the search now. This can take a moment the first time; "
            "results are kept until you search again so you can freely "
            "explore the rest of the page afterward."
        ),
    )

if find_combo_clicked:
    with st.spinner("Testing work-style combinations across all 6 stations..."):
        st.session_state["combo_results"] = find_best_combinations(
            baseline_df,
            capacity_change_pct=capacity_change_pct,
            downtime_minutes=downtime_minutes,
            bottleneck_station=bottleneck_arg,
            layout_modified=layout_modified,
            objective=objective,
            top_n=5,
        )
        st.session_state["combo_objective_used"] = objective

if "combo_results" in st.session_state and st.session_state["combo_results"]:
    combo_results = st.session_state["combo_results"]
    objective_used = st.session_state.get("combo_objective_used", objective)
    best = combo_results[0]

    st.markdown("#### 🧠 AI Suggested Combination")
    st.caption(
        "❓ This is the #1-ranked combination found by the search above, "
        "re-checked with the full simulation engine for accuracy."
    )
    st.success(advisor.generate_combination_insight(best, scenario_result, objective_used))

    explanation_rows = advisor.explain_combination_choices(best)
    st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)

    apply_clicked = st.button(
        "✅ Apply This Combination to the Sidebar",
        help=(
            "Sets every station's 'Work style' dropdown in the sidebar to "
            "match this recommended combination, clears any manual "
            "fine-tune overrides, and refreshes the whole page with the "
            "new settings."
        ),
    )
    if apply_clicked:
        for station, mode_name in best["combo"].items():
            st.session_state[f"mode_{station}"] = mode_name
            st.session_state[f"cap_{station}"] = 0
            st.session_state[f"down_{station}"] = 0
        st.rerun()

    with st.expander(f"📊 Compare Top {len(combo_results)} Combinations Found", expanded=False):
        st.caption(
            "❓ Each row is one full combination of work styles across all "
            "6 stations, ranked best-first for your chosen objective."
        )
        summary_rows = []
        for rank, item in enumerate(combo_results, start=1):
            row = {"Rank": rank}
            row.update(item["combo"])
            row["Throughput (units/hr)"] = item["simulation"]["throughput_units_per_hour"]
            row["Avg. Risk"] = item["avg_risk"]
            row["Energy Index"] = item["avg_energy_index"]
            row["Total Delay (min)"] = item["simulation"]["total_delay_minutes"]
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
else:
    st.info(
        "No combination search has been run yet. Choose an objective above "
        "and click **Find Best Combination** to see AI-suggested station "
        "work styles."
    )

st.divider()

# --------------------------------------------------------------------------
# Info / Next Steps
# --------------------------------------------------------------------------
with st.expander("ℹ️ About this Digital Twin & Next Steps", expanded=False):
    st.markdown(
        """
        ### What this app currently does
        - Models the 6-station glass production line using historical/sample data.
        - Runs configurable what-if scenarios (whole-line capacity, downtime,
          bottleneck, layout) **and** individual per-station adjustments,
          using transparent heuristic simulation logic.
        - Surfaces placeholder AI predictions, plain-English analysis, and
          prioritized, color-coded suggestions for planners.
        - Explains every input, metric, and table with a built-in `?`
          tooltip or caption so the app is usable without prior training.

        ### Planned Next Steps
        1. **Connect live IoT data**: Stream real-time sensor data (temperature,
           pressure, vibration, throughput counters) from PLCs/SCADA systems via
           MQTT/OPC-UA into this twin for live state updates.
        2. **Train real ML models**: Replace the placeholder cycle time and
           failure-pattern models with models trained on historical production
           and maintenance data (e.g. gradient boosting, LSTM for time series).
        3. **LLM-powered analysis**: Swap the rule-based `AIAdvisor` text
           generation for a real LLM call grounded in simulation data, for
           richer, more nuanced recommendations and natural-language Q&A.
        4. **Discrete-event simulation engine**: Upgrade the simulation logic
           to a full discrete-event simulation (e.g. using SimPy) to capture
           queueing, buffers, and stochastic variation more realistically.
        5. **Cloud analytics & deployment**: Deploy to a cloud platform
           (Azure/AWS/GCP) with a managed database, scheduled retraining
           pipelines, and role-based access for planners and operators.
        6. **Alerting & optimization**: Add automated alerts for predicted
           failures/bottlenecks and an optimization layer to recommend the
           best scenario configuration automatically.
        """
    )

st.caption("AI-Enabled Digital Twin — Glass Production Line | Demo build")
