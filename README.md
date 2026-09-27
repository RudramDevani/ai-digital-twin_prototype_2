# AI-Enabled Digital Twin for Glass Production Line

A Streamlit-based web application that simulates a **virtual replica (digital
twin)** of a 6-station glass production line. It learns from historical and
(eventually) live machine data, and allows production planners to run
**AI-supported what-if scenarios** before applying changes on the real
factory floor.

## Overview

The digital twin models a serial production chain of six stations:

1. **Raw Material Preparation**
2. **Melting Furnace**
3. **Forming & Molding**
4. **Annealing**
5. **Inspection**
6. **Packaging**

Planners can adjust scenario parameters — capacity, downtime, bottleneck
station, and layout changes — and instantly see the projected impact on
throughput, utilization, and delays, compared against the current baseline.

## Features

- 📊 **Whole-line scenario controls** — capacity change (%), downtime
  (minutes), bottleneck station selection, and layout modification toggle.
- 🔧 **Individual workstation "Work Styles"** — each station can be set to
  **Standard**, **High-Speed**, **Energy-Saving**, **Quality-Focus**, or
  **Maintenance Mode**, each a preset bundle of capacity/downtime/failure-risk/
  energy trade-offs, with an optional manual fine-tune slider underneath for
  even finer control.
- 🧬 **AI Combination Optimizer** — brute-force-tests every realistic
  work-style combination across all 6 stations (thousands of combinations)
  against a chosen objective (Balanced / Maximize Throughput / Minimize
  Failure Risk / Minimize Energy Use), then re-verifies the top candidates
  with the full simulation engine for accuracy.
- 🧠 **AI Suggestion Box for combinations** — presents the #1-ranked
  combination in plain English (predicted throughput, risk, energy vs. your
  current settings), a full per-station breakdown table, and a **one-click
  "Apply"** button that updates every sidebar dropdown to match.
- 🏭 **Per-station simulation results** — cycle time, utilization, and delay
  for every station in both the baseline and what-if scenario, including
  realistic cascading delay from an upstream station to those downstream.
- 📈 **Scenario comparison chart** — visual and tabular comparison of
  baseline vs. what-if throughput, utilization, and delay.
- 🤖 **AI predictions** — placeholder cycle time prediction and failure
  risk scoring (now work-style-aware), structured so they can be swapped
  for trained ML models.
- 🧠 **AI Analysis & Suggestions** — a plain-English breakdown of what's
  happening on the line, plus prioritized, color-coded, actionable
  recommendations (🛑 critical · ⚠️ warning · ℹ️ info · ✅ all clear).
- ❓ **Built-in help everywhere** — every slider, input, dropdown, checkbox,
  button, metric, and table has a `?` tooltip or caption explaining exactly
  what it does and how it affects the simulation, plus a glossary of key
  terms.
- 🧩 **Modular structure** — simulation logic, work-style presets, and AI
  models are separated from the UI, making it straightforward to extend or
  replace any layer.

## Project Structure

```
ai_digital_twin/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   └── sample_data.csv       # Example station dataset
├── models/
│   └── placeholder_model.py  # Placeholder ML models + AIAdvisor (analysis/suggestions)
└── utils/
    ├── simulation.py         # Simulation engine + combination optimizer
    └── work_styles.py        # Work-style presets (Standard/High-Speed/etc.)
```

## Installation

1. **Clone or download** this project folder.
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Locally

From the `ai_digital_twin/` project root, run:

```bash
streamlit run streamlit_app.py
```

Streamlit will start a local development server and open the app in your
default browser (typically at `http://localhost:8501`).

## Using the App

1. Use the **sidebar → Whole-Line Scenario Controls** to set a baseline
   what-if scenario:
   - Set a **capacity change (%)** to model faster/slower line speed.
   - Set **downtime (minutes)** on a chosen **bottleneck station**.
   - Toggle **layout modification** to model a material-flow improvement.
2. Open **sidebar → Individual Workstation Adjustments** and expand any of
   the 6 stations to pick a **Work Style** (Standard / High-Speed /
   Energy-Saving / Quality-Focus / Maintenance Mode), or dig into "Advanced:
   manual fine-tune" for raw percentage/minute control on top of it.
3. Scroll to **🧬 AI Combination Optimizer**, pick an objective (e.g.
   "Maximize Throughput" or "Minimize Failure Risk"), and click
   **Find Best Combination** to have the AI test every realistic work-style
   mix across all 6 stations and recommend the best one.
4. Review the **AI Suggested Combination** box and per-station breakdown,
   then click **Apply This Combination to the Sidebar** to instantly set
   every station to the recommended work style.
5. Review the **metrics**, **work style overview**, **per-station tables**,
   **comparison chart**, **AI predictions**, and **AI Analysis & Suggestions**
   sections to understand what's driving the results and what to do about it.
6. Hover the **`?` icon** next to any control, metric, or table for a
   plain-English explanation of what it means and how it affects the
   simulation.

## Future Improvements

- **IoT Integration**: Stream live sensor data (temperature, pressure,
  vibration, cycle counters) from PLCs/SCADA systems via MQTT or OPC-UA to
  drive the twin with real-time state instead of static sample data.
- **Machine Learning Models**: Replace the placeholder heuristics in
  `models/placeholder_model.py` with trained models (e.g. gradient boosting
  for cycle time prediction, classifiers or anomaly detectors for failure
  pattern recognition) built on historical production and maintenance logs.
- **Discrete-Event Simulation**: Upgrade `utils/simulation.py` to a full
  discrete-event simulation (e.g. using SimPy) to model queues, buffers, and
  stochastic variability between stations more realistically.
- **Cloud Analytics & Deployment**: Deploy to a cloud platform (Azure, AWS,
  or GCP) with a managed database/data lake, scheduled model retraining
  pipelines, and role-based access control for planners and operators.
- **Automated Alerts & Optimization**: Add proactive alerting for predicted
  failures or emerging bottlenecks, and an optimization layer that
  recommends the best scenario configuration automatically.

## License

This is a demo/reference project intended as a starting point for building
out a production-grade digital twin application.
