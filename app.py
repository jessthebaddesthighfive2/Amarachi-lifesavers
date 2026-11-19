import json
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# 1. CONSTANTS & NUR-3 HANDBOOK DEFAULTS
# ==============================================================================
TEAM_ID = "LIFE_SAVERS"
COURSE = "AP Calculus"
PATHWAY_ID = "NUR-3"
BUSINESS_NAME = "Nurse-led health education simulation"

def to_money(val) -> Decimal:
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def default_state() -> dict:
    return {
        "team_id": TEAM_ID,
        "course": COURSE,
        "pathway_id": PATHWAY_ID,
        "business_name": BUSINESS_NAME,
        "selected_level": "Advanced",
        "snapshot_id": "SNAP-001",
        "state_version": 1,
        "simulation_date": "2027-01-31",
        "base_period": "monthly",
        "working_days": 20,
        "currency": "USD",
        "inputs": {
            "base_price": 25.00,
            "D0": 80,
            "requested": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
            "ad_cost": 0.0,
            "refund_amount": 0.0,
            "loan_principal": 0.0,
            "interest": 0.0,
            "additional_capex": 0.0,
            "new_owner_cash": 0.0,
            "loan_cash": 0.0,
        },
        "startup": {
            "owner_contribution": 6000.00,
            "equipment_assets": 1500.00,
            "pre_opening_expense": 500.00,
            "opening_inventory": 0.00,
            "opening_cash": 4000.00,
            "retained_earnings": -500.00,
        },
        "variable_cost_lines": {
            "educator_delivery": 2.00,
            "materials": 2.00,
            "cleaning_consumables": 1.00,
        },
        "fixed_cost_lines": {
            "venue": 250.00,
            "scheduling_software": 100.00,
            "accessibility_support": 200.00,
            "admin_insurance_assumption": 150.00,
        },
        "offerings": [
            {"id": "OFF-1", "name": "General wellness education workshop (Primary)", "active": True},
            {"id": "OFF-2", "name": "Caregiver information session", "active": False},
            {"id": "OFF-3", "name": "First-aid awareness demonstration", "active": False},
        ],
        "current_cash": 4000.00,
        "ledger": [],
        "processed_run_ids": [],
        "decisions": [],
    }

# ==============================================================================
# 2. DETERMINISTIC NUR SIMULATION & AP CALCULUS ENGINE
# ==============================================================================
def compute_nur_period(
    inputs: dict,
    var_lines: dict,
    fix_lines: dict,
    opening_cash: Decimal,
    capacity_override: int | None = None,
) -> dict:
    price = to_money(inputs["base_price"])
    requested = int(inputs["requested"])
    staff_hours = float(inputs["staff_hours"])
    hours_per_unit = float(inputs["hours_per_unit"])
    venue_seats = int(inputs["venue_seats"])
    no_show_rate = float(inputs["no_show_rate"])

    staff_cap = math.floor(staff_hours / hours_per_unit) if hours_per_unit > 0 else 0
    derived_capacity = min(staff_cap, venue_seats)
    effective_capacity = int(capacity_override) if capacity_override is not None else derived_capacity

    booked = min(requested, effective_capacity)
    attended = math.floor(booked * (1.0 - no_show_rate))
    sold = attended
    lost_demand = max(0, requested - booked)

    var_per_unit = sum(to_money(v) for v in var_lines.values())
    fixed_expense = sum(to_money(v) for v in fix_lines.values())
    ad_cost = to_money(inputs.get("ad_cost", 0.0))
    refunds = to_money(inputs.get("refund_amount", 0.0))
    interest = to_money(inputs.get("interest", 0.0))
    principal = to_money(inputs.get("loan_principal", 0.0))
    capex = to_money(inputs.get("additional_capex", 0.0))
    new_owner = to_money(inputs.get("new_owner_cash", 0.0))
    loan_cash = to_money(inputs.get("loan_cash", 0.0))

    gross_revenue = price * Decimal(sold)
    net_revenue = gross_revenue - refunds
    variable_expense = var_per_unit * Decimal(sold)
    total_operating_expense = variable_expense + fixed_expense + ad_cost
    operating_profit = net_revenue - total_operating_expense
    net_profit = operating_profit - interest

    ending_cash = (
        opening_cash
        + net_revenue
        - variable_expense
        - fixed_expense
        - ad_cost
        - interest
        - principal
        - capex
        + new_owner
        + loan_cash
    )

    return {
        "staff_capacity": staff_cap,
        "venue_capacity": venue_seats,
        "effective_capacity": effective_capacity,
        "requested": requested,
        "booked": booked,
        "attended": attended,
        "sold": sold,
        "lost_demand": lost_demand,
        "var_per_unit": var_per_unit,
        "revenue": net_revenue,
        "variable_expense": variable_expense,
        "fixed_expense": fixed_expense,
        "ad_cost": ad_cost,
        "total_expense": total_operating_expense,
        "operating_profit": operating_profit,
        "net_profit": net_profit,
        "opening_cash": opening_cash,
        "ending_cash": ending_cash,
    }

def solve_ap_calculus_model(D0: float, p0: float, v: float, F: float, capacity: float) -> dict:
    # Linear demand q(p) = a - b*p on 0 <= p <= 2*p0
    a = 2.0 * D0
    b = D0 / p0 if p0 > 0 else 0.0
    p_max = 2.0 * p0

    # Unconstrained profit P(p) = (p - v)*(a - b*p) - F -> P'(p) = a + b*v - 2*b*p
    p_crit = (a + b * v) / (2.0 * b) if b > 0 else 0.0
    p_crit_clamped = min(max(0.0, p_crit), p_max)
    q_at_crit = max(0.0, a - b * p_crit_clamped)
    unconstrained_profit = (p_crit_clamped - v) * q_at_crit - F

    # Capacity corner where q(p) == capacity
    p_corner = (a - capacity) / b if b > 0 else 0.0
    p_corner_in_domain = 0.0 <= p_corner <= p_max

    # Candidate prices for constrained continuous optimum: endpoints, corner, feasible critical point
    candidates = [0.0, p_max]
    if p_corner_in_domain:
        candidates.append(p_corner)
    if 0.0 <= p_crit <= p_max and (a - b * p_crit) <= capacity:
        candidates.append(p_crit)

    def constrained_profit(p_val: float) -> float:
        q_val = max(0.0, a - b * p_val)
        s_val = min(capacity, q_val)
        return (p_val - v) * s_val - F

    best_p = max(candidates, key=constrained_profit)
    best_profit = constrained_profit(best_p)

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_crit": p_crit,
        "unconstrained_profit": unconstrained_profit,
        "p_corner": p_corner,
        "p_corner_in_domain": p_corner_in_domain,
        "constrained_opt_p": best_p,
        "constrained_opt_profit": best_profit,
        "unconstrained_feasible": (a - b * p_crit) <= capacity,
    }

# ==============================================================================
# 3. SNAPSHOT VALIDATION & PERSISTENCE
# ==============================================================================
def validate_and_load_snapshot(raw_text: str, current_version: int) -> tuple[bool, str, dict | None]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON syntax: {exc}", None

    required_keys = ["team_id", "course", "pathway_id", "state_version", "inputs", "current_cash", "ledger"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, f"Missing required snapshot fields: {', '.join(missing)}", None

    if data["team_id"] != TEAM_ID:
        return False, f"Rejected: Wrong team_id '{data['team_id']}' (expected '{TEAM_ID}').", None

    if int(data["state_version"]) < current_version:
        return (
            False,
            f"Rejected: Stale snapshot v{data['state_version']} cannot overwrite current state v{current_version}.",
            None,
        )

    if float(data["current_cash"]) < 0 or float(data["inputs"].get("base_price", 0)) <= 0:
        return False, "Rejected: Negative cash or non-positive base price detected.", None

    return True, f"Loaded snapshot {data.get('snapshot_id', 'N/A')} (v{data['state_version']}).", data

# ==============================================================================
# 4. STREAMLIT UI & STAGE 1 MODULES
# ==============================================================================
st.set_page_config(page_title="Life Savers | Career Venture Lab (NUR-3)", layout="wide")

if "sim" not in st.session_state:
    st.session_state.sim = default_state()

sim = st.session_state.sim

# Top Header & Guardrail Banner
st.title("Life Savers - Nurse-Led Health Education Simulation")
st.caption(
    f"Team ID: **{sim['team_id']}** | Pathway: **{sim['pathway_id']}** | Course: **{sim['course']}** | "
    f"State Version: **v{sim['state_version']}** | Snapshot: **{sim['snapshot_id']}**"
)
st.info(
    "Educational Simulation Notice: Models fictional general wellness workshops only. "
    "No medical diagnosis, treatment, or real patient data is used."
)

# Sidebar Controls: Challenge Level, Save/Load JSON, Reset
with st.sidebar:
    st.header("System Controls")
    level = st.selectbox(
        "Challenge Level (Teacher Approved)",
        ["Foundation", "Applied", "Advanced"],
        index=["Foundation", "Applied", "Advanced"].index(sim["selected_level"]),
    )
    if level != sim["selected_level"]:
        sim["selected_level"] = level
        st.rerun()

    st.divider()
    st.subheader("Save / Restore Snapshot (JSON)")
    snapshot_json = json.dumps(sim, indent=2)
    st.download_button(
        label="Export Snapshot JSON",
        data=snapshot_json,
        file_name=f"{TEAM_ID}_{PATHWAY_ID}_v{sim['state_version']}.json",
        mime="application/json",
    )

    uploaded_file = st.file_uploader("Import Snapshot JSON", type=["json"])
    if uploaded_file is not None:
        if st.button("Validate & Import JSON"):
            ok, msg, loaded = validate_and_load_snapshot(
                uploaded_file.getvalue().decode("utf-8"), sim["state_version"]
            )
            if ok and loaded:
                st.session_state.sim = loaded
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.divider()
    if st.button("Reset to Baseline Defaults"):
        st.session_state.sim = default_state()
        st.rerun()

# Navigation Tabs (Stage 1 Active + Stage 2/3 Placeholders)
tab_overview, tab_inputs, tab_math, tab_tests, tab_roadmap = st.tabs(
    [
        "1. Overview & Simulation Step",
        "2. Inputs, Capacity & Costs",
        "3. AP Calculus Panel",
        "4. Stage 1 Acceptance Tests",
        "5. Stage 2-4 Roadmap (Not Yet Implemented)",
    ]
)

# Current live preview metrics
live_res = compute_nur_period(
    sim["inputs"], sim["variable_cost_lines"], sim["fixed_cost_lines"], to_money(sim["current_cash"])
)

# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & SIMULATION STEP
# ------------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spendable Cash", f"${sim['current_cash']:,.2f}")
    col2.metric("Effective Capacity", f"{live_res['effective_capacity']} seats/mo")
    col3.metric("Projected Attended/Sold", f"{live_res['sold']} seats", f"-{live_res['lost_demand']} unserved")
    col4.metric("Projected Monthly Profit", f"${live_res['operating_profit']:,.2f}")

    st.subheader("Persistent AP Calculus Summary")
    calc_summary = solve_ap_calculus_model(
        D0=sim["inputs"]["D0"],
        p0=sim["inputs"]["base_price"],
        v=float(live_res["var_per_unit"]),
        F=float(live_res["fixed_expense"]),
        capacity=live_res["effective_capacity"],
    )
    st.markdown(
        f"- **Active Skill & Level:** Constrained Profit Optimization & Marginal Analysis (**{sim['selected_level']}**)\n"
        f"- **Demand Model:** $q(p) = {calc_summary['a']:.1f} - {calc_summary['b']:.2f}p$ on $0 \\le p \\le {calc_summary['p_max']:.2f}$ dollars\n"
        f"- **Constrained Continuous Optimum:** Price = **${calc_summary['constrained_opt_p']:.2f}**/seat "
        f"-> Max Continuous Profit = **${calc_summary['constrained_opt_profit']:.2f}**/mo "
        f"(Unconstrained critical point ${calc_summary['p_crit']:.2f} is "
        f"{'feasible' if calc_summary['unconstrained_feasible'] else 'INFEASIBLE due to capacity cap'})"
    )

    st.divider()
    st.subheader("Record & Advance One Simulation Month (Ledger Entry)")
    c_run, c_date, c_seed = st.columns(3)
    run_id_input = c_run.text_input("Unique Run ID", value=f"RUN-{sim['state_version']:03d}")
    sim_date_input = c_date.text_input("Simulation Date (YYYY-MM-DD)", value=sim["simulation_date"])
    seed_input = c_seed.number_input("Random Seed", value=2027, step=1)

    if st.button("Approve & Commit Month to Ledger", type="primary"):
        if run_id_input in sim["processed_run_ids"]:
            st.error(f"Duplicate Run Blocked: '{run_id_input}' has already been recorded.")
        else:
            sim["current_cash"] = float(live_res["ending_cash"])
            sim["state_version"] += 1
            sim["simulation_date"] = sim_date_input
            sim["snapshot_id"] = f"SNAP-{sim['state_version']:03d}"
            sim["processed_run_ids"].append(run_id_input)
            sim["ledger"].append(
                {
                    "run_id": run_id_input,
                    "state_version": sim["state_version"],
                    "date": sim_date_input,
                    "requested": live_res["requested"],
                    "capacity": live_res["effective_capacity"],
                    "booked": live_res["booked"],
                    "sold": live_res["sold"],
                    "revenue": float(live_res["revenue"]),
                    "total_expense": float(live_res["total_expense"]),
                    "profit": float(live_res["operating_profit"]),
                    "closing_cash": float(live_res["ending_cash"]),
                    "seed": int(seed_input),
                }
            )
            st.success(f"Committed {run_id_input}! Advanced to state v{sim['state_version']}.")
            st.rerun()

    st.subheader("Dated Event Ledger (Actual Simulated Months)")
    if sim["ledger"]:
        ledger_df = pd.DataFrame(sim["ledger"])
        st.dataframe(ledger_df, use_container_width=True)
        st.download_button(
            "Export Ledger CSV",
            data=ledger_df.to_csv(index=False),
            file_name=f"{TEAM_ID}_ledger_v{sim['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No months committed yet. Adjust inputs in Tab 2 and click 'Approve & Commit Month to Ledger'.")

# ------------------------------------------------------------------------------
# TAB 2: INPUTS, CAPACITY & COSTS
# ------------------------------------------------------------------------------
with tab_inputs:
    st.subheader("Workshop Demand, Pricing & Capacity Inputs")
    col_a, col_b = st.columns(2)
    with col_a:
        sim["inputs"]["base_price"] = st.number_input(
            "Seat Price ($/seat)", min_value=1.0, value=float(sim["inputs"]["base_price"]), step=1.0
        )
        sim["inputs"]["D0"] = st.number_input(
            "Baseline Demand D0 at Base Price (seats/mo)", min_value=0, value=int(sim["inputs"]["D0"]), step=1
        )
        sim["inputs"]["requested"] = st.slider(
            "Current Month Requested Seats", min_value=0, max_value=200, value=int(sim["inputs"]["requested"])
        )
        sim["inputs"]["no_show_rate"] = st.slider(
            "No-Show Rate (0.00 to 1.00)",
            min_value=0.0,
            max_value=1.0,
            value=float(sim["inputs"]["no_show_rate"]),
            step=0.05,
        )

    with col_b:
        sim["inputs"]["staff_hours"] = st.number_input(
            "Educator Staff Hours / Month", min_value=0.0, value=float(sim["inputs"]["staff_hours"]), step=1.0
        )
        sim["inputs"]["hours_per_unit"] = st.number_input(
            "Educator Hours per Seat", min_value=0.1, value=float(sim["inputs"]["hours_per_unit"]), step=0.1
        )
        sim["inputs"]["venue_seats"] = st.number_input(
            "Venue Seat Limit / Month", min_value=0, value=int(sim["inputs"]["venue_seats"]), step=5
        )
        sim["inputs"]["ad_cost"] = st.number_input(
            "Approved Monthly Marketing Expense ($)", min_value=0.0, value=float(sim["inputs"]["ad_cost"]), step=25.0
        )

    st.divider()
    c_var, c_fix = st.columns(2)
    with c_var:
        st.subheader("Variable Costs ($ per Attended Seat)")
        for k, val in sim["variable_cost_lines"].items():
            sim["variable_cost_lines"][k] = st.number_input(
                f"{k} ($/unit)", min_value=0.0, value=float(val), step=0.5, key=f"var_{k}"
            )
        st.write(f"**Total Variable Cost:** ${live_res['var_per_unit']:.2f} / seat")

    with c_fix:
        st.subheader("Monthly Fixed Costs ($ / Month)")
        for k, val in sim["fixed_cost_lines"].items():
            sim["fixed_cost_lines"][k] = st.number_input(
                f"{k} ($/mo)", min_value=0.0, value=float(val), step=10.0, key=f"fix_{k}"
            )
        st.write(f"**Total Fixed Cost:** ${live_res['fixed_expense']:.2f} / month")

# ------------------------------------------------------------------------------
# TAB 3: AP CALCULUS PANEL
# ------------------------------------------------------------------------------
with tab_math:
    st.subheader(f"AP Calculus Analysis - Level: {sim['selected_level']}")
    D0 = float(sim["inputs"]["D0"])
    p0 = float(sim["inputs"]["base_price"])
    v = float(live_res["var_per_unit"])
    F = float(live_res["fixed_expense"]) + float(sim["inputs"]["ad_cost"])
    cap = float(live_res["effective_capacity"])
    res_calc = solve_ap_calculus_model(D0, p0, v, F, cap)

    st.markdown(
        f"**Business Question:** What seat price $p$ maximizes monthly operating profit $P(p)$ given demand "
        f"$q(p) = {res_calc['a']:.1f} - {res_calc['b']:.2f}p$ and capacity cap $s \\le {cap:.0f}$ seats?\n\n"
        f"**Variables & Units:** $p$ = seat price ($/seat), $q(p)$ = demanded seats (seats/mo), "
        f"$s(p) = \\min({cap:.0f}, \\max(0, q(p)))$ = completed seats (seats/mo), $P(p)$ = monthly profit ($/mo), "
        f"$P'(p)$ = marginal profit with respect to price (($/mo) per ($/seat))."
    )

    # Interactive price slider for Transfer Check
    test_p = st.slider(
        "Transfer Check: Predict profit & derivative before moving price slider ($/seat)",
        min_value=0.0,
        max_value=float(res_calc["p_max"]),
        value=float(p0),
        step=0.5,
    )
    q_at_p = max(0.0, res_calc["a"] - res_calc["b"] * test_p)
    s_cont = min(cap, q_at_p)
    s_int = math.floor(s_cont)
    p_cont = (test_p - v) * s_cont - F
    p_int = (test_p - v) * s_int - F

    if abs(test_p - res_calc["p_corner"]) < 1e-6 and res_calc["p_corner_in_domain"]:
        deriv_str = "Undefined (capacity corner)"
    elif q_at_p > cap:
        deriv_str = f"{cap:.2f} ($/mo)/($/seat) [Capacity-constrained branch]"
    else:
        d_val = res_calc["a"] + res_calc["b"] * v - 2.0 * res_calc["b"] * test_p
        deriv_str = f"{d_val:.2f} ($/mo)/($/seat) [Demand-constrained branch]"

    m1, m2, m3 = st.columns(3)
    m1.metric("Continuous Profit P(p)", f"${p_cont:,.2f}")
    m2.metric("Rounded Integer-Seat Profit", f"${p_int:,.2f}", f"{s_int} whole seats")
    m3.metric("Derivative P'(p)", deriv_str)

    # Linked Profit & Derivative Plot on Aligned Axes
    prices = np.linspace(0, res_calc["p_max"], 400)
    q_vals = np.maximum(0.0, res_calc["a"] - res_calc["b"] * prices)
    s_vals = np.minimum(cap, q_vals)
    prof_vals = (prices - v) * s_vals - F
    unconst_prof_vals = (prices - v) * q_vals - F

    deriv_vals = np.where(
        q_vals > cap,
        cap,
        res_calc["a"] + res_calc["b"] * v - 2.0 * res_calc["b"] * prices,
    )
    # Break derivative line at capacity corner
    if res_calc["p_corner_in_domain"]:
        corner_idx = np.argmin(np.abs(prices - res_calc["p_corner"]))
        deriv_vals[corner_idx] = np.nan

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Monthly Profit P(p) vs. Price [Snapshot {sim['snapshot_id']}, v{sim['state_version']}]",
            "Marginal Profit P'(p) (Derivative with Capacity Corner)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=prices, y=prof_vals, name="Constrained Profit P(p)", line=dict(color="#1f77b4", width=3)),
        row=1,
        col=1,
    )
    if sim["selected_level"] in ["Applied", "Advanced"]:
        fig.add_trace(
            go.Scatter(
                x=prices,
                y=unconst_prof_vals,
                name="Unconstrained Profit (Ignores Cap)",
                line=dict(color="#7f7f7f", dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="black", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=prices, y=deriv_vals, name="P'(p)", line=dict(color="#d62728", width=2)),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1)

    if res_calc["p_corner_in_domain"]:
        fig.add_vline(
            x=res_calc["p_corner"],
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text=f"Capacity Corner p=${res_calc['p_corner']:.2f}",
        )

    fig.update_xaxes(title_text="Seat Price p ($/seat)", row=2, col=1)
    fig.update_yaxes(title_text="Profit ($/mo)", row=1, col=1)
    fig.update_yaxes(title_text="P'(p)", row=2, col=1)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    # Linked Input-Output Table
    sample_prices = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    rows = []
    for sp in sample_prices:
        if sp <= res_calc["p_max"]:
            sq = max(0.0, res_calc["a"] - res_calc["b"] * sp)
            ss = min(cap, sq)
            rows.append(
                {
                    "Price ($)": sp,
                    "Demand q(p)": round(sq, 2),
                    "Completed s(p) [Continuous]": round(ss, 2),
                    "Completed [Integer Floor]": math.floor(ss),
                    "Continuous Profit ($)": round((sp - v) * ss - F, 2),
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Advanced Extension: Campaign Rate Accumulation & Scenario Sensitivity
    if sim["selected_level"] == "Advanced":
        st.divider()
        st.subheader("Advanced Module: Campaign Rate Integral & Demand Sensitivity")
        st.markdown(
            "**Isolated Campaign Rate Exercise (Handbook AP Calculus Adapter):** "
            "Given revenue rate $r(t) = 100 + 20t$ ($/day) and cost rate $c(t) = 60 + 5t$ ($/day) over $0 \\le t \\le 10$ days:\n"
            "- Accumulated Revenue $= \\int_0^{10} (100 + 20t)\\,dt = [100t + 10t^2]_0^{10} = \\$2,000.00$\n"
            "- Accumulated Cost $= \\int_0^{10} (60 + 5t)\\,dt = [60t + 2.5t^2]_0^{10} = \\$850.00$\n"
            "- Net Accumulated Campaign Profit $= \\$2,000.00 - \\$850.00 = \\$1,150.00$ "
            "*(isolated campaign exercise; does not overwrite monthly ledger)*."
        )

        st.markdown("**Demand Uncertainty Scenarios (Baseline 1.00 / Upside 1.10 / Downside 0.90):**")
        scen_rows = []
        for label, mult in [("Downside (0.90x)", 0.90), ("Baseline (1.00x)", 1.00), ("Upside (1.10x)", 1.10)]:
            scen_inputs = dict(sim["inputs"], requested=math.floor(sim["inputs"]["D0"] * mult))
            scen_out = compute_nur_period(
                scen_inputs, sim["variable_cost_lines"], sim["fixed_cost_lines"], to_money(sim["current_cash"])
            )
            scen_rows.append(
                {
                    "Scenario": label,
                    "Requested Seats": scen_out["requested"],
                    "Sold Seats": scen_out["sold"],
                    "Lost Demand": scen_out["lost_demand"],
                    "Operating Profit ($)": float(scen_out["operating_profit"]),
                }
            )
        st.dataframe(pd.DataFrame(scen_rows), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: STAGE 1 ACCEPTANCE TESTS
# ------------------------------------------------------------------------------
with tab_tests:
    st.subheader("Automated Stage 1 Verification & Fixture Replay")
    st.warning("Fixture-test mode uses explicit capacity_override only to verify ledger math against the handbook.")

    if st.button("Run All Stage 1 Acceptance Checks"):
        base = default_state()
        results = []

        # 1. Three-month fixture replay
        fixture_rows = [
            ("2027-01", 80, 60, 0.0, 60, Decimal("1500.00"), Decimal("1000.00"), Decimal("500.00"), Decimal("4500.00")),
            ("2027-02", 88, 60, 100.0, 60, Decimal("1500.00"), Decimal("1100.00"), Decimal("400.00"), Decimal("4900.00")),
            ("2027-03", 72, 45, 0.0, 45, Decimal("1125.00"), Decimal("925.00"), Decimal("200.00"), Decimal("5100.00")),
        ]
        cash_cursor = Decimal("4000.00")
        for m, req, cap_ov, ad, exp_sold, exp_rev, exp_exp, exp_prof, exp_cash in fixture_rows:
            inp = dict(base["inputs"], requested=req, ad_cost=ad)
            out = compute_nur_period(
                inp, base["variable_cost_lines"], base["fixed_cost_lines"], cash_cursor, capacity_override=cap_ov
            )
            passed = (
                out["sold"] == exp_sold
                and out["revenue"] == exp_rev
                and out["total_expense"] == exp_exp
                and out["operating_profit"] == exp_prof
                and out["ending_cash"] == exp_cash
            )
            results.append(
                {
                    "Test": f"Fixture {m}",
                    "Status": "PASS" if passed else "FAIL",
                    "Details": f"Sold={out['sold']}, Rev=${out['revenue']}, Profit=${out['operating_profit']}, Cash=${out['ending_cash']}",
                }
            )
            cash_cursor = out["ending_cash"]

        # 2. NUR Business-specific no-show test
        noshow_inp = dict(base["inputs"], requested=80, no_show_rate=0.20)
        ns_out = compute_nur_period(
            noshow_inp, base["variable_cost_lines"], base["fixed_cost_lines"], Decimal("4000.00")
        )
        ns_pass = (
            ns_out["booked"] == 60
            and ns_out["attended"] == 48
            and ns_out["revenue"] == Decimal("1200.00")
            and ns_out["variable_expense"] == Decimal("240.00")
            and ns_out["operating_profit"] == Decimal("260.00")
        )
        results.append(
            {
                "Test": "NUR No-Show Rate = 0.20",
                "Status": "PASS" if ns_pass else "FAIL",
                "Details": f"Booked={ns_out['booked']}, Attended={ns_out['attended']}, Profit=${ns_out['operating_profit']}",
            }
        )

        # 3. Zero sales test
        zero_inp = dict(base["inputs"], requested=0)
        z_out = compute_nur_period(zero_inp, base["variable_cost_lines"], base["fixed_cost_lines"], Decimal("4000.00"))
        z_pass = z_out["revenue"] == Decimal("0.00") and z_out["operating_profit"] == Decimal("-700.00")
        results.append(
            {
                "Test": "Zero Sales Edge Case",
                "Status": "PASS" if z_pass else "FAIL",
                "Details": f"Revenue=${z_out['revenue']}, Profit=${z_out['operating_profit']}",
            }
        )

        # 4. AP Calculus acceptance fixture
        calc_test = solve_ap_calculus_model(D0=60, p0=30, v=10, F=500, capacity=40)
        c_pass = (
            abs(calc_test["constrained_opt_p"] - 40.0) < 1e-6
            and abs(calc_test["constrained_opt_profit"] - 700.0) < 1e-6
            and not calc_test["unconstrained_feasible"]
        )
        results.append(
            {
                "Test": "AP Calculus Constrained Optimum Fixture",
                "Status": "PASS" if c_pass else "FAIL",
                "Details": f"Optimum p=${calc_test['constrained_opt_p']:.2f}, Profit=${calc_test['constrained_opt_profit']:.2f}",
            }
        )

        # 5. Wrong team & stale version rejection
        wrong_team_ok, _, _ = validate_and_load_snapshot(
            json.dumps(dict(base, team_id="OTHER_TEAM")), current_version=1
        )
        stale_ok, _, _ = validate_and_load_snapshot(
            json.dumps(dict(base, state_version=1)), current_version=2
        )
        sec_pass = (not wrong_team_ok) and (not stale_ok)
        results.append(
            {
                "Test": "Wrong Team & Stale Version Rejection",
                "Status": "PASS" if sec_pass else "FAIL",
                "Details": "Rejected wrong team_id and stale state_version as required.",
            }
        )

        st.dataframe(pd.DataFrame(results), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: STAGE 2-4 ROADMAP
# ------------------------------------------------------------------------------
with tab_roadmap:
    st.subheader("Unimplemented Modules (Planned for Later Stages)")
    st.markdown(
        "- **Stage 2:** Full Report Center (Income Statement, Cash Flow, Budget Variance, Simplified Balance Sheet, PDF/CSV exports), "
        "Costed Marketing Studio (A/B variant comparison & campaign_id logging), and AI Recommendation Import Panel.\n"
        "- **Stage 3:** Cross-Course Connections Panel (2 non-math subjects) and Snapshot-Based Practice Assessment Generator.\n"
        "- **Stage 4 (Optional):** Demonstration Storefront (test orders only, no real payments)."
    )
