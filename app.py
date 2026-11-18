"""
Career Venture Lab - Complete Self-Contained Streamlit Application
Team ID: Life Savers
Pathway: NUR-3 (Nurse-Led Health Education Simulation | AP Calculus)

Single-file deployment ready for GitHub + Render:
  streamlit run app.py --server.address 0.0.0.0 --server.port $PORT

Simulation Guardrail:
  Fictional nonclinical health education workshops only.
  No medical diagnosis, clinical treatment, real patient data, or real payments.
"""

from __future__ import annotations

import copy
import io
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ==============================================================================
# 1. DETERMINISTIC ACCOUNTING & SIMULATION ENGINE (NUR-3)
# ==============================================================================

TWO_PLACES = Decimal("0.01")

EMBEDDED_MONTHLY_FIXTURE = [
    {"month": "2027-01", "requested": 80, "capacity_override": 60, "ad_cost": "0.00"},
    {"month": "2027-02", "requested": 88, "capacity_override": 60, "ad_cost": "100.00"},
    {"month": "2027-03", "requested": 72, "capacity_override": 45, "ad_cost": "0.00"},
]

REQUIRED_SNAPSHOT_KEYS = {
    "team_id",
    "course",
    "pathway_id",
    "snapshot_id",
    "state_version",
    "simulation_date",
    "funding",
    "balances",
    "parameters",
    "variable_cost_lines",
    "fixed_cost_lines",
    "ledger",
}


def money(val: Any) -> Decimal:
    """Convert any number or string into exact 2-decimal USD Decimal."""
    d = val if isinstance(val, Decimal) else Decimal(str(val))
    return d.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def fmt_usd(val: Any) -> str:
    d = money(val)
    sign = "-" if d < 0 else ""
    return f"{sign}${abs(d):,.2f}"


def default_nur3_state(team_id: str = "Life Savers", selected_level: str = "Advanced") -> Dict[str, Any]:
    """Exact NUR-3 baseline state from Career Venture Lab Student Handbook (pp. 37, 70, 83-85)."""
    return {
        "team_id": team_id,
        "course": "AP Calculus",
        "pathway_id": "NUR-3",
        "business_name": "Nurse-Led Health Education Simulation",
        "primary_offering": "General wellness education workshop",
        "secondary_offerings": [
            "Caregiver information session",
            "First-aid awareness demonstration",
        ],
        "selected_level": selected_level,
        "teacher_approved_skills": [
            "Constrained Profit Optimization with Derivatives",
            "Accumulated Revenue/Cost Rate Integration",
        ],
        "snapshot_id": "SNAP-000",
        "state_version": 1,
        "simulation_date": "2026-12-31",
        "base_period": "month",
        "working_days": 20,
        "calendar_days": 30,
        "currency": "USD",
        "random_seed": 2027,
        "assumptions": [
            "Fictional general health education workshops only; no clinical diagnosis or patient data.",
            "Simplified accrual accounting with immediate cash collection on attended seats.",
            "Taxes, inventory valuation, and depreciation are disabled at baseline.",
            "Continuous calculus curves are separated from integer-floored ledger sales.",
        ],
        "sources": [
            "Career Venture Lab Student Handbook (November 2026), Pathway NUR-3 (p. 37) & NUR Package (pp. 83-87)."
        ],
        "funding": {
            "owner_contribution": "6000.00",
            "equipment_assets": "1500.00",
            "pre_opening_expense": "500.00",
            "opening_inventory": "0.00",
            "opening_cash": "4000.00",
            "loan_balance": "0.00",
            "retained_earnings": "-500.00",
        },
        "balances": {
            "cash": "4000.00",
            "equipment_assets": "1500.00",
            "loan_debt": "0.00",
            "owner_equity": "6000.00",
            "retained_earnings": "-500.00",
        },
        "parameters": {
            "base_price": "25.00",
            "D0": 80,
            "staff_hours": 30.0,
            "hours_per_unit": 0.5,
            "venue_seats": 60,
            "no_show_rate": 0.0,
            "satisfaction_mean": 4.0,
        },
        "variable_cost_lines": {
            "educator_delivery": "2.00",
            "materials": "2.00",
            "cleaning_consumables": "1.00",
        },
        "fixed_cost_lines": {
            "venue": "250.00",
            "scheduling_software": "100.00",
            "accessibility_support": "200.00",
            "admin_insurance_assumption": "150.00",
        },
        "ledger": [],
        "processed_run_ids": [],
        "monthly_summaries": [],
        "campaigns": [],
        "decisions": [],
    }


def validate_inputs(state: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not str(state.get("team_id", "")).strip():
        errors.append("team_id must be non-empty.")
    if state.get("pathway_id") != "NUR-3":
        errors.append(f"Expected pathway_id 'NUR-3', got '{state.get('pathway_id')}'.")

    params = state["parameters"]
    if money(params["base_price"]) <= Decimal("0.00"):
        errors.append("base_price must be > 0.")
    if int(params["D0"]) < 0:
        errors.append("D0 must be >= 0.")
    if float(params["staff_hours"]) < 0:
        errors.append("staff_hours must be >= 0.")
    if float(params["hours_per_unit"]) <= 0:
        errors.append("hours_per_unit must be > 0.")
    if int(params["venue_seats"]) < 0:
        errors.append("venue_seats must be >= 0.")
    nsr = float(params["no_show_rate"])
    if not (0.0 <= nsr <= 1.0):
        errors.append("no_show_rate must be between 0 and 1.")

    for k, v in state["variable_cost_lines"].items():
        if money(v) < Decimal("0.00"):
            errors.append(f"Variable cost '{k}' cannot be negative.")
    for k, v in state["fixed_cost_lines"].items():
        if money(v) < Decimal("0.00"):
            errors.append(f"Fixed cost '{k}' cannot be negative.")
    return errors


def compute_capacity(state: Dict[str, Any], capacity_override: Optional[int] = None) -> Dict[str, Any]:
    params = state["parameters"]
    staff_cap = math.floor(float(params["staff_hours"]) / float(params["hours_per_unit"]))
    venue_cap = int(params["venue_seats"])
    derived_cap = min(staff_cap, venue_cap)

    if capacity_override is not None:
        effective_cap = int(capacity_override)
        bottleneck = "Fixture capacity_override"
    else:
        effective_cap = derived_cap
        if staff_cap < venue_cap:
            bottleneck = f"Educator hours ({staff_cap} seats)"
        elif venue_cap < staff_cap:
            bottleneck = f"Venue seats ({venue_cap} seats)"
        else:
            bottleneck = f"Tied: Educator hours & Venue ({effective_cap} seats)"

    return {
        "staff_capacity": staff_cap,
        "venue_capacity": venue_cap,
        "derived_capacity": derived_cap,
        "effective_capacity": effective_cap,
        "bottleneck": bottleneck,
    }


def unit_variable_cost(state: Dict[str, Any]) -> Decimal:
    return sum((money(v) for v in state["variable_cost_lines"].values()), Decimal("0.00"))


def monthly_fixed_cost(state: Dict[str, Any]) -> Decimal:
    return sum((money(v) for v in state["fixed_cost_lines"].values()), Decimal("0.00"))


def run_simulation_period(
    state: Dict[str, Any],
    *,
    run_id: str,
    period_date: str,
    requested: int,
    price: Optional[Any] = None,
    ad_cost: Any = "0.00",
    capacity_override: Optional[int] = None,
    loan_cash: Any = "0.00",
    loan_principal_repayment: Any = "0.00",
    interest_expense: Any = "0.00",
    new_owner_cash: Any = "0.00",
    additional_capex: Any = "0.00",
    refund_amount: Any = "0.00",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if requested < 0:
        raise ValueError("Requested seats cannot be negative.")
    if any(
        money(x) < Decimal("0.00")
        for x in [
            ad_cost,
            loan_cash,
            loan_principal_repayment,
            interest_expense,
            new_owner_cash,
            additional_capex,
            refund_amount,
        ]
    ):
        raise ValueError("Financial inputs cannot be negative.")

    if run_id in state["processed_run_ids"]:
        return state, {
            "status": "duplicate_ignored",
            "message": f"Run ID '{run_id}' already exists in ledger. No duplicate transaction recorded.",
        }

    errs = validate_inputs(state)
    if errs:
        raise ValueError("Invalid simulation state: " + "; ".join(errs))

    new_state = copy.deepcopy(state)
    params = new_state["parameters"]
    unit_price = money(price if price is not None else params["base_price"])
    no_show_rate = float(params["no_show_rate"])

    cap_info = compute_capacity(new_state, capacity_override=capacity_override)
    eff_cap = cap_info["effective_capacity"]

    booked = min(int(requested), eff_cap)
    unserved_demand = max(0, int(requested) - booked)
    attended = math.floor(booked * (1.0 - no_show_rate))
    no_shows = booked - attended
    sold = attended

    v_unit = unit_variable_cost(new_state)
    f_month = monthly_fixed_cost(new_state)
    mkt_cost = money(ad_cost)
    int_exp = money(interest_expense)
    ref_amt = money(refund_amount)
    loan_in = money(loan_cash)
    loan_prin = money(loan_principal_repayment)
    owner_in = money(new_owner_cash)
    capex_out = money(additional_capex)

    gross_revenue = money(unit_price * sold)
    net_revenue = money(gross_revenue - ref_amt)
    var_expense = money(v_unit * sold)
    total_operating_expense = money(var_expense + f_month + mkt_cost)
    operating_profit = money(net_revenue - total_operating_expense)
    net_profit = money(operating_profit - int_exp)

    opening_cash = money(new_state["balances"]["cash"])
    closing_cash = money(
        opening_cash
        + net_revenue
        - var_expense
        - f_month
        - mkt_cost
        - int_exp
        - loan_prin
        - capex_out
        + owner_in
        + loan_in
    )

    next_version = int(new_state["state_version"]) + 1
    new_state["state_version"] = next_version
    new_state["simulation_date"] = period_date
    new_state["snapshot_id"] = f"SNAP-{next_version - 1:03d}"
    new_state["processed_run_ids"].append(run_id)

    new_state["balances"]["cash"] = f"{closing_cash:.2f}"
    new_state["balances"]["equipment_assets"] = f"{money(new_state['balances']['equipment_assets']) + capex_out:.2f}"
    new_state["balances"]["loan_debt"] = f"{money(new_state['balances']['loan_debt']) + loan_in - loan_prin:.2f}"
    new_state["balances"]["owner_equity"] = f"{money(new_state['balances']['owner_equity']) + owner_in:.2f}"
    new_state["balances"]["retained_earnings"] = f"{money(new_state['balances']['retained_earnings']) + net_profit:.2f}"

    tx_counter = len(new_state["ledger"]) + 1

    def add_tx(
        event_type: str,
        category: str,
        amount: Decimal,
        qty: int = 0,
        u_price: Decimal = Decimal("0.00"),
        fin_type: str = "operating",
    ) -> None:
        nonlocal tx_counter
        if amount == Decimal("0.00"):
            return
        new_state["ledger"].append(
            {
                "transaction_id": f"TX-{tx_counter:04d}",
                "run_id": run_id,
                "date": period_date,
                "event_type": event_type,
                "category": category,
                "amount": f"{amount:.2f}",
                "quantity": qty,
                "unit_price": f"{u_price:.2f}",
                "offering_id": "OFF-NUR-01",
                "campaign_id": f"CMP-{run_id}" if category == "marketing" else "",
                "financing_type": fin_type,
                "state_version": next_version,
            }
        )
        tx_counter += 1

    add_tx("revenue", "workshop_sales", net_revenue, qty=sold, u_price=unit_price, fin_type="operating")
    add_tx("expense", "variable_cost", var_expense, qty=sold, u_price=v_unit, fin_type="operating")
    add_tx("expense", "fixed_cost", f_month, qty=1, u_price=f_month, fin_type="operating")
    if mkt_cost > Decimal("0.00"):
        add_tx("expense", "marketing", mkt_cost, qty=1, u_price=mkt_cost, fin_type="operating")
    if int_exp > Decimal("0.00"):
        add_tx("expense", "interest_expense", int_exp, fin_type="financing")
    if loan_in > Decimal("0.00"):
        add_tx("financing_inflow", "loan_proceeds", loan_in, fin_type="financing")
    if loan_prin > Decimal("0.00"):
        add_tx("financing_outflow", "loan_principal_repayment", loan_prin, fin_type="financing")
    if owner_in > Decimal("0.00"):
        add_tx("financing_inflow", "owner_contribution", owner_in, fin_type="equity")
    if capex_out > Decimal("0.00"):
        add_tx("investing_outflow", "equipment_capex", capex_out, fin_type="investing")

    summary = {
        "status": "committed",
        "run_id": run_id,
        "date": period_date,
        "state_version": next_version,
        "mode": "FIXTURE_OVERRIDE" if capacity_override is not None else "NORMAL",
        "requested": int(requested),
        "capacity": eff_cap,
        "bottleneck": cap_info["bottleneck"],
        "booked": booked,
        "unserved_demand": unserved_demand,
        "no_shows": no_shows,
        "sold": sold,
        "unit_price": f"{unit_price:.2f}",
        "revenue": f"{net_revenue:.2f}",
        "variable_expense": f"{var_expense:.2f}",
        "fixed_expense": f"{f_month:.2f}",
        "marketing_expense": f"{mkt_cost:.2f}",
        "total_operating_expense": f"{total_operating_expense:.2f}",
        "operating_profit": f"{operating_profit:.2f}",
        "interest_expense": f"{int_exp:.2f}",
        "net_profit": f"{net_profit:.2f}",
        "opening_cash": f"{opening_cash:.2f}",
        "closing_cash": f"{closing_cash:.2f}",
    }
    new_state["monthly_summaries"].append(summary)
    return new_state, summary


def replay_fixture_rows(
    base_state: Dict[str, Any],
    fixture_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = fixture_rows if fixture_rows is not None else EMBEDDED_MONTHLY_FIXTURE
    working = default_nur3_state(
        team_id=base_state.get("team_id", "Life Savers"),
        selected_level=base_state.get("selected_level", "Advanced"),
    )
    results = []
    for row in rows:
        month_str = str(row["month"]).strip()
        working, summary = run_simulation_period(
            working,
            run_id=f"FIXTURE-{month_str}",
            period_date=f"{month_str}-28",
            requested=int(row["requested"]),
            ad_cost=str(row["ad_cost"]),
            capacity_override=int(row["capacity_override"]),
        )
        results.append(summary)
    return working, results


def export_snapshot_json(state: Dict[str, Any]) -> str:
    return json.dumps(state, indent=2)


def import_snapshot_json(
    raw_json: str,
    *,
    expected_team_id: str,
    current_state_version: Optional[int] = None,
    reject_stale: bool = True,
) -> Dict[str, Any]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON format: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Snapshot JSON must be an object.")

    missing = REQUIRED_SNAPSHOT_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Snapshot missing required fields: {sorted(missing)}")

    if str(data.get("team_id", "")).strip() != expected_team_id.strip():
        raise ValueError(
            f"Team ID mismatch: expected '{expected_team_id}', got '{data.get('team_id')}'. Import rejected."
        )

    if data.get("pathway_id") != "NUR-3":
        raise ValueError(
            f"Pathway mismatch: expected 'NUR-3', got '{data.get('pathway_id')}'. Import rejected."
        )

    incoming_version = int(data["state_version"])
    if reject_stale and current_state_version is not None and incoming_version < int(current_state_version):
        raise ValueError(
            f"Stale state_version rejected: incoming v{incoming_version} is older than current v{current_state_version}."
        )

    errs = validate_inputs(data)
    if errs:
        raise ValueError("Snapshot validation failed: " + "; ".join(errs))

    return data


# ==============================================================================
# 2. AP CALCULUS COURSE ADAPTER (NUR-3)
# ==============================================================================

def ap_calculus_analysis(
    D0: float,
    p0: float,
    v: float,
    F: float,
    capacity: int,
    test_price: Optional[float] = None,
) -> Dict[str, Any]:
    a = 2.0 * float(D0)
    b = float(D0) / float(p0)
    p_max = 2.0 * float(p0)

    p_unconstrained = (a + b * v) / (2.0 * b)
    q_at_unconstrained = max(0.0, a - b * p_unconstrained)
    profit_unconstrained = (p_unconstrained - v) * q_at_unconstrained - F

    p_corner = max(0.0, min(p_max, (a - float(capacity)) / b))

    def eval_point(p: float) -> Dict[str, float]:
        q_raw = max(0.0, a - b * p)
        s_cont = min(float(capacity), q_raw)
        s_int = math.floor(s_cont)
        prof_cont = (p - v) * s_cont - F
        prof_int = (p - v) * s_int - F
        if abs(p - p_corner) < 1e-9 and 0 < p_corner < p_max:
            deriv = float("nan")
        elif p < p_corner:
            deriv = float(capacity)
        else:
            deriv = a + b * v - 2.0 * b * p
        return {
            "price": p,
            "demand_q": q_raw,
            "completed_continuous": s_cont,
            "completed_integer": s_int,
            "profit_continuous": prof_cont,
            "profit_integer": prof_int,
            "derivative_dP_dp": deriv,
        }

    candidates = [0.0, p_max, p_corner]
    if p_corner <= p_unconstrained <= p_max:
        candidates.append(p_unconstrained)

    evaluated_candidates = [eval_point(c) for c in candidates]
    best_cont = max(evaluated_candidates, key=lambda item: item["profit_continuous"])
    selected_eval = eval_point(float(test_price)) if test_price is not None else best_cont

    return {
        "a": a,
        "b": b,
        "price_domain": (0.0, p_max),
        "p_unconstrained": p_unconstrained,
        "q_at_unconstrained": q_at_unconstrained,
        "profit_unconstrained": profit_unconstrained,
        "unconstrained_is_feasible": q_at_unconstrained <= float(capacity) + 1e-9,
        "p_corner": p_corner,
        "constrained_optimum": best_cont,
        "candidates": evaluated_candidates,
        "selected_point": selected_eval,
    }


def ap_calculus_rate_integral(
    t_start: float = 0.0,
    t_end: float = 10.0,
    r0: float = 100.0,
    r_slope: float = 20.0,
    c0: float = 60.0,
    c_slope: float = 5.0,
) -> Dict[str, float]:
    def antideriv_r(t: float) -> float:
        return r0 * t + 0.5 * r_slope * (t ** 2)

    def antideriv_c(t: float) -> float:
        return c0 * t + 0.5 * c_slope * (t ** 2)

    acc_rev = antideriv_r(t_end) - antideriv_r(t_start)
    acc_cost = antideriv_c(t_end) - antideriv_c(t_start)
    acc_profit = acc_rev - acc_cost
    return {
        "t_start": t_start,
        "t_end": t_end,
        "accumulated_revenue": acc_rev,
        "accumulated_cost": acc_cost,
        "accumulated_profit": acc_profit,
    }


# ==============================================================================
# 3. STREAMLIT USER INTERFACE
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Life Savers | NUR-3 Career Venture Lab",
        page_icon="🩺",
        layout="wide",
    )

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = default_nur3_state(team_id="Life Savers", selected_level="Advanced")

    state = st.session_state.sim_state

    with st.sidebar:
        st.header("🩺 Life Savers (`NUR-3`)")
        st.caption("Nurse-Led Health Education Simulation • AP Calculus")
        st.info(
            "**Simulation Guardrail:** Fictional nonclinical health education workshops only. "
            "No medical diagnosis, clinical care, or real personal/patient data."
        )
        st.metric("Current Cash Balance", fmt_usd(state["balances"]["cash"]))
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("State Version", f"v{state['state_version']}")
        col_s2.metric("Snapshot ID", state["snapshot_id"])
        st.caption(f"Sim Date: `{state['simulation_date']}` | Ledger Rows: `{len(state['ledger'])}`")

        st.divider()
        st.subheader("Module Status")
        st.success("✅ **Core Engine & Math:** Active")
        st.success("✅ **Reports & Marketing:** Active")
        st.info("ℹ️ **AI Handoffs:** Manual Snapshot Export/Import")

    st.title("Career Venture Lab: Nurse-Led Health Education Studio (`NUR-3`)")
    st.markdown(
        f"**Team:** `{state['team_id']}` | **Course:** `{state['course']}` | "
        f"**Challenge Level:** `{state['selected_level']}` | **Primary Offering:** {state['primary_offering']}"
    )

    tabs = st.tabs(
        [
            "1. Overview & System Map",
            "2. Inputs & Funding",
            "3. Math Skills (AP Calculus)",
            "4. Run Simulation & Ledger",
            "5. Report Center",
            "6. Marketing Studio",
            "7. Save / Load Snapshot",
            "8. Acceptance Tests",
        ]
    )

    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW & SYSTEM MAP
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Business Operating Snapshot")
        cap_info = compute_capacity(state)
        v_cost = unit_variable_cost(state)
        f_cost = monthly_fixed_cost(state)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline Seat Price (p0)", fmt_usd(state["parameters"]["base_price"]))
        c2.metric("Monthly Capacity", f"{cap_info['effective_capacity']} seats", help=cap_info["bottleneck"])
        c3.metric("Variable Cost / Seat (v)", fmt_usd(v_cost))
        c4.metric("Monthly Fixed Cost (F)", fmt_usd(f_cost))

        st.markdown("#### Current Balance Sheet")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Spendable Cash", fmt_usd(state["balances"]["cash"]))
        b2.metric("Equipment & Access Assets", fmt_usd(state["balances"]["equipment_assets"]))
        b3.metric("Loan Debt", fmt_usd(state["balances"]["loan_debt"]))
        b4.metric(
            "Owner Equity / Ret. Earnings",
            f"{fmt_usd(state['balances']['owner_equity'])} / {fmt_usd(state['balances']['retained_earnings'])}",
        )

        st.markdown("#### NUR-3 System Map (Automatic vs. Student-Managed)")
        st.code(
            """[Marketing Studio] ──(approved ad campaign)──> [Customer Demand (D0)]
       ^                                                  │
       │                                         (seat requests)
(later demand shift)                                      v
       │                                   [Operations & Capacity Engine]
[Satisfaction & Attendance] <──(attended seats)── min(requests, floor(staff_hrs/0.5), venue_seats)
       │                                                  │
       └──(revenue, variable & fixed costs)──> [Dated Ledger & Reports] ──> [Student Approval]

• AUTOMATIC (Python Engine): Seat capacity, no-show rounding, revenue/cost math, cash reconciliation, duplicate run_id guard.
• STUDENT-MANAGED: Seat price, resource edits, approving simulation steps, verifying AI proposals, and writing Decision Memos.""",
            language="text",
        )

    # --------------------------------------------------------------------------
    # TAB 2: INPUTS & FUNDING
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Editable Business & Resource Parameters")
        st.caption("Adjust assumptions here, then commit a simulation period in Tab 4 to write to the ledger.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Workshop Demand & Capacity Inputs")
            new_price = st.number_input(
                "Base Price per Workshop Seat ($ USD)",
                min_value=1.0,
                max_value=500.0,
                value=float(state["parameters"]["base_price"]),
                step=1.0,
            )
            new_d0 = st.number_input(
                "Baseline Monthly Requested Seats (D0)",
                min_value=0,
                max_value=1000,
                value=int(state["parameters"]["D0"]),
                step=5,
            )
            new_staff_hrs = st.number_input(
                "Educator Staff Hours / Month",
                min_value=0.0,
                max_value=500.0,
                value=float(state["parameters"]["staff_hours"]),
                step=1.0,
            )
            new_hrs_unit = st.number_input(
                "Educator Hours per Seat",
                min_value=0.1,
                max_value=10.0,
                value=float(state["parameters"]["hours_per_unit"]),
                step=0.1,
            )
            new_venue = st.number_input(
                "Venue Seat Capacity / Month",
                min_value=0,
                max_value=1000,
                value=int(state["parameters"]["venue_seats"]),
                step=5,
            )
            new_noshow = st.slider(
                "No-Show Rate (0.00 to 1.00)",
                min_value=0.0,
                max_value=1.0,
                value=float(state["parameters"]["no_show_rate"]),
                step=0.05,
            )

        with col_right:
            st.markdown("##### Per-Seat Variable Cost Lines ($ / attended seat)")
            v_edu = st.number_input(
                "Educator Delivery ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["educator_delivery"]),
                step=0.5,
            )
            v_mat = st.number_input(
                "Workshop Materials ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["materials"]),
                step=0.5,
            )
            v_cln = st.number_input(
                "Cleaning & Consumables ($/seat)",
                min_value=0.0,
                value=float(state["variable_cost_lines"]["cleaning_consumables"]),
                step=0.5,
            )

            st.markdown("##### Monthly Fixed Cost Lines ($ / month)")
            f_ven = st.number_input(
                "Venue Rental ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["venue"]),
                step=10.0,
            )
            f_sof = st.number_input(
                "Scheduling Software ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["scheduling_software"]),
                step=10.0,
            )
            f_acc = st.number_input(
                "Accessibility & Translation Support ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["accessibility_support"]),
                step=10.0,
            )
            f_adm = st.number_input(
                "Admin & Insurance Assumption ($/mo)",
                min_value=0.0,
                value=float(state["fixed_cost_lines"]["admin_insurance_assumption"]),
                step=10.0,
            )

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Apply Parameter Changes", type="primary"):
            state["parameters"]["base_price"] = f"{money(new_price):.2f}"
            state["parameters"]["D0"] = int(new_d0)
            state["parameters"]["staff_hours"] = float(new_staff_hrs)
            state["parameters"]["hours_per_unit"] = float(new_hrs_unit)
            state["parameters"]["venue_seats"] = int(new_venue)
            state["parameters"]["no_show_rate"] = float(new_noshow)
            state["variable_cost_lines"] = {
                "educator_delivery": f"{money(v_edu):.2f}",
                "materials": f"{money(v_mat):.2f}",
                "cleaning_consumables": f"{money(v_cln):.2f}",
            }
            state["fixed_cost_lines"] = {
                "venue": f"{money(f_ven):.2f}",
                "scheduling_software": f"{money(f_sof):.2f}",
                "accessibility_support": f"{money(f_acc):.2f}",
                "admin_insurance_assumption": f"{money(f_adm):.2f}",
            }
            st.success("Parameters updated!")
            st.rerun()

        if col_btn2.button("Reset to NUR-3 Handbook Baseline"):
            st.session_state.sim_state = default_nur3_state(
                team_id=state["team_id"], selected_level=state["selected_level"]
            )
            st.success("Reset to NUR-3 baseline state.")
            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 3: MATH SKILLS & CHALLENGE LEVEL PANEL (AP CALCULUS)
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Math Skills & Challenge Level Panel — AP Calculus (`NUR-3`)")
        level = st.selectbox(
            "Select Challenge Level:",
            ["Foundation", "Applied", "Advanced"],
            index=["Foundation", "Applied", "Advanced"].index(state["selected_level"]),
        )
        if level != state["selected_level"]:
            state["selected_level"] = level
            st.rerun()

        p0 = float(state["parameters"]["base_price"])
        D0 = float(state["parameters"]["D0"])
        v = float(unit_variable_cost(state))
        F = float(monthly_fixed_cost(state))
        cap = compute_capacity(state)["effective_capacity"]

        test_p = st.slider(
            "Explore Workshop Seat Price p ($ USD) — Predict before moving!",
            min_value=0.0,
            max_value=float(2 * p0),
            value=float(p0),
            step=0.25,
        )
        calc = ap_calculus_analysis(D0=D0, p0=p0, v=v, F=F, capacity=cap, test_price=test_p)
        sel = calc["selected_point"]

        st.markdown(
            f"**Course:** AP Calculus | **Level:** `{level}` | **Domain:** $0 \\le p \\le {2*p0:.2f}$ USD/seat | "
            f"**Capacity Limit:** {cap} seats/month"
        )

        if level == "Foundation":
            st.info(
                "**Foundation Task:** Derive unconstrained profit $P(p) = (p - v)(a - bp) - F$, compute $P'(p) = a + bv - 2bp$, "
                "and test the critical point $p = (a + bv)/(2b)$ and endpoints."
            )
            st.latex(
                rf"q(p) = {calc['a']:.1f} - {calc['b']:.2f}p \quad\Rightarrow\quad "
                rf"P(p) = (p - {v:.2f})({calc['a']:.1f} - {calc['b']:.2f}p) - {F:.2f}"
            )
            st.latex(
                rf"P'(p) = {calc['a'] + calc['b']*v:.2f} - {2*calc['b']:.2f}p = 0 "
                rf"\implies p^*_{{\text{{unconstrained}}}} = \${calc['p_unconstrained']:.2f}"
            )
            feas_txt = (
                "feasible"
                if calc["unconstrained_is_feasible"]
                else f"INFEASIBLE (exceeds {cap}-seat capacity!)"
            )
            st.write(
                f"At $p = \\${calc['p_unconstrained']:.2f}$, unconstrained demand is "
                f"**{calc['q_at_unconstrained']:.2f} seats**, which is **{feas_txt}**."
            )

        elif level == "Applied":
            st.info(
                "**Applied Task:** Enforce completed seats $s(p) = \\min(\\text{capacity}, \\max(0, q(p)))$, check the capacity corner "
                "$p = (a - \\text{capacity})/b$, endpoints, and compare continuous calculus optima with integer-rounded sales."
            )
            st.latex(
                rf"s(p) = \min({cap},\, \max(0,\, {calc['a']:.1f} - {calc['b']:.2f}p)), \quad "
                rf"p_{{\text{{corner}}}} = \frac{{{calc['a']:.1f} - {cap}}}{{{calc['b']:.2f}}} = \${calc['p_corner']:.2f}"
            )
            opt = calc["constrained_optimum"]
            st.success(
                f"**Constrained Continuous Optimum:** $p = \\${opt['price']:.2f}$ → "
                f"Continuous Profit = **${opt['profit_continuous']:,.2f}**, Integer-Floored Profit ({opt['completed_integer']} seats) = **${opt['profit_integer']:,.2f}**."
            )

        else:
            st.info(
                "**Advanced Task:** Combine constrained boundary optimization with rate-of-change accumulation "
                "$\\int_{t_1}^{t_2} [r(t) - c(t)]\\,dt$ over a campaign window, and evaluate sensitivity."
            )
            opt = calc["constrained_optimum"]
            st.write(
                f"**1. Constrained Boundary Optimum:** Corner at $p = \\${calc['p_corner']:.2f}$ yields **${opt['profit_continuous']:,.2f}/mo** "
                f"(whereas unconstrained critical point $p = \\${calc['p_unconstrained']:.2f}$ demands {calc['q_at_unconstrained']:.1f} seats > {cap} capacity)."
            )
            st.markdown("**2. Time-Dependent Rate Accumulation Exercise (Handbook p. 70):**")
            st.latex(
                r"r(t) = 100 + 20t \;\text{(\$/day)}, \quad c(t) = 60 + 5t \;\text{(\$/day)} "
                r"\quad \text{on } 0 \le t \le 10 \text{ days}"
            )
            col_r1, col_r2 = st.columns(2)
            t_end = col_r1.slider("Campaign Window End Day (t_end)", min_value=1, max_value=30, value=10)
            integ = ap_calculus_rate_integral(0.0, float(t_end))
            col_r2.metric(
                f"Accumulated Net Change (0 to {t_end} days)",
                f"${integ['accumulated_profit']:,.2f}",
                delta=f"Rev ${integ['accumulated_revenue']:,.2f} - Cost ${integ['accumulated_cost']:,.2f}",
            )

        prices = np.linspace(0.0, 2.0 * p0, 241)
        q_vals = np.maximum(0.0, calc["a"] - calc["b"] * prices)
        s_cont_vals = np.minimum(float(cap), q_vals)
        s_int_vals = np.floor(s_cont_vals)
        prof_cont_vals = (prices - v) * s_cont_vals - F
        prof_int_vals = (prices - v) * s_int_vals - F

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax1.plot(prices, prof_cont_vals, label="Continuous Profit P(p)", linewidth=2, color="#1f77b4")
        ax1.step(
            prices,
            prof_int_vals,
            where="post",
            label="Integer-Floored Profit",
            linestyle="--",
            color="#ff7f0e",
            alpha=0.85,
        )
        ax1.axhline(0, color="black", linewidth=0.9, linestyle=":")
        ax1.axvline(calc["p_corner"], color="#2ca02c", linestyle="-.", label=f"Capacity Corner (${calc['p_corner']:.2f})")
        ax1.scatter([test_p], [sel["profit_continuous"]], color="#d62728", zorder=5, label=f"Selected p=${test_p:.2f}")
        ax1.set_ylabel("Monthly Profit ($ USD)")
        ax1.set_title(f"NUR-3 Profit & Derivative Analysis ({level} Mode) | Snapshot {state['snapshot_id']}")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        left_mask = prices < calc["p_corner"]
        right_mask = prices > calc["p_corner"]
        ax2.plot(
            prices[left_mask],
            np.full(np.sum(left_mask), float(cap)),
            color="#9467bd",
            linewidth=2,
            label="P'(p) = capacity (constrained branch)",
        )
        ax2.plot(
            prices[right_mask],
            calc["a"] + calc["b"] * v - 2 * calc["b"] * prices[right_mask],
            color="#8c564b",
            linewidth=2,
            label="P'(p) = a + bv - 2bp (demand branch)",
        )
        ax2.axhline(0, color="black", linewidth=0.9, linestyle=":")
        if 0 < calc["p_corner"] < 2.0 * p0:
            ax2.scatter(
                [calc["p_corner"]],
                [float(cap)],
                facecolors="none",
                edgecolors="#d62728",
                s=60,
                zorder=5,
                label="Undefined Corner",
            )
            ax2.scatter(
                [calc["p_corner"]],
                [calc["a"] + calc["b"] * v - 2 * calc["b"] * calc["p_corner"]],
                facecolors="none",
                edgecolors="#d62728",
                s=60,
                zorder=5,
            )
        ax2.set_xlabel("Workshop Seat Price p ($ USD / seat)")
        ax2.set_ylabel("Marginal Profit P'(p) ($/mo per $)")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            "Download Graph PNG",
            data=buf.getvalue(),
            file_name=f"NUR3_math_{state['snapshot_id']}.png",
            mime="image/png",
        )
        st.dataframe(pd.DataFrame([sel]), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: RUN SIMULATION & DATED LEDGER
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Advance Simulation Period (Normal Mode)")
        st.caption("Each approved run uses a unique run_id and advances state_version by +1.")

        with st.form("run_period_form"):
            c_a, c_b, c_c = st.columns(3)
            run_id_input = c_a.text_input("Unique Run ID", value=f"RUN-2027-{state['state_version']:02d}")
            date_input = c_b.text_input(
                "Simulation Date (YYYY-MM-DD)", value=f"2027-{min(12, state['state_version']):02d}-28"
            )
            req_input = c_c.number_input(
                "Requested Workshop Seats", min_value=0, value=int(state["parameters"]["D0"]), step=1
            )

            c_d, c_e, c_f = st.columns(3)
            price_input = c_d.number_input(
                "Seat Price ($)", min_value=0.01, value=float(state["parameters"]["base_price"]), step=1.0
            )
            ad_input = c_e.number_input("Approved Marketing Expense ($)", min_value=0.0, value=0.0, step=10.0)
            loan_input = c_f.number_input("New Simulated Loan Proceeds ($)", min_value=0.0, value=0.0, step=100.0)

            submitted = st.form_submit_button("Approve & Advance Simulation Once", type="primary")
            if submitted:
                try:
                    updated, res = run_simulation_period(
                        state,
                        run_id=run_id_input.strip(),
                        period_date=date_input.strip(),
                        requested=int(req_input),
                        price=price_input,
                        ad_cost=ad_input,
                        loan_cash=loan_input,
                    )
                    if res["status"] == "duplicate_ignored":
                        st.warning(res["message"])
                    else:
                        st.session_state.sim_state = updated
                        st.success(
                            f"Committed {run_id_input} (v{updated['state_version']}): Sold {res['sold']} seats | "
                            f"Revenue {fmt_usd(res['revenue'])} | Net Profit {fmt_usd(res['net_profit'])} | "
                            f"Closing Cash {fmt_usd(res['closing_cash'])}"
                        )
                        st.rerun()
                except ValueError as err:
                    st.error(str(err))

        st.markdown("#### Period Summaries")
        if state["monthly_summaries"]:
            st.dataframe(pd.DataFrame(state["monthly_summaries"]), use_container_width=True)
        else:
            st.info("No simulation periods committed yet. Advance a period above or test in Tab 8.")

        st.markdown("#### Dated Event Ledger")
        if state["ledger"]:
            ledger_df = pd.DataFrame(state["ledger"])
            st.dataframe(ledger_df, use_container_width=True)
            st.download_button(
                "Export Ledger CSV",
                data=ledger_df.to_csv(index=False),
                file_name=f"NUR3_ledger_v{state['state_version']}.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------------------------
    # TAB 5: REPORT CENTER (RECONCILED STATEMENTS)
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Reconciled Report Center")
        st.caption(
            f"Team: {state['team_id']} | Pathway: {state['pathway_id']} | Snapshot: {state['snapshot_id']} (v{state['state_version']}) | "
            "Viewing reports never creates transactions."
        )
        if not state["monthly_summaries"]:
            st.info("Advance at least one simulation period in Tab 4 to populate Income Statement, Cash-Flow, and Operations reports.")
        else:
            df_sum = pd.DataFrame(state["monthly_summaries"])
            tot_rev = sum(money(x) for x in df_sum["revenue"])
            tot_var = sum(money(x) for x in df_sum["variable_expense"])
            tot_fix = sum(money(x) for x in df_sum["fixed_expense"])
            tot_mkt = sum(money(x) for x in df_sum["marketing_expense"])
            tot_net = sum(money(x) for x in df_sum["net_profit"])

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("##### Cumulative Income Statement (Actual Simulated)")
                inc_df = pd.DataFrame(
                    [
                        {"Line Item": "Workshop Sales Revenue", "Amount ($)": f"{tot_rev:.2f}"},
                        {"Line Item": "Variable Cost of Sales", "Amount ($)": f"-{tot_var:.2f}"},
                        {"Line Item": "Gross Margin", "Amount ($)": f"{tot_rev - tot_var:.2f}"},
                        {"Line Item": "Monthly Fixed Operating Costs", "Amount ($)": f"-{tot_fix:.2f}"},
                        {"Line Item": "Own Marketing Expense", "Amount ($)": f"-{tot_mkt:.2f}"},
                        {"Line Item": "Net Operating Result", "Amount ($)": f"{tot_net:.2f}"},
                    ]
                )
                st.dataframe(inc_df, use_container_width=True, hide_index=True)

            with r2:
                st.markdown("##### Simplified Balance Sheet Reconciliation")
                cash_bal = money(state["balances"]["cash"])
                equip_bal = money(state["balances"]["equipment_assets"])
                debt_bal = money(state["balances"]["loan_debt"])
                eq_bal = money(state["balances"]["owner_equity"])
                re_bal = money(state["balances"]["retained_earnings"])
                bs_df = pd.DataFrame(
                    [
                        {"Account": "Spendable Cash", "Balance ($)": f"{cash_bal:.2f}"},
                        {"Account": "Equipment & Accessibility Assets", "Balance ($)": f"{equip_bal:.2f}"},
                        {"Account": "TOTAL ASSETS", "Balance ($)": f"{cash_bal + equip_bal:.2f}"},
                        {"Account": "Loan Debt", "Balance ($)": f"{debt_bal:.2f}"},
                        {"Account": "Owner Equity", "Balance ($)": f"{eq_bal:.2f}"},
                        {"Account": "Retained Earnings (incl. -$500 setup)", "Balance ($)": f"{re_bal:.2f}"},
                        {"Account": "TOTAL LIABILITIES + EQUITY", "Balance ($)": f"{debt_bal + eq_bal + re_bal:.2f}"},
                    ]
                )
                st.dataframe(bs_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Export Period Reports CSV",
                data=df_sum.to_csv(index=False),
                file_name=f"NUR3_reports_v{state['state_version']}.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------------------------
    # TAB 6: COSTED MARKETING STUDIO
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Marketing Studio (Costed Draft & A/B Variant Planner)")
        st.caption("Drafting ads does not automatically create demand or deduct cash until student-approved in a round.")

        mc1, mc2 = st.columns(2)
        with mc1:
            camp_id = st.text_input("Campaign ID", value="CMP-NUR-01")
            offering_choice = st.selectbox(
                "Target Offering",
                [state["primary_offering"]] + state["secondary_offerings"],
            )
            audience = st.text_input("Target Audience", value="Local community adults & caregivers")
            headline = st.text_input("Approved Educational Claim", value="Practical Wellness & Prevention Skills Workshop")
        with mc2:
            d_hrs = st.number_input("Design Labor Hours", min_value=0.0, value=2.0, step=0.5)
            d_rate = st.number_input("Design Labor Rate ($/hr)", min_value=0.0, value=15.0, step=1.0)
            assets_c = st.number_input("Licensed Assets Cost ($)", min_value=0.0, value=10.0, step=5.0)
            print_qty = st.number_input("Print Flyer Quantity", min_value=0, value=100, step=10)
            print_unit = st.number_input("Print Cost per Flyer ($)", min_value=0.0, value=0.30, step=0.05)
            dist_place = st.number_input("Distribution & Placement Cost ($)", min_value=0.0, value=30.0, step=5.0)

        total_camp_cost = money(d_hrs * d_rate + assets_c + print_qty * print_unit + dist_place)
        st.metric("Total Itemized Campaign Cost", fmt_usd(total_camp_cost))

        est_conv = st.number_input("Simulated Attributed Conversions (for CAC preview)", min_value=0, value=0, step=1)
        cac_display = "N/A (0 conversions)" if est_conv == 0 else fmt_usd(total_camp_cost / Decimal(est_conv))
        st.write(f"**Customer Acquisition Cost (CAC):** `{cac_display}`")
        st.text_area(
            "Printable Flyer Copy Preview (Fictional Contact Details)",
            value=(
                f"[{camp_id}] {headline}\n"
                f"Offering: {offering_choice} | Audience: {audience}\n"
                f"Price: {fmt_usd(state['parameters']['base_price'])} per seat\n"
                f"Note: Fictional educational simulation only — not medical advice. Contact: info@lifesavers-sim.example"
            ),
            height=120,
        )

    # --------------------------------------------------------------------------
    # TAB 7: SAVE / LOAD JSON SNAPSHOT
    # --------------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Durable Save / Restore JSON Snapshot")
        st.caption("Download your JSON snapshot after every decision round so you can restore it anytime.")

        json_str = export_snapshot_json(state)
        st.download_button(
            f"Download Snapshot JSON ({state['snapshot_id']} / v{state['state_version']})",
            data=json_str,
            file_name=f"{state['team_id'].replace(' ', '_')}_{state['pathway_id']}_v{state['state_version']}.json",
            mime="application/json",
        )

        uploaded = st.file_uploader("Upload a saved Snapshot JSON file", type=["json"])
        enforce_stale = st.checkbox("Reject stale state_version imports", value=True)
        if uploaded is not None:
            raw_text = uploaded.read().decode("utf-8")
            if st.button("Validate & Import Snapshot"):
                try:
                    restored = import_snapshot_json(
                        raw_text,
                        expected_team_id=state["team_id"],
                        current_state_version=state["state_version"],
                        reject_stale=enforce_stale,
                    )
                    st.session_state.sim_state = restored
                    st.success(f"Restored {restored['snapshot_id']} (v{restored['state_version']}).")
                    st.rerun()
                except ValueError as err:
                    st.error(f"Import Rejected: {err}")

    # --------------------------------------------------------------------------
    # TAB 8: ACCEPTANCE TESTS
    # --------------------------------------------------------------------------
    with tabs[7]:
        st.subheader("NUR-3 Automated Acceptance & Fixture Tests")
        if st.button("Run All Automated Checks", type="primary"):
            checks = []

            _, fix_res = replay_fixture_rows(state, EMBEDDED_MONTHLY_FIXTURE)
            expected_fix = [
                ("2027-01", 60, "1500.00", "1000.00", "500.00", "4500.00"),
                ("2027-02", 60, "1500.00", "1100.00", "400.00", "4900.00"),
                ("2027-03", 45, "1125.00", "925.00", "200.00", "5100.00"),
            ]
            for actual, (m, exp_sold, exp_rev, exp_exp, exp_prof, exp_cash) in zip(fix_res, expected_fix):
                passed = (
                    actual["sold"] == exp_sold
                    and actual["revenue"] == exp_rev
                    and actual["total_operating_expense"] == exp_exp
                    and actual["operating_profit"] == exp_prof
                    and actual["closing_cash"] == exp_cash
                )
                checks.append(
                    {
                        "Test": f"Fixture {m}",
                        "Expected": f"Sold={exp_sold}, Rev=${exp_rev}, Exp=${exp_exp}, Prof=${exp_prof}, Cash=${exp_cash}",
                        "Actual": f"Sold={actual['sold']}, Rev=${actual['revenue']}, Exp=${actual['total_operating_expense']}, Prof=${actual['operating_profit']}, Cash=${actual['closing_cash']}",
                        "Status": "PASS" if passed else "FAIL",
                    }
                )

            ns_state = default_nur3_state()
            ns_state["parameters"]["no_show_rate"] = 0.20
            _, ns_res = run_simulation_period(ns_state, run_id="TEST-NOSHOW", period_date="2027-01-28", requested=80)
            ns_pass = (
                ns_res["booked"] == 60
                and ns_res["sold"] == 48
                and ns_res["revenue"] == "1200.00"
                and ns_res["variable_expense"] == "240.00"
                and ns_res["operating_profit"] == "260.00"
            )
            checks.append(
                {
                    "Test": "No-Show Rate 20%",
                    "Expected": "Booked=60, Attended=48, Rev=$1200.00, Var=$240.00, Prof=$260.00",
                    "Actual": f"Booked={ns_res['booked']}, Attended={ns_res['sold']}, Rev=${ns_res['revenue']}, Var=${ns_res['variable_expense']}, Prof=${ns_res['operating_profit']}",
                    "Status": "PASS" if ns_pass else "FAIL",
                }
            )

            z_state = default_nur3_state()
            _, z_res = run_simulation_period(z_state, run_id="TEST-ZERO", period_date="2027-01-28", requested=0)
            z_pass = (
                z_res["revenue"] == "0.00"
                and z_res["variable_expense"] == "0.00"
                and z_res["operating_profit"] == "-700.00"
            )
            checks.append(
                {
                    "Test": "Zero Sales",
                    "Expected": "Rev=$0.00, Var=$0.00, Prof=-$700.00",
                    "Actual": f"Rev=${z_res['revenue']}, Var=${z_res['variable_expense']}, Prof=${z_res['operating_profit']}",
                    "Status": "PASS" if z_pass else "FAIL",
                }
            )

            st.dataframe(pd.DataFrame(checks), use_container_width=True)


if __name__ == "__main__":
    main()
