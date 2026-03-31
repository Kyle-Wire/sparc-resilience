"""Page 7 — Scenario Designer."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Scenarios")
st.markdown(
    "Define the *what-if* intervention scenarios you want to simulate. "
    "Single-variable scenarios test one change at a time; joint scenarios combine multiple."
)

actionable = st.session_state.get("actionable_variables", [])
predictors = st.session_state.get("predictors", [])
variables = actionable if actionable else predictors

if not variables:
    st.info("Define actionable variables on the **Variables** page first.")
    st.stop()


def _parse_increments(text: str) -> list[float]:
    """Parse a comma-separated string into a list of floats."""
    if not text:
        return []
    parts = []
    for part in text.split(","):
        part = part.strip()
        if part:
            try:
                parts.append(float(part))
            except ValueError:
                pass
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# Single-Variable Scenarios
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Single-Variable Scenarios")

scenarios: list[dict] = list(st.session_state.get("scenarios", []))

# Display existing
for i, sc in enumerate(scenarios):
    with st.expander(f"**{sc.get('name', f'Scenario {i+1}')}**", expanded=False):
        c1, c2, c3 = st.columns(3)
        sc["name"] = c1.text_input("Name", value=sc.get("name", ""), key=f"sc_name_{i}")
        sc["variable"] = c2.selectbox(
            "Variable", variables,
            index=variables.index(sc["variable"]) if sc.get("variable") in variables else 0,
            key=f"sc_var_{i}",
        )
        sc["direction"] = c3.selectbox(
            "Direction", ["increase", "decrease"],
            index=["increase", "decrease"].index(sc.get("direction", "increase")),
            key=f"sc_dir_{i}",
        )

        inc_str = st.text_input(
            "Increments (comma-separated)",
            value=", ".join(str(x) for x in sc.get("increments", [])),
            key=f"sc_inc_{i}",
        )
        sc["increments"] = _parse_increments(inc_str)

        c4, c5 = st.columns(2)
        sc["min_val"] = c4.number_input("Min", value=float(sc.get("min_val", 0)), key=f"sc_min_{i}")
        sc["max_val"] = c5.number_input("Max", value=float(sc.get("max_val", 100)), key=f"sc_max_{i}")
        sc["unit"] = st.text_input("Unit", value=sc.get("unit", ""), key=f"sc_unit_{i}")

        if st.button("Remove", key=f"sc_rem_{i}"):
            scenarios.pop(i)
            st.session_state["scenarios"] = scenarios
            st.rerun()

# Add new
with st.expander("+ Add new scenario"):
    c1, c2, c3 = st.columns(3)
    new_name = c1.text_input("Name", key="new_sc_name")
    new_var  = c2.selectbox("Variable", variables, key="new_sc_var")
    new_dir  = c3.selectbox("Direction", ["increase", "decrease"], key="new_sc_dir")
    new_inc  = st.text_input("Increments (comma-separated)", key="new_sc_inc",
                             help="e.g. 5, 10, 15, 20, 25")
    c4, c5 = st.columns(2)
    new_min = c4.number_input("Min", value=0.0, key="new_sc_min")
    new_max = c5.number_input("Max", value=100.0, key="new_sc_max")
    new_unit = st.text_input("Unit", key="new_sc_unit", value="percentage points")

    if st.button("Add scenario") and new_name:
        scenarios.append({
            "name": new_name,
            "variable": new_var,
            "direction": new_dir,
            "increments": _parse_increments(new_inc),
            "min_val": new_min,
            "max_val": new_max,
            "unit": new_unit,
        })
        st.session_state["scenarios"] = scenarios
        st.rerun()

st.session_state["scenarios"] = scenarios

# ══════════════════════════════════════════════════════════════════════════════
# Joint Scenarios
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## Joint Scenarios")
st.markdown("Combine multiple variable interventions into a single scenario.")

joint: list[dict] = list(st.session_state.get("joint_scenarios", []))

for i, js in enumerate(joint):
    with st.expander(f"**{js.get('name', f'Joint {i+1}')}**", expanded=False):
        js["name"] = st.text_input("Name", value=js.get("name", ""), key=f"js_name_{i}")
        js["auto_propagate_dag"] = st.checkbox(
            "Auto-propagate through DAG",
            value=js.get("auto_propagate_dag", True),
            key=f"js_prop_{i}",
        )

        interventions = js.get("interventions", [])
        updated_int = []
        for j, intv in enumerate(interventions):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            intv_var = c1.selectbox(
                "Variable", variables,
                index=variables.index(intv["variable"]) if intv.get("variable") in variables else 0,
                key=f"ji_var_{i}_{j}",
            )
            intv_dir = c2.selectbox(
                "Direction", ["increase", "decrease"],
                index=["increase", "decrease"].index(intv.get("direction", "increase")),
                key=f"ji_dir_{i}_{j}",
            )
            intv_inc = c3.number_input(
                "Increment", value=float(intv.get("increment", 0)),
                key=f"ji_inc_{i}_{j}",
            )
            if not c4.button("x", key=f"ji_rem_{i}_{j}"):
                updated_int.append({"variable": intv_var, "direction": intv_dir, "increment": intv_inc})
        js["interventions"] = updated_int

        # Add intervention
        if st.button("+ Add intervention", key=f"js_addint_{i}"):
            js["interventions"].append({"variable": variables[0], "direction": "increase", "increment": 10})
            st.rerun()

        if st.button("Remove joint scenario", key=f"js_rem_{i}"):
            joint.pop(i)
            st.session_state["joint_scenarios"] = joint
            st.rerun()

# Add new joint scenario
with st.expander("+ Add new joint scenario"):
    jn = st.text_input("Name", key="new_js_name")
    jp = st.checkbox("Auto-propagate through DAG", value=True, key="new_js_prop")
    if st.button("Add joint scenario") and jn:
        joint.append({
            "name": jn,
            "auto_propagate_dag": jp,
            "interventions": [],
        })
        st.session_state["joint_scenarios"] = joint
        st.rerun()

st.session_state["joint_scenarios"] = joint

# ── Summary ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## Summary")
col1, col2 = st.columns(2)
col1.metric("Single-variable scenarios", len(scenarios))
col2.metric("Joint scenarios", len(joint))

if scenarios:
    import pandas as pd
    summary_data = [
        {"Name": s["name"], "Variable": s["variable"],
         "Direction": s["direction"], "Steps": len(s.get("increments", []))}
        for s in scenarios
    ]
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
