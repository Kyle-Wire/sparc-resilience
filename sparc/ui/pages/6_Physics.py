"""Page 6 — Physics Constraints (Priors, Caps, Calibration)."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


st.title("Physics Constraints")
st.markdown(
    "Define literature-based priors, physical bounds, delta limits, and calibration "
    "settings that guardrail the scenario simulations."
)

actionable = st.session_state.get("actionable_variables", [])
predictors = st.session_state.get("predictors", [])
variables = actionable if actionable else predictors

if not variables:
    st.info("Select predictors on the **Variables** page first.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Literature Priors
# ══════════════════════════════════════════════════════════════════════════════
tab_priors, tab_caps, tab_cal, tab_val = st.tabs([
    "Literature Priors", "Bounds & Caps", "Calibration", "Validation Rules"
])

with tab_priors:
    st.markdown("## Literature Coefficients")
    st.markdown(
        "Enter the expected effect size of each variable on the target, "
        "based on published literature. These anchor the scenario predictions."
    )

    priors = dict(st.session_state.get("priors", {}))

    for var in variables:
        with st.expander(f"**{var}**", expanded=var in priors):
            existing = priors.get(var, {})
            c1, c2, c3 = st.columns(3)
            value = c1.number_input(
                "Coefficient", value=float(existing.get("value", 0.0)),
                format="%.4f", key=f"prior_val_{var}",
            )
            units = c2.text_input(
                "Units", value=existing.get("units", ""),
                key=f"prior_units_{var}",
                help="e.g. '°F per +10 pp' or '°F per +0.1'",
            )
            uncertainty = c3.number_input(
                "Uncertainty (±)", value=float(existing.get("uncertainty", 0.20)),
                min_value=0.0, max_value=1.0, step=0.05,
                key=f"prior_unc_{var}",
            )
            source = st.text_input(
                "Literature Source",
                value=existing.get("literature_source", existing.get("source", "")),
                key=f"prior_src_{var}",
            )
            confidence = st.selectbox(
                "Confidence",
                ["high", "medium", "low"],
                index=["high", "medium", "low"].index(existing.get("confidence", "medium")),
                key=f"prior_conf_{var}",
            )
            desc = st.text_input(
                "Description",
                value=existing.get("description", ""),
                key=f"prior_desc_{var}",
            )

            if value != 0.0 or units:
                priors[var] = {
                    "value": value,
                    "units": units,
                    "uncertainty": uncertainty,
                    "literature_source": source,
                    "confidence": confidence,
                    "description": desc,
                }

    st.session_state["priors"] = priors

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Bounds & Caps
# ══════════════════════════════════════════════════════════════════════════════
with tab_caps:
    st.markdown("## Variable Bounds")

    bounds = dict(st.session_state.get("caps_variable_bounds", {}))
    deltas = dict(st.session_state.get("caps_delta_constraints", {}))
    dim_ret = dict(st.session_state.get("caps_diminishing_return_thresholds", {}))

    for var in variables:
        with st.expander(f"**{var}**", expanded=var in bounds):
            # Bounds
            st.markdown("**Physical bounds**")
            b = bounds.get(var, {})
            c1, c2 = st.columns(2)
            bmin = c1.number_input("Min", value=float(b.get("min", 0.0)), key=f"bmin_{var}")
            bmax = c2.number_input("Max", value=float(b.get("max", 100.0)), key=f"bmax_{var}")
            bunits = st.text_input("Units", value=b.get("units", ""), key=f"bunits_{var}")

            if bmin != 0.0 or bmax != 100.0 or bunits:
                bounds[var] = {"min": bmin, "max": bmax, "units": bunits}

            # Delta constraints
            st.markdown("**Delta constraints (max intervention size)**")
            d = deltas.get(var, {})
            c3, c4 = st.columns(2)
            dmax_inc = c3.number_input(
                "Max increase", value=float(d.get("max_increase", 0.0)),
                min_value=0.0, key=f"dinc_{var}",
            )
            dmax_dec = c4.number_input(
                "Max decrease", value=float(d.get("max_decrease", 0.0)),
                min_value=0.0, key=f"ddec_{var}",
            )
            if dmax_inc > 0 or dmax_dec > 0:
                deltas[var] = {"max_increase": dmax_inc, "max_decrease": dmax_dec}

            # Diminishing returns
            dr_val = dim_ret.get(var, 10.0)
            dim_ret[var] = st.number_input(
                "Diminishing return threshold",
                value=float(dr_val), min_value=0.0, key=f"dr_{var}",
                help="Effects taper (sqrt decay) beyond this delta.",
            )

    st.session_state["caps_variable_bounds"] = bounds
    st.session_state["caps_delta_constraints"] = deltas
    st.session_state["caps_diminishing_return_thresholds"] = dim_ret

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Calibration
# ══════════════════════════════════════════════════════════════════════════════
with tab_cal:
    st.markdown("## Physics–Data Calibration")
    st.markdown(
        "Control how much the model trusts literature priors vs. data-estimated coefficients."
    )

    st.session_state["calibration_shrinkage_weight"] = st.slider(
        "Shrinkage weight (physics vs data)",
        min_value=0.0, max_value=1.0, step=0.05,
        value=float(st.session_state["calibration_shrinkage_weight"]),
        help="1.0 = pure physics priors, 0.0 = pure data. Default 0.7.",
    )

    st.session_state["calibration_max_deviation"] = st.number_input(
        "Max deviation from physics",
        min_value=0.0, max_value=2.0, step=0.1,
        value=float(st.session_state["calibration_max_deviation"]),
        help="Maximum allowed deviation from the literature prior.",
    )

    st.session_state["calibration_adaptive_weighting"] = st.checkbox(
        "Adaptive weighting",
        value=st.session_state["calibration_adaptive_weighting"],
        help="Automatically adjust shrinkage based on data quality.",
    )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Validation Rules
# ══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.markdown("## Validation Thresholds")

    validation = dict(st.session_state.get("caps_validation", {}))

    validation["max_single_cell_delta"] = st.number_input(
        "Max single-cell delta",
        value=float(validation.get("max_single_cell_delta", 10.0)),
        min_value=0.0, step=1.0,
    )
    validation["typical_max_delta"] = st.number_input(
        "Typical max delta",
        value=float(validation.get("typical_max_delta", 5.0)),
        min_value=0.0, step=0.5,
    )
    validation["max_zone_avg_delta"] = st.number_input(
        "Max zone average delta",
        value=float(validation.get("max_zone_avg_delta", 4.0)),
        min_value=0.0, step=0.5,
    )
    validation["sign_violation_action"] = st.selectbox(
        "Sign violation action",
        ["error", "warning", "dampen"],
        index=["error", "warning", "dampen"].index(
            validation.get("sign_violation_action", "error")
        ),
    )
    validation["magnitude_violation_action"] = st.selectbox(
        "Magnitude violation action",
        ["error", "warning"],
        index=["error", "warning"].index(
            validation.get("magnitude_violation_action", "warning")
        ),
    )

    st.session_state["caps_validation"] = validation
