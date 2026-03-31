"""Page 5 — DAG Builder."""
import streamlit as st
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sparc.ui.theme import SPECTRAL, get_matplotlib_style

st.title("Causal DAG Builder")
st.markdown(
    "Define the **Directed Acyclic Graph** that encodes your causal assumptions. "
    "Nodes are variables; edges represent causal relationships."
)

# ── helpers ───────────────────────────────────────────────────────────────────
NODE_TYPES = ["treatment", "mediator", "confounder", "outcome"]
TYPE_COLORS = {
    "treatment":  SPECTRAL[0],   # purple
    "mediator":   SPECTRAL[2],   # pink
    "confounder": SPECTRAL[4],   # orange
    "outcome":    SPECTRAL[5],   # yellow
}

predictors = st.session_state.get("predictors", [])
target     = st.session_state.get("target_column", "")
all_vars   = list(dict.fromkeys(predictors + ([target] if target else [])))

# ── Nodes ─────────────────────────────────────────────────────────────────────
st.markdown("## Nodes")

nodes: list[dict] = list(st.session_state.get("dag_nodes", []))
node_names = [n["name"] for n in nodes]

# Quick-add from predictors
missing = [v for v in all_vars if v not in node_names]
if missing:
    with st.expander("Quick-add variables as nodes"):
        to_add = st.multiselect("Select variables to add", missing)
        if st.button("Add selected as nodes"):
            for var in to_add:
                ntype = "outcome" if var == target else "treatment"
                nodes.append({"name": var, "type": ntype, "description": ""})
            st.session_state["dag_nodes"] = nodes
            st.rerun()

# Editable node table
if nodes:
    st.markdown("Edit node types and descriptions:")
    updated_nodes = []
    for i, node in enumerate(nodes):
        cols = st.columns([2, 2, 4, 1])
        with cols[0]:
            name = st.text_input("Name", value=node["name"], key=f"nname_{i}", disabled=True)
        with cols[1]:
            ntype = st.selectbox(
                "Type", NODE_TYPES,
                index=NODE_TYPES.index(node.get("type", "treatment")),
                key=f"ntype_{i}",
            )
        with cols[2]:
            desc = st.text_input("Description", value=node.get("description", ""), key=f"ndesc_{i}")
        with cols[3]:
            remove = st.button("x", key=f"nrem_{i}")
        if not remove:
            updated_nodes.append({"name": name, "type": ntype, "description": desc})
    st.session_state["dag_nodes"] = updated_nodes
    nodes = updated_nodes
    node_names = [n["name"] for n in nodes]
else:
    st.info("No nodes yet. Use the quick-add above or add variables on the Variables page first.")

# Manual add
with st.expander("Add a custom node"):
    c1, c2 = st.columns(2)
    new_name = c1.text_input("Node name", key="new_node_name")
    new_type = c2.selectbox("Node type", NODE_TYPES, key="new_node_type")
    new_desc = st.text_input("Description", key="new_node_desc")
    if st.button("Add node") and new_name and new_name not in node_names:
        nodes.append({"name": new_name, "type": new_type, "description": new_desc})
        st.session_state["dag_nodes"] = nodes
        st.rerun()

# ── Edges ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## Edges")

edges: list[dict] = list(st.session_state.get("dag_edges", []))

if node_names:
    # Edge editor
    if edges:
        updated_edges = []
        for i, edge in enumerate(edges):
            cols = st.columns([2, 2, 4, 1])
            with cols[0]:
                parent = st.selectbox(
                    "Parent", node_names,
                    index=node_names.index(edge["parent"]) if edge["parent"] in node_names else 0,
                    key=f"eparent_{i}",
                )
            with cols[1]:
                child = st.selectbox(
                    "Child", node_names,
                    index=node_names.index(edge["child"]) if edge["child"] in node_names else 0,
                    key=f"echild_{i}",
                )
            with cols[2]:
                mech = st.text_input("Mechanism", value=edge.get("mechanism", ""), key=f"emech_{i}")
            with cols[3]:
                remove = st.button("x", key=f"erem_{i}")
            if not remove:
                updated_edges.append({"parent": parent, "child": child, "mechanism": mech})
        st.session_state["dag_edges"] = updated_edges
        edges = updated_edges

    # Add new edge
    with st.expander("Add a new edge"):
        c1, c2 = st.columns(2)
        ep = c1.selectbox("Parent", node_names, key="new_edge_parent")
        ec = c2.selectbox("Child", node_names, key="new_edge_child")
        em = st.text_input("Mechanism / explanation", key="new_edge_mech")
        if st.button("Add edge") and ep and ec and ep != ec:
            edges.append({"parent": ep, "child": ec, "mechanism": em})
            st.session_state["dag_edges"] = edges
            st.rerun()
else:
    st.info("Add nodes above before defining edges.")

# ── Causal settings ──────────────────────────────────────────────────────────
st.divider()
st.markdown("## Causal Estimation Settings")
col_a, col_b = st.columns(2)
with col_a:
    est_options = ["hgb", "ols", "dml"]
    st.session_state["causal_estimator"] = st.selectbox(
        "Estimator",
        est_options,
        index=est_options.index(st.session_state.get("causal_estimator", "hgb")),
        help="**hgb** = non-linear monotone-constrained, **ols** = linear, **dml** = doubly-robust ML",
    )
with col_b:
    st.session_state["causal_estimate_cate"] = st.checkbox(
        "Estimate CATE (heterogeneous treatment effects)",
        value=st.session_state.get("causal_estimate_cate", False),
    )

# ── DAG Visualization ────────────────────────────────────────────────────────
st.divider()
st.markdown("## DAG Preview")

if nodes and edges:
    import networkx as nx
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(get_matplotlib_style())

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["name"], ntype=n["type"])
    for e in edges:
        if e["parent"] in G and e["child"] in G:
            G.add_edge(e["parent"], e["child"])

    # Cycle detection
    cycles = list(nx.simple_cycles(G))
    if cycles:
        st.error(f"**Cycle detected!** A DAG cannot have cycles: {cycles}")

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2)

    node_colors = [TYPE_COLORS.get(G.nodes[n].get("ntype", "treatment"), SPECTRAL[0]) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="medium")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555", arrows=True,
                           arrowsize=20, connectionstyle="arc3,rad=0.1", width=1.5)
    ax.set_title("Causal DAG", fontsize=14)
    ax.axis("off")

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=c, label=t.title()) for t, c in TYPE_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    st.pyplot(fig)
    plt.close(fig)
elif nodes:
    st.info("Add edges to see the DAG visualization.")
else:
    st.info("Add nodes and edges to see the DAG visualization.")
