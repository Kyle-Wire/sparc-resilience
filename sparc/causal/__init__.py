"""
causal — Causal inference module for SPARC
==========================================

Provides a formal Structural Causal Model (SCM) layer on top of the SPARC
spatial-regression pipeline.  Built around Microsoft DoWhy for identifiability
analysis and EconML for heterogeneous treatment-effect estimation.

Sub-modules
-----------
dag_definition
    Load, validate, and convert a user-supplied DAG (YAML / DAGitty JSON)
    into a ``networkx.DiGraph`` and DoWhy ``CausalModel``.

dag_validator
    Automated identifiability checks (backdoor / front-door criteria),
    instrument search, positivity / SUTVA diagnostics, and testable-
    implication reports.

counterfactual_engine
    Given trained SPARC models, a DAG, and an intervention specification,
    propagate causal effects through the graph and produce spatially-varying
    counterfactual predictions with uncertainty bounds.

causal_discovery
    Data-driven DAG validation via PC-Stable, DirectLiNGAM, and GES
    algorithms (causal-learn).  Compares learned structure against the
    expert DAG using SHD, precision/recall, and d-separation tests.

spatial_cate
    Geographically-weighted Conditional Average Treatment Effects via
    EconML's CausalForestDML with spatial coordinates as effect modifiers.
"""

__all__ = [
    "dag_definition",
    "dag_validator",
    "counterfactual_engine",
    "causal_discovery",
    "spatial_cate",
    # V2 Bayesian
    "mc3",
    "nuts",
]
