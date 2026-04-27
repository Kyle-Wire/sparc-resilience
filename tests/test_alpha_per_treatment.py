"""Tests for ProcessRateNet per-treatment alpha-field output (Phase 3, v4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sparc.models.process_rate_net import ProcessRateNet


@pytest.fixture()
def domain_config():
    return {
        "name": "uhi",
        "bounds": [1e-3, 1.0],
        "prior_mean": 0.05,
        "material_priors": {"impervious": 0.10, "tree": 0.02, "water": 0.005},
        "units": "m2/s",
    }


def test_default_single_treatment_output_shape(domain_config):
    """n_treatments=1 (default) must keep legacy (N, 1) output shape."""
    net = ProcessRateNet(n_inputs=4, domain_config=domain_config)
    x = torch.randn(8, 4)
    out = net(x)
    assert out.shape == (8, 1)
    # Bounds enforced
    assert (out >= domain_config["bounds"][0]).all()
    assert (out <= domain_config["bounds"][1]).all()
    # ``.squeeze(-1)`` still yields (N,) for legacy callers.
    assert out.squeeze(-1).shape == (8,)


def test_multi_treatment_output_shape(domain_config):
    net = ProcessRateNet(n_inputs=4, domain_config=domain_config, n_treatments=3)
    x = torch.randn(10, 4)
    out = net(x)
    assert out.shape == (10, 3)
    assert (out >= domain_config["bounds"][0]).all()
    assert (out <= domain_config["bounds"][1]).all()


def test_multi_treatment_heads_independent(domain_config):
    """Different heads must produce different outputs after a small training nudge."""
    torch.manual_seed(0)
    net = ProcessRateNet(n_inputs=4, domain_config=domain_config, n_treatments=2)
    x = torch.randn(32, 4)
    # Train head 0 toward higher rate, head 1 toward lower rate.
    target = torch.stack([
        torch.full((32,), 0.5),  # head 0 → 0.5
        torch.full((32,), 0.01),  # head 1 → 0.01
    ], dim=-1)
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        pred = net(x)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    pred = net(x).detach()
    assert pred[:, 0].mean() > pred[:, 1].mean(), (
        f"Heads collapsed: h0={pred[:, 0].mean():.4f}, h1={pred[:, 1].mean():.4f}"
    )


def test_compute_mixture_prior_shape(domain_config):
    net1 = ProcessRateNet(n_inputs=3, domain_config=domain_config)
    net2 = ProcessRateNet(n_inputs=3, domain_config=domain_config, n_treatments=4)
    lc = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.3, 0.2],
    ])
    p1 = net1.compute_mixture_prior(lc)
    p4 = net2.compute_mixture_prior(lc)
    assert p1.shape == (3, 1)
    assert p4.shape == (3, 4)
    # Per-row across treatments must be identical (broadcast).
    assert torch.allclose(p4[:, 0:1], p4[:, 1:2])
    assert torch.allclose(p4[:, 0:1], p1)


def test_network_shim_single_treatment(domain_config):
    """The legacy ``self.network`` accessor must still expose final layer for n_treatments=1."""
    net = ProcessRateNet(n_inputs=2, domain_config=domain_config)
    seq = net.network
    assert seq[-1].out_features == 1


def test_network_shim_raises_for_multi(domain_config):
    net = ProcessRateNet(n_inputs=2, domain_config=domain_config, n_treatments=2)
    with pytest.raises(AttributeError):
        _ = net.network


def test_legacy_constructor_signature(domain_config):
    """Old call site (no n_treatments kwarg) must continue to work unchanged."""
    net = ProcessRateNet(n_inputs=5, domain_config=domain_config, n_temporal_inputs=2)
    assert net.n_treatments == 1
    out = net(torch.randn(4, 7))
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Phase 3 step 2: treatment-list resolution from config / DAG
# ---------------------------------------------------------------------------


def test_resolve_treatments_from_explicit_config():
    from sparc.run.v2_neural_training import _resolve_treatment_list
    cfg = {"causal": {"treatments": ["impervious", "tree", "albedo"]}}
    assert _resolve_treatment_list(cfg) == ["impervious", "tree", "albedo"]


def test_resolve_treatments_from_dag(tmp_path):
    import json
    from sparc.run.v2_neural_training import _resolve_treatment_list
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(json.dumps({
        "nodes": [
            {"name": "impervious", "type": "treatment"},
            {"name": "tree", "type": "treatment"},
            {"name": "lst", "type": "outcome"},
            {"name": "humidity", "type": "confounder"},
        ],
        "edges": [
            {"parent": "impervious", "child": "lst"},
            {"parent": "tree", "child": "lst"},
            {"parent": "humidity", "child": "lst"},
        ],
    }))
    cfg = {"causal": {"dag_file": str(dag_file)}}
    treatments = _resolve_treatment_list(cfg)
    assert set(treatments) == {"impervious", "tree"}


def test_resolve_treatments_empty_when_no_dag():
    from sparc.run.v2_neural_training import _resolve_treatment_list
    assert _resolve_treatment_list({}) == []
    assert _resolve_treatment_list({"causal": {}}) == []


# ---------------------------------------------------------------------------
# Phase 3 step 3: persistence — (N, n_treatments) blob with metadata
# ---------------------------------------------------------------------------


def test_alpha_field_db_roundtrip_multi_head(tmp_path):
    """End-to-end: persist (N, n_treatments) via ArtifactStore, read back, verify metadata."""
    from sparc.registry.run_registry import RunRegistry
    from sparc.registry.store import ArtifactStore

    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)

    n_cells = 64
    treatments = ["impervious", "tree"]
    alpha_full = np.random.RandomState(0).uniform(0.01, 0.5, size=(n_cells, 2))

    store.write_blob(
        "2", "v2_alpha_field", alpha_full,
        serializer="pickle",
        producer="v2_neural_training",
        consumers=["scenario_simulator", "causal_effect_resolver"],
        metadata={
            "schema_version": 2,
            "shape": list(alpha_full.shape),
            "treatments": treatments,
            "n_treatments": 2,
            "collapse": "mean",
        },
    )

    loaded = store.read_any("2", "v2_alpha_field")
    assert loaded.shape == (n_cells, 2)
    assert np.allclose(loaded, alpha_full)


def test_alpha_field_back_compat_legacy_1d(tmp_path):
    """Legacy (N,) alpha blobs from pre-Phase-3 runs must still round-trip."""
    from sparc.registry.run_registry import RunRegistry
    from sparc.registry.store import ArtifactStore

    reg = RunRegistry(tmp_path)
    store = ArtifactStore(reg)
    legacy = np.linspace(0.01, 0.5, 32).astype(np.float32)
    store.write_blob(
        "2", "v2_alpha_field", legacy,
        serializer="pickle", producer="v2_neural_training",
    )
    loaded = store.read_any("2", "v2_alpha_field")
    assert loaded.ndim == 1
    assert loaded.shape == (32,)
