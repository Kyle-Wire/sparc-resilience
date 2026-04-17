"""
Integration tests for all SPARC V2 modules using brown4.csv example data.
"""

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures — load brown4.csv once, subsample for speed
# ---------------------------------------------------------------------------

DATA_PATH = "examples/brown4.csv"
FEATURES = ["Pct_Canopy", "Pct_Impervious", "NDVI", "Albedo", "Elevation_m", "Distance_from_water_m"]
TARGET = "AAT_z"
COORD_COLS = ["POINT_X", "POINT_Y"]
N_SUBSAMPLE = 500  # keep tests fast


@pytest.fixture(scope="module")
def brown_data():
    df = pd.read_csv(DATA_PATH)
    # Deterministic subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=N_SUBSAMPLE, replace=False)
    return df.iloc[idx].reset_index(drop=True)


@pytest.fixture(scope="module")
def arrays(brown_data):
    """Return numpy arrays ready for model consumption."""
    X = brown_data[FEATURES].values.astype(np.float32)
    y = brown_data[TARGET].values.astype(np.float32)
    coords = brown_data[COORD_COLS].values.astype(np.float32)
    return X, y, coords


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


# ===================================================================
# 1. Sinusoidal Encoding
# ===================================================================

class TestSinusoidalEncoding:
    def test_output_shape(self, arrays, device):
        from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding

        _, _, coords = arrays
        encoder = SinusoidalSpatialEncoding(n_frequencies=32)
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
        out = encoder(coords_t)

        expected_dim = encoder.output_dim  # 4 * (n_frequencies // 2)
        assert out.shape == (N_SUBSAMPLE, expected_dim), f"Expected (500, {expected_dim}), got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in sinusoidal encoding output"
        assert not torch.isinf(out).any(), "Inf in sinusoidal encoding output"

    def test_different_coords_differ(self, device):
        from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding

        encoder = SinusoidalSpatialEncoding(n_frequencies=16)
        c1 = torch.tensor([[100.0, 200.0]], device=device)
        c2 = torch.tensor([[300.0, 400.0]], device=device)
        e1 = encoder(c1)
        e2 = encoder(c2)
        assert not torch.allclose(e1, e2), "Different coords should produce different encodings"


# ===================================================================
# 2. Spatial Attention (SIREN + KNN)
# ===================================================================

class TestSpatialAttention:
    def test_siren_layer(self, device):
        from sparc.models.spatial_attention import SIRENLayer

        layer = SIRENLayer(6, 64, omega=30.0, is_first=True).to(device)
        x = torch.randn(N_SUBSAMPLE, 6, device=device)
        out = layer(x)
        assert out.shape == (N_SUBSAMPLE, 64)
        assert not torch.isnan(out).any(), "NaN in SIREN output"

    def test_sparse_spatial_attention(self, arrays, device):
        from sparc.models.spatial_attention import SparseSpatialAttention

        _, _, coords = arrays
        d_model = 32
        max_neighbors = 16

        attn = SparseSpatialAttention(d_model=d_model, n_heads=2, max_neighbors=max_neighbors).to(device)

        # Build KNN index
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        _, knn_idx = tree.query(coords, k=max_neighbors + 1)
        knn_idx = knn_idx[:, 1:]  # exclude self

        X = torch.randn(N_SUBSAMPLE, d_model, device=device)
        coords_t = torch.tensor(coords, device=device)
        knn_t = torch.tensor(knn_idx, dtype=torch.long, device=device)

        out, weights = attn(X, coords_t, knn_t)
        assert out.shape == (N_SUBSAMPLE, d_model), f"Expected ({N_SUBSAMPLE}, {d_model}), got {out.shape}"
        assert weights.shape[0] == N_SUBSAMPLE
        assert not torch.isnan(out).any(), "NaN in attention output"


# ===================================================================
# 3. Surrogates
# ===================================================================

class TestSurrogates:
    def test_differentiable_gwr(self, arrays, device):
        from sparc.models.surrogates import DifferentiableGWR

        X, y, coords = arrays
        n_vars = X.shape[1]
        model = DifferentiableGWR(n_vars=n_vars, n_spatial_features=2, hidden_dim=32).to(device)

        X_t = torch.tensor(X, device=device)
        coords_t = torch.tensor(coords, device=device)
        pred, beta = model(X_t, coords_t)

        assert pred.shape == (N_SUBSAMPLE,), f"Expected ({N_SUBSAMPLE},), got {pred.shape}"
        assert beta.shape == (N_SUBSAMPLE, n_vars), f"Beta shape: {beta.shape}"
        assert not torch.isnan(pred).any()

    def test_differentiable_gwrf(self, arrays, device):
        from sparc.models.surrogates import DifferentiableGWRF

        X, _, coords = arrays
        model = DifferentiableGWRF(n_vars=X.shape[1], n_spatial_features=2, hidden_dim=32).to(device)

        X_t = torch.tensor(X, device=device)
        coords_t = torch.tensor(coords, device=device)
        pred = model(X_t, coords_t)

        assert pred.shape == (N_SUBSAMPLE,)
        assert not torch.isnan(pred).any()

    def test_differentiable_ggpgam(self, arrays, device):
        from sparc.models.surrogates import DifferentiableGGPGAM

        X, _, coords = arrays
        n_spatial = 128  # sinusoidal encoding dimension
        model = DifferentiableGGPGAM(
            n_vars=X.shape[1], n_spatial_features=n_spatial, hidden_dim=32,
        ).to(device)

        X_t = torch.tensor(X, device=device)
        spatial_t = torch.randn(X.shape[0], n_spatial, device=device)
        pred = model(X_t, spatial_t)

        assert pred.shape == (N_SUBSAMPLE,)
        assert not torch.isnan(pred).any()

    def test_gradient_flows(self, arrays, device):
        """Verify surrogates are fully differentiable (gradients flow back)."""
        from sparc.models.surrogates import DifferentiableGWR

        X, y, coords = arrays
        model = DifferentiableGWR(n_vars=X.shape[1], n_spatial_features=2, hidden_dim=32).to(device)

        X_t = torch.tensor(X, device=device)
        coords_t = torch.tensor(coords, device=device)
        y_t = torch.tensor(y, device=device)

        pred, _ = model(X_t, coords_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowed through DifferentiableGWR"


# ===================================================================
# 4. Process Rate Network
# ===================================================================

class TestProcessRateNet:
    def test_output_in_bounds(self, arrays, device):
        from sparc.models.process_rate_net import ProcessRateNet

        X, _, _ = arrays
        net = ProcessRateNet(
            n_inputs=3,
            domain_config={
                "name": "thermal_admittance",
                "units": "W·m⁻²·K⁻¹",
                "bounds": [0.5, 12.0],
                "prior_mean": 4.0,
            },
        ).to(device)

        inp = torch.tensor(X[:, :3], device=device)
        alpha = net(inp)

        assert alpha.shape == (N_SUBSAMPLE, 1), f"Expected ({N_SUBSAMPLE}, 1), got {alpha.shape}"
        assert (alpha >= 0.5).all(), f"Alpha below lower bound: min={alpha.min().item()}"
        assert (alpha <= 12.0).all(), f"Alpha above upper bound: max={alpha.max().item()}"

    def test_mixture_prior(self, arrays, device):
        from sparc.models.process_rate_net import ProcessRateNet

        X, _, _ = arrays
        material_table = {"canopy": 0.3, "impervious": 0.7, "vegetation": 0.5}
        net = ProcessRateNet(
            n_inputs=3,
            domain_config={
                "name": "rate",
                "units": "",
                "bounds": [0.01, 5.0],
                "prior_mean": 1.0,
                "material_priors": material_table,
            },
        ).to(device)

        # land_cover fractions: (N, n_materials)
        land_cover = torch.tensor(X[:, :3], device=device).abs()
        land_cover = land_cover / land_cover.sum(dim=1, keepdim=True).clamp(min=1e-8)

        prior = net.compute_mixture_prior(land_cover, material_table)

        assert prior.shape == (N_SUBSAMPLE, 1), f"Mixture prior shape: {prior.shape}"
        assert torch.isfinite(prior).all(), "Prior should be finite"
        assert (prior >= 0.01).all() and (prior <= 5.0).all(), "Prior should be in bounds"


# ===================================================================
# 5. Neural Meta-Learner (SPARCMetaLearner)
# ===================================================================

class TestSPARCMetaLearner:
    def _build(self, arrays, device):
        from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet
        from scipy.spatial import cKDTree

        X, y, coords = arrays
        n_base = 3
        n_freq = 16
        max_neighbors = 16

        encoder = SinusoidalSpatialEncoding(n_frequencies=n_freq)
        coords_t = torch.tensor(coords, device=device)
        X_spatial = encoder(coords_t)
        d_spatial = X_spatial.shape[1]

        tree = cKDTree(coords)
        _, knn_idx = tree.query(coords, k=max_neighbors + 1)
        knn_idx = knn_idx[:, 1:]
        knn_t = torch.tensor(knn_idx, dtype=torch.long, device=device)

        model = SPARCMetaLearner(
            n_base_models=n_base,
            n_physics_features=X.shape[1],
            d_spatial=d_spatial,
            hidden_dim=64,
            dropout=0.1,
            thresholds=[0.25, 0.75],
            n_heads=2,
            max_neighbors=max_neighbors,
            siren_omega=30.0,
        ).to(device)

        process_net = ProcessRateNet(
            n_inputs=3,
            domain_config={"name": "test", "units": "", "bounds": [0.01, 1.0], "prior_mean": 0.5},
        ).to(device)

        base_preds = torch.randn(N_SUBSAMPLE, n_base, device=device)
        physics = torch.tensor(X, device=device)
        alpha = process_net(physics[:, :3])

        return model, process_net, base_preds, physics, X_spatial, coords_t, knn_t, alpha, y

    def test_forward(self, arrays, device):
        model, _, base_preds, physics, X_spatial, coords_t, knn_t, alpha, _ = self._build(arrays, device)

        T_pred, exceedance, attn_w = model(base_preds, physics, X_spatial, coords_t, knn_t, alpha)

        assert T_pred.shape == (N_SUBSAMPLE,), f"T_pred shape: {T_pred.shape}"
        assert len(exceedance) == 2, "Should have 2 exceedance heads"
        assert exceedance[0].shape == (N_SUBSAMPLE,)
        assert not torch.isnan(exceedance[0]).any(), "NaN in exceedance output"
        assert (exceedance[0] >= 0).all() and (exceedance[0] <= 1).all(), "Exceedance should be in [0,1]"
        assert not torch.isnan(T_pred).any()

    def test_mc_dropout_uncertainty(self, arrays, device):
        model, _, base_preds, physics, X_spatial, coords_t, knn_t, alpha, _ = self._build(arrays, device)

        mean, std = model.predict_with_uncertainty(
            base_preds, physics, X_spatial, coords_t, knn_t, alpha, n_samples=10,
        )

        assert mean.shape == (N_SUBSAMPLE,)
        assert std.shape == (N_SUBSAMPLE,)
        assert not torch.isnan(std).any(), "NaN in uncertainty"
        assert (std >= 0).all(), "Uncertainty should be non-negative"
        assert not torch.isnan(mean).any()

    def test_backward(self, arrays, device):
        model, process_net, base_preds, physics, X_spatial, coords_t, knn_t, _, y = self._build(arrays, device)

        # Re-compute alpha with gradient tracking
        alpha = process_net(physics[:, :3])

        y_t = torch.tensor(y, device=device)
        T_pred, _, _ = model(base_preds, physics, X_spatial, coords_t, knn_t, alpha)

        loss = ((T_pred - y_t) ** 2).mean()
        loss.backward()

        meta_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        pr_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in process_net.parameters())

        assert meta_has_grad, "No gradients in SPARCMetaLearner"
        assert pr_has_grad, "No gradients in ProcessRateNet"


# ===================================================================
# 6. Training Loss
# ===================================================================

class TestTrainingLoss:
    def test_joint_loss_runs(self, arrays, device):
        from sparc.training.loss import sparc_joint_loss

        X, y, coords = arrays
        N = N_SUBSAMPLE

        T_pred = torch.randn(N, device=device, requires_grad=True)
        y_t = torch.tensor(y, device=device)
        exc = [torch.sigmoid(torch.randn(N, device=device, requires_grad=True))]
        alpha = torch.sigmoid(torch.randn(N, 1, device=device, requires_grad=True))
        alpha_prior = torch.full((N, 1), 0.5, device=device)

        total, components = sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=exc,
            y_true=y_t,
            thresholds=[88.0],
            alpha=alpha,
            alpha_prior=alpha_prior,
            neighbor_idx=None,
            source_term=torch.zeros(N, device=device),
            resolution=30.0,
            surrogate_preds=[],
            surrogate_targets=[],
            lambda_physics=0.01,
            lambda_smooth=0.01,
            lambda_alpha_smooth=0.001,
            lambda_prior=1.0,
            lambda_base=0.0,
            lambda_neighbor=0.1,
            epoch=0,
        )

        assert total.shape == (), "Loss should be scalar"
        assert torch.isfinite(total), f"Loss not finite: {total.item()}"
        assert "mse" in components

        total.backward()
        assert T_pred.grad is not None, "No gradient on T_pred"


# ===================================================================
# 7. Training Optimizer
# ===================================================================

class TestOptimizer:
    def test_build_optimizer_and_scheduler(self, arrays, device):
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet
        from sparc.training.optimizer import build_optimizer, build_scheduler

        model = SPARCMetaLearner(
            n_base_models=3, n_physics_features=6, d_spatial=64,
            hidden_dim=64, n_heads=2, max_neighbors=16,
        ).to(device)

        process_net = ProcessRateNet(
            n_inputs=3,
            domain_config={"name": "test", "units": "", "bounds": [0.01, 1.0], "prior_mean": 0.5},
        ).to(device)

        optimizer = build_optimizer(model, process_net, surrogates={}, base_lr=1e-3)
        scheduler = build_scheduler(optimizer, n_epochs=50)

        assert optimizer is not None
        assert scheduler is not None
        assert len(optimizer.param_groups) > 0


# ===================================================================
# 8. Curriculum
# ===================================================================

class TestCurriculum:
    def test_lambda_schedule_stages(self):
        from sparc.training.curriculum import get_lambda_schedule

        target_lambdas = {
            "physics": 0.01,
            "smooth": 0.01,
            "alpha_smooth": 0.001,
            "base": 0.1,
            "neighbor": 0.1,
            "prior": 1.0,
            "mse": 1.0,
            "exceedance": 0.1,
        }

        # Stage A (warmup): physics should be 0
        lambdas_a = get_lambda_schedule(epoch=0, target_lambdas=target_lambdas)
        assert lambdas_a["physics"] == 0.0, f"Stage A physics should be 0, got {lambdas_a['physics']}"

        # Stage B (physics activation): physics should ramp up
        lambdas_b = get_lambda_schedule(epoch=15, target_lambdas=target_lambdas)
        assert lambdas_b["physics"] > 0, "Stage B physics should be > 0"

        # Stage C (joint): all lambdas active
        lambdas_c = get_lambda_schedule(epoch=50, target_lambdas=target_lambdas)
        assert lambdas_c["mse"] > 0
        assert lambdas_c["physics"] > 0
        assert lambdas_c["exceedance"] > 0


# ===================================================================
# 9. CMA-ES (quick smoke test)
# ===================================================================

class TestCMAES:
    def test_decode_encode_roundtrip(self):
        from sparc.training.cma_es import decode_vector, encode_defaults, DEFAULT_SPACE

        x0 = encode_defaults()
        params = decode_vector(x0)

        assert len(params) == 12, f"Expected 12 params, got {len(params)}"
        assert "learning_rate" in params
        assert "lambda_mse" in params
        assert params["learning_rate"] == pytest.approx(1e-3, rel=0.01)

    def test_cma_es_runs(self):
        """Quick CMA-ES with a trivial objective (2 generations)."""
        from sparc.training.cma_es import run_cma_es

        call_count = [0]

        def dummy_objective(params):
            call_count[0] += 1
            return sum(v ** 2 for v in params.values())

        result = run_cma_es(
            dummy_objective,
            max_generations=2,
            population_size=4,
            sigma0=0.3,
        )

        assert result.best_loss < float("inf")
        assert result.n_generations >= 1
        assert call_count[0] >= 4, "Should have evaluated at least one population"


# ===================================================================
# 10. MC³ (Bayesian DAG structure learning)
# ===================================================================

class TestMC3:
    def test_mc3_on_brown_data(self, brown_data):
        from sparc.causal.mc3 import (
            DAGStructure,
            PhysicsInformedGraphPrior,
            bge_score,
            run_mc3,
        )

        node_names = ["Pct_Canopy", "Pct_Impervious", "NDVI", "AAT_z"]
        data_sub = brown_data[node_names]

        # Build prior (Canopy→AAT, Impervious→AAT edges preferred)
        p = len(node_names)
        probs = np.full((p, p), 0.1)
        probs[0, 3] = 0.8  # Canopy → AAT
        probs[1, 3] = 0.8  # Impervious → AAT
        probs[2, 3] = 0.7  # NDVI → AAT
        np.fill_diagonal(probs, 0.0)
        prior = PhysicsInformedGraphPrior(edge_probs=probs)

        results = run_mc3(
            data=data_sub,
            node_names=node_names,
            prior=prior,
            n_iter=500,
            n_chains=2,
            seed=42,
        )

        assert results.best_dag is not None
        assert results.edge_inclusion_probs.shape == (4, 4)
        assert results.n_total == 500
        assert results.n_accepted > 0, "MC³ should accept at least some proposals"

        # The edge from Impervious → AAT should have reasonable inclusion prob
        imp_to_aat = results.edge_inclusion_probs[1, 3]
        print(f"  MC3: Impervious->AAT inclusion prob = {imp_to_aat:.3f}")

    def test_dag_structure_roundtrip(self):
        import networkx as nx
        from sparc.causal.mc3 import DAGStructure

        G = nx.DiGraph()
        G.add_edges_from([("A", "C"), ("B", "C")])
        dag = DAGStructure.from_networkx(G)

        assert dag.n_nodes == 3
        G2 = dag.to_networkx()
        assert set(G2.edges()) == set(G.edges())


# ===================================================================
# 11. NUTS Sampler
# ===================================================================

class TestNUTS:
    def test_nuts_simple_normal(self):
        """Sample from a known Normal(3, 1) posterior."""
        from sparc.causal.nuts import NUTSBlock, run_nuts

        def log_prob(params):
            x = params["mu"]
            return -0.5 * (x - 3.0) ** 2  # N(3, 1)

        blocks = [NUTSBlock(name="mu", dim=1, init=np.array([0.0]))]

        results = run_nuts(
            log_prob_fn=log_prob,
            blocks=blocks,
            n_samples=300,
            n_warmup=100,
            max_depth=6,
            init_step_size=0.5,
            seed=42,
        )

        mu_samples = results.samples["mu"][:, 0]
        posterior_mean = mu_samples.mean()

        assert abs(posterior_mean - 3.0) < 1.0, f"Posterior mean {posterior_mean:.2f} should be near 3.0"
        assert results.acceptance_rate > 0.3, f"Acceptance rate too low: {results.acceptance_rate:.2f}"
        assert results.n_divergences < results.log_probs.shape[0] * 0.5, "Too many divergences"

    def test_nuts_with_brown_treatment_effect(self, brown_data):
        """Estimate Pct_Impervious → AAT_z treatment effect via NUTS."""
        from sparc.causal.nuts import NUTSBlock, run_nuts

        y = torch.tensor(brown_data[TARGET].values, dtype=torch.float64)
        X = torch.tensor(brown_data["Pct_Impervious"].values, dtype=torch.float64)
        X = (X - X.mean()) / X.std()  # standardise
        y = (y - y.mean()) / y.std()

        def log_prob(params):
            beta = params["beta"]
            sigma2 = params["sigma2"]
            mu = X * beta[0]
            ll = -0.5 * torch.sum(torch.log(sigma2) + (y - mu) ** 2 / sigma2)
            lp = -0.5 * beta[0] ** 2 / 10.0 - 2.0 * torch.log(sigma2)
            return ll + lp

        blocks = [
            NUTSBlock(name="beta", dim=1, init=np.array([0.0])),
            NUTSBlock(name="sigma2", dim=1, init=np.array([0.0]), transform="log"),
        ]

        results = run_nuts(
            log_prob_fn=log_prob, blocks=blocks,
            n_samples=200, n_warmup=100, max_depth=6,
            init_step_size=0.01, seed=42,
        )

        beta_mean = results.samples["beta"][:, 0].mean()
        # Impervious should have a positive effect on temperature z-score
        print(f"  NUTS: beta(Pct_Impervious->AAT_z) = {beta_mean:.4f}")
        assert results.acceptance_rate > 0.1, f"Acceptance too low: {results.acceptance_rate:.2f}"


# ===================================================================
# 12. End-to-end mini training loop
# ===================================================================

class TestEndToEnd:
    def test_mini_training_loop(self, arrays, device):
        """Train SPARCMetaLearner for 3 epochs on brown4 subsample."""
        from sparc.features.sinusoidal_encoding import SinusoidalSpatialEncoding
        from sparc.models.neural_meta import SPARCMetaLearner
        from sparc.models.process_rate_net import ProcessRateNet
        from sparc.training.loss import sparc_joint_loss
        from sparc.training.curriculum import get_lambda_schedule
        from scipy.spatial import cKDTree

        X, y, coords = arrays
        N = len(y)
        n_base = 4
        n_freq = 16
        max_neighbors = 16

        # Encoding
        encoder = SinusoidalSpatialEncoding(n_frequencies=n_freq)
        coords_t = torch.tensor(coords, device=device)
        X_spatial = encoder(coords_t)
        d_spatial = X_spatial.shape[1]

        # KNN
        tree = cKDTree(coords)
        _, knn_idx = tree.query(coords, k=max_neighbors + 1)
        knn_idx = knn_idx[:, 1:]
        knn_t = torch.tensor(knn_idx, dtype=torch.long, device=device)

        # Models
        model = SPARCMetaLearner(
            n_base_models=n_base, n_physics_features=X.shape[1],
            d_spatial=d_spatial, hidden_dim=64, dropout=0.1,
            thresholds=[88.0], n_heads=2, max_neighbors=max_neighbors,
        ).to(device)

        process_net = ProcessRateNet(
            n_inputs=3,
            domain_config={"name": "test", "units": "", "bounds": [0.01, 1.0], "prior_mean": 0.5},
        ).to(device)

        # Fake base-model OOF preds (random for this test)
        base_preds = torch.randn(N, n_base, device=device)
        physics = torch.tensor(X, device=device)
        y_t = torch.tensor(y, device=device)

        all_params = list(model.parameters()) + list(process_net.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=1e-3)

        target_lambdas = {
            "physics": 0.01, "smooth": 0.01, "alpha_smooth": 0.001,
            "base": 0.0, "neighbor": 0.1, "prior": 1.0,
        }

        losses = []
        for epoch in range(3):
            lambdas = get_lambda_schedule(epoch, target_lambdas)
            alpha = process_net(physics[:, :3])
            alpha_prior = torch.full_like(alpha, 0.5)
            T_pred, exceedance, _ = model(
                base_preds, physics, X_spatial, coords_t, knn_t, alpha,
            )
            total, components = sparc_joint_loss(
                T_pred=T_pred,
                exceedance_preds=exceedance,
                y_true=y_t,
                thresholds=[88.0],
                alpha=alpha,
                alpha_prior=alpha_prior,
                neighbor_idx=None,
                source_term=torch.zeros(N, device=device),
                resolution=30.0,
                surrogate_preds=[],
                surrogate_targets=[],
                lambda_physics=lambdas.get("physics", 0.0),
                lambda_smooth=lambdas.get("smooth", 0.0),
                lambda_alpha_smooth=lambdas.get("alpha_smooth", 0.0),
                lambda_prior=lambdas.get("prior", 1.0),
                lambda_base=lambdas.get("base", 0.0),
                lambda_neighbor=lambdas.get("neighbor", 0.0),
                epoch=epoch,
            )
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            losses.append(total.item())

        assert all(np.isfinite(losses)), f"Non-finite losses: {losses}"
        # Loss should not explode
        assert losses[-1] < losses[0] * 100, f"Loss exploded: {losses}"
        print(f"  End-to-end losses: {[f'{l:.4f}' for l in losses]}")
