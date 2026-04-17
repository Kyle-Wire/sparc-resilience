"""
Tests for SPARC V3 physics integration and α(s) → simulation wiring.
"""

import numpy as np
import pytest
import torch


# =========================================================================
# PDE Operators
# =========================================================================

class TestPDEOperators:
    """Test finite-difference PDE operators on regular grids."""

    @pytest.fixture
    def regular_grid(self):
        """5x5 regular grid with known analytic function f = x² + y²."""
        xs = np.linspace(0, 4, 5)
        ys = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        coords_t = torch.tensor(coords, dtype=torch.float32)
        # f = x² + y² → ∇²f = 4 everywhere
        f_values = torch.tensor(
            (coords[:, 0] ** 2 + coords[:, 1] ** 2),
            dtype=torch.float32,
        )
        return coords_t, f_values

    @pytest.fixture
    def cardinal_neighbors(self, regular_grid):
        """Build cardinal neighbors for the 5x5 grid."""
        from sparc.run.v2_neural_training import _build_cardinal_neighbors
        coords_t, _ = regular_grid
        coords_np = coords_t.numpy()
        neighbor_idx, _ = _build_cardinal_neighbors(coords_np, resolution=1.0)
        return torch.tensor(neighbor_idx, dtype=torch.long)

    def test_laplacian_quadratic(self, regular_grid, cardinal_neighbors):
        """Laplacian of x² + y² should be 4."""
        from sparc.physics.pde_operators import laplacian
        coords_t, f_values = regular_grid
        lap, valid = laplacian(f_values, cardinal_neighbors, h=1.0)
        # Only check truly interior points (boundaries have self-referencing neighbors)
        truly_interior = (
            (coords_t[:, 0] > 0) & (coords_t[:, 0] < 4) &
            (coords_t[:, 1] > 0) & (coords_t[:, 1] < 4)
        )
        # Expand lap to full size for consistent indexing
        lap_full = torch.zeros(len(coords_t))
        lap_full[valid] = lap
        assert truly_interior.sum() > 0
        np.testing.assert_allclose(
            lap_full[truly_interior].numpy(), 4.0, atol=1e-4,
        )

    def test_gradient_magnitude_linear(self, regular_grid, cardinal_neighbors):
        """Gradient of linear function f = 3x + 4y should have |∇f| = 5."""
        from sparc.physics.pde_operators import gradient_magnitude
        coords_t, _ = regular_grid
        f_linear = 3.0 * coords_t[:, 0] + 4.0 * coords_t[:, 1]
        grad_mag, _, _, valid = gradient_magnitude(f_linear, cardinal_neighbors, h=1.0)
        truly_interior = (
            (coords_t[:, 0] > 0) & (coords_t[:, 0] < 4) &
            (coords_t[:, 1] > 0) & (coords_t[:, 1] < 4)
        )
        grad_full = torch.zeros(len(coords_t))
        grad_full[valid] = grad_mag
        np.testing.assert_allclose(
            grad_full[truly_interior].numpy(), 5.0, atol=1e-4,
        )

    def test_estimate_local_spacing(self, regular_grid, cardinal_neighbors):
        """Local spacing estimation on uniform grid should be ≈ 1.0."""
        from sparc.physics.pde_operators import estimate_local_spacing
        coords_t, _ = regular_grid
        h = estimate_local_spacing(coords_t, cardinal_neighbors, spacing='local')
        valid = cardinal_neighbors.min(dim=1).values >= 0
        h_interior = h[valid]
        np.testing.assert_allclose(
            h_interior.numpy(), 1.0, atol=0.1,
        )

    def test_directional_curvatures(self, regular_grid, cardinal_neighbors):
        """For f = x², ∂²f/∂x² = 2, ∂²f/∂y² = 0."""
        from sparc.physics.pde_operators import directional_curvatures
        coords_t, _ = regular_grid
        f_x2 = coords_t[:, 0] ** 2
        d2_dx2, d2_dy2, valid = directional_curvatures(f_x2, cardinal_neighbors, h=1.0)
        truly_interior = (
            (coords_t[:, 0] > 0) & (coords_t[:, 0] < 4) &
            (coords_t[:, 1] > 0) & (coords_t[:, 1] < 4)
        )
        if valid.any():
            dx2_full = torch.zeros(len(coords_t))
            dy2_full = torch.zeros(len(coords_t))
            dx2_full[valid] = d2_dx2
            dy2_full[valid] = d2_dy2
            np.testing.assert_allclose(dx2_full[truly_interior].numpy(), 2.0, atol=1e-4)
            np.testing.assert_allclose(dy2_full[truly_interior].numpy(), 0.0, atol=1e-4)


# =========================================================================
# PDE Encoder
# =========================================================================

class TestPDEEncoder:
    """Test that PDE-informed physics encoder produces correct shapes."""

    def test_forward_shape(self):
        from sparc.models.pde_encoder import PDEInformedPhysicsEncoder
        enc = PDEInformedPhysicsEncoder(
            n_physics_features=20,
            out_dim=64,
            omega=30.0,
        )
        x = torch.randn(50, 20)
        encoded, w_source = enc(x)
        assert encoded.shape == (50, 64)
        assert w_source.shape == (50, 1)
        # w_source should be in [0, 1] (sigmoid output)
        assert (w_source >= 0).all() and (w_source <= 1).all()

    def test_gradient_flow(self):
        from sparc.models.pde_encoder import PDEInformedPhysicsEncoder
        enc = PDEInformedPhysicsEncoder(n_physics_features=10, out_dim=32)
        x = torch.randn(20, 10, requires_grad=True)
        encoded, w = enc(x)
        loss = encoded.sum() + w.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# =========================================================================
# PDE Loss
# =========================================================================

class TestPDELoss:
    """Test staged PDE loss curriculum."""

    def test_staged_activation(self):
        from sparc.physics.pde_loss import compute_pde_loss, PDELossWeights
        N = 25
        T = torch.randn(N)
        alpha = torch.rand(N, 1) * 0.5 + 0.1
        source = torch.randn(N)
        # Simple 5x5 grid neighbors (mock)
        neighbor_idx = torch.full((N, 4), -1, dtype=torch.long)
        # Just set interior point 12 to have all 4 neighbors
        neighbor_idx[12] = torch.tensor([7, 17, 13, 11])
        h = 1.0
        weights = PDELossWeights()

        # Epoch 0: before PDE activation (pde_start_epoch=30), nothing active
        _, d0 = compute_pde_loss(T, alpha, source, neighbor_idx, h,
                                 weights=weights, epoch=0)
        assert d0["pde_heat_diffusion"] >= 0
        assert d0["pde_energy_balance"] == 0
        assert d0["pde_directional"] == 0

        # Epoch 36: heat_diffusion fully active, energy_balance ramping
        _, d36 = compute_pde_loss(T, alpha, source, neighbor_idx, h,
                                  weights=weights, epoch=36,
                                  energy_residual=torch.randn(N))
        assert d36["pde_energy_balance"] > 0

        # Epoch 55: all terms fully active (pde_epoch=25, all offsets < 20)
        _, d55 = compute_pde_loss(T, alpha, source, neighbor_idx, h,
                                  weights=weights, epoch=55,
                                  energy_residual=torch.randn(N),
                                  alpha_prior_field=torch.rand(N, 1))
        assert d55["pde_alpha_prior"] > 0


# =========================================================================
# Boundary Conditions
# =========================================================================

class TestBoundaryConditions:
    def test_detect_boundary_points(self):
        from sparc.physics.boundary_conditions import detect_boundary_points
        # 4 points: point 0 missing north, point 3 missing all
        neighbor_idx = torch.tensor([
            [-1, 1, 2, 3],  # missing north
            [0, 2, 3, -1],  # missing west
            [1, 3, -1, 0],  # missing east
            [-1, -1, -1, -1],  # corner
        ], dtype=torch.long)
        mask, edges = detect_boundary_points(neighbor_idx)
        assert mask.sum() == 4  # all are boundary
        assert edges["north"][0] == True
        assert edges["north"][1] == False

    def test_neumann_loss_constant_field(self):
        """Constant field should have zero Neumann loss."""
        from sparc.physics.boundary_conditions import neumann_loss
        T = torch.ones(10) * 5.0
        boundary_mask = torch.zeros(10, dtype=torch.bool)
        boundary_mask[0] = True
        neighbor_idx = torch.full((10, 4), -1, dtype=torch.long)
        # Point 0 missing north but has south=1
        neighbor_idx[0] = torch.tensor([-1, 1, 2, 3])
        for i in range(1, 10):
            neighbor_idx[i] = torch.tensor([max(0, i-3), min(9, i+3),
                                            min(9, i+1), max(0, i-1)])
        loss = neumann_loss(T, boundary_mask, neighbor_idx, h=1.0)
        assert loss.item() < 1e-6


# =========================================================================
# Energy Balance
# =========================================================================

class TestEnergyBalance:
    def test_net_radiation_sign(self):
        from sparc.physics.energy_balance import net_radiation
        T = torch.full((10,), 300.0)  # 300K
        albedo = torch.full((10,), 0.2)
        Q_star = net_radiation(T, albedo)
        # Net radiation should be positive (incoming > outgoing)
        assert (Q_star > 0).all()

    def test_sensible_heat_positive_laplacian(self):
        from sparc.physics.energy_balance import sensible_heat_flux
        lap = torch.ones(10) * 2.0  # positive laplacian
        QH = sensible_heat_flux(lap)
        # QH = -k·∇²T·d → negative for positive laplacian
        assert (QH < 0).all()


# =========================================================================
# Initial Conditions
# =========================================================================

class TestInitialConditions:
    def test_ic_from_ols(self):
        from sparc.physics.initial_conditions import compute_initial_condition
        torch.manual_seed(42)
        X = torch.randn(50, 5)
        beta_true = torch.tensor([1.0, -0.5, 0.3, 0.0, 0.2])
        y = X @ beta_true + torch.randn(50) * 0.1
        T0 = compute_initial_condition(X, y)
        assert T0.shape == (50,)
        # T0 should correlate with y
        corr = np.corrcoef(T0.numpy(), y.numpy())[0, 1]
        assert corr > 0.5

    def test_warmup_schedule(self):
        from sparc.physics.initial_conditions import warmup_ic_schedule
        loss = torch.tensor(1.0)
        # Before warmup: weight = 0
        w0 = warmup_ic_schedule(0, loss, warmup_epochs=20)
        assert w0.item() == 0.0
        # After double warmup: weight = max_weight
        w40 = warmup_ic_schedule(40, loss, warmup_epochs=20, max_weight=0.05)
        assert abs(w40.item() - 0.05) < 1e-6


# =========================================================================
# Neural Meta-Learner (Stream 2 replacement)
# =========================================================================

class TestNeuralMetaPDEStream:
    def test_meta_learner_forward_with_pde_encoder(self):
        from sparc.models.neural_meta import SPARCMetaLearner
        model = SPARCMetaLearner(
            n_base_models=3,
            n_physics_features=30,  # extended features
            d_spatial=16,
            hidden_dim=32,
            thresholds=[0.5],
        )
        N = 20
        base_preds = torch.randn(N, 3)
        physics = torch.randn(N, 30)
        spatial = torch.randn(N, 16)
        coords = torch.randn(N, 2)
        knn = torch.randint(0, N, (N, 10))
        alpha = torch.rand(N, 1)

        T_pred, exc, attn = model(base_preds, physics, spatial, coords, knn, alpha)
        assert T_pred.shape == (N,)
        assert len(exc) == 1
        assert attn.shape[0] == N
        # w_source should be stored
        assert model._last_w_source is not None
        assert model._last_w_source.shape == (N, 1)


# =========================================================================
# Gini Coefficient
# =========================================================================

class TestGiniCoefficient:
    def test_uniform_gini_zero(self):
        from sparc.interventions.scenario_simulator import spatial_gini_coefficient
        values = np.ones(100)
        gini = spatial_gini_coefficient(values)
        assert abs(gini) < 0.01

    def test_concentrated_gini_high(self):
        from sparc.interventions.scenario_simulator import spatial_gini_coefficient
        values = np.zeros(100)
        values[0] = 100.0
        gini = spatial_gini_coefficient(values)
        assert gini > 0.95

    def test_moderate_gini(self):
        from sparc.interventions.scenario_simulator import spatial_gini_coefficient
        np.random.seed(42)
        values = np.random.exponential(1.0, 1000)
        gini = spatial_gini_coefficient(values)
        # Exponential distribution has Gini ≈ 0.5
        assert 0.3 < gini < 0.7


# =========================================================================
# Loss integration
# =========================================================================

class TestLossIntegration:
    def test_joint_loss_backward_compatible(self):
        """Existing V2 calls should still work (no PDE/BC args)."""
        from sparc.training.loss import sparc_joint_loss
        N = 50
        T_pred = torch.randn(N, requires_grad=True)
        y = torch.randn(N)
        alpha = torch.rand(N, 1)
        alpha_prior = torch.full((N, 1), 0.5)
        neighbor_idx = torch.full((N, 4), -1, dtype=torch.long)
        source = torch.zeros(N)

        total, components = sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=[],
            y_true=y,
            thresholds=[],
            alpha=alpha,
            alpha_prior=alpha_prior,
            neighbor_idx=neighbor_idx,
            source_term=source,
            resolution=100.0,
            surrogate_preds=[],
            surrogate_targets=[],
            lambda_physics=0.01,
            lambda_smooth=0.01,
            lambda_alpha_smooth=0.001,
            lambda_prior=1.0,
            lambda_base=0.0,
            lambda_neighbor=0.0,
            epoch=0,
        )
        assert "mse" in components
        assert "pde_total" in components
        assert components["pde_total"] == 0.0  # not active
        total.backward()
        assert T_pred.grad is not None

    def test_joint_loss_with_pde(self):
        """V3 PDE loss should activate when lambda_pde > 0."""
        from sparc.training.loss import sparc_joint_loss
        N = 25
        T_pred = torch.randn(N, requires_grad=True)
        y = torch.randn(N)
        alpha = torch.rand(N, 1)
        neighbor_idx = torch.full((N, 4), -1, dtype=torch.long)
        neighbor_idx[12] = torch.tensor([7, 17, 13, 11])

        total, components = sparc_joint_loss(
            T_pred=T_pred,
            exceedance_preds=[],
            y_true=y,
            thresholds=[],
            alpha=alpha,
            alpha_prior=torch.full((N, 1), 0.5),
            neighbor_idx=neighbor_idx,
            source_term=torch.zeros(N),
            resolution=1.0,
            surrogate_preds=[],
            surrogate_targets=[],
            lambda_physics=0.0,
            lambda_smooth=0.0,
            lambda_alpha_smooth=0.0,
            lambda_prior=0.0,
            lambda_base=0.0,
            lambda_neighbor=0.0,
            epoch=36,
            h_field=1.0,
            lambda_pde=1.0,
        )
        assert components["pde_total"] > 0
        total.backward()


# =========================================================================
# Curriculum
# =========================================================================

class TestCurriculum:
    def test_pde_lambda_schedule(self):
        from sparc.training.curriculum import get_lambda_schedule
        target = {"pde": 0.1, "bc": 0.02, "physics": 0.01}

        # Before ramp_end: pde=0
        s = get_lambda_schedule(5, target, warmup_end=10, ramp_end=30)
        assert s["pde"] == 0.0
        assert s["bc"] == 0.0

        # At ramp_end: pde=0
        s = get_lambda_schedule(25, target, warmup_end=10, ramp_end=30)
        assert s["pde"] == 0.0

        # After ramp_end: pde=target
        s = get_lambda_schedule(35, target, warmup_end=10, ramp_end=30)
        assert s["pde"] == 0.1
        assert s["bc"] == 0.02
