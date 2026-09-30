"""Regression tests for the G5 physics-loss fixes (roadmap Appendix A12–A18).

* directional / anisotropy / gradient_flux default to weight 0 and are skipped
* PDELossWeights.from_config reads ``physics.pde_weights``
* the sheaf term is skipped (not crashing) when δ⁰ was built on the full
  graph but T_pred is a mini-batch
* energy_balance: ``ground_conduction_proxy`` replaces the misnamed
  ``sensible_heat_flux`` (kept as a deprecated alias)
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparc.physics import pde_loss as pde_loss_mod
from sparc.physics.pde_loss import PDELossWeights, compute_pde_loss


def _grid_neighbors(n_rows: int, n_cols: int) -> torch.Tensor:
    """Cardinal neighbours [N, S, E, W] on a regular grid (-1 = missing)."""
    idx = torch.full((n_rows * n_cols, 4), -1, dtype=torch.long)
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            if i > 0:
                idx[k, 0] = (i - 1) * n_cols + j
            if i < n_rows - 1:
                idx[k, 1] = (i + 1) * n_cols + j
            if j < n_cols - 1:
                idx[k, 2] = i * n_cols + j + 1
            if j > 0:
                idx[k, 3] = i * n_cols + j - 1
    return idx


def _inputs(n_rows=6, n_cols=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = n_rows * n_cols
    T = torch.randn(N, generator=g)
    alpha = torch.rand(N, 1, generator=g) * 0.5 + 0.1
    source = torch.randn(N, generator=g)
    return T, alpha, source, _grid_neighbors(n_rows, n_cols)


# Late enough that every staged term (offsets ≤ 20, ramp 5) is fully active.
LATE_EPOCH = 80


class TestDefaultWeights:
    def test_defaults_are_zero_for_uninformative_terms(self):
        w = PDELossWeights()
        assert w.directional == 0.0
        assert w.anisotropy == 0.0
        assert w.gradient_flux == 0.0
        assert w.heat_diffusion > 0

    def test_default_terms_contribute_zero_but_keys_present(self):
        T, alpha, source, nb = _inputs()
        _, d = compute_pde_loss(T, alpha, source, nb, 1.0, epoch=LATE_EPOCH)
        for key in ("pde_directional", "pde_anisotropy", "pde_gradient_flux"):
            assert key in d
            assert d[key] == 0.0
        assert d["pde_heat_diffusion"] > 0

    def test_zero_weight_blocks_are_skipped(self, monkeypatch):
        """With default weights the operators for the dead terms are never called."""
        def _boom(*a, **k):
            raise AssertionError("operator should not be evaluated for a 0-weight term")

        monkeypatch.setattr(pde_loss_mod, "directional_curvatures", _boom)
        monkeypatch.setattr(pde_loss_mod, "gradient_magnitude", _boom)
        T, alpha, source, nb = _inputs()
        total, d = compute_pde_loss(T, alpha, source, nb, 1.0, epoch=LATE_EPOCH)
        assert torch.isfinite(total)

    def test_explicit_weights_restore_legacy_terms(self):
        T, alpha, source, nb = _inputs()
        w = PDELossWeights(directional=1.0, anisotropy=1.0, gradient_flux=1.0)
        _, d = compute_pde_loss(T, alpha, source, nb, 1.0, weights=w, epoch=LATE_EPOCH)
        assert d["pde_anisotropy"] > 0
        assert d["pde_gradient_flux"] > 0
        assert "pde_directional" in d

    def test_directional_residual_is_identically_zero(self):
        """d2x + d2y − ∇² vanishes (up to float round-off) with the shared
        5-point stencil — the defect that motivated weight 0.  (Per-residual
        normalisation would then amplify pure round-off noise.)"""
        from sparc.physics.pde_operators import directional_curvatures, laplacian
        T, _, _, nb = _inputs()
        d2x, d2y, valid_d = directional_curvatures(T, nb, 1.0)
        lap, valid_l = laplacian(T, nb, 1.0)
        assert torch.equal(valid_d, valid_l)
        assert torch.allclose(d2x + d2y, lap, atol=1e-5)


class TestFromConfig:
    def test_reads_physics_pde_weights_and_ignores_unknown(self):
        cfg = {"physics": {"pde_weights": {
            "heat_diffusion": 2.0, "anisotropy": 0.3,
            "energy_balance": 9.0,          # legacy desktop key — ignored
            "not_a_term": 1.0,
        }}}
        w = PDELossWeights.from_config(cfg)
        assert w.heat_diffusion == 2.0
        assert w.anisotropy == 0.3
        assert w.directional == 0.0
        assert not hasattr(w, "energy_balance")

    @pytest.mark.parametrize("cfg", [None, {}, {"physics": {}}, {"physics": {"pde_weights": None}}])
    def test_missing_sections_give_defaults(self, cfg):
        assert PDELossWeights.from_config(cfg) == PDELossWeights()

    def test_compute_pde_loss_accepts_config_mapping(self):
        T, alpha, source, nb = _inputs()
        cfg = {"physics": {"pde_weights": {"gradient_flux": 0.5}}}
        _, d = compute_pde_loss(T, alpha, source, nb, 1.0, weights=cfg, epoch=LATE_EPOCH)
        assert d["pde_gradient_flux"] > 0


class TestSheafGuard:
    def _full_graph_delta(self, n_full: int):
        from sparc.physics.pde_operators import build_sheaf_laplacian
        nb_full = _grid_neighbors(10, n_full // 10)
        return build_sheaf_laplacian(nb_full, stalk_dim=2)

    def test_batch_sized_prediction_with_full_graph_delta_does_not_raise(self):
        delta = self._full_graph_delta(100)          # built on N_full = 100
        T, alpha, source, nb = _inputs(5, 5)          # batch of 25 points
        assert delta.shape[1] == 200 != 2 * T.shape[0]
        total, d = compute_pde_loss(
            T, alpha, source, nb, 1.0, epoch=LATE_EPOCH, sheaf_delta=delta,
        )
        assert d["pde_sheaf"] == 0.0
        assert torch.isfinite(total)

    def test_matching_delta_is_applied(self):
        delta = self._full_graph_delta(100)
        T, alpha, source, nb = _inputs(10, 10)        # full N = 100
        _, d = compute_pde_loss(
            T, alpha, source, nb, 1.0, epoch=LATE_EPOCH, sheaf_delta=delta,
        )
        assert d["pde_sheaf"] > 0


class TestEnergyBalanceRename:
    def test_ground_conduction_proxy(self):
        from sparc.physics.energy_balance import ground_conduction_proxy
        lap = torch.full((4,), 2.0)
        g = ground_conduction_proxy(lap, depth=0.5, k_thermal=1.5)
        assert torch.allclose(g, torch.full((4,), -1.5))

    def test_sensible_heat_alias_warns_and_matches(self):
        from sparc.physics.energy_balance import ground_conduction_proxy, sensible_heat_flux
        lap = torch.linspace(-1, 1, 5)
        with pytest.warns(DeprecationWarning, match="ground_conduction_proxy"):
            qh = sensible_heat_flux(lap)
        assert torch.allclose(qh, ground_conduction_proxy(lap))

    def test_uncalled_residual_removed(self):
        import sparc.physics.energy_balance as eb
        assert not hasattr(eb, "energy_balance_residual")
