"""Numerical verification of the advection–diffusion–relaxation operator
(physics plan W1): positivity of the exponentially fitted scheme over the
fit box, convergence to the analytic K₀ Green's function, manufactured
solutions, padding truncation, the residual and adjoint identities, the
downwind shift, the torch twins, the legacy central scheme, and what the
fitted scheme costs: accuracy at moderate cell Péclet numbers and the
identifiability of L above Pe = 1 (``grid_resolution``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import k0

from sparc.core import operators as ops

DX = 30.0
# The physics fit box: L from dx (log L ≥ log dx) and |v| up to v_max_m = 1000 m.
FIT_BOX_L = (1.0, 1.5, 2.0, 4.0, 8.5)            # × the smallest cell side
FIT_BOX_V = (0.0, 100.0, 300.0, 1000.0)          # m
ANGLES_DEG = (0.0, 30.0, 45.0, 90.0, 135.0, 200.0, 315.0)


def _vec(norm: float, deg: float) -> tuple[float, float]:
    return norm * math.cos(math.radians(deg)), norm * math.sin(math.radians(deg))


def _unit_response(L, v, dx, dy, scheme, reach_m=None):
    """G for a unit source in the centre cell of a window that holds its
    support (≈ 4L across, plus the downwind drift); returns (G, h)."""
    reach_m = reach_m if reach_m is not None else 4.0 * L + 3.0 * math.hypot(*v)
    h = int(math.ceil(reach_m / min(dx, dy))) + 4
    q = np.zeros((2 * h + 1, 2 * h + 1))
    q[h, h] = 1.0
    return ops.solve(q, L, v, dx=dx, dy=dy, scheme=scheme), h


# ---------------------------------------------------------------------------
# a. Positivity
# ---------------------------------------------------------------------------


def test_fitted_diffusion_bounds_and_limits():
    d = DX
    for L in (30.0, 45.0, 120.0, 256.0):
        assert ops.fitted_diffusion(L, 0.0, d) == L**2
        for v in (1e-9, 0.3, 50.0, 100.0, 300.0, 1000.0):
            D = ops.fitted_diffusion(L, v, d)
            assert D == ops.fitted_diffusion(L, -v, d)                # even in v: adjoint is −v
            assert D >= abs(v) * d / 2.0 and D >= L**2               # off-diagonal weights ≤ 0
            pe = v * d / (2.0 * L**2)
            if pe < 0.05:
                assert D / L**2 == pytest.approx(1.0 + pe**2 / 3.0, rel=1e-6)
    # large Pe → first-order upwind; L = 0 is that limit
    assert ops.fitted_diffusion(1.0, 1000.0, d) == pytest.approx(1000.0 * d / 2.0, rel=1e-12)
    assert ops.fitted_diffusion(0.0, 1000.0, d) == 1000.0 * d / 2.0
    # the series branch joins x·coth(x) smoothly at its threshold
    x0 = ops._SERIES_PE
    below, above = float(ops._xcothx(x0 * (1 - 1e-9))), float(ops._xcothx(x0 * (1 + 1e-9)))
    assert abs(above - below) < 1e-12
    assert ops.axis_diffusion(60.0, (300.0, -30.0), 30.0, 20.0, "central") == (3600.0, 3600.0)
    # each axis is fitted with its own cell side: D_x from dx, D_y from dy
    assert ops.axis_diffusion(60.0, (300.0, -250.0), 30.0, 20.0, "exponential") == (
        ops.fitted_diffusion(60.0, 300.0, 30.0), ops.fitted_diffusion(60.0, -250.0, 20.0))
    assert ops.fitted_diffusion(60.0, -250.0, 20.0) < 0.9 * ops.fitted_diffusion(60.0, -250.0, 30.0)


def test_each_axis_is_fitted_with_its_own_cell_side():
    """solve, apply_operator and the torch twins against a hand-built operator
    with D_x = fitted(L, vx, dx) and D_y = fitted(L, vy, dy) at Pe_y ≈ 1 on a
    dx ≠ dy grid.  (Building D_y from dx instead stays consistent between
    solve and apply and second order, so only a direct check catches it.)"""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    L, v, dx, dy = 60.0, (120.0, 360.0), 30.0, 20.0          # Pe_x = 0.5 on dx, Pe_y = 1.0 on dy
    Dx, Dy = ops.fitted_diffusion(L, v[0], dx), ops.fitted_diffusion(L, v[1], dy)
    q = rng.standard_normal((24, 30))
    pad = ops.pad_cells(L, math.hypot(*v), dy)
    py, px = ops._padded_shape(*q.shape, pad)
    buf = np.zeros((py, px))
    buf[pad:pad + 24, pad:pad + 30] = q
    lam_x, lam_y, sx, sy = ops.symbols_axes(py, px, dy, dx)
    Ghat = 1.0 / (1.0 + Dx * lam_x + Dy * lam_y + 1j * (v[0] * sx + v[1] * sy))
    want = np.fft.irfft2(np.fft.rfft2(buf) * Ghat, s=(py, px))[pad:pad + 24, pad:pad + 30]
    got = ops.solve(q, L, v, dx=dx, dy=dy, pad=pad, scheme="exponential")
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    got = ops.solve_torch(torch.tensor(q), t(L), t(v[0]), t(v[1]), dx, dy, pad=pad, scheme="exponential").numpy()
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    d2x, d2y = ops.second_differences(q, dx, dy)
    gx, gy = ops.central_gradient(q, dx, dy)
    want = (q - Dx * d2x - Dy * d2y + v[0] * gx + v[1] * gy)[1:-1, 1:-1]
    got = ops.apply_operator(q, L, v, dx=dx, dy=dy, scheme="exponential")[1:-1, 1:-1]
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    got = ops.apply_operator_torch(torch.tensor(q), t(L), t(v[0]), t(v[1]), dx, dy, scheme="exponential").numpy()
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


@pytest.mark.parametrize("dx,dy", [(30.0, 30.0), (30.0, 20.0), (20.0, 30.0)])
def test_exponential_green_function_is_non_negative_over_the_fit_box(dx, dy):
    worst = (0.0, None)
    for f in FIT_BOX_L:
        L = f * min(dx, dy)
        for vn in FIT_BOX_V:
            for deg in (ANGLES_DEG if vn else (0.0,)):
                G, _ = _unit_response(L, _vec(vn, deg), dx, dy, "exponential")
                r = G.min() / G.max()
                if r < worst[0]:
                    worst = (r, (L, vn, deg))
    assert worst[0] >= -1e-10, worst


def test_central_scheme_has_negative_lobes_at_high_cell_peclet():
    """Regression record of the defect (plan F3): L = dx, |v| = 1 km (cell Péclet 16.7)."""
    G, _ = _unit_response(30.0, (1000.0, 0.0), DX, DX, "central")
    assert G.min() / G.max() < -0.5
    G, _ = _unit_response(30.0, (1000.0, 0.0), DX, DX, "exponential")
    assert G.min() / G.max() >= -1e-10


# ---------------------------------------------------------------------------
# b. Analytic oracle: the continuous Green's function (Wilson & Miller 1978)
# ---------------------------------------------------------------------------


def _green_exact(X, Y, L, v):
    """(1/2πL²)·exp(v·r/2L²)·K₀(|r|·√(1/L² + |v|²/4L⁴)), a density in m⁻²."""
    kappa = math.sqrt(1.0 / L**2 + (v[0] ** 2 + v[1] ** 2) / (4.0 * L**4))
    return np.exp((v[0] * X + v[1] * Y) / (2.0 * L**2)) * k0(kappa * np.hypot(X, Y)) / (2.0 * np.pi * L**2)


def _ring_error(L, v, dx, scheme):
    """Median relative error of G/dx² against the oracle on the ring 1.5L–2L."""
    G, h = _unit_response(L, v, dx, dx, scheme)
    c = np.arange(-h, h + 1) * dx
    X, Y = np.meshgrid(c, c)                         # axis 1 is x, axis 0 is y
    ring = (np.hypot(X, Y) >= 1.5 * L) & (np.hypot(X, Y) <= 2.0 * L)
    exact = _green_exact(X[ring], Y[ring], L, v)
    return float(np.median(np.abs(G[ring] / dx**2 - exact) / exact))


@pytest.mark.parametrize("scheme", ops.SCHEMES)
@pytest.mark.parametrize("v", [(300.0, 0.0), _vec(300.0, 45.0)], ids=["along_x", "diagonal"])
def test_green_function_converges_to_the_analytic_oracle_at_second_order(v, scheme):
    L = 256.0
    e30, e15 = _ring_error(L, v, 30.0, scheme), _ring_error(L, v, 15.0, scheme)
    assert e30 < 5e-3, e30
    assert e30 / e15 >= 3.0, (e30, e15)


# ---------------------------------------------------------------------------
# c. Method of manufactured solutions (Roache 2002)
# ---------------------------------------------------------------------------


def _mms_error(L, v, s, dx, dy, scheme):
    """max |solve(q) − φ| / max φ for the Gaussian φ = exp(−r²/2s²) and
    q = (1 − L²∇² + v·∇)φ applied analytically, on ±5s."""
    hx, hy = int(round(5.0 * s / dx)), int(round(5.0 * s / dy))
    X, Y = np.meshgrid(np.arange(-hx, hx + 1) * dx, np.arange(-hy, hy + 1) * dy)
    r2 = X**2 + Y**2
    phi = np.exp(-r2 / (2.0 * s * s))
    q = phi * (1.0 - L**2 * (r2 / s**4 - 2.0 / s**2) - (v[0] * X + v[1] * Y) / s**2)
    return float(np.max(np.abs(ops.solve(q, L, v, dx=dx, dy=dy, scheme=scheme) - phi)))


def _mms_case(L, v, s, dx, dy):
    """A parameter set whose id states its per-axis cell Péclet numbers |v_axis|·d/(2L²)."""
    pe_x, pe_y = abs(v[0]) * dx / (2.0 * L * L), abs(v[1]) * dy / (2.0 * L * L)
    return pytest.param(L, v, s, dx, dy, id=f"L{L:g}_dx{dx:g}_dy{dy:g}_Pe{pe_x:.2f},{pe_y:.2f}")


# Oblique wind with Pe ≈ 0.3–0.5; Pe 1.25 along x; Pe 1.25 along y on the
# dx ≠ dy grid (so D_y, built from dy, carries the fitted correction).
MMS_CASES = [_mms_case(90.0, (150.0, -260.0), 330.0, 30.0, 30.0), _mms_case(90.0, (150.0, -260.0), 330.0, 30.0, 20.0),
             _mms_case(60.0, (300.0, 0.0), 300.0, 30.0, 30.0), _mms_case(60.0, (300.0, 0.0), 300.0, 30.0, 20.0),
             _mms_case(60.0, (0.0, 450.0), 300.0, 30.0, 20.0)]


@pytest.mark.parametrize("scheme", ops.SCHEMES)
@pytest.mark.parametrize("L,v,s,dx,dy", MMS_CASES)
def test_manufactured_solution_converges_at_second_order(L, v, s, dx, dy, scheme):
    coarse = _mms_error(L, v, s, dx, dy, scheme)
    fine = _mms_error(L, v, s, dx / 2.0, dy / 2.0, scheme)
    assert coarse < 0.02
    assert 3.5 <= coarse / fine <= 4.5, (coarse, fine)


# ---------------------------------------------------------------------------
# d. Truncation of the zero padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,v", [(120.0, (300.0, -200.0)), (30.0, (1000.0, 0.0)), (256.0, (0.0, 0.0)),
                                 (60.0, (300.0, 300.0))])
def test_doubling_the_padding_changes_responses_by_under_one_percent(L, v):
    q = np.random.default_rng(4).standard_normal((40, 48))
    pad = ops.pad_cells(L, math.hypot(*v), DX)
    a = ops.solve(q, L, v, dx=DX)
    b = ops.solve(q, L, v, dx=DX, pad=2 * pad)
    assert np.max(np.abs(a - b)) < 0.01 * np.max(np.abs(b))


# ---------------------------------------------------------------------------
# e. Residual identity, torch twins, gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_apply_operator_inverts_solve_on_the_interior(scheme):
    q = np.random.default_rng(5).standard_normal((40, 52))
    L, v = 45.0, (400.0, -250.0)                      # cell Péclet > 1 on both axes
    phi = ops.solve(q, L, v, dx=30.0, dy=20.0, scheme=scheme)
    resid = ops.apply_operator(phi, L, v, dx=30.0, dy=20.0, scheme=scheme) - q
    assert np.all(np.isnan(resid[0])) and np.all(np.isnan(resid[:, -1]))
    assert np.max(np.abs(resid[1:-1, 1:-1])) < 1e-10


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_torch_twins_match_numpy(scheme):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    q = rng.standard_normal((30, 36))
    L, v, dx, dy = 45.0, (400.0, -250.0), 30.0, 20.0
    pad = ops.pad_cells(L, math.hypot(*v), dy)
    ref = ops.solve(q, L, v, dx=dx, dy=dy, pad=pad, scheme=scheme)
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    got = ops.solve_torch(torch.tensor(q), t(L), t(v[0]), t(v[1]), dx, dy, pad=pad, scheme=scheme).numpy()
    assert np.max(np.abs(got - ref)) < 1e-12
    phi = rng.standard_normal((20, 24))
    ref = ops.apply_operator(phi, L, v, dx=dx, dy=dy, scheme=scheme)[1:-1, 1:-1]
    for args in ((L, v[0], v[1]), (t(L), t(v[0]), t(v[1]))):           # floats and tensors
        got = ops.apply_operator_torch(torch.tensor(phi), *args, dx, dy, scheme=scheme).numpy()
        assert np.max(np.abs(got - ref)) < 1e-9
    D = ops.fitted_diffusion_torch(t(L), t(v[0]), dx)
    assert float(D) == pytest.approx(ops.fitted_diffusion(L, v[0], dx), rel=1e-14)


def test_gradients_are_finite_at_zero_wind_and_match_finite_differences():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    q = torch.tensor(rng.standard_normal((24, 28)))
    w = torch.tensor(rng.standard_normal((24, 28)))
    dx, dy, pad = 30.0, 20.0, ops.pad_cells(120.0, 600.0, 20.0)

    def loss(L, vx, vy):
        return (w * ops.solve_torch(q, L, vx, vy, dx, dy, pad=pad)).sum()

    def grads(L, vx, vy):
        P = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (L, vx, vy)]
        loss(*P).backward()
        return np.array([float(p.grad) for p in P])

    # exactly zero and tiny winds (series branch of Pe·coth Pe), and just past it
    for vx, vy in ((0.0, 0.0), (1e-9, -1e-9), (1e-3, 0.0), (0.0, 30.0)):
        g = grads(60.0, vx, vy)
        assert np.all(np.isfinite(g)), (vx, vy, g)
    # D(L, v) itself: dD/dv = 0 and dD/dL = 2L at v = 0
    L = torch.tensor(60.0, dtype=torch.float64, requires_grad=True)
    v0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    ops.fitted_diffusion_torch(L, v0, dx).backward()
    assert float(v0.grad) == 0.0 and float(L.grad) == pytest.approx(120.0, rel=1e-14)
    # autograd = central finite differences at a strong, oblique wind
    x0 = np.array([60.0, 350.0, -220.0])
    g = grads(*x0)
    h = 1e-3
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        with torch.no_grad():
            fd = (float(loss(*(x0 + e))) - float(loss(*(x0 - e)))) / (2.0 * h)
        assert g[i] == pytest.approx(fd, rel=1e-6, abs=1e-9), (i, g[i], fd)
    # the stacker's operator is differentiable at v = 0 too
    phi = torch.tensor(rng.standard_normal((20, 24)))
    P = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (60.0, 0.0, 0.0)]
    ops.apply_operator_torch(phi, *P, dx, dy).pow(2).sum().backward()
    assert all(torch.isfinite(p.grad) for p in P)


# ---------------------------------------------------------------------------
# f. Adjoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ops.SCHEMES)
def test_adjoint_is_the_operator_with_reversed_wind(scheme):
    rng = np.random.default_rng(8)
    a, b = rng.standard_normal((30, 36)), rng.standard_normal((30, 36))
    L, v, dx, dy = 60.0, (300.0, -150.0), 30.0, 20.0
    pad = ops.pad_cells(L, math.hypot(*v), dy)
    lhs = np.sum(b * ops.solve(a, L, v, dx=dx, dy=dy, pad=pad, scheme=scheme))
    rhs = np.sum(a * ops.solve(b, L, (-v[0], -v[1]), dx=dx, dy=dy, pad=pad, scheme=scheme))
    assert lhs == pytest.approx(rhs, rel=1e-12)


# ---------------------------------------------------------------------------
# g. Downwind shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,v", [(60.0, (300.0, 0.0)), (30.0, _vec(150.0, 45.0)), (120.0, (0.0, -600.0)),
                                 (30.0, (1000.0, 0.0))])
def test_green_function_centroid_sits_v_downwind(L, v):
    """The first moment of G is v for the discrete operator (λ′(0) = 0,
    adv′(0) = v); the window holds ≈ 10 downwind decay lengths."""
    G, h = _unit_response(L, v, DX, DX, "exponential", reach_m=6.0 * L + 10.0 * math.hypot(*v))
    c = np.arange(-h, h + 1)
    X, Y = np.meshgrid(c, c)
    cx, cy = float((X * G).sum() / G.sum()), float((Y * G).sum() / G.sum())
    target = np.array(v) / DX                          # cells
    assert np.hypot(cx - target[0], cy - target[1]) <= 0.1 * np.hypot(*target)


# ---------------------------------------------------------------------------
# What the fitted scheme costs: accuracy, and L above a cell Péclet number of 1
# ---------------------------------------------------------------------------


def _mass_l1_error(L, v, dx, scheme):
    """Mass-weighted L1 error of G/dx² against the oracle beyond two cells of the source."""
    G, h = _unit_response(L, v, dx, dx, scheme)
    c = np.arange(-h, h + 1) * dx
    X, Y = np.meshgrid(c, c)
    far = np.hypot(X, Y) > 2.0 * dx
    exact = _green_exact(X[far], Y[far], L, v)
    return float(np.abs(G[far] / dx**2 - exact).sum() / exact.sum())


def test_positivity_is_not_bought_with_accuracy_at_small_peclet_but_is_at_moderate_peclet():
    """The module docstring's trade-off: at Pe = 0.07 the schemes are equally
    accurate (mass-weighted), at Pe = 1.25 the central scheme is more accurate."""
    small = [_mass_l1_error(256.0, (300.0, 0.0), DX, s) for s in ("exponential", "central")]
    assert small[0] < 3e-3 and small[1] < 3e-3 and abs(small[0] - small[1]) < 2e-4, small
    moderate = [_mass_l1_error(60.0, (300.0, 0.0), DX, s) for s in ("exponential", "central")]
    assert moderate[1] < moderate[0] < 0.06, moderate


def test_grid_resolution_reports_cell_peclet_and_effective_lengths():
    r = ops.grid_resolution(30.0, (1000.0, 0.0), DX)                 # the F3 worst case, wind along x
    assert r["cell_peclet_x"] == pytest.approx(1000.0 * DX / (2.0 * 900.0)) and r["cell_peclet_y"] == 0.0
    assert r["L_eff_x_m"] == pytest.approx(math.sqrt(ops.fitted_diffusion(30.0, 1000.0, DX)), rel=1e-14)
    assert r["L_eff_x_m"] == pytest.approx(122.47, abs=0.01) and r["L_eff_y_m"] == 30.0
    assert r["L_min_resolved_m"] == pytest.approx(math.sqrt(1000.0 * DX / 2.0)) and r["L_resolved"] is False
    c = ops.grid_resolution(30.0, (1000.0, 0.0), DX, scheme="central")
    assert (c["L_eff_x_m"], c["L_eff_y_m"]) == (30.0, 30.0) and c["L_resolved"] is False
    z = ops.grid_resolution(30.0, (0.0, 0.0), DX)
    assert (z["cell_peclet_x"], z["L_eff_x_m"], z["L_min_resolved_m"], z["L_resolved"]) == (0.0, 30.0, 0.0, True)
    # per-axis cell sides, and L_min_resolved_m is the Pe = 1 boundary
    r = ops.grid_resolution(60.0, (300.0, -300.0), 30.0, 20.0)
    assert r["cell_peclet_x"] == pytest.approx(1.25) and r["cell_peclet_y"] == pytest.approx(300.0 * 20.0 / 7200.0)
    Lb = r["L_min_resolved_m"]
    assert ops.grid_resolution(Lb * (1 + 1e-9), (300.0, -300.0), 30.0, 20.0)["L_resolved"] is True
    assert ops.grid_resolution(Lb * (1 - 1e-9), (300.0, -300.0), 30.0, 20.0)["L_resolved"] is False


def test_above_cell_peclet_one_the_grid_not_L_sets_the_fitted_diffusion():
    """Oblique wind of |v| = 1 km on 30 m cells: both axes have Pe > 1 at
    L = 30–45 m, and the fitted operator is the same for both L (L is not
    identified there), while the central one differs (through its negative
    lobes).  At L ≥ L_min_resolved_m (≈ 103 m here) L matters again."""
    q = np.random.default_rng(11).standard_normal((40, 40))
    v = (707.1, 707.1)

    def change(L1, L2, scheme):
        a, b = (ops.solve(q, L, v, dx=DX, scheme=scheme) for L in (L1, L2))
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    assert ops.grid_resolution(45.0, v, DX)["L_resolved"] is False
    assert change(30.0, 45.0, "exponential") < 1e-3
    assert change(30.0, 45.0, "central") > 0.2
    assert ops.grid_resolution(150.0, v, DX)["L_resolved"] is True
    assert change(150.0, 225.0, "exponential") > 0.2


# ---------------------------------------------------------------------------
# h. Legacy central scheme and scheme names
# ---------------------------------------------------------------------------


def _legacy_solve(q, L, v, dx, dy=None, pad=None):
    """The pre-W1 ``operators.solve`` (5-point Laplacian, central advection), verbatim."""
    dy = dx if dy is None else dy
    ny, nx = q.shape
    vx, vy = float(v[0]), float(v[1])
    if pad is None:
        pad = ops.pad_cells(L, math.hypot(vx, vy), min(dx, dy))
    py, px = ops._padded_shape(ny, nx, pad)
    buf = np.zeros((py, px), dtype=np.float64)
    buf[pad:pad + ny, pad:pad + nx] = q
    kx = 2.0 * np.pi * np.fft.rfftfreq(px, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(py, d=dy)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    lam = (4.0 / dx**2) * np.sin(KX * dx / 2.0) ** 2 + (4.0 / dy**2) * np.sin(KY * dy / 2.0) ** 2
    sx = np.sin(KX * dx) / dx
    sy = np.sin(KY * dy) / dy
    G = 1.0 / (1.0 + (L**2) * lam + 1j * (vx * sx + vy * sy))
    return np.fft.irfft2(np.fft.rfft2(buf) * G, s=(py, px))[pad:pad + ny, pad:pad + nx]


def test_central_scheme_reproduces_the_legacy_operator_exactly():
    rng = np.random.default_rng(9)
    q = rng.standard_normal((40, 52))
    for L, v, dy in ((150.0, (90.0, -45.0), None), (30.0, (1000.0, 0.0), None), (90.0, (30.0, -10.0), 20.0)):
        assert np.array_equal(ops.solve(q, L, v, dx=30.0, dy=dy, scheme="central"), _legacy_solve(q, L, v, 30.0, dy))
        legacy_apply = q - (L**2) * ops.laplacian(q, 30.0, dy) + v[0] * ops.central_gradient(q, 30.0, dy)[0] \
            + v[1] * ops.central_gradient(q, 30.0, dy)[1]
        assert np.array_equal(ops.apply_operator(q, L, v, dx=30.0, dy=dy, scheme="central"), legacy_apply,
                              equal_nan=True)
    lam, sx, sy = ops.symbols(64, 48, 20.0, 30.0)
    lam_x, lam_y, sx2, sy2 = ops.symbols_axes(64, 48, 20.0, 30.0)
    assert np.array_equal(lam, lam_x + lam_y) and np.array_equal(sx, sx2) and np.array_equal(sy, sy2)
    # at v = 0 the schemes coincide (D = L²)
    a = ops.solve(q, 120.0, (0.0, 0.0), dx=30.0, scheme="exponential")
    assert np.max(np.abs(a - ops.solve(q, 120.0, (0.0, 0.0), dx=30.0, scheme="central"))) < 1e-13
    assert ops.green_centre(0.0, (0.0, 0.0), 30.0) == pytest.approx(1.0)


def test_central_torch_twins_reproduce_the_legacy_formulas_exactly():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    q = torch.tensor(rng.standard_normal((24, 30)))
    L, vx, vy = (torch.tensor(x, dtype=torch.float64) for x in (120.0, 40.0, -25.0))
    pad = 12
    py, px = ops._padded_shape(24, 30, pad)
    buf = q.new_zeros((py, px))
    buf[pad:pad + 24, pad:pad + 30] = q
    lam, sx, sy = (torch.as_tensor(a) for a in ops.symbols(py, px, 30.0, 30.0))
    legacy = torch.fft.irfft2(torch.fft.rfft2(buf) / torch.complex(1.0 + (L**2) * lam, vx * sx + vy * sy),
                              s=(py, px))[pad:pad + 24, pad:pad + 30]
    assert torch.equal(ops.solve_torch(q, L, vx, vy, 30.0, pad=pad, scheme="central"), legacy)
    c = q[1:-1, 1:-1]
    lap = (q[1:-1, 2:] - 2.0 * c + q[1:-1, :-2]) / 30.0**2 + (q[2:, 1:-1] - 2.0 * c + q[:-2, 1:-1]) / 30.0**2
    gx, gy = (q[1:-1, 2:] - q[1:-1, :-2]) / (2.0 * 30.0), (q[2:, 1:-1] - q[:-2, 1:-1]) / (2.0 * 30.0)
    legacy = c - (L**2) * lap + vx * gx + vy * gy
    assert torch.equal(ops.apply_operator_torch(q, L, vx, vy, 30.0, scheme="central"), legacy)


def test_params_scheme_falls_back_to_the_legacy_central_scheme():
    """Callers that re-apply a fitted operator resolve its scheme from the
    model's params; a dict without the key is a run fitted before the scheme
    existed (central), never the current default."""
    assert ops.LEGACY_SCHEME == "central" != ops.DEFAULT_SCHEME
    for old in ({}, None, {"advection_scheme": None}, {"L_m": 45.0, "vx_m": 400.0, "vy_m": -250.0}):
        assert ops.params_scheme(old) == "central"
    assert ops.params_scheme({"advection_scheme": "Exponential"}) == "exponential"
    assert ops.params_scheme({"advection_scheme": "central"}) == "central"
    with pytest.raises(ValueError, match="advection scheme"):
        ops.params_scheme({"advection_scheme": "upwind"})


def test_unknown_scheme_is_rejected():
    q = np.zeros((8, 8))
    assert ops.check_scheme("Exponential") == "exponential" and ops.DEFAULT_SCHEME in ops.SCHEMES
    calls = [lambda: ops.check_scheme("upwind"),
             lambda: ops.solve(q, 60.0, (10.0, 0.0), scheme="upwind"),
             lambda: ops.apply_operator(q, 60.0, (10.0, 0.0), scheme="upwind"),
             lambda: ops.axis_diffusion(60.0, (10.0, 0.0), 30.0, scheme="upwind"),
             lambda: ops.green_centre(60.0, (10.0, 0.0), 30.0, scheme="upwind"),
             lambda: ops.green_mass(np.ones((8, 8)), 60.0, (10.0, 0.0), 30.0, scheme="upwind")]
    for call in calls:
        with pytest.raises(ValueError, match="advection scheme"):
            call()
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="advection scheme"):
        ops.solve_torch(torch.zeros((8, 8), dtype=torch.float64), 60.0, 0.0, 0.0, 30.0, scheme="upwind")
    with pytest.raises(ValueError, match="advection scheme"):
        ops.apply_operator_torch(torch.zeros((8, 8), dtype=torch.float64), 60.0, 0.0, 0.0, 30.0, scheme="upwind")
