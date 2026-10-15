"""Numerical verification of the advection–diffusion–relaxation operator
(physics plan W1): positivity of the hybrid and exponentially fitted schemes
over the fit box, convergence to the analytic K₀ Green's function,
manufactured solutions, padding truncation, the residual and adjoint
identities, the downwind shift, the torch twins and their derivatives, the
legacy central scheme (and the hybrid scheme's identity with it below a cell
Péclet number of 1), and what positivity costs: accuracy at moderate cell
Péclet numbers, the bias of the fitted L on continuum data, and how much L
still matters above Pe = 1 (``grid_resolution``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import k0

from sparc.core import operators as ops

DX = 30.0
POSITIVE = ("hybrid", "exponential")              # the schemes that keep G ≥ 0
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


def test_hybrid_diffusion_is_central_up_to_peclet_one_and_upwind_beyond():
    d = DX
    for L in (30.0, 45.0, 120.0, 256.0):
        for v in (0.0, 1e-9, 50.0, 100.0, 300.0, 1000.0):
            D, pe = ops.hybrid_diffusion(L, v, d), v * d / (2.0 * L**2)
            assert D == ops.hybrid_diffusion(L, -v, d)                # even in v: adjoint is −v
            assert D == (L**2 if pe <= 1.0 else v * d / 2.0)          # max(L², |v|·d/2), Patankar's hybrid
            assert D <= ops.fitted_diffusion(L, v, d)                 # never more diffusion than exponential fitting
    # the excess over L² is exactly 0 for central, and for hybrid on every axis with Pe ≤ 1
    assert ops.axis_excess(60.0, (300.0, -240.0), 30.0, 20.0, "central") == (0.0, 0.0)
    assert ops.axis_excess(60.0, (240.0, -240.0), 30.0, 30.0, "hybrid") == (0.0, 0.0)          # Pe = 1 exactly
    assert ops.axis_excess(60.0, (300.0, -240.0), 30.0, 20.0, "hybrid") == (300.0 * 15.0 - 3600.0, 0.0)
    assert ops.axis_diffusion(60.0, (300.0, -450.0), 30.0, 20.0, "hybrid") == (4500.0, 4500.0)  # D_y from dy


@pytest.mark.parametrize("scheme", POSITIVE)
def test_each_axis_is_fitted_with_its_own_cell_side(scheme):
    """solve, apply_operator and the torch twins against a hand-built operator
    with D_x = D(L, vx, dx) and D_y = D(L, vy, dy) at Pe_y = 1.25 on a
    dx ≠ dy grid (1.9 if D_y were built from dx).  (Building D_y from dx
    instead stays consistent between solve and apply and second order, so
    only a direct check catches it.)"""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    L, v, dx, dy = 60.0, (120.0, 450.0), 30.0, 20.0          # Pe_x = 0.5 on dx, Pe_y = 1.25 on dy
    f = ops.hybrid_diffusion if scheme == "hybrid" else ops.fitted_diffusion
    Dx, Dy = f(L, v[0], dx), f(L, v[1], dy)
    assert Dy > L**2
    q = rng.standard_normal((24, 30))
    pad = ops.pad_cells(L, math.hypot(*v), dy)
    py, px = ops._padded_shape(*q.shape, pad)
    buf = np.zeros((py, px))
    buf[pad:pad + 24, pad:pad + 30] = q
    lam_x, lam_y, sx, sy = ops.symbols_axes(py, px, dy, dx)
    Ghat = 1.0 / (1.0 + Dx * lam_x + Dy * lam_y + 1j * (v[0] * sx + v[1] * sy))
    want = np.fft.irfft2(np.fft.rfft2(buf) * Ghat, s=(py, px))[pad:pad + 24, pad:pad + 30]
    got = ops.solve(q, L, v, dx=dx, dy=dy, pad=pad, scheme=scheme)
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    got = ops.solve_torch(torch.tensor(q), t(L), t(v[0]), t(v[1]), dx, dy, pad=pad, scheme=scheme).numpy()
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    d2x, d2y = ops.second_differences(q, dx, dy)
    gx, gy = ops.central_gradient(q, dx, dy)
    want = (q - Dx * d2x - Dy * d2y + v[0] * gx + v[1] * gy)[1:-1, 1:-1]
    got = ops.apply_operator(q, L, v, dx=dx, dy=dy, scheme=scheme)[1:-1, 1:-1]
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))
    got = ops.apply_operator_torch(torch.tensor(q), t(L), t(v[0]), t(v[1]), dx, dy, scheme=scheme).numpy()
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


@pytest.mark.parametrize("scheme", POSITIVE)
@pytest.mark.parametrize("dx,dy", [(30.0, 30.0), (30.0, 20.0), (20.0, 30.0)])
def test_green_function_is_non_negative_over_the_fit_box(dx, dy, scheme):
    worst = (0.0, None)
    for f in FIT_BOX_L:
        L = f * min(dx, dy)
        for vn in FIT_BOX_V:
            for deg in (ANGLES_DEG if vn else (0.0,)):
                G, _ = _unit_response(L, _vec(vn, deg), dx, dy, scheme)
                r = G.min() / G.max()
                if r < worst[0]:
                    worst = (r, (L, vn, deg))
    assert worst[0] >= -1e-10, worst


def test_central_scheme_has_negative_lobes_at_high_cell_peclet():
    """Regression record of the defect (plan F3): L = dx, |v| = 1 km (cell Péclet 16.7)."""
    G, _ = _unit_response(30.0, (1000.0, 0.0), DX, DX, "central")
    assert G.min() / G.max() < -0.5
    for scheme in POSITIVE:
        G, _ = _unit_response(30.0, (1000.0, 0.0), DX, DX, scheme)
        assert G.min() / G.max() >= -1e-10, scheme


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
    pe = max(abs(v[0]) * dx, abs(v[1]) * dy) / (2.0 * L * L)
    if scheme == "hybrid" and pe > 1.0:
        # The coarse grid upwinds the axis past Pe = 1 (first order there); halving the cells brings
        # every Pe ≤ 1, where the hybrid scheme is the central one, so the error drops by more than 4.
        assert coarse / fine >= 3.5 and fine == _mms_error(L, v, s, dx / 2.0, dy / 2.0, "central")
    else:
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
    assert float(ops.hybrid_diffusion_torch(t(L), t(v[0]), dx)) == ops.hybrid_diffusion(L, v[0], dx)


def test_torch_diffusion_twins_match_numpy_values_and_analytic_derivatives():
    """Values against x·coth x evaluated directly (a few ulp for x > 0) and
    autograd against the analytic derivatives, on both sides of the series
    threshold of Pe·coth(Pe) (1e-2) and well past it.  With f(x) = x·coth x,
    D = L²·f(Pe), Pe = |v|·d/(2L²): dD/dv = (d/2)·f′(Pe)·sign(v) and
    dD/dL = 2L·(f − Pe·f′), f′(x) = coth x − x/sinh²x (2x/3 − 4x³/45 +
    12x⁵/945 + O(x⁷) for small x)."""
    torch = pytest.importorskip("torch")
    L, d = 60.0, 30.0
    for pe in (0.0, 1e-4, 5e-3, 0.99e-2, 1.01e-2, 0.3, 0.6, 2.0):
        v = pe * 2.0 * L * L / d
        f = pe / math.tanh(pe) if pe else 1.0
        if pe < 0.05:
            fp = 2 * pe / 3 - 4 * pe**3 / 45 + 12 * pe**5 / 945
        else:
            fp = 1 / math.tanh(pe) - pe / math.sinh(pe) ** 2
        assert ops.fitted_diffusion(L, v, d) == pytest.approx(L * L * f, rel=1e-13), pe
        Lt, vt = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (L, v))
        D = ops.fitted_diffusion_torch(Lt, vt, d)
        D.backward()
        assert float(D.detach()) == pytest.approx(L * L * f, rel=1e-13), pe
        assert float(vt.grad) == pytest.approx(d / 2.0 * fp, rel=1e-9, abs=1e-15), pe
        assert float(Lt.grad) == pytest.approx(2.0 * L * (f - pe * fp), rel=1e-12), pe
        # hybrid: L² (dD/dL = 2L, dD/dv = 0) up to Pe = 1, |v|·d/2 (dD/dL = 0, dD/dv = d/2) beyond
        Lt, vt = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (L, v))
        D = ops.hybrid_diffusion_torch(Lt, vt, d)
        D.backward()
        assert float(D.detach()) == ops.hybrid_diffusion(L, v, d)
        assert (float(Lt.grad), float(vt.grad)) == ((2.0 * L, 0.0) if pe < 1.0 else (0.0, d / 2.0)), pe


@pytest.mark.parametrize("scheme", POSITIVE)
def test_gradients_are_finite_at_zero_wind_and_match_finite_differences(scheme):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    q = torch.tensor(rng.standard_normal((24, 28)))
    w = torch.tensor(rng.standard_normal((24, 28)))
    dx, dy, pad = 30.0, 20.0, ops.pad_cells(120.0, 600.0, 20.0)

    def loss(L, vx, vy):
        return (w * ops.solve_torch(q, L, vx, vy, dx, dy, pad=pad, scheme=scheme)).sum()

    def grads(L, vx, vy):
        P = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (L, vx, vy)]
        loss(*P).backward()
        return np.array([float(p.grad) for p in P])

    # exactly zero and tiny winds (series branch of Pe·coth Pe), and just past it
    for vx, vy in ((0.0, 0.0), (1e-9, -1e-9), (1e-3, 0.0), (0.0, 30.0)):
        g = grads(60.0, vx, vy)
        assert np.all(np.isfinite(g)), (vx, vy, g)
    # autograd = central finite differences at a strong, oblique wind (Pe 1.46 on x, 0.61 on y)
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
    ops.apply_operator_torch(phi, *P, dx, dy, scheme=scheme).pow(2).sum().backward()
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


@pytest.mark.parametrize("scheme", POSITIVE)
@pytest.mark.parametrize("L,v", [(60.0, (300.0, 0.0)), (30.0, _vec(150.0, 45.0)), (120.0, (0.0, -600.0)),
                                 (30.0, (1000.0, 0.0))])
def test_green_function_centroid_sits_v_downwind(L, v, scheme):
    """The first moment of G is v for the discrete operator (λ′(0) = 0,
    adv′(0) = v); the window holds ≈ 10 downwind decay lengths."""
    G, h = _unit_response(L, v, DX, DX, scheme, reach_m=6.0 * L + 10.0 * math.hypot(*v))
    c = np.arange(-h, h + 1)
    X, Y = np.meshgrid(c, c)
    cx, cy = float((X * G).sum() / G.sum()), float((Y * G).sum() / G.sum())
    target = np.array(v) / DX                          # cells
    assert np.hypot(cx - target[0], cy - target[1]) <= 0.1 * np.hypot(*target)


# ---------------------------------------------------------------------------
# What positivity costs: accuracy, the fitted L, and L above a cell Péclet number of 1
# ---------------------------------------------------------------------------


def _mass_l1_error(L, v, dx, scheme):
    """Mass-weighted L1 error of G/dx² against the oracle beyond two cells of the source."""
    G, h = _unit_response(L, v, dx, dx, scheme)
    c = np.arange(-h, h + 1) * dx
    X, Y = np.meshgrid(c, c)
    far = np.hypot(X, Y) > 2.0 * dx
    exact = _green_exact(X[far], Y[far], L, v)
    return float(np.abs(G[far] / dx**2 - exact).sum() / exact.sum())


def test_positivity_costs_accuracy_only_where_central_is_not_positive():
    """The module docstring's numbers: at Pe = 0.07 the schemes are equally
    accurate (mass-weighted); at Pe = 0.88 on both axes the hybrid scheme is
    the central one and the exponential one is about twice as far off; at
    Pe = 1.25 central < hybrid < exponential."""
    small = {s: _mass_l1_error(256.0, (300.0, 0.0), DX, s) for s in ops.SCHEMES}
    assert max(small.values()) < 3e-3 and max(small.values()) - min(small.values()) < 2e-4, small
    oblique = {s: _mass_l1_error(60.0, (212.1, 212.1), DX, s) for s in ops.SCHEMES}
    assert oblique["hybrid"] == oblique["central"] < 0.5 * oblique["exponential"], oblique
    moderate = {s: _mass_l1_error(60.0, (300.0, 0.0), DX, s) for s in ops.SCHEMES}
    assert moderate["central"] < moderate["hybrid"] < moderate["exponential"] < 0.06, moderate


def _continuum_solve(q, L, v, dx, pad=200):
    """The *continuous* operator's solution for the trigonometric interpolant of
    ``q`` (exact for a band-limited source): Ĝ = 1/(1 + L²|k|² + i·v·k) with the
    exact wavenumbers, on a zero-padded grid."""
    ny, nx = q.shape
    py, px = ops._padded_shape(ny, nx, pad)
    buf = np.zeros((py, px))
    buf[pad:pad + ny, pad:pad + nx] = q
    kx = 2.0 * np.pi * np.fft.rfftfreq(px, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(py, d=dx)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    G = 1.0 / (1.0 + L**2 * (KX**2 + KY**2) + 1j * (v[0] * KX + v[1] * KY))
    return np.fft.irfft2(np.fft.rfft2(buf) * G, s=(py, px))[pad:pad + ny, pad:pad + nx]


@pytest.fixture(scope="module")
def smooth_source():
    """40 Gaussians of σ = 60 m (2 cells; spectrum ≈ 3e-9 at Nyquist, so the
    sampled source is band-limited) at least 4σ inside a 48 × 48 grid."""
    rng = np.random.default_rng(5)
    n, sig, margin = 48, 60.0, 8 * DX
    centres = rng.uniform(margin, n * DX - margin, size=(40, 2))
    amp = rng.standard_normal(40)
    X, Y = np.meshgrid(np.arange(n) * DX, np.arange(n) * DX)
    return sum(a * np.exp(-((X - u) ** 2 + (Y - w) ** 2) / (2.0 * sig * sig)) for a, (u, w) in zip(amp, centres))


@pytest.mark.parametrize("L,v,exp_ratio_max", [(60.0, (212.1, 212.1), 0.85), (90.0, (300.0, 300.0), 0.97)],
                         ids=["L60_Pe0.88_oblique", "L90_Pe0.56_oblique"])
def test_fit_recovers_the_continuum_L_with_the_default_scheme(smooth_source, L, v, exp_ratio_max):
    """Data from the continuous operator (no inverse crime), sampled on 30 m
    cells, at cell Péclet numbers below 1 with an oblique wind.  A fit pins
    the diffusion D, not L: the hybrid scheme (D = L² here, like central)
    recovers L within a few percent, from a start well below it (where the
    hybrid operator is flat in L, so the fit must start on the central one),
    while the exponential scheme's extra diffusion L²·(Pe·coth Pe − 1) makes
    its L run low (the reason it is not the default)."""
    from sparc.core.physics import fit_operator

    y = 2.0 * _continuum_solve(smooth_source, L, v, DX) + 0.5
    mask = np.ones(y.shape, dtype=bool)

    def fit(scheme):
        return fit_operator(smooth_source, y, mask, DX, L_init=0.6 * L, v_max=1000.0, v_penalty=0.0, max_iter=80,
                            scheme=scheme)

    h = fit("hybrid")
    assert abs(h["L_m"] / L - 1.0) < 0.03, h["L_m"]
    assert math.hypot(h["vx_m"] - v[0], h["vy_m"] - v[1]) < 0.05 * math.hypot(*v)
    assert h["L_resolved"] is True and h["resolution_warning"] is None
    e = fit("exponential")
    assert e["L_m"] < exp_ratio_max * L, e["L_m"]
    if L == 60.0:                     # below Pe = 1 the hybrid fit is the central fit, bit for bit
        c = fit("central")
        assert (c["L_m"], c["vx_m"], c["vy_m"], c["rmse"]) == (h["L_m"], h["vx_m"], h["vy_m"], h["rmse"])


def test_grid_resolution_reports_cell_peclet_effective_lengths_and_elasticities():
    r = ops.grid_resolution(30.0, (1000.0, 0.0), DX)                 # the F3 worst case, wind along x (hybrid)
    assert r["cell_peclet_x"] == pytest.approx(1000.0 * DX / (2.0 * 900.0)) and r["cell_peclet_y"] == 0.0
    assert r["L_eff_x_m"] == math.sqrt(1000.0 * DX / 2.0) and r["L_eff_y_m"] == 30.0
    assert (r["L_elasticity_x"], r["L_elasticity_y"]) == (0.0, 2.0)
    assert r["L_min_resolved_m"] == pytest.approx(math.sqrt(1000.0 * DX / 2.0)) and r["L_resolved"] is False
    e = ops.grid_resolution(30.0, (1000.0, 0.0), DX, scheme="exponential")
    assert e["L_eff_x_m"] == pytest.approx(math.sqrt(ops.fitted_diffusion(30.0, 1000.0, DX)), rel=1e-14)
    assert e["L_eff_x_m"] == pytest.approx(122.47, abs=0.01) and e["L_elasticity_x"] < 1e-12
    c = ops.grid_resolution(30.0, (1000.0, 0.0), DX, scheme="central")
    assert (c["L_eff_x_m"], c["L_eff_y_m"], c["L_elasticity_x"]) == (30.0, 30.0, 2.0) and c["L_resolved"] is False
    z = ops.grid_resolution(30.0, (0.0, 0.0), DX)
    assert (z["cell_peclet_x"], z["L_eff_x_m"], z["L_min_resolved_m"], z["L_resolved"]) == (0.0, 30.0, 0.0, True)
    # per-axis cell sides, and L_min_resolved_m is the Pe = 1 boundary
    r = ops.grid_resolution(60.0, (300.0, -300.0), 30.0, 20.0)
    assert r["cell_peclet_x"] == pytest.approx(1.25) and r["cell_peclet_y"] == pytest.approx(300.0 * 20.0 / 7200.0)
    Lb = r["L_min_resolved_m"]
    assert ops.grid_resolution(Lb * (1 + 1e-9), (300.0, -300.0), 30.0, 20.0)["L_resolved"] is True
    assert ops.grid_resolution(Lb * (1 - 1e-9), (300.0, -300.0), 30.0, 20.0)["L_resolved"] is False
    # the exponential elasticity 4Pe/sinh(2Pe) at the docstring's points, and against D itself
    for pe, want in ((1.0, 1.1), (1.5, 0.6), (2.0, 0.3), (3.0, 0.06)):
        L = 60.0
        el = ops.grid_resolution(L, (pe * 2.0 * L * L / DX, 0.0), DX, scheme="exponential")["L_elasticity_x"]
        assert el == pytest.approx(want, abs=0.01 if pe < 3 else 0.001), pe
        D = lambda L_: ops.fitted_diffusion(L_, pe * 2.0 * L * L / DX, DX)  # noqa: E731
        h = 1e-5
        fd = (math.log(D(L * (1 + h))) - math.log(D(L * (1 - h)))) / (math.log(1 + h) - math.log(1 - h))
        assert el == pytest.approx(fd, rel=1e-6), pe


def test_above_cell_peclet_one_L_loses_influence_on_the_positive_schemes():
    """Oblique wind of |v| = 1 km on 30 m cells.  At L = 30–45 m (Pe 5–12 on
    both axes) the hybrid operator does not depend on L at all and the
    exponential one hardly does, while the central one does (through its
    negative lobes).  At L ≥ L_min_resolved_m (≈ 103 m) L matters again."""
    q = np.random.default_rng(11).standard_normal((40, 40))
    v = (707.1, 707.1)
    pad = ops.pad_cells(225.0, math.hypot(*v), DX)                   # the same padding for every L compared

    def change(L1, L2, scheme):
        a, b = (ops.solve(q, L, v, dx=DX, pad=pad, scheme=scheme) for L in (L1, L2))
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    assert ops.grid_resolution(45.0, v, DX)["L_resolved"] is False
    assert change(30.0, 45.0, "hybrid") < 1e-12
    assert change(30.0, 45.0, "exponential") < 1e-3
    assert change(30.0, 45.0, "central") > 0.2
    assert ops.grid_resolution(150.0, v, DX)["L_resolved"] is True
    for scheme in ops.SCHEMES:
        assert change(150.0, 225.0, scheme) > 0.2, scheme


def test_just_above_cell_peclet_one_the_exponential_L_is_weakly_not_un_identified():
    """Pe = 1.2 on both axes (same wind): a 10 % change of L moves the
    exponential solve almost as much as at the resolved boundary (Pe = 1), as
    the elasticity 4Pe/sinh(2Pe) says (0.88 vs 1.10); the hybrid operator does
    not move at all there."""
    q = np.random.default_rng(11).standard_normal((40, 40))
    v = (707.1, 707.1)
    half = v[0] * DX / 2.0
    pad = ops.pad_cells(120.0, math.hypot(*v), DX)                   # the same padding for every L compared

    def change(L, scheme, f=1.1):
        a, b = (ops.solve(q, x, v, dx=DX, pad=pad, scheme=scheme) for x in (L, f * L))
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    L1, L12 = math.sqrt(half), math.sqrt(half / 1.2)              # Pe = 1 and 1.2
    at_boundary, above = change(L1, "exponential"), change(L12, "exponential")
    assert above > 0.75 * at_boundary > 0.04, (above, at_boundary)
    r = ops.grid_resolution(L12, v, DX, scheme="exponential")
    assert r["L_elasticity_x"] == pytest.approx(4.0 * 1.2 / math.sinh(2.4)) == r["L_elasticity_y"]
    assert r["L_resolved"] is False
    assert change(L12, "hybrid", f=0.9) < 1e-12                    # Pe 1.2 → 1.48 on both axes: flat in L


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


def test_hybrid_is_bit_identical_to_central_up_to_cell_peclet_one():
    """Where every Pe ≤ 1 the hybrid scheme adds nothing, so solve, apply and
    the torch twins (values and gradients) are the central ones bit for bit:
    runs whose winds stay below Pe = 1 (Providence: ≤ 0.23) do not move."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    q = rng.standard_normal((30, 36))
    for L, v, dy in ((150.0, (90.0, -45.0), None), (60.0, (240.0, -240.0), None), (90.0, (300.0, -10.0), 20.0)):
        assert max(abs(v[0]) * 30.0, abs(v[1]) * (dy or 30.0)) / (2 * L * L) <= 1.0
        assert np.array_equal(ops.solve(q, L, v, dx=30.0, dy=dy, scheme="hybrid"),
                              ops.solve(q, L, v, dx=30.0, dy=dy, scheme="central"))
        assert np.array_equal(ops.apply_operator(q, L, v, dx=30.0, dy=dy, scheme="hybrid"),
                              ops.apply_operator(q, L, v, dx=30.0, dy=dy, scheme="central"), equal_nan=True)
        out = {}
        for scheme in ("hybrid", "central"):
            P = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (L, *v)]
            phi = ops.solve_torch(torch.tensor(q), *P, 30.0, dy, pad=12, scheme=scheme)
            res = ops.apply_operator_torch(torch.tensor(q), *P, 30.0, dy, scheme=scheme)
            (phi.sum() + res.pow(2).sum()).backward()
            out[scheme] = (phi.detach(), res.detach(), [p.grad for p in P])
        assert torch.equal(out["hybrid"][0], out["central"][0]) and torch.equal(out["hybrid"][1], out["central"][1])
        assert all(torch.equal(a, b) for a, b in zip(out["hybrid"][2], out["central"][2]))
    # past Pe = 1 on an axis they differ
    a, b = (ops.solve(q, 60.0, (300.0, 0.0), dx=30.0, scheme=s) for s in ("hybrid", "central"))
    assert np.max(np.abs(a - b)) > 1e-3 * np.max(np.abs(b))


def test_params_scheme_falls_back_to_the_legacy_central_scheme():
    """Callers that re-apply a fitted operator resolve its scheme from the
    model's params; a dict without the key is a run fitted before the scheme
    existed (central), never the current default."""
    assert ops.LEGACY_SCHEME == "central" != ops.DEFAULT_SCHEME
    for old in ({}, None, {"advection_scheme": None}, {"L_m": 45.0, "vx_m": 400.0, "vy_m": -250.0}):
        assert ops.params_scheme(old) == "central"
    assert ops.params_scheme({"advection_scheme": "Exponential"}) == "exponential"
    assert ops.params_scheme({"advection_scheme": "hybrid"}) == "hybrid"
    assert ops.params_scheme({"advection_scheme": "central"}) == "central"
    with pytest.raises(ValueError, match="advection scheme"):
        ops.params_scheme({"advection_scheme": "upwind"})


def test_unknown_scheme_is_rejected():
    q = np.zeros((8, 8))
    assert ops.check_scheme("Exponential") == "exponential" and ops.DEFAULT_SCHEME == "hybrid" in ops.SCHEMES
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
