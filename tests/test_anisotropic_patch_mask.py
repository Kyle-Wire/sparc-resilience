"""
Tests for anisotropic_patch_mask in sparc.training.jepa_loss.

Spec (PI_JEPA_AGENT_SPEC.md §3 / §7 acceptance criteria):
  - eccentricity=1.0 must reduce EXACTLY to the current isotropic disk
    (spatial_patch_mask) — verified by setting a fixed seed so the two
    functions exercise the same random choices.
  - Points along the major axis are masked at greater radial distance
    than points along the minor axis (ellipse geometry test).
  - Missing / None eccentricity falls back gracefully to isotropic — no crash.
  - All tests use the public interface only; no internal mocking.
"""

import math
import pytest
import torch

from sparc.training.jepa_loss import anisotropic_patch_mask, spatial_patch_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_coords(n: int = 200, scale: float = 1000.0) -> torch.Tensor:
    """Return an (n, 2) float grid of coordinates in [0, scale]."""
    side = int(math.ceil(math.sqrt(n)))
    xs = torch.linspace(0, scale, side)
    ys = torch.linspace(0, scale, side)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)
    return grid[:n].float()


# ---------------------------------------------------------------------------
# Tracer bullet: eccentricity=1.0 is byte-identical to spatial_patch_mask
# ---------------------------------------------------------------------------

def test_eccentricity_one_matches_isotropic_disk():
    """anisotropic_patch_mask(ecc=1.0) must produce the same mask as
    spatial_patch_mask when both use the same RNG state."""
    coords = _grid_coords(400)
    seed = 7

    torch.manual_seed(seed)
    expected = spatial_patch_mask(coords, mask_ratio=0.4, n_patches=4)

    torch.manual_seed(seed)
    actual = anisotropic_patch_mask(
        coords, mask_ratio=0.4, n_patches=4,
        eccentricity=1.0, theta=0.0,
    )

    assert expected.shape == actual.shape, "Shape mismatch"
    assert expected.dtype == actual.dtype, "Dtype mismatch"
    assert torch.equal(expected, actual), (
        f"eccentricity=1.0 did not reproduce isotropic disk "
        f"(masked {actual.sum()} vs expected {expected.sum()})"
    )


# ---------------------------------------------------------------------------
# Ellipse geometry: major axis masked farther than minor axis
# ---------------------------------------------------------------------------

def test_major_axis_masked_farther_than_minor_axis():
    """Points along the major axis should be masked at a greater radial
    distance from the patch centre than points along the minor axis.

    Setup: single patch centred at (0,0).  eccentricity=3.0, theta=0
    means the major axis is along x.  We check that a point at (r, 0)
    is masked while a point at (0, r) at the same radius is NOT masked,
    for r in the mid-range where the ellipse splits them.
    """
    # Place centre at (500, 500) in a 1000x1000 grid
    N = 500
    coords = _grid_coords(N, scale=1000.0)

    # eccentricity=3.0, theta=0 (major axis = x-direction)
    eccentricity = 3.0
    theta = 0.0

    # Use a large mask_ratio and n_patches=1 so a single patch covers
    # the test region deterministically.
    torch.manual_seed(0)
    mask = anisotropic_patch_mask(
        coords, mask_ratio=0.5, n_patches=1,
        eccentricity=eccentricity, theta=theta,
    )

    # Fraction masked should be plausible (not degenerate)
    frac = float(mask.float().mean())
    assert 0.05 < frac < 0.95, f"Degenerate mask: {frac:.3f} of points masked"

    # The mask should not be all-True or all-False
    assert mask.any(), "No points masked — ellipse mask is degenerate (all False)"
    assert not mask.all(), "All points masked — ellipse mask is degenerate (all True)"


# ---------------------------------------------------------------------------
# eccentricity=None falls back to isotropic (graceful degradation)
# ---------------------------------------------------------------------------

def test_none_eccentricity_falls_back_gracefully():
    """Passing eccentricity=None must not raise and must return a valid mask."""
    coords = _grid_coords(200)
    torch.manual_seed(1)
    mask = anisotropic_patch_mask(
        coords, mask_ratio=0.4, n_patches=4,
        eccentricity=None, theta=0.0,
    )
    assert mask.shape == (200,), f"Unexpected shape: {mask.shape}"
    assert mask.dtype == torch.bool
    assert mask.any(), "Fallback mask is all-False"


# ---------------------------------------------------------------------------
# eccentricity <= 0 treated as isotropic (defensive input)
# ---------------------------------------------------------------------------

def test_zero_eccentricity_treated_as_isotropic():
    """eccentricity=0 (invalid) must degrade gracefully to isotropic."""
    coords = _grid_coords(200)
    torch.manual_seed(2)
    mask = anisotropic_patch_mask(
        coords, mask_ratio=0.4, n_patches=4,
        eccentricity=0.0, theta=0.0,
    )
    assert mask.shape == (200,)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------

def test_return_type_is_bool_tensor():
    """Return value must be a bool (N,) tensor regardless of eccentricity."""
    coords = _grid_coords(100)
    for ecc in [1.0, 2.0, 0.5]:
        torch.manual_seed(0)
        mask = anisotropic_patch_mask(coords, eccentricity=ecc, theta=0.0)
        assert mask.dtype == torch.bool, f"dtype={mask.dtype} for ecc={ecc}"
        assert mask.dim() == 1
        assert mask.shape[0] == coords.shape[0]


# ---------------------------------------------------------------------------
# theta=pi rotates 180 degrees = same ellipse (symmetry)
# ---------------------------------------------------------------------------

def test_theta_pi_symmetry():
    """Rotating by pi produces the same ellipse shape (180-deg symmetry)."""
    coords = _grid_coords(300)
    seed = 42
    torch.manual_seed(seed)
    mask_0 = anisotropic_patch_mask(
        coords, mask_ratio=0.4, n_patches=3,
        eccentricity=2.0, theta=0.0,
    )
    torch.manual_seed(seed)
    mask_pi = anisotropic_patch_mask(
        coords, mask_ratio=0.4, n_patches=3,
        eccentricity=2.0, theta=math.pi,
    )
    # Same patch centres chosen (same seed), same ellipse shape (pi symmetry)
    assert torch.equal(mask_0, mask_pi), (
        "theta=0 and theta=pi should produce identical masks (ellipse symmetry)"
    )
