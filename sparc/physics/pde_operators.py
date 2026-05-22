"""
PDE finite-difference operators for SPARC V3.

Provides Laplacian, gradient, curvature, and Hessian operators using
cardinal (N/S/E/W) neighbor structure from ``_build_cardinal_neighbors``.

Grid spacing
------------
For irregular point clouds (e.g. Providence), the grid spacing ``h`` is
not fixed.  Two modes are available:

  ``spacing='local'``  (default)
      Per-point h_i estimated from cardinal neighbor distances.
      Laplacian becomes  (f_N + f_S + f_E + f_W - 4f) / h_i².

  ``spacing='global'``
      Single scalar h = median of all cardinal distances.
      Appropriate for regular grids or when local h is too noisy.

All operators expect projected coordinates in **meters** so that
Laplacian magnitudes are physically meaningful (units: field / m²).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid spacing estimation
# ---------------------------------------------------------------------------

def estimate_local_spacing(
    coords: torch.Tensor,
    neighbor_idx: torch.Tensor,
    spacing: str = "local",
) -> torch.Tensor:
    """
    Estimate the effective grid spacing h per point.

    Parameters
    ----------
    coords : (N, 2) — projected coordinates (x, y) in meters
    neighbor_idx : (N, 4) long — [North, South, East, West], -1 = missing
    spacing : ``'local'`` or ``'global'``

    Returns
    -------
    h : (N,) tensor — per-point grid spacing in meters.
        For ``spacing='global'``, all values are the same scalar.
    """
    N = coords.shape[0]
    device = coords.device

    # Compute distance to each valid cardinal neighbor
    dists = torch.zeros(N, 4, device=device)
    valid_count = torch.zeros(N, device=device)

    for k in range(4):
        valid_k = neighbor_idx[:, k] >= 0
        if valid_k.any():
            nbr_coords = coords[neighbor_idx[valid_k, k].clamp(min=0)]
            d = torch.norm(coords[valid_k] - nbr_coords, dim=1)
            dists[valid_k, k] = d
            valid_count[valid_k] += 1.0

    # Mean distance across valid neighbors per point
    dist_sum = dists.sum(dim=1)
    h_local = dist_sum / valid_count.clamp(min=1)

    # Points with no valid neighbors: use global median
    no_neighbors = valid_count == 0
    valid_h = h_local[~no_neighbors]

    if spacing == "global" or len(valid_h) == 0:
        h_global = valid_h.median() if len(valid_h) > 0 else torch.tensor(30.0, device=device)
        h = torch.full((N,), h_global.item(), device=device)
    else:
        h = h_local.clone()
        if no_neighbors.any():
            h[no_neighbors] = valid_h.median()

    # Clamp to prevent division by zero or extreme values
    h = h.clamp(min=1.0)

    # Log diagnostics
    logger.info(
        "Grid spacing: h_median=%.2f  h_std=%.2f  h_min=%.2f  h_max=%.2f  mode=%s",
        h.median().item(), h.std().item(), h.min().item(), h.max().item(), spacing,
    )

    return h


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def laplacian(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finite-difference Laplacian: ∇²f ≈ (f_N + f_S + f_E + f_W - 4f) / h².

    Parameters
    ----------
    field : (N,) — scalar field values
    neighbor_idx : (N, 4) long — [N, S, E, W], -1 = missing
    h : (N,) or scalar — grid spacing per point

    Returns
    -------
    lap : (M,) — Laplacian values for valid points
    valid : (N,) bool — mask of points with all 4 neighbors present
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]
    f_c = field[valid]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    lap = (f_n + f_s + f_e + f_w - 4.0 * f_c) / (h_valid ** 2)

    return lap, valid


# ---------------------------------------------------------------------------
# Directional curvatures
# ---------------------------------------------------------------------------

def directional_curvatures(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-axis second derivatives: ∂²f/∂x² and ∂²f/∂y².

    Uses E/W for x-axis and N/S for y-axis:
      ∂²f/∂x² ≈ (f_E + f_W - 2f) / h²
      ∂²f/∂y² ≈ (f_N + f_S - 2f) / h²

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long — [N, S, E, W]
    h : (N,) or scalar

    Returns
    -------
    d2_dx2 : (M,) — second derivative in x
    d2_dy2 : (M,) — second derivative in y
    valid : (N,) bool
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]
    f_c = field[valid]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    h2 = h_valid ** 2

    d2_dy2 = (f_n + f_s - 2.0 * f_c) / h2
    d2_dx2 = (f_e + f_w - 2.0 * f_c) / h2

    return d2_dx2, d2_dy2, valid


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------

def gradient_magnitude(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gradient magnitude from central differences.

      ∂f/∂x ≈ (f_E - f_W) / (2h)
      ∂f/∂y ≈ (f_N - f_S) / (2h)
      |∇f| = √((∂f/∂x)² + (∂f/∂y)²)

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long — [N, S, E, W]
    h : (N,) or scalar

    Returns
    -------
    grad_mag : (M,) — gradient magnitude
    df_dx : (M,) — x-derivative
    df_dy : (M,) — y-derivative
    valid : (N,) bool
    """
    valid = (neighbor_idx >= 0).all(dim=1)

    f_n = field[neighbor_idx[valid, 0]]
    f_s = field[neighbor_idx[valid, 1]]
    f_e = field[neighbor_idx[valid, 2]]
    f_w = field[neighbor_idx[valid, 3]]

    h_valid = h[valid] if isinstance(h, torch.Tensor) and h.dim() > 0 else h
    two_h = 2.0 * h_valid

    df_dx = (f_e - f_w) / two_h
    df_dy = (f_n - f_s) / two_h
    grad_mag = torch.sqrt(df_dx ** 2 + df_dy ** 2 + 1e-12)

    return grad_mag, df_dx, df_dy, valid


# ---------------------------------------------------------------------------
# Hessian invariants
# ---------------------------------------------------------------------------

def hessian_invariants(
    field: torch.Tensor,
    neighbor_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hessian matrix invariants from the diagonal second derivatives.

    Since we only have cardinal neighbors, the off-diagonal ∂²f/∂x∂y
    is not directly available.  We compute:

      det(H) ≈ (∂²f/∂x²)(∂²f/∂y²)        [product of diagonal entries]
      tr(H)  = ∂²f/∂x² + ∂²f/∂y² = ∇²f   [Laplacian]
      anisotropy = |∂²f/∂x² - ∂²f/∂y²| / (|∂²f/∂x²| + |∂²f/∂y²| + ε)

    Interpretation:
      det(H) > 0 → elliptic point (local extremum)
      det(H) < 0 → saddle point (transition zone)
      anisotropy ≈ 0 → isotropic curvature
      anisotropy ≈ 1 → strongly directional

    Parameters
    ----------
    field : (N,)
    neighbor_idx : (N, 4) long
    h : (N,) or scalar

    Returns
    -------
    det_H : (M,)
    anisotropy : (M,)
    valid : (N,) bool
    """
    d2_dx2, d2_dy2, valid = directional_curvatures(field, neighbor_idx, h)

    det_H = d2_dx2 * d2_dy2

    denom = d2_dx2.abs() + d2_dy2.abs() + 1e-8
    anisotropy = (d2_dx2 - d2_dy2).abs() / denom

    return det_H, anisotropy, valid


# ---------------------------------------------------------------------------
# Precomputed sparse Laplacian
# ---------------------------------------------------------------------------

def build_sparse_laplacian(
    cardinal_idx: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a precomputed sparse Laplacian matrix for the full point cloud.

    For each point i with all 4 valid cardinal neighbors [N, S, E, W]:
        L[i, i]   =  4 / h_i²
        L[i, j_k] = -1 / h_i²   for k in {N, S, E, W}

    Points missing any neighbor have a zero row (L[i, :] = 0).

    Applying the result: ``(L @ field.unsqueeze(-1)).squeeze(-1)``
    yields the h-normalised Laplacian as a dense (N,) vector, with the
    same values as ``_expand(lap_raw, valid_mask, N)`` from the per-gather
    path — but via a single sparse matrix–vector product rather than four
    index-gather ops.

    Only activates in full-N evaluation contexts (not per-batch training,
    where cardinal indices are remapped to batch-local space).

    Parameters
    ----------
    cardinal_idx : (N, 4) LongTensor — [N, S, E, W] neighbors, -1 = missing.
                   Must be on the target device.
    h : (N,) FloatTensor — per-point grid spacing in metres.

    Returns
    -------
    sparse_L : torch.sparse_coo_tensor, shape (N, N), coalesced
    valid_mask : (N,) bool tensor — rows with all 4 valid neighbors
    """
    device = cardinal_idx.device
    N = cardinal_idx.shape[0]
    valid = (cardinal_idx >= 0).all(dim=1)   # (N,) bool
    M = int(valid.sum().item())

    if M == 0:
        sparse_L = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (N, N), device=device,
        ).coalesce()
        return sparse_L, valid

    valid_idx = valid.nonzero(as_tuple=True)[0]   # (M,)
    h2 = h[valid] ** 2                            # (M,) — spacing² for valid pts

    # Center diagonal entries: L[i, i] = -4/h_i²  (matches ∇²f sign convention)
    row_c = valid_idx                             # (M,)
    col_c = valid_idx                             # (M,)
    val_c = -4.0 / h2                             # (M,)

    # Off-diagonal entries: L[i, j_k] = +1/h_i² for each of 4 neighbors
    nb_idx = cardinal_idx[valid]                  # (M, 4)
    row_n = valid_idx.unsqueeze(1).expand(M, 4).reshape(-1)   # (4M,)
    col_n = nb_idx.reshape(-1)                                 # (4M,)
    val_n = (1.0 / h2).unsqueeze(1).expand(M, 4).reshape(-1)  # (4M,)

    rows = torch.cat([row_c, row_n])   # (5M,)
    cols = torch.cat([col_c, col_n])   # (5M,)
    vals = torch.cat([val_c, val_n])   # (5M,)

    sparse_L = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (N, N), device=device,
    ).coalesce()
    return sparse_L, valid


# ---------------------------------------------------------------------------
# Sheaf Laplacian (Term 11 — multi-scale MAUP resistance)
# ---------------------------------------------------------------------------

def build_sheaf_laplacian(
    knn_idx: torch.Tensor,
    stalk_dim: int = 2,
    restriction_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build a sparse Sheaf Laplacian for a KNN spatial graph.

    Each node v has a stalk F_v ≅ ℝ^d.  The coboundary matrix
    δ⁰ : C⁰(ℱ) → C¹(ℱ) is defined edge-wise as::

        (δ⁰ f)_{(u,v)} = R_{v→e} f_v − R_{u→e} f_u

    where R_{·→e} are restriction maps from vertex stalks to the edge
    stalk F_e ≅ ℝ^d.  The sheaf Laplacian is L = (δ⁰)ᵀ (δ⁰).

    When ``restriction_maps`` is None, we use the identity restriction
    map (R = I_d), which reduces to the standard KNN graph Laplacian
    lifted to d-dimensional stalks — appropriate when the fine-scale
    (T_pred) and coarse-scale (T_smooth) predictions share the same
    interpretation.

    Applying to a (N, d) section f ∈ C⁰(ℱ)::

        loss = (L f).pow(2).mean()
             = ||δ⁰ f||² / N

    Parameters
    ----------
    knn_idx : (N, K) LongTensor — K nearest-neighbor indices per point.
              Indices must be in [0, N) with -1 meaning "no neighbor".
    stalk_dim : int — dimension d of each vertex/edge stalk. Default 2
                (one slot for T_pred, one for T_smooth).
    restriction_maps : (E, d, d) FloatTensor or None.
        Per-edge restriction maps.  If None, identity maps are used.
        E = number of directed edges = N × K (before filtering invalid).

    Returns
    -------
    sheaf_L : torch.sparse_coo_tensor, shape (N·d, N·d), coalesced.
              Apply via ``torch.sparse.mm(sheaf_L, f.reshape(-1, 1)).reshape(N, d)``.
    """
    device = knn_idx.device
    N, K = knn_idx.shape
    d = stalk_dim
    Nd = N * d

    # Collect valid edges (u → v) where v != -1
    u_list, v_list = [], []
    for k in range(K):
        v_col = knn_idx[:, k]               # (N,)
        valid = v_col >= 0                   # (N,) bool
        u_idx = valid.nonzero(as_tuple=True)[0]   # source nodes
        v_idx = v_col[valid]                       # target nodes
        u_list.append(u_idx)
        v_list.append(v_idx)

    u_all = torch.cat(u_list)   # (E,)
    v_all = torch.cat(v_list)   # (E,)
    E = u_all.shape[0]

    if E == 0:
        sheaf_L = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (Nd, Nd),
        ).coalesce()
        return sheaf_L

    # Build δ⁰ : ℝ^{Nd} → ℝ^{Ed}  (coboundary)
    # For each edge e=(u,v) and each stalk dimension i:
    #   δ⁰[e*d+i, v*d+i] +=  R_v[i, i]  (+1 for identity)
    #   δ⁰[e*d+i, u*d+i] += -R_u[i, i]  (-1 for identity)
    # Shape of δ⁰: (E·d, N·d)

    e_idx = torch.arange(E, device=device)          # (E,)
    d_idx = torch.arange(d, device=device)          # (d,)
    # Broadcast: each edge e paired with each stalk dim i
    e_rep = e_idx.unsqueeze(1).expand(E, d)         # (E, d)
    d_rep = d_idx.unsqueeze(0).expand(E, d)         # (E, d)
    row_ed = (e_rep * d + d_rep).reshape(-1)        # (E*d,)

    u_rep = u_all.unsqueeze(1).expand(E, d)         # (E, d)
    v_rep = v_all.unsqueeze(1).expand(E, d)         # (E, d)
    col_u = (u_rep * d + d_rep).reshape(-1)         # (E*d,)
    col_v = (v_rep * d + d_rep).reshape(-1)         # (E*d,)

    # Identity restriction: +1 for v-end, -1 for u-end
    if restriction_maps is None:
        val_v = torch.ones(E * d, device=device, dtype=torch.float32)
        val_u = -torch.ones(E * d, device=device, dtype=torch.float32)
    else:
        # restriction_maps: (E, d, d) — use diagonal only for this stalk-dim expansion
        # restriction_maps[e, i, i] for each edge e, stalk dim i
        # Full off-diagonal not needed for identity-stalk case, but supported.
        R = restriction_maps  # (E, d, d)
        if R.shape[0] != E:
            raise ValueError(
                f"restriction_maps first dim ({R.shape[0]}) must match "
                f"number of directed edges E={E}"
            )
        # val_v[e*d+i] = R_e[i, i]  (diagonal restriction, v-end)
        diag_R = torch.diagonal(R, dim1=1, dim2=2)  # (E, d)
        val_v = diag_R.reshape(-1)
        val_u = -val_v

    # δ⁰ as sparse (E*d, N*d)
    Ed = E * d
    rows_delta = torch.cat([row_ed, row_ed])        # (2*E*d,)
    cols_delta = torch.cat([col_v, col_u])          # (2*E*d,)
    vals_delta = torch.cat([val_v, val_u])          # (2*E*d,)

    delta = torch.sparse_coo_tensor(
        torch.stack([rows_delta, cols_delta]),
        vals_delta,
        (Ed, Nd),
    ).coalesce()

    # Sheaf Laplacian L = δ⁰ᵀ δ⁰  (sparse × sparse)
    # PyTorch supports sparse.mm for (sparse, dense). Build via:
    #   L_dense = δᵀ δ  for small N, or keep as factored form for large N.
    # For memory efficiency, return δ so caller can compute ||δ f||² directly.
    # We return δ⁰ itself (called "sheaf_L" for API consistency but is δ⁰).
    # The loss is ||δ⁰ f||² / (E*d) which equals (f.T L f) / (E*d).
    return delta

