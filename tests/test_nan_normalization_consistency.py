"""
Regression test: Phase 1 and Phase 2/LOO must apply NaN normalization consistently.

Root cause: NaN pixels were being encoded as (0 - X_mean) / X_std in Phase 2/LOO
(nan_to_num BEFORE normalize) but as 0 in Phase 1 (nan_to_num AFTER normalize).

For features with large X_mean (e.g. albedo X_mean≈11960), NaN pixels became
-5.33σ out-of-distribution at Phase 2/LOO — garbage trunk embeddings → negative
spatial correlations in LOO eval (train62-65 regression).

The fix: both Phase 2 and LOO use nan_to_num AFTER normalization so NaN = 0 in
normalized space everywhere.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "train_multicity_jepa.py"


# ---------------------------------------------------------------------------
# AST helper
# ---------------------------------------------------------------------------

def _source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _func_src(tree: ast.Module, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found in {_SCRIPT.name}")


# ---------------------------------------------------------------------------
# Phase 1 reference: run_jepa_pretraining must use NaN → 0 AFTER normalize
# ---------------------------------------------------------------------------

def test_phase1_nan_after_normalize():
    """Phase 1 must call nan_to_num AFTER the (X - mean)/std normalization."""
    src = _func_src(_tree(), "run_jepa_pretraining")
    # nan_to_num wraps the normalize expression "(all_X - X_mean) / X_std"
    assert "nan_to_num" in src, "run_jepa_pretraining must call nan_to_num"
    # The normalization happens inside nan_to_num call  
    assert "nan_to_num((all_X - X_mean)" in src or "nan_to_num((all_X -" in src, (
        "Phase 1 must call nan_to_num(( X - mean ) / std ) — NaN fills after divide"
    )


# ---------------------------------------------------------------------------
# Phase 2: run_city_finetune must match Phase 1 convention
# ---------------------------------------------------------------------------

def test_phase2_nan_after_normalize():
    """run_city_finetune must call nan_to_num AFTER normalization (Phase 1 convention)."""
    src = _func_src(_tree(), "run_city_finetune")
    # The old (buggy) pattern: nan_to_num(..., nan=0.0) then X_raw = (X_raw - X_mean)/X_std
    # The fix: (X_raw - X_mean)/X_std then nan_to_num
    assert "nan_to_num((X_raw - X_mean)" in src or "nan_to_num((X_raw -" in src, (
        "run_city_finetune must apply nan_to_num AFTER normalization so NaN = 0 in "
        "normalized space (matches Phase 1). The old order nan_to_num → normalize "
        "maps NaN to (0-X_mean)/X_std, which is far OOD for high-mean features "
        "like albedo (X_mean≈11960 → -5.33σ) and causes negative LOO correlation."
    )


def test_phase2_nan_not_before_normalize():
    """run_city_finetune must NOT do nan_to_num on raw features before normalizing."""
    src = _func_src(_tree(), "run_city_finetune")
    # Detect the old buggy pattern: a standalone nan_to_num call on the raw feature
    # matrix (before normalization), which would produce the wrong encoded value.
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "nan_to_num" in line and "X_mean" not in line and "X_std" not in line:
            # There may be legitimate nan_to_num calls on labels — skip those
            if "anomaly" in line or "label" in line.lower() or "valid" in line:
                continue
            # A standalone nan_to_num on features (no normalization in same expression)
            # is only OK if it's in the else branch (when normalizer is None)
            context = "\n".join(lines[max(0, i-3): i+3])
            assert "else" in context or "normalizer is None" in context or "normalizer" not in "\n".join(lines[max(0,i-5):i]), (
                f"Suspicious nan_to_num before normalization at line:\n  {line}\n"
                "This is the NaN-before-normalize bug that causes negative LOO correlations."
            )


# ---------------------------------------------------------------------------
# Phase 3: run_loo_eval must match Phase 1 convention
# ---------------------------------------------------------------------------

def test_loo_nan_after_normalize():
    """run_loo_eval must call nan_to_num AFTER normalization (Phase 1 convention)."""
    src = _func_src(_tree(), "run_loo_eval")
    assert "nan_to_num((X_raw - X_mean)" in src or "nan_to_num((X_raw -" in src, (
        "run_loo_eval must apply nan_to_num AFTER normalization so NaN = 0 in "
        "normalized space (matches Phase 1). Mismatched NaN encoding causes "
        "garbage trunk embeddings on the holdout city and negative correlation."
    )


# ---------------------------------------------------------------------------
# Numeric consistency: verify Phase 1 ≡ Phase 2 for NaN pixels
# ---------------------------------------------------------------------------

def test_nan_pixel_encodes_to_zero_in_normalized_space():
    """
    NaN pixels must encode to 0.0 in normalized space in ALL phases.

    This test directly verifies the mathematical invariant: given a feature
    column with X_mean=11960 and X_std=2244 (realistic albedo statistics),
    a NaN pixel must produce 0.0 — not -5.33 — after normalization.
    """
    X_mean = np.array([11960.0], dtype=np.float32)
    X_std  = np.array([2244.0],  dtype=np.float32)

    # Simulate a pixel with NaN albedo value
    x_raw = np.array([[np.nan]], dtype=np.float32)

    # Phase 1 convention: normalize then nan_to_num
    x_phase1 = np.nan_to_num((x_raw - X_mean) / X_std, nan=0.0)
    assert x_phase1[0, 0] == pytest.approx(0.0), (
        f"Phase 1 convention: NaN should encode to 0.0 in normalized space, "
        f"got {x_phase1[0, 0]}"
    )

    # Old (buggy) Phase 2 convention: nan_to_num then normalize
    x_buggy = (np.nan_to_num(x_raw, nan=0.0) - X_mean) / X_std
    expected_buggy = (0.0 - 11960.0) / 2244.0  # ≈ -5.33
    assert x_buggy[0, 0] == pytest.approx(expected_buggy, rel=1e-4), (
        "Buggy Phase 2 convention sanity check: should produce -5.33"
    )
    assert abs(x_phase1[0, 0] - x_buggy[0, 0]) > 5.0, (
        "The two conventions must differ significantly for high-mean features "
        "(confirms the bug was real and the fix is meaningful)"
    )

    # New (fixed) Phase 2 convention: normalize then nan_to_num (same as Phase 1)
    x_fixed = np.nan_to_num((x_raw - X_mean) / X_std, nan=0.0)
    assert x_fixed[0, 0] == pytest.approx(0.0), (
        "Fixed Phase 2 convention: NaN must encode to 0.0 (matches Phase 1)"
    )
    assert x_fixed[0, 0] == pytest.approx(x_phase1[0, 0]), (
        "Fixed Phase 2 must match Phase 1 exactly for NaN pixels"
    )
