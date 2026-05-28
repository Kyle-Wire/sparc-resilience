"""
Spatiotemporal cross-validation strategies for the SPARC pipeline.

Provides three methods:
1. expanding_window  — train on all prior time periods, test on next
2. sliding_window    — fixed-length training window slides forward
3. spatiotemporal_block — joint spatial+temporal blocking (Roberts et al. 2017)
"""

import numpy as np


def expanding_window_splits(
    time_values: np.ndarray,
    n_splits: int = 5,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window temporal CV.

    Each fold uses all prior time steps for training and the next chunk
    of time steps for testing, with an optional gap to guard against
    temporal autocorrelation leakage.
    """
    unique_times = np.sort(np.unique(time_values))
    n_times = len(unique_times)
    if n_times < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} unique time steps for "
            f"{n_splits} expanding-window splits, got {n_times}"
        )

    # Reserve the first chunk as minimum training, split the rest
    chunk_size = max(1, (n_times - 1) // n_splits)
    folds = []

    for i in range(n_splits):
        test_start = 1 + i * chunk_size  # at least 1 time step for initial training
        test_end = min(test_start + chunk_size, n_times)
        if test_start >= n_times:
            break

        train_times = unique_times[:max(1, test_start - gap)]
        test_times = unique_times[test_start:test_end]

        train_mask = np.isin(time_values, train_times)
        test_mask = np.isin(time_values, test_times)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    return folds


def sliding_window_splits(
    time_values: np.ndarray,
    n_splits: int = 5,
    gap: int = 0,
    window_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Sliding-window temporal CV.

    Fixed-size training window moves forward through time.
    """
    unique_times = np.sort(np.unique(time_values))
    n_times = len(unique_times)

    if window_size is None:
        window_size = max(1, (n_times - n_splits) // 2)

    folds = []
    step = max(1, (n_times - window_size) // n_splits)

    for i in range(n_splits):
        train_start = i * step
        train_end = train_start + window_size
        test_start = train_end + gap
        test_end = min(test_start + step, n_times)

        if test_start >= n_times:
            break

        train_times = unique_times[train_start:train_end]
        test_times = unique_times[test_start:test_end]

        train_mask = np.isin(time_values, train_times)
        test_mask = np.isin(time_values, test_times)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    return folds


def spatiotemporal_block_splits(
    coords: np.ndarray,
    time_values: np.ndarray,
    n_splits: int = 5,
    block_size: float | None = None,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Joint spatial-temporal blocking CV (Roberts et al. 2017).

    Combines spatial block assignment with temporal block assignment
    so that neither nearby locations nor nearby time steps leak
    between train and test.
    """
    n = len(coords)
    unique_times = np.sort(np.unique(time_values))
    n_times = len(unique_times)

    # --- Spatial blocks ---
    if n == 0:
        raise ValueError("coords is empty — cannot create spatiotemporal folds")
    bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
    if block_size is None:
        extent = np.linalg.norm(bounds[1] - bounds[0])
        block_size = extent / (n_splits * 2)

    n_blocks_x = max(1, int(np.ceil((bounds[1, 0] - bounds[0, 0]) / block_size)))
    n_blocks_y = max(1, int(np.ceil((bounds[1, 1] - bounds[0, 1]) / block_size)))

    spatial_block = np.zeros(n, dtype=int)
    for i in range(n):
        bx = int((coords[i, 0] - bounds[0, 0]) / block_size)
        by = int((coords[i, 1] - bounds[0, 1]) / block_size)
        spatial_block[i] = bx * n_blocks_y + by

    # --- Temporal blocks ---
    time_map = {t: idx for idx, t in enumerate(unique_times)}
    time_idx = np.array([time_map[t] for t in time_values])
    t_block_size = max(1, n_times // n_splits)
    temporal_block = time_idx // t_block_size

    # --- Combined block ID ---
    n_t_blocks = temporal_block.max() + 1
    combined_block = spatial_block * n_t_blocks + temporal_block

    unique_blocks = np.unique(combined_block)
    np.random.shuffle(unique_blocks)

    folds = []
    for fold_idx in range(n_splits):
        fold_blocks = unique_blocks[fold_idx::n_splits]
        test_mask = np.isin(combined_block, fold_blocks)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        # Apply temporal gap: remove train rows whose time is within `gap` steps of any test time
        if gap > 0:
            test_time_set = set(time_idx[test_idx])
            buffer_times = set()
            for tt in test_time_set:
                for g in range(1, gap + 1):
                    buffer_times.add(tt - g)
                    buffer_times.add(tt + g)
            buffer_mask = np.array([time_idx[i] in buffer_times for i in train_idx])
            train_idx = train_idx[~buffer_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    return folds


def spatiotemporal_kfold(
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    time_values: np.ndarray,
    config: dict,
    block_size: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    High-level entry point: create spatiotemporal CV folds from sparc.config.

    Reads ``config['temporal']['temporal_cv']`` and dispatches to the
    appropriate strategy.

    Falls back to pure spatial blocking (via existing ``spatial_kfold``)
    when temporal mode is disabled.
    """
    tcv = config.get('temporal', {}).get('temporal_cv', {})
    method = tcv.get('method', 'expanding_window')
    n_splits = tcv.get('n_splits', 5)
    gap = tcv.get('gap', 0)

    print(f"Spatiotemporal CV: method={method}, n_splits={n_splits}, gap={gap}")

    if method == 'expanding_window':
        folds = expanding_window_splits(time_values, n_splits, gap)
    elif method == 'sliding_window':
        folds = sliding_window_splits(time_values, n_splits, gap)
    elif method == 'spatiotemporal_block':
        folds = spatiotemporal_block_splits(coords, time_values, n_splits, block_size, gap)
    else:
        raise ValueError(f"Unknown temporal CV method: {method}")

    for i, (tr, te) in enumerate(folds):
        print(f"  Fold {i+1}: Train={len(tr)}, Test={len(te)}")

    return folds
