"""
Fold-Aware Laplacian Eigenmaps
==============================
Generates Laplacian eigenmaps that respect train/test separation to prevent
spatial information leakage through the feature space.

Problem: Computing eigenmaps on ALL data before splitting means the graph
structure encodes test-fold cell positions. This inflates R² by ~0.01-0.03.

Solution: Recompute the Laplacian on training data only, then use Nyström
out-of-sample extension to project test points into the training eigenspace.

Author: SPARC Labs
Date: February 2026
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from libpysal.weights import KNN as PysalKNN
from sklearn.neighbors import NearestNeighbors


def generate_fold_aware_laplacian(coords_train, coords_test, n_components=150,
                                   k_neighbors=10):
    """
    Generate Laplacian eigenmaps that respect train/test separation.

    Approach:
    1. Build the KNN graph using ONLY training coordinates
    2. Compute eigenvectors on the training graph
    3. Use Nyström out-of-sample extension to project test points
       into the training eigenspace WITHOUT rebuilding the graph

    Parameters
    ----------
    coords_train : array-like, shape (n_train, 2)
        Training coordinates (X, Y)
    coords_test : array-like, shape (n_test, 2)
        Test coordinates (X, Y)
    n_components : int, default=150
        Number of eigenmaps to compute
    k_neighbors : int, default=10
        Number of nearest neighbours for the spatial weights matrix

    Returns
    -------
    eigenvectors_train : ndarray, shape (n_train, n_components)
        Eigenvectors computed on training graph
    eigenvectors_test : ndarray, shape (n_test, n_components)
        Nyström-projected eigenvectors for test points
    eigenvalues : ndarray, shape (n_components,)
        Corresponding eigenvalues (skip the trivial constant one)
    """
    coords_train = np.asarray(coords_train)
    coords_test = np.asarray(coords_test)

    # ------------------------------------------------------------------
    # Step 1: Build graph on training data only
    # ------------------------------------------------------------------
    k_nn = min(k_neighbors, len(coords_train) - 1)
    w_train = PysalKNN.from_array(coords_train, k=k_nn)
    L_train = laplacian(w_train.sparse, normed=True)

    # ------------------------------------------------------------------
    # Step 2: Compute eigenvectors on training graph
    # ------------------------------------------------------------------
    k_eigsh = min(n_components + 1, L_train.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(L_train, k=k_eigsh, which='SM')

    # Skip the first (constant) eigenvector
    eigenvalues = eigenvalues[1:]
    eigenvectors_train = eigenvectors[:, 1:]

    # Truncate to requested n_components
    n_actual = min(n_components, eigenvectors_train.shape[1])
    eigenvalues = eigenvalues[:n_actual]
    eigenvectors_train = eigenvectors_train[:, :n_actual]

    # ------------------------------------------------------------------
    # Step 3: Nyström extension for test points
    # ------------------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=k_nn)
    nn.fit(coords_train)
    distances, indices = nn.kneighbors(coords_test)

    # Gaussian kernel weights with adaptive bandwidth
    bandwidth = np.median(distances[:, -1]) + 1e-10
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)  # Row-normalise

    # Nyström: test eigenvector ≈ weighted average of training eigenvectors
    eigenvectors_test = np.zeros((len(coords_test), eigenvectors_train.shape[1]))
    for i in range(len(coords_test)):
        neighbor_vecs = eigenvectors_train[indices[i]]       # (k, n_comp)
        eigenvectors_test[i] = (weights[i, :, np.newaxis] * neighbor_vecs).sum(axis=0)

    # Match column norms to training scale
    for j in range(eigenvectors_test.shape[1]):
        train_norm = np.linalg.norm(eigenvectors_train[:, j])
        test_norm = np.linalg.norm(eigenvectors_test[:, j])
        if test_norm > 1e-10:
            eigenvectors_test[:, j] *= train_norm / test_norm

    return eigenvectors_train, eigenvectors_test, eigenvalues


class FoldAwareLaplacianEigenmap:
    """
    Drop-in replacement for :class:`LaplacianEigenmap` that recomputes the
    graph per CV fold.  The ``fit_transform`` method operates on training
    coordinates, and ``transform`` projects new (test) coordinates via
    Nyström extension.

    Parameters
    ----------
    n_components : int
        Number of eigenmaps to produce
    k_neighbors : int
        KNN parameter for the spatial weights graph
    """

    def __init__(self, n_components=150, k_neighbors=10):
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.eigenvectors_train_ = None
        self.eigenvalues_ = None
        self.coords_train_ = None
        self.eigenmap_cols_ = None

    def fit_transform(self, coords_train):
        """Build graph on *training* coordinates and return eigenmaps."""
        coords_train = np.asarray(coords_train)
        self.coords_train_ = coords_train

        k_nn = min(self.k_neighbors, len(coords_train) - 1)
        w = PysalKNN.from_array(coords_train, k=k_nn)
        L = laplacian(w.sparse, normed=True)

        k_eigsh = min(self.n_components + 1, L.shape[0] - 2)
        eigenvalues, eigenvectors = eigsh(L, k=k_eigsh, which='SM')

        self.eigenvalues_ = eigenvalues[1:]
        self.eigenvectors_train_ = eigenvectors[:, 1:]

        n_actual = min(self.n_components, self.eigenvectors_train_.shape[1])
        self.eigenvalues_ = self.eigenvalues_[:n_actual]
        self.eigenvectors_train_ = self.eigenvectors_train_[:, :n_actual]

        self.eigenmap_cols_ = [f'LAPEIG_{i+1}' for i in range(n_actual)]
        return self.eigenvectors_train_

    def transform(self, coords_test):
        """Nyström out-of-sample extension for *test* coordinates."""
        if self.eigenvectors_train_ is None:
            raise RuntimeError("Call fit_transform() first.")

        coords_test = np.asarray(coords_test)
        k_nn = min(self.k_neighbors, len(self.coords_train_) - 1)

        nn = NearestNeighbors(n_neighbors=k_nn)
        nn.fit(self.coords_train_)
        distances, indices = nn.kneighbors(coords_test)

        bandwidth = np.median(distances[:, -1]) + 1e-10
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)

        eigenvectors_test = np.zeros((len(coords_test),
                                      self.eigenvectors_train_.shape[1]))
        for i in range(len(coords_test)):
            neighbor_vecs = self.eigenvectors_train_[indices[i]]
            eigenvectors_test[i] = (weights[i, :, np.newaxis] * neighbor_vecs).sum(axis=0)

        for j in range(eigenvectors_test.shape[1]):
            train_norm = np.linalg.norm(self.eigenvectors_train_[:, j])
            test_norm = np.linalg.norm(eigenvectors_test[:, j])
            if test_norm > 1e-10:
                eigenvectors_test[:, j] *= train_norm / test_norm

        return eigenvectors_test
