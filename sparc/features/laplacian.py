import numpy as np
import pandas as pd
from libpysal.weights import KNN as PysalKNN
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

class LaplacianEigenmap:
    """
    A class to generate Laplacian Eigenmaps for spatial data using graph Laplacian.
    
    Parameters:
    -----------
    n_components : int, default=10
        Number of Laplacian Eigenmaps to generate
    k_neighbors : int, default=5
        Number of nearest neighbors for spatial weights matrix
    target_crs : str, optional
        Target CRS to project the data to before generating eigenmaps
    """
    
    def __init__(self, n_components=10, k_neighbors=5, target_crs=None):
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.target_crs = target_crs
        self.eigenmap_cols_ = None
        self.coords_ = None
        self.embedding_ = None
    
    def fit_transform(self, coords):
        """
        Generate Laplacian Eigenmaps for the input coordinates.
        
        Parameters:
        -----------
        coords : array-like, shape (n_samples, 2)
            Input coordinates (X, Y)
            
        Returns:
        --------
        array-like, shape (n_samples, n_components)
            Generated Laplacian Eigenmaps
        """
        coords = np.asarray(coords)
        if coords.shape[1] != 2:
            raise ValueError("coords must be a 2D array with shape (n_samples, 2)")

        n = coords.shape[0]
        # Adaptive k: cap neighbours for large datasets to control
        # SWM density and eigsh cost.  Beyond ~20K points the full k
        # often exceeds what's needed for a good Laplacian.
        k_eff = self.k_neighbors
        if n > 10_000:
            k_eff = min(self.k_neighbors, max(5, int(np.sqrt(n) / 2)))
            if k_eff != self.k_neighbors:
                print(f"  Laplacian adaptive k: {self.k_neighbors} -> {k_eff} (n={n:,})")
            
        try:
            # Create spatial weights matrix
            w_laplacian = PysalKNN.from_array(coords, k=k_eff)
            
            # Compute Laplacian matrix
            L = laplacian(w_laplacian.sparse, normed=True)
            
            # Calculate number of eigenvalues to compute
            k_eigsh = min(self.n_components + 1, L.shape[0] - 2)
            
            if k_eigsh < 2:
                raise ValueError("Not enough samples or n_components too low for eigsh")
            
            # Compute eigenvalues and eigenvectors
            vals, vecs = eigsh(L, k=k_eigsh, which='SM')
            
            # Extract eigenmaps (skip first eigenvector as it's constant)
            eigenmaps = vecs[:, 1:min(vecs.shape[1], self.n_components + 1)]
            
            # Store for derivative computation
            self.coords_ = coords.copy()
            self.embedding_ = eigenmaps.copy()
            
            # Store column names
            self.eigenmap_cols_ = [f'LAPEIG_{i+1}' for i in range(eigenmaps.shape[1])]
            
            return eigenmaps
            
        except Exception as e:
            print(f"Error generating Laplacian Eigenmaps: {e}")
            return np.zeros((coords.shape[0], self.n_components))
    
    def compute_manifold_derivatives(self, y, X, feature_names, output_dir):
        """
        Compute temperature sensitivities on the Laplacian manifold.
        
        Uses local linear regression to estimate:
        1. dT/dz_j for each eigen-dimension
        2. dz_j/dX_i for variable-projected gradients
        3. Chain rule: dT/dX_i = Σ_j (dT/dz_j * dz_j/dX_i)
        
        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target variable (temperature)
        X : array-like, shape (n_samples, n_features)
            Feature matrix (raw predictors)
        feature_names : list
            Names of features in X
        output_dir : str or Path
            Directory to save derivative outputs
        """
        if self.embedding_ is None or self.coords_ is None:
            print("Error: fit_transform() must be called first to generate embeddings.")
            return
        
        print("\n" + "="*80)
        print("EXTRACTING LAPLACIAN MANIFOLD DERIVATIVES")
        print("="*80)
        
        from pathlib import Path
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import NearestNeighbors
        
        output_dir = Path(output_dir)
        laplacian_dir = output_dir / 'laplacian_derivatives'
        laplacian_dir.mkdir(parents=True, exist_ok=True)
        
        # Physics sign rules
        physics_signs = {
            "Pct_Impervious": +1,
            "Pct_Canopy": -1,
            "NDVI": -1,
            "Albedo": -1
        }
        
        Z = self.embedding_
        n_components = Z.shape[1]
        n_samples = Z.shape[0]
        
        print(f"\nComputing manifold gradients for {n_components} eigen-dimensions...")
        print(f"Using {n_samples} samples with {len(feature_names)} features")
        
        # 1. Compute dT/dz_j using local linear regression
        print("\nStep 1: Computing dT/dz_j (temperature gradient on manifold)...")
        manifold_grads = np.zeros((n_samples, n_components))
        nn = NearestNeighbors(n_neighbors=min(50, n_samples))
        nn.fit(Z)
        
        for i in tqdm(range(n_samples), desc="Local gradients"):
            _, indices = nn.kneighbors(Z[i:i+1])
            Z_local = Z[indices[0]]
            y_local = y[indices[0]]
            
            # Handle constant y case
            if np.std(y_local) < 1e-6:
                manifold_grads[i] = 0
                continue
            
            try:
                reg = LinearRegression()
                reg.fit(Z_local, y_local)
                manifold_grads[i] = reg.coef_
            except:
                manifold_grads[i] = 0
        
        # Save eigen-dimension derivatives
        print(f"\nSaving {n_components} eigen-dimension derivative surfaces...")
        for j in range(n_components):
            deriv_df = pd.DataFrame({
                'POINT_X': self.coords_[:, 0],
                'POINT_Y': self.coords_[:, 1],
                'dT_dz': manifold_grads[:, j],
                'z_value': Z[:, j]
            })
            deriv_df.to_csv(
                laplacian_dir / f'laplacian_derivative_z{j+1}.csv',
                index=False
            )
        
        # 2. Compute dz_j/dX_i and apply chain rule for variable-projected derivatives
        print("\nStep 2: Computing variable-projected derivatives (chain rule)...")
        
        for feat_idx, feat_name in enumerate(tqdm(feature_names, desc="Variable derivatives")):
            variable_gradient = np.zeros(n_samples)
            
            for i in range(n_samples):
                _, indices = nn.kneighbors(Z[i:i+1])
                X_local = X[indices[0], feat_idx]
                
                # Handle constant X case
                if np.std(X_local) < 1e-6:
                    continue
                
                # For each eigen-dimension, compute dz_j/dX_i
                for j in range(n_components):
                    z_local = Z[indices[0], j]
                    
                    # Use covariance-based gradient estimate
                    try:
                        cov_matrix = np.cov(z_local, X_local)
                        var_X = cov_matrix[1, 1]
                        
                        if var_X > 1e-6:
                            dz_dX = cov_matrix[0, 1] / var_X
                            # Chain rule: dT/dX_i = Σ_j (dT/dz_j * dz_j/dX_i)
                            variable_gradient[i] += manifold_grads[i, j] * dz_dX
                    except:
                        continue
            
            # Physics-constrained gradient correction (preserves diagnostics)
            variable_gradient_corrected = variable_gradient.copy()
            sign_violations_lap = np.zeros(n_samples, dtype=int)
            if feat_name in physics_signs:
                expected_sign = physics_signs[feat_name]
                if expected_sign == -1:
                    violation_mask = variable_gradient > 0
                else:
                    violation_mask = variable_gradient < 0
                sign_violations_lap[violation_mask] = 1
                if expected_sign == -1:
                    variable_gradient_corrected = np.minimum(variable_gradient_corrected, 0.0)
                else:
                    variable_gradient_corrected = np.maximum(variable_gradient_corrected, 0.0)
            
            # Save variable-projected derivatives with sign violation diagnostics
            deriv_df = pd.DataFrame({
                'POINT_X': self.coords_[:, 0],
                'POINT_Y': self.coords_[:, 1],
                'laplacian_derivative_raw': variable_gradient,
                'laplacian_derivative_constrained': variable_gradient_corrected,
                'sign_violation': sign_violations_lap,
                'baseline_value': X[:, feat_idx]
            })
            deriv_df.to_csv(
                laplacian_dir / f'laplacian_derivative_{feat_name}.csv',
                index=False
            )
        
        print(f"\n✅ Laplacian manifold derivatives saved to: {laplacian_dir}")
        print(f"   Eigen-dimension derivatives: {n_components}")
        print(f"   Variable-projected derivatives: {len(feature_names)}")
        print(f"   Total files: {n_components + len(feature_names)}")


def generate_laplacian_eigenmaps(gdf, n_eigenmaps=10, k_neighbors=5, target_crs=None):
    """
    Generate Laplacian Eigenmaps for spatial data using graph Laplacian.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Input GeoDataFrame containing spatial data
    n_eigenmaps : int, default=10
        Number of Laplacian Eigenmaps to generate
    k_neighbors : int, default=5
        Number of nearest neighbors for spatial weights matrix
    target_crs : str, optional
        Target CRS to project the data to before generating eigenmaps
        
    Returns:
    --------
    GeoDataFrame
        Original GeoDataFrame enriched with Laplacian Eigenmaps
    list
        Names of the generated Laplacian Eigenmap columns
    """
    # Create a copy to avoid modifying the original
    data_for_modeling = gdf.copy()
    
    # Project to target CRS if specified
    if target_crs is not None and str(data_for_modeling.crs).upper() != str(target_crs).upper():
        print(f"Projecting GeoDataFrame from {data_for_modeling.crs} to {target_crs}...")
        data_for_modeling = data_for_modeling.to_crs(target_crs)
    
    # Add projected coordinates
    data_for_modeling['projected_X'] = data_for_modeling.geometry.x
    data_for_modeling['projected_Y'] = data_for_modeling.geometry.y
    
    print(f"\nGenerating {n_eigenmaps} Laplacian Eigenmaps using k={k_neighbors} for SWM...")
    
    # Prepare coordinates for Laplacian
    coords_for_laplacian = np.array(list(zip(data_for_modeling.geometry.x, data_for_modeling.geometry.y)))

    # Adaptive k for large datasets
    n = len(coords_for_laplacian)
    k_eff = k_neighbors
    if n > 10_000:
        k_eff = min(k_neighbors, max(5, int(np.sqrt(n) / 2)))
        if k_eff != k_neighbors:
            print(f"  Laplacian adaptive k: {k_neighbors} -> {k_eff} (n={n:,})")
    
    try:
        # Create spatial weights matrix
        print("Creating spatial weights matrix...")
        w_laplacian = PysalKNN.from_array(coords_for_laplacian, k=k_eff)
        
        # Compute Laplacian matrix
        print("Computing Laplacian matrix...")
        L = laplacian(w_laplacian.sparse, normed=True)
        
        # Calculate number of eigenvalues to compute
        k_eigsh = min(n_eigenmaps + 1, L.shape[0] - 2)
        
        if k_eigsh < 2:
            print("Warning: Not enough samples or n_eigenmaps too low for eigsh. Skipping Laplacian Eigenmaps.")
            return data_for_modeling, []
        
        # Compute eigenvalues and eigenvectors with progress bar
        print("Computing eigenvalues and eigenvectors...")
        with tqdm(total=100, desc="Computing eigenmaps") as pbar:
            vals, vecs = eigsh(L, k=k_eigsh, which='SM')
            pbar.update(100)
        
        # Extract eigenmaps (skip first eigenvector as it's constant)
        eigenmaps_to_add = vecs[:, 1:min(vecs.shape[1], n_eigenmaps + 1)]
        
        # Create DataFrame with eigenmaps
        eigenmap_df_cols = {f'LAPEIG_{i+1}': eigenmaps_to_add[:, i] for i in range(eigenmaps_to_add.shape[1])}
        eigenmap_df = pd.DataFrame(eigenmap_df_cols, index=data_for_modeling.index)
        
        # Join eigenmaps to original data
        data_for_modeling = data_for_modeling.join(eigenmap_df)
        laplacian_cols = list(eigenmap_df.columns)
        
        print(f"Successfully generated and added {len(laplacian_cols)} Laplacian Eigenmaps.")
        return data_for_modeling, laplacian_cols
        
    except Exception as e:
        print(f"Error generating Laplacian Eigenmaps: {e}. Proceeding without them.")
        return data_for_modeling, [] 