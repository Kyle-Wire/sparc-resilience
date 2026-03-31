#!/usr/bin/env python3
"""
Laplacian Eigenmap t-SNE Visualizer

This script creates comprehensive t-SNE visualizations of Laplacian eigenmaps
to understand the spatial structure captured by these features in the meta-ensemble.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# When installed via `pip install -e .`, the package root is already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sparc.config.config import load_config
from sparc.data.data_utils import load_and_preprocess_data
from sparc.features.laplacian import LaplacianEigenmap
from sparc.run.pipeline_paths import PipelinePaths


class LaplacianTSNEVisualizer:
    """
    A comprehensive visualizer for Laplacian eigenmaps using t-SNE
    """
    
    def __init__(self, config_path=None):
        """Initialize the visualizer with configuration"""
        self.config = load_config(config_path)
        self.paths = PipelinePaths()
        self.laplacian_features = None
        self.coords = None
        self.target_values = None
        self.data = None
        
    def load_data(self, data_file=None):
        """Load and preprocess the spatial data"""
        print("=== Loading Spatial Data ===")
        
        if data_file is None:
            data_file = self.paths.get_data_file()
        
        # Load the data
        self.data = load_and_preprocess_data(
            raw_data_path=data_file,
            identifier_col=self.config['variables']['identifier'],
            target_col=self.config['variables']['target'],
            coords_cols=self.config['variables']['coordinates'],
            predictor_cols=self.config['predictors']['base_model'],
            initial_crs=self.config['crs']['initial'],
            target_crs=self.config['crs']['target_projected']
        )
        
        # Extract coordinates and target
        self.coords = self.data[['projected_X', 'projected_Y']].values
        self.target_values = self.data[self.config['variables']['target']].values
        
        print(f"Loaded {len(self.data)} spatial observations")
        print(f"Coordinate range: X[{self.coords[:, 0].min():.2f}, {self.coords[:, 0].max():.2f}], "
              f"Y[{self.coords[:, 1].min():.2f}, {self.coords[:, 1].max():.2f}]")
        print(f"Target variable ({self.config['variables']['target']}) range: "
              f"[{self.target_values.min():.3f}, {self.target_values.max():.3f}]")
        
    def generate_laplacian_features(self):
        """Generate Laplacian eigenmap features using the same parameters as meta-ensemble"""
        print("\n=== Generating Laplacian Eigenmap Features ===")
        
        # Use the same parameters as in the meta-ensemble
        n_components = self.config['laplacian_params']['n_eigenmaps']
        k_neighbors = self.config['laplacian_params']['k_for_swm']
        
        print(f"Parameters: n_components={n_components}, k_neighbors={k_neighbors}")
        
        # Create Laplacian eigenmap generator
        laplacian = LaplacianEigenmap(
            n_components=n_components,
            k_neighbors=k_neighbors,
            target_crs=None
        )
        
        # Generate features
        self.laplacian_features = laplacian.fit_transform(self.coords)
        self.feature_names = laplacian.eigenmap_cols_
        
        print(f"Generated {self.laplacian_features.shape[1]} Laplacian eigenmap features")
        print(f"Feature matrix shape: {self.laplacian_features.shape}")
        
        # Display some statistics
        print(f"\nLaplacian feature statistics:")
        print(f"  Mean absolute value: {np.abs(self.laplacian_features).mean():.6f}")
        print(f"  Standard deviation: {self.laplacian_features.std():.6f}")
        print(f"  Value range: [{self.laplacian_features.min():.6f}, {self.laplacian_features.max():.6f}]")
        
    def compute_tsne_embeddings(self, perplexity_values=None, n_components_list=None):
        """Compute t-SNE embeddings with different parameters"""
        print("\n=== Computing t-SNE Embeddings ===")
        
        if self.laplacian_features is None:
            raise ValueError("Must generate Laplacian features first")
        
        # Use config parameters if not provided
        if perplexity_values is None:
            perplexity_values = self.config['tsne_params']['perplexity_values']
        if n_components_list is None:
            n_components_list = self.config['tsne_params']['n_components_list']
        
        # Standardize features for t-SNE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.laplacian_features)
        
        self.tsne_results = {}
        
        for n_components in n_components_list:
            for perplexity in perplexity_values:
                print(f"Computing {n_components}D t-SNE with perplexity={perplexity}...")
                
                # Apply PCA preprocessing for high-dimensional data
                if X_scaled.shape[1] > 50:
                    pca = PCA(n_components=50)
                    X_pca = pca.fit_transform(X_scaled)
                    print(f"  Applied PCA preprocessing: {X_scaled.shape[1]} -> {X_pca.shape[1]} dimensions")
                else:
                    X_pca = X_scaled
                
                # Compute t-SNE using config parameters
                tsne = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    max_iter=self.config['tsne_params']['n_iter'],
                    random_state=self.config['tsne_params']['random_state'],
                    learning_rate=self.config['tsne_params']['learning_rate'],
                    init=self.config['tsne_params']['init']
                )
                
                embedding = tsne.fit_transform(X_pca)
                
                key = f"{n_components}D_perp{perplexity}"
                self.tsne_results[key] = {
                    'embedding': embedding,
                    'perplexity': perplexity,
                    'n_components': n_components,
                    'kl_divergence': tsne.kl_divergence_
                }
                
                print(f"  Final KL divergence: {tsne.kl_divergence_:.4f}")
        
        print(f"\nGenerated {len(self.tsne_results)} t-SNE embeddings")
        
    def create_static_visualizations(self):
        """Create static matplotlib visualizations"""
        print("\n=== Creating Static Visualizations ===")
        
        # Create figure with subplots for different coloring schemes
        n_perplexities = len(set(result['perplexity'] for result in self.tsne_results.values() if result['n_components'] == 2))
        fig, axes = plt.subplots(3, n_perplexities, figsize=(5*n_perplexities, 15))
        if n_perplexities == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot 2D t-SNE results with different coloring schemes
        perplexity_values = sorted(set(result['perplexity'] for result in self.tsne_results.values() if result['n_components'] == 2))
        
        for i, perplexity in enumerate(perplexity_values):
            key = f"2D_perp{perplexity}"
            embedding = self.tsne_results[key]['embedding']
            kl_div = self.tsne_results[key]['kl_divergence']
            
            # Plot 1: Colored by spatial coordinates (X)
            scatter = axes[0, i].scatter(
                embedding[:, 0], embedding[:, 1],
                c=self.coords[:, 0], cmap='viridis',
                alpha=0.7, s=20
            )
            axes[0, i].set_title(f'Colored by X-coordinate\nPerplexity={perplexity}, KL={kl_div:.3f}')
            axes[0, i].set_xlabel('t-SNE 1')
            axes[0, i].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[0, i], label='X-coordinate')
            
            # Plot 2: Colored by spatial coordinates (Y)
            scatter = axes[1, i].scatter(
                embedding[:, 0], embedding[:, 1],
                c=self.coords[:, 1], cmap='plasma',
                alpha=0.7, s=20
            )
            axes[1, i].set_title(f'Colored by Y-coordinate\nPerplexity={perplexity}')
            axes[1, i].set_xlabel('t-SNE 1')
            axes[1, i].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[1, i], label='Y-coordinate')
            
            # Plot 3: Colored by target variable
            scatter = axes[2, i].scatter(
                embedding[:, 0], embedding[:, 1],
                c=self.target_values, cmap='RdYlBu_r',
                alpha=0.7, s=20
            )
            axes[2, i].set_title(f'Colored by {self.config["variables"]["target"]}\nPerplexity={perplexity}')
            axes[2, i].set_xlabel('t-SNE 1')
            axes[2, i].set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=axes[2, i], label=self.config["variables"]["target"])
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.paths.laplacian_tsne_plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Static visualization saved to: {output_path}")
        
        # Create additional spatial comparison plot
        self._create_spatial_comparison_plot()
        
    def _create_spatial_comparison_plot(self):
        """Create a side-by-side comparison of spatial and t-SNE embeddings"""
        print("Creating spatial comparison plot...")
        
        # Use the best 2D embedding (lowest KL divergence)
        best_2d_key = min([k for k in self.tsne_results.keys() if '2D' in k], 
                         key=lambda k: self.tsne_results[k]['kl_divergence'])
        best_embedding = self.tsne_results[best_2d_key]['embedding']
        best_perplexity = self.tsne_results[best_2d_key]['perplexity']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original spatial coordinates
        scatter1 = axes[0].scatter(
            self.coords[:, 0], self.coords[:, 1],
            c=self.target_values, cmap='RdYlBu_r',
            alpha=0.7, s=20
        )
        axes[0].set_title(f'Original Spatial Distribution\n({self.config["variables"]["target"]})')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        axes[0].set_aspect('equal')
        plt.colorbar(scatter1, ax=axes[0], label=self.config["variables"]["target"])
        
        # t-SNE embedding
        scatter2 = axes[1].scatter(
            best_embedding[:, 0], best_embedding[:, 1],
            c=self.target_values, cmap='RdYlBu_r',
            alpha=0.7, s=20
        )
        axes[1].set_title(f't-SNE Embedding of Laplacian Features\n(Perplexity={best_perplexity})')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1], label=self.config["variables"]["target"])
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = str(self.paths.laplacian_tsne_plot).replace('.png', '_spatial_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Spatial comparison plot saved to: {comparison_path}")
        
    def create_interactive_visualizations(self):
        """Create interactive plotly visualizations"""
        print("\n=== Creating Interactive Visualizations ===")
        
        # Create subplot for different perplexity values
        perplexity_values = sorted(set(result['perplexity'] for result in self.tsne_results.values() if result['n_components'] == 2))
        n_plots = len(perplexity_values)
        
        # Create 2D interactive plots with multiple coloring options
        self._create_interactive_2d_plots(perplexity_values)
        
        # Create 3D visualization if available
        if any(result['n_components'] == 3 for result in self.tsne_results.values()):
            self._create_interactive_3d_plot()
        
        # Create interactive spatial comparison
        self._create_interactive_spatial_comparison()
        
    def _create_interactive_2d_plots(self, perplexity_values):
        """Create 2D interactive plots with different coloring schemes"""
        # Create tabs for different coloring schemes
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Find the best embedding for detailed analysis
        best_2d_key = min([k for k in self.tsne_results.keys() if '2D' in k], 
                         key=lambda k: self.tsne_results[k]['kl_divergence'])
        best_embedding = self.tsne_results[best_2d_key]['embedding']
        best_perplexity = self.tsne_results[best_2d_key]['perplexity']
        
        # Create comprehensive interactive plot with multiple views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Colored by {self.config["variables"]["target"]}',
                'Colored by X-coordinate',
                'Colored by Y-coordinate', 
                'Spatial Clustering'
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # Create enhanced hover text
        hover_text = [
            f"Point {idx}<br>"
            f"X: {self.coords[idx, 0]:.2f}<br>"
            f"Y: {self.coords[idx, 1]:.2f}<br>"
            f"{self.config['variables']['target']}: {self.target_values[idx]:.3f}<br>"
            f"t-SNE1: {best_embedding[idx, 0]:.3f}<br>"
            f"t-SNE2: {best_embedding[idx, 1]:.3f}<br>"
            f"Distance from origin: {np.sqrt(self.coords[idx, 0]**2 + self.coords[idx, 1]**2):.2f}"
            for idx in range(len(best_embedding))
        ]
        
        # Plot 1: Colored by target variable
        fig.add_trace(
            go.Scatter(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=self.target_values,
                    colorscale='RdYlBu_r',
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title=self.config["variables"]["target"], x=0.45, len=0.4)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Target Variable'
            ),
            row=1, col=1
        )
        
        # Plot 2: Colored by X-coordinate
        fig.add_trace(
            go.Scatter(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=self.coords[:, 0],
                    colorscale='Viridis',
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title='X-coordinate', x=1.02, len=0.4)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='X-coordinate'
            ),
            row=1, col=2
        )
        
        # Plot 3: Colored by Y-coordinate
        fig.add_trace(
            go.Scatter(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=self.coords[:, 1],
                    colorscale='Plasma',
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title='Y-coordinate', x=0.45, y=0.1, len=0.4)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Y-coordinate'
            ),
            row=2, col=1
        )
        
        # Plot 4: Spatial clustering (distance from centroid)
        centroid = self.coords.mean(axis=0)
        distances = np.sqrt(((self.coords - centroid)**2).sum(axis=1))
        fig.add_trace(
            go.Scatter(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=distances,
                    colorscale='Turbo',
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title='Distance from Centroid', x=1.02, y=0.1, len=0.4)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Spatial Distance'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Interactive t-SNE Visualization of Laplacian Eigenmaps<br>Best Configuration: Perplexity={best_perplexity}, KL divergence={self.tsne_results[best_2d_key]['kl_divergence']:.4f}",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(2):
            for j in range(2):
                fig.update_xaxes(title_text="t-SNE 1", row=i+1, col=j+1)
                fig.update_yaxes(title_text="t-SNE 2", row=i+1, col=j+1)
        
        # Save interactive plot
        interactive_path = str(self.paths.laplacian_tsne_plot).replace('.png', '_interactive.html')
        fig.write_html(interactive_path)
        print(f"2D interactive visualization saved to: {interactive_path}")
        
    def _create_interactive_3d_plot(self):
        """Create 3D interactive visualization"""
        print("Creating 3D interactive visualization...")
        
        # Use the best 3D result (lowest perplexity for stability)
        best_3d_key = min([k for k in self.tsne_results.keys() if '3D' in k], 
                         key=lambda k: self.tsne_results[k]['perplexity'])
        embedding_3d = self.tsne_results[best_3d_key]['embedding']
        
        # Create enhanced hover text for 3D
        hover_text_3d = [
            f"Point {idx}<br>"
            f"X: {self.coords[idx, 0]:.2f}<br>"
            f"Y: {self.coords[idx, 1]:.2f}<br>"
            f"{self.config['variables']['target']}: {self.target_values[idx]:.3f}<br>"
            f"t-SNE1: {embedding_3d[idx, 0]:.3f}<br>"
            f"t-SNE2: {embedding_3d[idx, 1]:.3f}<br>"
            f"t-SNE3: {embedding_3d[idx, 2]:.3f}"
            for idx in range(len(embedding_3d))
        ]
        
        fig_3d = go.Figure(data=[
            go.Scatter3d(
                x=embedding_3d[:, 0],
                y=embedding_3d[:, 1],
                z=embedding_3d[:, 2],
                mode='markers',
                marker=dict(
                    color=self.target_values,
                    colorscale='RdYlBu_r',
                    size=4,
                    opacity=0.7,
                    colorbar=dict(title=self.config["variables"]["target"])
                ),
                text=hover_text_3d,
                hovertemplate='%{text}<extra></extra>',
                name='3D t-SNE'
            )
        ])
        
        fig_3d.update_layout(
            title=f"3D t-SNE Visualization (Perplexity={self.tsne_results[best_3d_key]['perplexity']})",
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700
        )
        
        # Save 3D plot
        interactive_3d_path = str(self.paths.laplacian_tsne_plot).replace('.png', '_3d_interactive.html')
        fig_3d.write_html(interactive_3d_path)
        print(f"3D interactive visualization saved to: {interactive_3d_path}")
        
    def _create_interactive_spatial_comparison(self):
        """Create interactive spatial comparison plot"""
        print("Creating interactive spatial comparison...")
        
        # Use the best 2D embedding
        best_2d_key = min([k for k in self.tsne_results.keys() if '2D' in k], 
                         key=lambda k: self.tsne_results[k]['kl_divergence'])
        best_embedding = self.tsne_results[best_2d_key]['embedding']
        best_perplexity = self.tsne_results[best_2d_key]['perplexity']
        
        # Create side-by-side comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Original Spatial Distribution', 't-SNE Embedding of Laplacian Features'],
            horizontal_spacing=0.1
        )
        
        # Enhanced hover text
        hover_text = [
            f"Point {idx}<br>"
            f"X: {self.coords[idx, 0]:.2f}<br>"
            f"Y: {self.coords[idx, 1]:.2f}<br>"
            f"{self.config['variables']['target']}: {self.target_values[idx]:.3f}"
            for idx in range(len(self.coords))
        ]
        
        # Original spatial coordinates
        fig.add_trace(
            go.Scatter(
                x=self.coords[:, 0],
                y=self.coords[:, 1],
                mode='markers',
                marker=dict(
                    color=self.target_values,
                    colorscale='RdYlBu_r',
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title=self.config["variables"]["target"], x=0.45)
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Spatial'
            ),
            row=1, col=1
        )
        
        # t-SNE embedding
        fig.add_trace(
            go.Scatter(
                x=best_embedding[:, 0],
                y=best_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=self.target_values,
                    colorscale='RdYlBu_r',
                    size=5,
                    opacity=0.7,
                    showscale=False
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='t-SNE'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Spatial vs t-SNE Comparison (Perplexity={best_perplexity})",
            height=600,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="X Coordinate", row=1, col=1)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        
        # Save comparison plot
        comparison_path = str(self.paths.laplacian_tsne_plot).replace('.png', '_spatial_comparison_interactive.html')
        fig.write_html(comparison_path)
        print(f"Interactive spatial comparison saved to: {comparison_path}")
        
    def save_tsne_coordinates(self):
        """Save t-SNE coordinates for further analysis"""
        print("\n=== Saving t-SNE Coordinates ===")
        
        # Create DataFrame with all results
        results_df = pd.DataFrame({
            'X': self.coords[:, 0],
            'Y': self.coords[:, 1],
            self.config["variables"]["target"]: self.target_values
        })
        
        # Add t-SNE coordinates
        for key, result in self.tsne_results.items():
            embedding = result['embedding']
            n_comp = result['n_components']
            perp = result['perplexity']
            
            if n_comp == 2:
                results_df[f'tsne1_perp{perp}'] = embedding[:, 0]
                results_df[f'tsne2_perp{perp}'] = embedding[:, 1]
            elif n_comp == 3:
                results_df[f'tsne1_3d_perp{perp}'] = embedding[:, 0]
                results_df[f'tsne2_3d_perp{perp}'] = embedding[:, 1]
                results_df[f'tsne3_3d_perp{perp}'] = embedding[:, 2]
        
        # Save coordinates
        output_path = self.paths.laplacian_tsne_coords
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        print(f"t-SNE coordinates saved to: {output_path}")
        print(f"Saved {len(results_df)} observations with {len(results_df.columns)} features")
        
    def analyze_spatial_structure(self):
        """Analyze the spatial structure revealed by t-SNE"""
        print("\n=== Analyzing Spatial Structure ===")
        
        # Find the best 2D embedding (lowest KL divergence)
        best_2d_key = min([k for k in self.tsne_results.keys() if '2D' in k], 
                         key=lambda k: self.tsne_results[k]['kl_divergence'])
        best_embedding = self.tsne_results[best_2d_key]['embedding']
        best_perplexity = self.tsne_results[best_2d_key]['perplexity']
        best_kl = self.tsne_results[best_2d_key]['kl_divergence']
        
        print(f"Best 2D embedding: Perplexity={best_perplexity}, KL divergence={best_kl:.4f}")
        
        # Compute correlations between t-SNE dimensions and spatial coordinates
        from scipy.stats import pearsonr, spearmanr
        
        correlations = {}
        for i, tsne_dim in enumerate(['t-SNE 1', 't-SNE 2']):
            for j, coord_name in enumerate(['X', 'Y']):
                pearson_r, pearson_p = pearsonr(best_embedding[:, i], self.coords[:, j])
                spearman_r, spearman_p = spearmanr(best_embedding[:, i], self.coords[:, j])
                
                correlations[f"{tsne_dim} vs {coord_name}"] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
        
        print("\nCorrelations between t-SNE dimensions and spatial coordinates:")
        for comparison, stats in correlations.items():
            print(f"  {comparison}:")
            print(f"    Pearson r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.3f})")
            print(f"    Spearman r={stats['spearman_r']:.3f} (p={stats['spearman_p']:.3f})")
        
        # Analyze relationship with target variable
        target_correlations = {}
        for i, tsne_dim in enumerate(['t-SNE 1', 't-SNE 2']):
            pearson_r, pearson_p = pearsonr(best_embedding[:, i], self.target_values)
            spearman_r, spearman_p = spearmanr(best_embedding[:, i], self.target_values)
            
            target_correlations[tsne_dim] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
        
        print(f"\nCorrelations between t-SNE dimensions and {self.config['variables']['target']}:")
        for dim, stats in target_correlations.items():
            print(f"  {dim}:")
            print(f"    Pearson r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.3f})")
            print(f"    Spearman r={stats['spearman_r']:.3f} (p={stats['spearman_p']:.3f})")
        
        return correlations, target_correlations
        
    def run_full_analysis(self, data_file=None):
        """Run the complete t-SNE analysis pipeline"""
        print("Starting Laplacian Eigenmap t-SNE Analysis")
        print("=" * 50)
        
        try:
            # Load data
            self.load_data(data_file)
            
            # Generate Laplacian features
            self.generate_laplacian_features()
            
            # Compute t-SNE embeddings
            self.compute_tsne_embeddings()
            
            # Create visualizations
            self.create_static_visualizations()
            self.create_interactive_visualizations()
            
            # Save coordinates
            self.save_tsne_coordinates()
            
            # Analyze structure
            spatial_corr, target_corr = self.analyze_spatial_structure()
            
            print("\n" + "=" * 50)
            print("Analysis completed successfully!")
            print(f"Static plots saved to: {self.paths.laplacian_tsne_plot}")
            print(f"Interactive plots saved to: {str(self.paths.laplacian_tsne_plot).replace('.png', '_interactive.html')}")
            print(f"Coordinates saved to: {self.paths.laplacian_tsne_coords}")
            
            return spatial_corr, target_corr
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Main function to run the t-SNE visualization"""
    try:
        # Create visualizer
        visualizer = LaplacianTSNEVisualizer()
        
        # Run full analysis
        spatial_correlations, target_correlations = visualizer.run_full_analysis()
        
        if spatial_correlations is not None:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()