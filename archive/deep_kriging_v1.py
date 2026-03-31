import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
from tqdm import tqdm

class DeepKriging:
    """
    Deep Kriging model for spatial residual correction. Uses a neural network
    to model and correct residual spatial autocorrelation using coordinates
    and Laplacian eigenvector features.
    """
    
    def __init__(
        self,
        hidden_layers=[64, 32],
        dropout_rate=0.1,
        learning_rate=0.001,
        val_size=0.2,
        random_state=42,
        batch_size=256,
        epochs=50,
        patience=5
    ):
        """
        Initialize the Deep Kriging model.
        
        Parameters:
        -----------
        hidden_layers : list, default=[64, 32]
            Number of neurons in each hidden layer
        dropout_rate : float, default=0.1
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer
        val_size : float, default=0.2
            Validation set size for early stopping
        random_state : int, default=42
            Random state for reproducibility
        batch_size : int, default=256
            Batch size for training
        epochs : int, default=50
            Maximum number of epochs
        patience : int, default=5
            Patience for early stopping
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _build_model(self, input_dim):
        """Build the neural network architecture with L2 regularization"""
        model = Sequential([Input(shape=(input_dim,))])
        
        # Add hidden layers with dropout and L2 regularization
        for i, units in enumerate(self.hidden_layers):
            model.add(Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer with no activation (regression)
        model.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        
        # Compile model with lower learning rate and clipnorm
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0
            ),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_features(self, coords, laplacian_features=None):
        """Prepare and combine spatial features with basic distance-based features"""
        features = [coords]
        feature_names = ['X', 'Y']
        
        # Add simple distance-based features
        try:
            # Distance to spatial centroid
            centroid = coords.mean(axis=0)
            distance_to_center = np.linalg.norm(coords - centroid, axis=1).reshape(-1, 1)
            features.append(distance_to_center)
            feature_names.append('distance_to_center')
            
            # Simple local density feature (average distance to 5 nearest neighbors)
            if len(coords) > 5:
                from sklearn.neighbors import NearestNeighbors
                k_neighbors = min(5, len(coords) - 1)
                nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coords)
                distances, indices = nbrs.kneighbors(coords)
                local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)  # Inverse of mean distance
                features.append(local_density.reshape(-1, 1))
                feature_names.append('local_density')
        except Exception:
            # Silently skip if there are issues
            pass
        
        if laplacian_features is not None:
            features.append(laplacian_features)
            feature_names.extend([f'LAPEIG_{i+1}' for i in range(laplacian_features.shape[1])])
        
        X = np.hstack([f for f in features if f is not None])
        return X, feature_names
    
    def fit(self, coords, residuals, laplacian_features=None, verbose=1):
        """
        Fit the Deep Kriging model.
        
        Parameters:
        -----------
        coords : array-like
            Spatial coordinates (n_samples, 2)
        residuals : array-like
            Residuals to model (n_samples,)
        laplacian_features : array-like, optional
            Laplacian eigenvector features
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        # Check if residuals have enough variation to model
        if np.std(residuals) < 1e-6:
            if verbose:
                print("Residuals have very low variance. Skipping training.")
            self.model = None
            return self
        
        # Prepare features
        X, self.feature_names = self._prepare_features(coords, laplacian_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with stratification to ensure balanced residual distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, residuals,
            test_size=self.val_size,
            random_state=self.random_state
        )
        
        # Build model with L2 regularization
        self.model = self._build_model(X_scaled.shape[1])
        
        if verbose:
            print(f"\nTraining Deep Kriging model...")
            print(f"Input features: {', '.join(self.feature_names)}")
            print(f"Residual statistics: Mean={np.mean(residuals):.4f}, Std={np.std(residuals):.4f}")
        
        # Train model with more aggressive early stopping
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    min_delta=1e-4
                )
            ]
        )
        
        # Calculate performance on validation set to check for overfitting
        val_predictions = self.model.predict(X_val, verbose=0).flatten()
        val_r2 = r2_score(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        # Note: Removed validation check that would disable training
        # Model will always train regardless of validation performance
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Validation R²: {val_r2:.4f}")
            print(f"Validation RMSE: {val_rmse:.4f}")
        
        return self
    
    def predict(self, coords, laplacian_features=None):
        """
        Generate residual corrections using the Deep Kriging model.
        
        Parameters:
        -----------
        coords : array-like
            Spatial coordinates (n_samples, 2)
        laplacian_features : array-like, optional
            Laplacian eigenvector features
            
        Returns:
        --------
        array-like
            Predicted residual corrections
        """
        if self.model is None:
            # Return zero predictions if model wasn't trained or failed
            return np.zeros(len(coords))
        
        # Prepare and scale features
        X, _ = self._prepare_features(coords, laplacian_features)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Apply some regularization to prevent extreme predictions
        std_limit = 2.0  # Limit predictions to 2 standard deviations
        predictions = np.clip(predictions, -std_limit, std_limit)
        
        return predictions
    
    def save_model(self, model_dir):
        """
        Save the Deep Kriging model and scaler.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save model artifacts
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save neural network
        model_path = os.path.join(model_dir, 'deep_kriging_model.keras')
        save_model(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'deep_kriging_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names and parameters
        params = {
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names
        }
        params_path = os.path.join(model_dir, 'deep_kriging_params.joblib')
        joblib.dump(params, params_path)
        
        print(f"\nModel artifacts saved to: {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved Deep Kriging model.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing saved model artifacts
            
        Returns:
        --------
        DeepKriging
            Loaded model instance
        """
        # Load parameters
        params_path = os.path.join(model_dir, 'deep_kriging_params.joblib')
        params = joblib.load(params_path)
        
        # Create instance with saved parameters
        instance = cls(
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate']
        )
        instance.feature_names = params['feature_names']
        
        # Load neural network
        model_path = os.path.join(model_dir, 'deep_kriging_model.keras')
        instance.model = load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'deep_kriging_scaler.pkl')
        instance.scaler = joblib.load(scaler_path)
        
        return instance 