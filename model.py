"""
Model implementation using scikit-learn MLPRegressor for CPU-optimized training.

This module provides the neural network architecture and training logic 
for predicting signal parameters from Fourier features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Optional, Dict, Any
import pickle
import time

class SignalModel:
    """
    A neural network model for predicting signal parameters from Fourier features.
    
    Uses scikit-learn's MLPRegressor for CPU-optimized training and inference.
    """
    
    def __init__(self, 
                 hidden_layer_sizes: Tuple[int, ...] = (64, 64, 64),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 max_iter: int = 500,
                 random_state: int = 42):
        """
        Initialize the neural network model.
        
        Args:
            hidden_layer_sizes (tuple): Number of neurons in each hidden layer
            activation (str): Activation function ('relu', 'tanh', 'logistic')
            solver (str): Optimizer ('adam', 'lbfgs', 'sgd')
            alpha (float): L2 regularization parameter
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X (np.ndarray): Feature matrix with shape (n_samples, n_features)
            y (np.ndarray): Target matrix with shape (n_samples, 2) [amplitude, frequency]
            test_size (float): Fraction of data to use for testing
            
        Returns:
            dict: Training results and metrics
        """
        print("Starting model training...")
        start_time = time.time()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'n_iterations': self.model.n_iter_,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'loss_curve': self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else None
        }
        
        self.is_trained = True
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Converged after {self.model.n_iter_} iterations")
        print(f"Test R² Score: Amplitude={test_metrics['r2_amplitude']:.4f}, "
              f"Frequency={test_metrics['r2_frequency']:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted parameters [amplitude, frequency]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various performance metrics."""
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Per-parameter metrics
        # Amplitude (first column)
        metrics['mse_amplitude'] = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        metrics['mae_amplitude'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        metrics['r2_amplitude'] = r2_score(y_true[:, 0], y_pred[:, 0])
        
        # Frequency (second column)
        metrics['mse_frequency'] = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        metrics['mae_frequency'] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        metrics['r2_frequency'] = r2_score(y_true[:, 1], y_pred[:, 1])
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.training_history = model_data['training_history']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

def visualize_training_results(training_history: Dict[str, Any],
                             save_path: Optional[str] = None) -> None:
    """
    Visualize training results including loss curves and metrics.
    
    Args:
        training_history (dict): Training history from model.train()
        save_path (str, optional): Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    if training_history['loss_curve'] is not None:
        axes[0, 0].plot(training_history['loss_curve'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss Curve')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Loss curve not available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Training Loss Curve')
    
    # Training vs Test metrics comparison
    train_metrics = training_history['train_metrics']
    test_metrics = training_history['test_metrics']
    
    metrics_names = ['MSE', 'MAE', 'R²']
    train_values = [train_metrics['mse'], train_metrics['mae'], train_metrics['r2']]
    test_values = [test_metrics['mse'], test_metrics['mae'], test_metrics['r2']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, train_values, width, label='Train', alpha=0.7)
    axes[0, 1].bar(x + width/2, test_values, width, label='Test', alpha=0.7)
    axes[0, 1].set_title('Training vs Test Metrics')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-parameter R² scores
    amp_r2 = [train_metrics['r2_amplitude'], test_metrics['r2_amplitude']]
    freq_r2 = [train_metrics['r2_frequency'], test_metrics['r2_frequency']]
    
    x = ['Train', 'Test']
    axes[1, 0].plot(x, amp_r2, 'ro-', linewidth=2, label='Amplitude', markersize=8)
    axes[1, 0].plot(x, freq_r2, 'bo-', linewidth=2, label='Frequency', markersize=8)
    axes[1, 0].set_title('R² Score by Parameter')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training summary
    summary_text = f"""Training Summary:
    
Time: {training_history['training_time']:.2f} seconds
Iterations: {training_history['n_iterations']}

Test Performance:
MSE: {test_metrics['mse']:.4f}
MAE: {test_metrics['mae']:.4f}
R²: {test_metrics['r2']:.4f}

Amplitude R²: {test_metrics['r2_amplitude']:.4f}
Frequency R²: {test_metrics['r2_frequency']:.4f}"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results saved to {save_path}")
    
    plt.show()

def visualize_predictions(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_path: Optional[str] = None) -> None:
    """
    Create scatter plots comparing true vs predicted values.
    
    Args:
        y_true (np.ndarray): True parameter values
        y_pred (np.ndarray): Predicted parameter values
        save_path (str, optional): Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Amplitude predictions
    axes[0].scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, s=20)
    
    # Perfect prediction line
    amp_min, amp_max = y_true[:, 0].min(), y_true[:, 0].max()
    axes[0].plot([amp_min, amp_max], [amp_min, amp_max], 'r--', linewidth=2, label='Perfect')
    
    axes[0].set_xlabel('True Amplitude')
    axes[0].set_ylabel('Predicted Amplitude')
    axes[0].set_title('Amplitude Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Calculate R² for amplitude
    r2_amp = r2_score(y_true[:, 0], y_pred[:, 0])
    axes[0].text(0.05, 0.95, f'R² = {r2_amp:.4f}', transform=axes[0].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Frequency predictions
    axes[1].scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, s=20, color='orange')
    
    # Perfect prediction line
    freq_min, freq_max = y_true[:, 1].min(), y_true[:, 1].max()
    axes[1].plot([freq_min, freq_max], [freq_min, freq_max], 'r--', linewidth=2, label='Perfect')
    
    axes[1].set_xlabel('True Frequency (Hz)')
    axes[1].set_ylabel('Predicted Frequency (Hz)')
    axes[1].set_title('Frequency Prediction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Calculate R² for frequency
    r2_freq = r2_score(y_true[:, 1], y_pred[:, 1])
    axes[1].text(0.05, 0.95, f'R² = {r2_freq:.4f}', transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction scatter plots saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # Test the model implementation
    try:
        # Load data
        from feature_extractor import extract_features_batch
        
        data = np.load('signals.npz')
        signals = data['signals']
        labels = data['labels']
        
        print("Testing model implementation...")
        print(f"Dataset shape: signals={signals.shape}, labels={labels.shape}")
        
        # Extract features
        print("Extracting Fourier features...")
        num_modes = 20
        features = extract_features_batch(signals, num_modes)
        print(f"Features shape: {features.shape}")
        
        # Create and train model
        print("Creating and training model...")
        model = SignalModel(
            hidden_layer_sizes=(64, 64, 64),
            max_iter=300,
            random_state=42
        )
        
        # Train the model
        training_history = model.train(features, labels, test_size=0.2)
        
        # Visualize results
        visualize_training_results(
            training_history,
            save_path='visualizations/plots/training_results.png'
        )
        
        # Test predictions
        X_test = features[-100:]  # Last 100 samples
        y_test = labels[-100:]
        y_pred = model.predict(X_test)
        
        visualize_predictions(
            y_test, y_pred,
            save_path='visualizations/plots/prediction_scatter.png'
        )
        
        # Save the model
        model.save_model('trained_model.pkl')
        
        print("Model implementation test complete!")
        
    except FileNotFoundError:
        print("signals.npz not found. Run data_generator.py first to create the dataset.")
        
        # Create dummy data for testing
        print("Creating dummy data for testing...")
        np.random.seed(42)
        dummy_features = np.random.randn(100, 40)  # 20 modes * 2
        dummy_labels = np.random.uniform([0.5, 1.0], [5.0, 10.0], (100, 2))
        
        model = SignalModel(hidden_layer_sizes=(32, 32), max_iter=100)
        training_history = model.train(dummy_features, dummy_labels)
        
        print(f"Dummy test completed. Training time: {training_history['training_time']:.2f}s")
        print("Run the full pipeline: data_generator.py -> feature_extractor.py -> model.py")