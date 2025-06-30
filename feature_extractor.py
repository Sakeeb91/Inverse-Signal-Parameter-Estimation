"""
Feature extraction module using Fast Fourier Transform (FFT).

This module implements the core feature engineering approach inspired by 
Fourier Neural Operators, transforming high-dimensional time-series signals
into compact, information-rich frequency domain representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
from scipy.fft import fft, fftfreq

def extract_fourier_features(signal: np.ndarray, 
                           num_modes: int,
                           normalize: bool = True) -> np.ndarray:
    """
    Extracts Fourier features from a 1D signal using FFT.
    
    This is the core feature engineering function that transforms a high-dimensional
    signal into a compact representation using the first k Fourier modes.
    
    Args:
        signal (np.ndarray): Input signal of shape (n_samples,)
        num_modes (int): Number of Fourier coefficients (modes) to keep
        normalize (bool): Whether to normalize features by signal length
        
    Returns:
        np.ndarray: Feature vector of size 2 * num_modes containing
                   concatenated real and imaginary parts
    """
    # Apply Fast Fourier Transform
    fft_coeffs = fft(signal)
    
    # Keep only the first num_modes coefficients (low frequencies)
    truncated_coeffs = fft_coeffs[:num_modes]
    
    # Normalize by signal length if requested
    if normalize:
        truncated_coeffs = truncated_coeffs / len(signal)
    
    # Separate real and imaginary parts and concatenate
    real_parts = np.real(truncated_coeffs)
    imag_parts = np.imag(truncated_coeffs)
    
    features = np.concatenate([real_parts, imag_parts])
    
    return features

def extract_features_batch(signals: np.ndarray, 
                          num_modes: int,
                          normalize: bool = True) -> np.ndarray:
    """
    Extract Fourier features from a batch of signals.
    
    Args:
        signals (np.ndarray): Batch of signals with shape (n_samples, signal_length)
        num_modes (int): Number of Fourier modes to extract
        normalize (bool): Whether to normalize features
        
    Returns:
        np.ndarray: Feature matrix with shape (n_samples, 2 * num_modes)
    """
    n_samples = signals.shape[0]
    feature_dim = 2 * num_modes
    features = np.zeros((n_samples, feature_dim))
    
    for i in range(n_samples):
        features[i] = extract_fourier_features(signals[i], num_modes, normalize)
    
    return features

def analyze_frequency_content(signal: np.ndarray, 
                            sampling_rate: float,
                            title: str = "Frequency Analysis") -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze the frequency content of a signal and visualize it.
    
    Args:
        signal (np.ndarray): Input signal
        sampling_rate (float): Sampling rate in Hz
        title (str): Title for the plot
        
    Returns:
        tuple: (frequencies, magnitudes) for further analysis
    """
    # Compute FFT
    fft_coeffs = fft(signal)
    frequencies = fftfreq(len(signal), 1/sampling_rate)
    
    # Take magnitude and only positive frequencies
    magnitudes = np.abs(fft_coeffs)
    pos_freq_idx = frequencies >= 0
    frequencies = frequencies[pos_freq_idx]
    magnitudes = magnitudes[pos_freq_idx]
    
    return frequencies, magnitudes

def visualize_fourier_analysis(signal: np.ndarray,
                             sampling_rate: float,
                             num_modes_list: list = [5, 10, 20, 40],
                             save_path: Optional[str] = None) -> None:
    """
    Visualize the effect of different numbers of Fourier modes.
    
    Args:
        signal (np.ndarray): Input signal to analyze
        sampling_rate (float): Sampling rate in Hz
        num_modes_list (list): List of different mode numbers to compare
        save_path (str, optional): Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Time vector
    t = np.linspace(0, len(signal)/sampling_rate, len(signal))
    
    # Original signal
    axes[0, 0].plot(t, signal, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency spectrum
    frequencies, magnitudes = analyze_frequency_content(signal, sampling_rate)
    axes[0, 1].plot(frequencies[:len(frequencies)//2], magnitudes[:len(magnitudes)//2])
    axes[0, 1].set_title('Frequency Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature dimensionalities for different modes
    mode_dims = [2 * modes for modes in num_modes_list]
    axes[0, 2].bar(range(len(num_modes_list)), mode_dims, alpha=0.7)
    axes[0, 2].set_title('Feature Dimensions vs Fourier Modes')
    axes[0, 2].set_xlabel('Mode Configuration')
    axes[0, 2].set_ylabel('Feature Dimension')
    axes[0, 2].set_xticks(range(len(num_modes_list)))
    axes[0, 2].set_xticklabels([f'{m} modes' for m in num_modes_list])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Feature extraction for different mode numbers
    for i, num_modes in enumerate(num_modes_list[:3]):
        features = extract_fourier_features(signal, num_modes)
        
        ax = axes[1, i]
        ax.plot(features[:num_modes], 'ro-', label='Real parts', alpha=0.7)
        ax.plot(features[num_modes:], 'bo-', label='Imaginary parts', alpha=0.7)
        ax.set_title(f'Features with {num_modes} modes')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fourier analysis saved to {save_path}")
    
    plt.show()

def compare_feature_dimensions(signals: np.ndarray,
                             mode_range: range = range(5, 51, 5),
                             save_path: Optional[str] = None) -> dict:
    """
    Compare feature extraction with different numbers of Fourier modes.
    
    Args:
        signals (np.ndarray): Batch of signals for analysis
        mode_range (range): Range of mode numbers to test
        save_path (str, optional): Path to save the comparison plot
        
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {
        'modes': list(mode_range),
        'feature_dims': [],
        'compression_ratios': [],
        'sample_features': []
    }
    
    original_dim = signals.shape[1]
    
    for num_modes in mode_range:
        feature_dim = 2 * num_modes
        results['feature_dims'].append(feature_dim)
        results['compression_ratios'].append(original_dim / feature_dim)
        
        # Extract features for first signal as example
        sample_features = extract_fourier_features(signals[0], num_modes)
        results['sample_features'].append(sample_features)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Feature dimensions
    axes[0].plot(results['modes'], results['feature_dims'], 'bo-', linewidth=2)
    axes[0].axhline(y=original_dim, color='r', linestyle='--', 
                   label=f'Original dim: {original_dim}')
    axes[0].set_xlabel('Number of Fourier Modes')
    axes[0].set_ylabel('Feature Dimension')
    axes[0].set_title('Feature Dimension vs Fourier Modes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Compression ratios
    axes[1].plot(results['modes'], results['compression_ratios'], 'go-', linewidth=2)
    axes[1].set_xlabel('Number of Fourier Modes')
    axes[1].set_ylabel('Compression Ratio')
    axes[1].set_title('Compression Ratio vs Fourier Modes')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature comparison saved to {save_path}")
    
    plt.show()
    
    return results

def get_optimal_modes(signals: np.ndarray,
                     labels: np.ndarray,
                     mode_range: range = range(5, 51, 5)) -> int:
    """
    Determine optimal number of Fourier modes based on information content.
    
    This is a simplified heuristic that looks at the variance explained
    by different numbers of modes.
    
    Args:
        signals (np.ndarray): Training signals
        labels (np.ndarray): Corresponding labels
        mode_range (range): Range of modes to test
        
    Returns:
        int: Recommended number of modes
    """
    variances = []
    
    for num_modes in mode_range:
        features = extract_features_batch(signals[:100], num_modes)  # Sample for speed
        variance = np.var(features, axis=0).sum()
        variances.append(variance)
    
    # Find the elbow point (simplified)
    variances = np.array(variances)
    # Normalize variances
    variances = variances / variances.max()
    
    # Find point where improvement becomes marginal
    improvements = np.diff(variances)
    optimal_idx = np.where(improvements < 0.1)[0]
    
    if len(optimal_idx) > 0:
        optimal_modes = list(mode_range)[optimal_idx[0]]
    else:
        optimal_modes = list(mode_range)[len(mode_range)//2]  # Middle value as fallback
    
    print(f"Recommended number of Fourier modes: {optimal_modes}")
    return optimal_modes

if __name__ == '__main__':
    # Load sample data for testing
    try:
        data = np.load('signals.npz')
        signals = data['signals']
        labels = data['labels']
        
        print("Testing feature extraction...")
        print(f"Signal shape: {signals.shape}")
        
        # Test single signal feature extraction
        test_signal = signals[0]
        features_10 = extract_fourier_features(test_signal, num_modes=10)
        features_20 = extract_fourier_features(test_signal, num_modes=20)
        
        print(f"Original signal length: {len(test_signal)}")
        print(f"Features with 10 modes: {len(features_10)}")
        print(f"Features with 20 modes: {len(features_20)}")
        
        # Test batch processing
        features_batch = extract_features_batch(signals[:10], num_modes=20)
        print(f"Batch features shape: {features_batch.shape}")
        
        # Visualize analysis
        visualize_fourier_analysis(
            test_signal, 
            sampling_rate=256,  # From data_generator
            save_path='visualizations/plots/fourier_analysis.png'
        )
        
        # Compare different feature dimensions
        compare_feature_dimensions(
            signals[:100],
            save_path='visualizations/plots/feature_comparison.png'
        )
        
        # Find optimal modes
        optimal = get_optimal_modes(signals, labels)
        
        print("Feature extraction module test complete!")
        
    except FileNotFoundError:
        print("signals.npz not found. Run data_generator.py first to create the dataset.")
        
        # Create a simple test signal for demonstration
        print("Creating test signal for demonstration...")
        t = np.linspace(0, 4, 1024)
        test_signal = 2.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(1024)
        
        features = extract_fourier_features(test_signal, num_modes=20)
        print(f"Test signal length: {len(test_signal)}")
        print(f"Extracted features length: {len(features)}")
        print("Run data_generator.py to create the full dataset for complete testing.")