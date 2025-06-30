"""
Data generator for synthetic signals with varying amplitude and frequency.

This module generates 1D time-series signals composed of sine waves with noise
for training the inverse parameter estimation model.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Global constants for signal generation
SIGNAL_LENGTH = 1024  # Number of points in each signal
T_MAX = 4.0          # Time duration of the signal in seconds
SAMPLING_RATE = SIGNAL_LENGTH / T_MAX  # Samples per second

def generate_signal(amplitude: float, frequency: float, 
                   noise_level: float = 0.1, 
                   length: int = SIGNAL_LENGTH,
                   t_max: float = T_MAX,
                   random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generates a single sine wave signal with additive Gaussian noise.
    
    Args:
        amplitude (float): Amplitude of the sine wave
        frequency (float): Frequency of the sine wave in Hz
        noise_level (float): Standard deviation of noise relative to amplitude
        length (int): Number of sample points
        t_max (float): Duration of signal in seconds
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.ndarray: Generated noisy signal of shape (length,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create time vector
    t = np.linspace(0, t_max, length)
    
    # Generate pure sine wave
    pure_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * amplitude, length)
    
    return pure_signal + noise

def generate_dataset(num_samples: int = 2000,
                    amp_range: Tuple[float, float] = (0.5, 5.0),
                    freq_range: Tuple[float, float] = (1.0, 10.0),
                    noise_level: float = 0.1,
                    save_path: str = 'signals.npz',
                    random_seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset of signals and their corresponding parameters.
    
    Args:
        num_samples (int): Number of signals to generate
        amp_range (tuple): (min_amplitude, max_amplitude)
        freq_range (tuple): (min_frequency, max_frequency) in Hz
        noise_level (float): Standard deviation of noise relative to amplitude
        save_path (str): Path to save the dataset
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: (signals, labels) where signals has shape (num_samples, SIGNAL_LENGTH)
               and labels has shape (num_samples, 2) with [amplitude, frequency]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    signals = []
    labels = []
    
    print(f"Generating {num_samples} synthetic signals...")
    print(f"Amplitude range: {amp_range}")
    print(f"Frequency range: {freq_range} Hz")
    print(f"Noise level: {noise_level}")
    
    for i in range(num_samples):
        # Generate random parameters
        amplitude = np.random.uniform(*amp_range)
        frequency = np.random.uniform(*freq_range)
        
        # Create the signal
        signal = generate_signal(amplitude, frequency, noise_level)
        
        signals.append(signal)
        labels.append([amplitude, frequency])
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{num_samples} signals")
    
    # Convert to numpy arrays
    signals = np.array(signals)
    labels = np.array(labels)
    
    # Save the dataset
    np.savez_compressed(save_path, signals=signals, labels=labels)
    print(f"Dataset saved to {save_path}")
    print(f"Dataset shape: signals={signals.shape}, labels={labels.shape}")
    
    return signals, labels

def visualize_sample_signals(signals: np.ndarray, 
                           labels: np.ndarray, 
                           num_samples: int = 6,
                           save_path: Optional[str] = None) -> None:
    """
    Visualize a few sample signals from the dataset.
    
    Args:
        signals (np.ndarray): Array of signals
        labels (np.ndarray): Array of corresponding labels
        num_samples (int): Number of signals to plot
        save_path (str, optional): Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Time vector for plotting
    t = np.linspace(0, T_MAX, SIGNAL_LENGTH)
    
    # Select random samples
    indices = np.random.choice(len(signals), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        signal = signals[idx]
        amplitude, frequency = labels[idx]
        
        axes[i].plot(t, signal, 'b-', alpha=0.7, linewidth=1)
        axes[i].set_title(f'Signal {idx}: A={amplitude:.2f}, f={frequency:.2f}Hz')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample signals saved to {save_path}")
    
    plt.show()

def analyze_dataset_statistics(labels: np.ndarray) -> None:
    """
    Analyze and print statistics of the generated dataset.
    
    Args:
        labels (np.ndarray): Array of labels with shape (num_samples, 2)
    """
    amplitudes = labels[:, 0]
    frequencies = labels[:, 1]
    
    print("\nDataset Statistics:")
    print("=" * 40)
    print(f"Number of samples: {len(labels)}")
    print(f"Signal length: {SIGNAL_LENGTH} points")
    print(f"Signal duration: {T_MAX} seconds")
    print(f"Sampling rate: {SAMPLING_RATE:.1f} Hz")
    
    print(f"\nAmplitude statistics:")
    print(f"  Mean: {amplitudes.mean():.3f}")
    print(f"  Std: {amplitudes.std():.3f}")
    print(f"  Min: {amplitudes.min():.3f}")
    print(f"  Max: {amplitudes.max():.3f}")
    
    print(f"\nFrequency statistics:")
    print(f"  Mean: {frequencies.mean():.3f} Hz")
    print(f"  Std: {frequencies.std():.3f} Hz")
    print(f"  Min: {frequencies.min():.3f} Hz")
    print(f"  Max: {frequencies.max():.3f} Hz")

if __name__ == '__main__':
    # Generate the dataset
    signals, labels = generate_dataset(
        num_samples=2000,
        amp_range=(0.5, 5.0),
        freq_range=(1.0, 10.0),
        noise_level=0.1,
        save_path='signals.npz',
        random_seed=42
    )
    
    # Analyze dataset statistics
    analyze_dataset_statistics(labels)
    
    # Visualize sample signals
    visualize_sample_signals(
        signals, labels, 
        num_samples=6,
        save_path='visualizations/plots/signal_examples.png'
    )
    
    print("\nDataset generation complete!")
    print("Run 'python data_generator.py' to regenerate the dataset.")