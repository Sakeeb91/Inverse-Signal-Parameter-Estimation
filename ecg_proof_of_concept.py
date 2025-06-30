"""
Proof of Concept: ECG Arrhythmia Detection using our FFT Feature Extractor

This demonstrates how our existing Fourier feature extraction can be applied
to real ECG data for medical signal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from feature_extractor import extract_fourier_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def generate_synthetic_ecg(duration=2.0, fs=360, arrhythmia_type='normal'):
    """
    Generate synthetic ECG signals for different arrhythmia types.
    This simulates real ECG data until we download the MIT-BIH database.
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    if arrhythmia_type == 'normal':
        # Normal sinus rhythm ~72 BPM
        heart_rate = 72
        ecg = generate_normal_ecg(t, heart_rate)
        
    elif arrhythmia_type == 'pvc':
        # Premature Ventricular Contraction
        heart_rate = 75
        ecg = generate_pvc_ecg(t, heart_rate)
        
    elif arrhythmia_type == 'atrial_fib':
        # Atrial Fibrillation - irregular rhythm
        ecg = generate_afib_ecg(t)
        
    elif arrhythmia_type == 'tachycardia':
        # Ventricular Tachycardia - fast rate
        heart_rate = 150
        ecg = generate_tachycardia_ecg(t, heart_rate)
        
    else:
        raise ValueError(f"Unknown arrhythmia type: {arrhythmia_type}")
    
    # Add realistic noise
    noise = np.random.normal(0, 0.05, len(ecg))
    return ecg + noise

def generate_normal_ecg(t, heart_rate):
    """Generate normal ECG waveform"""
    freq = heart_rate / 60  # Convert BPM to Hz
    
    # P wave
    p_wave = 0.1 * np.sin(2 * np.pi * freq * t)
    
    # QRS complex (dominant)
    qrs_complex = 0.8 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-((t % (1/freq) - 0.3) / 0.05)**2)
    
    # T wave
    t_wave = 0.2 * np.sin(2 * np.pi * freq * 0.5 * t + np.pi/4)
    
    return p_wave + qrs_complex + t_wave

def generate_pvc_ecg(t, heart_rate):
    """Generate ECG with Premature Ventricular Contractions"""
    normal_ecg = generate_normal_ecg(t, heart_rate)
    
    # Add premature beats at random intervals
    pvc_times = np.random.choice(t, size=int(len(t) * 0.1), replace=False)
    for pvc_time in pvc_times:
        # Wide, abnormal QRS complex
        pvc_mask = np.abs(t - pvc_time) < 0.1
        normal_ecg[pvc_mask] += 0.5 * np.sin(10 * np.pi * (t[pvc_mask] - pvc_time))
    
    return normal_ecg

def generate_afib_ecg(t):
    """Generate Atrial Fibrillation ECG (irregular rhythm)"""
    # Irregular RR intervals
    irregular_rhythm = np.sin(2 * np.pi * np.cumsum(np.random.exponential(1.2, len(t))))
    
    # High frequency fibrillatory waves
    fib_waves = 0.05 * np.sum([np.sin(2 * np.pi * f * t + np.random.random() * 2 * np.pi) 
                              for f in np.random.uniform(4, 8, 5)], axis=0)
    
    return irregular_rhythm + fib_waves

def generate_tachycardia_ecg(t, heart_rate):
    """Generate Ventricular Tachycardia ECG"""
    normal_ecg = generate_normal_ecg(t, heart_rate)
    
    # Add high frequency components
    tachy_component = 0.3 * np.sin(2 * np.pi * 5 * (heart_rate / 60) * t)
    
    return normal_ecg + tachy_component

def create_ecg_dataset(n_samples_per_class=200):
    """Create a balanced ECG dataset for classification"""
    
    arrhythmia_types = ['normal', 'pvc', 'atrial_fib', 'tachycardia']
    
    all_signals = []
    all_labels = []
    
    print("Generating synthetic ECG dataset...")
    
    for i, arrhythmia_type in enumerate(arrhythmia_types):
        print(f"Generating {n_samples_per_class} {arrhythmia_type} ECG signals...")
        
        for _ in range(n_samples_per_class):
            # Generate ECG signal (2 seconds at 360 Hz = 720 samples)
            ecg_signal = generate_synthetic_ecg(
                duration=2.0, 
                fs=360, 
                arrhythmia_type=arrhythmia_type
            )
            
            all_signals.append(ecg_signal)
            all_labels.append(i)  # Numeric labels
    
    return np.array(all_signals), np.array(all_labels), arrhythmia_types

def extract_ecg_features(signals, num_modes=30):
    """Extract features from ECG signals using our FFT method"""
    
    print(f"Extracting FFT features with {num_modes} modes...")
    
    features = []
    for signal in signals:
        # Apply our existing FFT feature extraction
        fft_features = extract_fourier_features(signal, num_modes)
        
        # Add ECG-specific features
        heart_rate = estimate_heart_rate(signal)
        rr_variability = estimate_rr_variability(signal)
        
        # Combine features
        combined_features = np.concatenate([
            fft_features,  # Our FFT features
            [heart_rate, rr_variability]  # Medical features
        ])
        
        features.append(combined_features)
    
    return np.array(features)

def estimate_heart_rate(ecg_signal, fs=360):
    """Estimate heart rate from ECG signal"""
    # Simple peak detection for R-waves
    peaks = find_peaks(ecg_signal, height=np.max(ecg_signal) * 0.5, distance=fs//3)
    
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs  # RR intervals in seconds
        heart_rate = 60 / np.mean(rr_intervals)  # Convert to BPM
    else:
        heart_rate = 70  # Default if no peaks detected
    
    return heart_rate

def estimate_rr_variability(ecg_signal, fs=360):
    """Estimate RR interval variability (HRV)"""
    peaks = find_peaks(ecg_signal, height=np.max(ecg_signal) * 0.5, distance=fs//3)
    
    if len(peaks) > 2:
        rr_intervals = np.diff(peaks) / fs
        rr_variability = np.std(rr_intervals)
    else:
        rr_variability = 0.05  # Default variability
    
    return rr_variability

def find_peaks(signal, height=None, distance=None):
    """Simple peak detection algorithm"""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if height is None or signal[i] >= height:
                if distance is None or not peaks or (i - peaks[-1]) >= distance:
                    peaks.append(i)
    return np.array(peaks)

def train_ecg_classifier(features, labels):
    """Train ECG arrhythmia classifier"""
    
    print("Training ECG classifier...")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train Random Forest classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    
    return classifier, X_test, y_test, y_pred

def visualize_ecg_analysis(signals, labels, arrhythmia_types, features, classifier):
    """Create comprehensive ECG analysis visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Sample ECG signals for each arrhythmia type
    for i, arrhythmia_type in enumerate(arrhythmia_types):
        ax = axes[0, i//2] if i < 2 else axes[1, i-2]
        
        # Find first signal of this type
        signal_idx = np.where(labels == i)[0][0]
        signal = signals[signal_idx]
        
        t = np.linspace(0, 2, len(signal))
        ax.plot(t, signal, 'b-', linewidth=1)
        ax.set_title(f'{arrhythmia_type.replace("_", " ").title()} ECG')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)
    
    # 2. Feature importance from classifier
    feature_importance = classifier.feature_importances_
    ax = axes[2, 0]
    
    # Split FFT features vs medical features
    fft_importance = feature_importance[:-2]  # All but last 2
    medical_importance = feature_importance[-2:]  # Last 2
    
    x_fft = np.arange(len(fft_importance))
    ax.bar(x_fft, fft_importance, alpha=0.7, label='FFT Features')
    ax.bar([len(fft_importance), len(fft_importance)+1], medical_importance, 
           alpha=0.7, label='Medical Features', color='orange')
    
    ax.set_title('Feature Importance for ECG Classification')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Importance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Frequency analysis comparison
    ax = axes[2, 1]
    
    for i, arrhythmia_type in enumerate(arrhythmia_types):
        signal_idx = np.where(labels == i)[0][0]
        signal = signals[signal_idx]
        
        # Compute power spectrum
        freqs = np.fft.fftfreq(len(signal), 1/360)[:len(signal)//2]
        spectrum = np.abs(np.fft.fft(signal))[:len(signal)//2]
        
        ax.plot(freqs[:50], spectrum[:50], label=arrhythmia_type.replace('_', ' ').title(), alpha=0.7)
    
    ax.set_title('Frequency Spectrum Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/plots/ecg_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(y_test, y_pred, arrhythmia_types):
    """Create confusion matrix for ECG classification"""
    
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[t.replace('_', ' ').title() for t in arrhythmia_types],
                yticklabels=[t.replace('_', ' ').title() for t in arrhythmia_types])
    
    plt.title('ECG Arrhythmia Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('visualizations/plots/ecg_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main proof of concept execution"""
    
    print("🏥 ECG Arrhythmia Detection - Proof of Concept")
    print("=" * 50)
    
    # 1. Generate synthetic ECG dataset
    signals, labels, arrhythmia_types = create_ecg_dataset(n_samples_per_class=100)
    
    print(f"Dataset created: {len(signals)} ECG signals")
    print(f"Signal shape: {signals[0].shape}")
    print(f"Classes: {arrhythmia_types}")
    
    # 2. Extract features using our FFT method
    features = extract_ecg_features(signals, num_modes=25)
    
    print(f"Features extracted: {features.shape}")
    print(f"Feature dimension: {features.shape[1]} (50 FFT + 2 medical)")
    
    # 3. Train classifier
    classifier, X_test, y_test, y_pred = train_ecg_classifier(features, labels)
    
    # 4. Evaluate performance
    print("\n📊 Classification Results:")
    print("=" * 30)
    print(classification_report(y_test, y_pred, 
                              target_names=[t.replace('_', ' ').title() for t in arrhythmia_types]))
    
    # 5. Create visualizations
    print("\n🎨 Creating visualizations...")
    visualize_ecg_analysis(signals, labels, arrhythmia_types, features, classifier)
    create_confusion_matrix(y_test, y_pred, arrhythmia_types)
    
    # 6. Demonstrate real-time prediction
    print("\n🔍 Real-time prediction example:")
    print("=" * 35)
    
    test_signal = generate_synthetic_ecg(arrhythmia_type='pvc')
    test_features = extract_ecg_features([test_signal], num_modes=25)
    prediction = classifier.predict(test_features)[0]
    confidence = classifier.predict_proba(test_features)[0]
    
    print(f"Generated: PVC ECG signal")
    print(f"Predicted: {arrhythmia_types[prediction].replace('_', ' ').title()}")
    print(f"Confidence: {confidence[prediction]:.2f}")
    print(f"All probabilities: {dict(zip(arrhythmia_types, confidence))}")
    
    print("\n✅ Proof of concept completed successfully!")
    print("📁 Visualizations saved to visualizations/plots/")
    print("\n🚀 Next steps:")
    print("   1. Download real MIT-BIH ECG database")
    print("   2. Replace synthetic data with real ECG recordings")
    print("   3. Implement deep learning architecture")
    print("   4. Add real-time monitoring capabilities")

if __name__ == '__main__':
    main()