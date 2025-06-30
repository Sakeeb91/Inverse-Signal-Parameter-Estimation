"""
Streamlit web application for Inverse Signal Parameter Estimation.

This app provides an interactive interface for training and testing the 
Fourier feature-based neural network model for signal parameter prediction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os

from data_generator import generate_signal, generate_dataset, SIGNAL_LENGTH, T_MAX
from feature_extractor import extract_features_batch, extract_fourier_features
from model import SignalModel

# Configure Streamlit page
st.set_page_config(
    page_title="Fourier Feature Signal Predictor",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main title and description
st.title("💡 Inverse Signal Parameter Estimation with Fourier Features")
st.markdown("""
This application demonstrates how **Fourier Transform** creates powerful, low-dimensional features 
for neural networks. We predict the **amplitude** and **frequency** that generated a noisy sine wave 
using FFT-based feature extraction inspired by **Fourier Neural Operators**.
""")

st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📊 Dataset Parameters")
    num_samples = st.slider("Number of Training Samples", 500, 5000, 2000, 250)
    amp_range = st.slider("Amplitude Range", 0.1, 10.0, (0.5, 5.0), 0.1)  
    freq_range = st.slider("Frequency Range (Hz)", 0.5, 20.0, (1.0, 10.0), 0.5)
    noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.01)
    
    st.subheader("🔧 Feature Engineering")
    num_modes = st.slider("Number of Fourier Modes (k)", 5, 50, 20, 1)
    st.caption(f"Feature dimension: **{2 * num_modes}** (from {SIGNAL_LENGTH} raw points)")
    
    compression_ratio = SIGNAL_LENGTH / (2 * num_modes)
    st.metric("Compression Ratio", f"{compression_ratio:.1f}x")
    
    st.subheader("🧠 Model Architecture")
    hidden_layers = st.selectbox("Hidden Layer Architecture", 
                                [(64, 64, 64), (128, 64, 32), (32, 32), (128, 128)],
                                format_func=lambda x: f"{len(x)} layers: {x}")
    max_iterations = st.slider("Max Training Iterations", 100, 1000, 300, 50)
    learning_rate_init = st.select_slider("Learning Rate", 
                                         [0.0001, 0.0005, 0.001, 0.005, 0.01], 
                                         0.001)

# Initialize session state
if 'dataset_generated' not in st.session_state:
    st.session_state.dataset_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Workflow")
    
    # Step 1: Dataset Generation
    st.subheader("Step 1: Generate Synthetic Dataset")
    
    if st.button("🔄 Generate New Dataset", type="primary"):
        with st.spinner("Generating synthetic signals..."):
            # Generate dataset
            signals, labels = generate_dataset(
                num_samples=num_samples,
                amp_range=amp_range,
                freq_range=freq_range,
                noise_level=noise_level,
                save_path='signals.npz'
            )
            
            st.session_state.signals = signals
            st.session_state.labels = labels
            st.session_state.dataset_generated = True
            st.session_state.model_trained = False  # Reset model training
            
        st.success(f"✅ Generated {num_samples} synthetic signals!")
        
        # Display dataset statistics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Dataset Size", f"{signals.shape[0]:,} samples")
            st.metric("Signal Length", f"{signals.shape[1]:,} points")
        with col_b:
            st.metric("Amplitude Range", f"{labels[:, 0].min():.2f} - {labels[:, 0].max():.2f}")
            st.metric("Frequency Range", f"{labels[:, 1].min():.2f} - {labels[:, 1].max():.2f} Hz")

    # Load existing dataset if available
    elif os.path.exists('signals.npz') and not st.session_state.dataset_generated:
        try:
            data = np.load('signals.npz')
            st.session_state.signals = data['signals']
            st.session_state.labels = data['labels']
            st.session_state.dataset_generated = True
            st.info("📂 Loaded existing dataset from signals.npz")
        except:
            st.warning("⚠️ Could not load existing dataset. Please generate a new one.")

    if st.session_state.dataset_generated:
        # Step 2: Feature Extraction and Model Training
        st.subheader("Step 2: Extract Features & Train Model")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Extracting Fourier features and training model..."):
                # Extract features
                features = extract_features_batch(st.session_state.signals, num_modes)
                
                # Create and train model
                model = SignalModel(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=max_iterations,
                    random_state=42
                )
                
                # Record training time
                start_time = time.time()
                training_history = model.train(features, st.session_state.labels)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.training_history = training_history
                st.session_state.model_trained = True
                
            st.success("✅ Model training completed!")
            
            # Display training results
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Training Time", f"{training_history['training_time']:.2f}s")
            with col_b:
                st.metric("Iterations", training_history['n_iterations'])
            with col_c:
                st.metric("Test R²", f"{training_history['test_metrics']['r2']:.4f}")

with col2:
    st.header("📈 Performance Insights")
    
    if st.session_state.model_trained:
        metrics = st.session_state.training_history['test_metrics']
        
        # Performance metrics
        st.metric("🎯 Overall R² Score", f"{metrics['r2']:.4f}")
        st.metric("📐 Amplitude R²", f"{metrics['r2_amplitude']:.4f}")
        st.metric("🔊 Frequency R²", f"{metrics['r2_frequency']:.4f}")
        st.metric("📉 Mean Absolute Error", f"{metrics['mae']:.4f}")
        
        # Feature efficiency
        st.markdown("### 🔬 Feature Efficiency")
        original_dim = SIGNAL_LENGTH
        feature_dim = 2 * num_modes
        st.metric("Original Dimension", f"{original_dim:,}")
        st.metric("Feature Dimension", f"{feature_dim}")
        st.metric("Compression", f"{original_dim/feature_dim:.1f}x smaller")

# Visualization Section
if st.session_state.model_trained:
    st.header("📊 Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Training Loss", "📈 Predictions", "🎵 Signal Examples", "🔍 Interactive Test"])
    
    with tab1:
        # Training loss curve
        if st.session_state.training_history['loss_curve'] is not None:
            loss_curve = st.session_state.training_history['loss_curve']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=loss_curve,
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Model Training Loss Curve",
                xaxis_title="Iteration",
                yaxis_title="Loss",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loss curve not available for this training session.")
    
    with tab2:
        # Prediction scatter plots
        model = st.session_state.model
        features = st.session_state.features
        labels = st.session_state.labels
        
        # Make predictions on test set
        test_indices = np.arange(int(0.8 * len(features)), len(features))
        X_test = features[test_indices]
        y_test = labels[test_indices]
        y_pred = model.predict(X_test)
        
        # Create subplot for amplitude and frequency
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Amplitude Prediction', 'Frequency Prediction')
        )
        
        # Amplitude scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_test[:, 0], y=y_pred[:, 0],
                mode='markers',
                name='Amplitude',
                marker=dict(color='blue', opacity=0.6),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Perfect prediction line for amplitude
        amp_range_plot = [y_test[:, 0].min(), y_test[:, 0].max()]
        fig.add_trace(
            go.Scatter(
                x=amp_range_plot, y=amp_range_plot,
                mode='lines',
                name='Perfect',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Frequency scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_test[:, 1], y=y_pred[:, 1],
                mode='markers',
                name='Frequency',
                marker=dict(color='orange', opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Perfect prediction line for frequency
        freq_range_plot = [y_test[:, 1].min(), y_test[:, 1].max()]
        fig.add_trace(
            go.Scatter(
                x=freq_range_plot, y=freq_range_plot,
                mode='lines',
                name='Perfect',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="True Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="True Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Predicted Frequency (Hz)", row=1, col=2)
        
        fig.update_layout(height=400, title_text="Prediction Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Sample signals visualization
        st.subheader("Example Signals from Dataset")
        
        # Select random samples
        n_samples_show = 6
        random_indices = np.random.choice(len(st.session_state.signals), n_samples_show, replace=False)
        
        # Create time vector
        t = np.linspace(0, T_MAX, SIGNAL_LENGTH)
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Signal {i}: A={st.session_state.labels[i, 0]:.2f}, f={st.session_state.labels[i, 1]:.2f}Hz' 
                          for i in random_indices]
        )
        
        for idx, signal_idx in enumerate(random_indices):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            signal = st.session_state.signals[signal_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=t, y=signal,
                    mode='lines',
                    name=f'Signal {signal_idx}',
                    showlegend=False,
                    line=dict(width=1)
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=500, title_text="Sample Signals from Training Dataset")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Interactive testing
        st.subheader("🧪 Test the Model Interactively")
        
        col_test_1, col_test_2 = st.columns([2, 1])
        
        with col_test_1:
            # Generate custom signal for testing
            st.markdown("**Generate a test signal:**")
            test_amp = st.slider("Test Amplitude", 0.5, 5.0, 2.5, 0.1)
            test_freq = st.slider("Test Frequency (Hz)", 1.0, 10.0, 5.0, 0.1)
            test_noise = st.slider("Test Noise Level", 0.0, 0.3, 0.1, 0.01)
            
            if st.button("🎲 Generate Random Test Signal"):
                test_amp = np.random.uniform(0.5, 5.0)
                test_freq = np.random.uniform(1.0, 10.0)
                st.rerun()
            
            # Generate test signal
            test_signal = generate_signal(test_amp, test_freq, test_noise)
            
            # Visualize test signal
            t_test = np.linspace(0, T_MAX, SIGNAL_LENGTH)
            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(
                x=t_test, y=test_signal,
                mode='lines',
                name='Test Signal',
                line=dict(color='green', width=2)
            ))
            fig_test.update_layout(
                title="Generated Test Signal",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig_test, use_container_width=True)
        
        with col_test_2:
            # Make prediction
            test_features = extract_fourier_features(test_signal, num_modes).reshape(1, -1)
            test_prediction = st.session_state.model.predict(test_features)[0]
            
            st.markdown("**True Parameters:**")
            st.metric("True Amplitude", f"{test_amp:.3f}")
            st.metric("True Frequency", f"{test_freq:.3f} Hz")
            
            st.markdown("**Predicted Parameters:**")
            st.metric("Predicted Amplitude", f"{test_prediction[0]:.3f}", 
                     delta=f"{test_prediction[0] - test_amp:.3f}")
            st.metric("Predicted Frequency", f"{test_prediction[1]:.3f} Hz", 
                     delta=f"{test_prediction[1] - test_freq:.3f}")
            
            # Calculate errors
            amp_error = abs(test_prediction[0] - test_amp) / test_amp * 100
            freq_error = abs(test_prediction[1] - test_freq) / test_freq * 100
            
            st.markdown("**Prediction Errors:**")
            st.metric("Amplitude Error", f"{amp_error:.2f}%")
            st.metric("Frequency Error", f"{freq_error:.2f}%")

# Information section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### 🔬 How It Works
    
    1. **Signal Generation**: Creates synthetic sine waves with random amplitudes and frequencies, plus Gaussian noise
    2. **Feature Extraction**: Uses Fast Fourier Transform (FFT) to convert 1024-point signals into compact Fourier features
    3. **Model Training**: Trains a Multi-Layer Perceptron using scikit-learn to predict parameters from features
    4. **Interactive Testing**: Allows real-time testing with custom or random signals
    
    ### 🎯 Key Innovations
    
    - **Dimensionality Reduction**: 1024 → 40 features (20 modes) = 25x compression
    - **Information Preservation**: Low-frequency Fourier modes capture essential signal characteristics
    - **CPU Optimization**: Uses scikit-learn MLPRegressor for efficient CPU-only training
    - **Real-time Inference**: Fast prediction suitable for interactive applications
    
    ### 📚 Scientific Background
    
    This approach is inspired by **Fourier Neural Operators (FNO)**, which operate in the frequency domain 
    to achieve resolution-invariance and computational efficiency. The key insight is that most signal 
    information is concentrated in low-frequency components, allowing aggressive dimensionality reduction 
    without loss of predictive power.
    
    ### 🌍 Real-World Applications
    
    - **Structural Health Monitoring**: Analyze building vibrations to detect damage
    - **Power Grid Analysis**: Monitor electrical grid frequency stability
    - **Medical Signal Processing**: Classify ECG/EEG patterns for diagnosis
    - **Audio Processing**: Instrument recognition and acoustic analysis
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>🤖 <strong>Inverse Signal Parameter Estimation with Fourier Features</strong></p>
<p>Built with Streamlit • Powered by scikit-learn • Inspired by Fourier Neural Operators</p>
</div>
""", unsafe_allow_html=True)