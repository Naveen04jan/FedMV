import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, MultiHeadAttention, LayerNormalization,
    Dropout, GlobalAveragePooling1D, Add
)
from pykalman import KalmanFilter

def apply_kalman_filter(data):
    """
    Apply Kalman filter to smooth noisy time series data
    
    Args:
        data: 1D numpy array of time series values
    
    Returns:
        Smoothed 1D numpy array
    """
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means.flatten()

def create_dataset(features, target, time_step):
    """
    Create windowed time series dataset
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        target: numpy array of shape (n_samples, 1)
        time_step: window size for time series
    
    Returns:
        X: numpy array of shape (n_samples - time_step, time_step, n_features)
        y: numpy array of shape (n_samples - time_step, 1)
    """
    X, y = [], []
    
    for i in range(len(features) - time_step):
        # Take a window of features
        X.append(features[i:(i + time_step)])
        # Predict the target at the next time step
        y.append(target[i + time_step])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def build_temporal_fusion_model(input_shape, num_heads=4, dropout_rate=0.2):
    """
    Build a Temporal Fusion Transformer-inspired model for time series prediction
    Architecture: LSTM + Multi-Head Attention + Skip Connections
    
    Args:
        input_shape: tuple of (time_steps, n_features)
        num_heads: number of attention heads
        dropout_rate: dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape, name='time_varying_inputs')
    
    # LSTM layers for temporal feature extraction
    lstm_out = LSTM(64, return_sequences=True, name='lstm_1')(inputs)
    lstm_out = LayerNormalization(epsilon=1e-6, name='norm_1')(lstm_out)
    lstm_out = Dropout(dropout_rate, name='dropout_1')(lstm_out)
    
    lstm_out = LSTM(64, return_sequences=True, name='lstm_2')(lstm_out)
    lstm_out = LayerNormalization(epsilon=1e-6, name='norm_2')(lstm_out)
    
    # Multi-head self-attention mechanism
    attention_out = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=64,
        name='multi_head_attention'
    )(lstm_out, lstm_out)
    
    # Add & Normalize (residual connection)
    attention_out = Add(name='skip_connection')([attention_out, lstm_out])
    attention_out = LayerNormalization(epsilon=1e-6, name='norm_3')(attention_out)
    
    # Dropout for regularization
    attention_out = Dropout(dropout_rate, name='dropout_2')(attention_out)
    
    # Global average pooling to aggregate temporal information
    pooled = GlobalAveragePooling1D(name='global_avg_pooling')(attention_out)
    
    # Feed-forward network
    dense_out = Dense(32, activation='relu', name='dense_1')(pooled)
    dense_out = Dropout(dropout_rate, name='dropout_3')(dense_out)
    
    dense_out = Dense(16, activation='relu', name='dense_2')(dense_out)
    
    # Output layer
    output = Dense(1, name='output')(dense_out)
    
    # Build and compile model
    model = Model(inputs=inputs, outputs=output, name='temporal_fusion_model')
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model