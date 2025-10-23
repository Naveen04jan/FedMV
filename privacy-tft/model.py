import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GRU, MultiHeadAttention, LayerNormalization,
    Dropout, Concatenate, TimeDistributed, GlobalAveragePooling1D
)
from pykalman import KalmanFilter

def apply_kalman_filter(data):
    kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
    (filtered_state_means, _) = kf.filter(data)
    return filtered_state_means

def create_dataset(static_data, time_varying_data, target_data, time_step=6):
    X_static, X_time_varying, y = [], [], []
    for i in range(len(time_varying_data) - time_step):
        X_static.append(static_data[i])
        X_time_varying.append(time_varying_data[i:(i + time_step)])
        if target_data is not None:
            y.append(target_data[i + time_step])
    
    X_static = np.array(X_static)
    X_time_varying = np.array(X_time_varying)
    if target_data is not None:
        y = np.array(y)
    else:
        y = None
        
    return X_static, X_time_varying, y

def build_tft_inspired_model(static_input_shape, time_varying_input_shape, num_heads=4, dropout_rate=0.2):
    # Inputs
    static_inputs = Input(shape=static_input_shape)
    time_varying_inputs = Input(shape=time_varying_input_shape)
    
    # Static feature processing
    static_context = Dense(32, activation='relu')(static_inputs)
    static_context = LayerNormalization(epsilon=1e-6)(static_context)
    
    # Time-varying feature processing
    time_varying_processed = TimeDistributed(Dense(32, activation='relu'))(time_varying_inputs)
    time_varying_processed = LayerNormalization(epsilon=1e-6)(time_varying_processed)
    
    # GRU layer
    gru_out = GRU(64, return_sequences=True)(time_varying_processed)
    
    # Multi-head attention layer
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=64)(gru_out, gru_out)
    
    # Add & Normalize
    attention = LayerNormalization(epsilon=1e-6)(attention + gru_out)
    
    # Dropout
    attention = Dropout(dropout_rate)(attention)
    
    # Global average pooling
    pooled = GlobalAveragePooling1D()(attention)
    
    # Concatenate with static context
    combined = Concatenate()([pooled, static_context])
    
    # Final dense layers
    output = Dense(32, activation='relu')(combined)
    output = Dense(1)(output)
    
    model = Model(inputs=[static_inputs, time_varying_inputs], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model