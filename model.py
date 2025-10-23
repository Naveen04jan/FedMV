import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GRU, MultiHeadAttention, LayerNormalization,
    Dropout, Concatenate, TimeDistributed, GlobalAveragePooling1D
)
from pykalman import KalmanFilter

# --- MC DROPOUT CHANGE: Custom Dropout Layer ---
# This layer will allow us to force dropout to be active during inference.
class MCDropout(Dropout):
    def call(self, inputs, training=None):
        # The 'training' flag is the key. If we pass training=True,
        # this layer will perform dropout even during model.predict().
        return super().call(inputs, training=True)
# --- END MC DROPOUT CHANGE ---

def apply_kalman_filter(data):
    # Add a check for empty or very short data
    if data is None or len(data) == 0:
        return np.array([])
        
    try:
        kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
        (filtered_state_means, _) = kf.filter(data)
        return filtered_state_means.flatten() # Ensure 1D output
    except Exception as e:
        # Catch errors from Kalman (e.g., singular matrix)
        print(f"Kalman filter failed on data with len {len(data)}: {e}. Returning original data.")
        return data

def create_dataset(static_data, time_varying_data, target_data, time_step=6):
    X_static, X_time_varying, y = [], [], []
    
    # Get the total number of samples we can create
    num_samples = len(time_varying_data) - time_step

    # Check if we have enough data to create even one sample
    if num_samples <= 0:
        if target_data is not None:
            return np.array([]), np.array([]), np.array([])
        else:
            return np.array([]), np.array([]), None

    for i in range(num_samples):
        X_static.append(static_data[i])
        X_time_varying.append(time_varying_data[i:(i + time_step)])
        
        if target_data is not None:
            y.append(target_data[i + time_step][0])
    
    try:
        X_static = np.array(X_static)
        X_time_varying = np.array(X_time_varying)
        
        if target_data is not None:
            y = np.array(y)
        else:
            y = None

        return X_static, X_time_varying, y
        
    except ValueError as e:
        print(f"ERROR in create_dataset np.array conversion: {e}. Returning empty arrays.")
        return np.array([]), np.array([]), np.array([])


def build_tft_inspired_model(static_input_shape, time_varying_input_shape, num_heads=4, dropout_rate=0.2):
    # Inputs
    static_inputs = Input(shape=static_input_shape)
    time_varying_inputs = Input(shape=time_varying_input_shape)
    
    # --- SHARED BASE ---
    static_context = Dense(32, activation='relu')(static_inputs)
    static_context = LayerNormalization(epsilon=1e-6)(static_context)
    
    time_varying_processed = TimeDistributed(Dense(32, activation='relu'))(time_varying_inputs)
    time_varying_processed = LayerNormalization(epsilon=1e-6)(time_varying_processed)
    
    gru_out = GRU(64, return_sequences=True)(time_varying_processed)
    
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=64)(gru_out, gru_out)
    
    attention = LayerNormalization(epsilon=1e-6)(attention + gru_out)
    
    # --- MC DROPOUT CHANGE: Use the custom layer ---
    # This will now apply dropout during both training and evaluation
    attention = MCDropout(dropout_rate)(attention)
    # --- END MC DROPOUT CHANGE ---
    
    pooled = GlobalAveragePooling1D()(attention)
    
    combined = Concatenate()([pooled, static_context])
    
    # --- PERSONALIZED HEAD ---
    output = Dense(32, activation='relu', name='personalized_dense')(combined)
    output = Dense(1, name='personalized_output')(output)
    
    model = Model(inputs=[static_inputs, time_varying_inputs], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Helper functions for personalization (no changes needed)
def get_global_weights(model):
    global_weights = []
    for layer in model.layers:
        if layer.name not in ['personalized_dense', 'personalized_output']:
            global_weights.extend(layer.get_weights())
    return global_weights

def set_global_weights(model, global_weights):
    weight_idx = 0
    for layer in model.layers:
        if layer.name not in ['personalized_dense', 'personalized_output']:
            num_weight_tensors = len(layer.get_weights())
            if num_weight_tensors > 0:
                layer_weights = global_weights[weight_idx : weight_idx + num_weight_tensors]
                layer.set_weights(layer_weights)
                weight_idx += num_weight_tensors
    
    if weight_idx != len(global_weights):
        raise ValueError(f"Mismatch in weights: expected {len(global_weights)}, but used {weight_idx}")

