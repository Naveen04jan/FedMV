import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import flwr as fl
import tensorflow as tf
from model import build_temporal_fusion_model, create_dataset, apply_kalman_filter
from config import EPOCHS, WINDOW_SIZE, DATA_DIR, NUM_CLIENTS, SERVER_ADDRESS
import json



# -----------------------------
# Differential Privacy Helpers
# -----------------------------
def apply_client_side_clipping(model_weights, clipping_norm):
    """Clips the model weights using a fixed L2 norm threshold."""
    clipped_weights = []
    for weight in model_weights:
        flat = weight.flatten()
        norm = np.linalg.norm(flat)
        if norm > clipping_norm:
            weight = weight * (clipping_norm / norm)
        clipped_weights.append(weight)
    return clipped_weights


def add_noise_to_weights(clipped_weights, noise_multiplier, clipping_norm):
    """Adds Gaussian noise to clipped weights for DP."""
    noisy_weights = []
    for weight in clipped_weights:
        flat = weight.flatten()
        stddev = noise_multiplier * clipping_norm
        noise = np.random.normal(0, stddev, flat.shape)
        noisy = flat + noise
        noisy_weights.append(noisy.reshape(weight.shape))
    return noisy_weights

# -----------------------------
# Flower Client with DP
# -----------------------------
class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, client_id, clipping_norm=1.0, noise_multiplier=0.1):
        self.client_id = client_id
        self.csv_files = self.load_csv_files(client_id)
        self.epochs = EPOCHS
        self.clipping_norm = clipping_norm
        self.noise_multiplier = noise_multiplier
        
        # Initialize model with fixed input shape
        # Shape: (WINDOW_SIZE, 2) for [disc, start_time] features after Kalman filtering
        self.model = build_temporal_fusion_model(input_shape=(WINDOW_SIZE, 2))

    def load_csv_files(self, client_id):
        with open('file_allocation.json', 'r') as f:
            allocation = json.load(f)
        return allocation[f"client_{client_id}"]

    def preprocess_data(self, file):
        """
        Preprocess Alibaba dataset with columns: start_time, cpu, disc
        Target: cpu
        Features: disc, start_time (normalized) with Kalman filtering
        """
        df = pd.read_csv(file)
        
        # Keep only required columns
        df = df[["start_time", "cpu", "disk"]].dropna()
        
        # Remove infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        if len(df) < WINDOW_SIZE + 1:
            raise ValueError(f"File {file} has insufficient data")
        
        # Normalize start_time (convert to relative time from first timestamp)
        df['start_time'] = df['start_time'] - df['start_time'].min()
        
        # Apply Kalman filter to smooth the time series
        df['disk'] = apply_kalman_filter(df['disk'].values)
        df['start_time'] = apply_kalman_filter(df['start_time'].values)
        df['cpu'] = apply_kalman_filter(df['cpu'].values)
        
        # Extract features and target
        # Features: [disc, start_time]
        features = df[['disk', 'start_time']].values
        target = df['cpu'].values.reshape(-1, 1)
        
        # Scale the data
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target)
        
        return features_scaled, target_scaled, feature_scaler, target_scaler

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        total_train_mse = 0
        total_train_mae = 0
        total_samples = 0

        for file in self.csv_files:
            try:
                features, target, _, target_scaler = self.preprocess_data(file)
                
                if len(features) <= WINDOW_SIZE:
                    continue
                    
                # 75-25 train-test split
                training_size = int(len(features) * 0.75)
                
                features_train = features[:training_size]
                target_train = target[:training_size]
                
                # Create windowed dataset
                X, y = create_dataset(features_train, target_train, WINDOW_SIZE)
                
                if len(X) == 0:
                    continue
                
                # Train for 1 epoch (per federated round)
                history = self.model.fit(
                    X, y,
                    epochs=1,
                    batch_size=32,
                    verbose=0
                )

                # Calculate metrics
                train_loss = self.model.evaluate(X, y, verbose=0)
                mse, mae = train_loss[0], train_loss[1]
                
                total_train_mse += mse
                total_train_mae += mae
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing file {file} during fitting: {str(e)}")
                continue

        if total_samples == 0:
            return self.model.get_weights(), 0, {}

        avg_train_mse = total_train_mse / total_samples
        avg_train_mae = total_train_mae / total_samples

        # -----------------------------
        # DP step: Clip + Add Noise
        # -----------------------------
        clipped_weights = apply_client_side_clipping(self.model.get_weights(), self.clipping_norm)
        noisy_weights = add_noise_to_weights(clipped_weights, self.noise_multiplier, self.clipping_norm)

        return noisy_weights, total_samples, {
            'train_mse': float(avg_train_mse),
            'train_mae': float(avg_train_mae)
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        total_mse = 0
        total_mae = 0
        total_samples = 0

        for file in self.csv_files:
            try:
                features, target, _, target_scaler = self.preprocess_data(file)
                
                if len(features) <= WINDOW_SIZE:
                    continue
                    
                # 75-25 train-test split
                training_size = int(len(features) * 0.75)
                
                features_test = features[training_size:]
                target_test = target[training_size:]
                
                # Create windowed dataset
                X, y = create_dataset(features_test, target_test, WINDOW_SIZE)
                
                if len(X) == 0:
                    continue
                
                # Evaluate model
                test_loss = self.model.evaluate(X, y, verbose=0)
                mse, mae = test_loss[0], test_loss[1]
                
                total_mse += mse
                total_mae += mae
                total_samples += 1
                
            except Exception as e:
                print(f"Error processing file {file} during evaluation: {str(e)}")
                continue

        if total_samples == 0:
            return float('inf'), 0, {}

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples

        return float(avg_mse), total_samples, {
            'test_mse': float(avg_mse),
            'test_mae': float(avg_mae)
        }

def main():
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    clipping_norm = 1.0
    noise_multiplier = 0.1
    client = TimeSeriesClient(client_id, clipping_norm, noise_multiplier)
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

if __name__ == "__main__":
    main()