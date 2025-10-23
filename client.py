import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import flwr as fl
import tensorflow as tf
from model import (
    build_tft_inspired_model, create_dataset, apply_kalman_filter,
    get_global_weights, set_global_weights
)
from config import EPOCHS, WINDOW_SIZE, DATA_DIR, NUM_CLIENTS, SERVER_ADDRESS
import json

class TimeSeriesClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.csv_files = self.load_csv_files(client_id)
        self.epochs = EPOCHS
        
       
        self.mc_samples = 50
       

        static_shape, time_varying_shape = self._get_input_shapes()
        if static_shape is None or time_varying_shape is None:
            raise ValueError("No valid data found to initialize model")

        self.model = build_tft_inspired_model(static_shape, time_varying_shape)

    # _get_input_shapes, load_csv_files, preprocess_data, and get_parameters are unchanged
    def _get_input_shapes(self):
        for file in self.csv_files:
            try:
                static_data, time_varying_data, _, _, _, _ = self.preprocess_data(file)
                if len(static_data) > WINDOW_SIZE:
                    X_static, X_time_varying, _ = create_dataset(static_data, time_varying_data, None, WINDOW_SIZE)
                    if len(X_static) > 0:
                        return X_static.shape[1:], X_time_varying.shape[1:]
            except Exception as e:
                print(f"Error processing file {file} during shape check: {str(e)}")
                continue
        return None, None

    def load_csv_files(self, client_id):
        with open('file_allocation.json', 'r') as f:
            allocation = json.load(f)
        return allocation[f"client_{client_id}"]

    def preprocess_data(self, file):
        df = pd.read_csv(file)
        df = df[["timestamp", "min_cpu", "avg_cpu", "max_cpu", "vm virtual core count",
                 "vm memory (gb)", "vm_category"]].dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
        #df['min_cpu'] = apply_kalman_filter(df['min_cpu'].values)
        #df['avg_cpu'] = apply_kalman_filter(df['avg_cpu'].values)
        #df['max_cpu'] = apply_kalman_filter(df['max_cpu'].values)

        df = pd.get_dummies(df, columns=['vm_category'])
        expected_categories = [
            'vm_category_t2.micro', 'vm_category_t2.small',
            'vm_category_t2.medium', 'vm_category_t2.large',
            'vm_category_t2.xlarge', 'vm_category_t2.2xlarge'
        ]
        for cat in expected_categories:
            if cat not in df.columns:
                df[cat] = 0

        static_features = ['vm virtual core count', 'vm memory (gb)'] + expected_categories
        time_varying_features = ['timestamp', 'min_cpu', 'max_cpu']
        target = ['avg_cpu']

        static_data = df[static_features].values
        time_varying_data = df[time_varying_features].values
        target_data = df[target].values

        static_scaler = MinMaxScaler(feature_range=(0, 1))
        time_varying_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        static_data_scaled = static_scaler.fit_transform(static_data)
        time_varying_data_scaled = time_varying_scaler.fit_transform(time_varying_data)
        target_data_scaled = target_scaler.fit_transform(target_data)

        return static_data_scaled, time_varying_data_scaled, target_data_scaled, static_scaler, time_varying_scaler, target_scaler

    def get_parameters(self, config):
        print(f"\n[Client {self.client_id}] Sending global weights...")
        return get_global_weights(self.model)

    # fit method is unchanged
    def fit(self, parameters, config):
        print(f"\n[Client {self.client_id}] Receiving global weights...")
        set_global_weights(self.model, parameters)
        
        total_train_mse = 0
        total_train_mae = 0
        total_samples = 0

        for file in self.csv_files:
            try:
                static_data, time_varying_data, target_data, _, _, _ = self.preprocess_data(file)
                if len(static_data) <= WINDOW_SIZE:
                    continue

                training_size = int(len(static_data) * 0.75)
                static_train = static_data[:training_size]
                time_varying_train = time_varying_data[:training_size]
                target_train = target_data[:training_size]

                X_static, X_time_varying, y = create_dataset(static_train, time_varying_train, target_train, WINDOW_SIZE)
                
                if len(y) == 0:
                    continue

                self.model.fit([X_static, X_time_varying], y,
                               epochs=1, batch_size=64, verbose=0)

                mse, mae = self.model.evaluate([X_static, X_time_varying], y, verbose=0)
                
                total_train_mse += mse * len(y)
                total_train_mae += mae * len(y)
                total_samples += len(y)

            except Exception as e:
                print(f"Error processing file {file} during fitting: {str(e)}")
                continue

        if total_samples == 0:
            print(f"\n[Client {self.client_id}] No data processed, returning empty weights.")
            return get_global_weights(self.model), 0, {}

        avg_train_mse = total_train_mse / total_samples
        avg_train_mae = total_train_mae / total_samples

        return get_global_weights(self.model), total_samples, {
            'train_mse': float(avg_train_mse),
            'train_mae': float(avg_train_mae)
        }

    def evaluate(self, parameters, config):
        print(f"\n[Client {self.client_id}] Evaluating with new global weights...")
        set_global_weights(self.model, parameters)
        
        total_mse = 0
        total_mae = 0
        total_samples = 0
        
       
        total_prediction_variance = 0.0
        

        for file in self.csv_files:
            try:
                static_data, time_varying_data, target_data, _, _, _ = self.preprocess_data(file)
                if len(static_data) <= WINDOW_SIZE:
                    continue

                training_size = int(len(static_data) * 0.75)
                static_test = static_data[training_size:]
                time_varying_test = time_varying_data[training_size:]
                target_test = target_data[training_size:]

                X_static, X_time_varying, y = create_dataset(static_test, time_varying_test, target_test, WINDOW_SIZE)
                
                if len(y) == 0:
                    continue

               
                mc_predictions = []
                for _ in range(self.mc_samples):
                    y_pred = self.model.predict([X_static, X_time_varying], verbose=0)
                    mc_predictions.append(y_pred)

                
                mc_predictions = np.stack(mc_predictions, axis=0)  # Shape: (mc_samples, num_data_points, 1)

                
                mean_predictions = np.mean(mc_predictions, axis=0) # Shape: (num_data_points, 1)
                prediction_variance = np.var(mc_predictions, axis=0) # Shape: (num_data_points, 1)
                
                
                avg_file_variance = np.mean(prediction_variance)
                
               
                mse = np.mean(np.square(y - mean_predictions.flatten()))
                mae = np.mean(np.abs(y - mean_predictions.flatten()))
                # --- END MC DROPOUT CHANGE ---
                
                total_mse += mse * len(y)
                total_mae += mae * len(y)
                total_prediction_variance += avg_file_variance * len(y)
                total_samples += len(y)

            except Exception as e:
                print(f"Error processing file {file} during evaluation: {str(e)}")
                continue

        if total_samples == 0:
            return float('inf'), 0, {}

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        
        
        avg_prediction_std = np.sqrt(total_prediction_variance / total_samples)
      

        return float(avg_mse), total_samples, {
            'test_mse': float(avg_mse),
            'test_mae': float(avg_mae),
           
            'avg_prediction_std': float(avg_prediction_std)
        }

def main():
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    
    client = TimeSeriesClient(client_id)
    
    # Start the client using the new start_client API
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client()
    )

if __name__ == "__main__":
    main()

