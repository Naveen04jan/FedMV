import os
import flwr as fl
import numpy as np
import json
from config import SERVER_ADDRESS, NUM_CLIENTS

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        num_rounds = kwargs.pop('num_rounds', 1)
        super().__init__(**kwargs)
        self.metrics_history = []
        self.train_metrics_history = []
        self.best_metrics = {
            'mse': (float('inf'), -1),
            'rmse': (float('inf'), -1),
            'mae': (float('inf'), -1),
            # --- MC DROPOUT CHANGE: Track new metric ---
            'avg_prediction_std': (float('inf'), -1)
        }
        self.num_rounds = num_rounds
        self.best_weights = None
        self.aggregated_weights = None

    def aggregate_fit(self, rnd, results, failures):
        valid_results = []
        for client, fit_res in results:
            if fit_res.num_examples > 0:
                valid_results.append((client, fit_res))
            else:
                print(f"\nWarning: Discarding result from client {client.cid} (0 samples).")
        
        if not valid_results:
             print(f"\nWarning: Round {rnd} received 0 valid fit results. Reusing old weights.")
             return self.aggregated_weights, {}

        aggregated_weights, _ = super().aggregate_fit(rnd, valid_results, failures)

        train_mse_values = [fit_res.metrics['train_mse'] for _, fit_res in valid_results]
        train_mae_values = [fit_res.metrics['train_mae'] for _, fit_res in valid_results]
        
        train_rmse_values = [np.sqrt(mse) for mse in train_mse_values]

        avg_train_mse = np.mean(train_mse_values)
        avg_train_rmse = np.mean(train_rmse_values)
        avg_train_mae = np.mean(train_mae_values)

        self.train_metrics_history.append({
            'round': rnd,
            'train_mse': avg_train_mse,
            'train_rmse': avg_train_rmse,
            'train_mae': avg_train_mae
        })

        print(f"\nAggregated training metrics for round {rnd}:")
        print(f"Train MSE: {avg_train_mse:.4f}")
        print(f"Train RMSE: {avg_train_rmse:.4f}")
        print(f"Train MAE: {avg_train_mae:.4f}")

        self.aggregated_weights = aggregated_weights

        return aggregated_weights, {}

    def aggregate_evaluate(self, rnd, results, failures):
        if failures:
            print(f"\nWarning: Round {rnd} encountered {len(failures)} failures.")

        valid_results = []
        for client, eval_res in results:
            if eval_res.num_examples > 0:
                valid_results.append((client, eval_res))
        
        if not valid_results:
            print(f"\nWarning: Round {rnd} received 0 valid eval results.")
            return float('inf'), {}

        # Get evaluation metrics from clients
        test_mse_values = [r.metrics['test_mse'] for _, r in valid_results]
        test_mae_values = [r.metrics['test_mae'] for _, r in valid_results]
        
        # --- MC DROPOUT CHANGE: Aggregate new metric ---
        pred_std_values = [r.metrics['avg_prediction_std'] for _, r in valid_results]
        avg_pred_std = np.mean(pred_std_values)
        # --- END MC DROPOUT CHANGE ---
        
        test_rmse_values = [np.sqrt(mse) for mse in test_mse_values]

        avg_mse = np.mean(test_mse_values)
        avg_rmse = np.mean(test_rmse_values)
        avg_mae = np.mean(test_mae_values)

        current_metrics = {
            'round': rnd,
            'mse': avg_mse,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'avg_prediction_std': avg_pred_std # Log new metric
        }
        self.metrics_history.append(current_metrics)

        # Update best metrics
        if avg_mse < self.best_metrics['mse'][0]:
            self.best_metrics['mse'] = (avg_mse, rnd)
            self.best_weights = self.aggregated_weights
            self.save_best_weights()
        
        if avg_rmse < self.best_metrics['rmse'][0]:
            self.best_metrics['rmse'] = (avg_rmse, rnd)
            
        if avg_mae < self.best_metrics['mae'][0]:
            self.best_metrics['mae'] = (avg_mae, rnd)
            
        # --- MC DROPOUT CHANGE: Track best (lowest) uncertainty ---
        if avg_pred_std < self.best_metrics['avg_prediction_std'][0]:
            self.best_metrics['avg_prediction_std'] = (avg_pred_std, rnd)
        # --- END MC DROPOUT CHANGE ---

        print(f"\nEvaluation metrics for round {rnd}:")
        print(f"Test MSE: {avg_mse:.4f}")
        print(f"Test RMSE: {avg_rmse:.4f}")
        print(f"Test MAE: {avg_mae:.4f}")
        # --- MC DROPOUT CHANGE: Print new metric ---
        print(f"Avg Prediction Std Dev: {avg_pred_std:.4f}")
        # --- END MC DROPOUT CHANGE ---

        if rnd == self.num_rounds:
            self.save_metrics_history()
            self.print_final_summary()

        return avg_mse, {}

    def save_best_weights(self):
        if self.best_weights is not None:
            weights_ndarrays = fl.common.parameters_to_ndarrays(self.best_weights)
            np.savez("best_weights.npz", *weights_ndarrays)
            print("\nBest weights saved to best_weights.npz")

    def save_metrics_history(self):
        metrics_data = {
            'train_metrics': self.train_metrics_history,
            'eval_metrics': self.metrics_history
        }
        with open("metrics_history.json", "w") as f:
            json.dump(metrics_data, f, indent=4)
        print("\nMetrics history saved to metrics_history.json")

    def print_final_summary(self):
        print("\nFinal Training Summary:")
        print("=" * 50)
        for metric, (value, round_num) in self.best_metrics.items():
            if value != float('inf') and value != float('-inf'):
                print(f"Best {metric.upper()}: {value:.4f} (achieved in round {round_num})")
        print("=" * 50)

def main():
    num_rounds = 10

    strategy = CustomStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
        num_rounds=num_rounds
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )

if __name__ == "__main__":
    main()

