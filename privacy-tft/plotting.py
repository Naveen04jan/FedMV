import json
import matplotlib.pyplot as plt

# Load metrics from JSON file
with open("metrics_history.json", "r") as f:
   metrics = json.load(f)

train_metrics = metrics['train_metrics']
eval_metrics = metrics['eval_metrics']

rounds = [m['round'] for m in train_metrics]

# Extract metrics 
train_mse = [m['train_mse'] for m in train_metrics]
train_mae = [m['train_mae'] for m in train_metrics]

eval_mse = [m['mse'] for m in eval_metrics]
eval_mae = [m['mae'] for m in eval_metrics]

# Create figure with more height and adjust margins
plt.figure(figsize=(10, 14))

# Adjust the overall layout to make room for the title
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4)

# Add main title with adjusted position
plt.suptitle('Model Performance Metrics (PWS = 30 minutes)', 
            fontsize=16, 
            y=0.97)

# MSE Plot (top)
plt.subplot(2, 1, 1)
plt.plot(rounds, train_mse, label='Train MSE', marker='o')
plt.plot(rounds, eval_mse, label='Test MSE', marker='o')
plt.xlabel('Rounds')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


# MAE Plot (bottom)
plt.subplot(2, 1, 2)
plt.plot(rounds, train_mae, label='Train MAE', marker='o')
plt.plot(rounds, eval_mae, label='Test MAE', marker='o')
plt.xlabel('Rounds')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust for title overlap
plt.show()
