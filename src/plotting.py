import os
import config
import matplotlib.pyplot as plt

font = { 'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15 }
plt.rc('font', **font)


def plot_true_vs_predicted(true, predicted):

    print('[INFO] Plotting true values vs predicted values...')
    plt.figure(figsize=(12, 10), dpi=100)
    plt.scatter(true, predicted, c="b", alpha=0.25)
    plt.title("True values vs Predicted values")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.savefig(os.path.join(config.RUN_LOG_DIR, '/true_vs_predicted.png'))