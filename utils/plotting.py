import matplotlib.pyplot as plt
import os

def plot_curves(
    data: dict, 
    title: str, 
    ylabel: str, 
    save_path: str,
    xlabel: str = "Epoch"
):
    """
    Plot multiple curves on the same chart.
    data: {'train_loss': [0.5, 0.4, ...], 'val_loss': [0.6, 0.5, ...]}
    """
    plt.figure(figsize=(10, 6))
    for label, values in data.items():
        plt.plot(values, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def save_training_plots(history: dict, assets_dir: str, experiment_name: str):
    """
    Helper to save common training plots.
    """
    # Loss plot
    loss_data = {k: v for k, v in history.items() if 'loss' in k.lower()}
    if loss_data:
        plot_curves(
            loss_data, 
            f"Loss Curves - {experiment_name}", 
            "Loss", 
            os.path.join(assets_dir, f"{experiment_name}_loss.png")
        )
    
    # BLEU plot
    bleu_data = {k: v for k, v in history.items() if 'bleu' in k.lower()}
    if bleu_data:
        plot_curves(
            bleu_data, 
            f"BLEU Scores - {experiment_name}", 
            "BLEU (0-100)", 
            os.path.join(assets_dir, f"{experiment_name}_bleu.png")
        )
