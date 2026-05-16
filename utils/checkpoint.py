import torch
import os
import logging

def save_checkpoint(
    state: dict, 
    checkpoint_dir: str, 
    experiment_name: str, 
    epoch: int, 
    is_best: bool = False,
    metric_name: str = "val_loss",
    metric_value: float = 0.0
):
    """
    Save model checkpoint with metadata.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filename = f"{experiment_name}_epoch{epoch}.pt"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # Add metadata
    state['epoch'] = epoch
    state[metric_name] = metric_value
    
    torch.save(state, filepath)
    logging.info(f"Saved checkpoint to {filepath}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pt")
        torch.save(state, best_path)
        logging.info(f"Saved NEW BEST checkpoint to {best_path}")

def load_checkpoint(filepath: str, model, optimizer=None, device=None):
    """
    Load model and optimizer state from a checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    logging.info(f"Loaded checkpoint from {filepath} (Epoch {epoch})")
    
    return model, optimizer, epoch
