import logging
import os
import wandb
from typing import Dict, Any, Optional

def setup_logger(log_dir: str = "logs", log_name: str = "training.log"):
    """
    Setup a logger that outputs to both console and a file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_name)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)
    
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

def init_wandb(project_name: str, experiment_name: str, config: Dict[str, Any], notes: Optional[str] = None):
    """
    Initialize Weights & Biases for experiment tracking.
    """
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        notes=notes
    )

def log_metrics(metrics: Dict[str, float], step: int):
    """
    Log metrics to WandB and console (optional).
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)
    
    # Standard logging
    metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logging.info(f"Step {step}: {metric_str}")
