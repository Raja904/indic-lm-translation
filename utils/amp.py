import torch
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    """
    A simple wrapper for torch.cuda.amp to handle mixed precision training.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)

    def step(self, loss, optimizer, model_parameters=None, clip_grad=None):
        """
        Perform a single optimization step with scaling.
        """
        # Backward pass
        self.scaler.scale(loss).backward()

        # Unscale for gradient clipping
        if clip_grad is not None and model_parameters is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_parameters, clip_grad)

        # Optimizer step
        self.scaler.step(optimizer)
        
        # Update scaler
        self.scaler.update()
        
        # Zero gradients
        optimizer.zero_grad()

    def get_autocast_context(self):
        """
        Returns the autocast context manager.
        """
        return autocast(enabled=self.enabled)
