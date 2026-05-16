import torch
from tqdm import tqdm
import logging
from .logging import log_metrics
from .checkpoint import save_checkpoint
from .amp import MixedPrecisionTrainer

class BaseTrainer:
    """
    Base Trainer class with common training and validation loops.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        config,
        experiment_name: str,
        mixed_precision: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.experiment_name = experiment_name
        self.amp_trainer = MixedPrecisionTrainer(enabled=mixed_precision)
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_bleu': []}

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # Prepare data (override this if batch structure is different)
            src, trg = batch[0].to(self.device), batch[1].to(self.device)
            
            with self.amp_trainer.get_autocast_context():
                output = self.model(src, trg)
                # Reshape output/target if needed for CrossEntropyLoss
                # e.g., output: [batch, seq, vocab], trg: [batch, seq]
                loss = self.criterion(output.view(-1, output.size(-1)), trg.view(-1))

            self.amp_trainer.step(loss, self.optimizer, self.model.parameters(), clip_grad=1.0)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                src, trg = batch[0].to(self.device), batch[1].to(self.device)
                
                with self.amp_trainer.get_autocast_context():
                    output = self.model(src, trg)
                    loss = self.criterion(output.view(-1, output.size(-1)), trg.view(-1))
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def fit(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Log to WandB/Console
            log_metrics({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)
            
            # Save checkpoints
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            save_checkpoint(
                state={
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                checkpoint_dir="checkpoints",
                experiment_name=self.experiment_name,
                epoch=epoch,
                is_best=is_best,
                metric_value=val_loss
            )
            
        logging.info("Training complete.")
        return self.history
