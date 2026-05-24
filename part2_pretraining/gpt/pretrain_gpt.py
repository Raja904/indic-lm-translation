import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import sentencepiece as spm
import wandb
import sys
import glob

try:
    from part2_pretraining.gpt.gpt import GPTModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from part2_pretraining.gpt.gpt import GPTModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = {
    "vocab_size": 16000,
    "hidden_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "n_kv_heads": 4,
    "ffn_dim": 4096,
    "max_seq_len": 256,
    "dropout": 0.1,
    "batch_size": 64,
    "epochs": 12,
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "clip": 1.0,
    "pad_idx": 0,
    "checkpoint_dir": "checkpoints/gpt",
    "experiment_name": "gpt_clm_pretrain",
    "wandb_project": "adivaani-nmt",
    "train_file": "data/processed/train.mr",
    "val_file": "data/processed/val.mr",
    "sp_model": "tokenizer/spm_hi_mr.model",
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
class CLMDataset(Dataset):
    def __init__(self, file_path, sp_model_path, max_seq_len=256, sos_id=2, eos_id=3):
        self.sp_model_path = sp_model_path
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.max_seq_len = max_seq_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
            
    def __len__(self):
        return len(self.lines)
        
    def __getitem__(self, idx):
        text = self.lines[idx]
        tokens = self.sp.encode(text)
        
        # Add SOS and EOS
        tokens = [self.sos_id] + tokens + [self.eos_id]
        
        # Truncate to max_seq_len + 1 (since we need N+1 tokens to get N inputs and N labels)
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[:self.max_seq_len + 1]
            
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def collate_fn_clm(batch, pad_id=0):
    input_ids, labels = zip(*batch)
    
    lengths = [len(x) for x in input_ids]
    max_len = max(lengths)
    
    padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long) # -100 is ignored by CrossEntropyLoss
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    
    for i in range(len(batch)):
        seq_len = len(input_ids[i])
        padded_inputs[i, :seq_len] = input_ids[i]
        padded_labels[i, :seq_len] = labels[i]
        attention_mask[i, :seq_len] = 1
        
    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask,
        "labels": padded_labels
    }

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def get_lr_multiplier(step, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0

def evaluate_epoch(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast():
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    seed_everything(42)
    wandb.init(project=config["wandb_project"], name=config["experiment_name"], config=config)
    
    # Dataset and Dataloader
    dataset = CLMDataset(config["train_file"], config["sp_model"], max_seq_len=config["max_seq_len"])
    val_dataset = CLMDataset(config["val_file"], config["sp_model"], max_seq_len=config["max_seq_len"])
    
    def collate_wrapper(batch):
        return collate_fn_clm(batch, pad_id=config["pad_idx"])
        
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_wrapper, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_wrapper, num_workers=0)
    
    # Initialize Model
    model = GPTModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr_multiplier(step, config["warmup_steps"]))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler()
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    global_step = 0
    start_epoch = 0
    
    # ===== AUTO-RESUME FROM LATEST CHECKPOINT =====
    checkpoint_files = glob.glob(os.path.join(config["checkpoint_dir"], "gpt_epoch_*.pt"))
    if checkpoint_files:
        # Get the latest epoch checkpoint
        checkpoint_path = sorted(checkpoint_files)[-1]
        print(f"Found latest checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        
        print(f"✅ Resumed training from epoch {start_epoch}, global_step {global_step}")
    else:
        print("⚠️ No checkpoints found, starting from epoch 0")
    
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, config["vocab_size"]), labels.view(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            global_step += 1
            
            if global_step % 100 == 0:
                perplexity = math.exp(loss.item()) if loss.item() < 100 else float('inf')
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | PPL: {perplexity:.4f}")
                wandb.log({"train_loss": loss.item(), "perplexity": perplexity, "epoch": epoch, "step": global_step, "lr": scheduler.get_last_lr()[0]})
                
            # Save mid-epoch checkpoint every 2000 steps to prevent progress loss
            if global_step % 2000 == 0:
                step_ckpt = os.path.join(config["checkpoint_dir"], f"gpt_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss.item(),
                }, step_ckpt)
                print(f"Saved mid-epoch checkpoint to {step_ckpt}")
                
        # Validation phase
        val_loss = evaluate_epoch(model, val_dataloader, criterion, device, config)
        val_perplexity = math.exp(val_loss) if val_loss < 100 else float('inf')
        print(f"Epoch {epoch} | Train completed | Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.4f}")
        wandb.log({"val_loss": val_loss, "val_perplexity": val_perplexity, "epoch": epoch, "step": global_step})

        # Save comprehensive checkpoint per epoch
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"gpt_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss.item(),
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }, checkpoint_path)
        print(f"Saved epoch checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
