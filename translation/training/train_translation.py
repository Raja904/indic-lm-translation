import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import sacrebleu

# Local imports using try/except
try:
    from part1_seq2seq.data.dataset import TranslationDataset, collate_fn
    from translation.models.translation_model import TranslationModel
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from part1_seq2seq.data.dataset import TranslationDataset, collate_fn
    from translation.models.translation_model import TranslationModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = {
    "vocab_size": 16000,
    "hidden_dim": 768,
    "batch_size": 32,
    "epochs": 4,
    "learning_rate": 5e-5,  # smaller lr, finetuning pretrained weights
    "warmup_steps": 500,
    "clip": 1.0,
    "pad_idx": 0,
    "sos_idx": 2,
    "eos_idx": 3,
    "teacher_forcing_ratio": 0.5,
    "bert_checkpoint": "checkpoints/bert/bert_best.pt",
    "gpt_checkpoint": "checkpoints/gpt/gpt_best.pt",
    "train_hi": "data/processed/train.hi",
    "train_mr": "data/processed/train.mr",
    "val_hi": "data/processed/val.hi",
    "val_mr": "data/processed/val.mr",
    "sp_model": "tokenizer/spm_hi_mr.model",
    "checkpoint_dir": "checkpoints/translation",
    "experiment_name": "translation_bert_gpt",
    "wandb_project": "adivaani-nmt",
}


def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_lr_multiplier(step, warmup_steps):
    """Calculates linear learning rate warmup multiplier."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def decode_greedy(model, src, sp_model, device, max_decode_len=50, sos_idx=2, eos_idx=3, pad_idx=0):
    """
    Greedy decoding for a single source sentence using the Transformer model.
    """
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    
    with torch.no_grad():
        # src shape: (1, src_len)
        # Start target sequence with SOS token
        # tgt_ids shape: (1, 1)
        tgt_ids = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        
        for _ in range(max_decode_len):
            # Forward pass: logits shape: (1, current_tgt_len, vocab_size)
            outputs = raw_model(src, tgt_ids)
            
            # Extract prediction for the last position
            # next_token shape: scalar
            next_token = outputs[0, -1, :].argmax().item()
            
            # Append predicted token to sequence
            # next_token_tensor shape: (1, 1)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            # tgt_ids shape: (1, current_tgt_len + 1)
            tgt_ids = torch.cat([tgt_ids, next_token_tensor], dim=1)
            
            # Stop decoding if we hit EOS
            if next_token == eos_idx:
                break
                
        predicted_tokens = tgt_ids[0].tolist()
        # Filter out special tokens for clean text decoding
        clean_tokens = [t for t in predicted_tokens if t not in (sos_idx, eos_idx, pad_idx)]
        # Decode back to string
        decoded_string = sp_model.decode(clean_tokens)
        
    return decoded_string


def compute_bleu_chrf(model, loader, sp_model, device, num_batches=10):
    """
    Compute BLEU and CHRF scores on a subset of the validation set.
    """
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            # src_batch shape: (batch_size, src_len)
            src_batch = batch["src"].to(device)
            # tgt_out_batch shape: (batch_size, tgt_len)
            tgt_out_batch = batch["tgt_out"].tolist()
            
            batch_size = src_batch.shape[0]
            
            for j in range(batch_size):
                # Extract single source sequence
                # src shape: (1, src_len)
                src = src_batch[j].unsqueeze(0)
                
                # Perform greedy decoding
                pred_str = decode_greedy(
                    model=model,
                    src=src,
                    sp_model=sp_model,
                    device=device,
                    max_decode_len=50,
                    sos_idx=config["sos_idx"],
                    eos_idx=config["eos_idx"],
                    pad_idx=config["pad_idx"]
                )
                hypotheses.append(pred_str)
                
                # Decode reference sentence, stopping at EOS
                ref_tokens = []
                for token_id in tgt_out_batch[j]:
                    if token_id == config["eos_idx"]:
                        break
                    if token_id not in [config["pad_idx"], config["sos_idx"]]:
                        ref_tokens.append(token_id)
                        
                ref_str = sp_model.decode(ref_tokens)
                references.append(ref_str)
                
    if not hypotheses:
        return 0.0, 0.0
        
    # corpus metrics: sacrebleu expects list of system hypotheses and list of references lists
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references], force=True).score
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    
    return bleu_score, chrf_score


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, epoch, global_step):
    """
    Runs training for one epoch and logs loss values.
    """
    model.train()
    epoch_loss = 0
    is_cuda = (device.type == 'cuda')
    
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    
    for batch in pbar:
        # src shape: (batch_size, src_len)
        src = batch["src"].to(device)
        # tgt_in shape: (batch_size, tgt_len - 1)
        tgt_in = batch["tgt_in"].to(device)
        # tgt_out shape: (batch_size, tgt_len - 1)
        tgt_out = batch["tgt_out"].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if is_cuda:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                # logits shape: (batch_size, tgt_len - 1, vocab_size)
                logits = model(src, tgt_in)
                vocab_size = logits.shape[-1]
                
                # Flatten logits: (batch_size * (tgt_len - 1), vocab_size)
                # Flatten target outputs: (batch_size * (tgt_len - 1))
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            # logits shape: (batch_size, tgt_len - 1, vocab_size)
            logits = model(src, tgt_in)
            vocab_size = logits.shape[-1]
            
            # Flatten logits: (batch_size * (tgt_len - 1), vocab_size)
            # Flatten target outputs: (batch_size * (tgt_len - 1))
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            optimizer.step()
            
        scheduler.step()
        global_step += 1
        epoch_loss += loss.item()
        
        # Live logging printing progress every 100 steps
        if global_step % 100 == 0:
            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
            wandb.log({"train_step_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0], "step": global_step})
            
        # Save mid-epoch checkpoint every 2000 steps to prevent progress loss
        if global_step % 2000 == 0:
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            step_ckpt = os.path.join(config["checkpoint_dir"], f"translation_step_{global_step}.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state': raw_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if is_cuda else None,
                'loss': loss.item(),
            }, step_ckpt)
            print(f"Saved mid-epoch checkpoint to {step_ckpt}")
            
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return epoch_loss / len(loader), global_step


def evaluate_epoch(model, loader, criterion, device):
    """
    Runs evaluation for one epoch.
    """
    model.eval()
    epoch_loss = 0
    is_cuda = (device.type == 'cuda')
    
    with torch.no_grad():
        for batch in loader:
            # src shape: (batch_size, src_len)
            src = batch["src"].to(device)
            # tgt_in shape: (batch_size, tgt_len - 1)
            tgt_in = batch["tgt_in"].to(device)
            # tgt_out shape: (batch_size, tgt_len - 1)
            tgt_out = batch["tgt_out"].to(device)
            
            if is_cuda:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    # logits shape: (batch_size, tgt_len - 1, vocab_size)
                    logits = model(src, tgt_in)
                    vocab_size = logits.shape[-1]
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            else:
                # logits shape: (batch_size, tgt_len - 1, vocab_size)
                logits = model(src, tgt_in)
                vocab_size = logits.shape[-1]
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)


def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project=config["wandb_project"], name=config["experiment_name"], config=config)
    
    # 1. Datasets and DataLoaders
    print("Loading datasets...")
    train_dataset = TranslationDataset(
        hi_path=config["train_hi"],
        mr_path=config["train_mr"],
        sp_model_path=config["sp_model"],
        max_len=100
    )
    val_dataset = TranslationDataset(
        hi_path=config["val_hi"],
        mr_path=config["val_mr"],
        sp_model_path=config["sp_model"],
        max_len=100
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # 2. Build NMT Model
    print("Building TranslationModel...")
    model = TranslationModel(
        bert_checkpoint_path=config["bert_checkpoint"],
        gpt_checkpoint_path=config["gpt_checkpoint"],
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        pad_idx=config["pad_idx"],
        device=device,
        ffn_dim=4096
    )
    
    # DataParallel wrapper if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 3. Setup Optimizer, Warmup Scheduler, Loss and GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(step, config["warmup_steps"])
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_idx"])
    
    is_cuda = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_loss = float('inf')
    sp_model = train_dataset.sp
    
    start_epoch = 1
    global_step = 0
    
    # Auto-resume logic if continuation checkpoint exists
    latest_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_latest.pt")
    if os.path.exists(latest_ckpt_path):
        print(f"Found existing translation checkpoint at {latest_ckpt_path}. Loading weights to resume...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(checkpoint["model_state"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None and is_cuda:
            scaler.load_state_dict(checkpoint["scaler_state"])
            
        start_epoch = checkpoint["epoch"] + 1
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
            
        print(f"Resuming training from Epoch {start_epoch} (Step {global_step})!")
        
    # 4. Training Loop
    print("Starting training loop...")
    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step
        )
        
        print(f"Epoch {epoch} complete. Running evaluation...")
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        
        # Calculate BLEU/CHRF on validation subset (10 batches)
        bleu, chrf = compute_bleu_chrf(model, val_loader, sp_model, device, num_batches=10)
        
        # Log epoch metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu": bleu,
            "chrf": chrf,
            "epoch": epoch,
            "global_step": global_step
        })
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.2f} | CHRF: {chrf:.2f}")
        
        # Checkpointing
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # Save best model by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_best.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "bleu": bleu,
                "config": config,
            }, best_ckpt_path)
            print(f"  --> Saved improved best validation checkpoint to {best_ckpt_path}")
            
        # Save latest model for resume functionality
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if is_cuda else None,
            "val_loss": val_loss,
            "bleu": bleu,
            "config": config,
        }, latest_ckpt_path)
        print(f"  --> Saved latest epoch checkpoint to {latest_ckpt_path}")
        
    print("Training complete.")


if __name__ == "__main__":
    main()
