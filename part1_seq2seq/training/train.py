import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
import sacrebleu

# Local imports
from part1_seq2seq.data.dataset import TranslationDataset, collate_fn
from part1_seq2seq.models.encoder import Encoder
from part1_seq2seq.models.decoder import Decoder
from part1_seq2seq.models.seq2seq import Seq2Seq
from part1_seq2seq.inference.beam_search import decode_beam_search
from torch.utils.data import DataLoader

config = {
    "vocab_size": 32000,          # Adjusted to match our trained 32k tokenizer
    "embedding_dim": 256,
    "hidden_dim": 512,
    "n_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,
    "epochs": 16,
    "learning_rate": 1.5e-4,
    "teacher_forcing_ratio": 0.5,
    "clip": 1.0,
    "pad_idx": 0,
    "sos_idx": 2,
    "eos_idx": 3,
    "max_len": 100,
    # Using the tokenizer name we actually generated in our directory
    "sp_model": "tokenizer/spm_hi_mr.model",
    "train_hi": "data/processed/train.hi",
    "train_mr": "data/processed/train.mr",
    "val_hi": "data/processed/val.hi",
    "val_mr": "data/processed/val.mr",
    "checkpoint_dir": "checkpoints",
    "experiment_name": "lstm_random_exp_a",
    "wandb_project": "adivaani-nmt",
    # Beam Search Configs
    "beam_size": 4,
    "length_penalty_alpha": 0.6,
    "no_repeat_bigram": True,
    "repetition_penalty": 1.5,
    "temperature": 0.8,
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

def train_epoch(model, loader, optimizer, criterion, scaler, device, clip):
    """
    Train for a single epoch.
    """
    model.train()
    epoch_loss = 0
    
    from tqdm import tqdm
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        src = batch["src"].to(device)
        src_len = batch["src_len"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        
        # Reconstruct full tgt tensor for seq2seq
        tgt = torch.cat([tgt_in, tgt_out[:, -1:]], dim=1)
        
        optimizer.zero_grad()
        
        # Mixed precision context
        is_cuda = device.type == 'cuda'
        
        # Only use autocast if CUDA is available
        if is_cuda:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                outputs = model(src, src_len, tgt, teacher_forcing_ratio=config["teacher_forcing_ratio"])
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                targets = tgt_out.contiguous().view(-1)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(src, src_len, tgt, teacher_forcing_ratio=config["teacher_forcing_ratio"])
            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim)
            targets = tgt_out.contiguous().view(-1)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    return epoch_loss / len(loader)

def evaluate_epoch(model, loader, criterion, device):
    """
    Evaluate the model using teacher_forcing_ratio = 0.0 (autoregressive decoding).
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            src_len = batch["src_len"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            
            # Reconstruct full tgt tensor for seq2seq
            tgt = torch.cat([tgt_in, tgt_out[:, -1:]], dim=1)
            
            is_cuda = device.type == 'cuda'
            if is_cuda:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    # NO teacher forcing during evaluation
                    outputs = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
                    output_dim = outputs.shape[-1]
                    outputs = outputs.contiguous().view(-1, output_dim)
                    targets = tgt_out.contiguous().view(-1)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                targets = tgt_out.contiguous().view(-1)
                loss = criterion(outputs, targets)
                
            epoch_loss += loss.item()
            
    return epoch_loss / len(loader)

def decode_greedy(model, src, src_len, sp_model, device, max_decode_len=50):
    """
    Greedy decoding for a single source sentence.
    """
    # Extract underlying raw model to avoid DataParallel batch size 1 splitting errors on multiple GPUs
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    
    with torch.no_grad():
        # Create a dummy target tensor filled with pad_idx
        # shape: (1, max_decode_len)
        dummy_tgt = torch.full((1, max_decode_len), config["pad_idx"], dtype=torch.long).to(device)
        
        # First token must be <sos>
        dummy_tgt[0, 0] = config["sos_idx"]
        
        # Run raw model with teacher_forcing=0.0 so it auto-regressively predicts its own next tokens
        # outputs shape: (1, max_decode_len - 1, vocab_size)
        outputs = raw_model(src, src_len, dummy_tgt, teacher_forcing_ratio=0.0)
        
        # Get argmax over vocab dimension
        # predictions shape: (1, max_decode_len - 1)
        predictions = outputs.argmax(dim=2)
        
        # Extract token list and stop at <eos>
        predicted_tokens = []
        for token_id in predictions[0].tolist():
            if token_id == config["eos_idx"]:
                break
            if token_id not in [config["sos_idx"], config["pad_idx"]]:
                predicted_tokens.append(token_id)
                
        # Decode back to string
        decoded_string = sp_model.decode(predicted_tokens)
        
    return decoded_string

def compute_bleu_chrf(model, loader, sp_model, device, num_batches=10):
    """
    Run beam search decode on subset of validation batches to compute BLEU and CHRF.
    """
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            src_batch = batch["src"].to(device)
            src_len_batch = batch["src_len"].to(device)
            tgt_out_batch = batch["tgt_out"].tolist()
            
            batch_size = src_batch.shape[0]
            
            for j in range(batch_size):
                # Extract single source and length
                # src shape: (1, seq_len)
                src = src_batch[j].unsqueeze(0)
                src_len = src_len_batch[j].unsqueeze(0)
                
                # Get prediction string using Beam Search
                pred_str = decode_beam_search(
                    model=model,
                    src=src,
                    src_len=src_len,
                    sp_model=sp_model,
                    device=device,
                    beam_size=config["beam_size"],
                    max_decode_len=50,
                    length_penalty_alpha=config["length_penalty_alpha"],
                    no_repeat_bigram=config["no_repeat_bigram"],
                    repetition_penalty=config["repetition_penalty"],
                    temperature=config["temperature"],
                    pad_idx=config["pad_idx"],
                    sos_idx=config["sos_idx"],
                    eos_idx=config["eos_idx"]
                )
                hypotheses.append(pred_str)
                
                # Extract reference string (stop at <eos>)
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
        
    # Note: sacrebleu's .score attribute already returns a 0-100 scale.
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references], force=True).score
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references]).score
    
    return bleu_score, chrf_score

def init_weights(m):
    """
    Initialize weights with xavier_uniform for linear layers, 
    orthogonal for LSTM weights.
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                if 'lstm' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

def main():
    # 1. Setup Environment
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Init wandb
    wandb.init(project=config["wandb_project"], name=config["experiment_name"], config=config)
    
    # 3. Build Datasets and DataLoaders
    print("Loading data...")
    train_dataset = TranslationDataset(
        hi_path=config["train_hi"],
        mr_path=config["train_mr"],
        sp_model_path=config["sp_model"],
        max_len=config["max_len"]
    )
    val_dataset = TranslationDataset(
        hi_path=config["val_hi"],
        mr_path=config["val_mr"],
        sp_model_path=config["sp_model"],
        max_len=config["max_len"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # 4. Build Model
    print("Building model...")
    encoder = Encoder(
        vocab_size=config["vocab_size"], 
        embedding_dim=config["embedding_dim"], 
        hidden_dim=config["hidden_dim"], 
        n_layers=config["n_layers"], 
        dropout=config["dropout"], 
        pad_idx=config["pad_idx"]
    )
    decoder = Decoder(
        vocab_size=config["vocab_size"], 
        embedding_dim=config["embedding_dim"], 
        hidden_dim=config["hidden_dim"], 
        n_layers=config["n_layers"], 
        dropout=config["dropout"], 
        pad_idx=config["pad_idx"]
    )
    model = Seq2Seq(encoder, decoder, pad_idx=config["pad_idx"], device=device)
    
    # Initialize weights
    model.apply(init_weights)
    
    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 5. Setup Optimizer, Loss, and Scaler
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_idx"])
    
    # Enable scaler only if we are using CUDA to prevent warnings/issues
    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_val_loss = float('inf')
    best_bleu = 0.0
    sp_model = train_dataset.sp
    
    start_epoch = 1
    # Auto-resume: Check if a checkpoint exists and load it
    latest_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_latest.pt")
    if os.path.exists(latest_ckpt_path):
        print(f"Found existing checkpoint at {latest_ckpt_path}. Loading weights to resume...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        # Load weights (handle DataParallel wrapping)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # Override learning rate with the updated value from config
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["learning_rate"]
        print(f"Loaded optimizer state and updated learning rate to {config['learning_rate']}")
        
        start_epoch = checkpoint["epoch"] + 1
        if "val_loss" in checkpoint:
            best_val_loss = checkpoint["val_loss"]
        if "bleu" in checkpoint:
            best_bleu = checkpoint["bleu"]
        print(f"Resuming training from Epoch {start_epoch}!")
    
    # 6. Training Loop
    print("Starting training loop...")
    for epoch in range(start_epoch, config["epochs"] + 1):
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config["clip"])
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        
        # Evaluate metrics on validation subset
        bleu, chrf = compute_bleu_chrf(model, val_loader, sp_model, device, num_batches=10)
        
        # Log to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu": bleu,
            "chrf": chrf,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | BLEU: {bleu:.2f} | CHRF: {chrf:.2f}")
        
        # Checkpointing
        # Extract clean state_dict (without 'module.' prefix) if model is wrapped in DataParallel
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        
        # 1. Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "bleu": bleu,
                "config": config,
            }, ckpt_path)
            print(f"  --> Saved improved best validation loss checkpoint to {ckpt_path}")
            
        # 2. Save best BLEU model separately
        if bleu > best_bleu:
            best_bleu = bleu
            bleu_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_best_bleu.pt")
            torch.save({
                "epoch": epoch,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "bleu": bleu,
                "config": config,
            }, bleu_ckpt_path)
            print(f"  --> Saved improved BLEU checkpoint to {bleu_ckpt_path} (BLEU: {bleu:.2f})")
            
        # 3. Save continuation checkpoints individually: lstm_random_exp_a_epoch_X.pt
        epoch_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "bleu": bleu,
            "config": config,
        }, epoch_ckpt_path)
        print(f"  --> Saved individual epoch checkpoint to {epoch_ckpt_path}")
            
        # 4. Always save latest model to resume training if interrupted
        latest_ckpt_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "bleu": bleu,
            "config": config,
        }, latest_ckpt_path)
        
        # 5. Save fixed qualitative translations after every epoch using Beam Search
        predefined_queries = [
            "यह एक बहुत अच्छा दिन है।",
            "मेरा नाम राजीव है।",
            "मैं स्कूल जा रहा हूँ।",
            "भारत एक महान देश है।",
            "आप कैसे हैं?"
        ]
        samples_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch}_samples.txt")
        with open(samples_path, "w", encoding="utf-8") as f_samp:
            f_samp.write(f"=== Qualitative Samples Epoch {epoch} ===\n\n")
            for query in predefined_queries:
                # Format query string to tensor IDs
                hi_ids = sp_model.encode(query)
                src_ids = [config["sos_idx"]] + hi_ids + [config["eos_idx"]]
                src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
                src_len_tensor = torch.LongTensor([len(src_ids)]).to(device)
                
                # Decode using beam search
                res = decode_beam_search(
                    model=model,
                    src=src_tensor,
                    src_len=src_len_tensor,
                    sp_model=sp_model,
                    device=device,
                    beam_size=config["beam_size"],
                    max_decode_len=50,
                    length_penalty_alpha=config["length_penalty_alpha"],
                    no_repeat_bigram=config["no_repeat_bigram"],
                    repetition_penalty=config["repetition_penalty"],
                    temperature=config["temperature"],
                    pad_idx=config["pad_idx"],
                    sos_idx=config["sos_idx"],
                    eos_idx=config["eos_idx"]
                )
                f_samp.write(f"Hindi:   {query}\n")
                f_samp.write(f"Marathi: {res}\n")
                f_samp.write("-" * 40 + "\n")
        print(f"  --> Saved qualitative samples to {samples_path}")
        
        # 3. Save metrics to a CSV history file for easy graph plotting later
        history_path = os.path.join(config["checkpoint_dir"], f"{config['experiment_name']}_history.csv")
        write_header = not os.path.exists(history_path)
        with open(history_path, "a", encoding="utf-8") as f_hist:
            if write_header:
                f_hist.write("epoch,train_loss,val_loss,bleu,chrf\n")
            f_hist.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{bleu:.4f},{chrf:.4f}\n")
            
    print("Training complete")

if __name__ == "__main__":
    main()
