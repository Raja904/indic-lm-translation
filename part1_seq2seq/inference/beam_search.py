import torch
import torch.nn as nn
import torch.nn.functional as F

def decode_beam_search(
    model, 
    src, 
    src_len, 
    sp_model, 
    device, 
    beam_size=4, 
    max_decode_len=100, 
    length_penalty_alpha=0.6, 
    no_repeat_bigram=True, 
    repetition_penalty=1.5, 
    temperature=0.8,
    pad_idx=0,
    sos_idx=2,
    eos_idx=3
):
    """
    Beam Search decoding for Seq2Seq model with length normalization, 
    repetition penalties, bigram blocking, and temperature scaling.
    """
    # Extract underlying raw model to avoid DataParallel batch size 1 splitting errors on multiple GPUs
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    
    # 1. Run encoder once to get hidden, cell, and encoder outputs
    with torch.no_grad():
        encoder_outputs, hidden, cell = raw_model.encoder(src, src_len)
        mask = raw_model.create_mask(src)
        
    # A beam state is represented as a dictionary:
    # {
    #     "tokens": [sos_idx, ...],
    #     "log_prob": float, (sum of log probabilities of tokens)
    #     "hidden": tensor of shape (n_layers, 1, hidden_dim),
    #     "cell": tensor of shape (n_layers, 1, hidden_dim),
    #     "score": float (length-normalized score)
    # }
    initial_beam = {
        "tokens": [sos_idx],
        "log_prob": 0.0,
        "hidden": hidden,
        "cell": cell,
        "score": 0.0
    }
    
    active_beams = [initial_beam]
    completed_beams = []
    
    # Decode step-by-step
    for step in range(1, max_decode_len):
        if not active_beams:
            break
            
        candidates = []
        
        for beam in active_beams:
            # Get last token in current beam
            last_token = beam["tokens"][-1]
            decoder_input = torch.LongTensor([last_token]).to(device)
            
            with torch.no_grad():
                logits, new_hidden, new_cell, _ = raw_model.decoder(
                    decoder_input, beam["hidden"], beam["cell"], encoder_outputs, mask
                )
            
            # Apply Temperature scaling
            if temperature > 0.0 and temperature != 1.0:
                logits = logits / temperature
                
            # Apply repetition blocking and penalties
            # A. Prevent immediate token repetition (cannot select last_token again)
            if len(beam["tokens"]) >= 1:
                prev_token = beam["tokens"][-1]
                logits[0, prev_token] = -1e9
                
            # B. No-repeat bigram blocking
            if no_repeat_bigram and len(beam["tokens"]) >= 2:
                # Find all bigrams already present in the sequence
                existing_bigrams = set()
                for i in range(len(beam["tokens"]) - 1):
                    existing_bigrams.add((beam["tokens"][i], beam["tokens"][i+1]))
                
                # The next bigram would be (last_token, next_token)
                # If this bigram already exists, block the next_token
                for token_id in range(logits.shape[1]):
                    if (prev_token, token_id) in existing_bigrams:
                        logits[0, token_id] = -1e9
                        
            # C. Configurable general repetition penalty (penalize any previously generated tokens)
            if repetition_penalty > 0.0:
                unique_tokens = set(t for t in beam["tokens"] if t not in [sos_idx, pad_idx])
                for token_id in unique_tokens:
                    # Subtract penalty from logits to reduce likelihood of choosing it
                    logits[0, token_id] -= repetition_penalty
                    
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Retrieve top beam_size + 2 candidates (over-retrieve slightly to account for blocked tokens)
            topk_log_probs, topk_indices = log_probs.topk(min(beam_size + 2, logits.shape[1]), dim=-1)
            
            for log_p, token_idx in zip(topk_log_probs[0].tolist(), topk_indices[0].tolist()):
                if log_p < -1e8:
                    continue  # Filter out blocked tokens
                    
                new_tokens = beam["tokens"] + [token_idx]
                new_log_prob = beam["log_prob"] + log_p
                
                # Calculate length normalized score
                length = len(new_tokens)
                score = new_log_prob / (length ** length_penalty_alpha)
                
                candidate = {
                    "tokens": new_tokens,
                    "log_prob": new_log_prob,
                    "hidden": new_hidden,
                    "cell": new_cell,
                    "score": score
                }
                
                if token_idx == eos_idx:
                    completed_beams.append(candidate)
                else:
                    candidates.append(candidate)
                    
        # Sort all accumulated candidates across all beams by length-normalized score descending
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        # Retain the top beam_size candidates
        active_beams = candidates[:beam_size]
        
    # If any beams reached <eos>, select the highest-scoring completed one.
    # Otherwise, fall back to the highest-scoring active beam.
    if completed_beams:
        best_beam = max(completed_beams, key=lambda x: x["score"])
    elif active_beams:
        best_beam = active_beams[0]
    else:
        return ""
        
    # Decode token IDs back to a string, skipping padding and special tokens
    decoded_tokens = [t for t in best_beam["tokens"] if t not in [sos_idx, eos_idx, pad_idx]]
    return sp_model.decode(decoded_tokens)
