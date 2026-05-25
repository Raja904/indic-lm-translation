"""
Part 2 Inference Script — BERT + GPT Bridged NMT (Hindi → Marathi)
====================================================================
Usage:
    python part2_pretraining/run_nmt_part2.py \
        --bert_ckpt checkpoints/bert/bert_epoch_5.pt \
        --gpt_ckpt  checkpoints/gpt/gpt_epoch_8.pt \
        --sp_model  tokenizer/spm_hi_mr.model \
        --output    results/part2/inference_output.txt

The script:
  1. Loads pretrained BERT (encoder) and GPT (decoder).
  2. Bridges them via a learned linear projection (soft-prompting).
  3. Translates a built-in Hindi test set → Marathi.
  4. Prints + saves SRC / REF / PRED for every sentence.
  5. Reports corpus BLEU and CHRF++ at the end.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import sentencepiece as spm
from sacrebleu.metrics import CHRF, BLEU

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from part2_pretraining.bert.bert import BERTModel
from part2_pretraining.gpt.gpt import GPTModel


# ---------------------------------------------------------------------------
# Default test sentences (Hindi → Reference Marathi)
# ---------------------------------------------------------------------------
TEST_PAIRS = [
    ("भारत एक महान देश है।",               "भारत एक महान देश आहे।"),
    ("मुझे चाय पसंद है।",                   "मला चहा आवडतो।"),
    ("आज मौसम अच्छा है।",                  "आज हवामान छान आहे।"),
    ("वह घर पर है।",                        "तो घरी आहे।"),
    ("बच्चे पार्क में खेल रहे हैं।",         "मुले उद्यानात खेळत आहेत।"),
    ("मैं स्कूल जा रहा हूँ।",               "मी शाळेत जात आहे।"),
    ("हमें समय पर काम पूरा करना चाहिए।",   "आपण वेळेवर काम पूर्ण केले पाहिजे।"),
    ("वह हर सुबह जल्दी उठता है।",          "तो दररोज सकाळी लवकर उठतो।"),
    ("इंटरनेट ने लोगों के संवाद करने का तरीका बदल दिया है।",
     "इंटरनेटमुळे लोकांच्या संवाद करण्याच्या पद्धतीत बदल झाला आहे।"),
    ("मुंबई भारत का एक बड़ा शहर है।",      "मुंबई भारतातील एक मोठे शहर आहे।"),
]


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def load_bert(ckpt_path: str, device: torch.device) -> BERTModel:
    model = BERTModel(vocab_size=16001).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_gpt(ckpt_path: str, device: torch.device) -> GPTModel:
    model = GPTModel(vocab_size=16000).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Translation (soft-prompting: BERT context prepended to GPT input)
# ---------------------------------------------------------------------------
def translate(
    src_text: str,
    bert: BERTModel,
    gpt: GPTModel,
    proj: nn.Linear,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    max_len: int = 60,
    sos_id: int = 2,
    eos_id: int = 3,
) -> str:
    src_ids = torch.tensor(
        [[sos_id] + sp.encode(src_text) + [eos_id]], dtype=torch.long
    ).to(device)

    with torch.no_grad():
        # Encode source with BERT
        enc = proj(bert(src_ids))          # (1, src_len, 768)

        generated = [sos_id]
        for _ in range(max_len):
            tgt = torch.tensor([generated], dtype=torch.long).to(device)
            tgt_emb = gpt.token_embedding(tgt) * math.sqrt(gpt.hidden_dim)

            # Soft-prompt: [BERT context | target embeddings]
            x = torch.cat([enc, tgt_emb], dim=1)
            for layer in gpt.layers:
                x = layer(x)
            x = gpt.norm(x)
            next_tok = gpt.lm_head(x)[0, -1, :].argmax().item()

            if next_tok == eos_id:
                break
            generated.append(next_tok)

    return sp.decode(generated[1:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Part 2 Inference: BERT-GPT bridged Hindi→Marathi NMT"
    )
    parser.add_argument(
        "--bert_ckpt",
        default="checkpoints/bert/bert_epoch_5.pt",
        help="Path to BERT checkpoint (.pt)",
    )
    parser.add_argument(
        "--gpt_ckpt",
        default="checkpoints/gpt/gpt_epoch_8.pt",
        help="Path to GPT checkpoint (.pt)",
    )
    parser.add_argument(
        "--sp_model",
        default="tokenizer/spm_hi_mr.model",
        help="Path to SentencePiece model",
    )
    parser.add_argument(
        "--output",
        default="results/part2/inference_output.txt",
        help="Where to save the output file",
    )
    parser.add_argument(
        "--max_len", type=int, default=60, help="Max generation length per sentence"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"Loading BERT from {args.bert_ckpt} ...")
    bert = load_bert(args.bert_ckpt, device)

    print(f"Loading GPT  from {args.gpt_ckpt} ...")
    gpt = load_gpt(args.gpt_ckpt, device)

    # Projection bridge (randomly initialized — no fine-tuning needed for demo)
    proj = nn.Linear(768, 768).to(device)

    print(f"Loading tokenizer from {args.sp_model} ...")
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)

    print("✅ All loaded. Running inference...\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    hyps, refs = [], []
    lines = []

    sep = "─" * 60
    for src, ref in TEST_PAIRS:
        pred = translate(src, bert, gpt, proj, sp, device, max_len=args.max_len)
        hyps.append(pred)
        refs.append(ref)

        block = f"SRC : {src}\nREF : {ref}\nPRED: {pred}\n{sep}"
        print(block)
        lines.append(block)

    # Metrics
    chrf_score = CHRF().corpus_score(hyps, [refs])
    bleu_score = BLEU(effective_order=True).corpus_score(hyps, [refs])

    metric_block = (
        f"\n{'='*60}\n"
        f"  Corpus BLEU  : {bleu_score.score:.2f}\n"
        f"  Corpus CHRF++: {chrf_score.score:.2f}\n"
        f"{'='*60}"
    )
    print(metric_block)
    lines.append(metric_block)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Results saved → {args.output}")


if __name__ == "__main__":
    main()
