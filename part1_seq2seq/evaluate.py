import argparse
import torch
from utils.config import load_config, Config
from utils.metrics import get_translation_metrics
from utils.device import get_device

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate LSTM Seq2Seq NMT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/part1/lstm_base.yaml")
    args = parser.parse_args()

    cfg = Config(load_config(args.config))
    device = get_device()

    # TODO: Load Model
    # model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    
    # TODO: Generate translations
    hypotheses = ["नमस्ते", "कसे आहात"]
    references = ["नमस्ते", "कसे आहात"]
    
    metrics = get_translation_metrics(hypotheses, references)
    print(f"Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    evaluate()
