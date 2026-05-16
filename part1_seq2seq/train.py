import argparse
from utils.config import load_config, Config
from utils.seed import seed_everything
from utils.device import get_device
from utils.logging import setup_logger, init_wandb

def main():
    parser = argparse.ArgumentParser(description="Train LSTM Seq2Seq NMT")
    parser.add_argument("--config", type=str, default="configs/part1/lstm_base.yaml", help="Path to config file")
    args = parser.parse_args()

    # 1. Load config
    cfg_dict = load_config(args.config)
    cfg = Config(cfg_dict)

    # 2. Seed everything
    seed_everything(cfg.get("seed", 42))

    # 3. Setup logger and WandB
    logger = setup_logger(log_name=f"{cfg.experiment_name}.log")
    init_wandb(
        project_name="indic-nmt-part1",
        experiment_name=cfg.experiment_name,
        config=cfg_dict
    )

    # 4. Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # 5. TODO: Initialize Tokenizer, Dataset, Dataloader
    # 6. TODO: Initialize Model, Optimizer, Criterion
    # 7. TODO: Initialize Trainer and call fit()
    
    logger.info("Starting training for Part 1 (Seq2Seq)...")

if __name__ == "__main__":
    main()
