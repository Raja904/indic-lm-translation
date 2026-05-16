import argparse
from utils.config import load_config, Config
from utils.seed import seed_everything
from utils.device import get_device
from utils.logging import setup_logger, init_wandb

def main():
    parser = argparse.ArgumentParser(description="GPT-style Pretraining")
    parser.add_argument("--config", type=str, default="configs/part2/gpt_base.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    cfg = Config(cfg_dict)
    seed_everything(cfg.get("seed", 42))
    logger = setup_logger(log_name=f"{cfg.experiment_name}.log")
    init_wandb(project_name="indic-pretraining", experiment_name=cfg.experiment_name, config=cfg_dict)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info("Starting GPT pretraining...")

if __name__ == "__main__":
    main()
