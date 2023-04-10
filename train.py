import os

from dataset.DiscriminatorDataset import DiscriminatorDataset
from utils import load_cfg, load_args

from logger import get_logger
logger = get_logger("Train logger")

import pdb

def main(cfg):
    """Load dataset"""
    train_set = DiscriminatorDataset(cfg)
    

if __name__ == "__main__":
    args = load_args()
    cfg = load_cfg(args.config_file)
    logger.info(f"Loaded config\n{cfg}")
    
    main(cfg)