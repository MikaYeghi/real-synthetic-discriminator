import yaml
import argparse
from yacs.config import CfgNode as CN

def load_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = CN(yaml.load(f, Loader=yaml.UnsafeLoader))
    return cfg

def load_args():
    parser = argparse.ArgumentParser(description="Real-synthetic discriminator argument parser")
    parser.add_argument('--config-file', type=str, help="path to the config file")
    
    args = parser.parse_args()
    
    return args