import yaml
import argparse
from yacs.config import CfgNode as CN
from torch.nn import BCELoss

from model_zoo.ResNet101 import ResNet101

import pdb

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, masks_batch):
        model.train()
        yhat = model(images_batch)
        loss = loss_fn(masks_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def load_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = CN(yaml.load(f, Loader=yaml.UnsafeLoader))
    return cfg

def load_args():
    parser = argparse.ArgumentParser(description="Real-synthetic discriminator argument parser")
    parser.add_argument('--config-file', type=str, help="path to the config file")
    
    args = parser.parse_args()
    
    return args

def load_model(cfg):
    model_name = cfg.MODEL.NAME
    if model_name == 'resnet101':
        model = ResNet101(device=cfg.DEVICE)
    else:
        raise NotImplementedError
    
    return model

def get_loss_fn(cfg):
    loss_name = cfg.LOSS.NAME
    if loss_name == 'BCELoss':
        loss = BCELoss()
    else:
        raise NotImplementedError
        
    return loss