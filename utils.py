import os
import yaml
import torch
import argparse
from torch.nn import BCELoss
from yacs.config import CfgNode as CN
from sklearn.metrics import accuracy_score

from model_zoo.ResNet101 import ResNet101

from logger import get_logger
logger = get_logger("Utils logger")

import pdb

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, labels_batch):
        model.train()
        yhat = model(images_batch)
        loss = loss_fn(yhat, labels_batch)
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

def save_state(model, epoch, iter_counter, save_dir, is_final=False):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "iter_counter": iter_counter,
    }
    if is_final:
        save_path = os.path.join(save_dir, "model_final.pth")
    else:
        save_path = os.path.join(save_dir, f"model_{iter_counter + 1}.pth")
    torch.save(checkpoint, save_path)
    
def load_state(model, state_path):
    if state_path is None:
        logger.info("No checkpoint path. Starting from scratch.")
        return model, 0, 0
    else:
        checkpoint = torch.load(state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        iter_counter = checkpoint['iter_counter']
        logger.info(f"Loaded checkpoint from {state_path}")
        return model, epoch, iter_counter
    
def compute_metrics(total_gt, total_preds):
    accuracy = accuracy_score(total_gt, total_preds)
    
    results = {
        "accuracy": accuracy
    }
    return results