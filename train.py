import os
import torch
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

from dataset.DiscriminatorDataset import DiscriminatorDataset
from utils import load_cfg, load_args, load_model, get_loss_fn, make_train_step, save_state, load_state, compute_metrics

from logger import get_logger
logger = get_logger("Train logger")

import pdb

def run_eval(model, val_loader, max_iter):
    total_preds = torch.empty((0, 1))
    total_gt = torch.empty((0, 1))
    max_iter = min(len(val_loader), max_iter)
    with torch.no_grad():
        for idx, (images_batch, labels_batch) in enumerate(val_loader):
            if idx >= max_iter:
                break
            
            model.eval()
            preds = model(images_batch)
            
            # Update the global lists
            total_preds = torch.cat((total_preds, preds.cpu()))
            total_gt = torch.cat((total_gt, labels_batch.cpu()))
    total_preds = (total_preds > 0.5).float()
    
    # Compute the metrics
    results = compute_metrics(total_gt, total_preds)
    
    return results

def main(cfg):
    """Load dataset"""
    train_set = DiscriminatorDataset(cfg, mode="train")
    val_set = DiscriminatorDataset(cfg, mode="validation")
    print("-" * 80)
    
    """Create the dataloaders"""
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE)
    
    """Model"""
    model = load_model(cfg)
    
    """Loss function, optimizer, scheduler, logging"""
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    loss_fn = get_loss_fn(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.BASE_LR)
    scheduler = ExponentialLR(optimizer, gamma=cfg.LR_GAMMA)
    writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, "logs"))
    
    """Training setup"""
    train_step = make_train_step(model, loss_fn, optimizer)
    model, epoch, iter_counter = load_state(model, cfg.MODEL.WEIGHTS)
    
    """Train"""
    logger.info("Start training")
    for epoch in range(cfg.N_EPOCHS):
        t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, labels_batch in t:
            # Compute the loss
            loss = train_step(images_batch, labels_batch)
            loss = loss * cfg.LOSS.COEFFICIENT
            
            # Save the progress
            if iter_counter % cfg.SAVE_FREQ == 0 and iter_counter != 0:
                save_state(model, epoch, iter_counter, cfg.OUTPUT_DIR, is_final=False)
            
            # Run evaluation
            if iter_counter % cfg.EVAL.FREQ == 0 and iter_counter != 0:
                results = run_eval(model, val_loader, cfg.EVAL.MAX_ITER)
                
                # Log the results of validation
                for k, v in results.items():
                    writer.add_scalar(k, v, iter_counter)
            
            # Logging
            writer.add_scalar(cfg.LOSS.NAME, loss, iter_counter)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], iter_counter)
            
            iter_counter += 1
            
            t.set_description(f"Epoch #{epoch + 1}")
        
        # LR scheduler step
        scheduler.step()
            
    # Save the final model
    save_state(model, epoch + 1, iter_counter, cfg.OUTPUT_DIR, is_final=True)
            
if __name__ == "__main__":
    args = load_args()
    cfg = load_cfg(args.config_file)
    
    print("-" * 80)
    logger.info(f"Loaded config\n{cfg}")
    print("-" * 80)
    
    main(cfg)