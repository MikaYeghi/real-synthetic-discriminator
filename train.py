import os
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.DiscriminatorDataset import DiscriminatorDataset
from utils import load_cfg, load_args, load_model, get_loss_fn, make_train_step, save_state, load_state

from logger import get_logger
logger = get_logger("Train logger")

import pdb

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
    
    """Loss function, optimizer, logging"""
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    loss_fn = get_loss_fn(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.BASE_LR)
    writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, "logs"))
    
    """Training setup"""
    train_step = make_train_step(model, loss_fn, optimizer)
    model, epoch, iter_counter = load_state(model, cfg.MODEL.WEIGHTS)
    
    """Train"""
    for epoch in range(cfg.N_EPOCHS):
        t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, labels_batch in tqdm(train_loader):
            # Compute the loss
            loss = train_step(images_batch, labels_batch)
            loss = loss * cfg.LOSS.COEFFICIENT
            
            # Log the loss
            writer.add_scalar(cfg.LOSS.NAME, loss, iter_counter)
            
            # Save the progress
            if iter_counter % cfg.SAVE_FREQUENCY == 0 and iter_counter != 0:
                save_state(model, epoch, iter_counter, cfg.OUTPUT_DIR, is_final=False)
            
            iter_counter += 1
            
    # Save the final model
    save_state(model, epoch + 1, iter_counter, cfg.OUTPUT_DIR, is_final=True)
            
if __name__ == "__main__":
    args = load_args()
    cfg = load_cfg(args.config_file)
    
    print("-" * 80)
    logger.info(f"Loaded config\n{cfg}")
    print("-" * 80)
    
    main(cfg)