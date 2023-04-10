import os
import glob
from random import shuffle
from torch.utils.data import Dataset

from logger import get_logger
logger = get_logger("Dataset logger")

import pdb

class DiscriminatorDataset(Dataset):
    def __init__(self, cfg):
        super(DiscriminatorDataset, self).__init__()
        
        self.cfg = cfg
        self.metadata = self.extract_metadata(cfg)
        
    def extract_metadata(self, cfg):
        metadata = []
        data_path = cfg.DATA_PATH
        image_format = cfg.IMAGE_FORMAT
        real_images_dir = os.path.join(data_path, "real")
        synthetic_images_dir = os.path.join(data_path, "synthetic")
        
        # Load images
        real_images = glob.glob(real_images_dir + "/*" + image_format)
        synthetic_images = glob.glob(synthetic_images_dir + "/*" + image_format)
        
        # Real - 0
        # Synthetic - 1
        for image_path in real_images:
            metadata_ = {
                "image_path": image_path,
                "category": 0
            }
            metadata.append(metadata_)
        for image_path in synthetic_images:
            metadata_ = {
                "image_path": image_path,
                "category": 1
            }
            metadata.append(metadata_)
            
        if cfg.SHUFFLE:
            shuffle(metadata)
        
        logger.info(f"Loaded a dataset\n-Real: {len(real_images)}\n-Synthetic: {len(synthetic_images)}\n-Total: {len(metadata)}")
        
        return metadata
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        pass