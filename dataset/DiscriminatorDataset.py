import os
import glob
import torch
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset
from torchvision import transforms

from logger import get_logger
logger = get_logger("Dataset logger")

import pdb

class DiscriminatorDataset(Dataset):
    def __init__(self, cfg, mode="train", device='cuda'):
        super(DiscriminatorDataset, self).__init__()
        
        self.cfg = cfg
        self.device = device
        self.metadata = self.extract_metadata(cfg, mode=mode)
        self.transforms = self.get_transforms(cfg)
        
    def extract_metadata(self, cfg, mode):
        metadata = []
        data_path = cfg.TRAIN_PATH if mode == 'train' else cfg.VALIDATION_PATH
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
        
        logger.info(f"Loaded a {mode} dataset\n-Real: {len(real_images)}\n-Synthetic: {len(synthetic_images)}\n-Total: {len(metadata)}")
        
        return metadata
    
    def get_transforms(self, cfg):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Image data
        data = self.metadata[idx]
        
        # Load the image
        image = Image.open(data['image_path'])
        
        # Convert the image to tensor and move to the device
        image = self.transforms(image).float().to(self.device)
        
        # Extract the label
        label = data['category']
        
        # Convert the label to torch tensor
        label = torch.tensor(label).unsqueeze(0).float().to(self.device)
        
        return image, label