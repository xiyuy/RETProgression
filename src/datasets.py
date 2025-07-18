import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

class JoslinData(Dataset):
    """Dataset for medical image classification with optimized data loading"""
    
    def __init__(self, data_dir, annotations_file, img_dir, transform=transforms.ToTensor()):
        """
        Initialize JoslinData dataset
        
        Args:
            data_dir: Base directory containing the data
            annotations_file: CSV file with annotations
            img_dir: Directory containing the images
            transform: Transformations to apply to the images
        """
        self.img_dir = os.path.join(data_dir, img_dir)
        self.img_labels = pd.read_csv(os.path.join(data_dir, annotations_file), index_col=0)
        # self.img_labels = self.img_labels.iloc[:1000, :]  # Limit to first 1000 samples
        self.transform = transform
        self.label_map = {'NMTM': 0, 'MTM': 1}
        # self.label_map = {'NMTM (Non-Referable)': 0, 'MTM (Referable)': 1}
        
        # Pre-compute image paths for faster access
        self.img_paths = [os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0]}.jpg") 
                          for idx in range(len(self.img_labels))]
        
        # Log dataset statistics
        self._log_dataset_info()
    
    def _log_dataset_info(self):
        """Log dataset information and class distribution"""
        class_counts = self.img_labels.iloc[:, 1].value_counts()
        total_samples = len(self.img_labels)
        class_distribution = {label: f"{count} ({count/total_samples*100:.1f}%)" 
                             for label, count in class_counts.items()}
        
        logging.info(f"JoslinData: Loaded {total_samples} samples from {self.img_dir}")
        logging.info(f"Class distribution: {class_distribution}")
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """Get image and label for a given index with robust error handling"""
        img_path = self.img_paths[idx]
        
        # try:
        # Open and transform image
        image = Image.open(img_path)
        image_transformed = self.transform(image) if self.transform else image
        
        # Get label
        label = self.img_labels.iloc[idx, 1]
        label_tensor = torch.tensor(self.label_map[label], dtype=torch.long)
        # label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_transformed, label_tensor
            
        # except Exception as e:
        #     logging.warning(f"Error in __getitem__ for {img_path}: {str(e)}")
        #     # Return fallback tensor with expected shape
        #     return torch.zeros((3, 224, 224)), torch.tensor(0, dtype=torch.long)


def get_transforms(augmentation_strength='moderate', resolution=224):
    """
    Get train and validation transforms based on specified augmentation strength
    
    Args:
        augmentation_strength: 'none', 'moderate', or 'strong'
        resolution: Image resolution (height and width in pixels)
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    # Base validation transform
    val_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    
    # Select train transform based on augmentation strength
    if augmentation_strength == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])
    elif augmentation_strength == 'moderate':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)), # Make sure augmentation is valid from the clinical perspective
        ])
    elif augmentation_strength == 'strong':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
    
    return {
        'train': train_transform,
        'val': val_transform
    }

