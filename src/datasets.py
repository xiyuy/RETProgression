import os 
import pandas as pd 
from PIL import Image 
import torch 
from torchvision import transforms 
from torch.utils.data import Dataset 
import logging 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class JoslinData(Dataset): 
    def __init__(self, data_dir, annotations_file, img_dir, transform=transforms.ToTensor()): 
        """
        Initialize JoslinData dataset with optimized data loading.
        
        Args:
            data_dir: Base directory containing the data
            annotations_file: CSV file with annotations
            img_dir: Directory containing the images
            transform: Transformations to apply to the images
        """
        self.img_dir = os.path.join(data_dir, img_dir) 
        self.img_labels = pd.read_csv(os.path.join(data_dir, annotations_file), index_col=0)
        self.transform = transform 
        self.label_map = {'NMTM (Non-Referable)': 0, 'MTM (Referable)': 1} # Create a map
        
        # Pre-compute all image paths for faster access
        self.img_paths = [os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.jpeg') 
                        for idx in range(len(self.img_labels))] 
        
        # Calculate class distribution for logging
        class_counts = self.img_labels.iloc[:, 1].value_counts() 
        total_samples = len(self.img_labels) 
        class_distribution = {label: f"{count} ({count/total_samples*100:.1f}%)" 
                            for label, count in class_counts.items()} 
        
        # Log dataset initialization
        logging.info(f"JoslinData: Loaded {len(self.img_labels)} samples from {annotations_file}") 
        logging.info(f"Class distribution: {class_distribution}") 
        
    def __len__(self): 
        return len(self.img_labels) 
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            # Open the image file 
            image = Image.open(img_path)
            
            # Apply transformation with explicit error handling
            try:
                if self.transform:
                    image_transformed = self.transform(image)
                else:
                    image_transformed = image
            except Exception as transform_error:
                print(f"Transform error: {transform_error}")
                raise
            
            # Get the label
            label = self.img_labels.iloc[idx, 1]
            label_tensor = torch.tensor(self.label_map[label], dtype=torch.long)
            
            return image_transformed, label_tensor
            
        except Exception as e:
            print(f"Error in __getitem__ for {img_path}: {str(e)}")
            # Create a proper fallback tensor with the expected shape
            default_img = torch.zeros((3, 224, 224))
            return default_img, torch.tensor(0, dtype=torch.long)

def get_transforms(augmentation_strength='moderate'):
    """
    Returns train and validation transforms based on the specified augmentation strength.
    
    Args:
        augmentation_strength: 'none', 'moderate', or 'strong'
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    # Base validation transform - always the same
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # No augmentation - just resize and convert to tensor
    if augmentation_strength == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    # Moderate augmentation - balanced settings
    elif augmentation_strength == 'moderate':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Less aggressive crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # Fundus images are rotation invariant
            transforms.RandomRotation(15),    # Reduced rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Reduced color distortion
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Gentler affine
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Reduced erasing
        ])
    
    # Strong augmentation - original enhanced settings
    elif augmentation_strength == 'strong':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
    
    return {
        'train': train_transform,
        'val': val_transform
    }

def visualize_augmentations(dataset, num_samples=4, num_augmentations=5, save_path=None):
    """
    Visualize augmentations applied to random samples from the dataset.
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of different samples to visualize
        num_augmentations: Number of augmentations to apply to each sample
        save_path: Path to save the visualization
    """
    # Get a few random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Create a figure to show original images and augmentations
    fig, axs = plt.subplots(num_samples, num_augmentations + 1, figsize=(12, 3 * num_samples))
    
    # If only one sample, ensure axs is 2D
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get original image
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0, 1]
        
        # Display original
        axs[i, 0].imshow(img_np)
        axs[i, 0].set_title(f"Original (Class {label.item()})")
        axs[i, 0].axis('off')
        
        # Display augmentations
        for j in range(1, num_augmentations + 1):
            # Apply transform again to get new augmentation
            img_aug, _ = dataset[idx]
            img_aug_np = img_aug.permute(1, 2, 0).numpy()
            img_aug_np = np.clip(img_aug_np, 0, 1)
            
            axs[i, j].imshow(img_aug_np)
            axs[i, j].set_title(f"Aug {j}")
            axs[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved augmentation visualization to {save_path}")
        plt.close()
    else:
        plt.show()