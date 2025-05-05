import os 
import pandas as pd 
from PIL import Image 
import torch 
from torchvision import transforms 
from torch.utils.data import Dataset 
import io 
from concurrent.futures import ThreadPoolExecutor 
import logging 

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
        """
        Get an item from the dataset with optimized image loading.
        
        Args:
            idx: Index of the item to retrieve
        
        Returns:
            tuple: (image, label) where image is the transformed image and label is the class index
        """
        # Use pre-computed path instead of constructing it each time
        img_path = self.img_paths[idx] 
        
        # Optimized image loading
        try: 
            # Open the image file directly
            image = Image.open(img_path) 
            
            # Apply transformation if provided
            if self.transform: 
                image = self.transform(image) 
            
            # Get the label and convert to tensor
            label = self.img_labels.iloc[idx, 1] 
            label_tensor = torch.tensor(self.label_map[label], dtype=torch.long) 
            
            return image, label_tensor 
        
        except Exception as e: 
            logging.error(f"Error loading image {img_path}: {str(e)}") 
            # Return a default value in case of error
            if self.transform: 
                default_img = torch.zeros((3, 224, 224)) 
            else: 
                default_img = Image.new('RGB', (224, 224)) 
            return default_img, torch.tensor(0, dtype=torch.long) 