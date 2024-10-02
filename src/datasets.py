import os
import pandas as pd
# from torchvision.io import read_image
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset


class JoslinData(Dataset):
    def __init__(self, data_dir, annotations_file, img_dir, transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(data_dir, annotations_file))
        self.img_dir = os.path.join(data_dir, img_dir)
        self.transform = transform # transform is None
        self.transform = transforms.Compose([
            # transforms.PILToTensor(), # Convert PIL image to tensor w/o normalize to range [0,1]
            transforms.ToTensor(),  # Convert PIL image to tensor and normalize to range [0,1].
            # lambda x: x/255.0, # Normalize the tensor to range [0,1]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_file_name = str(self.img_labels.iloc[idx, 1]) + '.jpeg' 
        img_path = os.path.join(self.img_dir, img_file_name)
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
