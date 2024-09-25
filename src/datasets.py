import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms

from torch.utils.data import Dataset


class JoslinData(Dataset):
    def __init__(self, data_dir, annotations_file, img_dir, transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(data_dir, annotations_file))
        self.img_dir = os.path.join(data_dir, img_dir)
        # self.transform = transforms.Compose([
        #     # transforms.ToTensor(),  # Converts to float and normalizes to [0, 1]
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_file_name = str(self.img_labels.iloc[idx, 0]) + '.jpeg' 
        # img_path = os.path.join(self.img_dir, img_file_name)
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        # image = image.float() / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
