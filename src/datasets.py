# import os
# import pandas as pd
# from torchvision.io import read_image

# from torch.utils.data import Dataset


# class JoslinData(Dataset):
#     def __init__(self, data_dir, annotations_file, img_dir, transform=None,
#                  target_transform=None):
#         self.img_labels = pd.read_csv(os.join(data_dir, annotations_file))
#         self.img_dir = os.join(data_dir, img_dir)
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

import os
import pandas as pd
from google.cloud import storage
from torchvision.io import read_image
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

class JoslinData(Dataset):
    def __init__(self, bucket_name, annotations_file_path, indices = None, transform=None, target_transform=None):
        # Initialize Google Cloud Storage client
        self.client = storage.Client() #gcp
        self.bucket = self.client.get_bucket(bucket_name) #gcp

        # Load labels from the CSV file, filter out rows with missing labels
        self.img_labels = pd.read_csv(annotations_file_path, index_col=0)  #gcp
        
        self.img_labels.dropna(subset=['DR'], inplace=True) #gcp
        self.classes = self.img_labels['DR'].unique().tolist() #gcp

        # Get the list of available image ids in the bucket
        available_images = [blob.name.split('.')[0] for blob in self.bucket.list_blobs(max_results=500)] #gcp
        
        #image ids to string and then getting their labels
        self.img_labels.index = self.img_labels.index.astype(str) #gcp
        self.img_labels = self.img_labels[self.img_labels.index.isin(available_images)]  # Filter to include only available images
        
        # If indices are provided, use only the subset of data
        if indices is not None:
            self.img_labels = self.img_labels.iloc[indices]

        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.index[idx]
        img_path = f"{img_id}.jpeg"

        # Fetch the image from GCP bucket
        blob = self.bucket.blob(img_path)
        img_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Convert to tensor
        image = self.transform(image) if self.transform else torch.tensor(image)

        # Get the corresponding label
        label = self.img_labels.iloc[idx, 2]  # Assuming labels are in the first column
        label = self.target_transform(label) if self.target_transform else label

        return image, label
