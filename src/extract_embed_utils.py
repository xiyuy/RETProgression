import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm

# ---------- Image Loader ----------

class ImageDatasetWithPaths(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_id = os.path.basename(path)
        return image, image_id

# ---------- Metadata ----------

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    expected_columns = {'image_id', 'study_date', 'label'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {expected_columns - set(df.columns)}")
    return df

# ---------- Model Loader ----------

def load_model(model_type, checkpoint_path=None):
    model_map = {
        'resnet': 'resnet50',
        'vit': 'vit_base_patch16_224',
        'swinv2': 'swinv2_base_window12to16_192to256_22kft1k'
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = timm.create_model(model_map[model_type], pretrained=True)
    model.reset_classifier(0)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = state_dict.get('model', state_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")

    model.eval()
    return model

# ---------- Embedding Extraction ----------

def extract_embeddings(model, dataloader, device='cuda'):
    model = model.to(device)
    embeddings, image_ids = [], []

    with torch.no_grad():
        for images, ids in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu())
            image_ids.extend(ids)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, image_ids

# ---------- Save Results ----------

def build_embedding_dataframe(embeddings, image_ids, metadata_df):
    embedding_dim = embeddings.shape[1]
    embedding_df = pd.DataFrame(embeddings.numpy(), columns=[f'embedding_{i}' for i in range(embedding_dim)])
    embedding_df.insert(0, 'image_id', image_ids)

    merged_df = pd.merge(embedding_df, metadata_df, on='image_id', how='left')
    if merged_df['label'].isnull().sum() > 0:
        print("Warning: Some image IDs were not found in metadata.")
    return merged_df

def save_embeddings_to_csv(df, output_dir, output_filename='image_embeddings.csv'):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    df.to_csv(out_path, index=False)
    print(f"Saved embeddings to {out_path}")
