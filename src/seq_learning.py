# Data Description
# For each patient, we have at least two images from different study dates.
# Each image was converted into a d-dimensional embedding (i.e., vector).
# Therefore, we have a sequence of embeddings for each patient. 
# These embeddings are stored in a csv file, where each row includes the 
# following variables / columns:
# image_id, study_date, disease_label, embedding_0, embedding_1, ...
# Note: Each embedding element is represented as a column

# Code Description
# Given a sequence of image embeddings, we would like to predict the 
# probability of the disease progression in the next year. We will 
# use sequential learning models (e.g., LSTM) to generate the probability.

import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from tqdm import tqdm

# -------- Dataset --------
class PatientSequenceDataset(Dataset):
    def __init__(self, df, embedding_dim):
        self.data = []
        grouped = df.groupby('patient_id')
        for patient_id, group in grouped:
            group = group.sort_values('study_date')
            embeddings = group[[f'embedding_{i}' for i in range(embedding_dim)]].values
            label = group['label'].values[-1]  # target is based on the last image
            self.data.append((torch.tensor(embeddings, dtype=torch.float32),
                              torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# -------- Model --------
class LSTMProgressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]  # last layer's hidden state
        return self.classifier(out)

# -------- Utilities --------
def preprocess_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Extract patient_id from image_id (assuming it's the prefix before an underscore)
    df['patient_id'] = df['image_id'].apply(lambda x: x.split('_')[0])

    # Convert study_date to sortable format
    df['study_date'] = pd.to_datetime(df['study_date'])

    # Encode label (assuming binary classification: 0 = no progression, 1 = progression)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    return df

def train_model(model, dataloader, num_epochs=10, lr=1e-3, device='cuda'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# -------- Main --------
def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Learning on Image Embeddings")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--embedding_dim', type=int, required=True, help='Dimensionality of image embeddings')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main(args):
    print("Loading and preprocessing data...")
    df = preprocess_dataframe(args.csv_path)

    dataset = PatientSequenceDataset(df, args.embedding_dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x))

    print("Initializing model...")
    model = LSTMProgressionModel(input_dim=args.embedding_dim)

    print("Training model...")
    train_model(model, dataloader, num_epochs=args.num_epochs, device=args.device)

def collate_batch(batch):
    sequences, labels = zip(*batch)
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.stack(labels)

if __name__ == "__main__":
    args = parse_args()
    main(args)
