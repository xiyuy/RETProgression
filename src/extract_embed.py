import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from extract_embed_utils import (
    ImageDatasetWithPaths, load_metadata, load_model,
    extract_embeddings, build_embedding_dataframe, save_embeddings_to_csv
)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract image embeddings from pretrained models.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output CSV')
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'swinv2'], help='Model type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    return parser.parse_args()

def main(args):
    print("Loading metadata...")
    metadata_df = load_metadata(args.csv_path)

    print("Preparing image dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = ImageDatasetWithPaths(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Loading model...")
    model = load_model(args.model_type, args.checkpoint_path)

    print("Extracting embeddings...")
    embeddings, image_ids = extract_embeddings(model, dataloader, device=args.device)

    print("Merging embeddings with metadata...")
    final_df = build_embedding_dataframe(embeddings, image_ids, metadata_df)

    print("Saving to CSV...")
    save_embeddings_to_csv(final_df, args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
