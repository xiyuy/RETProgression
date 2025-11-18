# test_inference.py
import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from PIL import Image, UnidentifiedImageError

# Project imports
from datasets import JoslinData
from custom_metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix_with_stats

# -------------------------
# Logging
# -------------------------
def configure_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'test_inference.log')
    try:
        with open(log_file, 'w') as f:
            f.write(f"=== New Test Inference Run: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    except Exception as e:
        print(f"Warning: Could not clear log file: {e}")

    logger = logging.getLogger()
    logger.handlers = []
    fmt = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a'); fh.setFormatter(fmt)
    ch = logging.StreamHandler();                ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch); logger.setLevel(logging.INFO)
    return logger

# -------------------------
# Model
# -------------------------
def load_model(checkpoint_path, model_name, num_classes=2, img_size=1024, device="cuda"):
    model = create_model(model_name, pretrained=False, num_classes=num_classes, img_size=img_size)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    logging.info(f"Loaded checkpoint: {checkpoint_path}")
    logging.info(f"Model: {model_name}, img_size={img_size}, num_classes={num_classes}")
    for key in ['val_acc','val_balanced_acc','val_f1','val_auc']:
        if isinstance(ckpt, dict) and key in ckpt:
            logging.info(f"Checkpoint {key}: {ckpt[key]:.4f}")
    return model

# -------------------------
# Prefilter helpers
# -------------------------
def build_present_index(img_folder):
    """Return dicts for quick lookups: present_files (name->True), stem2name (stem->one filename)."""
    present_files = {}
    stem2name = {}
    try:
        with os.scandir(img_folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                stem, _ = os.path.splitext(name)
                present_files[name] = True
                # prefer first seen; overwrite not needed in flat folder
                stem2name.setdefault(stem, name)
    except FileNotFoundError:
        pass
    return present_files, stem2name

def normalize_csv_value(v):
    """Normalize CSV entry to (basename, stem), stripping paths and trailing '.0' from stem."""
    s = str(v).strip()
    name = os.path.basename(s)
    stem, ext = os.path.splitext(name)
    if stem.endswith(".0"):  # CSV float artifact like '49281.0'
        stem = stem[:-2]
        name = stem + ext
    return name, stem

def is_1024x1024(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
        return (w == 1024 and h == 1024)
    except (UnidentifiedImageError, OSError):
        return False

def prepare_annotations(data_dir, img_dir, annotations_file, out_dir, enforce_1024=True, log_examples=12, progress_every=20000):
    """
    Create a filtered CSV whose first column is the ACTUAL filename present in img_dir.
    Matching is by exact filename OR by stem (ID).
    Optionally keep only 1024x1024 images (header check).
    """
    os.makedirs(out_dir, exist_ok=True)
    src_csv = annotations_file if os.path.isabs(annotations_file) else os.path.join(data_dir, annotations_file)
    df = pd.read_csv(src_csv)
    if df.shape[1] < 1:
        raise ValueError("Annotations CSV must have at least one column (ID or filename).")
    fname_col = df.columns[0]

    img_folder = os.path.join(data_dir, img_dir)
    present_files, stem2name = build_present_index(img_folder)
    if not present_files:
        logging.warning(f"No files found in image folder: {img_folder}")

    keep_rows = []
    out_names = []
    misses = []

    n = len(df)
    t0 = time.time()
    for i, v in enumerate(df[fname_col].values):
        name, stem = normalize_csv_value(v)
        chosen_name = None

        if name in present_files:
            chosen_name = name
        elif stem in stem2name:
            chosen_name = stem2name[stem]

        if chosen_name is not None:
            abs_path = os.path.join(img_folder, chosen_name)
            if (not enforce_1024) or is_1024x1024(abs_path):
                keep_rows.append(True)
                out_names.append(chosen_name)
            else:
                keep_rows.append(False)
                out_names.append(None)
        else:
            keep_rows.append(False)
            out_names.append(None)
            if len(misses) < log_examples:
                misses.append((v, name, stem))

        if progress_every and (i+1) % progress_every == 0:
            dt = time.time() - t0
            logging.info(f"[prefilter] checked {i+1:,}/{n:,} rows (~{(i+1)/max(dt,1):.1f} rows/s)")

    keep = np.array(keep_rows, dtype=bool)
    kept, dropped = int(keep.sum()), int((~keep).sum())
    logging.info(f"Prefilter: keeping {kept:,} / {n:,}; dropping {dropped:,} "
                 f"({'missing/nomatch' + ('/not 1024x1024' if enforce_1024 else '')}).")

    if dropped > 0 and misses:
        logging.info("Examples of unmatched entries (raw -> basename -> stem):")
        for raw, name, stem in misses:
            logging.info(f"  {raw!r} -> {name!r} -> {stem!r}")

    # Build filtered DF with first column replaced by the resolved filename
    kept_df = df.loc[keep].copy()
    kept_df.iloc[:, 0] = np.array([n for n in out_names if n is not None])

    suffix = "_PRESENT_BYNAME_1024.csv" if enforce_1024 else "_PRESENT_BYNAME.csv"
    filtered_csv = os.path.join(out_dir, os.path.splitext(os.path.basename(src_csv))[0] + suffix)
    kept_df.to_csv(filtered_csv, index=False)
    return filtered_csv, kept, dropped

# -------------------------
# Dataset (no resize)
# -------------------------
def load_test_dataset(data_dir, annotations_file, img_dir):
    """
    No resizing: we only include wanted images via prefilter; here we just ToTensor().
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),  # NO Resize
    ])
    ds = JoslinData(
        data_dir=data_dir,
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=tfm
    )
    logging.info(f"Loaded {len(ds)} test samples from {os.path.basename(annotations_file)} (img_dir={img_dir})")
    return ds

# -------------------------
# Evaluate
# -------------------------
def evaluate(model, loader, device, normalization_transform=None):
    model.eval()
    results = []

    total = len(loader.dataset)
    processed = 0
    start = time.time(); last = start; interval = 2.0

    logging.info(f"Starting test evaluation on {total} samples ({len(loader)} batches)...")

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch["image"]; targets = batch.get("label", None)
            else:
                raise ValueError("Unexpected batch format from DataLoader.")

            bs = inputs.size(0)
            inputs = inputs.to(device, non_blocking=True)
            if normalization_transform is not None:
                inputs = normalization_transform(inputs)

            outputs = model(inputs)                 # logits [B, 2]
            probs   = F.softmax(outputs, dim=1)     # [B, 2]
            preds   = torch.argmax(probs, dim=1)    # [B]

            base_ds = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset

            for i in range(bs):
                idx = bidx * loader.batch_size + i
                try:
                    img_filename = base_ds.img_labels.iloc[idx, 0]
                except Exception:
                    img_filename = f"sample_{idx}"

                row = {
                    "image_filename": img_filename,
                    "predicted_label": int(preds[i].cpu().item()),
                    "probability_class_0": float(probs[i, 0].cpu().item()),
                    "probability_class_1": float(probs[i, 1].cpu().item()),
                    "logit_0": float(outputs[i, 0].detach().cpu().item()),
                    "logit_1": float(outputs[i, 1].detach().cpu().item()),
                    "logit_margin": float((outputs[i, 1] - outputs[i, 0]).detach().cpu().item())
                }
                if targets is not None:
                    row["true_label"] = int(targets[i].cpu().item())
                    row["prediction_correct"] = int(preds[i].cpu().item() == targets[i].cpu().item())
                results.append(row)

            processed += bs
            now = time.time()
            if (now - last > interval) or (bidx == len(loader) - 1):
                elapsed = now - start
                spd = processed / max(elapsed, 1e-6)
                remaining = (total - processed) / max(spd, 1e-6)
                bar_len = 30
                filled = int(bar_len * processed / max(total, 1))
                bar = '█' * filled + '░' * (bar_len - filled)
                msg = (f"Progress: [{bar}] {processed}/{total} ({processed/total*100:.1f}%) | "
                       f"Batch {bidx+1}/{len(loader)} | Elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                       f"ETA {time.strftime('%H:%M:%S', time.gmtime(remaining))} | {spd:.1f} samp/s")
                print(msg); logging.info(msg); last = now

    logging.info(f"Finished test evaluation on {len(results)} samples.")
    return results

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Run model on a NEW test set and save predictions (+ logits).")
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    p.add_argument('--data_dir',   required=True, help='Root data directory')
    p.add_argument('--annotations_file', required=True, help='CSV with test image list (ID or filename in first column)')
    p.add_argument('--img_dir',    required=True, help='Relative image folder used by the dataset')
    p.add_argument('--output_dir', default='test_results', help='Where to save outputs')
    p.add_argument('--model_name', default='swinv2_large_window12to16_192to256.ms_in22k_ft_in1k')
    p.add_argument('--img_size', type=int, default=1024)  # model config only; NOT used to resize inputs
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--unlabeled', action='store_true', help='Set if your test CSV has no labels (skip metrics)')
    p.add_argument('--no_autofilter', action='store_true', help='Disable prefilter to existing files')
    p.add_argument('--enforce_1024', action='store_true', default=True, help='Keep only 1024x1024 images')
    p.add_argument('--no_normalize', action='store_true', help='Disable ImageNet normalization before inference')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    configure_logging(args.output_dir)

    # device sanity
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available; using CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    logging.info(f"Device: {device}")

    try:
        # 1) Prefilter annotations to files that exist (by exact name or stem/ID), and (optionally) 1024x1024
        ann_for_load = args.annotations_file
        if not args.no_autofilter:
            ann_for_load, kept, dropped = prepare_annotations(
                args.data_dir, args.img_dir, args.annotations_file, args.output_dir,
                enforce_1024=args.enforce_1024
            )
            if kept == 0:
                logging.warning("Prefilter kept 0 rows. Check CSV first column and image folder contents.")
            elif dropped > 0:
                logging.warning(f"Autofiltered annotations saved to: {ann_for_load}")
        else:
            logging.info("Auto-filter disabled; using annotations as-is.")

        # 2) model + data (NO resizing)
        model = load_model(args.checkpoint, args.model_name, num_classes=2, img_size=args.img_size, device=device)
        test_ds = load_test_dataset(args.data_dir, ann_for_load, img_dir=args.img_dir)

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        normalization = None if args.no_normalize else transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 3) run
        logging.info("\nEvaluating on NEW test set (existing files only; no resizing)...")
        results = evaluate(model, test_loader, device, normalization_transform=normalization)

        # 4) dataframe
        df = pd.DataFrame(results)

        ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
        preds_csv = os.path.join(args.output_dir, f'predictions_{ckpt_tag}.csv')

        if 'true_label' not in df.columns:
            df['true_label'] = -1
            df['prediction_correct'] = np.nan

        df.to_csv(preds_csv, index=False)
        logging.info(f"Per-sample predictions saved to {preds_csv}")

        # 5) metrics (if labels exist and not --unlabeled)
        has_labels = (not args.unlabeled) and ('true_label' in df.columns) and df['true_label'].nunique() >= 1
        if has_labels:
            y_true = df['true_label'].values.astype(int)
            y_pred = df['predicted_label'].values.astype(int)
            y_prob = df['probability_class_1'].values

            metrics = confusion_matrix_with_stats(y_true, y_pred)

            try:
                auc = roc_auc_score(y_true, y_prob)
            except Exception as e:
                logging.warning(f"AUC computation failed: {e}")
                auc = float('nan')

            summary = {
                'total_samples': len(df),
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision'],
                'f1_score': metrics['f1_score'],
                'auc_roc': auc,
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'TN': metrics['TN'],
            }
            summary_df = pd.DataFrame([summary])
            summary_csv = os.path.join(args.output_dir, f'summary_{ckpt_tag}.csv')
            summary_df.to_csv(summary_csv, index=False)
            logging.info(f"Summary metrics saved to {summary_csv}")

            # confusion matrix CSV
            cm = pd.DataFrame([[metrics['TN'], metrics['FP']],
                               [metrics['FN'], metrics['TP']]],
                              index=['actual_0','actual_1'], columns=['pred_0','pred_1'])
            cm.to_csv(os.path.join(args.output_dir, f'confusion_matrix_{ckpt_tag}.csv'), index=True)

            # correct / errors CSVs
            err_df = df[df['prediction_correct'] == 0].copy()
            if len(err_df) > 0:
                err_csv = os.path.join(args.output_dir, f'errors_{ckpt_tag}.csv')
                err_df.to_csv(err_csv, index=False)
                logging.info(f"Errors saved to {err_csv} | FP={(err_df['true_label']==0).sum()}, FN={(err_df['true_label']==1).sum()}")

            correct_df = df[df['prediction_correct'] == 1].copy()
            if len(correct_df) > 0:
                corr_csv = os.path.join(args.output_dir, f'correct_{ckpt_tag}.csv')
                correct_df.to_csv(corr_csv, index=False)
                logging.info(f"Correct predictions saved to {corr_csv}")

        else:
            logging.info("No labels provided (or --unlabeled passed). Skipping metrics.")

        logging.info("\nTest inference completed successfully!")

    except Exception as e:
        logging.error(f"Error during test inference: {e}")
        import traceback; logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
