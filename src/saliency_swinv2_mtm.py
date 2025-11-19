# saliency_swinv2_mtm.py
import argparse
import os
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Optional: Integrated Gradients (pip install captum)
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except Exception:
    HAS_CAPTUM = False

# timm for SwinV2
try:
    import timm
except Exception as e:
    raise RuntimeError("This script expects 'timm' to be installed. pip install timm") from e


ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


# def build_model(model_name: str, num_classes: int, ckpt: str = None, device="cuda"):
#     model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
#     if ckpt:
#         ckpt_path = Path(ckpt)
#         if ckpt_path.is_file():
#             try:
#                 obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # PyTorch >= 2.4
#             except TypeError:
#                 obj = torch.load(ckpt_path, map_location="cpu")  # older versions: no weights_only arg
#             state = obj.get("state_dict", obj.get("model", obj))
#             clean = {k.replace("module.", ""): v for k, v in state.items()}
#             missing, unexpected = model.load_state_dict(clean, strict=False)
#             print(f"[ckpt] loaded with missing={missing}, unexpected={unexpected}")
#         else:
#             print(f"[warn] checkpoint not found at {ckpt_path}, using pretrained backbone head.")
#     model.eval().to(device)
#     return model


def build_model(model_name: str, num_classes: int, ckpt: str = None, device="cuda"):
    import torch
    import timm
    from pathlib import Path

    # IMPORTANT: match the exact arch you trained
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, img_size=1024)

    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            print(f"[warn] checkpoint not found at {ckpt_path}, using randomly initialized weights.")
        else:
            obj = None
            # 1) try safe load
            try:
                obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                print("[ckpt] loaded with weights_only=True")
            except Exception as e:
                print("[ckpt] Falling back to weights_only=False (trusted checkpoint). Reason:", type(e).__name__)
                # 2) trusted fallback
                obj = torch.load(ckpt_path, map_location="cpu")

            # pick a state_dict inside wrapper dicts
            state = None
            if isinstance(obj, dict):
                for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
                    if k in obj and isinstance(obj[k], dict):
                        state = obj[k]
                        break
                # if top-level looks like a bare state_dict
                if state is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
                    state = obj

            if state is None:
                raise RuntimeError(
                    "Could not find a model state_dict in checkpoint. "
                    f"Top-level keys: {list(obj.keys())[:12] if isinstance(obj, dict) else type(obj)}"
                )

            # strip DistributedDataParallel prefix if present
            clean = {k.replace("module.", ""): v for k, v in state.items()}

            missing, unexpected = model.load_state_dict(clean, strict=False)
            print(f"[ckpt] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:   print("  missing (first 12):", missing[:12])
            if unexpected:print("  unexpected (first 12):", unexpected[:12])

            # if too many missing keys, fail loudly (arch mismatch)
            if len(missing) > 50:
                raise RuntimeError(
                    "Too many missing keys. Ensure --model EXACTLY matches the training architecture and "
                    "num_classes matches the trained head."
                )

    model.eval().to(device)
    return model


def get_eval_transform(model, img_size: int = 1024):
    """
    Evaluation transform for SwinV2:
    - Forces 1024x1024 resolution to match training.
    - Uses bilinear interpolation.
    """
    cfg = getattr(model, "default_cfg", {})
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std  = cfg.get("std",  (0.229, 0.224, 0.225))

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), 
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm, mean, std


def tensor_to_uint8_rgb(t, mean, std):
    t = t.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def normalize_map(x):
    x = x - x.min()
    denom = x.max() + 1e-8
    return x / denom


def overlay_heatmap(rgb_uint8, heatmap_2d, alpha=0.45, cmap="jet"):
    heat = normalize_map(heatmap_2d)
    colored = plt.get_cmap(cmap)(heat)[..., :3]  # HxWx3 in 0..1
    overlay = (1 - alpha) * (rgb_uint8 / 255.0) + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def saliency_vanilla(model, x, target_idx=None):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if target_idx is None:
        target_idx = logits.argmax(dim=1).item()
    loss = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    loss.backward()
    sal = x.grad.detach().abs().max(dim=1)[0]  # [1,H,W] -> [H,W]
    return normalize_map(sal[0].cpu().numpy()), target_idx


def grad_cam_swinv2(model, x, target_idx=None, target_layer=None):
    """
    Grad-CAM for timm SwinV2 models.

    Args:
        model: SwinV2 model (timm)
        x: input tensor [1, 3, H, W] on the same device as model
        target_idx: class index to explain; if None, use argmax
        target_layer: layer to hook; defaults to model.norm for SwinV2

    Returns:
        cam_np: numpy array [H, W] in [0, 1]
        target_idx: int, class index actually used
    """
    model.eval()

    # ----- choose target layer -----
    if target_layer is None:
        if hasattr(model, "norm"):
            target_layer = model.norm
        else:
            raise ValueError("Model has no attribute 'norm'; please pass target_layer explicitly.")

    features = {}
    grads = {}

    def fwd_hook(module, inp, out):
        features["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        grads["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # ----- forward & pick target logit -----
    logits = model(x)
    if logits.ndim == 1:
        if target_idx is None:
            target_idx = 0
        score = logits.sum()
    elif logits.ndim == 2 and logits.shape[1] == 1:
        if target_idx is None:
            target_idx = 0
        score = logits[:, 0].sum()
    else:
        if target_idx is None:
            target_idx = int(logits.argmax(dim=1).item())
        score = logits[:, target_idx].sum()

    # ----- backward -----
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    handle_fwd.remove()
    handle_bwd.remove()

    fmap = features.get("value", None)
    grad = grads.get("value", None)
    if fmap is None or grad is None:
        raise RuntimeError("Grad-CAM hooks did not capture features/gradients. Check target_layer.")

    # ------------------------------------------------------------------
    # CASE A: token representation [B, L, C]  -> 1D CAM of length L
    # ------------------------------------------------------------------
    if fmap.dim() == 3:
        # fmap, grad: [B, L, C]
        B, L, C = fmap.shape
        assert B == 1, "This function assumes batch size 1 for visualization."

        # Grad-CAM weights over channels (average over tokens L)
        #   weights: [C]
        weights = grad.mean(dim=1)[0]              # [C]
        weights = weights.view(1, 1, C)            # [1,1,C]

        # Weighted sum over channels -> [1, L]
        cam_1d = (fmap * weights).sum(dim=2)       # [1, L]
        cam_1d = F.relu(cam_1d)

        # ----- reshape 1D CAM to 2D grid BEFORE interpolation -----
        # infer grid size; for Swin this should be square
        H_feat = W_feat = int(L ** 0.5)
        if H_feat * W_feat != L:
            # fall back to non-square if needed
            H_feat = x.shape[2] // 32  # assuming overall stride 32
            W_feat = x.shape[3] // 32
            if H_feat * W_feat != L:
                raise ValueError(f"Cannot reshape tokens of length {L} to a 2D grid.")

        cam = cam_1d.view(1, 1, H_feat, W_feat)    # [B=1,1,H_feat,W_feat]

    # ------------------------------------------------------------------
    # CASE B: conv-like representation [B, C, H_feat, W_feat]
    # ------------------------------------------------------------------
    elif fmap.dim() == 4:
        # Could be [B, C, H, W] or [B, H, W, C]
        B, a, b, c = fmap.shape
        if a <= b and a <= c:
            # assume [B, C, H, W]
            fmap_c = fmap
            grad_c = grad
        else:
            # assume [B, H, W, C]
            fmap_c = fmap.permute(0, 3, 1, 2).contiguous()
            grad_c = grad.permute(0, 3, 1, 2).contiguous()

        # standard Grad-CAM
        weights = grad_c.mean(dim=(2, 3), keepdim=True)       # [B, C, 1, 1]
        cam = (weights * fmap_c).sum(dim=1, keepdim=True)     # [B, 1, H_feat, W_feat]
        cam = F.relu(cam)

    else:
        raise ValueError(f"Unsupported feature map dimension: {fmap.shape}")

    # ----- upsample to input size & normalize -----
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy(), target_idx


def saliency_integrated_gradients(model, x, target_idx=None, steps=64, baseline="black"):
    """
    Compute saliency map using Integrated Gradients.

    Works for:
      - Binary models that output a single logit: shape [B] or [B, 1]
      - Multi-class models that output logits: shape [B, C] with C > 1
    """
    if not HAS_CAPTUM:
        raise RuntimeError("captum is not installed. pip install captum")

    model.eval()

    # Ensure x requires grad
    x = x.clone().detach()
    x.requires_grad_(True)

    # Forward once to inspect output shape
    with torch.no_grad():
        logits = model(x)

    # Decide if we can / should use a target index
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        # Scalar / single-logit output (binary classifier)
        use_target = False
        # target_idx is ignored in this case; IG is computed w.r.t. the scalar output
        effective_target_idx = None
    else:
        # Multi-class output: shape [B, C] with C > 1
        use_target = True
        if target_idx is None:
            # Assume batch size 1 (your script uses single images)
            effective_target_idx = int(logits.argmax(dim=1).item())
        else:
            effective_target_idx = int(target_idx)

    # Baseline choice
    if baseline == "black":
        b = torch.zeros_like(x)
    elif baseline == "random":
        b = torch.rand_like(x) * 0.1
    else:
        # default to black if unknown string
        b = torch.zeros_like(x)

    ig = IntegratedGradients(model)

    if use_target:
        # Multi-class case: select class via target index
        attributions = ig.attribute(
            inputs=x,
            baselines=b,
            target=effective_target_idx,
            n_steps=steps,
        )
    else:
        # Scalar / single-logit case: do NOT pass target
        attributions = ig.attribute(
            inputs=x,
            baselines=b,
            n_steps=steps,
        )

    # Collapse channels -> [H, W] for the first (and only) image in batch
    sal = attributions.abs().sum(dim=1, keepdim=False)[0]
    sal = sal.detach().cpu().numpy()

    # Return normalized map and the target index (if any)
    return normalize_map(sal), effective_target_idx


def read_csv_ids_labels(csv_path):
    """
    Expects columns: ID and one of {label, MTM_binary_DR, MTM_binary, MTM, y} (case-insensitive).
    Returns list of (id_str, label_str).
    """
    candidates = {"label", "mtm_binary_dr", "mtm_binary", "mtm", "y"}
    rows = []
    with open(csv_path, "r", newline="") as f:
        sn = csv.DictReader(f)
        keys = {k.lower(): k for k in sn.fieldnames}
        if "id" not in keys:
            raise ValueError("CSV must have an 'ID' column (case-insensitive).")
        # find the first matching label-like column
        label_key = None
        for c in candidates:
            if c in keys:
                label_key = keys[c]
                break
        if label_key is None:
            raise ValueError("CSV must have one label column among: label, MTM_binary_DR, MTM_binary, MTM, y.")

        id_key = keys["id"]
        for r in sn:
            rows.append((r[id_key].strip(), r[label_key].strip()))
    return rows


def find_image_by_id(root_dir: Path, img_id: str):
    """
    Robust, case-insensitive search. Matches:
      - exact basename (with common extensions, any case)
      - filenames that start with the ID (e.g., 360501_*.*)
      - filenames that contain the ID as a token (underscore/dash or anywhere)
    """
    # 1) direct match for common extensions (lower + upper)
    for ext in ALLOWED_EXTS:
        for cand in [root_dir / f"{img_id}{ext}",
                     root_dir / f"{img_id}{ext.upper()}"]:
            if cand.exists():
                return cand

    # 2) start-with pattern
    hits = [h for h in root_dir.rglob(f"{img_id}*") if h.is_file()]
    if hits:
        allowed_hits = [h for h in hits if h.suffix.lower() in ALLOWED_EXTS]
        return (allowed_hits or hits)[0]

    # 3) anywhere in name, prefer allowed ext
    hits = [h for h in root_dir.rglob(f"*{img_id}*")
            if h.is_file() and h.suffix.lower() in ALLOWED_EXTS]
    if hits:
        return hits[0]

    return None


def map_label_to_index(label_str: str):
    """
    Map label to (index, canonical_name).

    Assumes:
        MTM  -> 1
        NMTM -> 0
    """
    s = str(label_str).strip().lower()

    # exact or common variants
    if s in {"mtm"}:
        return 1, "MTM"
    if s in {"nmtm", "nonmtm", "non-mtm"}:
        return 0, "Non-MTM"

    # fallback for numeric encodings (optional)
    if s in {"1", "true", "yes"}:
        return 1, "MTM"
    if s in {"0", "false", "no"}:
        return 0, "Non-MTM"

    # if unknown, default to Non-MTM but warn
    print(f"[warn] Unknown label '{label_str}' — defaulting to Non-MTM.")
    return 0, "Non-MTM"


@torch.no_grad()
def predict(model, x):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.item(), conf.item(), probs[0].cpu().numpy()


def save_panel(orig_uint8, overlay_van, overlay_other, out_path,
               pred_name, pred_conf, true_name=None, other_method_name=None):
    """
    orig_uint8: original RGB image (H,W,3) uint8
    overlay_van: vanilla gradient overlay (H,W,3) uint8
    overlay_other: overlay for IG or Grad-CAM (or None)
    other_method_name: string for title, e.g. 'Integrated Gradients' or 'Grad-CAM'
    """
    ncols = 3 if overlay_other is not None else 2
    plt.figure(figsize=(12 if ncols == 3 else 8, 6))

    # Original
    plt.subplot(1, ncols, 1)
    plt.imshow(orig_uint8)
    plt.axis("off")
    plt.title("Original")

    # Vanilla Gradient
    plt.subplot(1, ncols, 2)
    plt.imshow(overlay_van)
    plt.axis("off")
    t2 = f"Vanilla Grad\npred={pred_name} ({pred_conf:.2f})"
    if true_name:
        t2 += f"\ntrue={true_name}"
    plt.title(t2)

    # Other method (IG or Grad-CAM)
    if overlay_other is not None:
        plt.subplot(1, ncols, 3)
        plt.imshow(overlay_other)
        plt.axis("off")
        plt.title(other_method_name or "Other method")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="SwinV2 MTM saliency maps from CSV-defined splits")
    ap.add_argument("--images-dir", type=str, required=True, help="Directory containing ALL images")
    ap.add_argument("--csv-test", type=str, required=True, help="CSV for test split (ID,label)")
    ap.add_argument("--model", type=str, default="swinv2_tiny_window16_256",
                    help="timm model (e.g., swinv2_tiny_window16_256)")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to trained checkpoint (optional)")
    ap.add_argument(
        "--method",
        type=str,
        default="ig",
        choices=["ig", "gradcam", "none"],
        help="Secondary saliency method to show along with vanilla gradient "
             "(ig, gradcam, or none). Default: ig."
    )
    ap.add_argument("--outdir", type=str, default="outputs_mtm_saliency", help="Output directory")

    # selection controls
    ap.add_argument("--mode", type=str, choices=["all", "selected"], default="all",
                    help="'all' = all test CSV images; 'selected' = only provided IDs")
    ap.add_argument("--ids", type=str, default="", help="Space/comma-separated image IDs (no extension)")
    ap.add_argument("--ids-file", type=str, default="", help="Text file: one image ID per line")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, num_classes=2, ckpt=args.ckpt, device=device)
    tfm, mean, std = get_eval_transform(model)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_dir) if hasattr(args, "images-dir") else Path(args.images_dir)  # safety


    # Load test CSV (ID,label)
    test_rows = read_csv_ids_labels(args.csv_test)
    label_by_id = {img_id: lab for img_id, lab in test_rows}

    # Determine target IDs
    if args.mode == "all":
        target_ids = [img_id for img_id, _ in test_rows]
    else:
        target_ids = []
        if args.ids:
            for tok in args.ids.replace(",", " ").split():
                if tok.strip():
                    target_ids.append(tok.strip())
        if args.ids_file:
            with open(args.ids_file, "r") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        target_ids.append(s)
        if not target_ids:
            raise ValueError("Selected mode needs --ids or --ids-file.")

    # Process
    missing_files, missing_labels = [], []
    print(f"[info] processing {len(target_ids)} image(s) in mode='{args.mode}'")

    for img_id in target_ids:
        p = find_image_by_id(images_root, img_id)
        if p is None:
            missing_files.append(img_id); continue

        true_label_str = label_by_id.get(img_id, None)
        if true_label_str is None:
            missing_labels.append(img_id)
        true_idx, true_name = map_label_to_index(true_label_str if true_label_str is not None else "Non-MTM")

        pil = Image.open(p).convert("RGB")
        orig_uint8 = np.array(pil)
        x = tfm(pil).unsqueeze(0).to(device)

        # predict
        # pred_idx, pred_conf, _ = predict(model, x)
        pred_idx, pred_conf, probs = predict(model, x)
        print(f"probs = [p0={probs[0]:.3f}, p1={probs[1]:.3f}]  pred_idx={pred_idx}")  
        pred_name = "MTM" if pred_idx == 1 else "Non-MTM"

        # vanilla saliency
        sal_van, _ = saliency_vanilla(model, x, target_idx=pred_idx)
        sal_van_t = torch.from_numpy(sal_van)[None, None].float()
        sal_van_up = F.interpolate(sal_van_t, size=orig_uint8.shape[:2], mode="bilinear", align_corners=False)[0,0].numpy()
        overlay_van = overlay_heatmap(orig_uint8, sal_van_up, alpha=0.45, cmap="jet")

        # secondary method: IG or Grad-CAM (or none)
        overlay_other = None
        other_method_name = None

        if args.method == "ig":
            if not HAS_CAPTUM:
                print("[warn] captum not available; skipping Integrated Gradients.")
            else:
                sal_ig, _ = saliency_integrated_gradients(
                    model, x, target_idx=pred_idx, steps=8, baseline="black"
                )
                sal_ig_t = torch.from_numpy(sal_ig)[None, None].float()
                sal_ig_up = F.interpolate(
                    sal_ig_t,
                    size=orig_uint8.shape[:2],
                    mode="bilinear",
                    align_corners=False
                )[0, 0].numpy()
                overlay_other = overlay_heatmap(orig_uint8, sal_ig_up, alpha=0.45, cmap="jet")
                other_method_name = "Integrated Gradients"

        elif args.method == "gradcam":
            # Option 2: hook patch embedding conv instead of model.norm
            target_layer = None
            if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
                target_layer = model.patch_embed.proj
            else:
                # fallback to old default if the arch is different
                target_layer = None

            sal_cam, _ = grad_cam_swinv2(
                model,
                x,
                target_idx=pred_idx,
                target_layer=target_layer
            )

            sal_cam_t = torch.from_numpy(sal_cam)[None, None].float()
            sal_cam_up = F.interpolate(
                sal_cam_t,
                size=orig_uint8.shape[:2],
                mode="bilinear",
                align_corners=False
            )[0, 0].numpy()
            overlay_other = overlay_heatmap(orig_uint8, sal_cam_up, alpha=0.45, cmap="jet")
            other_method_name = "Grad-CAM"

        else:
            # args.method == "none": no secondary overlay
            overlay_other = None
            other_method_name = None

        
        # save
        out_name = f"saliency_{img_id}_pred-{pred_name}_p{pred_conf:.2f}.png"
        out_path = outdir / out_name
        save_panel(
            orig_uint8,
            overlay_van,
            overlay_other,
            out_path,
            pred_name,
            pred_conf,
            true_name=true_name if true_label_str is not None else None,
            other_method_name=other_method_name,
        )


    if missing_files:
        print(f"[warn] {len(missing_files)} ID(s) not found as files: {missing_files[:10]}{' ...' if len(missing_files)>10 else ''}")
    if missing_labels:
        print(f"[warn] {len(missing_labels)} ID(s) missing labels in CSV (defaulted to Non-MTM): {missing_labels[:10]}{' ...' if len(missing_labels)>10 else ''}")


if __name__ == "__main__":
    main()
