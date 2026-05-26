# gradcam_swinv2_gradable.py
import argparse
from pathlib import Path
import csv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import timm
except Exception as e:
    raise RuntimeError("This script expects 'timm' to be installed. pip install timm") from e


ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# ── Label convention (INVERTED to match metrics notebook) ─
# Raw CSV: 1=Gradable, 0=Ungradable
# After inversion (matching metrics): 1=Ungradable, 0=Gradable
CLASS_NAMES = {0: "Grad", 1: "Ungrad"}


def build_trained_model(model_name, num_classes, ckpt, img_size, device="cuda"):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, img_size=img_size)

    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print("[ckpt] loaded with weights_only=True")
    except Exception as e:
        print("[ckpt] Falling back to weights_only=False. Reason:", type(e).__name__)
        obj = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break
        if state is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state = obj

    if state is None:
        raise RuntimeError(f"Could not find state_dict in checkpoint. Keys: {list(obj.keys())[:12]}")

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 50:
        raise RuntimeError("Too many missing keys — check --model matches training architecture.")

    model.eval().to(device)
    return model


def build_untrained_model(model_name, num_classes, img_size, device="cuda"):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, img_size=img_size)
    model.eval().to(device)
    return model


def get_eval_transform(model, img_size):
    cfg = getattr(model, "default_cfg", {})
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std  = cfg.get("std",  (0.229, 0.224, 0.225))
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm


def normalize_map(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)


def apply_jet_colormap(x_norm):
    """Pure NumPy jet colormap — no matplotlib needed. Input: 2D array in [0,1]. Output: HxWx3 uint8."""
    r = np.clip(1.5 - np.abs(4 * x_norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x_norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x_norm - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def heatmap_to_uint8(heatmap_2d, cmap="jet"):
    return apply_jet_colormap(normalize_map(heatmap_2d))


def overlay_heatmap(rgb_uint8, heatmap_2d, alpha=0.45, cmap="jet"):
    colored = apply_jet_colormap(normalize_map(heatmap_2d)) / 255.0
    overlay = (1 - alpha) * (rgb_uint8 / 255.0) + alpha * colored
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def grad_cam_swinv2_stage4(model, x, target_idx=None):
    model.eval()
    if not hasattr(model, "layers") or len(model.layers) == 0:
        raise ValueError("Model does not expose 'layers'; cannot locate Stage 4.")

    target_layer = model.layers[-1]
    features, grads = {}, {}

    def fwd_hook(module, inp, out): features["value"] = out
    def bwd_hook(module, grad_in, grad_out): grads["value"] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    if logits.ndim == 1:
        target_idx = target_idx or 0
        score = logits.sum()
    elif logits.ndim == 2 and logits.shape[1] == 1:
        target_idx = target_idx or 0
        score = logits[:, 0].sum()
    else:
        if target_idx is None:
            target_idx = int(logits.argmax(dim=1).item())
        score = logits[:, target_idx].sum()

    model.zero_grad(set_to_none=True)
    score.backward()
    h_fwd.remove()
    h_bwd.remove()

    fmap = features.get("value")
    grad = grads.get("value")
    if fmap is None or grad is None:
        raise RuntimeError("Grad-CAM hooks did not capture features/gradients.")

    if fmap.dim() == 3:
        B, L, C = fmap.shape
        weights = grad.mean(dim=1)[0].view(1, 1, C)
        cam_1d = F.relu((fmap * weights).sum(dim=2))
        H = W = int(L ** 0.5)
        if H * W != L:
            H, W = x.shape[2] // 32, x.shape[3] // 32
        cam = cam_1d.view(1, 1, H, W)
    elif fmap.dim() == 4:
        if fmap.shape[1] == fmap.shape[2] and fmap.shape[3] > fmap.shape[1]:
            fmap, grad = fmap.permute(0,3,1,2).contiguous(), grad.permute(0,3,1,2).contiguous()
        weights = grad.mean(dim=(2,3), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
    else:
        raise ValueError(f"Unsupported feature map shape: {fmap.shape}")

    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = normalize_map(cam[0, 0])
    return cam.detach().cpu().numpy(), target_idx


def read_csv_ids_labels(csv_path):
    candidates = {"gradable_binary_dr", "label", "gradable", "y"}
    rows = []
    with open(csv_path, newline="") as f:
        sn = csv.DictReader(f)
        keys = {k.lower(): k for k in sn.fieldnames}
        if "id" not in keys:
            raise ValueError("CSV must have an 'ID' column.")
        label_key = None
        for c in candidates:
            if c in keys:
                label_key = keys[c]
                break
        if label_key is None:
            raise ValueError("CSV must have a 'gradable_binary_DR' (or similar) column.")
        print(f"[csv] using label column: '{label_key}'")
        id_key = keys["id"]
        for r in sn:
            rows.append((r[id_key].strip(), r[label_key].strip()))
    return rows


def find_image_by_id(root_dir, img_id):
    for ext in ALLOWED_EXTS:
        for cand in [root_dir / f"{img_id}{ext}", root_dir / f"{img_id}{ext.upper()}"]:
            if cand.exists():
                return cand
    hits = [h for h in root_dir.rglob(f"{img_id}*") if h.is_file() and h.suffix.lower() in ALLOWED_EXTS]
    if hits:
        return hits[0]
    hits = [h for h in root_dir.rglob(f"*{img_id}*") if h.is_file() and h.suffix.lower() in ALLOWED_EXTS]
    return hits[0] if hits else None


def map_label_to_index(label_str):
    """
    Applies same inversion as metrics notebook (1-y_true).
    Raw CSV: 1=Gradable, 0=Ungradable
    After inversion: 1=Ungradable, 0=Gradable
    """
    s = str(label_str).strip().lower()
    if s in {"1", "true", "yes", "gradable"}:
        # Raw label 1 (Gradable) → inverted to 0
        return 0, "Grad"
    if s in {"0", "false", "no", "ungradable"}:
        # Raw label 0 (Ungradable) → inverted to 1
        return 1, "Ungrad"
    print(f"[warn] Unknown label '{label_str}' — defaulting to Gradable.")
    return 0, "Grad"


@torch.no_grad()
def predict(model, x):
    logits = model(x)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim == 2 and logits.shape[1] == 1:
        prob1 = torch.sigmoid(logits[:, 0])
        probs = torch.stack([1 - prob1, prob1], dim=1)
    else:
        probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.item(), conf.item(), probs[0].cpu().numpy()


def add_legend(rgb_uint8, text, pad=10):
    img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x, y = pad, pad
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            draw.text((x+dx, y+dy), text, fill=(0,0,0), font=font)
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return np.array(img)


def save_uint8_rgb(rgb_uint8, out_path, legend_text=None):
    if legend_text:
        rgb_uint8 = add_legend(rgb_uint8, legend_text)
    Image.fromarray(rgb_uint8).save(out_path)
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM for Gradable/Ungradable SwinV2 classifier")
    ap.add_argument("--images-dir",     required=True,  help="Directory containing images")
    ap.add_argument("--csv-test",       required=True,  help="CSV with ID and gradable_binary_DR columns")
    ap.add_argument("--model",          default="swinv2_large_window12to16_192to256.ms_in22k_ft_in1k")
    ap.add_argument("--ckpt",           required=True,  help="Path to trained checkpoint")
    ap.add_argument("--img-size",       type=int, default=1024)
    ap.add_argument("--outdir",         default="outputs_gradable_saliency")
    ap.add_argument("--target-source",  default="trained", choices=["trained", "untrained", "label"])
    ap.add_argument("--alpha",          type=float, default=0.45)
    ap.add_argument("--cmap",           default="jet")
    ap.add_argument("--mode",           choices=["all", "selected"], default="all")
    ap.add_argument("--ids",            default="", help="Space/comma-separated image IDs")
    ap.add_argument("--ids-file",       default="", help="Text file with one ID per line")
    ap.add_argument("--raw-images-dir", default="", help="Optional directory with raw (unprocessed) images to also copy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained   = build_trained_model(args.model, num_classes=2, ckpt=args.ckpt,
                                    img_size=args.img_size, device=device)
    untrained = build_untrained_model(args.model, num_classes=2,
                                      img_size=args.img_size, device=device)

    tfm             = get_eval_transform(trained, args.img_size)
    outdir          = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    images_root     = Path(args.images_dir)
    raw_images_root = Path(args.raw_images_dir) if args.raw_images_dir else None

    test_rows   = read_csv_ids_labels(args.csv_test)
    label_by_id = {img_id: lab for img_id, lab in test_rows}

    if args.mode == "all":
        target_ids = [img_id for img_id, _ in test_rows]
    else:
        target_ids = []
        if args.ids:
            target_ids += [t.strip() for t in args.ids.replace(",", " ").split() if t.strip()]
        if args.ids_file:
            with open(args.ids_file) as f:
                target_ids += [l.strip() for l in f if l.strip()]
        if not target_ids:
            raise ValueError("--mode selected requires --ids or --ids-file.")

    missing_files, missing_labels = [], []
    print(f"[info] processing {len(target_ids)} image(s), mode='{args.mode}'")

    for img_id in target_ids:
        p = find_image_by_id(images_root, img_id)
        if p is None:
            missing_files.append(img_id)
            continue

        true_label_str = label_by_id.get(img_id)
        if true_label_str is None:
            missing_labels.append(img_id)
        true_idx, true_name = map_label_to_index(true_label_str or "0")

        pil        = Image.open(p).convert("RGB")
        orig_uint8 = np.array(pil)
        x          = tfm(pil).unsqueeze(0).to(device)

        # ── Predictions ───────────────────────────────────────
        pred_idx_t, pred_conf_t, probs_t = predict(trained,   x)
        pred_idx_u, pred_conf_u, probs_u = predict(untrained, x)

        pred_name_t = CLASS_NAMES[1 - pred_idx_t]
        pred_name_u = CLASS_NAMES[1 - pred_idx_u]

        # ── Print before inversion (raw model output) ─────────
        raw_true  = int(true_label_str) if true_label_str is not None else "N/A"
        raw_name  = "Gradable" if raw_true == 1 else "Ungradable"
        print(f"\n[ID {img_id}] ── BEFORE INVERSION (raw CSV / model output space) ──")
        print(f"  True label      : {raw_true} ({raw_name})")
        print(f"  Trained pred    : {pred_idx_t} ({'Gradable' if pred_idx_t == 1 else 'Ungradable'})  conf={pred_conf_t:.3f}")
        print(f"  Untrained pred  : {pred_idx_u} ({'Gradable' if pred_idx_u == 1 else 'Ungradable'})  conf={pred_conf_u:.3f}")

        # ── Print after inversion (metrics notebook space) ────
        inv_true      = 1 - raw_true if isinstance(raw_true, int) else "N/A"
        inv_pred_t    = 1 - pred_idx_t
        inv_pred_u    = 1 - pred_idx_u
        inv_name      = "Ungradable" if inv_true == 1 else "Gradable"
        print(f"[ID {img_id}] ── AFTER INVERSION (metrics notebook space) ──────────")
        print(f"  True label      : {inv_true} ({inv_name})")
        print(f"  Trained pred    : {inv_pred_t} ({'Ungradable' if inv_pred_t == 1 else 'Gradable'})  conf={pred_conf_t:.3f}")
        print(f"  Untrained pred  : {inv_pred_u} ({'Ungradable' if inv_pred_u == 1 else 'Gradable'})  conf={pred_conf_u:.3f}")
        print(f"  p0={probs_t[0]:.3f} p1={probs_t[1]:.3f} [trained] | p0={probs_u[0]:.3f} p1={probs_u[1]:.3f} [untrained]")

        if args.target_source == "trained":
            target_idx = pred_idx_t
        elif args.target_source == "untrained":
            target_idx = pred_idx_u
        else:
            target_idx = true_idx

        # ── Per-image output folder ───────────────────────────
        # Format: {ID}_true-{true}_train-{pred_trained}_untrain-{pred_untrained}
        folder_name = f"{img_id}_true-{true_name}_train-{pred_name_t}-p{pred_conf_t:.2f}_untrain-{pred_name_u}-p{pred_conf_u:.2f}"
        img_outdir  = outdir / folder_name
        img_outdir.mkdir(parents=True, exist_ok=True)

        # ── Grad-CAM ──────────────────────────────────────────
        cam_t, _ = grad_cam_swinv2_stage4(trained,   x, target_idx=target_idx)
        cam_u, _ = grad_cam_swinv2_stage4(untrained, x, target_idx=target_idx)

        def upsample(cam):
            t = torch.from_numpy(cam)[None, None].float()
            return F.interpolate(t, size=orig_uint8.shape[:2],
                                 mode="bilinear", align_corners=False)[0, 0].numpy()

        cam_t_up  = upsample(cam_t)
        cam_u_up  = upsample(cam_u)
        overlay_t = overlay_heatmap(orig_uint8, cam_t_up, alpha=args.alpha, cmap=args.cmap)
        overlay_u = overlay_heatmap(orig_uint8, cam_u_up, alpha=args.alpha, cmap=args.cmap)

        legend_t = f"pred={pred_name_t} ({pred_conf_t:.2f}), true={true_name}"
        legend_u = f"pred={pred_name_u} ({pred_conf_u:.2f}), true={true_name}"

        # ── Save Grad-CAM outputs ─────────────────────────────
        save_uint8_rgb(heatmap_to_uint8(cam_t_up),  img_outdir / "gradcam_trained.png",    legend_t)
        save_uint8_rgb(heatmap_to_uint8(cam_u_up),  img_outdir / "gradcam_untrained.png",  legend_u)
        save_uint8_rgb(overlay_t,                   img_outdir / "overlay_trained.png",    legend_t)
        save_uint8_rgb(overlay_u,                   img_outdir / "overlay_untrained.png",  legend_u)

        # ── Save original (preprocessed) image ───────────────
        Image.fromarray(orig_uint8).save(img_outdir / "original.png")
        print(f"[saved] {img_outdir / 'original.png'}")

        # ── Save raw image if directory provided ──────────────
        if raw_images_root is not None:
            p_raw = find_image_by_id(raw_images_root, img_id)
            if p_raw is not None:
                raw_uint8 = np.array(Image.open(p_raw).convert("RGB"))
                Image.fromarray(raw_uint8).save(img_outdir / "raw.png")
                print(f"[saved] {img_outdir / 'raw.png'}")
            else:
                print(f"[warn] Raw image not found for ID {img_id} in {raw_images_root}")

    if missing_files:
        print(f"[warn] {len(missing_files)} ID(s) not found: {missing_files[:10]}")
    if missing_labels:
        print(f"[warn] {len(missing_labels)} ID(s) missing labels (defaulted to Gradable): {missing_labels[:10]}")


if __name__ == "__main__":
    main()