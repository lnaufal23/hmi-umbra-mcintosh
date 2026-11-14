"""
train_seg_crops.py
==================
Three-class sunspot segmentation (umbra, penumbra, photosphere) trained on active-region
crops of the opt_F images (Section 4.1). The production model is the U-Net (ResNet-34).

Input   : <seg_crops>/{year}/*.png + <seg_masks_crop>/{year}/*_mask.png
Pad     : native size -> padded to 256x256 with 245 (opt_F off-disk level);
          crops larger than 256 px are first reduced to 256 px
Mask pad: padded with the photosphere class
Architectures: U-Net (ResNet34) or DeepLabV3+ (ResNet50)

Classes:
    0 = umbra      (mask value 0)
    1 = penumbra   (mask value 127)
    2 = photosphere (mask value 255)

Run:
    python train_seg_crops.py --arch unet   --epochs 50 --batch 16
    python train_seg_crops.py --arch deeplab --epochs 50 --batch 12

Inference crop:
    python train_seg_crops.py --infer --arch unet \
        --infer-img /path/to/crop.png \
        --checkpoint <seg_runs>/unet_resnet34/best_model.pt

Inference full disk (sliding window):
    python train_seg_crops.py --infer-fulldisk --arch unet \
        --infer-img /path/to/fulldisk.png \
        --checkpoint <seg_runs>/unet_resnet34/best_model.pt \
        --infer-out pred_fulldisk.png
"""

from __future__ import annotations
import os, sys, json, time, random, argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

from hmi_umbra import paths

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CROPS_ROOT      = paths.get("seg_crops")
MASK_CROP_ROOT  = paths.get("seg_masks_crop")
MASK_ROOT   = paths.get("seg_masks_fulldisk_train")
RUN_DIR     = paths.get("seg_runs")

PAD_SIZE    = 256
PAD_FILL    = 245    # off-disk value opt_F
MASK_FILL   = 255    # photosphere value used to pad the mask

NUM_CLASSES = 3
SEED        = 42
CLASS_NAMES = ["umbra", "penumbra", "photosphere"]

# Class weights: inverse frequency (sunspot pixels are rare)
CLASS_WEIGHTS = torch.tensor([10.0, 5.0, 1.0])

# ─── LOGGING ──────────────────────────────────────────────────────────────────

def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("seg_crops")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
        fh = logging.FileHandler(run_dir / "train.log", mode="w")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    for h in log.handlers:
        h.setFormatter(fmt)
    return log

# ─── PAD UTILS ────────────────────────────────────────────────────────────────

def pad_to_square(arr: np.ndarray, fill: int, target: int = PAD_SIZE) -> np.ndarray:
    """
    Pad an (H,W) array to target x target.
    If a dimension exceeds target, resize first (bilinear for images, nearest for masks).
    """
    h, w = arr.shape
    # Resize when larger than target
    if h > target or w > target:
        scale = target / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        pil   = Image.fromarray(arr.astype(np.uint8))
        resample = Image.BILINEAR if fill == PAD_FILL else Image.NEAREST
        pil   = pil.resize((new_w, new_h), resample)
        arr   = np.array(pil)
        h, w  = arr.shape

    # Pad to target x target (centered)
    out    = np.full((target, target), fill, dtype=np.uint8)
    r_off  = (target - h) // 2
    c_off  = (target - w) // 2
    out[r_off:r_off+h, c_off:c_off+w] = arr
    return out


def decode_mask(arr: np.ndarray) -> np.ndarray:
    """Convert mask PNG (0/127/255) → class index (0/1/2)."""
    out = np.full(arr.shape, 2, dtype=np.uint8)
    out[arr == 0]   = 0
    out[arr == 127] = 1
    return out

# ─── BUILD PAIRS ──────────────────────────────────────────────────────────────

def build_pairs(crops_root: str, mask_root: str) -> list[dict]:
    """
    Scan all crop JSON files and match them with their crop masks.
    Return list of dict: {crop_png, mask_png, x1, y1, x2, y2, stem, year, bbox_id, meta}
    """
    pairs = []
    for year_dir in sorted(Path(crops_root).iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        mask_dir = Path(mask_root) / year

        for json_path in sorted(year_dir.glob("*.json")):
            if json_path.name == "README.json":
                continue
            try:
                meta = json.load(open(json_path))
            except Exception:
                continue

            crop_png  = Path(meta["crop_png_path"])
            stem      = meta["stem"]
            bbox_id   = int(meta["bbox_id"])
            mask_crop_png = Path(MASK_CROP_ROOT) / year / f"{stem}_bb{bbox_id:02d}_mask.png"

            if not crop_png.exists() or not mask_crop_png.exists():
                continue

            pairs.append({
                "crop_png": str(crop_png),
                "mask_png": str(mask_crop_png),
                "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                "stem":    stem,
                "year":    year,
                "bbox_id": meta["bbox_id"],
                "extent_max_deg": meta.get("extent_max_deg", 0.0),
                "mu_cos":         meta.get("mu_cos", 1.0),
            })
    return pairs

# ─── DATASET ──────────────────────────────────────────────────────────────────

class CropSegDataset(Dataset):
    def __init__(self, pairs: list[dict], augment: bool = True):
        self.pairs   = pairs
        self.augment = augment

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05,
                     rotate=(-15, 15), p=0.4),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.3),
            ToTensorV2(),
        ]) if augment else A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]

        # Load the crop PNG and its pre-cut mask
        crop_arr = np.array(Image.open(p["crop_png"]).convert("L"))
        mask_arr = np.array(Image.open(p["mask_png"]).convert("L"))

        # Clamp the size in case of a 1-px rounding difference
        h, w = crop_arr.shape
        mask_arr = mask_arr[:h, :w]

        # Decode mask → class index
        mask_cls = decode_mask(mask_arr)

        # Pad to 256x256
        img_pad  = pad_to_square(crop_arr, fill=PAD_FILL,  target=PAD_SIZE)
        mask_pad = pad_to_square(mask_cls, fill=2,          target=PAD_SIZE)
        # mask_pad is 2 (photosphere) in the padded area

        # Augment
        transformed = self.transform(
            image=img_pad.astype(np.uint8),
            mask=mask_pad.astype(np.uint8),
        )
        img_t  = transformed["image"].float() / 255.0   # (1, H, W)
        mask_t = transformed["mask"].long()              # (H, W)

        # 3-channel (imagenet pretrain expects RGB)
        img_t = img_t.repeat(3, 1, 1)

        return {
            "image":   img_t,
            "mask":    mask_t,
            "stem":    p["stem"],
            "bbox_id": p["bbox_id"],
        }

# ─── MODEL ────────────────────────────────────────────────────────────────────

def build_model(arch: str) -> nn.Module:
    if arch == "unet":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=NUM_CLASSES,
        )
    return smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights="imagenet",
        in_channels=3, classes=NUM_CLASSES,
    )

# ─── LOSS ─────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, dice_weight: float = 0.5):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(weight=weights)
        self.dice_w = dice_weight
        self.ce_w   = 1.0 - dice_weight

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(NUM_CLASSES):
            p = probs[:, c]
            t = (targets == c).float()
            loss += 1 - (2*(p*t).sum() + 1) / (p.sum() + t.sum() + 1)
        return loss / NUM_CLASSES

    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + \
               self.dice_w * self.dice_loss(logits, targets)

# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    metrics, iou_list, f1_list = {}, [], []
    for c in range(NUM_CLASSES):
        tp = ((preds==c)&(targets==c)).sum().item()
        fp = ((preds==c)&(targets!=c)).sum().item()
        fn = ((preds!=c)&(targets==c)).sum().item()
        iou = tp/(tp+fp+fn+1e-8)
        f1  = 2*tp/(2*tp+fp+fn+1e-8)
        metrics[f"iou_{CLASS_NAMES[c]}"] = round(iou, 4)
        metrics[f"f1_{CLASS_NAMES[c]}"]  = round(f1, 4)
        iou_list.append(iou); f1_list.append(f1)
    metrics["mIoU"] = round(np.mean(iou_list), 4)
    metrics["mF1"]  = round(np.mean(f1_list),  4)
    return metrics

# ─── EPOCH ────────────────────────────────────────────────────────────────────

def compute_metrics_from_conf(conf_mat: torch.Tensor) -> dict:
    metrics, iou_list, f1_list = {}, [], []
    for c in range(NUM_CLASSES):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp
        iou = tp/(tp+fp+fn+1e-8)
        f1  = 2*tp/(2*tp+fp+fn+1e-8)
        metrics[f"iou_{CLASS_NAMES[c]}"] = round(iou, 4)
        metrics[f"f1_{CLASS_NAMES[c]}"]  = round(f1, 4)
        iou_list.append(iou); f1_list.append(f1)
    metrics["mIoU"] = round(np.mean(iou_list), 4)
    metrics["mF1"]  = round(np.mean(f1_list),  4)
    return metrics

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    # Incremental confusion matrix, so that no large tensors are accumulated
    conf_mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "val ", leave=False):
            imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            logits = model(imgs)
            if isinstance(logits, dict): logits = logits["out"]
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(1).cpu().view(-1)
            tgts  = masks.cpu().view(-1)
            for c_true in range(NUM_CLASSES):
                for c_pred in range(NUM_CLASSES):
                    conf_mat[c_true, c_pred] += ((tgts==c_true)&(preds==c_pred)).sum()
    return total_loss/len(loader), compute_metrics_from_conf(conf_mat)

# ─── INFERENCE: SINGLE CROP ───────────────────────────────────────────────────

def infer_crop(model, img_path: str, device) -> np.ndarray:
    """Inference on one crop PNG -> class mask (H_orig, W_orig)."""
    model.eval()
    orig = np.array(Image.open(img_path).convert("L"))
    h_orig, w_orig = orig.shape

    img_pad = pad_to_square(orig, fill=PAD_FILL, target=PAD_SIZE)
    t = torch.from_numpy(img_pad.astype(np.float32)/255.0)\
              .unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)

    with torch.no_grad():
        logits = model(t)
        if isinstance(logits, dict): logits = logits["out"]
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Unpad: take the central region of the original size
    h_pad, w_pad = PAD_SIZE, PAD_SIZE
    scale = PAD_SIZE / max(h_orig, w_orig) if max(h_orig, w_orig) > PAD_SIZE else 1.0
    h_scaled = max(1, int(h_orig * scale))
    w_scaled = max(1, int(w_orig * scale))
    r_off = (h_pad - h_scaled) // 2
    c_off = (w_pad - w_scaled) // 2
    pred_crop = pred[r_off:r_off+h_scaled, c_off:c_off+w_scaled]

    # Resize back to the original size if it was scaled
    if scale < 1.0:
        pred_crop = np.array(
            Image.fromarray(pred_crop).resize((w_orig, h_orig), Image.NEAREST)
        )
    return pred_crop

# ─── INFERENCE: FULL DISK SLIDING WINDOW ──────────────────────────────────────

def infer_full_disk(model, img_path: str, device,
                    patch_size: int = PAD_SIZE, overlap: float = 0.5) -> np.ndarray:
    """Sliding window inference → full disk (4096×4096) class map."""
    model.eval()
    img = np.array(Image.open(img_path).convert("L"))
    H, W   = img.shape
    P      = patch_size
    stride = int(P * (1 - overlap))

    accum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    win   = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32)

    rows = list(range(0, H-P+1, stride)) + [H-P]
    cols = list(range(0, W-P+1, stride)) + [W-P]

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = img[r:r+P, c:c+P].astype(np.float32) / 255.0
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)\
                         .repeat(1,3,1,1).to(device)
                logits = model(t)
                if isinstance(logits, dict): logits = logits["out"]
                probs = F.softmax(logits, 1).squeeze(0).cpu().numpy()
                accum[:, r:r+P, c:c+P] += probs * win
                count[r:r+P, c:c+P]    += win

    return (accum / np.maximum(count, 1e-8)).argmax(0).astype(np.uint8)


def save_pred_png(pred_cls: np.ndarray, out_path: str):
    out = np.full(pred_cls.shape, 255, dtype=np.uint8)
    out[pred_cls == 0] = 0
    out[pred_cls == 1] = 127
    Image.fromarray(out, "L").save(out_path)

# ─── ARGS ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",        default="unet", choices=["unet","deeplab"])
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--workers",     type=int,   default=8)
    p.add_argument("--no-augment",  action="store_true")
    # Inference
    p.add_argument("--infer",           action="store_true", help="Inference 1 crop")
    p.add_argument("--infer-fulldisk",  action="store_true", help="Inference full disk")
    p.add_argument("--infer-img",       type=str, default=None)
    p.add_argument("--checkpoint",      type=str, default=None)
    p.add_argument("--infer-out",       type=str, default="pred_mask.png")
    return p.parse_args()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.arch}_resnet{'34' if args.arch=='unet' else '50'}"
    run_dir  = Path(RUN_DIR) / run_name
    log      = setup_logger(run_dir)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ── Inference crop ──────────────────────────────────────────────────────
    if args.infer:
        if not args.infer_img or not args.checkpoint:
            log.error("--infer requires --infer-img and --checkpoint"); sys.exit(1)
        model = build_model(args.arch).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        pred = infer_crop(model, args.infer_img, device)
        save_pred_png(pred, args.infer_out)
        log.info(f"Saved: {args.infer_out}"); return

    # ── Inference full disk ─────────────────────────────────────────────────
    if args.infer_fulldisk:
        if not args.infer_img or not args.checkpoint:
            log.error("--infer-fulldisk requires --infer-img and --checkpoint"); sys.exit(1)
        model = build_model(args.arch).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        log.info(f"Sliding window inference: {args.infer_img}")
        pred = infer_full_disk(model, args.infer_img, device)
        save_pred_png(pred, args.infer_out)
        log.info(f"Saved: {args.infer_out}"); return

    # ── Training ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Training: {run_name} | {device}")
    log.info(f"Epochs={args.epochs} | Batch={args.batch} | LR={args.lr}")
    log.info(f"Pad={PAD_SIZE}×{PAD_SIZE} | Fill={PAD_FILL}")
    log.info("=" * 60)

    # Build pairs
    log.info("Building pairs...")
    pairs = build_pairs(CROPS_ROOT, MASK_ROOT)
    log.info(f"  Total pairs: {len(pairs)}")
    if not pairs:
        log.error("No pairs found."); sys.exit(1)

    # Split 70/15/15, stratified by year so that the temporal distribution is even
    years = [p["year"] for p in pairs]
    train_pairs, temp = train_test_split(
        pairs, test_size=0.30, random_state=SEED, stratify=years)
    years_temp = [p["year"] for p in temp]
    val_pairs, test_pairs = train_test_split(
        temp, test_size=0.50, random_state=SEED, stratify=years_temp)

    log.info(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # Save the split
    for name, sp in [("train",train_pairs),("val",val_pairs),("test",test_pairs)]:
        pd.DataFrame(sp).to_csv(run_dir/f"{name}_split.csv", index=False)

    # Dataset & loader
    train_ds = CropSegDataset(train_pairs, augment=not args.no_augment)
    val_ds   = CropSegDataset(val_pairs,   augment=False)
    log.info(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=False)

    # Model, loss, optimizer
    model     = build_model(args.arch).to(device)
    criterion = CombinedLoss(CLASS_WEIGHTS.to(device), dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_miou, best_epoch, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion,
                                        optimizer, device, train=True)
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion,
                                        None, device, train=False)
        scheduler.step()

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"mIoU {train_m['mIoU']:.4f}/{val_m['mIoU']:.4f} | "
            f"iou_umb {val_m['iou_umbra']:.4f} | "
            f"iou_pen {val_m['iou_penumbra']:.4f} | "
            f"{time.time()-t0:.0f}s"
        )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        row.update({f"train_{k}": v for k,v in train_m.items()})
        row.update({f"val_{k}":   v for k,v in val_m.items()})
        history.append(row)

        if val_m["mIoU"] > best_miou:
            best_miou, best_epoch = val_m["mIoU"], epoch
            torch.save(model.state_dict(), run_dir/"best_model.pt")
            log.info(f"  * Best saved (mIoU={best_miou:.4f})")

        torch.save(model.state_dict(), run_dir/"last_model.pt")
        pd.DataFrame(history).to_csv(run_dir/"history.csv", index=False)

    log.info(f"DONE. Best epoch {best_epoch}, mIoU {best_miou:.4f}")
    with open(run_dir/"config.json","w") as f:
        json.dump({**vars(args), "best_epoch": best_epoch, "best_miou": best_miou,
                   "n_train": len(train_pairs), "n_val": len(val_pairs),
                   "n_test": len(test_pairs), "pad_size": PAD_SIZE,
                   "pad_fill": PAD_FILL}, f, indent=2)



if __name__ == "__main__":
    main()
