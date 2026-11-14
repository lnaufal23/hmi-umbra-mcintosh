#!/usr/bin/env python3
"""Full-disk U-Net masks in a single pass (whole 4096x4096 opt_F image, no tiling; the U-Net is
fully convolutional), Section 4.1. Preprocessing as in training (/255, replicated to 3 channels).
Off-disk pixels (opt_F >= 245) are set to photosphere.
Output <masks_fulldisk>/{year}/{stem}_pred.png : 0=umbra, 127=penumbra, 255=photosphere.
Existing masks are skipped."""
import glob
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image

from hmi_umbra import paths

CKPT = os.path.join(paths.get("seg_runs"), "unet_resnet34", "best_model.pt")
SRC = paths.get("optf_png")
OUT = paths.get("masks_fulldisk")


def main():
    files = sorted(glob.glob(f"{SRC}/*/hmi.ic_720s.*.png"))
    todo = [f for f in files if not os.path.exists(f"{OUT}/{Path(f).parent.name}/{Path(f).stem}_pred.png")]
    print(f"opt_F PNG {len(files)} | remaining {len(todo)}", flush=True)
    if not todo:
        return
    dev = torch.device("cuda:0")
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    m.load_state_dict(ck); m = m.to(dev).eval()
    rd = lambda f: (f, np.array(Image.open(f).convert("L")))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = deque(ex.submit(rd, f) for f in todo[:3]); nxt = len(futs)
        for i in range(len(todo)):
            f, img = futs.popleft().result()   # the image leaves the queue once taken
            if nxt < len(todo):
                futs.append(ex.submit(rd, todo[nxt])); nxt += 1
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                cls = m(x).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
            out = np.full(cls.shape, 255, np.uint8); out[cls == 0] = 0; out[cls == 1] = 127
            out[img >= 245] = 255
            od = Path(OUT) / Path(f).parent.name; od.mkdir(parents=True, exist_ok=True)
            tmp = od / (Path(f).stem + "_pred.png.tmp")
            Image.fromarray(out, "L").save(tmp, format="PNG", compress_level=3)
            os.replace(tmp, od / (Path(f).stem + "_pred.png"))
            if (i + 1) % 200 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)} | {el / (i + 1):.2f} s/image | ETA {(len(todo) - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
