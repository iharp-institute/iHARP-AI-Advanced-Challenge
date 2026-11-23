#!/usr/bin/env python3
"""Baseline submission: Attention U-Net inference for englacial layer segmentation.

Called by ingestion:
  python model.py --weights_dir <dir> --images <dir> --output <dir>
Writes <stem>_pred.png (binary 0/255, same HxW as input) per image.
Loads the single non-.py file in --weights_dir: a torch.save dict with
keys model, base_filters, patch_size (as produced by train.py).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

torch.backends.cudnn.enabled = False
EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, intermediate_ch):
        super().__init__()
        self.skip = nn.Conv2d(skip_ch, intermediate_ch, 1)
        self.gate = nn.Conv2d(gate_ch, intermediate_ch, 1)
        self.psi = nn.Conv2d(intermediate_ch, 1, 1)

    def forward(self, skip, gating):
        return skip * torch.sigmoid(self.psi(F.relu(self.skip(skip) + self.gate(gating))))


class AttentionUNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([ConvBlock(1 if i == 0 else f[i-1], f[i]) for i in range(4)])
        self.bottleneck = ConvBlock(f[3], f[3] * 2)
        self.att = nn.ModuleList([AttentionGate(f[i], f[i+1] if i < 3 else f[3] * 2, max(f[i]//2, 1)) for i in range(3, -1, -1)])
        self.dec = nn.ModuleList([ConvBlock(f[i] + (f[i+1] if i < 3 else f[3]*2), f[i]) for i in range(3, -1, -1)])
        self.out = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for i, (att, dec) in enumerate(zip(self.att, self.dec)):
            skip = skips[-i-1]
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat((att(skip, x), x), dim=1))
        return self.out(x)


@torch.no_grad()  # inference_mode needs torch>=1.9; no_grad works on all versions
def predict(model, image, device, patch=128, stride=64, batch=4):
    """Hann-weighted overlap-add probability map, same HxW as image."""
    h, w = image.shape
    m = patch // 2
    x = np.pad(image, ((m, m), (m, m)), mode="reflect" if min(h, w) > m else "edge")
    x = np.pad(x, ((0, max(0, patch - x.shape[0])), (0, max(0, patch - x.shape[1]))), mode="edge")
    eh, ew = x.shape

    def starts(n):
        s = list(range(0, n - patch + 1, stride))
        return s if s[-1] == n - patch else s + [n - patch]

    coords = [(y, c) for y in starts(eh) for c in starts(ew)]
    ax = np.maximum(np.hanning(patch).astype(np.float32), 0.05)
    wt = np.outer(ax, ax).astype(np.float32)
    acc = np.zeros((eh, ew), np.float32)
    den = np.zeros((eh, ew), np.float32)
    model.eval()
    for i in range(0, len(coords), batch):
        bc = coords[i:i+batch]
        p = np.stack([x[y:y+patch, c:c+patch] for y, c in bc]).astype(np.float32)[:, None] / 255.0
        probs = torch.sigmoid(model(torch.from_numpy(p).to(device)))[:, 0].cpu().numpy()
        for (y, c), pr in zip(bc, probs):
            acc[y:y+patch, c:c+patch] += pr * wt
            den[y:y+patch, c:c+patch] += wt
    return (acc / den)[m:m+h, m:m+w]


def find_weights(d: Path):
    files = [p for p in d.iterdir()
             if p.is_file() and p.suffix.lower() != ".py" and not p.name.startswith((".", "__"))]
    if not files:
        raise FileNotFoundError(f"No model file found in {d}")
    return max(files, key=lambda p: p.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    wpath = find_weights(Path(a.weights_dir))
    print(f"torch {torch.__version__}", flush=True)
    try:
        ckpt = torch.load(wpath, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(wpath, map_location="cpu")
    model = AttentionUNet(ckpt["base_filters"]).to(device)
    model.load_state_dict(ckpt["model"])
    patch = int(ckpt["patch_size"])
    print(f"Loaded {wpath.name} | device {device} | patch {patch}", flush=True)

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in EXTS)
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = np.asarray(Image.open(p).convert("L"))
        prob = predict(model, img, device, patch, min(a.stride, patch), a.batch_size)
        Image.fromarray((prob >= a.threshold).astype(np.uint8) * 255).save(out / f"{p.stem}_pred.png")
        el = time.time() - t0
        print(f"[{i}/{len(paths)}] {p.name} {img.shape[1]}x{img.shape[0]} | "
              f"{el:.0f}s elapsed, ~{el / i * (len(paths) - i):.0f}s left", flush=True)


if __name__ == "__main__":
    main()
