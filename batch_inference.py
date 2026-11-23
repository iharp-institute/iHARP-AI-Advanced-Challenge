"""Batch Attention U-Net inference and binary segmentation metrics.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from englacial_attention_unet_pytorch import AttentionUNet, predict_tiled


IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def metrics_from_counts(tp, fp, fn):
    # No positives in either prediction or target counts as a perfect empty mask.
    if tp + fp + fn == 0:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def list_images(directory):
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("No supported images found in {}".format(path))
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate filename stems in {} (e.g. same name with .jpg and .png)".format(path))
    return paths


def imwrite_checked(path, array):
    if not cv2.imwrite(str(path), array):
        raise IOError("Could not save {}".format(path))


@torch.inference_mode()
def predict_overlap_blend(model, image, device, patch_size=128, stride=64,
                          batch_size=4, blend="hann"):
    """Overlap-add probability maps; smooth boundaries before thresholding.

    Reflect-pad by half a patch so original image edges receive central-window
    predictions too. Use strictly positive weights to avoid uncovered pixels.
    """
    if not (1 <= stride <= patch_size):
        raise ValueError("stride must be in [1, patch_size]")
    h, w = image.shape
    margin = patch_size // 2
    expanded = np.pad(image, ((margin, margin), (margin, margin)), mode="reflect"
                      if min(image.shape) > margin else "edge")
    eh, ew = expanded.shape
    extra_h = max(0, patch_size - eh)
    extra_w = max(0, patch_size - ew)
    if extra_h or extra_w:
        expanded = np.pad(expanded, ((0, extra_h), (0, extra_w)), mode="edge")
    eh, ew = expanded.shape

    def starts(length):
        positions = list(range(0, length - patch_size + 1, stride))
        if positions[-1] != length - patch_size:
            positions.append(length - patch_size)
        return positions

    coords = [(y, x) for y in starts(eh) for x in starts(ew)]
    if blend == "hann":
        axis = np.hanning(patch_size).astype(np.float32)
        axis = np.maximum(axis, 0.05)
        weights = np.outer(axis, axis).astype(np.float32)
    elif blend == "uniform":
        weights = np.ones((patch_size, patch_size), dtype=np.float32)
    else:
        raise ValueError("Unknown blend: {}".format(blend))

    weighted_sum = np.zeros((eh, ew), dtype=np.float32)
    weight_sum = np.zeros((eh, ew), dtype=np.float32)
    model.eval()
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        patches = np.stack([expanded[y:y+patch_size, x:x+patch_size]
                            for y, x in batch_coords]).astype(np.float32)[:, None] / 255.0
        logits = model(torch.from_numpy(patches).to(device))
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
        for (y, x), prob in zip(batch_coords, probs):
            weighted_sum[y:y+patch_size, x:x+patch_size] += prob * weights
            weight_sum[y:y+patch_size, x:x+patch_size] += weights
    if np.any(weight_sum <= 0):
        raise RuntimeError("Some pixels were not covered by inference windows")
    restored = weighted_sum / weight_sum
    return restored[margin:margin+h, margin:margin+w]


def run(args):
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    # The iHARP server can run CUDA kernels but cannot load its cuDNN runtime.
    # This has to be set in the *inference* process, not only during training.
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")  # trusted, locally trained checkpoint only
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a training checkpoint with 'model' state_dict")
    model = AttentionUNet(base=checkpoint["base_filters"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    patch_size = int(checkpoint["patch_size"])
    print("Device: {} | cuDNN enabled: {} | patch size: {}".format(
        device, torch.backends.cudnn.enabled, patch_size))

    images = list_images(args.images)
    ground_truth = None
    if args.masks:
        ground_truth = {p.stem: p for p in list_images(args.masks)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    missing = []

    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError("Unreadable radargram {}".format(path))
        gt = None
        if ground_truth is not None:
            gt_path = ground_truth.get(path.stem)
            if gt_path is None:
                missing.append(path.name)
                if args.require_masks:
                    raise FileNotFoundError("No ground-truth mask matching {}".format(path.name))
            else:
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise IOError("Unreadable ground-truth mask {}".format(gt_path))
                if gt.shape != image.shape:
                    raise ValueError("Dimension mismatch: {} {} vs {} {}. Evaluation requires aligned native-resolution masks.".format(
                        path.name, image.shape, gt_path.name, gt.shape))

        if args.stride == patch_size and args.blend == "uniform":
            # Exact legacy non-overlapping prediction for comparison.
            prob = predict_tiled(model, image, device, patch_size=patch_size,
                                 batch_size=args.batch_size)
        else:
            prob = predict_overlap_blend(model, image, device,
                                         patch_size=patch_size, stride=args.stride,
                                         batch_size=args.batch_size, blend=args.blend)
        prediction = prob >= args.threshold
        mask_out = output / (path.stem + "_pred.png")
        imwrite_checked(mask_out, prediction.astype(np.uint8) * 255)
        if args.save_probability:
            imwrite_checked(output / (path.stem + "_probability.png"),
                            np.rint(np.clip(prob, 0, 1) * 255).astype(np.uint8))

        row = {"filename": path.name, "height": image.shape[0], "width": image.shape[1],
               "prediction": mask_out.name, "has_ground_truth": gt is not None,
               "predicted_foreground_fraction": float(prediction.mean())}
        if gt is not None:
            truth = gt > 0  # supports black/white 0/1 or 0/255 masks
            tp = int(np.count_nonzero(prediction & truth))
            fp = int(np.count_nonzero(prediction & ~truth))
            fn = int(np.count_nonzero(~prediction & truth))
            tn = int(np.count_nonzero(~prediction & ~truth))
            counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
            for key in total:
                total[key] += counts[key]
            row.update(counts)
            row.update(metrics_from_counts(tp, fp, fn))
            row["truth_foreground_fraction"] = float(truth.mean())
            print("[{}/{}] {} | IoU {:.4f} P {:.4f} R {:.4f} F1 {:.4f}".format(
                index, len(images), path.name, row["iou"], row["precision"],
                row["recall"], row["f1"]))
        else:
            print("[{}/{}] {} | prediction saved; no ground truth".format(index, len(images), path.name))
        rows.append(row)

    fieldnames = ["filename", "height", "width", "prediction", "has_ground_truth",
                  "predicted_foreground_fraction", "truth_foreground_fraction",
                  "tp", "fp", "fn", "tn", "iou", "precision", "recall", "f1"]
    with (output / "per_image_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    evaluated = [r for r in rows if r["has_ground_truth"]]
    summary = {"checkpoint": str(args.checkpoint), "threshold": args.threshold,
               "stride": args.stride, "blend": args.blend,
               "num_images": len(images), "num_evaluated": len(evaluated),
               "num_missing_masks": len(missing), "missing_masks": missing,
               "empty_mask_convention": "both empty => IoU/precision/recall/F1 = 1; otherwise undefined precision or recall = 0"}
    if evaluated:
        summary["macro_average"] = {
            metric: float(np.mean([r[metric] for r in evaluated]))
            for metric in ("iou", "precision", "recall", "f1")
        }
        summary["micro_average"] = metrics_from_counts(total["tp"], total["fp"], total["fn"])
        summary["aggregate_counts"] = total
    with (output / "summary_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:\n" + json.dumps(summary, indent=2))
    print("Saved predictions, per_image_metrics.csv, summary_metrics.json in", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint fold_N_best.pt")
    parser.add_argument("--images", required=True, help="Input radargram PNG directory")
    parser.add_argument("--masks", help="Ground-truth mask directory; match by filename stem")
    parser.add_argument("--output", default="batch_predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4, help="Number of 128x128 patches per GPU batch")
    parser.add_argument("--stride", type=int, default=64,
                        help="Inference stride in pixels; 64 gives 50%% overlap with 128-pixel patches")
    parser.add_argument("--blend", choices=("hann", "uniform"), default="hann",
                        help="Hann-weighted or uniform probability blending")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:1")
    parser.add_argument("--save-probability", action="store_true", help="Also save grayscale probability PNGs")
    parser.add_argument("--require-masks", action="store_true", help="Fail if any input lacks a corresponding mask")
    parser.add_argument("--enable-cudnn", action="store_false", dest="disable_cudnn",
                        help="Use only if cuDNN is known to work on your server")
    parser.set_defaults(disable_cudnn=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
