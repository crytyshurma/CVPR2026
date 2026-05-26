import os
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError as e:
    raise ImportError(
        "This script needs scikit-image. Install it with:\n"
        "pip install scikit-image"
    ) from e


def natural_key(path: Path):
    """Sort paths like 00_01.png, 00_02.png, 01_00.png in human order."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def load_gray_float(path: Path) -> np.ndarray:
    """
    Load an image as grayscale float32 in [0, 1].
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    """
    Compute PSNR and SSIM for a pair of grayscale images in [0, 1].
    """
    psnr_val = sk_psnr(gt, pred, data_range=1.0)
    ssim_val = sk_ssim(gt, pred, data_range=1.0)
    return psnr_val, ssim_val


def find_scene_pairs(gt_root: Path, pred_root: Path):
    """
    Returns a list of (scene_name, gt_scene_dir, pred_scene_dir).
    Supports two common layouts:

    1) gt_root contains images directly, pred_root contains images directly
    2) gt_root contains scene folders, pred_root contains matching scene folders
       or matching scene/re_views folders
    """
    gt_root = gt_root.resolve()
    pred_root = pred_root.resolve()

    # Case 1: both roots directly contain images
    gt_images = sorted([p for p in gt_root.glob("*.png")], key=natural_key)
    pred_images = sorted([p for p in pred_root.glob("*.png")], key=natural_key)
    if gt_images and pred_images:
        return [("single_scene", gt_root, pred_root)]

    # Case 2: gt_root contains subfolders
    scene_pairs = []
    for gt_scene_dir in sorted([p for p in gt_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        scene_name = gt_scene_dir.name

        # Pred may be:
        #   pred_root/scene_name/re_views
        # or pred_root/scene_name
        candidate1 = pred_root / scene_name / "re_views"
        candidate2 = pred_root / scene_name

        if candidate1.exists() and candidate1.is_dir():
            pred_scene_dir = candidate1
        elif candidate2.exists() and candidate2.is_dir():
            pred_scene_dir = candidate2
        else:
            continue

        scene_pairs.append((scene_name, gt_scene_dir, pred_scene_dir))

    return scene_pairs


def evaluate_scene(scene_name: str, gt_dir: Path, pred_dir: Path, output_csv: Path = None):
    gt_files = sorted([p for p in gt_dir.glob("*.png")], key=natural_key)
    pred_files = sorted([p for p in pred_dir.glob("*.png")], key=natural_key)

    if len(gt_files) == 0:
        raise FileNotFoundError(f"No PNG files found in ground-truth folder: {gt_dir}")
    if len(pred_files) == 0:
        raise FileNotFoundError(f"No PNG files found in prediction folder: {pred_dir}")

    if len(gt_files) != len(pred_files):
        print(
            f"[Warning] Scene '{scene_name}': ground truth has {len(gt_files)} images, "
            f"predictions have {len(pred_files)} images. Comparing up to the smaller count."
        )

    n = min(len(gt_files), len(pred_files))
    psnr_list = []
    ssim_list = []

    rows = []
    for i in range(n):
        gt = load_gray_float(gt_files[i])
        pred = load_gray_float(pred_files[i])

        if gt.shape != pred.shape:
            raise ValueError(
                f"Image size mismatch in scene '{scene_name}' at index {i}:\n"
                f"GT:   {gt_files[i].name} -> {gt.shape}\n"
                f"PRED: {pred_files[i].name} -> {pred.shape}"
            )

        psnr_val, ssim_val = compute_metrics(gt, pred)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        rows.append({
            "scene": scene_name,
            "index": i,
            "gt_file": gt_files[i].name,
            "pred_file": pred_files[i].name,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    print(f"\nScene: {scene_name}")
    print(f"  Images compared : {n}")
    print(f"  Average PSNR    : {avg_psnr:.4f}")
    print(f"  Average SSIM    : {avg_ssim:.4f}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_csv.exists()

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["scene", "index", "gt_file", "pred_file", "psnr", "ssim"]
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        print(f"  Per-image results appended to: {output_csv}")

    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed LF views using PSNR and SSIM.")
    parser.add_argument(
        "--gt_root",
        type=str,
        default="LF_data/8x8/BasicLFSR/Lego_Knights",
        help="Ground-truth LF folder or root folder containing scene subfolders."
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="Dst/test/from_LF_data/BasicLFSR/Lego_Knights",
        help="Prediction folder or root folder containing scene subfolders."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Dst/test/from_LF_data/evaluation_results.csv",
        help="CSV file to save per-image metrics."
    )

    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    output_csv = Path(args.output_csv)

    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth path not found: {gt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_root}")

    scene_pairs = find_scene_pairs(gt_root, pred_root)

    if not scene_pairs:
        raise RuntimeError(
            "Could not find matching image folders.\n"
            "Check that your gt_root and pred_root point to the correct directories."
        )

    all_psnr = []
    all_ssim = []

    for scene_name, gt_dir, pred_dir in scene_pairs:
        avg_psnr, avg_ssim = evaluate_scene(scene_name, gt_dir, pred_dir, output_csv)
        all_psnr.append(avg_psnr)
        all_ssim.append(avg_ssim)

    print("\n================ Summary ================")
    print(f"Scenes evaluated : {len(scene_pairs)}")
    print(f"Overall PSNR     : {float(np.mean(all_psnr)):.4f}")
    print(f"Overall SSIM     : {float(np.mean(all_ssim)):.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()