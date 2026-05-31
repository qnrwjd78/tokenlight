from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from tokenlight.data import load_tensor_image
from tokenlight.eval import lpips_distance, psnr, ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL with pred and target paths")
    parser.add_argument("--root", default=".")
    parser.add_argument("--lpips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    records = []
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for record in tqdm(records):
        pred = load_tensor_image(root / record["pred"]).unsqueeze(0).to(device)
        target = load_tensor_image(root / record["target"]).unsqueeze(0).to(device)
        totals["psnr"] += float(psnr(pred, target).cpu())
        totals["ssim"] += float(ssim(pred, target).cpu())
        if args.lpips:
            totals["lpips"] += float(lpips_distance(pred, target).cpu())
    count = max(len(records), 1)
    print({key: value / count for key, value in totals.items() if args.lpips or key != "lpips"})


if __name__ == "__main__":
    main()
