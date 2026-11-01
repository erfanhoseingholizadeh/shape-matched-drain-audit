# Computes the uniform logit shift that moves the mean predicted probability on the reference films by 0.02 and by 0.10 (the margin and the INVERT
# constant).
# Run: python3 analysis/reachability_constants.py --help

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from torch.utils.data import DataLoader

from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX, NIHPneumothoraxDataset
from cxr_shortcuts.data.manifests import VAL_SUBSET_SEED, attach_val_subset
from cxr_shortcuts.models.backbones import build_backbone

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def mean_delta_p_for_shift(logits: np.ndarray, shift: float) -> float:
    return float(np.mean(sigmoid(logits + shift) - sigmoid(logits)))

def solve_uniform_shift_for_target_mean_dp(logits: np.ndarray, target: float) -> float:
    f = lambda z: mean_delta_p_for_shift(logits, z) - target
    lo, hi = -50.0, 50.0
    assert f(lo) < 0 < f(hi), f"target {target} not bracketed: f(lo)={f(lo)}, f(hi)={f(hi)}"
    return float(brentq(f, lo, hi, xtol=1e-10))

@torch.no_grad()
def predict_pneumothorax(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logit_batches, label_batches = [], []
    for images, labels in dataloader:
        images = images.to(device)
        raw_logits = model(images).cpu().numpy()[:, PNEUMOTHORAX_LABEL_INDEX]
        logit_batches.append(raw_logits)
        label_batches.append(labels.numpy()[:, PNEUMOTHORAX_LABEL_INDEX])
    return np.concatenate(logit_batches), np.concatenate(label_batches)

def reachability_summary(neg_logits: np.ndarray) -> dict:
    neg_probs = sigmoid(neg_logits)
    shift_for_pass = solve_uniform_shift_for_target_mean_dp(neg_logits, 0.10)
    shift_for_fail_floor = solve_uniform_shift_for_target_mean_dp(neg_logits, 0.02)
    return {
        "n_negatives": int(len(neg_logits)),
        "median_baseline_p": float(np.median(neg_probs)),
        "median_baseline_logit": float(np.median(neg_logits)),
        "mean_baseline_p": float(np.mean(neg_probs)),
        "uniform_logit_shift_for_mean_dp_0.10_PASS": shift_for_pass,
        "uniform_logit_shift_for_mean_dp_0.02_FAIL_floor": shift_for_fail_floor,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nih-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append",
                         help="repeatable: one checkpoint per seed, in seed order (0,1,2,3,4)")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    tagged = attach_val_subset(manifest, seed=VAL_SUBSET_SEED)
    rows = tagged[(tagged["split"] == "val") & (tagged["val_subset"] == "report")]
    dataset = NIHPneumothoraxDataset(
        rows, image_root=args.nih_root, split="val", image_size=args.image_size, label_columns=list(NIH_14_LABELS),
    )
    patient_ids_all = rows.reset_index(drop=True)["patient_id"].to_numpy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    csv_lines = []
    if write_header:
        csv_lines.append("seed,patient_id,logit,prob\n")

    per_seed_results: dict = {}
    pooled_neg_logits = []

    for seed_idx, ckpt in enumerate(args.checkpoint):
        model = build_backbone(args.backbone, num_classes=len(NIH_14_LABELS), pretrained=False)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model = model.to(args.device)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        logits, labels = predict_pneumothorax(model, dataloader, args.device)

        neg_mask = labels == 0
        neg_logits = logits[neg_mask]
        neg_pids = patient_ids_all[neg_mask]
        neg_probs = sigmoid(neg_logits)

        for pid, z, p in zip(neg_pids, neg_logits, neg_probs):
            csv_lines.append(f"{seed_idx},{pid},{z},{p}\n")

        summary = reachability_summary(neg_logits)
        per_seed_results[seed_idx] = {"checkpoint": str(ckpt), **summary}
        pooled_neg_logits.append(neg_logits)

        print(f"[seed {seed_idx}] n_neg={summary['n_negatives']} median_p={summary['median_baseline_p']:.4g} "
              f"shift_for_0.02={summary['uniform_logit_shift_for_mean_dp_0.02_FAIL_floor']:.4f} "
              f"shift_for_0.10={summary['uniform_logit_shift_for_mean_dp_0.10_PASS']:.4f}")

    with open(args.output_csv, "a") as f:
        f.writelines(csv_lines)

    pooled = np.concatenate(pooled_neg_logits)
    pooled_summary = reachability_summary(pooled)
    print(f"[pooled, n_seeds={len(args.checkpoint)}] n_neg={pooled_summary['n_negatives']} "
          f"median_p={pooled_summary['median_baseline_p']:.4g} "
          f"shift_for_0.02={pooled_summary['uniform_logit_shift_for_mean_dp_0.02_FAIL_floor']:.4f} "
          f"shift_for_0.10={pooled_summary['uniform_logit_shift_for_mean_dp_0.10_PASS']:.4f}")

    results = {
        "backbone": args.backbone,
        "per_seed": per_seed_results,
        "pooled": pooled_summary,
        "per_seed_median_p_spread": {
            "values": [per_seed_results[s]["median_baseline_p"] for s in per_seed_results],
            "min": min(per_seed_results[s]["median_baseline_p"] for s in per_seed_results),
            "max": max(per_seed_results[s]["median_baseline_p"] for s in per_seed_results),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
