# Trains one classifier (DenseNet-121, ResNet-50 or ViT-B/16) on NIH ChestX-ray14 by empirical risk minimisation.
# Run: python3 -m cxr_shortcuts.runners.train --help

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cxr_shortcuts.data.datasets import NIHPneumothoraxDataset
from cxr_shortcuts.data.gate_guard import load_manifest, load_table
from cxr_shortcuts.data.leakage import assert_no_leakage
from cxr_shortcuts.data.manifests import VAL_SUBSET_SEED, assert_no_patient_leakage, attach_val_subset, verify_manifest_file
from cxr_shortcuts.eval.population_log import log_population
from cxr_shortcuts.eval.schema import config_hash, git_sha, require_git_sha
from cxr_shortcuts.methods.erm import erm_train_step_microbatched
from cxr_shortcuts.models.backbones import build_backbone
from cxr_shortcuts.models.checkpoint import load_checkpoint, save_checkpoint

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_pos_weight(manifest: pd.DataFrame, label_columns: Sequence[str], split: str = "train") -> torch.Tensor:
    rows = manifest[manifest["split"] == split]
    weights: list[float] = []
    for col in label_columns:
        n_pos = int(rows[col].sum())
        n_neg = len(rows) - n_pos
        if n_pos == 0:
            raise ValueError(f"label column {col!r} has zero positives in split={split!r}; cannot compute pos_weight")
        weights.append(n_neg / n_pos)
    return torch.tensor(weights[0]) if len(weights) == 1 else torch.tensor(weights)

def _predict_probs_and_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    label_index: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_batches, labels_batches = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.numpy()
            if label_index is not None:
                probs = probs[:, label_index]
                labels_np = labels_np[:, label_index]
            probs_batches.append(probs)
            labels_batches.append(labels_np)
    return np.concatenate(probs_batches), np.concatenate(labels_batches)

def _write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    os.replace(tmp_path, path)

def resolved_run_output_dir(base_output_dir: str, seed: int, image_size: int) -> str:
    return f"{base_output_dir}_seed{seed}_res{image_size}"

def run_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    dataloader: DataLoader,
    epochs: int,
    checkpoint_path: Path,
    device: str = "cpu",
    pos_weight: Optional[torch.Tensor] = None,
    val_dataloader: Optional[DataLoader] = None,
    val_label_index: Optional[int] = None,
    best_checkpoint_path: Optional[Path] = None,
    selection_metrics_path: Optional[Path] = None,
    micro_batch: Optional[int] = None,
    max_batches_per_epoch: Optional[int] = None,
) -> int:
    checkpoint_path = Path(checkpoint_path)
    start_epoch = 0
    if checkpoint_path.exists():
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    best_metric = None
    best_epoch = None
    meta_path = Path(str(best_checkpoint_path) + ".meta.json") if best_checkpoint_path is not None else None
    if meta_path is not None and meta_path.exists():
        prior_best = json.loads(meta_path.read_text())
        best_metric = prior_best["best_metric"]
        best_epoch = prior_best["best_epoch"]

    last_completed = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        steps = 0
        for images, labels in dataloader:
            if max_batches_per_epoch is not None and steps >= max_batches_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)
            erm_train_step_microbatched(model, optimizer, images, labels, pos_weight=pos_weight, micro_batch=micro_batch)
            steps += 1
        if scheduler is not None:
            scheduler.step()
        print(f"epoch {epoch + 1}/{epochs} trained", flush=True)
        print(f"epoch {epoch + 1}/{epochs} epoch_train_seconds {time.time() - t_epoch:.1f} optimizer_steps {steps}", flush=True)
        log_peak_gpu_memory(device, epoch, epochs)

        if val_dataloader is not None:
            t_eval = time.time()
            probs, labels_np = _predict_probs_and_labels(model, val_dataloader, device, label_index=val_label_index)
            metric = float(roc_auc_score(labels_np, probs))
            print(f"epoch {epoch + 1}/{epochs} val_select_auc {metric:.4f}", flush=True)
            print(f"epoch {epoch + 1}/{epochs} epoch_eval_seconds {time.time() - t_eval:.1f}", flush=True)
            if selection_metrics_path is not None:
                selection_metrics_path = Path(selection_metrics_path)
                selection_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(selection_metrics_path, "a") as f:
                    f.write(json.dumps({"epoch": epoch, "val_select_auc": metric}) + "\n")
            if best_metric is None or metric > best_metric:
                best_metric, best_epoch = metric, epoch
                if best_checkpoint_path is not None:
                    best_checkpoint_path = Path(best_checkpoint_path)
                    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), best_checkpoint_path)
                    _write_json_atomic(meta_path, {"best_epoch": best_epoch, "best_metric": best_metric})

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch)
        last_completed = epoch

    return last_completed

def archive_final_checkpoint(output_dir: Path, archive_dir: Path) -> Path:
    output_dir = Path(output_dir)
    destination = Path(archive_dir) / output_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "checkpoint.pt", destination / "checkpoint.pt")
    shutil.copy2(output_dir / "run_metadata.json", destination / "run_metadata.json")
    return destination

def write_run_metadata(output_dir: Path, config: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sha = require_git_sha()
    metadata = {
        "config": config,
        "config_hash": config_hash(config),
        "git_sha": sha,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

def read_manifest_through_guard(path, gate_manifest_path=None, **table_kwargs) -> pd.DataFrame:
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    if "gate_subset" in header:
        return load_manifest(path, **table_kwargs)
    kw = {} if gate_manifest_path is None else {"manifest_path": Path(gate_manifest_path)}
    return load_table(path, id_col="image_id", id_kind="image", **kw, **table_kwargs)

def log_peak_gpu_memory(device: str, epoch: int, epochs: int) -> Optional[int]:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    mib = int(torch.cuda.max_memory_reserved(device) // (1024 * 1024))
    print(f"epoch {epoch + 1}/{epochs} peak_gpu_reserved_mib {mib}", flush=True)
    return mib

def run_from_config(config: dict) -> int:
    set_all_seeds(config["seed"])
    output_dir = Path(config["output_dir"])
    write_run_metadata(output_dir, config)
    print(f"run start: output_dir={output_dir} seed={config['seed']} git_sha={git_sha()}", flush=True)

    device = config.get("device", "cpu")

    label_columns = list(config["train"].get("label_columns", ["pneumothorax"]))
    model = build_backbone(config["backbone"], num_classes=len(label_columns), pretrained=config.get("pretrained", True))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"].get("lr_step_size", 5),
        gamma=config["train"].get("lr_gamma", 0.5),
    )

    manifest_path = Path(config["data"]["manifest_path"])
    verify_manifest_file(manifest_path)
    manifest = read_manifest_through_guard(manifest_path, config["data"].get("gate_manifest_path"))
    assert_no_patient_leakage(manifest)

    log_population(output_dir, "train_patients", manifest.loc[manifest["split"] == "train", "patient_id"].unique(), "patient")

    assert_no_leakage(
        patient_manifest=manifest,
        acknowledged_unknown=frozenset(
            {
                "donor_train_eval_donor_id_split",
                "donor_train_eval_patient_split",
                "donor_source_vs_audit_target_patient_overlap",
                "gate_subset_locked",
                "donor_eval_outside_training_split",
                "donor_pilot_flagged",
                "screener_patient_split",
                "screener_audit_target_overlap",
                "discriminator_patient_split",
                "discriminator_donor_disjointness",
                "dfr_eval_disjointness",
                "jtt_train_only",
                "checkpoint_selection_validation_only",
                "operating_point_validation_only",
                "normalization_stats_train_only",
                "siim_nih_patient_mapping",
                "control_source_screener_confirmed_drain_free",
                "dev_slice_disjoint_from_eval_arms",
            }
        ),
    )

    dataset = NIHPneumothoraxDataset(
        manifest,
        image_root=config["data"]["nih_root"],
        split="train",
        image_size=config["data"].get("image_size", 224),
        label_columns=label_columns,

        image_path_column=config["data"].get("image_path_column"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        pin_memory=(device != "cpu"),
        num_workers=config["train"].get("num_workers", 0),
    )

    pos_weight = None
    if config["train"].get("class_weighting", False):
        pos_weight = compute_pos_weight(manifest, label_columns, split="train").to(device)

    val_dataloader = None
    val_label_index = None
    best_checkpoint_path = None
    selection_metrics_path = None
    if config["train"].get("checkpoint_selection", False):
        tagged = attach_val_subset(manifest, seed=config["train"].get("val_subset_seed", VAL_SUBSET_SEED))
        val_select_manifest = tagged[(tagged["split"] == "val") & (tagged["val_subset"] == "select")]
        log_population(output_dir, "val_select_patients", val_select_manifest["patient_id"].unique(), "patient")
        log_population(output_dir, "val_report_patients",
                       tagged.loc[(tagged["split"] == "val") & (tagged["val_subset"] == "report"), "patient_id"].unique(), "patient")
        val_dataset = NIHPneumothoraxDataset(
            val_select_manifest,
            image_root=config["data"]["nih_root"],
            split="val",
            image_size=config["data"].get("image_size", 224),
            label_columns=label_columns,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["train"].get("eval_batch_size", config["train"]["batch_size"]),
            shuffle=False,
            pin_memory=(device != "cpu"),
            num_workers=config["train"].get("num_workers", 0),
        )

        val_label_index = None
        if len(label_columns) > 1:
            lower_columns = [c.lower() for c in label_columns]
            if "pneumothorax" not in lower_columns:
                raise ValueError(
                    f"checkpoint_selection requires a pneumothorax label column among label_columns, "
                    f"got {label_columns!r}"
                )
            val_label_index = lower_columns.index("pneumothorax")
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        selection_metrics_path = output_dir / "val_select_metrics.jsonl"

    last_completed = run_training(
        model,
        optimizer,
        scheduler,
        dataloader,
        epochs=config["train"]["epochs"],
        checkpoint_path=output_dir / "checkpoint.pt",
        device=device,
        pos_weight=pos_weight,
        val_dataloader=val_dataloader,
        val_label_index=val_label_index,
        best_checkpoint_path=best_checkpoint_path,
        selection_metrics_path=selection_metrics_path,
        micro_batch=config["train"].get("micro_batch"),
        max_batches_per_epoch=config["train"].get("max_batches_per_epoch"),
    )

    archive_dir = config.get("final_checkpoint_archive_dir")
    if archive_dir is not None and last_completed == config["train"]["epochs"] - 1:
        archive_final_checkpoint(output_dir, Path(archive_dir))

    return last_completed

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 ERM training entry point")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)

    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument(
        "--device", default="cpu",
        help="'cpu' or 'cuda' -- a pure execution-environment choice, not a pre-registered sweep "
        "axis like --seed/--image-size, so it defaults to 'cpu' rather than being required.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config["seed"] = args.seed
    config["data"]["image_size"] = args.image_size
    config["device"] = args.device
    config["output_dir"] = resolved_run_output_dir(config["output_dir"], args.seed, args.image_size)
    run_from_config(config)

if __name__ == "__main__":
    main()
