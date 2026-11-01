# Builds and scores the shape-by-content two-by-two on the negative targets (tube or disk, rendered or tissue).
# Run: python3 scoring/twobytwo_audit.py --help

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

ARMS5 = ("before", "insertion", "control", "blob", "disk_tissue")
CELLS = {"tube_rendered": "insertion", "tube_tissue": "control", "disk_rendered": "blob", "disk_tissue": "disk_tissue"}

def recover_disk(blob_placed: np.ndarray, generate_equal_area_blob, place_mask):
    area = int(blob_placed.sum())
    _, local = generate_equal_area_blob(area, 255.0, 0.0)
    rows, cols = np.nonzero(blob_placed)
    lrows, lcols = np.nonzero(local)
    offset = (int(rows.min() - lrows.min()), int(cols.min() - lcols.min()))
    if not np.array_equal(place_mask(local, blob_placed.shape, offset), blob_placed):
        raise AssertionError("regenerated disk mask does not reproduce the persisted blob_mask at the recovered offset")
    return local, offset

def build_disk_tissue(before, blob_placed, control_source, generate_equal_area_blob, place_mask, log_additive_blend_patch):
    local, offset = recover_disk(blob_placed, generate_equal_area_blob, place_mask)
    h, w = local.shape
    patch = control_source[offset[0]: offset[0] + h, offset[1]: offset[1] + w]
    if patch.shape != local.shape:
        raise AssertionError("disk crop falls outside the control-source image")
    out = log_additive_blend_patch(before, patch, local, offset, float(patch.mean()))
    if not np.array_equal(out[~blob_placed], before[~blob_placed]):
        raise AssertionError("the disk-tissue arm changed pixels outside the disk mask")
    return out

def run_score(args) -> None:
    import torch
    from cxr_shortcuts.data.image_io import default_image_loader
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved
    from cxr_shortcuts.edits.drain.geometry import place_mask
    from cxr_shortcuts.edits.drain.insertion import log_additive_blend_patch
    from cxr_shortcuts.eval.population_log import log_population
    from scoring import fresh_composite_audit as p64
    from scoring.blob_control_audit import generate_equal_area_blob

    p64.BACKBONE = args.arch
    anchor = json.loads(args.anchor_json.read_text())
    comp = p64.load_verified_composite(args.composite_dir, anchor["composite_manifest_sha256"])
    man = comp["manifest"]
    ids = man["kept"]
    if args.smoke_n:
        ids = ids[: args.smoke_n]
        comp["arrays"] = {a: v[: args.smoke_n] for a, v in comp["arrays"].items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_ids = [man["meta"][i]["control_source_id"] for i in ids]
    assert_not_reserved(ids, "twobytwo_audit targets")
    assert_not_reserved(src_ids, "twobytwo_audit control sources")
    loader = default_image_loader(args.nih_root)
    disk = []
    for i, image_id in enumerate(ids):
        blob_placed = np.load(args.composite_dir / man["meta"][image_id]["file"])["blob_mask"]
        disk.append(build_disk_tissue(comp["arrays"]["before"][i], blob_placed, loader(src_ids[i]),
                                      generate_equal_area_blob, place_mask, log_additive_blend_patch))
    digest = hashlib.sha256(b"".join(a.tobytes() for a in disk)).hexdigest()
    arrays = {a: comp["arrays"][a] for a in ("before", "insertion", "control", "blob")} | {"disk_tissue": disk}
    entries = json.loads(args.checkpoints.read_text())
    if args.smoke_n:
        entries = entries[:1]
    header = ["label", "image_id", "patient_id"] + [f"logit_{a}" for a in ARMS5]
    csv_path = args.out_dir / f"scores_{args.arch}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for e in entries:
            model = p64.build_model(e["checkpoint"], args.device)
            lg = {a: p64.logits_for(model, arrays[a], args.device) for a in ARMS5}
            p64.log(f"scored {e['label']}: " + " ".join(f"{a}={np.mean(lg[a] - lg['before']):.4f}" for a in ARMS5[1:]))
            for i, image_id in enumerate(ids):
                w.writerow([e["label"], image_id, man["meta"][image_id]["patient_id"]] + [lg[a][i] for a in ARMS5])
            del model
    log_population(args.out_dir, f"twobytwo_{args.arch}_targets", ids, "image")
    log_population(args.out_dir, f"twobytwo_{args.arch}_control_sources", src_ids, "image")
    (args.out_dir / f"summary_{args.arch}.json").write_text(json.dumps({
        "arch": args.arch, "n_targets": len(ids), "composite_manifest_sha256": man["manifest_sha256"],
        "disk_tissue_arrays_sha256": digest, "checkpoints": entries, "commit": args.commit}, indent=1))
    p64.log("DONE")

def boot(cluster_ids, values, n_resamples=2000):
    from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices
    values = np.asarray(values, float)
    reps = []

    def stat(idx):
        v = float(values[idx].mean())
        reps.append(v)
        return v

    ci = cluster_bootstrap_ci_indices(cluster_ids, stat, n_resamples=n_resamples, seed=0)
    return {"mean": float(values.mean()), "ci": [float(ci[0]), float(ci[1])], "se": float(np.std(reps, ddof=1))}

def analyze(df: pd.DataFrame) -> dict:
    d = {c: df[f"logit_{a}"] - df["logit_before"] for c, a in CELLS.items()}
    p = df["patient_id"].astype(str)
    out = {"n_rows": int(len(df)), "n_images": int(df.image_id.nunique()), "n_patients": int(p.nunique()), "n_checkpoints": int(df.label.nunique()),
           "cells": {c: boot(p, v) for c, v in d.items()}}
    content = ((d["tube_rendered"] - d["tube_tissue"]) + (d["disk_rendered"] - d["disk_tissue"])) / 2
    shape = ((d["tube_rendered"] - d["disk_rendered"]) + (d["tube_tissue"] - d["disk_tissue"])) / 2
    inter = (d["tube_rendered"] - d["tube_tissue"]) - (d["disk_rendered"] - d["disk_tissue"])
    out["decomposition"] = {"content_main_effect": boot(p, content), "shape_main_effect": boot(p, shape), "interaction": boot(p, inter),
                            "content_effect_on_tubes": boot(p, d["tube_rendered"] - d["tube_tissue"]),
                            "content_effect_on_disks": boot(p, d["disk_rendered"] - d["disk_tissue"])}
    out["per_checkpoint_cell_means"] = {lab: {c: float(v[df.label == lab].mean()) for c, v in d.items()} for lab in sorted(df.label.unique())}
    return out

def run_analyze(args) -> None:
    res = {}
    for arch, path in (("densenet121", args.densenet), ("resnet50", args.resnet)):
        res[arch] = analyze(pd.read_csv(path, dtype={"patient_id": str}))
    args.out.write_text(json.dumps(res, indent=1))
    for arch, r in res.items():
        print(arch, {c: round(x["mean"], 4) for c, x in r["cells"].items()}, {k: round(x["mean"], 4) for k, x in r["decomposition"].items()})

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("score")
    s.add_argument("--composite-dir", required=True, type=Path)
    s.add_argument("--anchor-json", required=True, type=Path)
    s.add_argument("--checkpoints", required=True, type=Path, help="JSON list of {label, checkpoint}")
    s.add_argument("--nih-root", required=True, type=Path)
    s.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
    s.add_argument("--out-dir", required=True, type=Path)
    s.add_argument("--device", default="cpu")
    s.add_argument("--commit", default="")
    s.add_argument("--smoke-n", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--densenet", required=True, type=Path)
    a.add_argument("--resnet", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    run_score(args) if args.mode == "score" else run_analyze(args)

if __name__ == "__main__":
    main()
