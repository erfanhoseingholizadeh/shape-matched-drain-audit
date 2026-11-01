# Computes the validity clauses on the reporting half for each further-seed and near-margin checkpoint.
# Run: python3 scoring/further_run_validity.py score --manifest CSV --nih-root DIR --checkpoints JSON --out-dir DIR [--device cuda]

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

REPORT_DIGEST_PREFIX = "b8f26e9e9cd805dd"
REPORT_PATIENTS = 2311
BAR_A, FLOOR_B = 0.82, 0.001
N_RESAMPLES, BOOT_SEED = 2000, 0

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def checkpoint_metrics(labels: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOT_SEED) -> dict:
    from sklearn.metrics import roc_auc_score
    from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices
    point = float(roc_auc_score(labels, probs))
    lo, hi = cluster_bootstrap_ci_indices(patient_ids, lambda ix: float(roc_auc_score(labels[ix], probs[ix])), n_resamples=n_resamples, ci=0.95, seed=seed)
    neg = probs[labels == 0]
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi), "median_neg_p": float(np.median(neg)), "n_negatives": int(len(neg)), "n_positives": int((labels == 1).sum())}

def run_score(a) -> int:
    import torch
    from torch.utils.data import DataLoader
    from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX, NIHPneumothoraxDataset
    from cxr_shortcuts.data.manifests import VAL_SUBSET_SEED, attach_val_subset
    from cxr_shortcuts.eval.population_log import ids_digest, log_population
    from cxr_shortcuts.models.backbones import build_backbone
    from cxr_shortcuts.runners.train import read_manifest_through_guard

    manifest = read_manifest_through_guard(a.manifest, None, dtype={"patient_id": str})
    tagged = attach_val_subset(manifest, seed=VAL_SUBSET_SEED)
    rows = tagged[(tagged["split"] == "val") & (tagged["val_subset"] == "report")].reset_index(drop=True)
    digest = ids_digest(rows["patient_id"].unique())
    if digest[:16] != REPORT_DIGEST_PREFIX or rows["patient_id"].nunique() != REPORT_PATIENTS:
        raise SystemExit(f"derived report half has {rows['patient_id'].nunique()} patients, digest {digest[:16]}; registered {REPORT_PATIENTS}, {REPORT_DIGEST_PREFIX}")
    entries = json.loads(a.checkpoints.read_text())
    if a.smoke_n:
        rows = rows.sort_values("image_id").head(a.smoke_n).reset_index(drop=True)
        entries = entries[:1]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log_population(a.out_dir, "val_report_patients", rows["patient_id"].unique(), "patient")
    log_population(a.out_dir, "val_report_images", rows["image_id"], "image")
    logged = {}
    for e in entries:
        idx = Path(e["checkpoint"]).parent / "populations" / "index.json"
        rec = json.loads(idx.read_text()).get("val_report_patients") if idx.exists() else None
        logged[e["label"]] = None if rec is None else rec["sha256"]
        if not a.smoke_n and (rec is None or rec["sha256"] != digest):
            raise SystemExit(f"{e['label']}: the job's own logged val_report_patients digest {None if rec is None else rec['sha256'][:16]} != the derived report half {digest[:16]}")
    models = []
    for e in entries:
        m = build_backbone(e["backbone"], num_classes=len(NIH_14_LABELS), pretrained=False)
        m.load_state_dict(torch.load(e["checkpoint"], map_location="cpu", weights_only=False))
        models.append(m.to(a.device).eval())
    ds = NIHPneumothoraxDataset(rows, image_root=a.nih_root, split="val", image_size=224, label_columns=list(NIH_14_LABELS))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    probs, labels = [[] for _ in models], []
    with torch.no_grad():
        for n, (x, y) in enumerate(dl):
            x = x.to(a.device)
            for k, m in enumerate(models):
                probs[k].append(torch.sigmoid(m(x))[:, PNEUMOTHORAX_LABEL_INDEX].float().cpu().numpy().astype(np.float64))
            labels.append(y.numpy()[:, PNEUMOTHORAX_LABEL_INDEX])
            if n % 20 == 0:
                log(f"batch {n}/{len(dl)}")
    labels = np.concatenate(labels).astype(int)
    P = np.stack([np.concatenate(p) for p in probs], axis=1)
    pats = rows["patient_id"].to_numpy()
    res = {"report_half": {"patients": int(rows["patient_id"].nunique()), "images": int(len(rows)), "patients_sha256": digest}, "bootstrap": {"B": 2000 if not a.smoke_n else 50, "seed": BOOT_SEED, "construction": "cluster_bootstrap_ci_indices, ci 0.95"},
           "smoke_n": a.smoke_n, "commit": a.commit, "checkpoints": {}}
    for k, e in enumerate(entries):
        met = checkpoint_metrics(labels, P[:, k], pats, n_resamples=2000 if not a.smoke_n else 50)
        res["checkpoints"][e["label"]] = {**met, "family": e.get("family"), "backbone": e["backbone"], "checkpoint": e["checkpoint"], "checkpoint_sha256": sha256_file(Path(e["checkpoint"])), "logged_report_patients_sha256": logged[e["label"]]}
        log(f"{e['label']}: auc {met['auc']:.4f} (ci {met['ci_low']:.4f} to {met['ci_high']:.4f}) median negative p {met['median_neg_p']:.6g}")
    np.savez(a.out_dir / "val_report_probs.npz", image_id=rows["image_id"].to_numpy().astype(str), patient_id=pats.astype(str), label=labels, probs=P, checkpoints=np.array([e["label"] for e in entries]))
    (a.out_dir / "validity_scores.json").write_text(json.dumps(res, indent=1))
    log("DONE score")
    return 0

def run_analyze(a) -> int:
    from analysis import clean_recompute as C56
    res = json.loads(a.scores.read_text())
    out = {"bars": {"clause_a_auc_at_least": BAR_A, "clause_b_median_negative_p_at_least": FLOOR_B}, "report_half": res["report_half"], "smoke_n": res.get("smoke_n", 0), "checkpoints": {}}
    for lab, r in res["checkpoints"].items():
        ca, cb = C56.classify_clause_a(r["auc"], r["ci_high"]), C56.classify_clause_b(r["median_neg_p"])
        out["checkpoints"][lab] = {**r, "clause_a": ca, "clause_b": cb, "valid": bool(ca == "clears" and cb == "clears")}
    fams = {}
    for lab, r in out["checkpoints"].items():
        fams.setdefault(r.get("family"), []).append(r["valid"])
    out["by_family"] = {f: {"checkpoints": len(v), "valid": int(sum(v))} for f, v in fams.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_family"], indent=1))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--nih-root", type=Path, required=True)
    s.add_argument("--checkpoints", type=Path, required=True, help="JSON list of {label, checkpoint, backbone, family}")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--num-workers", type=int, default=8)
    s.add_argument("--smoke-n", type=int, default=0)
    s.add_argument("--commit", default="")
    an = sub.add_parser("analyze")
    an.add_argument("--scores", type=Path, required=True)
    an.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    return run_score(a) if a.cmd == "score" else run_analyze(a)

if __name__ == "__main__":
    sys.exit(main())
