# Builds the four arms for the exploratory positive targets and scores them with the checkpoints.
# Run: python3 scoring/positive_audit.py --help

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

COMPOSITE_SEED = 7101
IMAGE_SIZE = 224
K = 8
ARMS = ["before", "tube", "disk"] + [f"tube_tissue_{k}" for k in range(K)] + [f"disk_tissue_{k}" for k in range(K)]

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest_digest(file_hashes: dict) -> str:
    return hashlib.sha256("\n".join(f"{k} {file_hashes[k]}" for k in sorted(file_hashes)).encode()).hexdigest()

def target_seed(image_id: str) -> int:
    return int(hashlib.sha256(f"{COMPOSITE_SEED}:{image_id}".encode()).hexdigest()[:12], 16)

def run_prepare(args) -> None:
    import pandas as pd
    from data_prep import exploratory_populations as P

    pop = P.derive(REPO_ROOT)
    checks = P.check(pop)
    if not all(checks.values()):
        raise SystemExit(f"PC-31 membership check failed: {[k for k, v in checks.items() if not v]}")
    from cxr_shortcuts.eval.population_log import ids_digest
    out = {"k": pop["k"], "donor_seed": pop["seed"], "composite_seed": COMPOSITE_SEED, "strata": {}, "pc31_checks": checks}
    all_targets = []
    for name, s in pop["strata"].items():
        out["strata"][name] = {
            "targets": [[r.image_id, str(r.patient_id)] for r in s["targets"].itertuples()],
            "assignment": {i: [[d, str(p)] for d, p in lst] for i, lst in s["assignment"].items()},
            "targets_drain_labelled_elsewhere": s["targets_drain_labelled_elsewhere"],
        }
        all_targets += list(s["targets"].image_id)
    out["digests"] = {name: {"targets_images": ids_digest(s["targets"].image_id),
                             "donor_images": ids_digest([d for lst in s["assignment"].values() for d, _ in lst])}
                      for name, s in pop["strata"].items()}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    rca = pd.read_csv(args.rca_csv)
    sub = rca[rca["Image Index"].isin(all_targets)]
    if sub["Image Index"].nunique() != len(all_targets):
        raise SystemExit(f"CheXmask rows found for {sub['Image Index'].nunique()} of {len(all_targets)} targets")
    sub.to_csv(args.out_masks, index=False)
    log(f"prepared {len(all_targets)} targets; input json sha256 {sha256_file(args.out_json)}; masks csv sha256 {sha256_file(args.out_masks)}")

def composite_one(image_id, patient_id, donors, loader, lung_mask, left_mask, right_mask, rng, p64, p70, tube_ranges):
    from cxr_shortcuts.edits.drain.control_edit import apply_control_edit
    from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, place_mask, placement_lung_side
    from cxr_shortcuts.edits.drain.insertion import insert_drain, log_additive_blend_patch
    from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
    from cxr_shortcuts.runners.composite_and_score import _draw_chord_orientation_deg, _draw_tube_params
    from scoring.blob_control_audit import blob_offset_from_tube_centroid, generate_equal_area_blob
    from scoring.negative_target_audit import BLEND_MODE, FILL_INTENSITY, STRIPE_INTENSITY

    target = loader(image_id)
    background_level = float(target[lung_mask].mean())
    chord = _draw_chord_orientation_deg(left_mask, right_mask, rng)
    caliber_px, length_px, curvature = _draw_tube_params(tube_ranges, rng)
    patch, mask, donor_bg = generate_synthetic_tube_patch(
        length_px=length_px, caliber_px=caliber_px, rng=rng, curvature=curvature, background_level=background_level,
        chord_orientation_deg=chord, fill_intensity=FILL_INTENSITY, stripe_intensity=STRIPE_INTENSITY)
    try:
        insertion = insert_drain(target, lung_mask, patch, mask, rng, blend_mode=BLEND_MODE, donor_background_level=donor_bg,
                                 target_patient_id=patient_id, control_source_patient_id=donors[0][1])
    except NoValidPlacementError:
        return None, "placement"
    if placement_lung_side(mask, insertion.offset, left_mask, right_mask) is None:
        return None, "straddle"
    blob_patch, blob_mask = generate_equal_area_blob(int(mask.sum()), FILL_INTENSITY, donor_bg)
    blob_offset = blob_offset_from_tube_centroid(mask, insertion.offset, blob_mask)
    bh, bw = blob_mask.shape
    if blob_offset[0] < 0 or blob_offset[1] < 0 or blob_offset[0] + bh > target.shape[0] or blob_offset[1] + bw > target.shape[1]:
        return None, "blob_oob"
    disk = log_additive_blend_patch(target, blob_patch, blob_mask, blob_offset, donor_bg)
    disk_placed = place_mask(blob_mask, target.shape, blob_offset)
    tube_tissue, disk_tissue = [], []
    for donor_id, _ in donors:
        donor_img = loader(donor_id)
        if donor_img.shape != target.shape:
            raise AssertionError(f"donor {donor_id} shape {donor_img.shape} != target {image_id} shape {target.shape}")
        tube_tissue.append(apply_control_edit(target, donor_img, insertion, blend_mode=BLEND_MODE).image)
        disk_tissue.append(p70.build_disk_tissue(target, disk_placed, donor_img, generate_equal_area_blob, place_mask, log_additive_blend_patch))
    m = insertion.placement_mask
    f = lambda a, b, mm: float(np.abs(a.astype(np.float64) - b.astype(np.float64))[mm].mean())
    mad = {"insertion": f(insertion.image, target, m), "control": float(np.mean([f(t, target, m) for t in tube_tissue])),
           "blob": f(disk, target, disk_placed)}
    arrays = {"before": target, "tube": insertion.image, "disk": disk, "tube_tissue": np.stack(tube_tissue), "disk_tissue": np.stack(disk_tissue),
              "tube_mask": m, "disk_mask": disk_placed}
    return (arrays, mad), None

def run_composite(args) -> None:
    import pandas as pd
    from cxr_shortcuts.data.image_io import default_image_loader
    from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved, reserved_ids_path
    from cxr_shortcuts.eval.population_log import ids_digest, log_population
    from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges
    from scoring import fresh_composite_audit as p64
    from scoring import twobytwo_audit as p70

    spec = json.loads(args.input_json.read_text())
    if spec["k"] != K:
        raise SystemExit(f"input K {spec['k']} != script K {K}")
    for name, s in spec["strata"].items():
        if ids_digest([t[0] for t in s["targets"]]) != spec["digests"][name]["targets_images"]:
            raise SystemExit(f"{name}: target ids do not match the frozen digest")
        if ids_digest([d for lst in s["assignment"].values() for d, _ in lst]) != spec["digests"][name]["donor_images"]:
            raise SystemExit(f"{name}: donor ids do not match the frozen digest")
    log(f"reserved-id file {reserved_ids_path()} sha256={sha256_file(reserved_ids_path())}")
    targets = [(name, t[0], t[1]) for name, s in spec["strata"].items() for t in s["targets"]]
    if args.smoke_n:
        keep = {n: {t[0] for t in s["targets"][: args.smoke_n]} for n, s in spec["strata"].items()}
        targets = [t for t in targets if t[1] in keep[t[0]]]
    donors_of = {i: lst for s in spec["strata"].values() for i, lst in s["assignment"].items()}
    all_donor_ids = [d for _, i, _ in targets for d, _ in donors_of[i]]
    assert_not_reserved([t[1] for t in targets], "positive_audit targets")
    assert_not_reserved(all_donor_ids, "positive_audit donors")
    rca = pd.read_csv(args.rca_csv)
    relevant = rca[rca["Image Index"].isin([t[1] for t in targets])]
    if relevant["Image Index"].nunique() != len(targets):
        raise SystemExit("CheXmask rows missing for some targets")
    lung_masks = load_chexmask_lung_masks(relevant)
    lateral = load_chexmask_lateral_masks(relevant)
    loader = default_image_loader(args.nih_root)
    tube_ranges = TubeGenerationRanges()
    args.composite_dir.mkdir(parents=True, exist_ok=True)
    file_hashes, meta, drops, mad = {}, {}, {}, {"insertion": [], "control": [], "blob": []}
    for n, (stratum, image_id, patient_id) in enumerate(targets):
        rng = np.random.default_rng(target_seed(image_id))
        res, why = composite_one(image_id, patient_id, donors_of[image_id], loader, lung_masks[image_id], *lateral[image_id], rng, p64, p70, tube_ranges)
        if res is None:
            drops.setdefault(stratum, {}).setdefault(why, []).append(image_id)
            continue
        arrays, m = res
        fname = f"{Path(image_id).stem}.npz"
        np.savez(args.composite_dir / fname, **arrays)
        file_hashes[fname] = sha256_file(args.composite_dir / fname)
        meta[image_id] = {"file": fname, "patient_id": patient_id, "stratum": stratum, "donors": donors_of[image_id]}
        for key in mad:
            mad[key].append(m[key])
        if (n + 1) % 20 == 0:
            log(f"composited {n + 1}/{len(targets)}")
    floors = p64.floor_checks(mad)
    manifest = {"k": K, "composite_seed": COMPOSITE_SEED, "kept": sorted(meta), "meta": meta, "file_hashes": file_hashes,
                "manifest_sha256": manifest_digest(file_hashes), "input_json_sha256": sha256_file(args.input_json)}
    (args.composite_dir / "composite_manifest.json").write_text(json.dumps(manifest, indent=1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept_by_stratum = {s: [i for i in manifest["kept"] if meta[i]["stratum"] == s] for s in spec["strata"]}
    for s, ids in kept_by_stratum.items():
        log_population(args.out_dir, f"composited_{s}_targets", ids, "image")
        log_population(args.out_dir, f"composited_{s}_donors", [d for i in ids for d, _ in meta[i]["donors"]], "image")
    (args.out_dir / "composite_summary.json").write_text(json.dumps({
        "n_targets_attempted": len(targets), "n_kept": len(manifest["kept"]), "kept_by_stratum": {s: len(v) for s, v in kept_by_stratum.items()},
        "drops": {s: {w: len(v) for w, v in d.items()} for s, d in drops.items()}, "dropped_ids": drops, "floors": floors, "smoke_n": args.smoke_n,
        "composite_manifest_sha256": manifest["manifest_sha256"], "input_json_sha256": manifest["input_json_sha256"],
        "reserved_ids_file_sha256": sha256_file(reserved_ids_path()), "commit": args.commit}, indent=1))
    log(f"DONE composite: kept {len(manifest['kept'])} of {len(targets)}; floors {floors}")

def load_verified(comp_dir: Path, expected_digest: str | None) -> dict:
    manifest = json.loads((comp_dir / "composite_manifest.json").read_text())
    if manifest_digest(manifest["file_hashes"]) != manifest["manifest_sha256"]:
        raise SystemExit("composite manifest's own digest does not match its file list")
    if expected_digest is not None and manifest["manifest_sha256"] != expected_digest:
        raise SystemExit(f"manifest digest {manifest['manifest_sha256']} != expected {expected_digest}")
    for fname, h in manifest["file_hashes"].items():
        if sha256_file(comp_dir / fname) != h:
            raise SystemExit(f"sha256 mismatch for {fname}")
    log(f"verified {len(manifest['file_hashes'])} composite files against manifest {manifest['manifest_sha256']}")
    return manifest

def resized_stack(arrays: dict) -> np.ndarray:
    from PIL import Image
    ims = [arrays["before"], arrays["tube"], arrays["disk"], *arrays["tube_tissue"], *arrays["disk_tissue"]]
    return np.stack([np.array(Image.fromarray(a).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) for a in ims])

def run_score(args) -> None:
    import torch
    from cxr_shortcuts.data.transforms import IMAGENET_MEAN, IMAGENET_STD, to_model_tensor
    from cxr_shortcuts.eval.population_log import log_population
    from scoring import fresh_composite_audit as p64

    manifest = load_verified(args.composite_dir, args.expected_digest)
    entries = json.loads(args.checkpoints.read_text())
    if args.smoke_n:
        entries = entries[:1]
    stacks, first = {}, None
    for image_id in manifest["kept"]:
        with np.load(args.composite_dir / manifest["meta"][image_id]["file"]) as z:
            arrs = {k: z[k] for k in ("before", "tube", "disk", "tube_tissue", "disk_tissue")}
        stacks[image_id] = resized_stack(arrs)
        if first is None:
            first = (arrs["before"], stacks[image_id][0])
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def to_input(u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(u8).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        return (x - mean) / std

    ref = to_model_tensor(first[0], size=IMAGE_SIZE)
    if not torch.allclose(to_input(first[1][None])[0], ref, atol=1e-6):
        raise SystemExit("the cached-resize input path does not reproduce to_model_tensor")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"scores_{args.arch}.csv"
    header = ["label", "family", "stratum", "image_id", "patient_id"] + [f"logit_{a}" for a in ARMS]
    scored = {}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for e in entries:
            p64.BACKBONE = args.arch
            model = p64.build_model(e["checkpoint"], args.device)
            allowed = set(e["strata"])
            ids = [i for i in manifest["kept"] if manifest["meta"][i]["stratum"] in allowed]
            scored.setdefault(e["family"], set()).update(ids)
            with torch.no_grad():
                for image_id in ids:
                    lg = model(to_input(stacks[image_id]).to(args.device))[:, p64.PNEUMOTHORAX_LABEL_INDEX].detach().cpu().numpy().astype(np.float64)
                    m = manifest["meta"][image_id]
                    w.writerow([e["label"], e["family"], m["stratum"], image_id, m["patient_id"], *lg.tolist()])
            log(f"scored {e['label']} on {len(ids)} targets")
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    for fam, ids in scored.items():
        log_population(args.out_dir, f"scored_{args.arch}_{fam}_targets", ids, "image")
    (args.out_dir / f"summary_{args.arch}.json").write_text(json.dumps({
        "arch": args.arch, "composite_manifest_sha256": manifest["manifest_sha256"], "checkpoints": entries, "commit": args.commit,
        "checkpoints_file_sha256": sha256_file(args.checkpoints), "n_targets_in_composite": len(manifest["kept"])}, indent=1))
    log("DONE score")

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--rca-csv", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-masks", required=True, type=Path)
    c = sub.add_parser("composite")
    c.add_argument("--input-json", required=True, type=Path)
    c.add_argument("--rca-csv", required=True, type=Path, help="the targets' CheXmask rows written by `prepare`")
    c.add_argument("--nih-root", required=True, type=Path)
    c.add_argument("--composite-dir", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)
    c.add_argument("--smoke-n", type=int, default=0)
    c.add_argument("--commit", default="")
    s = sub.add_parser("score")
    s.add_argument("--composite-dir", required=True, type=Path)
    s.add_argument("--checkpoints", required=True, type=Path, help="JSON list of {label, checkpoint, family, strata}")
    s.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
    s.add_argument("--out-dir", required=True, type=Path)
    s.add_argument("--expected-digest", default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--smoke-n", type=int, default=0)
    s.add_argument("--commit", default="")
    args = ap.parse_args()
    {"prepare": run_prepare, "composite": run_composite, "score": run_score}[args.mode](args)

if __name__ == "__main__":
    main()
