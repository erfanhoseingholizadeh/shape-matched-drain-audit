# Runs the frozen protocol's stages (prepare, composite, score) for the held-out half and, with the same code, the development-half comparator.
# Run: python3 scoring/heldout_run.py --help

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

COMPOSITE_SEED = 7201
IMAGE_SIZE = 224
M = 8
MAX_ATTEMPTS = 24
PROTOCOL_FILE = "docs/confirmatory_protocol.md"

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

def attempt_seed(image_id: str, attempt: int) -> int:
    return int(hashlib.sha256(f"{COMPOSITE_SEED}:{image_id}:{attempt}".encode()).hexdigest()[:12], 16)

def arm_names(m: int = M) -> list:
    return ["before"] + [f"{a}_{k}" for k in range(m) for a in ("tube", "disk", "tube_tissue", "disk_tissue")]

def run_prepare(args) -> None:
    import pandas as pd
    from data_prep import heldout_populations as G
    from cxr_shortcuts.data.gate_guard import confirmatory_entry, load_table
    from cxr_shortcuts.eval.population_log import ids_digest, log_population

    entry = None
    if args.half == "gate":
        if not args.protocol_commit:
            raise SystemExit("half=gate requires --protocol-commit")
        entry = confirmatory_entry(args.protocol_commit, purpose="heldout confirmatory prepare", log_path=args.out_dir / "gate_reads.jsonl")
    pop = G.derive(args.half, entry)
    checks = G.check(pop)
    if args.smoke_n:
        keep = list(pop["targets"].image_id[: args.smoke_n])
        pop["targets"] = pop["targets"][pop["targets"].image_id.isin(keep)].reset_index(drop=True)
        pop["assignment"] = {i: v for i, v in pop["assignment"].items() if i in set(keep)}
    if not all(checks.values()):
        raise SystemExit(f"membership check failed: {[k for k, v in checks.items() if not v]}")
    tg, a = pop["targets"], pop["assignment"]
    spec = {"half": args.half, "m": G.M, "donor_seed": G.DONOR_SEED, "composite_seed": COMPOSITE_SEED, "protocol_commit": args.protocol_commit, "smoke_n": args.smoke_n,
            "targets": [[r.image_id, str(r.patient_id), r.view_position] for r in tg.itertuples()],
            "assignment": {i: [[d, str(p)] for d, p in lst] for i, lst in a.items()},
            "targets_drain_labelled_elsewhere": pop["targets_drain_labelled_elsewhere"], "donor_stats": pop["stats"],
            "targets_before_pool_drop": pop["targets_before_pool_drop"], "dropped_pool_below_m": pop["dropped_pool_below_m"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.out_dir / "inputs.json"
    inputs.write_text(json.dumps(spec, indent=1))
    ccsv = G.CHEXMASK
    if args.half == "gate":
        rows = entry.gate_table(ccsv, id_col="Image Index", id_kind="image")
    else:
        rows = load_table(ccsv, id_col="Image Index", id_kind="image", manifest_path=G.MANIFEST)
    rows = rows[rows["Image Index"].isin(set(tg.image_id))]
    if rows["Image Index"].nunique() != len(tg):
        raise SystemExit(f"CheXmask rows found for {rows['Image Index'].nunique()} of {len(tg)} targets")
    masks = args.out_dir / "chexmask_targets.csv"
    rows.to_csv(masks, index=False)
    donor_images = [d for lst in a.values() for d, _ in lst]
    rec = {"half": args.half, "protocol_commit": args.protocol_commit, "inputs_sha256": sha256_file(inputs), "masks_sha256": sha256_file(masks),
           "n_targets": len(tg), "n_targets_before_pool_drop": pop["targets_before_pool_drop"], "n_dropped_pool_below_m": len(pop["dropped_pool_below_m"]),
           "targets_images_digest": ids_digest(tg.image_id), "donor_images_digest": ids_digest(donor_images), "donor_stats": pop["stats"],
           "n_drain_labelled_elsewhere": len(pop["targets_drain_labelled_elsewhere"]), "view_counts": tg.view_position.value_counts().to_dict(), "utc": datetime.now(timezone.utc).isoformat()}
    (args.out_dir / "prepare_record.json").write_text(json.dumps(rec, indent=1))
    (args.out_dir / "prepare_checks.json").write_text(json.dumps({"checks": checks, "protocol_commit": args.protocol_commit, "half": args.half}, indent=1))
    log_population(args.out_dir, f"{args.half}_targets_images", tg.image_id, "image")
    log_population(args.out_dir, f"{args.half}_targets_patients", tg.patient_id, "patient")
    log_population(args.out_dir, f"{args.half}_donor_images", donor_images, "image")
    log_population(args.out_dir, f"{args.half}_donor_patients", [p for lst in a.values() for _, p in lst], "patient")
    log(f"prepared half={args.half}: {len(tg)} targets (of {pop['targets_before_pool_drop']}), views {rec['view_counts']}, drain-elsewhere {rec['n_drain_labelled_elsewhere']}; inputs sha256 {rec['inputs_sha256']}")

def composite_target(image_id, patient_id, donors, loader, lung_mask, left_mask, right_mask, tube_ranges, p64, p70, m=M, max_attempts=MAX_ATTEMPTS):
    from cxr_shortcuts.edits.drain.control_edit import apply_control_edit
    from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, place_mask, placement_lung_side
    from cxr_shortcuts.edits.drain.insertion import insert_drain, log_additive_blend_patch
    from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
    from cxr_shortcuts.runners.composite_and_score import _draw_chord_orientation_deg, _draw_tube_params
    from scoring.blob_control_audit import blob_offset_from_tube_centroid, generate_equal_area_blob
    from scoring.negative_target_audit import BLEND_MODE, FILL_INTENSITY, STRIPE_INTENSITY

    target = loader(image_id)
    background_level = float(target[lung_mask].mean())
    tube, disk, tube_t, disk_t, tmask, dmask, mad = [], [], [], [], [], [], {"insertion": [], "control": [], "blob": []}
    drops, attempts = {"placement": 0, "straddle": 0, "blob_oob": 0}, 0
    donor_cache: dict = {}
    for attempt in range(max_attempts):
        if len(tube) == m:
            break
        attempts += 1
        k = len(tube)
        donor_id, donor_patient = donors[k]
        rng = np.random.default_rng(attempt_seed(image_id, attempt))
        chord = _draw_chord_orientation_deg(left_mask, right_mask, rng)
        caliber_px, length_px, curvature = _draw_tube_params(tube_ranges, rng)
        patch, mask, donor_bg = generate_synthetic_tube_patch(
            length_px=length_px, caliber_px=caliber_px, rng=rng, curvature=curvature, background_level=background_level,
            chord_orientation_deg=chord, fill_intensity=FILL_INTENSITY, stripe_intensity=STRIPE_INTENSITY)
        try:
            insertion = insert_drain(target, lung_mask, patch, mask, rng, blend_mode=BLEND_MODE, donor_background_level=donor_bg,
                                     target_patient_id=patient_id, control_source_patient_id=donor_patient)
        except NoValidPlacementError:
            drops["placement"] += 1
            continue
        if placement_lung_side(mask, insertion.offset, left_mask, right_mask) is None:
            drops["straddle"] += 1
            continue
        blob_patch, blob_mask = generate_equal_area_blob(int(mask.sum()), FILL_INTENSITY, donor_bg)
        blob_offset = blob_offset_from_tube_centroid(mask, insertion.offset, blob_mask)
        bh, bw = blob_mask.shape
        if blob_offset[0] < 0 or blob_offset[1] < 0 or blob_offset[0] + bh > target.shape[0] or blob_offset[1] + bw > target.shape[1]:
            drops["blob_oob"] += 1
            continue
        disk_img = log_additive_blend_patch(target, blob_patch, blob_mask, blob_offset, donor_bg)
        disk_placed = place_mask(blob_mask, target.shape, blob_offset)
        if donor_id not in donor_cache:
            donor_cache[donor_id] = loader(donor_id)
        donor_img = donor_cache[donor_id]
        if donor_img.shape != target.shape:
            raise AssertionError(f"donor {donor_id} shape {donor_img.shape} != target {image_id} shape {target.shape}")
        tt = apply_control_edit(target, donor_img, insertion, blend_mode=BLEND_MODE).image
        dt = p70.build_disk_tissue(target, disk_placed, donor_img, generate_equal_area_blob, place_mask, log_additive_blend_patch)
        mm = insertion.placement_mask
        f = lambda a, b, msk: float(np.abs(a.astype(np.float64) - b.astype(np.float64))[msk].mean())
        mad["insertion"].append(f(insertion.image, target, mm)); mad["control"].append(f(tt, target, mm)); mad["blob"].append(f(disk_img, target, disk_placed))
        tube.append(insertion.image); disk.append(disk_img); tube_t.append(tt); disk_t.append(dt); tmask.append(mm); dmask.append(disk_placed)
    if len(tube) < m:
        return None, {"reason": "fewer_than_m_valid", "valid": len(tube), "drops": drops, "attempts": attempts}
    arrays = {"before": target, "tube": np.stack(tube), "disk": np.stack(disk), "tube_tissue": np.stack(tube_t), "disk_tissue": np.stack(disk_t),
              "tube_mask": np.stack(tmask), "disk_mask": np.stack(dmask)}
    return (arrays, mad), {"drops": drops, "attempts": attempts}

def _write_read_log(path: Path, protocol_commit, purpose: str, table: str, ids: list) -> None:
    from cxr_shortcuts.eval.population_log import ids_digest
    rec = {"utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": protocol_commit, "caller": str(Path(__file__).resolve().name), "purpose": purpose,
           "table": table, "n_rows": len(ids), "ids_digest": ids_digest(ids)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

def run_composite(args) -> None:
    import pandas as pd
    from cxr_shortcuts.data.image_io import default_image_loader
    from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved, reserved_ids_path
    from cxr_shortcuts.eval.population_log import ids_digest, log_population
    from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges
    from scoring import fresh_composite_audit as p64
    from scoring import twobytwo_audit as p70

    spec_sha = sha256_file(args.input_json)
    if spec_sha != args.expected_inputs_sha256:
        raise SystemExit(f"inputs sha256 {spec_sha} != the prepare record's {args.expected_inputs_sha256}")
    spec = json.loads(args.input_json.read_text())
    if spec["m"] != M or spec["composite_seed"] != COMPOSITE_SEED:
        raise SystemExit("inputs were prepared with a different M or composite seed than this script")
    log(f"reserved-id file {reserved_ids_path()} sha256={sha256_file(reserved_ids_path())}; inputs sha256 {spec_sha}; half={spec['half']}")
    targets = spec["targets"]
    if args.smoke_n:
        targets = targets[: args.smoke_n]
    donors_of = spec["assignment"]
    all_donor_ids = [d for t in targets for d, _ in donors_of[t[0]]]
    assert_not_reserved([t[0] for t in targets], "heldout targets")
    assert_not_reserved(all_donor_ids, "heldout donors")
    reads = args.out_dir / "image_reads.jsonl"
    _write_read_log(reads, spec["protocol_commit"], f"heldout composite targets (half={spec['half']})", "nih_images:targets", [t[0] for t in targets])
    _write_read_log(reads, spec["protocol_commit"], f"heldout composite donors (half={spec['half']})", "nih_images:donors", sorted(set(all_donor_ids)))
    rca = pd.read_csv(args.rca_csv)
    relevant = rca[rca["Image Index"].isin([t[0] for t in targets])]
    if relevant["Image Index"].nunique() != len(targets):
        raise SystemExit("CheXmask rows missing for some targets")
    lung_masks = load_chexmask_lung_masks(relevant)
    lateral = load_chexmask_lateral_masks(relevant)
    loader = default_image_loader(args.nih_root)
    tube_ranges = TubeGenerationRanges()
    args.composite_dir.mkdir(parents=True, exist_ok=True)
    file_hashes, meta, drops, mad = {}, {}, {}, {"insertion": [], "control": [], "blob": []}
    for n, (image_id, patient_id, view) in enumerate(targets):
        res, info = composite_target(image_id, patient_id, [tuple(d) for d in donors_of[image_id]], loader, lung_masks[image_id], *lateral[image_id], tube_ranges, p64, p70)
        if res is None:
            drops[image_id] = info
            continue
        arrays, m = res
        fname = f"{Path(image_id).stem}.npz"
        np.savez(args.composite_dir / fname, **arrays)
        file_hashes[fname] = sha256_file(args.composite_dir / fname)
        meta[image_id] = {"file": fname, "patient_id": patient_id, "view_position": view, "donors": donors_of[image_id], **info,
                          "drain_labelled_elsewhere": image_id in set(spec["targets_drain_labelled_elsewhere"])}
        for key in mad:
            mad[key].extend(m[key])
        if (n + 1) % 10 == 0:
            log(f"composited {n + 1}/{len(targets)}")
    floors = p64.floor_checks(mad)
    manifest = {"m": M, "composite_seed": COMPOSITE_SEED, "kept": sorted(meta), "meta": meta, "file_hashes": file_hashes, "manifest_sha256": manifest_digest(file_hashes),
                "input_json_sha256": spec_sha, "half": spec["half"], "protocol_commit": spec["protocol_commit"]}
    (args.composite_dir / "composite_manifest.json").write_text(json.dumps(manifest, indent=1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_population(args.out_dir, f"composited_{spec['half']}_targets", manifest["kept"], "image")
    log_population(args.out_dir, f"composited_{spec['half']}_donors", [d for i in manifest["kept"] for d, _ in meta[i]["donors"]], "image")
    attempts = [meta[i]["attempts"] for i in meta]
    (args.out_dir / "composite_summary.json").write_text(json.dumps({
        "half": spec["half"], "n_targets_attempted": len(targets), "n_kept": len(manifest["kept"]), "dropped": drops, "floors": floors, "smoke_n": args.smoke_n,
        "attempts_per_kept_target": {"mean": float(np.mean(attempts)) if attempts else None, "max": int(max(attempts)) if attempts else None},
        "composite_manifest_sha256": manifest["manifest_sha256"], "input_json_sha256": spec_sha, "reserved_ids_file_sha256": sha256_file(reserved_ids_path()), "commit": args.commit}, indent=1))
    log(f"DONE composite: kept {len(manifest['kept'])} of {len(targets)}; drops {len(drops)}; floors {floors}")

def load_verified(comp_dir: Path, expected_digest) -> dict:
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

def stack_arrays(arrays: dict) -> np.ndarray:
    from PIL import Image
    ims = [arrays["before"]]
    for k in range(arrays["tube"].shape[0]):
        ims += [arrays["tube"][k], arrays["disk"][k], arrays["tube_tissue"][k], arrays["disk_tissue"][k]]
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
    ids_all = manifest["kept"][: args.smoke_n] if args.smoke_n else manifest["kept"]
    stacks, first = {}, None
    for image_id in ids_all:
        with np.load(args.composite_dir / manifest["meta"][image_id]["file"]) as z:
            arrs = {k: z[k] for k in ("before", "tube", "disk", "tube_tissue", "disk_tissue")}
        stacks[image_id] = stack_arrays(arrs)
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
    names = arm_names()
    header = ["label", "family", "image_id", "patient_id", "view_position"] + [f"logit_{a}" for a in names]
    with open(args.out_dir / f"scores_{args.arch}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for e in entries:
            p64.BACKBONE = args.arch
            model = p64.build_model(e["checkpoint"], args.device)
            with torch.no_grad():
                for image_id in ids_all:
                    lg = model(to_input(stacks[image_id]).to(args.device))[:, p64.PNEUMOTHORAX_LABEL_INDEX].detach().cpu().numpy().astype(np.float64)
                    m = manifest["meta"][image_id]
                    w.writerow([e["label"], e["family"], image_id, m["patient_id"], m["view_position"], *lg.tolist()])
            log(f"scored {e['label']} on {len(ids_all)} targets")
            del model
    log_population(args.out_dir, f"scored_{args.arch}_targets", ids_all, "image")
    (args.out_dir / f"summary_{args.arch}.json").write_text(json.dumps({
        "arch": args.arch, "composite_manifest_sha256": manifest["manifest_sha256"], "checkpoints": entries, "commit": args.commit,
        "checkpoints_file_sha256": sha256_file(args.checkpoints), "n_targets_in_composite": len(manifest["kept"])}, indent=1))
    log("DONE score")

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--half", required=True, choices=["dev", "gate"])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--protocol-commit", default=None)
    p.add_argument("--smoke-n", type=int, default=0)
    c = sub.add_parser("composite")
    c.add_argument("--input-json", required=True, type=Path)
    c.add_argument("--expected-inputs-sha256", required=True)
    c.add_argument("--rca-csv", required=True, type=Path)
    c.add_argument("--nih-root", required=True, type=Path)
    c.add_argument("--composite-dir", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)
    c.add_argument("--smoke-n", type=int, default=0)
    c.add_argument("--commit", default="")
    s = sub.add_parser("score")
    s.add_argument("--composite-dir", required=True, type=Path)
    s.add_argument("--checkpoints", required=True, type=Path)
    s.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
    s.add_argument("--out-dir", required=True, type=Path)
    s.add_argument("--expected-digest", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--smoke-n", type=int, default=0)
    s.add_argument("--commit", default="")
    args = ap.parse_args()
    {"prepare": run_prepare, "composite": run_composite, "score": run_score}[args.mode](args)

if __name__ == "__main__":
    main()
