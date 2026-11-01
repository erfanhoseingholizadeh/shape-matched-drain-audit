# Derives the held-out targets, donor pools and donor assignment under the frozen protocol.
# Run: library module, imported by the scripts

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "src"))
from cxr_shortcuts.data.gate_guard import load_manifest, load_table

MIN_RCA = 0.7
M = 8
DONOR_SEED = 7200
MANIFEST = R / "data/processed/manifests/nih_pneumothorax.csv"
NEATX = R / "data/raw/neatx/neatx.csv"
CHEXMASK = R / "data/raw/chexmask/chestxray8.csv"
DISC = R / "data/processed/manifests/discriminator_split.csv"
RESERVED_ARMS = ("eval_paired", "eval_unpaired_reference")

def _image_for(patient: str, images: list, seed: int) -> str:
    return images[int(hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest()[:12], 16) % len(images)]

def assign_donors(targets: pd.DataFrame, pool: pd.DataFrame, m: int = M, seed: int = DONOR_SEED) -> tuple:
    assignment, stats, dropped = {}, {}, []
    for view in sorted(targets.view_position.unique()):
        tv = targets[targets.view_position == view].sort_values("image_id")
        pv = pool[pool.view_position == view]
        patients = np.array(sorted(pv.patient_id.unique()))
        by_patient = pv.groupby("patient_id").image_id.apply(lambda s: sorted(s)).to_dict()
        if len(patients) < m:
            dropped += list(tv.image_id)
            stats[view] = {"targets": len(tv), "pool_patients": len(patients), "dropped_pool_below_m": True}
            continue

        def stream():
            cycle = 0
            while True:
                yield from patients[np.random.default_rng(seed + cycle).permutation(len(patients))]
                cycle += 1

        it, used = stream(), {}
        for row in tv.itertuples():
            chosen = []
            while len(chosen) < m:
                p = next(it)
                if p not in chosen:
                    chosen.append(p)
            for p in chosen:
                used[p] = used.get(p, 0) + 1
            assignment[row.image_id] = [(_image_for(p, by_patient[p], seed), p) for p in chosen]
        slots = m * len(tv)
        stats[view] = {"targets": len(tv), "pool_patients": len(patients), "slots": slots, "patients_used": len(used), "max_uses_of_one_patient": max(used.values()),
                       "reuse_slots": slots - len(used), "dropped_pool_below_m": False}
    return assignment, stats, dropped

def _frames(half: str, entry, root: Path = R) -> dict:
    mpath = root / "data/processed/manifests/nih_pneumothorax.csv"
    ncsv, ccsv, dcsv = root / "data/raw/neatx/neatx.csv", root / "data/raw/chexmask/chestxray8.csv", root / "data/processed/manifests/discriminator_split.csv"
    if half == "dev":
        nih = load_manifest(mpath, dtype={"patient_id": str})
        nih = nih[(nih.split == "test") & (nih.gate_subset == "dev")]
        neatx = load_table(ncsv, id_col="Image Index", id_kind="image", manifest_path=mpath)
        rca = load_table(ccsv, id_col="Image Index", id_kind="image", manifest_path=mpath, usecols=["Image Index", "Dice RCA (Mean)"])
        disc = load_table(dcsv, id_col="patient_id", id_kind="patient", manifest_path=mpath, dtype={"patient_id": str})
    else:
        nih = entry.gate_manifest(mpath, dtype={"patient_id": str})
        neatx = entry.gate_table(ncsv, id_col="Image Index", id_kind="image")
        rca = entry.gate_table(ccsv, id_col="Image Index", id_kind="image", usecols=["Image Index", "Dice RCA (Mean)"])
        disc = entry.gate_table(dcsv, id_col="patient_id", id_kind="patient", dtype={"patient_id": str})
    neatx = neatx.rename(columns={"Image Index": "image_id"})[["image_id", "Drain"]]
    rca = rca.drop_duplicates("Image Index").set_index("Image Index")["Dice RCA (Mean)"]
    reserved = set(disc.loc[disc.discriminator_arm.isin(RESERVED_ARMS), "patient_id"].astype(str))
    return {"nih": nih, "neatx": neatx, "rca": rca, "reserved": reserved, "disc_rows": len(disc)}

def derive(half: str, entry=None, root: Path = R, m: int = M, seed: int = DONOR_SEED) -> dict:
    if half not in ("dev", "gate"):
        raise ValueError(half)
    if half == "gate" and entry is None:
        raise ValueError("the gate half is derived only through a ConfirmatoryEntry")
    f = _frames(half, entry, root)
    nih = f["nih"].merge(f["neatx"], on="image_id", how="left")
    nih["rca"] = nih.image_id.map(f["rca"])
    nih["pneumothorax"] = nih.pneumothorax.astype(str) == "True"
    drain_patients = set(nih.loc[nih.Drain == 1, "patient_id"])
    ok = nih[~nih.patient_id.isin(f["reserved"]) & (nih.rca >= MIN_RCA)]
    tgt = ok[(ok.Drain == 0) & ok.pneumothorax].sort_values("image_id")[["image_id", "patient_id", "view_position"]].reset_index(drop=True)
    target_patients = set(nih.loc[nih.Drain.notna() & nih.pneumothorax, "patient_id"])
    pool = ok[(~ok.pneumothorax) & ~ok.patient_id.isin(drain_patients) & ~ok.patient_id.isin(target_patients)][["image_id", "patient_id", "view_position"]]
    assignment, stats, dropped = assign_donors(tgt, pool, m, seed)
    kept = tgt[~tgt.image_id.isin(dropped)].reset_index(drop=True)
    return {"half": half, "m": m, "seed": seed, "targets": kept, "targets_before_pool_drop": len(tgt), "dropped_pool_below_m": dropped, "assignment": assignment, "stats": stats,
            "targets_drain_labelled_elsewhere": sorted(kept.loc[kept.patient_id.isin(drain_patients), "image_id"]),
            "reserved": f["reserved"], "drain_patients": drain_patients, "target_patients": target_patients, "pool": pool, "half_rows": nih[["image_id", "patient_id", "view_position"]]}

def check(pop: dict) -> dict:
    res, a, tg = {}, pop["assignment"], pop["targets"]
    dp = {p for lst in a.values() for _, p in lst}
    view_of_image = dict(zip(pop["half_rows"].image_id, pop["half_rows"].view_position))
    tp = set(tg.patient_id)
    half_ids = set(pop["half_rows"].image_id)
    res["targets and donor images all inside the half"] = set(tg.image_id) <= half_ids and {i for lst in a.values() for i, _ in lst} <= half_ids
    res["targets avoid reserved-arm patients"] = not (tp & pop["reserved"])
    res["donors avoid reserved-arm patients"] = not (dp & pop["reserved"])
    res["donors avoid drain-labelled patients"] = not (dp & pop["drain_patients"])
    res["donor patients disjoint from every NEATX-labelled pneumothorax patient"] = not (dp & pop["target_patients"])
    res["every target has m distinct donor patients"] = all(len(lst) == pop["m"] and len({p for _, p in lst}) == pop["m"] for lst in a.values()) and set(a) == set(tg.image_id)
    res["every donor image matches its target's view position"] = all(view_of_image[i] == tv for t, tv in zip(tg.image_id, tg.view_position) for i, _ in a[t])
    res["every donor image is a pneumothorax-negative image of its patient"] = set(i for lst in a.values() for i, _ in lst) <= set(pop["pool"].image_id)
    res["non-empty"] = len(tg) > 0
    return res
