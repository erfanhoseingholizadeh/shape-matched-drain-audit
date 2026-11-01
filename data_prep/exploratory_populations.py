# Derives the exploratory audit's targets, donor pools and donor assignment from the manifest and the NEATX labels.
# Run: library module, imported by the scripts

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "src"))
from cxr_shortcuts.data.gate_guard import load_manifest, load_table
from cxr_shortcuts.eval.population_log import log_population
from data_prep.clean_population import val_halves

MIN_RCA = 0.7
K_DONORS = 8
DONOR_SEED = 7100
STRATA = ("dev_half", "R_int_C", "R_int_Q")
ORIGINAL_STRATA = STRATA
PLANTED_STRATA = ("dev_half", "R_int_C")

def _frames(root: Path = R) -> dict:

    mpath = root / "data/processed/manifests/nih_pneumothorax.csv"
    neatx = load_table(root / "data/raw/neatx/neatx.csv", id_col="Image Index", id_kind="image", manifest_path=mpath).rename(columns={"Image Index": "image_id"})[["image_id", "Drain"]]
    nih = load_manifest(mpath, dtype={"patient_id": str})
    disc = load_table(root / "data/processed/manifests/discriminator_split.csv", id_col="patient_id", id_kind="patient", manifest_path=mpath, dtype={"patient_id": str})
    rca = load_table(root / "data/raw/chexmask/chestxray8.csv", id_col="Image Index", id_kind="image", manifest_path=mpath,
                     usecols=["Image Index", "Dice RCA (Mean)"]).drop_duplicates("Image Index").set_index("Image Index")["Dice RCA (Mean)"]
    reserved = set(disc.loc[disc.discriminator_arm.isin(("eval_paired", "eval_unpaired_reference")), "patient_id"])
    return {"neatx": neatx, "nih": nih, "rca": rca, "reserved": reserved, "halves": val_halves(mpath)}

def derive(root: Path = R, k: int = K_DONORS, seed: int = DONOR_SEED) -> dict:
    f = _frames(root)
    h = f["halves"]
    r_int, r_str, s_str = set(h["R_int"]), set(h["R_str"]), set(h["S_str"])
    nih = f["nih"]

    nih = nih[((nih.split == "val") & nih.patient_id.isin(r_int)) | ((nih.split == "test") & (nih.gate_subset == "dev"))].copy()
    nih = nih.merge(f["neatx"], on="image_id", how="left")
    nih["rca"] = nih.image_id.map(f["rca"])
    source_rows = {
        "dev_half": nih[nih.split == "test"],
        "R_int_C": nih[(nih.split == "val") & nih.patient_id.isin(r_int & r_str)],
        "R_int_Q": nih[(nih.split == "val") & nih.patient_id.isin(r_int & s_str)],
    }
    out = {"k": k, "seed": seed, "strata": {}, "reserved": f["reserved"], "halves": {n: set(h[n]) for n in ("S_int", "R_int", "S_str", "R_str")}}
    rng = np.random.default_rng(seed)
    for name in STRATA:
        df = source_rows[name]
        drain_patients = set(df.loc[df.Drain == 1, "patient_id"])
        ok = df[~df.patient_id.isin(f["reserved"]) & (df.rca >= MIN_RCA)]
        tgt = ok[ok.Drain == 0].sort_values("image_id")
        assert tgt.pneumothorax.astype(bool).all()
        target_patients = set(df.loc[df.Drain.notna() & df.pneumothorax.astype(bool), "patient_id"])
        pool = ok[(~ok.pneumothorax.astype(bool)) & ~ok.patient_id.isin(drain_patients) & ~ok.patient_id.isin(target_patients)]
        patients = np.array(sorted(pool.patient_id.unique()))
        need = k * len(tgt)
        assert need <= len(patients), (name, need, len(patients))
        order = rng.permutation(len(patients))[:need]
        chosen = patients[order].reshape(len(tgt), k)
        by_patient = pool.groupby("patient_id").image_id.apply(lambda s: sorted(s)).to_dict()
        assignment = {}
        for i, image_id in enumerate(tgt.image_id):
            donors = []
            for p in chosen[i]:
                imgs = by_patient[p]
                donors.append((imgs[int(rng.integers(len(imgs)))], p))
            assignment[image_id] = donors
        out["strata"][name] = {
            "targets": tgt[["image_id", "patient_id"]].reset_index(drop=True),
            "targets_drain_labelled_elsewhere": sorted(tgt.loc[tgt.patient_id.isin(drain_patients), "image_id"]),
            "donor_pool_images": int(len(pool)), "donor_pool_patients": int(len(patients)),
            "drain_patients": drain_patients, "assignment": assignment,
        }
    return out

def check(pop: dict) -> dict:
    h, res = pop["halves"], {}
    for name, s in pop["strata"].items():
        tp = set(s["targets"].patient_id)
        dp = {p for lst in s["assignment"].values() for _, p in lst}
        di = [i for lst in s["assignment"].values() for i, _ in lst]
        res[f"{name}: targets avoid reserved-arm patients"] = not (tp & pop["reserved"])
        res[f"{name}: donors avoid reserved-arm patients"] = not (dp & pop["reserved"])
        res[f"{name}: donors avoid drain-labelled patients"] = not (dp & s["drain_patients"])
        res[f"{name}: donor patients disjoint from target patients"] = not (dp & tp)
        res[f"{name}: every target has K donors, K distinct patients"] = all(len(v) == pop["k"] and len({p for _, p in v}) == pop["k"] for v in s["assignment"].values())
        res[f"{name}: donor patients unique across the stratum (singleton clusters)"] = len(dp) == len(di) == len(set(di)) == pop["k"] * len(s["targets"])
        res[f"{name}: non-empty"] = len(s["targets"]) > 0
    val_strata = {"R_int_C": h["R_int"] & h["R_str"], "R_int_Q": h["R_int"] & h["S_str"]}
    for name, members in val_strata.items():
        s = pop["strata"][name]
        allp = set(s["targets"].patient_id) | {p for lst in s["assignment"].values() for _, p in lst}
        res[f"{name}: all target and donor patients inside the stratum"] = allp <= members
        res[f"{name}: disjoint from S_int"] = not (allp & h["S_int"])
    res["R_int_C disjoint from S_str"] = not ({*pop["strata"]["R_int_C"]["targets"].patient_id} & h["S_str"])
    res["R_int_Q inside S_str (planted families' select half)"] = {*pop["strata"]["R_int_Q"]["targets"].patient_id} <= h["S_str"]
    res["strata disjoint from each other"] = len({*pop["strata"]["dev_half"]["targets"].patient_id} & {*pop["strata"]["R_int_C"]["targets"].patient_id}) == 0 and \
        len({*pop["strata"]["R_int_C"]["targets"].patient_id} & {*pop["strata"]["R_int_Q"]["targets"].patient_id}) == 0
    return res

def log_all(pop: dict, out_dir) -> None:
    for name, s in pop["strata"].items():
        log_population(out_dir, f"{name}_targets_images", s["targets"].image_id, "image")
        log_population(out_dir, f"{name}_targets_patients", s["targets"].patient_id, "patient")
        log_population(out_dir, f"{name}_donor_images", [i for lst in s["assignment"].values() for i, _ in lst], "image")
        log_population(out_dir, f"{name}_donor_patients", [p for lst in s["assignment"].values() for _, p in lst], "patient")
