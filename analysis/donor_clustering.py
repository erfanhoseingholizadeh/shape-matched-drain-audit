# Descriptive summaries of the recorded held-out outputs, including the effect of clustering donors as well as targets.
# Run: python3 analysis/donor_clustering.py --out review/further_runs/gate_planted_donor.json

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))
from analysis.heldout_analysis import B_RESAMPLES, MARGIN, M, bootstrap, contrast_arrays, effects
from analysis.weak_vs_original import joint_bootstrap, primary_per_target, summ

GATE_DIR = ROOT / "review/heldout/confirmatory/cluster_outputs"
INPUTS = ROOT / "review/heldout/confirmatory/inputs.json"
MANIFEST = ROOT / "review/further_runs/gate_meta/composite_manifest.json"
ARCH_GLOB = {"densenet121": "gate_s_dn_*/scores_densenet121.csv", "resnet50": "gate_s_rn_*/scores_resnet50.csv"}

def load(arch: str) -> pd.DataFrame:
    files = sorted(GATE_DIR.glob(ARCH_GLOB[arch]))
    return pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files], ignore_index=True)

def planted_minus_zero(arch: str) -> dict:
    df = load(arch)
    planted = sorted(df[df.family != "original"].label.unique())
    strengths = sorted({l.rsplit("_seed", 1)[0] for l in planted})
    assert "zero" in strengths
    ids, patients, _ = primary_per_target(df, [l for l in planted if l.startswith("zero_")])
    sets = {}
    for s in strengths:
        labs = [l for l in planted if l.rsplit("_seed", 1)[0] == s]
        i2, p2, v = primary_per_target(df, labs)
        assert i2 == ids and (p2 == patients).all()
        sets[s] = v
    reps = joint_bootstrap(patients, sets)
    pts = {k: float(v.mean()) for k, v in sets.items()}
    n_ck = {s: len([l for l in planted if l.rsplit("_seed", 1)[0] == s]) for s in strengths}
    out = {"n_targets": len(ids), "n_patients": int(len(set(patients))), "n_checkpoints": n_ck, "margin": MARGIN[arch], "strengths_present": strengths,
           "pooled_primary": {s: summ(pts[s], reps[s]) for s in strengths},
           "minus_zero": {s: summ(pts[s] - pts["zero"], reps[s] - reps["zero"]) for s in strengths if s != "zero"}}
    return out

def cgm_variances(y: np.ndarray, target_id: np.ndarray, donor_id: np.ndarray) -> dict:
    n = len(y)
    e = y - y.mean()

    def v(cluster):
        s = pd.Series(e).groupby(cluster).sum()
        return float((s ** 2).sum()) / n ** 2
    vt, vd = v(target_id), v(donor_id)
    vi = v(np.array([f"{a}|{b}" for a, b in zip(target_id, donor_id)]))
    return {"V_target": vt, "V_donor": vd, "V_target_by_donor": vi, "V_two_way": vt + vd - vi, "V_donor_equals_intersection": bool(abs(vd - vi) < 1e-15 * max(1.0, vd))}

def donor_dependence(arch: str, assignment: dict) -> dict:
    df = load(arch)
    labels = sorted(df[df.family == "original"].label.unique())
    ids, patients, views, R, T, B, D = effects(df, labels)
    prim = (R - T).mean(axis=1)
    n_t, m = prim.shape
    donor_pat = np.array([[assignment[i][k][1] for k in range(m)] for i in ids])
    y = prim.ravel()
    tid = np.repeat(np.asarray(patients), m)
    did = donor_pat.ravel()
    var = cgm_variances(y, tid, did)
    mean = float(y.mean())
    z = 1.959963984540054
    avg = {"primary": prim}
    nested = bootstrap(patients, avg)["primary"]

    per_t = prim.mean(axis=1)
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(0)
    reps = np.empty(B_RESAMPLES)
    for i in range(B_RESAMPLES):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        reps[i] = per_t[idx].mean()
    se2, se1 = var["V_two_way"] ** 0.5, var["V_target"] ** 0.5
    return {"n_targets": n_t, "n_patients": int(len(uniq)), "slots": int(y.size), "distinct_donor_patients": int(len(set(did.tolist()))), "mean": mean, "variances": var,
            "interval_nested_registered": nested, "interval_target_bootstrap_slots_fixed": summ(mean, reps),
            "interval_two_way_normal_95": [mean - z * se2, mean + z * se2], "interval_one_way_target_normal_95": [mean - z * se1, mean + z * se1],
            "se": {"two_way": se2, "one_way_target": se1, "nested_bootstrap_ci95_halfwidth_over_z": (nested["ci95"][1] - nested["ci95"][0]) / (2 * z), "target_bootstrap_sd": float(reps.std(ddof=1))}}

def recorded_metadata(assignment: dict) -> dict:
    man = json.loads(MANIFEST.read_text())
    meta = man["meta"]
    kept = man["kept"]
    donors = [tuple(d) for k in kept for d in meta[k]["donors"]]
    per_patient = Counter(p for _, p in donors)
    tp = Counter(meta[k]["patient_id"] for k in kept)
    drops = Counter()
    for k in kept:
        drops.update(meta[k]["drops"])
    att = [meta[k]["attempts"] for k in kept]
    views = Counter(meta[k]["view_position"] for k in kept)
    inp = json.loads(INPUTS.read_text())
    return {"n_targets": len(kept), "targets_by_view": dict(views), "target_patients": len(tp), "targets_per_patient": {"min": min(tp.values()), "median": float(np.median(list(tp.values()))), "max": max(tp.values()), "patients_with_one_target": int(sum(1 for v in tp.values() if v == 1))},
            "slots": len(donors), "distinct_donor_images": len({d for d, _ in donors}), "distinct_donor_patients": len(per_patient), "slots_per_donor_patient": {"max": max(per_patient.values()), "min": min(per_patient.values())},
            "donor_patients_by_number_of_slots": dict(Counter(per_patient.values())), "donor_patients_serving_two_targets": int(sum(1 for v in per_patient.values() if v > 1)), "donors_per_target": sorted(set(len(meta[k]["donors"]) for k in kept)),
            "drops_total_by_type": dict(drops), "attempts_per_kept_target": {"min": min(att), "median": float(np.median(att)), "max": max(att), "sum": int(sum(att))},
            "targets_with_drain_label_elsewhere": int(sum(1 for k in kept if meta[k]["drain_labelled_elsewhere"])), "donor_stats_recorded": inp["donor_stats"],
            "placement_side_and_boundary_distance": "not recorded in the gate metadata; they live in the persisted composite arrays (images), which this analysis does not open",
            "manifest_sha256_of_file": __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest()}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    inp = json.loads(INPUTS.read_text())
    assignment = inp["assignment"]
    out = {"status": "DESCRIPTIVE, UNREGISTERED; recorded gate outputs only; no gate image read", "resamples": B_RESAMPLES, "seed": 0, "m": M,
           "1.1_planted_minus_zero": {arch: planted_minus_zero(arch) for arch in ARCH_GLOB},
           "1.2_donor_dependence": {arch: donor_dependence(arch, assignment) for arch in ARCH_GLOB},
           "1.3_recorded_metadata": recorded_metadata(assignment)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    for arch in ARCH_GLOB:
        print(arch, "planted minus zero:", {s: (round(v["mean"], 4), [round(x, 4) for x in v["ci95"]]) for s, v in out["1.1_planted_minus_zero"][arch]["minus_zero"].items()})
        d = out["1.2_donor_dependence"][arch]
        print(arch, "donor dependence:", round(d["mean"], 4), "nested", [round(x, 4) for x in d["interval_nested_registered"]["ci95"]], "target-boot", [round(x, 4) for x in d["interval_target_bootstrap_slots_fixed"]["ci95"]],
              "two-way", [round(x, 4) for x in d["interval_two_way_normal_95"]], "identity", d["variances"]["V_donor_equals_intersection"])

if __name__ == "__main__":
    main()
