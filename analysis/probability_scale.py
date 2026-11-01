# Summarises each arm on the probability scale and counts threshold crossings on the exploratory development-half targets.
# Run: python3 analysis/probability_scale.py [--out review/edit_artifacts/probability_scale.json]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from analysis import patient_bootstrap as BT

SEED_CI = 92001
SCORES = {"densenet121": REPO / "review/positive_audit/run/score_dn_p0_originals/scores_densenet121.csv", "resnet50": REPO / "review/positive_audit/run/score_rn_originals/scores_resnet50.csv"}
THRESHOLDS = {a: REPO / f"review/population_assignment/outputs/assigned_{a}.json" for a in SCORES}
RECORDED = REPO / "review/positive_audit/positive_analysis.json"
ARMS = ["tube", "tube_tissue", "disk", "disk_tissue"]
N_TARGETS, N_PATIENTS, N_PAIRS = 77, 39, 385

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))

def arm_arrays(s: pd.DataFrame) -> dict:
    lt = {k: s[[f"logit_{k}_{d}" for d in range(8)]].to_numpy(dtype=np.float64) for k in ("tube_tissue", "disk_tissue")}
    lg = {"before": s["logit_before"].to_numpy(dtype=np.float64), "tube": s["logit_tube"].to_numpy(dtype=np.float64), "disk": s["logit_disk"].to_numpy(dtype=np.float64)}
    return {"p_before": sigmoid(lg["before"]), "p_tube": sigmoid(lg["tube"]), "p_disk": sigmoid(lg["disk"]),
            "p_donor": {k: sigmoid(v) for k, v in lt.items()}, "logit": lg, "logit_donor": lt}

def changes(a: dict) -> dict:
    pb = a["p_before"]
    return {"tube": a["p_tube"] - pb, "tube_tissue": a["p_donor"]["tube_tissue"].mean(axis=1) - pb, "disk": a["p_disk"] - pb, "disk_tissue": a["p_donor"]["disk_tissue"].mean(axis=1) - pb}

def distribution(v: np.ndarray, inverse: np.ndarray, w: np.ndarray) -> dict:
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {**BT.mean_with_interval(v, inverse, w), "median": float(med), "q25": float(q25), "q75": float(q75)}

def crossing_shares(a: dict, tau: np.ndarray, inverse: np.ndarray, w: np.ndarray) -> dict:
    out = {"above_threshold_rate_unedited": BT.mean_with_interval((a["p_before"] >= tau).astype(float), inverse, w)}
    below = a["p_before"] < tau
    for arm in ARMS:
        p = a["p_donor"][arm] if arm.endswith("_tissue") else a[f"p_{arm}"][:, None]
        above = (p >= tau[:, None])
        out[arm] = {"above_threshold_rate": BT.mean_with_interval(above.mean(axis=1), inverse, w),
                    "up_crossing": BT.mean_with_interval((below[:, None] & above).mean(axis=1), inverse, w),
                    "down_crossing": BT.mean_with_interval(((~below)[:, None] & ~above).mean(axis=1), inverse, w)}
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description='the probability-scale summary and the threshold-crossing shares for the 77 exploratory development-half targets (review/edit_artifacts/plan.md, section 3).')
    ap.add_argument("--out", default=str(REPO / "review/edit_artifacts/probability_scale.json"))
    args = ap.parse_args()
    from cxr_shortcuts.eval.population_log import ids_digest
    recorded = json.loads(RECORDED.read_text())
    ids_file = REPO / "review/positive_audit/populations/dev_half_targets_images.ids.txt"
    committed_ids = [ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()]
    res = {"status": "EXPLORATORY and DESCRIPTIVE: 77 development-half targets, five original checkpoints per architecture, one placement and eight donors per target (the positive-audit stage design); probabilities are the "
                     "sigmoid of the raw logit and are uncalibrated; the thresholds are data-derived operating points of the never-selected validation half, not clinical operating points",
           "bootstrap": {"B": BT.B, "seed": SEED_CI, "unit": "patient"}, "architectures": {}}
    for arch, path in SCORES.items():
        s = pd.read_csv(path)
        s = s[s.stratum == "dev_half"].reset_index(drop=True)
        assert s.image_id.nunique() == N_TARGETS and s.patient_id.nunique() == N_PATIENTS and len(s) == N_PAIRS, f"{arch}: {s.image_id.nunique()} targets, {s.patient_id.nunique()} patients, {len(s)} pairs"
        assert ids_digest(s.image_id.unique()) == ids_digest(committed_ids), f"{arch}: the scored ids are not the committed development-half list"
        a = arm_arrays(s)

        rec = recorded[arch]["by_stratum"]["dev_half"]
        lc = {"tube_rendered": a["logit"]["tube"] - a["logit"]["before"], "tube_tissue": a["logit_donor"]["tube_tissue"].mean(axis=1) - a["logit"]["before"],
              "disk_rendered": a["logit"]["disk"] - a["logit"]["before"], "disk_tissue": a["logit_donor"]["disk_tissue"].mean(axis=1) - a["logit"]["before"]}
        lc["primary"] = a["logit"]["tube"] - a["logit_donor"]["tube_tissue"].mean(axis=1)
        for k, v in lc.items():
            assert abs(float(v.mean()) - rec[k]["mean"]) < 1e-9, f"{arch} {k}: recomputed {v.mean()} != recorded {rec[k]['mean']}"
        _, inverse = BT.patient_index(s["patient_id"].to_numpy())
        w = BT.draw(int(inverse.max()) + 1, SEED_CI)
        ch = changes(a)
        prim = a["p_tube"] - a["p_donor"]["tube_tissue"].mean(axis=1)
        out = {"n_pairs": len(s), "n_targets": int(s.image_id.nunique()), "n_patients": int(s.patient_id.nunique()),
               "wiring_check_logit_means_match_recorded": True,
               "mean_p_before": float(a["p_before"].mean()), "probability_change": {arm: distribution(ch[arm], inverse, w) for arm in ARMS},
               "logit_change": {"tube": distribution(lc["tube_rendered"], inverse, w), "tube_tissue": distribution(lc["tube_tissue"], inverse, w), "disk": distribution(lc["disk_rendered"], inverse, w),
                                "disk_tissue": distribution(lc["disk_tissue"], inverse, w), "primary": distribution(lc["primary"], inverse, w)},
               "primary_contrast_probability": distribution(prim, inverse, w), "crossings": {}}
        thr = json.loads(THRESHOLDS[arch].read_text())["per_seed"]
        ck_seed = s["label"].str.extract(r"seed(\d+)$")[0].astype(int).to_numpy()
        for name, getter in (("sens090", lambda k: thr[str(k)]["band"]["0.9"]["threshold"]), ("youden", lambda k: thr[str(k)]["youden"]["threshold"])):
            tau = np.array([getter(k) for k in ck_seed])
            per_seed = {str(k): float(getter(k)) for k in range(5)}
            out["crossings"][name] = {"threshold_per_seed": per_seed, "threshold_range": {"min": min(per_seed.values()), "max": max(per_seed.values())}, **crossing_shares(a, tau, inverse, w)}
        res["architectures"][arch] = out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    for arch, o in res["architectures"].items():
        print(f"{arch}: {o['n_pairs']} pairs ({o['n_targets']} targets, {o['n_patients']} patients); mean p_before {o['mean_p_before']:.4f}")
        for arm in ARMS:
            r = o["probability_change"][arm]
            print(f"  {arm:12s} change mean {r['mean']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] median {r['median']:+.4f} IQR [{r['q25']:+.4f},{r['q75']:+.4f}]")
        r = o["primary_contrast_probability"]
        print(f"  primary      change mean {r['mean']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] median {r['median']:+.4f} IQR [{r['q25']:+.4f},{r['q75']:+.4f}]")
        for name in ("sens090", "youden"):
            c = o["crossings"][name]
            print(f"  {name}: unedited above-threshold {c['above_threshold_rate_unedited']['mean']:.3f}; " + "; ".join(
                f"{arm} above {c[arm]['above_threshold_rate']['mean']:.3f} up {c[arm]['up_crossing']['mean']:.3f} down {c[arm]['down_crossing']['mean']:.3f}" for arm in ARMS))

if __name__ == "__main__":
    main()
