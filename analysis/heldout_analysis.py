# Analyses the held-out run: pooled contrasts, two-stage bootstrap intervals, the margin and the registered verdict rule. Reads score CSVs only.
# Run: python3 analysis/heldout_analysis.py --arch densenet121 --scores a.csv b.csv --inputs inputs.json --out out.json [--comparator-scores dev.csv --comparator-inputs dev_inputs.json]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))

M = 8
MARGIN = {"densenet121": 0.156, "resnet50": 0.1823}
INVERT = {"densenet121": 0.73, "resnet50": 0.8418}
CONTRASTS = ("tube_rendered", "tube_tissue", "disk_rendered", "disk_tissue", "primary", "tissue_shape", "rendered_shape", "content_on_disks", "interaction")
B_RESAMPLES = 2000

def contrast_arrays(r, t, b, d) -> dict:
    return {"tube_rendered": r, "tube_tissue": t, "disk_rendered": b, "disk_tissue": d, "primary": r - t, "tissue_shape": t - d, "rendered_shape": r - b,
            "content_on_disks": b - d, "interaction": (r - t) - (b - d)}

def effects(df: pd.DataFrame, labels: list, m: int = M):
    ids = sorted(df[df.label == labels[0]].image_id)
    per = {k: [] for k in "rtbd"}
    for lab in labels:
        sub = df[df.label == lab].set_index("image_id").loc[ids]
        base = sub["logit_before"].to_numpy()[:, None]
        per["r"].append(np.stack([sub[f"logit_tube_{k}"].to_numpy() for k in range(m)], axis=1) - base)
        per["b"].append(np.stack([sub[f"logit_disk_{k}"].to_numpy() for k in range(m)], axis=1) - base)
        per["t"].append(np.stack([sub[f"logit_tube_tissue_{k}"].to_numpy() for k in range(m)], axis=1) - base)
        per["d"].append(np.stack([sub[f"logit_disk_tissue_{k}"].to_numpy() for k in range(m)], axis=1) - base)
    meta = df[df.label == labels[0]].set_index("image_id").loc[ids]
    stack = lambda k: np.stack(per[k], axis=1)
    return ids, meta["patient_id"].astype(str).to_numpy(), meta["view_position"].to_numpy(), stack("r"), stack("t"), stack("b"), stack("d")

def bootstrap(patients, vals: dict, n_resamples: int = B_RESAMPLES, seed: int = 0, keep_reps: bool = False) -> dict:
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(seed)
    m = next(iter(vals.values())).shape[1]
    reps = {c: np.empty(n_resamples) for c in vals}
    for i in range(n_resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        pick = rng.integers(0, m, size=(len(idx), m))
        for c, v in vals.items():
            reps[c][i] = np.take_along_axis(v[idx], pick, axis=1).mean(1).mean()
    out = {c: {"mean": float(v.mean()), "ci90": [float(np.percentile(reps[c], 5)), float(np.percentile(reps[c], 95))],
               "ci95": [float(np.percentile(reps[c], 2.5)), float(np.percentile(reps[c], 97.5))]} for c, v in vals.items()}
    out["_n_targets"], out["_n_patients"] = int(len(patients)), int(len(uniq))
    return (out, reps) if keep_reps else out

def verdict(primary: dict, margin: float, invert: float) -> dict:
    lo90, hi90 = primary["ci90"]
    lo95, hi95 = primary["ci95"]
    equivalent = bool(lo90 > -margin and hi90 < margin)
    effect = bool(lo95 > 0 or hi95 < 0)
    label = "EQUIVALENT" if equivalent else ("EFFECT" if effect else "INCONCLUSIVE")
    sign = None if not effect else ("positive (reliance-consistent)" if lo95 > 0 else "negative (not reliance-consistent; an effect, not a null)")
    return {"label": label, "equivalent_90_inside_margin": equivalent, "effect_95_excludes_zero": effect, "effect_inside_margin_flag": bool(equivalent and effect),
            "effect_sign": sign, "margin": margin, "mean": primary["mean"], "ci95_upper_end": hi95, "mean_over_invert_constant": primary["mean"] / invert, "invert_constant": invert}

def variance_components(patients, y: np.ndarray, per_placement: np.ndarray) -> dict:
    df = pd.DataFrame({"y": y, "p": patients})
    g = df.groupby("p")["y"]
    n, P = len(df), g.ngroups
    sizes = g.size()
    n0 = (n - (sizes ** 2).sum() / n) / (P - 1)
    msb = float((sizes * (g.mean() - df["y"].mean()) ** 2).sum() / (P - 1))
    msw = float(((df["y"] - g.transform("mean")) ** 2).sum() / (n - P))
    pv = float(per_placement.var(axis=1, ddof=1).mean())
    return {"n_targets": n, "n_patients": P, "MSB": msb, "MSW": msw, "sigma_a2": max(0.0, (msb - msw) / n0), "sigma_w2": msw,
            "single_placement_variance": pv, "placement_part_of_sigma_w2": pv / per_placement.shape[1]}

def analyze_set(df: pd.DataFrame, labels: list, arch: str, with_reps: bool = False, seed: int = 0):
    ids, patients, views, R, T, B, D = effects(df, labels)
    arrays = contrast_arrays(R, T, B, D)
    avg = {c: a.mean(axis=1) for c, a in arrays.items()}
    res = bootstrap(patients, avg, seed=seed, keep_reps=with_reps)
    boot, reps = res if with_reps else (res, None)
    out = {"pooled": boot, "labels": labels}
    out["verdict"] = verdict(boot["primary"], MARGIN[arch], INVERT[arch])
    out["variance_components"] = variance_components(patients, avg["primary"].mean(1), avg["primary"])
    per_ck = {}
    for j, lab in enumerate(labels):
        per_ck[lab] = bootstrap(patients, {c: a[:, j, :] for c, a in arrays.items()})
    out["per_checkpoint_descriptive"] = per_ck
    means = np.array([[a[:, j, :].mean() for j in range(len(labels))] for c, a in arrays.items() if c == "primary"][0])
    if len(labels) > 1:
        from scipy.stats import t as tdist
        se = means.std(ddof=1) / np.sqrt(len(means))
        h = float(tdist.ppf(0.975, len(means) - 1) * se)
        out["recipe_level_secondary"] = {"mean": float(means.mean()), "t95": [float(means.mean() - h), float(means.mean() + h)], "n_checkpoints": len(means),
                                         "variance_across_checkpoints": float(means.var(ddof=1)), "note": "five-checkpoint limit; the between-checkpoint variance is not in the primary interval"}
    out["_meta"] = {"ids": ids, "patients": patients, "views": views}
    return (out, reps) if with_reps else out

def strata_results(df: pd.DataFrame, labels: list, arch: str, flag_ids: set) -> dict:
    ids, patients, views, R, T, B, D = effects(df, labels)
    avg = {c: a.mean(axis=1) for c, a in contrast_arrays(R, T, B, D).items()}
    out = {}
    ids = np.array(ids)
    masks = {"drain_labelled_elsewhere": np.isin(ids, list(flag_ids)), "not_drain_labelled_elsewhere": ~np.isin(ids, list(flag_ids)),
             "view_AP": views == "AP", "view_PA": views == "PA"}
    for name, mk in masks.items():
        if mk.sum() < 2 or len(set(patients[mk])) < 2:
            out[name] = {"n_targets": int(mk.sum()), "note": "too few clusters for an interval"}
            continue
        out[name] = bootstrap(patients[mk], {c: v[mk] for c, v in avg.items()})
    return out

def planted_curve(df: pd.DataFrame, arch: str, manip_strong_passes) -> dict:
    labs = sorted(df[df.family != "original"].label.unique())
    strengths = sorted({l.rsplit("_seed", 1)[0] for l in labs})
    out = {"strengths": {}, "positive_control": None}
    for s in strengths:
        L = [l for l in labs if l.rsplit("_seed", 1)[0] == s]
        ids, patients, views, R, T, B, D = effects(df, L)
        avg = {c: a.mean(axis=1) for c, a in contrast_arrays(R, T, B, D).items()}
        boot = bootstrap(patients, {"primary": avg["primary"], "tube_rendered": avg["tube_rendered"], "tube_tissue": avg["tube_tissue"]})
        seed_means = {l: float(contrast_arrays(*[x[:, [i], :] for x in (R, T, B, D)])["primary"].mean()) for i, l in enumerate(L)}
        out["strengths"][s] = {"labels": L, "primary": boot["primary"], "tube_rendered": boot["tube_rendered"], "tube_tissue": boot["tube_tissue"], "per_seed_primary_mean": seed_means}
    if "strong" in out["strengths"]:
        s = out["strengths"]["strong"]
        reaches = bool(s["primary"]["mean"] >= INVERT[arch] and s["primary"]["ci95"][0] > 0 and all(v > 0 for v in s["per_seed_primary_mean"].values()))
        has = reaches if arch != "resnet50" else bool(reaches and manip_strong_passes)
        out["positive_control"] = {"strong_reaches_invert_with_ci_and_all_seeds_positive": reaches, "manipulation_check_passed_at_strong_part4": manip_strong_passes if arch == "resnet50" else "n/a (DenseNet-121: passed in the manipulation-check stage and 65)",
                                   "confirmatory_population_has_a_positive_control": has, "invert_constant": INVERT[arch]}
    return out

def difference_vs(gate_reps: np.ndarray, gate_mean: float, other_reps: np.ndarray, other_mean: float) -> dict:
    d = gate_reps - other_reps
    return {"gate_minus_other": gate_mean - other_mean, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))]}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the half's inputs.json (drain-labelled-elsewhere list, realized counts)")
    ap.add_argument("--comparator-scores", type=Path, nargs="*", default=[])
    ap.add_argument("--manip-json", type=Path, default=None, help="Part 4's ResNet-50 readout JSON (manipulation_check_passes_at_strong)")
    ap.add_argument("--positive_audit-json", type=Path, default=SCRIPTS_DIR.parent / "review/positive_audit/positive_analysis.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in a.scores], ignore_index=True)
    inputs = json.loads(a.inputs.read_text())
    originals = sorted(df[df.family == "original"].label.unique())
    if len(originals) != 5:
        raise SystemExit(f"expected five original checkpoints, found {originals}")
    manip = None
    if a.manip_json and a.manip_json.exists():
        manip = bool(json.loads(a.manip_json.read_text())["manipulation_check_passes_at_strong"])
    res, reps = analyze_set(df, originals, a.arch, with_reps=True)
    meta = res.pop("_meta")
    out = {"arch": a.arch, "half": inputs["half"], "protocol_commit": inputs.get("protocol_commit"), "originals": res,
           "realized": {"n_targets_kept": len(meta["ids"]), "n_patients": int(len(set(meta["patients"]))), "projected_in_protocol": {"images": 157, "patients": 57},
                        "view_counts": {v: int((meta["views"] == v).sum()) for v in sorted(set(meta["views"]))}},
           "strata": strata_results(df, originals, a.arch, set(inputs["targets_drain_labelled_elsewhere"])),
           "planted": planted_curve(df, a.arch, manip),
           "donor_stats": inputs.get("donor_stats")}
    ok = df.groupby("label").image_id.nunique()
    out["targets_scored_per_checkpoint"] = {k: int(v) for k, v in ok.items()}
    try:
        from analysis.positive_analysis import margin_sensitivity
        out["margin_sensitivity_recomputed_on_gate_positives_baselines"] = margin_sensitivity(df, originals)
    except Exception as e:
        out["margin_sensitivity_recomputed_on_gate_positives_baselines"] = {"error": str(e)}
    if a.comparator_scores:
        cdf = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in a.comparator_scores], ignore_index=True)
        cres, creps = analyze_set(cdf, sorted(cdf[cdf.family == "original"].label.unique()), a.arch, with_reps=True, seed=1)
        cres.pop("_meta")
        out["dev_same_design_comparator"] = {"originals": cres, "gate_minus_dev_same_design_primary": difference_vs(reps["primary"], res["pooled"]["primary"]["mean"], creps["primary"], cres["pooled"]["primary"]["mean"]),
                                             "note": "difference between independent bootstrap runs (gate seed 0, dev seed 1) on different data; same design, different half"}
    p71 = json.loads(a.positive_audit_json.read_text())[a.arch]["by_source"]
    cmp71 = {}
    for src in ("dev_half", "R_int"):
        prim = p71[src]["primary"]
        se = (prim["ci95"][1] - prim["ci95"][0]) / (2 * 1.96)
        rng = np.random.default_rng(1)
        nrm = rng.normal(prim["mean"], se, size=len(reps["primary"]))
        cmp71[src] = {"positive_audit_estimate": prim["mean"], "positive_audit_ci95": prim["ci95"], "positive_audit_verdict_90_inside_margin": bool(prim["ci90"][0] > -MARGIN[a.arch] and prim["ci90"][1] < MARGIN[a.arch]),
                      **difference_vs(reps["primary"], res["pooled"]["primary"]["mean"], nrm, prim["mean"]),
                      "note": "the positive-audit stage's design differs from this protocol's (M, view matching, donors per placement): the difference is confounded by design; the interval uses a normal approximation to the positive-audit stage's bootstrap interval"}
    out["beside_positive_audit_exploratory"] = cmp71
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))
    v, p = out["originals"]["verdict"], out["originals"]["pooled"]["primary"]
    print(f"{a.arch} half={inputs['half']} n={out['realized']['n_targets_kept']} primary {p['mean']:+.4f} 90% ({p['ci90'][0]:+.4f},{p['ci90'][1]:+.4f}) 95% ({p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}) -> {v['label']}"
          f"{' [EFFECT inside the margin]' if v['effect_inside_margin_flag'] else ''}; sign {v['effect_sign']}")

if __name__ == "__main__":
    main()
