# Computes the pooled contrasts, intervals, variance components and the registered verdict rule for the exploratory positive targets.
# Run: python3 analysis/positive_analysis.py --densenet scores_densenet121.csv --resnet scores_resnet50.csv --inputs exploratory_inputs.json --out out.json

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 8
MARGIN = {"densenet121": 0.156, "resnet50": 0.1823}
INVERT = {"densenet121": 0.73, "resnet50": 0.8418}
CONTRASTS = ("tube_rendered", "tube_tissue", "disk_rendered", "disk_tissue", "primary", "content_on_disks", "shape_main", "interaction", "content_main")
TT = [f"logit_tube_tissue_{k}" for k in range(K)]
DT = [f"logit_disk_tissue_{k}" for k in range(K)]

def arrays(df: pd.DataFrame, labels: list | None = None):
    labels = labels or sorted(df.label.unique())
    sub = df[df.label.isin(labels)]
    ids = sorted(sub.image_id.unique())
    per = {}
    for lab in labels:
        d = sub[sub.label == lab].set_index("image_id").loc[ids]
        base = d["logit_before"].to_numpy()
        per[lab] = (d["logit_tube"].to_numpy() - base, d["logit_disk"].to_numpy() - base,
                    d[TT].to_numpy() - base[:, None], d[DT].to_numpy() - base[:, None])
    r = np.mean([v[0] for v in per.values()], axis=0)
    b = np.mean([v[1] for v in per.values()], axis=0)
    tt = np.mean([v[2] for v in per.values()], axis=0)
    dt = np.mean([v[3] for v in per.values()], axis=0)
    meta = sub.drop_duplicates("image_id").set_index("image_id").loc[ids]
    return ids, meta["patient_id"].astype(str).to_numpy(), meta["stratum"].to_numpy(), r, b, tt, dt

def contrast_values(r, b, tt_mean, dt_mean) -> dict:
    primary = r - tt_mean
    on_disks = b - dt_mean
    return {"tube_rendered": r, "tube_tissue": tt_mean, "disk_rendered": b, "disk_tissue": dt_mean, "primary": primary,
            "content_on_disks": on_disks, "shape_main": ((r - b) + (tt_mean - dt_mean)) / 2, "interaction": primary - on_disks,
            "content_main": (primary + on_disks) / 2}

def nested_bootstrap(patients, r, b, tt, dt, n_resamples: int = 2000, seed: int = 0) -> dict:
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(seed)
    k = tt.shape[1]
    point = {c: float(v.mean()) for c, v in contrast_values(r, b, tt.mean(1), dt.mean(1)).items()}
    reps = {c: np.empty(n_resamples) for c in CONTRASTS}
    for i in range(n_resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        d = rng.integers(0, k, size=(len(idx), k))
        ttm = np.take_along_axis(tt[idx], d, axis=1).mean(1)
        dtm = np.take_along_axis(dt[idx], d, axis=1).mean(1)
        for c, v in contrast_values(r[idx], b[idx], ttm, dtm).items():
            reps[c][i] = v.mean()
    return {c: {"mean": point[c], "ci90": [float(np.percentile(reps[c], 5)), float(np.percentile(reps[c], 95))],
                "ci95": [float(np.percentile(reps[c], 2.5)), float(np.percentile(reps[c], 97.5))]} for c in CONTRASTS} | \
           {"_n_targets": int(len(r)), "_n_patients": int(len(uniq))}

def verdict(primary: dict, margin: float, invert: float) -> dict:
    lo90, hi90 = primary["ci90"]
    lo95, hi95 = primary["ci95"]
    equivalent = bool(lo90 > -margin and hi90 < margin)
    effect = bool(lo95 > 0 or hi95 < 0)
    label = "EQUIVALENT" if equivalent else ("EFFECT" if effect else "INCONCLUSIVE")
    return {"label": label, "equivalent_90_inside_margin": equivalent, "effect_95_excludes_zero": effect,
            "effect_inside_margin_flag": bool(equivalent and effect), "margin": margin, "mean": primary["mean"],
            "mean_over_invert_constant": primary["mean"] / invert, "invert_constant": invert}

def variance_components(patients, r, tt) -> dict:
    y = r - tt.mean(1)
    df = pd.DataFrame({"y": y, "p": patients})
    g = df.groupby("p")["y"]
    n, P = len(df), g.ngroups
    sizes = g.size()
    n0 = (n - (sizes ** 2).sum() / n) / (P - 1)
    msb = float((sizes * (g.mean() - df["y"].mean()) ** 2).sum() / (P - 1))
    msw = float(((df["y"] - g.transform("mean")) ** 2).sum() / (n - P))
    sw2 = msw
    sa2 = max(0.0, (msb - msw) / n0)
    z = r[:, None] - tt
    sd2 = float(z.var(axis=1, ddof=1).mean())
    return {"n_targets": n, "n_patients": P, "n0": float(n0), "MSB": msb, "MSW": msw, "sigma_a2": sa2, "sigma_w2": sw2,
            "donor_var_single": sd2, "donor_part_of_sigma_w2": sd2 / tt.shape[1], "residual_placement_part": sw2 - sd2 / tt.shape[1]}

def checkpoint_component(df: pd.DataFrame, labels: list) -> dict:
    means = []
    for lab in labels:
        _, _, _, r, _, tt, _ = arrays(df, [lab])
        means.append(float((r - tt.mean(1)).mean()))
    return {"per_checkpoint_primary_means": means, "variance_across_checkpoints": float(np.var(means, ddof=1)) if len(means) > 1 else None}

def sig(z):
    return 1 / (1 + np.exp(-z))

def margin_sensitivity(df: pd.DataFrame, labels: list, target: float = 0.02) -> dict:
    from scipy.optimize import brentq
    out = {}
    for lab in labels:
        z = df[df.label == lab].drop_duplicates("image_id")["logit_before"].to_numpy()
        f = lambda s, sign: float(np.mean(sig(z + s) - sig(z))) - sign * target
        up, dn = brentq(f, 0, 50, args=(1,), xtol=1e-10), brentq(f, -50, 0, args=(-1,), xtol=1e-10)
        out[lab] = {"up": up, "down": abs(dn), "tighter": min(up, abs(dn)), "mean_baseline_p": float(sig(z).mean())}
    out["_mean_tighter"] = float(np.mean([v["tighter"] for k, v in out.items() if not k.startswith("_")]))
    return out

def analyse_arch(arch: str, df: pd.DataFrame, flagged: set) -> dict:
    orig = sorted(df[df.family == "original"].label.unique())
    res = {"arch": arch, "original_checkpoints": orig}
    if not orig:
        return res
    o = df[df.family == "original"]
    ids, pat, strat, r, b, tt, dt = arrays(o, orig)
    res["pooled"] = nested_bootstrap(pat, r, b, tt, dt)
    res["verdict"] = verdict(res["pooled"]["primary"], MARGIN[arch], INVERT[arch])
    src = np.where(strat == "dev_half", "dev_half", "R_int")
    res["by_source"], res["by_stratum"] = {}, {}
    for name, sel in (("dev_half", src == "dev_half"), ("R_int", src == "R_int")):
        res["by_source"][name] = nested_bootstrap(pat[sel], r[sel], b[sel], tt[sel], dt[sel])
    for s in sorted(set(strat)):
        sel = strat == s
        res["by_stratum"][s] = nested_bootstrap(pat[sel], r[sel], b[sel], tt[sel], dt[sel])
    fl = np.array([i in flagged for i in ids])
    res["drain_labelled_elsewhere"] = {name: nested_bootstrap(pat[m], r[m], b[m], tt[m], dt[m]) for name, m in (("flagged", fl), ("not_flagged", ~fl)) if m.sum() > 1}
    res["variance_components"] = variance_components(pat, r, tt)
    res["variance_components_dev_half_only"] = variance_components(pat[src == "dev_half"], r[src == "dev_half"], tt[src == "dev_half"])
    res["checkpoint_component"] = checkpoint_component(o, orig)
    cm = res["checkpoint_component"]["per_checkpoint_primary_means"]
    m, s, n = float(np.mean(cm)), float(np.std(cm, ddof=1)), len(cm)
    from scipy.stats import t as tdist
    hw = float(tdist.ppf(0.975, n - 1) * s / np.sqrt(n))
    res["recipe_level_secondary"] = {"mean": m, "t95": [m - hw, m + hw], "n_checkpoints": n, "note": "five-checkpoint limit; the between-checkpoint variance is not in the primary interval"}
    res["per_checkpoint_descriptive"] = {}
    for lab in orig:
        i2, p2, _, r2, b2, tt2, dt2 = arrays(o, [lab])
        res["per_checkpoint_descriptive"][lab] = nested_bootstrap(p2, r2, b2, tt2, dt2, n_resamples=500)
    res["margin_sensitivity"] = margin_sensitivity(o, orig)
    planted = df[df.family != "original"]
    res["planted_descriptive"] = {}
    for fam in sorted(planted.family.unique()):
        fdf = planted[planted.family == fam]
        res["planted_descriptive"][fam] = {}
        for strength in sorted({lab.split("_seed")[0] for lab in fdf.label.unique()}):
            labs = sorted(l for l in fdf.label.unique() if l.split("_seed")[0] == strength)
            i2, p2, s2, r2, b2, tt2, dt2 = arrays(fdf, labs)
            res["planted_descriptive"][fam][strength] = {"labels": labs, "boot": nested_bootstrap(p2, r2, b2, tt2, dt2, n_resamples=1000)}
    return res

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--densenet", type=Path, nargs="+", required=True)
    ap.add_argument("--resnet", type=Path, nargs="+", default=None)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.inputs.read_text())
    flagged = {i for s in spec["strata"].values() for i in s["targets_drain_labelled_elsewhere"]}
    out = {}
    for arch, path in (("densenet121", a.densenet), ("resnet50", a.resnet)):
        if not path:
            continue
        out[arch] = analyse_arch(arch, pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in path], ignore_index=True), flagged)
    a.out.write_text(json.dumps(out, indent=1))
    for arch, r in out.items():
        if "verdict" in r:
            print(arch, r["verdict"]["label"], {k: round(v["mean"], 4) for k, v in r["pooled"].items() if not k.startswith("_")})

if __name__ == "__main__":
    main()
