# Reads out the 512-pixel replication (pooled contrasts, verdicts, the two-by-two and the 224-pixel micro-batching control).
# Run: python3 analysis/px512_readout.py --scores-dn CSV --scores-rn CSV --validity J --margins J --in-hand J --out J [--b 2000]

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
from analysis import positive_analysis as P71
from analysis import further_run_readout as R79

LABEL_RE = re.compile(r"^(densenet121|resnet50)_(orig|zero|strong)_seed(\d+)$")
K = R79.K
B_DEFAULT = R79.B_DEFAULT

def classify(label: str) -> tuple[str, str, int]:
    m = LABEL_RE.match(label)
    if not m:
        raise ValueError(f"unrecognised E5 checkpoint label {label!r}")
    return m.group(1), m.group(2), int(m.group(3))

def summarize(arch: str, point: float, reps: np.ndarray, margin_512: float, invert_512: float, n_positive: int | None = None, n_ckpt: int | None = None) -> dict:
    s = {"mean": point, **R79.interval(reps)}
    v512 = P71.verdict(s, margin_512, invert_512)
    v224 = P71.verdict(s, P71.MARGIN[arch], P71.INVERT[arch])
    out = {**s, "sign": "negative" if point < 0 else "positive", "margin_512": margin_512, "verdict_512_margin": v512["label"], "equivalent_90_inside_512_margin": v512["equivalent_90_inside_margin"],
           "effect_95_excludes_zero": v512["effect_95_excludes_zero"], "effect_inside_margin_flag": v512["effect_inside_margin_flag"], "margin_224": P71.MARGIN[arch], "verdict_224_margin": v224["label"],
           "verdicts_differ_between_margins": bool(v512["label"] != v224["label"])}
    if n_ckpt is not None:
        out["invert_met_512"] = bool(point >= invert_512 and out["ci95"][0] > 0 and n_positive is not None and n_positive >= 4 and n_ckpt == 5)
    return out

def recipe_level(values: list[float], margin: float) -> dict:
    v = np.asarray(values, dtype=float)
    n, mean, sd = len(v), float(v.mean()), float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    out = {"n": n, "mean": mean, "sd": sd, "df": n - 1}
    for lvl, name in ((0.90, "ci90"), (0.95, "ci95")):
        t = float(stats.t.ppf(0.5 + lvl / 2, n - 1))
        out[name] = [mean - t * se, mean + t * se]
    out["ci90_inside_margin"] = bool(out["ci90"][0] > -margin and out["ci90"][1] < margin)
    out["ci95_excludes_zero"] = bool(out["ci95"][0] > 0 or out["ci95"][1] < 0)
    return out

def two_by_two(df: pd.DataFrame, labels: list[str], patients: np.ndarray, ids: list[str], b: int = B_DEFAULT, seed: int = 0) -> dict:
    disk_t = [f"logit_disk_tissue_{k}" for k in range(K)]
    cells = {"R_t": [], "T_t": [], "R_d": [], "T_d": []}
    for lab in labels:
        sub = df[df.label == lab].set_index("image_id").loc[ids]
        base = sub["logit_before"].to_numpy()
        cells["R_t"].append(sub["logit_tube"].to_numpy() - base)
        cells["R_d"].append(sub["logit_disk"].to_numpy() - base)
        cells["T_t"].append(sub[P71.TT].to_numpy() - base[:, None])
        cells["T_d"].append(sub[disk_t].to_numpy() - base[:, None])
    Rt, Rd = np.mean(cells["R_t"], axis=0), np.mean(cells["R_d"], axis=0)
    Tt, Td = np.mean(cells["T_t"], axis=0), np.mean(cells["T_d"], axis=0)

    def contrasts(rt, rd, tt_mean, td_mean):
        return {"content_main": float((((rt - tt_mean) + (rd - td_mean)) / 2).mean()), "shape_main": float((((rt - rd) + (tt_mean - td_mean)) / 2).mean()), "interaction": float(((rt - tt_mean) - (rd - td_mean)).mean())}

    point = contrasts(Rt, Rd, Tt.mean(1), Td.mean(1))
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(b) for k in point}
    for i in range(b):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        d = rng.integers(0, K, size=(len(idx), K))
        c = contrasts(Rt[idx], Rd[idx], np.take_along_axis(Tt[idx], d, axis=1).mean(1), np.take_along_axis(Td[idx], d, axis=1).mean(1))
        for k in c:
            reps[k][i] = c[k]
    return {k: {"mean": point[k], **R79.interval(reps[k])} for k in point}

def compare_224(arch: str, orig_valid_prim: dict, set_points: dict, in_hand: dict) -> dict:
    h = in_hand[arch]
    matched = {int(k): v for k, v in h["e4_matched_seeds_5_to_9"]["per_checkpoint"].items()}
    diffs = {s: orig_valid_prim[s] - matched[s] for s in sorted(orig_valid_prim) if s in matched}
    out = {"matched_seed_differences_512_minus_224_e4": diffs, "e4_matched_mean_224": h["e4_matched_seeds_5_to_9"]["mean"], "detectable_difference_95": h["e4_matched_seeds_5_to_9"]["detectable_difference_of_two_five_checkpoint_means_95pct"]}
    if len(diffs) >= 2:
        out["matched_difference_t_interval"] = recipe_level(list(diffs.values()), h["margin_224"])
    if "orig" in set_points:
        out["family_mean_512"] = set_points["orig"]
        out["minus_e4_ten_mean"] = set_points["orig"] - h["e4_seeds_5_to_14_selected_on_S_str"]["mean"]
        out["minus_five_224_originals_selected_on_S_int_mean"] = set_points["orig"] - h["originals_seeds_0_to_4_selected_on_S_int"]["mean"]
    for s in ("zero", "strong"):
        if s in set_points:
            out[f"planted_{s}_512_minus_224"] = set_points[s] - h["planted_224_seeds_0_to_2_selected_on_S_str"][s]["raw_mean"]
    return out

def control_224(arch: str, e5_prim: dict, ctl_prim: dict, in_hand: dict) -> dict:
    h = in_hand[arch]
    e4 = {int(k): v for k, v in h["e4_matched_seeds_5_to_9"]["per_checkpoint"].items()}
    seeds = sorted(set(e5_prim) & set(ctl_prim) & set(e4))
    if not seeds:
        return {"seeds": [], "note": "no seed has a valid E4, control and E5 original checkpoint"}
    per = {s: {"e4_224_batch128": e4[s], "control_224_micro32": ctl_prim[s], "e5_512_micro32": e5_prim[s], "micro_batching_effect": ctl_prim[s] - e4[s],
               "resolution_effect_at_fixed_micro_batch": e5_prim[s] - ctl_prim[s], "total_512_minus_224": e5_prim[s] - e4[s]} for s in seeds}
    keys = ("micro_batching_effect", "resolution_effect_at_fixed_micro_batch", "total_512_minus_224")
    mean = {k: float(np.mean([per[s][k] for s in seeds])) for k in keys}
    limit = float(1.96 * np.sqrt(2.0 / len(seeds)) * h["e4_matched_seeds_5_to_9"]["sd"])
    return {"seeds": seeds, "per_seed": per, "mean_over_seeds": mean, "seed_noise_limit": limit, "n_seeds": len(seeds),
            "attribution_to_resolution_allowed": bool(abs(mean["resolution_effect_at_fixed_micro_batch"]) > abs(mean["micro_batching_effect"]) and abs(mean["micro_batching_effect"]) < limit),
            "note": "descriptive, no threshold, no interval is registered for the mean over seeds; the limit is the size below which a difference of n-seed means is not distinguishable from seed noise"}

def control_block(arch: str, ctl_df: pd.DataFrame, ctl_validity: dict | None, e5_prim_valid: dict, ids_e5: list, in_hand: dict, b: int = B_DEFAULT) -> dict:
    ids, patients, per = R79.per_checkpoint(ctl_df)
    if ids != ids_e5:
        raise ValueError("the control scores do not cover exactly the targets of the E5 scores")
    prim = R79.checkpoint_primary(per)
    keep = [l for l in per if ctl_validity is None or ctl_validity["checkpoints"][f"{arch}/{l}"]["valid"]]
    sets = {l: R79.set_arrays(per, [l]) for l in keep}
    point, reps = R79.joint_bootstrap(patients, sets, b) if sets else ({}, {})
    ctl_prim = {int(l.rsplit("seed", 1)[1]): prim[l] for l in keep}
    return {"checkpoints": sorted(per), "valid_checkpoints": sorted(keep), "per_checkpoint_primary": {l: prim[l] for l in sorted(per)},
            "per_checkpoint_interval": {l: {"mean": point[l], **R79.interval(reps[l])} for l in keep}, **control_224(arch, e5_prim_valid, ctl_prim, in_hand)}

def readout_arch(arch: str, df: pd.DataFrame, validity: dict, margins: dict, in_hand: dict | None, b: int = B_DEFAULT, ctl_df: pd.DataFrame | None = None, ctl_validity: dict | None = None) -> dict:
    ids, patients, per = R79.per_checkpoint(df)
    prim = R79.checkpoint_primary(per)
    fam_labels: dict[str, list] = {"orig": [], "zero": [], "strong": []}
    for lab in per:
        a, fam, seed = classify(lab)
        assert a == arch, f"label {lab} is not an {arch} checkpoint"
        fam_labels[fam].append(lab)
    is_valid = lambda lab: bool(validity["checkpoints"][f"{arch}/{lab}"]["valid"])
    valid = {f: sorted(l for l in v if is_valid(l)) for f, v in fam_labels.items()}
    mg = margins["archs"][arch]
    if "margin_512" not in mg:
        raise SystemExit(f"{arch}: no valid original checkpoint, so no 512 margin exists; no verdict is computed")
    m512, i512 = mg["margin_512"], mg["invert_512"]
    sets, names = {}, {}
    for fam in fam_labels:
        for kind, lab in (("valid", valid[fam]), ("all", sorted(fam_labels[fam]))):
            if lab:
                sets[f"{fam}_{kind}"] = R79.set_arrays(per, lab)
                names[f"{fam}_{kind}"] = lab
    point, reps = R79.joint_bootstrap(patients, sets, b)
    out = {"arch": arch, "B": b, "seed": R79.SEED, "targets": len(ids), "patients": int(len(np.unique(patients))), "margin_512": m512, "invert_512": i512, "margins_file_pooled": mg["pooled"], "originals_valid": mg["originals_valid"],
           "sets": {}}
    for name, lab in names.items():
        fam = name.split("_")[0]
        pos = int(sum(prim[l] > 0 for l in lab)) if fam == "orig" else None
        row = {"checkpoints": lab, "per_checkpoint_primary": {l: prim[l] for l in lab}, "pooled": summarize(arch, point[name], reps[name], m512, i512, pos, len(lab) if fam == "orig" else None),
               "checkpoints_negative": int(sum(prim[l] < 0 for l in lab)), "checkpoints_above_512_margin": int(sum(prim[l] > m512 for l in lab)), "checkpoints_above_224_margin": int(sum(prim[l] > P71.MARGIN[arch] for l in lab))}
        out["sets"][name] = row
    if "orig_valid" in out["sets"] and len(valid["orig"]) >= 2:
        out["recipe_level_originals"] = {"at_512_margin": recipe_level([prim[l] for l in valid["orig"]], m512), "at_224_margin": recipe_level([prim[l] for l in valid["orig"]], P71.MARGIN[arch])}
    for fam in ("strong",):
        for kind in ("valid", "all"):
            a_, z_ = f"{fam}_{kind}", f"zero_{kind}"
            if a_ in sets and z_ in sets:
                mz = R79.minus_zero(point[a_], point[z_], reps[a_], reps[z_], m512)
                mz["margin_224"] = P71.MARGIN[arch]
                mz["relation_to_224_margin"] = R79.relation_to_margin(mz["ci95"], P71.MARGIN[arch])
                out["sets"][a_]["planted_minus_zero"] = mz
    if valid["orig"]:
        out["two_by_two_originals"] = two_by_two(df, valid["orig"], patients, ids, b)
    if in_hand is not None:
        pts = {f: point[f"{f}_valid"] for f in ("orig", "zero", "strong") if f"{f}_valid" in point}
        out["compare_224"] = compare_224(arch, {classify(l)[2]: prim[l] for l in valid["orig"]}, pts, in_hand)
        if ctl_df is not None:
            out["control_224"] = control_block(arch, ctl_df, ctl_validity, {classify(l)[2]: prim[l] for l in valid["orig"]}, ids, in_hand, b)
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dn", type=Path, required=True)
    ap.add_argument("--scores-rn", type=Path, required=True)
    ap.add_argument("--validity", type=Path, required=True)
    ap.add_argument("--margins", type=Path, required=True)
    ap.add_argument("--in-hand", type=Path)
    ap.add_argument("--scores-ctl-dn", type=Path, help="224 control score CSV, DenseNet-121 (E5 addendum)")
    ap.add_argument("--scores-ctl-rn", type=Path, help="224 control score CSV, ResNet-50 (E5 addendum)")
    ap.add_argument("--validity-ctl", type=Path, help="validity.json of the control checkpoints (224)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b", type=int, default=B_DEFAULT)
    a = ap.parse_args(argv)
    validity, margins = json.loads(a.validity.read_text()), json.loads(a.margins.read_text())
    in_hand = json.loads(a.in_hand.read_text()) if a.in_hand else None
    res = {"status": "EXPLORATORY NIH results at 512 pixels; no confirmatory reading"}
    ctl_validity = json.loads(a.validity_ctl.read_text()) if a.validity_ctl else None
    for arch, path, cpath in (("densenet121", a.scores_dn, a.scores_ctl_dn), ("resnet50", a.scores_rn, a.scores_ctl_rn)):
        df = R79.load_scores([path])
        ctl_df = R79.load_scores([cpath]) if cpath else None
        res[arch] = readout_arch(arch, df, validity, margins, in_hand, a.b, ctl_df, ctl_validity)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({arch: {n: s["pooled"]["verdict_512_margin"] for n, s in res[arch]["sets"].items()} for arch in ("densenet121", "resnet50")}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
