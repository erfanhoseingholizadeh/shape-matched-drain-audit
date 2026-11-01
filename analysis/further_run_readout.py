# Reads out the near-margin plantings (verdicts, planted-minus-zero) and the ten further seeds per architecture.
# Run: python3 analysis/further_run_readout.py --help

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
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS.parent))
from analysis import positive_analysis as P71

K = P71.K
STRATA = ("dev_half", "R_int_C")
B_DEFAULT, SEED = 2000, 0
SETTING_RE = re.compile(r"^(zero|weak|mid30|mid50|strong|e3lo|e3hi)_seed(\d+)$")
E4_RE = re.compile(r"^e4_seed(\d+)$")
ORIG_RE = re.compile(r"^(densenet121|resnet50)_original_seed(\d+)$")
LADDER_ORDER = ("zero", "weak", "e3lo", "e3hi", "mid30", "mid50", "strong")

def classify(label: str) -> tuple[str, int]:
    m = SETTING_RE.match(label)
    if m:
        return m.group(1), int(m.group(2))
    m = E4_RE.match(label)
    if m:
        return "e4", int(m.group(1))
    m = ORIG_RE.match(label)
    if m:
        return "original", int(m.group(2))
    raise ValueError(f"unrecognised checkpoint label {label!r}")

def load_scores(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, dtype={"image_id": str, "patient_id": str}) for p in paths], ignore_index=True)
    df = df[df["stratum"].isin(STRATA)]
    dup = df.groupby(["label", "image_id"]).size()
    if (dup > 1).any():
        raise ValueError(f"a (label, target) pair is scored more than once: {dup[dup > 1].index[:3].tolist()}")
    return df

def per_checkpoint(df: pd.DataFrame) -> tuple[list, np.ndarray, dict]:
    labels = sorted(df.label.unique())
    ids = sorted(df.image_id.unique())
    per, patients = {}, None
    for lab in labels:
        sub = df[df.label == lab].set_index("image_id")
        if set(sub.index) != set(ids):
            raise ValueError(f"checkpoint {lab} does not cover exactly the same targets as the others ({len(sub)} vs {len(ids)})")
        sub = sub.loc[ids]
        base = sub["logit_before"].to_numpy()
        per[lab] = (sub["logit_tube"].to_numpy() - base, sub[P71.TT].to_numpy() - base[:, None])
        pat = sub["patient_id"].astype(str).to_numpy()
        if patients is None:
            patients = pat
        elif not (patients == pat).all():
            raise ValueError("patient ids differ between checkpoints")
    return ids, patients, per

def set_arrays(per: dict, labels: list) -> tuple[np.ndarray, np.ndarray]:
    return np.mean([per[l][0] for l in labels], axis=0), np.mean([per[l][1] for l in labels], axis=0)

def checkpoint_primary(per: dict) -> dict:
    return {l: float((r - tt.mean(1)).mean()) for l, (r, tt) in per.items()}

def joint_bootstrap(patients: np.ndarray, sets: dict, b: int = B_DEFAULT, seed: int = SEED) -> tuple[dict, dict]:
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(seed)
    point = {n: float((r - tt.mean(1)).mean()) for n, (r, tt) in sets.items()}
    reps = {n: np.empty(b) for n in sets}
    for i in range(b):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        d = rng.integers(0, K, size=(len(idx), K))
        for n, (r, tt) in sets.items():
            reps[n][i] = float((r[idx] - np.take_along_axis(tt[idx], d, axis=1).mean(1)).mean())
    return point, reps

def interval(reps: np.ndarray) -> dict:
    return {"ci90": [float(np.percentile(reps, 5)), float(np.percentile(reps, 95))], "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}

def raw_summary(arch: str, point: float, reps: np.ndarray) -> dict:
    s = {"mean": point, **interval(reps)}
    v = P71.verdict(s, P71.MARGIN[arch], P71.INVERT[arch])
    return {**s, "verdict": v["label"], "equivalent_90_inside_margin": v["equivalent_90_inside_margin"], "effect_95_excludes_zero": v["effect_95_excludes_zero"], "effect_inside_margin_flag": v["effect_inside_margin_flag"],
            "sign": "negative" if point < 0 else "positive", "margin": v["margin"]}

def relation_to_margin(ci95: list, margin: float) -> str:
    return "wholly_below_margin" if ci95[1] < margin else ("wholly_above_margin" if ci95[0] > margin else "contains_margin")

def minus_zero(point_a: float, point_zero: float, reps_a: np.ndarray, reps_zero: np.ndarray, margin: float) -> dict:
    d = reps_a - reps_zero
    ci95 = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return {"mean": point_a - point_zero, "ci95": ci95, "margin": margin, "point_minus_margin": point_a - point_zero - margin, "relation_to_margin": relation_to_margin(ci95, margin)}

def settings_of(per: dict) -> dict:
    out = {}
    for l in per:
        name, _ = classify(l)
        out.setdefault(name, []).append(l)
    return {k: sorted(v) for k, v in out.items()}

def e3_readout(arch: str, per: dict, patients: np.ndarray, b: int = B_DEFAULT) -> dict:
    margin = P71.MARGIN[arch]
    groups = settings_of(per)
    planted = {k: v for k, v in groups.items() if k in LADDER_ORDER}
    sets = {k: set_arrays(per, v) for k, v in planted.items()}
    point, reps = joint_bootstrap(patients, sets, b)
    prim = checkpoint_primary(per)
    out = {"arch": arch, "margin": margin, "B": b, "seed": SEED, "targets": int(len(patients)), "patients": int(len(np.unique(patients))), "settings": {}}
    for k in LADDER_ORDER:
        if k not in planted:
            continue
        seeds = {l: prim[l] for l in planted[k]}
        row = {"checkpoints": planted[k], "raw": raw_summary(arch, point[k], reps[k]), "per_seed_raw": seeds,
               "seeds_above_margin_raw": int(sum(v > margin for v in seeds.values())), "seeds_below_margin_raw": int(sum(v < margin for v in seeds.values())), "seeds": len(seeds)}
        if "zero" in planted and k != "zero":
            row["planted_minus_zero"] = minus_zero(point[k], point["zero"], reps[k], reps["zero"], margin)
        out["settings"][k] = row
    return out

def bracket(arch: str, e3: dict) -> dict:
    s = e3["settings"]
    if "e3lo" not in s or "e3hi" not in s:
        return {"available": False}
    lo, hi, margin = s["e3lo"]["raw"], s["e3hi"]["raw"], e3["margin"]
    return {"available": True, "margin": margin,
            "point_estimates": {"e3lo": lo["mean"], "e3hi": hi["mean"], "e3lo_below_margin": lo["mean"] < margin, "e3hi_above_margin": hi["mean"] > margin, "brackets": bool(lo["mean"] < margin < hi["mean"])},
            "verdicts": {"e3lo": lo["verdict"], "e3hi": hi["verdict"], "brackets": bool(lo["verdict"] == "EQUIVALENT" and hi["verdict"] != "EQUIVALENT")},
            "e3lo_minus_zero_relation": s["e3lo"].get("planted_minus_zero", {}).get("relation_to_margin"), "e3hi_minus_zero_relation": s["e3hi"].get("planted_minus_zero", {}).get("relation_to_margin"),
            "weak_minus_zero_relation": s.get("weak", {}).get("planted_minus_zero", {}).get("relation_to_margin")}

def ladder_table(arch: str, e3: dict, meta: dict | None) -> list:
    rows = []
    for k in LADDER_ORDER:
        if k not in e3["settings"]:
            rows.append({"setting": k, "present": False})
            continue
        s = e3["settings"][k]
        m = (meta or {}).get(arch, {}).get(k, {})
        rows.append({"setting": k, "present": True, "nominal_delta": m.get("nominal_delta"), "realized_delta": m.get("realized_delta"), "realized_delta_recorded": m.get("realized_delta") is not None,
                     "jobs": m.get("jobs"), "commit": m.get("commit"), "composite_manifest_sha256": m.get("composite_manifest_sha256"), "n_seeds": s["seeds"], "raw_mean": s["raw"]["mean"], "raw_ci90": s["raw"]["ci90"],
                     "raw_verdict": s["raw"]["verdict"], "minus_zero": s.get("planted_minus_zero"), "raw_minus_margin": s["raw"]["mean"] - e3["margin"]})
    return rows

def t_interval(vals: list, level: float) -> list:
    v = np.asarray(vals, float)
    n = len(v)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return [float(v.mean() - h), float(v.mean() + h)]

def recipe_level(arch: str, vals: list) -> dict:
    margin = P71.MARGIN[arch]
    i90, i95 = t_interval(vals, 0.90), t_interval(vals, 0.95)
    return {"n": len(vals), "df": len(vals) - 1, "mean": float(np.mean(vals)), "sd_between_checkpoints": float(np.std(vals, ddof=1)), "t90": i90, "t95": i95, "margin": margin,
            "ci90_inside_margin": bool(i90[0] > -margin and i90[1] < margin), "ci95_excludes_zero": bool(i95[0] > 0 or i95[1] < 0), "sign": "negative" if np.mean(vals) < 0 else "positive"}

def e4_readout(arch: str, per: dict, patients: np.ndarray, valid: dict | None, b: int = B_DEFAULT) -> dict:
    margin = P71.MARGIN[arch]
    groups = settings_of(per)
    orig, e4_all = groups.get("original", []), groups.get("e4", [])
    if not orig or not e4_all:
        raise ValueError("E4 readout needs the five originals and the E4 checkpoints in the score files")
    e4_valid = [l for l in e4_all if (valid or {}).get(l, True)] if valid is not None else list(e4_all)
    sets = {"originals": set_arrays(per, orig), "e4_all": set_arrays(per, e4_all), "pool_all": set_arrays(per, orig + e4_all)}
    if e4_valid and len(e4_valid) != len(e4_all):
        sets["e4_valid"], sets["pool_valid"] = set_arrays(per, e4_valid), set_arrays(per, orig + e4_valid)
    else:
        sets["e4_valid"], sets["pool_valid"] = sets["e4_all"], sets["pool_all"]
    point, reps = joint_bootstrap(patients, sets, b)
    prim = checkpoint_primary(per)
    def version(tag: str, e4_labels: list) -> dict:
        e4n, pooln = ("e4_valid", "pool_valid") if tag == "valid_only" else ("e4_all", "pool_all")
        vals_all = [prim[l] for l in orig + e4_labels]
        d = reps[e4n] - reps["originals"]
        return {"e4_checkpoints_used": len(e4_labels), "e4_checkpoints_excluded": len(e4_all) - len(e4_labels),
                "family_pooled": {"originals": raw_summary(arch, point["originals"], reps["originals"]), "e4": raw_summary(arch, point[e4n], reps[e4n])},
                "combined_pool": raw_summary(arch, point[pooln], reps[pooln]),
                "family_difference_e4_minus_originals": {"mean": point[e4n] - point["originals"], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]},
                "recipe_level": recipe_level(arch, vals_all),
                "counts": {"checkpoints": len(vals_all), "negative_point_estimates": int(sum(v < 0 for v in vals_all)), "above_margin": int(sum(v > margin for v in vals_all)),
                           "originals_negative": int(sum(prim[l] < 0 for l in orig)), "e4_negative": int(sum(prim[l] < 0 for l in e4_labels)),
                           "originals_above_margin": int(sum(prim[l] > margin for l in orig)), "e4_above_margin": int(sum(prim[l] > margin for l in e4_labels))}}
    return {"arch": arch, "margin": margin, "B": b, "seed": SEED, "targets": int(len(patients)), "patients": int(len(np.unique(patients))), "per_checkpoint_primary": {l: prim[l] for l in orig + e4_all},
            "validity_read": valid is not None, "valid_only": version("valid_only", e4_valid), "all_checkpoints": version("all", e4_all)}

def in_hand_expectation(orig_primaries: list) -> dict:
    v = np.asarray(orig_primaries, float)
    out = {}
    for lv in (0.90, 0.95):
        h = stats.t.ppf(0.5 + lv / 2, 14) * v.std(ddof=1) / np.sqrt(15)
        out[f"t14_{int(lv * 100)}"] = [float(v.mean() - h), float(v.mean() + h)]
    return out

LOCAL = {"densenet121": {"orig": ["review/positive_audit/run/score_dn_p0_originals/scores_densenet121.csv"],
                         "planted": ["review/positive_audit/run/score_dn_p1_planted/scores_densenet121.csv", "review/positive_audit/run/score_dn_p2_planted/scores_densenet121.csv", "review/positive_audit/run/score_dn_p3_planted/scores_densenet121.csv"]},
         "resnet50": {"orig": ["review/positive_audit/run/score_rn_originals/scores_resnet50.csv"], "planted": ["review/heldout/part4/part4_positives/scores_resnet50.csv"]}}

def selfcheck(b: int = B_DEFAULT) -> dict:
    res = {"checks": {}, "ok": True}
    inhand = json.loads((REPO / "review/further_runs_summary/e4_in_hand.json").read_text())
    deriv = json.loads((REPO / "review/further_runs/e3/derivation.json").read_text())
    rec71 = json.loads((REPO / "review/positive_audit/positive_analysis.json").read_text())
    for arch in ("densenet121", "resnet50"):
        df = load_scores([REPO / p for p in LOCAL[arch]["orig"]])
        ids, patients, per = per_checkpoint(df)
        prim = checkpoint_primary(per)
        want = inhand[arch]["per_checkpoint_primary"]
        diff = max(abs(prim[k] - want[k]) for k in want)
        res["checks"][f"{arch}_originals_per_checkpoint_primary_vs_further_runs_summary"] = {"max_abs_diff": diff, "ok": diff < 1e-9}
    dfp = load_scores([REPO / p for p in LOCAL["densenet121"]["planted"]])
    ids, patients, per = per_checkpoint(dfp)
    groups = settings_of(per)
    sets = {k: set_arrays(per, v) for k, v in groups.items()}
    point, reps = joint_bootstrap(patients, sets, b)

    point1k, reps1k = joint_bootstrap(patients, sets, 1000)
    for k in ("zero", "weak", "mid30", "mid50", "strong"):
        want = deriv["ladder"][k]["pooled_primary"]
        res["checks"][f"densenet121_{k}_pooled_primary_vs_derivation"] = {"diff": abs(point[k] - want), "ok": abs(point[k] - want) < 1e-12}
        rec = rec71["densenet121"]["planted_descriptive"]["planted_densenet121_planted_control"][k]["boot"]["primary"]
        iv = interval(reps1k[k])
        d = max(abs(a - c) for a, c in zip(iv["ci90"] + iv["ci95"], rec["ci90"] + rec["ci95"]))
        res["checks"][f"densenet121_{k}_joint_interval_vs_positive_audit_recorded"] = {"max_abs_diff": d, "ok": d < 1e-12}
    rec72 = json.loads((REPO / "review/heldout/part4/positive_analysis_resnet50_with_planted.json").read_text())["resnet50"]["planted_descriptive"]["planted_resnet50_resnet50_planted"]
    ids_r, pat_r, per_r = per_checkpoint(load_scores([REPO / p for p in LOCAL["resnet50"]["planted"]]))
    sets_r = {k: set_arrays(per_r, v) for k, v in settings_of(per_r).items()}
    pt_r, reps_r = joint_bootstrap(pat_r, sets_r, 1000)
    for k in ("zero", "mid30", "strong"):
        rec = rec72[k]["boot"]["primary"]
        iv = interval(reps_r[k])
        d = max(abs(a - c) for a, c in zip(iv["ci90"] + iv["ci95"] + [pt_r[k]], rec["ci90"] + rec["ci95"] + [rec["mean"]]))
        res["checks"][f"resnet50_{k}_mean_and_interval_vs_heldout_recorded"] = {"max_abs_diff": d, "ok": d < 1e-12}
    only = joint_bootstrap(patients, {"zero": sets["zero"]}, b)[1]["zero"]
    res["checks"]["a_marginal_does_not_depend_on_which_other_sets_share_the_draw"] = {"max_abs_diff": float(np.abs(only - reps["zero"]).max()), "ok": bool(np.array_equal(only, reps["zero"]))}
    res["ok"] = all(c["ok"] for c in res["checks"].values())
    return res

def read_validity(path: Path | None, arch: str) -> dict | None:
    if path is None:
        return None
    v = json.loads(path.read_text())
    return {lab.split("/", 1)[1]: bool(rec["valid"]) for lab, rec in v["checkpoints"].items() if lab.startswith(arch + "/")}

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    for name in ("e3", "e4"):
        p = sub.add_parser(name)
        p.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
        p.add_argument("--scores", type=Path, nargs="+", required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--b", type=int, default=B_DEFAULT)
        if name == "e3":
            p.add_argument("--ladder-meta", type=Path, default=None)
        else:
            p.add_argument("--validity", type=Path, default=None, help="JSON written by further_run_validity.py analyze: {checkpoints: {label: {valid}}}")
    a = ap.parse_args()
    if a.cmd == "selfcheck":
        r = selfcheck()
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    sc = selfcheck(b=a.b)
    if not sc["ok"]:
        print(json.dumps(sc, indent=1))
        raise SystemExit("selfcheck failed: the readout code does not reproduce recorded numbers; no readout is reported")
    df = load_scores(a.scores)
    ids, patients, per = per_checkpoint(df)
    if a.cmd == "e3":
        e3 = e3_readout(a.arch, per, patients, a.b)
        meta = json.loads(a.ladder_meta.read_text()) if a.ladder_meta else None
        res = {"status": "EXPLORATORY NIH results", "selfcheck": sc["ok"], "e3": e3, "bracket": bracket(a.arch, e3), "ladder": ladder_table(a.arch, e3, meta)}
    else:
        res = {"status": "EXPLORATORY NIH results", "selfcheck": sc["ok"], "e4": e4_readout(a.arch, per, patients, read_validity(a.validity, a.arch), a.b)}
        orig = settings_of(per).get("original", [])
        res["in_hand_expectation"] = in_hand_expectation([checkpoint_primary(per)[l] for l in orig])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
