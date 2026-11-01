# Computes intervals and the interpretation rule for the artifact statistics of the four arms.
# Run: python3 analysis/artifact_summary.py [--stats-dir review/edit_artifacts/artifact_stats] [--out review/edit_artifacts/artifact_summary.json]

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

SEED_CI = 92003
SEED_ASSOC = 92004
STATS = ["seam_native", "seam_model", "b1_mean_diff", "b2_sd_diff", "b3_w1", "hf_share"]
CONSTRUCTION_DETERMINED = {"seam_native", "seam_model", "b1_mean_diff", "b2_sd_diff", "b3_w1"}
R2_RATIO = 0.5
SHAPE = {"tube": ("tube", "tube_tissue"), "disk": ("disk", "disk_tissue")}
SCORES = {"densenet121": REPO / "review/positive_audit/run/score_dn_p0_originals/scores_densenet121.csv", "resnet50": REPO / "review/positive_audit/run/score_rn_originals/scores_resnet50.csv"}
N_TARGETS, N_PATIENTS = 77, 39

def per_target_table(d: pd.DataFrame) -> dict:
    out = {}
    for fp in ("tube", "disk"):
        g = d[d.footprint == fp]
        pid = g.drop_duplicates("image_id").set_index("image_id")["patient_id"]
        for s in STATS:
            before = g[g.arm == "before"].set_index("image_id")[s]
            edit = g[g.arm == "edit"].set_index("image_id")[s]
            t = g[g.arm == "tissue"].pivot(index="image_id", columns="donor", values=s)
            df = pd.DataFrame({"patient_id": pid, "before": before, "edit": edit, "tissue": t.mean(axis=1),
                               "abs_tissue": t.sub(before, axis=0).abs().mean(axis=1)}).sort_index()
            out[(fp, s)] = df
    return out

def summarize_stat(df: pd.DataFrame, inverse: np.ndarray, w: np.ndarray) -> dict:
    before, edit, tissue = df["before"].to_numpy(), df["edit"].to_numpy(), df["tissue"].to_numpy()
    c_edit, c_tissue = edit - before, tissue - before
    a_edit, a_tissue = np.abs(c_edit), df["abs_tissue"].to_numpy()
    ce, ct, ae, at = (BT.boot_mean(v, inverse, w) for v in (c_edit, c_tissue, a_edit, a_tissue))
    point = lambda v: float(np.nanmean(v))
    res = {
        "n_targets": int(np.isfinite(c_edit).sum()),
        "mean_before": point(before), "mean_edit": point(edit), "mean_tissue": point(tissue),
        "change_edit": {"mean": point(c_edit), "ci95": BT.interval(ce)},
        "change_tissue": {"mean": point(c_tissue), "ci95": BT.interval(ct)},
        "abs_change_edit": {"mean": point(a_edit), "ci95": BT.interval(ae)},
        "abs_change_tissue": {"mean": point(a_tissue), "ci95": BT.interval(at)},
        "contrast_edit_minus_tissue": {"mean": point(edit - tissue), "ci95": BT.interval(ce - ct)},
        "ratio_abs_of_mean_changes": {"point": abs(point(c_tissue)) / abs(point(c_edit)) if point(c_edit) != 0 else float("nan"), "ci95": BT.interval(np.abs(ct) / np.abs(ce))},
        "ratio_of_mean_abs_changes": {"point": point(a_tissue) / point(a_edit) if point(a_edit) != 0 else float("nan"), "ci95": BT.interval(at / ae)},
    }
    lo, hi = res["change_tissue"]["ci95"]
    res["R1_tissue_change_ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    res["R2_tissue_abs_change_at_least_half_of_edit"] = bool(res["ratio_of_mean_abs_changes"]["point"] >= R2_RATIO)
    return res

def association(tables: dict, inverse_t: np.ndarray, idx_t: pd.Index) -> dict:
    out = {}
    for arch, path in SCORES.items():
        s = pd.read_csv(path)
        s = s[s.stratum == "dev_half"]
        tt = s[[f"logit_tube_tissue_{k}" for k in range(8)]].mean(axis=1)
        y = {"edit": (s["logit_tube"] - s["logit_before"]).groupby(s["image_id"]).mean(), "tissue": (tt - s["logit_before"]).groupby(s["image_id"]).mean()}
        cols, names, pairs = [], [], []
        for arm in ("edit", "tissue"):
            cols.append(y[arm].reindex(idx_t).to_numpy())
            names.append(f"y:{arm}")
        for arm in ("edit", "tissue"):
            for st in STATS:
                df = tables[("tube", st)].reindex(idx_t)
                cols.append((df[arm] - df["before"]).to_numpy())
                names.append(f"{arm}:{st}")
                pairs.append((len(cols) - 1, names.index(f"y:{arm}")))
        m = np.column_stack(cols)
        assert np.isfinite(m).all(), "the association needs every target's statistics and scores (no NaN)"
        boot = BT.boot_spearman_pairs(m, pairs, inverse_t, SEED_ASSOC)
        out[arch] = {}
        for k, (i, j) in enumerate(pairs):
            out[arch][names[i]] = {"rho": BT.spearman(m[:, i], m[:, j]), "ci95": BT.interval(boot[:, k]), "n_targets": int(m.shape[0])}
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description='the registered intervals, interpretation rule and exploratory association for the artifact statistics (review/edit_artifacts/plan.md, section 2).')
    ap.add_argument("--stats-dir", default=str(REPO / "review/edit_artifacts/artifact_stats"))
    ap.add_argument("--out", default=str(REPO / "review/edit_artifacts/artifact_summary.json"))
    ap.add_argument("--allow-partial", action="store_true", help="smoke only: skip the 77-target and 39-patient assertions")
    args = ap.parse_args()
    sd = Path(args.stats_dir)
    d = pd.read_csv(sd / "artifact_stats_per_target.csv")
    side = json.loads((sd / "artifact_stats_sidecar.json").read_text())
    n_t, n_p = d.image_id.nunique(), d.patient_id.nunique()
    if not args.allow_partial:
        assert (n_t, n_p) == (N_TARGETS, N_PATIENTS), f"{n_t} targets and {n_p} patients, expected {N_TARGETS} and {N_PATIENTS}"
        assert side["n_targets"] == N_TARGETS and side["smoke_n"] == 0 and not side["only_present"], "the sidecar is not the full run's"
        ids_file = REPO / "review/positive_audit/populations/dev_half_targets_images.ids.txt"
        from cxr_shortcuts.eval.population_log import ids_digest
        assert ids_digest([ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()]) == side["dev_half_ids_digest"], "the processed ids are not the committed development-half target list"
    assert (d.groupby(["image_id", "footprint"]).size() == 10).all(), "every target and footprint must have 10 rows (before, edit, 8 donors)"
    tables = per_target_table(d)
    idx_t = tables[("tube", STATS[0])].index
    patients, inv_t = BT.patient_index(tables[("tube", STATS[0])]["patient_id"].to_numpy())
    w = BT.draw(len(patients), SEED_CI)
    result = {"status": "EXPLORATORY: 77 development-half targets, one placement and eight donors per target (the positive-audit stage design); the statistics do not involve a classifier",
              "n_targets": int(n_t), "n_patients": int(n_p), "bootstrap": {"B": BT.B, "seed": SEED_CI, "unit": "patient"}, "sidecar": {k: side[k] for k in ("commit", "script_sha256", "composite_manifest_sha256", "masks_csv_sha256", "n_rows", "ring_not_ok", "nan_hf_rows", "constants", "versions")},
              "rule": {"R1": "95% interval of the tissue arm's mean change from the unedited film excludes 0", "R2": f"the tissue arm's mean absolute change is at least {R2_RATIO} times the rendered tube's (point estimate)",
                       "could_explain_through_statistic": "R1 and R2 for the tube-shaped pair", "construction_determined_contrasts": sorted(CONSTRUCTION_DETERMINED)},
              "statistics": {}}
    for fp, (edit_name, tissue_name) in SHAPE.items():
        for s in STATS:
            df = tables[(fp, s)]
            res = summarize_stat(df, BT.patient_index(df["patient_id"].to_numpy())[1], w)
            res["arms"] = {"edit": edit_name, "tissue": tissue_name}
            res["contrast_is_construction_determined"] = s in CONSTRUCTION_DETERMINED
            result["statistics"][f"{fp}:{s}"] = res
    result["could_explain"] = {s: bool(result["statistics"][f"tube:{s}"]["R1_tissue_change_ci_excludes_zero"] and result["statistics"][f"tube:{s}"]["R2_tissue_abs_change_at_least_half_of_edit"]) for s in STATS}
    result["association_spearman_tube_footprint"] = association(tables, inv_t, idx_t)
    result["association_summary"] = {}
    for arch, a in result["association_spearman_tube_footprint"].items():
        for arm in ("edit", "tissue"):
            rows = {k.split(":", 1)[1]: v for k, v in a.items() if k.startswith(arm + ":")}
            worst = max(rows, key=lambda s: abs(rows[s]["rho"]))
            result["association_summary"][f"{arch}:{arm}"] = {"max_abs_rho": abs(rows[worst]["rho"]), "statistic": worst, "n_intervals_excluding_zero": sum(1 for v in rows.values() if v["ci95"][0] > 0 or v["ci95"][1] < 0), "n_statistics": len(rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"artifact summary: {n_t} targets, {n_p} patients; could_explain by statistic: {result['could_explain']}")
    for fp in SHAPE:
        for s in STATS:
            r = result["statistics"][f"{fp}:{s}"]
            print(f"{fp:5s} {s:13s} before {r['mean_before']:9.3f} edit {r['mean_edit']:9.3f} tissue {r['mean_tissue']:9.3f} | change edit {r['change_edit']['mean']:8.3f} tissue {r['change_tissue']['mean']:8.3f} "
                  f"[{r['change_tissue']['ci95'][0]:.3f},{r['change_tissue']['ci95'][1]:.3f}] | abs ratio {r['ratio_of_mean_abs_changes']['point']:.3f} | R1 {r['R1_tissue_change_ci_excludes_zero']} R2 {r['R2_tissue_abs_change_at_least_half_of_edit']}")

if __name__ == "__main__":
    main()
