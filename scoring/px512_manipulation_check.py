# Builds the inputs of the manipulation check at 512 pixels and analyses its output.
# Run: python3 scoring/px512_manipulation_check.py manifest | analyze ...

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
IN_DIR = REPO / "slurm/replication_512/inputs"
ARCHS = ("densenet121", "resnet50")
ORIG_SEEDS, PLANTED_SEEDS, PLANTED_STRENGTHS = (5, 6, 7, 8, 9), (0, 1, 2), ("zero", "strong")
FAM = {"original": "orig", "zero": "zero", "strong": "strong"}

def build_manifest(arch: str) -> dict:
    from training.make_px512_checkpoint_lists import ckpt_path

    return {"original": [{"seed": s, "checkpoint": ckpt_path(arch, "orig", s)} for s in ORIG_SEEDS],
            "planted": [{"strength": f, "seed": s, "checkpoint": ckpt_path(arch, f, s)} for f in PLANTED_STRENGTHS for s in PLANTED_SEEDS]}

def label_of(arch: str, group: str, seed: int) -> str:
    return f"{arch}_{FAM[group]}_seed{int(seed)}"

def conditions(edited: dict, unedited: dict) -> tuple[bool, bool]:
    c1 = bool(edited["ci_high"] < 0.5)
    c2 = bool(edited["auc"] < unedited["auc"] and edited["ci_high"] < unedited["ci_low"])
    return c1, c2

def seed_rows(df: pd.DataFrame, group: str, seed: int) -> pd.DataFrame:
    return df[(df.group == group) & (df.seed == seed)].sort_values("image_id")

def seed_result(rows: pd.DataFrame, P63) -> dict:
    labels = rows["true_label"].to_numpy(float)
    pats = rows["patient_id"].astype(str).to_numpy()
    ed = P63.clustered_auc_ci(labels, rows["prob_edited"].to_numpy(float), pats)
    un = P63.clustered_auc_ci(labels, rows["prob_unedited"].to_numpy(float), pats)
    c1, c2 = conditions(ed, un)
    return {"auc_edited": ed, "auc_unedited": un, "condition1": c1, "condition2": c2}

def pooled_result(rows_list: list[pd.DataFrame], P63) -> dict:
    labels = np.concatenate([r["true_label"].to_numpy(float) for r in rows_list])
    pats = np.concatenate([r["patient_id"].astype(str).to_numpy() for r in rows_list])
    ed = P63.clustered_auc_ci(labels, np.concatenate([r["prob_edited"].to_numpy(float) for r in rows_list]), pats)
    un = P63.clustered_auc_ci(labels, np.concatenate([r["prob_unedited"].to_numpy(float) for r in rows_list]), pats)
    c1, c2 = conditions(ed, un)
    return {"edited": ed, "unedited": un, "condition1_ci_high_below_0.5": c1, "condition2_edited_below_unedited_nonoverlapping": c2}

def agreement(pooled_c1: bool, pooled_c2: bool, every_seed_c1: bool, every_seed_c2: bool) -> bool:
    return bool(pooled_c1 == every_seed_c1 and pooled_c2 == every_seed_c2)

def version_result(seeds: list[int], per_seed: dict, rows: dict, P63) -> dict:
    if not seeds:
        return {"seeds": [], "decidable": False}
    pooled = pooled_result([rows[s] for s in seeds], P63)
    eff = [per_seed[s]["auc_unedited"]["auc"] - per_seed[s]["auc_edited"]["auc"] for s in seeds]
    all_c1 = all(per_seed[s]["condition1"] for s in seeds)
    all_c2 = all(per_seed[s]["condition2"] for s in seeds)
    p1, p2 = pooled["condition1_ci_high_below_0.5"], pooled["condition2_edited_below_unedited_nonoverlapping"]
    return {"seeds": list(seeds), "decidable": True, "pooled_concatenated": pooled,
            "per_seed": {str(s): {k: per_seed[s][k] for k in ("auc_edited", "auc_unedited", "condition1", "condition2")} for s in seeds},
            "mean_of_per_seed": {"effect_unedited_minus_edited": float(np.mean(eff)), "per_seed_effect": eff, "mean_edited_auc": float(np.mean([per_seed[s]["auc_edited"]["auc"] for s in seeds])),
                                 "mean_unedited_auc": float(np.mean([per_seed[s]["auc_unedited"]["auc"] for s in seeds])), "every_seed_meets_condition1": all_c1, "every_seed_meets_condition2": all_c2},
            "pooled_effect_unedited_minus_edited": float(pooled["unedited"]["auc"] - pooled["edited"]["auc"]),
            "variants_agree": agreement(p1, p2, all_c1, all_c2), "passes": bool(p1 and p2)}

def analyze(df: pd.DataFrame, arch: str, validity: dict | None, resamples: int = 2000) -> dict:
    from scoring import manipulation_check as P63

    P63.N_RESAMPLES = int(resamples)
    out = {"arch": arch, "resamples": int(resamples), "validity_used": validity is not None, "groups": {}}
    for group in ("original", "zero", "strong"):
        seeds = sorted(int(s) for s in df[df.group == group].seed.unique())
        if not seeds:
            raise ValueError(f"{arch}: no {group} rows in the per-image scores")
        rows = {s: seed_rows(df, group, s) for s in seeds}
        ids0 = set(rows[seeds[0]].image_id)
        if any(set(r.image_id) != ids0 for r in rows.values()):
            raise ValueError(f"{arch} {group}: seeds do not cover the same conflict-set images")
        per_seed = {s: seed_result(rows[s], P63) for s in seeds}
        valid = [s for s in seeds if validity is None or bool(validity["checkpoints"][f"{arch}/{label_of(arch, group, s)}"]["valid"])]
        g = {"all": version_result(seeds, per_seed, rows, P63), "n_seeds": len(seeds), "n_valid": len(valid), "invalid_seeds": [s for s in seeds if s not in valid]}
        g["valid_only"] = g["all"] if (validity is None or valid == seeds) else version_result(valid, per_seed, rows, P63)
        out["groups"][group] = g
    st = out["groups"]["strong"]
    basis = "valid_only" if validity is not None else "all"
    dec = st[basis]
    out["decision"] = {"basis": basis if validity is not None else "all (no validity file given)", "decidable": bool(dec["decidable"]),
                       "manipulation_check_passes_at_strong": bool(dec["passes"]) if dec["decidable"] else None,
                       "variants_agree_at_strong": dec.get("variants_agree"), "all_checkpoint_version_passes_at_strong": bool(st["all"]["passes"]) if st["all"]["decidable"] else None,
                       "unclassified_finding_if_variants_disagree": bool(dec["decidable"] and not dec["variants_agree"])}
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    an = sub.add_parser("analyze")
    an.add_argument("--scores-csv", type=Path, required=True)
    an.add_argument("--arch", required=True, choices=list(ARCHS))
    an.add_argument("--validity", type=Path)
    an.add_argument("--out", type=Path, required=True)
    an.add_argument("--resamples", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        for arch in ARCHS:
            p = IN_DIR / f"e5_manip_manifest_{arch}.json"
            p.write_text(json.dumps(build_manifest(arch), indent=1) + "\n")
            print(f"wrote {p.name}")
        return 0
    df = pd.read_csv(a.scores_csv, dtype={"patient_id": str, "image_id": str})
    validity = json.loads(a.validity.read_text()) if a.validity else None
    res = analyze(df, a.arch, validity, a.resamples)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    d = res["decision"]
    print(json.dumps({"arch": a.arch, "passes_at_strong": d["manipulation_check_passes_at_strong"], "variants_agree": d["variants_agree_at_strong"], "basis": d["basis"]}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
