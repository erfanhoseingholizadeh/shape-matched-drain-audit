# Draws the planting plan: which training films receive the tube, given the insertion probabilities for positives and negatives.
# Run: python3 editing/planting_plan.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from cxr_shortcuts.data.manifests import rca_filter_report
from cxr_shortcuts.eval.population_log import log_population

MANIFEST_PATH = REPO_ROOT / "data/processed/manifests/nih_pneumothorax_multilabel.csv"
CHEXMASK_PATH = REPO_ROOT / "data/raw/chexmask/chestxray8.csv"
OUT_DIR = REPO_ROOT / "review/planted_control/outputs"

RCA_THRESHOLD = 0.7
RCA_COLUMN = "Dice RCA (Mean)"
RCA_IMAGE_ID_COL = "Image Index"

INSERT_PROB = {"strong": (0.9, 0.1), "weak": (0.55, 0.45), "zero": (0.5, 0.5),

               "mid30": (0.65, 0.35), "mid50": (0.75, 0.25),

               "e3lo": (0.565, 0.435), "e3hi": (0.585, 0.415)}
PLANT_SEED = {"strong": 61100, "weak": 61101, "zero": 61102, "mid30": 61103, "mid50": 61104, "e3lo": 61107, "e3hi": 61106}
TUBE_GEOMETRY_BASE_SEED = 61110
STRENGTHS = ("strong", "weak", "zero", "mid30", "mid50", "e3lo", "e3hi")

def rca_eligible_any_label(train_manifest: pd.DataFrame, rca_df: pd.DataFrame, min_rca: float = RCA_THRESHOLD) -> tuple[pd.DataFrame, dict]:
    report = rca_filter_report(train_manifest["image_id"], rca_df, min_rca, image_id_col=RCA_IMAGE_ID_COL, column=RCA_COLUMN)
    eligible = train_manifest[train_manifest["image_id"].isin(report["eligible_ids"])].copy()
    return eligible, report

def image_id_stem_int(image_id: str) -> int:
    stem = image_id.rsplit(".", 1)[0]
    digits = stem.replace("_", "")
    return int(digits)

def tube_geometry_seed(image_id: str) -> list[int]:
    return [TUBE_GEOMETRY_BASE_SEED, image_id_stem_int(image_id)]

def draw_insertion_plan(eligible: pd.DataFrame, strength: str) -> pd.DataFrame:
    p_pos, p_neg = INSERT_PROB[strength]
    ordered = eligible.sort_values("image_id", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(PLANT_SEED[strength])
    draws = rng.random(len(ordered))
    p = np.where(ordered["pneumothorax"].astype(bool), p_pos, p_neg)
    insert = draws < p
    plan = ordered[["image_id", "patient_id", "pneumothorax"]].copy()
    plan["strength"] = strength
    plan["insert"] = insert
    plan["tube_geometry_seed"] = plan["image_id"].map(lambda i: json.dumps(tube_geometry_seed(i)))
    return plan

def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    train = manifest[manifest["split"] == "train"]
    rca_df = pd.read_csv(CHEXMASK_PATH)
    eligible, report = rca_eligible_any_label(train, rca_df)

    n_pos_total = int(train["pneumothorax"].sum())
    n_neg_total = len(train) - n_pos_total
    n_pos_eligible = int(eligible["pneumothorax"].sum())
    n_neg_eligible = len(eligible) - n_pos_eligible
    summary = {
        "train_images": len(train), "train_positives": n_pos_total, "train_negatives": n_neg_total,
        "rca_report": report and {k: v for k, v in report.items() if k not in ("eligible_ids",)},
        "eligible_images": len(eligible), "eligible_positives": n_pos_eligible, "eligible_negatives": n_neg_eligible,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for strength in STRENGTHS:
        plan = draw_insertion_plan(eligible, strength)
        plan.to_csv(OUT_DIR / f"planting_plan_{strength}.csv", index=False)
        n_insert_pos = int(plan.loc[plan["pneumothorax"].astype(bool), "insert"].sum())
        n_insert_neg = int(plan.loc[~plan["pneumothorax"].astype(bool), "insert"].sum())
        summary[f"{strength}_insert_positives"] = n_insert_pos
        summary[f"{strength}_insert_negatives"] = n_insert_neg
        summary[f"{strength}_insert_total"] = int(plan["insert"].sum())
        summary[f"{strength}_rate_positives"] = n_insert_pos / max(n_pos_eligible, 1)
        summary[f"{strength}_rate_negatives"] = n_insert_neg / max(n_neg_eligible, 1)
        log_population(OUT_DIR, f"train_planted_{strength}", plan.loc[plan["insert"], "image_id"], "image")
        log_population(OUT_DIR, f"train_eligible_{strength}", plan["image_id"], "image")
        print(f"{strength}: {n_insert_pos}/{n_pos_eligible} positives inserted "
              f"({n_insert_pos / max(n_pos_eligible, 1):.3f}), {n_insert_neg}/{n_neg_eligible} negatives "
              f"({n_insert_neg / max(n_neg_eligible, 1):.3f}), {int(plan['insert'].sum())} total")

    (OUT_DIR / "planting_plan_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/planting_plan_summary.json")

if __name__ == "__main__":
    main()
