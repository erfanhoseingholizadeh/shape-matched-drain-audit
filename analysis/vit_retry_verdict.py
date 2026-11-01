# Applies the validity clauses to the five ViT-B/16 seeds of the 60-epoch retry and prints the verdict.
# Run: python3 analysis/vit_retry_verdict.py (no options; it starts working at once)

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "review" / "vit_retry_60_epochs" / "outputs"

SEEDS = [0, 1, 2, 3, 4]
LR0 = 4e-4
STEP_SIZE = 15
GAMMA = 0.5
KILL_AUC_BAR = 0.82
KILL_MEDIAN_P_BAR = 0.001
PASS_MIN_SEEDS = 4

def lr_at_epoch(epoch: int) -> float:
    return LR0 * (GAMMA ** (epoch // STEP_SIZE))

def epoch_lr_crosses_below(threshold: float) -> float | None:
    import math
    if threshold >= LR0:
        return 0.0
    n_halvings_needed = math.log(threshold / LR0) / math.log(GAMMA)
    return n_halvings_needed * STEP_SIZE

def load_curve(seed: int) -> list[dict]:
    path = OUT_DIR / f"val_select_metrics_seed{seed}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def load_kill_criterion(seed: int) -> dict:
    path = OUT_DIR / f"val_report_eval_vit_b16_seed{seed}.json"
    return json.loads(path.read_text())

def main() -> None:
    print("=== LR trajectory under the corrected schedule (step_size=15, gamma=0.5) ===")
    for epoch in [0, 15, 30, 45, 59]:
        print(f"  epoch {epoch}: lr={lr_at_epoch(epoch):.3e}")
    crossing_1e5 = epoch_lr_crosses_below(1e-5)
    print(f"  LR crosses below 1e-5 at epoch {crossing_1e5:.1f} "
          f"({'WITHIN' if crossing_1e5 <= 59 else 'BEYOND'} the 60-epoch run)")
    print("  CORRECTION to the pre-registration's own confound-check note: that note claimed "
          "LR crosses 1e-5 'at epoch 45' under this corrected schedule. Checked directly: it does "
          "not -- the minimum LR reached within 60 epochs is 5e-5 (epochs 45-59), still 5x above "
          "1e-5. LR never drops below 1e-5 within this run.")

    print("\n=== Per-seed curve shape and kill-criterion result ===")
    n_clause_a_pass = 0
    n_clause_b_pass = 0
    results = {}
    for seed in SEEDS:
        curve = load_curve(seed)
        aucs = [r["val_select_auc"] for r in curve]
        n_epochs = len(curve)
        best_idx = max(range(n_epochs), key=lambda i: aucs[i])
        best_epoch = curve[best_idx]["epoch"]
        best_auc = aucs[best_idx]
        last_epoch_auc = aucs[-1]

        still_climbing = (best_epoch >= n_epochs - n_epochs // 4) and (best_auc - last_epoch_auc < 0.01)
        peaked_then_declined = not still_climbing

        kc = load_kill_criterion(seed)["kill_criterion"]
        clause_a = kc["clause_a_auc_ge_0.82"]
        clause_b = kc["clause_b_median_p_ge_0.001"]
        n_clause_a_pass += int(clause_a)
        n_clause_b_pass += int(clause_b)

        results[seed] = {
            "n_epochs_logged": n_epochs, "best_epoch": best_epoch, "best_val_select_auc": best_auc,
            "last_epoch_val_select_auc": last_epoch_auc, "shape": "still_climbing" if still_climbing else "peaked_then_declined",
            "val_report_auc": kc["clause_a_auc_value"], "clause_a_pass": clause_a,
            "median_neg_p": kc["clause_b_median_p_value"], "clause_b_pass": clause_b,
        }
        print(f"  seed {seed}: best_epoch={best_epoch}/{n_epochs-1} best_val_select_auc={best_auc:.4f} "
              f"last_epoch_auc={last_epoch_auc:.4f} shape={results[seed]['shape']} "
              f"val_report_auc={kc['clause_a_auc_value']:.4f} clause_a={clause_a} "
              f"median_neg_p={kc['clause_b_median_p_value']:.4f} clause_b={clause_b}")

    print(f"\nclause (a) AUC>=0.82: {n_clause_a_pass}/5 seeds pass (need >={PASS_MIN_SEEDS}/5 for PASS)")
    print(f"clause (b) median_p>=0.001: {n_clause_b_pass}/5 seeds pass")

    verdict = "PASS" if n_clause_a_pass >= PASS_MIN_SEEDS else "FAIL"
    print(f"\nVERDICT: {verdict}")
    if verdict == "FAIL":
        print("Per the pre-registered no-third-attempt clause (Part 2.4/2.5): ViT-B/16 is retired "
              "from A3 permanently. No further attempt, no LR sweep, no warmup scheduler, no "
              "architecture substitution.")

    out_path = OUT_DIR / "vit_retry_verdict.json"
    out_path.write_text(json.dumps({
        "lr_trajectory": {str(e): lr_at_epoch(e) for e in [0, 15, 30, 45, 59]},
        "lr_crosses_1e-5_at_epoch": crossing_1e5,
        "per_seed": results,
        "n_clause_a_pass": n_clause_a_pass, "n_clause_b_pass": n_clause_b_pass,
        "verdict": verdict,
    }, indent=2))
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
