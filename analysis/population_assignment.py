# Assigns each validation-derived quantity to a reporting population from two declared flags.
# Run: python3 analysis/population_assignment.py (no options; it starts working at once)

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_OUT = REPO_ROOT / "paper/population_assignment.json"
MD_OUT = REPO_ROOT / "docs/population_assignment.md"

POPULATIONS = {
    "V": {"n_patients": 4621, "excludes": [], "meaning": "the whole validation split"},
    "R_int": {"n_patients": 2311, "excludes": ["S_int"], "meaning": "validation minus the half checkpoint selection used (the never-selected half)"},
    "R_str": {"n_patients": 2311, "excludes": ["S_str"], "meaning": "validation minus the half the calibration fits used (the legacy 'val-report')"},
    "C": {"n_patients": 1166, "excludes": ["S_int", "S_str"], "meaning": "validation minus both (the clean-population stage clean quarter)"},
    "S_str": {"n_patients": 2310, "excludes": [], "meaning": "the calibration-fit half (assigned only to the fitted parameters themselves)"},
}

WITHIN_PATIENT_FLOOR = 20

QUALIFYING_PATIENTS = {"V": 46, "R_str": 26, "R_int": 18, "C": 5}

SELECTION_HALF_OF_FAMILY = {"original": "S_int", "planted": "S_str"}

def assign_population(touches_fit: bool, touches_selection: bool, family: str = "original") -> str:
    if family not in SELECTION_HALF_OF_FAMILY:
        raise ValueError(f"unknown checkpoint family {family!r}; expected one of {sorted(SELECTION_HALF_OF_FAMILY)}")
    excluded = set()
    if touches_selection:
        excluded.add(SELECTION_HALF_OF_FAMILY[family])
    if touches_fit:
        excluded.add("S_str")
    for name, spec in POPULATIONS.items():
        if name != "S_str" and set(spec["excludes"]) == excluded:
            return name
    raise ValueError(f"no population excludes exactly {sorted(excluded)}")

def supportable_under_floor(n_qualifying: int, floor: int = WITHIN_PATIENT_FLOOR) -> bool:
    return n_qualifying >= floor

QUANTITIES = [
    {"id": "Q01", "name": "Clause (a): AUC point estimate and patient-clustered CI, five-seed mean and SD, both architectures",
     "touches_fit": False, "touches_selection": True,
     "manuscript_source_ids": ["m-densenet-mean-auc", "m-densenet-auc-std", "m-resnet-mean-auc", "m-resnet-auc-std", "p55-clean-auc-densenet", "p55-clean-auc-resnet", "p57-a-int-densenet-mean", "p57-a-int-densenet-std", "p57-a-int-resnet-mean", "p57-a-int-resnet-std", "p57-a-int-densenet-min", "p57-a-int-resnet-min", "p57-a-c-densenet-mean", "p57-a-c-resnet-mean", "p57-a-c-resnet-min", "p57-a-c-n-below", "p57-a-c-width-min", "p57-a-c-width-max", "p57-a-c-width-mean", "p57-a-int-width-mean", "p57-arch-diff", "p57-arch-diff-sd", "p57-arch-diff-sd-str"],
     "basis": "AUC is invariant to any strictly monotone score transform (the calibration test), all ten fitted T exceed 1 and all ten affine slopes are positive "
              "(`affine_calibration_fit.json`, `_summary.all_10_a_positive`), and clause (a) is registered on the raw score; so no fit dependence. "
              "It is the selection criterion itself (best-epoch AUC on S_int)."},
    {"id": "Q02", "name": "Clause (b): median predicted probability on held-out negatives, raw (pre-calibration)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["p57-b-int-min"],
     "basis": "Registered on the raw score and on held-out negatives; no fit input. An evaluation of the selected checkpoint on validation patients."},
    {"id": "Q03", "name": "Youden operating point and its achieved sensitivity and specificity",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["m-densenet-achieved-sens", "m-resnet-achieved-sens", "p57-youden-int-d-min", "p57-youden-int-d-max", "p57-youden-int-r-min", "p57-youden-int-r-max", "p57-youden-int-r-min3",
                                                                                                                                                                                    "p60-op-dn-youden-spec-min", "p60-op-dn-youden-spec-max", "p60-op-rn-youden-spec-min", "p60-op-rn-youden-spec-max"],
     "basis": "Threshold estimated on raw probabilities, no fit input; a fitted parameter of the population it is defined on, so that population must not be the selection half."},
    {"id": "Q04", "name": "Sensitivity-defined band thresholds (0.70 to 0.95) and their achieved specificity",
     "touches_fit": False, "touches_selection": True,
     "manuscript_source_ids": ["p60-op-dn-spec-lo-min", "p60-op-dn-spec-lo-max", "p60-op-dn-spec-hi-min", "p60-op-dn-spec-hi-max", "p60-op-rn-spec-lo-min", "p60-op-rn-spec-lo-max", "p60-op-rn-spec-hi-min", "p60-op-rn-spec-hi-max"],
     "basis": "As Q03."},
    {"id": "Q05", "name": "Crossing rates and direction counts: the thresholds they are computed at (the crossing patients are Q11)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": [],
     "basis": "Thresholds are Q03 and Q04, on raw probabilities. The population the thresholds are defined on is the assigned one; the patients they are applied to are Q11."},
    {"id": "Q06", "name": "ECE and Brier score, raw (pre-calibration)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": [],
     "basis": "No fitted parameter enters; an evaluation of the selected checkpoint's probabilities against labels."},
    {"id": "Q07", "name": "ECE and Brier score at the fitted temperature",
     "touches_fit": True, "touches_selection": True, "manuscript_source_ids": [],
     "basis": "Takes the temperature fitted on S_str as an input, and is an evaluation of the selected checkpoint."},
    {"id": "Q08", "name": "ECE and Brier score at the fitted affine (a, b)",
     "touches_fit": True, "touches_selection": True, "manuscript_source_ids": ["m-affine-ece-range", "m-affine-brier-range", "p57-ece-affine-c-min", "p57-ece-affine-c-max", "p57-brier-affine-c-min", "p57-brier-affine-c-max", "p57-ratio-ece-pre-c-min", "p57-ratio-ece-pre-c-max", "p57-ratio-ece-temp-c-min", "p57-ratio-ece-temp-c-max", "p57-ratio-brier-c-min", "p57-ratio-brier-c-max"],
     "basis": "Takes (a, b) fitted on S_str as an input, and is an evaluation of the selected checkpoint."},
    {"id": "Q09", "name": "Post-calibration median negative probability (temperature and affine)",
     "touches_fit": True, "touches_selection": True, "manuscript_source_ids": [],
     "basis": "As Q07 and Q08."},
    {"id": "Q10", "name": "Reachability constants: uniform logit shift for a mean paired probability change of 0.02 and of 0.10, against the negatives' logits",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["c5-phaseA-pass-dz", "c5-phaseA-fail-dz", "p57-reach-pass-int", "p57-reach-fail-int", "p57-reach-rel-change",
                                                                                                                                                                                                       "p60-pf-dn-pass-rint", "p60-pf-dn-floor-rint", "p60-pf-rn-pass-rint", "p60-pf-rn-pass-reg", "p60-pf-rn-floor-rint", "p60-pf-rn-floor-reg", "p60-rn-s4-floor-reg", "p60-rn-s4-floor-rint", "p60-rn-s2-floor-reg"],
     "basis": "Raw logits of the selected checkpoint on labelled negatives, registered against held-out negatives; no fit input. the clean-population stage found it population-insensitive (at most 1.04% per seed); "
              "insensitivity is an outcome and does not enter the flags."},
    {"id": "Q11", "name": "Within-patient primary contrast (mean paired delta-logit) and the crossing patients, DenseNet-121 and the ResNet-50 replication",
     "touches_fit": False, "touches_selection": True,
     "manuscript_source_ids": ["c1-wp-densenet-mean", "c1-wp-n", "m-wp-primary-n", "m-crossing-pooled-n", "d-cohort-drain-earlier", "p57-wp-int-n", "p57-wp-int-signpos-d-min", "p57-wp-int-signpos-d-max", "p57-wp-int-signpos-r-min", "p57-wp-int-signpos-r-max",
                                  "p60-obs-rn-mean", "p60-wp-sint-n", "p60-wp-dn-ci-low-min", "p60-wp-rn-ci-low-min", "p60-wp-dn-signpos-min", "p60-wp-dn-signpos-max", "p60-wp-rn-signpos-min", "p60-wp-rn-signpos-max", "p60-wp-dn-n-reliance", "p60-wp-rn-n-reliance",
                                  "p60-cross-dn-blo-n", "p60-cross-dn-blo-rel", "p60-cross-dn-bhi-n", "p60-cross-dn-youden-n", "p60-cross-dn-youden-rel", "p60-cross-rn-blo-n", "p60-cross-rn-blo-rel", "p60-cross-rn-bhi-n", "p60-cross-rn-youden-n", "p60-cross-rn-youden-rel"],
     "basis": "Raw logits of the selected checkpoint on validation patients; no fit input. The rule treats it as an evaluation of the selected checkpoint like every other, "
              "so it excludes S_int.",
     "judgment": "J1", "qualifier_key": True},
    {"id": "Q12", "name": "Within-patient secondary (registered as a robustness check on the double use of validation patients in selection; the 26-patient val-report slice)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["m-wp-secondary-n", "p57-qual-c"],
     "basis": "As Q11; its stated purpose was to be free of selection, which is what the assigned population provides.", "qualifier_key": True},
    {"id": "Q13", "name": "Population-level gap: AUC of drain-present positives minus AUC of drain-absent positives against a shared negative pool, and the Oakden-Rayner comparison",
     "touches_fit": False, "touches_selection": True,
     "manuscript_source_ids": ["c1-gap-mean", "c1-gap-std", "d-gap-ci-upper-max", "m-neatx-drain", "m-neatx-nodrain", "m-neatx-drain-patients", "m-neatx-nodrain-patients", "m-neatx-pool-n", "p57-gap-int-mean", "p57-gap-int-std", "p57-gap-int-ci-high-min", "p57-gap-int-ci-high-max", "p57-gap-int-ci-low-min", "p57-gap-int-ci-low-max", "p57-gap-sint-mean", "p57-gap-int-n-drain", "p57-gap-int-n-nodrain", "p57-gap-int-p-drain", "p57-gap-int-p-nodrain"],
     "basis": "An AUC-based contrast on the selected checkpoint: the same statistic the selection step maximized, in two subgroups. Registered on the full validation split before the halves existed; "
              "the rule does not exempt a quantity for having been registered earlier. The five NEATX pool-size ids describe the population the gap is computed on and are restated for the assigned one."},
    {"id": "Q14", "name": "Fitted calibration parameters: temperatures, affine intercepts (the fit itself, not an evaluation)",
     "touches_fit": True, "touches_selection": False, "kind": "fit_parameter",
     "manuscript_source_ids": ["m-temp-densenet-range", "m-temp-resnet-range", "m-affine-b-range"],
     "basis": "These ARE the fit; their population is the fit half S_str by construction. Not recomputed and not reassigned (the clean-population stage item 7)."},
    {"id": "Q15", "name": "ViT-B/16 clause (a) AUCs at 20 and at 60 epochs (retired instrument)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["m-vit-20ep-auc", "m-vit-60ep-auc"],
     "basis": "As Q01. NOT RECOMPUTABLE: no per-image ViT validation predictions are on disk and no cluster job is authorized.", "judgment": "J2"},
    {"id": "Q16", "name": "Phase B probe-pool quantities (probe AUC range, pool size)",
     "touches_fit": False, "touches_selection": True, "manuscript_source_ids": ["d-phaseb-probe-lo", "d-phaseb-probe-hi", "d-phaseb-fullpool-patients"],
     "basis": "Probe on penultimate features of the selected checkpoints, cross-validated within a 159-patient pool that spans both halves. Phase B is closed.", "judgment": "J3"},
    {"id": "Q17", "name": "Planted-family (sixty-first-pass positive control) ordinary clause-(a) AUC on labelled validation patients, and any comparison of it with the conflict-set AUC",
     "touches_fit": False, "touches_selection": True, "selection_family": "planted", "manuscript_source_ids": ["p64-manip-strong-auc", "p64-manip-zero-auc", "p64-manip-weak-auc", "p65-manip-mid30-effect", "p65-manip-mid50-effect", "p65-manip-strong-effect", "p65-manip-weak-effect"],
     "basis": "The nine planted checkpoints select on the canonical string-sorted half, S_str, so the population held out from THEIR selection is R_str, "
              "not R_int (R_int is held out from the original family only). Registered before any planted-family validation AUC exists; none is reported in the manuscript. "
              "the manipulation-check EFFECT (unedited minus edited conflict-set AUC) of the two intermediate strengths, of weak and strong is this quantity's "
              "sibling on the same conflict set (val-report, R_str); the six intermediate checkpoints select on the same canonical half as the nine."},
    {"id": "Q18", "name": "E5 (512 pixels) and 224-pixel micro-batch control checkpoints: ordinary clause (a) and (b) validity on labelled validation patients, and the report half's size",
     "touches_fit": False, "touches_selection": True, "selection_family": "planted", "manuscript_source_ids": ["p83-e5v-dn-orig-n", "p83-e5v-dn-orig-nvalid", "p83-e5v-dn-zero-n", "p83-e5v-dn-zero-nvalid", "p83-e5v-dn-strong-n", "p83-e5v-dn-strong-nvalid", "p83-e5v-dn-auc-min", "p83-e5v-dn-auc-max", "p83-e5v-dn-medp-min", "p83-e5v-dn-ctl-n", "p83-e5v-dn-ctl-nvalid", "p83-e5v-rn-orig-n", "p83-e5v-rn-orig-nvalid", "p83-e5v-rn-zero-n", "p83-e5v-rn-zero-nvalid", "p83-e5v-rn-strong-n", "p83-e5v-rn-strong-nvalid", "p83-e5v-rn-auc-min", "p83-e5v-rn-auc-max", "p83-e5v-rn-medp-min", "p83-e5v-rn-ctl-n", "p83-e5v-rn-ctl-nvalid", "p83-e5v-n-patients", "p83-e5v-n-images"],
     "basis": "All 22 E5 checkpoints and all 8 control checkpoints select on the canonical string-sorted half S_str (every job gates on the logged select-half digest, exit 4 otherwise), so the population held out from THEIR selection is R_str, not R_int, exactly as for Q17. No calibration is fitted, so touches_fit is false. The validity script read R_str (2,311 patients, digest b8f26e9e...); the logged digest equals it for every checkpoint. Registered in the E5 registration ('Validity') before any 512 checkpoint existed; the artifact is a recorded validity.json whose population is checked by the job-time digest, not by a registry claim (unenforced at the artifact level beyond that)."},
]

SELECTION_DIAGNOSTIC_SOURCE_IDS = [
    "p55-gap-densenet", "p55-gap-resnet", "p55-gap-retired", "p55-plateau-densenet", "p55-plateau-resnet",
    "p55-max-diff-int", "p55-min-diff-str", "p55-sel-in-report-patients", "p55-sel-in-report-images",

    "p57-sim-n-draws", "p57-sim-p-any", "p57-sim-p-two", "p57-sim-tail-d", "p57-sim-tail-r", "p57-mech-ks-p", "p57-mech-n-alt", "p57-mech-n-alt-r",
    "p57-mech-sd-ratio-d", "p57-mech-sd-ratio-r", "p57-quarter-others-min-d", "p57-quarter-others-max-d",
    "p57-sel-minus-comp-d", "p57-sel-minus-comp-r", "p57-sel-minus-comp-ret",

    "p60-comp-p-pos", "p60-comp-p-neg", "p60-comp-alpha", "p60-comp-pow-pos-3q", "p60-comp-pow-pos-half", "p60-comp-pow-neg-fifth",
]

_NOT_VALIDATION_DERIVED_IDS = {
    "fresh-composite audit numbers and draw-to-draw numbers (test-split composite pixels, no validation population) and the exposure count; the per-seed summary of those same draws (p66-s1-) and the recipe-level interval over the same draws (p67-)": lambda i: i.startswith(("p64-", "p65-", "p66-s1-", "p67-")) and not i.startswith(("p64-manip-", "p65-manip-")),
    "NEATX coverage of the test-split targets (a metadata join of image and patient ids, no score, no validation population)": lambda i: i.startswith("p69-"),
    "the rewritten text's audit numbers (negative targets and gate positives and donors from the test split; the exploratory positives mix development-half targets with never-selected-half targets, held out from the original checkpoints' selection and not from the planted checkpoints', PC-31), planted-checkpoint contrasts, geometry arithmetic on recorded arm means, population counts and the timing of the gate read; interventional composite measurements, no validation AUC, calibration or clause statistic": lambda i: i.startswith("q-"),
    "the exploratory positives' audit contrasts for the planting ladder, the further seeds, the generative edit and the rendered-tube companion (PC-31; interventional composite and generated-edit measurements on test-split and never-selected-half targets, no validation AUC, calibration or clause statistic of a classifier), the drain detector's own out-of-fold statistics (a different instrument with its own cross-fitted populations, checked by a committed script and not a classifier validation population), and population counts of those sets": lambda i: i.startswith("p83-") and not i.startswith("p83-e5v-"),
    "external literature or dataset-release number": lambda i: i.startswith(("i-", "d-roentmod-", "d-winkler-", "ds-")),
    "split or pool count (manifest facts and population-claim counts, no score involved)": lambda i: i in {
        "m-total-images", "m-total-patients", "m-train-images", "m-train-patients", "m-val-images", "m-val-patients", "m-test-images", "m-test-patients",
        "m-train-pneumo-pos", "m-train-pneumo-prev", "m-gate-dev-images", "m-gate-dev-patients", "m-gate-gate-images", "m-gate-gate-patients", "m-eligible-images", "m-eligible-patients",
        "m-train-primary-patients", "m-train-extended-only-patients", "m-eval-paired-patients", "m-eval-unpaired-patients", "m-train-extended-patients",
        "m-val-select-images", "m-val-select-patients", "m-val-report-images", "m-val-report-patients", "m-overlap-paired-images", "m-overlap-paired-patients",
        "m-overlap-unpaired-images", "m-overlap-unpaired-patients", "m-overlap-union-images", "m-overlap-gate-images", "p55-smoke-paired-images",
        "p55-smoke-unpaired-images", "p55-arm-union-patients", "p55-arm-union-images", "p55-sweep-n-claims", "p55-sweep-holds", "p55-sweep-fails",
        "p55-sweep-notdet", "p55-sweep-tierN", "p56-registry-n-claims", "p56-registry-holds", "p55-drop-in-pool", "m-phaseb-population-n",
        "m-phaseb-population-patients", "m-phaseb-population-pos", "m-phaseb-population-neg",
        "p57-clean-n-patients", "p57-rint-n-patients", "p57-clean-pos-patients",
        "p60-comp-n-q-pos", "p60-comp-n-c-neg", "p60-comp-n-q-neg", "p60-q-n"},
    "interventional or composite measurement (test split or composite pixels, no validation population)": lambda i: i in {
        "c2-densenet-confirm", "c2-densenet-confirm-ci", "c2-resnet-confirm", "d-ins-vs-base-densenet", "d-ctl-vs-base-densenet", "m-insertion-mad",
        "m-control-mad", "d-shuffle-rows",

        "p57-a2-densenet-confirm-margin", "p57-a3-resnet-confirm-margin",

        "p60-int-n-patients", "p60-pf-dn-mde", "p60-pf-dn-se", "p60-pf-dn-mde-ctl", "p60-pf-dn-power", "p60-pf-dn-mde-seed-min", "p60-pf-dn-mde-seed-max",
        "p60-pf-rn-mde-ub", "p60-pf-rn-mde-lb", "p60-pf-rn-mde-seed4", "p60-pf-rn-mde-seed2", "p60-pf-eq-dn-own", "p60-pf-eq-dn-a2", "p60-pf-eq-rn",
        "p60-rn-s4-delta", "p60-rn-s4-margin-reg", "p60-rn-s4-margin-rint", "p60-rn-s4-ci-hi", "p60-rn-s2-delta", "p60-rn-pooled-margin", "p60-rn-ins-base-npos",

        "p60-pf-rn-mde", "p60-pf-rn-mde-ctl", "p60-pf-rn-power", "p60-pf-rn-se",

        "c2-resnet-confirm-ci-lo", "c2-resnet-confirm-ci-hi"},
    "training configuration or training-split quantity": lambda i: i in {
        "m-pos-weight", "m-batch-size", "m-lr", "m-old-lr", "m-densenet-resnet-step", "m-densenet-resnet-gamma", "m-densenet-resnet-epochs"},
    "value predicted from training prevalence, no validation data": lambda i: i == "m-affine-b-predicted",
    "retired pre-Phase-A instrument on its own population (historical contrast)": lambda i: i == "c5-prephaseA-dz",
}

def is_declared_not_validation_derived(entry: dict) -> bool:
    if entry["instrument"] in ("design", "realism-panel") or entry.get("kind") in ("comparison", "table"):

        return True
    return any(pred(entry["id"]) for pred in _NOT_VALIDATION_DERIVED_IDS.values())

def build_table() -> list:
    rows = []
    for q in QUANTITIES:
        row = {k: v for k, v in q.items() if k not in ("qualifier_key",)}
        family = q.get("selection_family", "original")
        assigned = "S_str" if q.get("kind") == "fit_parameter" else assign_population(q["touches_fit"], q["touches_selection"], family)
        row["assigned"] = assigned
        row["held_out_from_family"] = family if q["touches_selection"] else None
        row["assigned_n_patients"] = POPULATIONS[assigned]["n_patients"]
        if q.get("qualifier_key"):
            n = QUALIFYING_PATIENTS[assigned]
            row["n_qualifying_patients"] = n
            row["supportable_at_assigned_population"] = supportable_under_floor(n)
        rows.append(row)
    return rows

HEADER = """# Population assignment for validation-derived quantities

Generated by `analysis/population_assignment.py`; do not edit the table by hand. Committed before any
quantity was recomputed against it (this file's first commit precedes `analysis/assigned_recompute.py` in the git history).

## The rule

Populations, all inside the 4,621-patient validation split V: **S_int** is the half the ten checkpoints were selected on (the population-claims stage's identification);
**S_str** is the half both calibration fits used; **R_int** = V minus S_int (2,311 patients); **R_str** = V minus S_str (2,311; the legacy "val-report");
**C** = V minus both (1,166).

Every quantity declares two flags, and the population is a function of them: **V minus S_int if the quantity touches selection, minus S_str if it
touches the calibration fit.**

- **touches_fit**: the quantity's computation takes as an input a parameter estimated on S_str (a fitted temperature T, or an affine (a, b)).
  Decidable from the code. AUC and every rank-based statistic do not: a strictly monotone transform cannot change them, and all ten fitted T exceed 1
  and all ten affine slopes are positive.
- **touches_selection**: the quantity is an evaluation of a selected checkpoint on labelled validation patients, reported as a characterization of that
  checkpoint on patients it was not developed on. The definition is structural, as the population-claim PC-04 was (the population-claims stage: "the criterion is
  structural, not magnitude-based"), and so every such quantity declares True. There is no per-quantity argument that a value is insensitive to
  selection. That argument is what would turn the assignment into a menu, and its own evidence (a heuristic bound of +0.0036 on argmax-epoch
  optimism for AUC, nothing comparable for any other statistic) does not exist for most quantities.

So the cell "selection only" gives R_int and the cell "fit and selection" gives C. The other two cells (neither; fit only) would give V and R_str; they are
empty for every evaluation quantity, by the second definition, and `tests/test_population_assignment.py` fails if one is ever occupied. The one
quantity kind outside the rule is the fitted parameters themselves (Q14), whose population is S_str by construction.

## What this changes from the clean-population stage, and why it is not a rescue

the clean-population stage assigned C to every recomputed quantity. That was over-broad (Reviewer Correction Record #13): the calibration-fit exclusion applies only to
quantities that take the fit as an input. Under the rule, clause (a), clause (b), the operating points, raw ECE and Brier and the reachability constants
move from C to R_int; only the calibrated ECE, Brier and median negative probability stay on C. The move was made after clause (a) failed on two ResNet-50
seeds on C, and that timing is recorded in the study record, not hidden here. The rule was applied to every quantity, including the ones
it hurts: the population-level gap and the within-patient contrast move to R_int as well, and the second of those cannot be supported there.

## What existed before this file was committed (disclosure, as the clean-population stage's protocol item 8 did)

Clause-(a) AUCs and CIs on R_int (the population-claims stage and the clean-population stage sensitivity check); C values for everything in Q01 to Q10; the recorded val-report (R_str)
and full-validation values of every quantity; the interval-width, random-half, split-mechanism and quarter-table outputs of this pass. Existing but not read by me before
this commit: the R_int clause-(b), Youden, band-threshold and reachability values inside `sensitivity_<arch>.json` (the clean-population stage script computed them and reported only
clause (a)). Not yet existing: R_int values for the gap, raw and calibrated ECE and Brier, the crossing counts at R_int thresholds, and the S_int and R_int parts of
the within-patient primary.

## Judgment calls the rule did not settle (flagged, not resolved silently)

- **J1. The within-patient primary and the crossing patients (Q11).** The rule assigns R_int, where 18 patients qualify, below the registered floor of 20. Under the
  owner's 1.4 the estimate cannot be reported as the primary at 18. The n = 46 full-validation figure has been the registered primary since before the halves existed
  and its double use in selection is disclosed; whether "touches selection" should reach a within-patient difference of two positive films, or only the statistics selection
  optimises, is the owner's reading to make. Until it is made, the 46-patient figure stays in the manuscript labelled as computed on a population that includes selection
  patients, with the R_int patients' sign count and raw distribution beside it, and is not called held-out.
- **J2. The ViT-B/16 AUCs (Q15)** are contaminated by the same overlap and cannot be recomputed on R_int (no per-image predictions on disk). They stay only as the historical
  basis of the retirement decision, labelled as measured on the legacy half, not as a held-out measurement.
- **J3. Phase B (Q16).** The rule would restrict the 159-patient probe pool to its R_int patients. Phase B is closed and halving the pool makes its registered
  probe bar unreachable by an earlier stage arithmetic, so the quantities are not reassigned; Methods already says the pool spans both halves.

## The table
"""

def render_markdown(table: list) -> str:
    lines = [HEADER, "| id | quantity | fit | selection | assigned (patients) | manuscript source ids | supportable at assigned population | judgment |", "|---|---|---|---|---|---|---|---|"]
    for r in table:
        sup = ""
        if "supportable_at_assigned_population" in r:
            sup = f"{r['n_qualifying_patients']} qualifying vs floor {WITHIN_PATIENT_FLOOR}: " + ("yes" if r["supportable_at_assigned_population"] else "**no**")
        elif r["id"] == "Q15":
            sup = "not recomputable"
        lines.append(f"| {r['id']} | {r['name']} | {'yes' if r['touches_fit'] else 'no'} | {'yes' if r['touches_selection'] else 'no'} | "
                     f"{r['assigned']} ({r['assigned_n_patients']:,}) | {', '.join(r['manuscript_source_ids']) or '(text claim; no numeric token)'} | {sup} | {r.get('judgment', '')} |")
    lines += ["", "Reasons for each row's flags:", ""]
    for r in table:
        lines.append(f"- **{r['id']}**: {r['basis']}")
    lines += ["", "Declared measurements of the selection contamination itself (population dictated by the measurement, not held-out claims): "
              + ", ".join(f"`{i}`" for i in SELECTION_DIAGNOSTIC_SOURCE_IDS) + ".", ""]
    return "\n".join(lines)

def main() -> int:
    table = build_table()
    JSON_OUT.write_text(json.dumps({"populations": POPULATIONS, "within_patient_floor": WITHIN_PATIENT_FLOOR, "qualifying_patients": QUALIFYING_PATIENTS,
                                    "quantities": table, "selection_diagnostic_source_ids": SELECTION_DIAGNOSTIC_SOURCE_IDS}, indent=2))
    MD_OUT.write_text(render_markdown(table))
    print(f"wrote {JSON_OUT.relative_to(REPO_ROOT)} and {MD_OUT.relative_to(REPO_ROOT)}: {len(table)} quantities")
    for r in table:
        print(f"  {r['id']}  fit={r['touches_fit']!s:5} sel={r['touches_selection']!s:5} -> {r['assigned']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
