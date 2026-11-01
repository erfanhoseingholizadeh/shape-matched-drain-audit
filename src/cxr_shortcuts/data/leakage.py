# Checks the split for patient and image leakage before training.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

from cxr_shortcuts.data.manifests import assert_no_discriminator_arm_leakage, assert_no_patient_leakage
from cxr_shortcuts.edits.drain.donor_library import assert_no_donor_leakage

ALL_SURFACES = frozenset(
    {

        "classifier_patient_split",
        "donor_train_eval_donor_id_split",
        "donor_train_eval_patient_split",
        "donor_source_vs_audit_target_patient_overlap",
        "gate_subset_locked",
        "donor_eval_outside_training_split",

        "discriminator_patient_split",

        "donor_pilot_flagged",
        "screener_patient_split",
        "screener_audit_target_overlap",

        "discriminator_donor_disjointness",
        "dfr_eval_disjointness",
        "jtt_train_only",
        "checkpoint_selection_validation_only",
        "operating_point_validation_only",
        "normalization_stats_train_only",
        "siim_nih_patient_mapping",

        "control_source_screener_confirmed_drain_free",

        "dev_slice_disjoint_from_eval_arms",
    }
)

def assert_no_donor_patient_leakage(donor_manifest: pd.DataFrame, patient_id_col: str = "patient_id") -> None:
    train_patients = set(donor_manifest.loc[donor_manifest["split"] == "train", patient_id_col])
    eval_patients = set(donor_manifest.loc[donor_manifest["split"] == "eval", patient_id_col])
    overlap = train_patients & eval_patients
    if overlap:
        raise AssertionError(
            f"donor leakage at the PATIENT level between train/eval splits: {sorted(overlap)}"
        )

def assert_no_donor_source_audit_target_overlap(
    donor_source_patient_ids: Iterable[str],
    audit_target_patient_ids: Iterable[str],
) -> None:
    overlap = set(donor_source_patient_ids) & set(audit_target_patient_ids)
    if overlap:
        preview = sorted(overlap)[:10]
        suffix = "..." if len(overlap) > 10 else ""
        raise AssertionError(
            f"{len(overlap)} patient(s) supply both a donor-source image and an "
            f"audit-target image: {preview}{suffix}"
        )

def assert_eval_donors_outside_training_split(
    donor_manifest: pd.DataFrame,
    classifier_patient_split: Mapping[str, str],
    patient_id_col: str = "patient_id",
) -> None:
    eval_patients = set(donor_manifest.loc[donor_manifest["split"] == "eval", patient_id_col])
    training_patients = {p for p, split in classifier_patient_split.items() if split == "train"}
    overlap = eval_patients & training_patients
    if overlap:
        raise AssertionError(
            f"eval donor(s) sourced from classifier-TRAINING-split patient(s): {sorted(overlap)} -- "
            f"Δp for these donors partly measures memorization, not causal reliance"
        )

def assert_dev_slice_disjoint_from_eval_arms(
    dev_slice_patient_ids: Iterable[str],
    eval_arm_patient_ids: Iterable[str],
) -> None:
    overlap = set(dev_slice_patient_ids) & set(eval_arm_patient_ids)
    if overlap:
        raise AssertionError(
            f"dev-slice patient(s) also present in a registered eval arm (eval_paired/"
            f"eval_unpaired_reference): {sorted(overlap)} -- this would let rendering-"
            f"iteration choices see the population eval_paired's single pre-registered "
            f"measurement is supposed to be protected from"
        )

def assert_no_leakage(
    *,
    patient_manifest: Optional[pd.DataFrame] = None,
    discriminator_manifest: Optional[pd.DataFrame] = None,
    donor_manifest: Optional[pd.DataFrame] = None,
    donor_manifest_with_patient_ids: Optional[pd.DataFrame] = None,
    classifier_patient_split: Optional[Mapping[str, str]] = None,
    donor_source_patient_ids: Optional[Iterable[str]] = None,
    audit_target_patient_ids: Optional[Iterable[str]] = None,
    gate_subset_locked: Optional[bool] = None,
    dev_slice_patient_ids: Optional[Iterable[str]] = None,
    eval_arm_patient_ids: Optional[Iterable[str]] = None,
    acknowledged_unknown: frozenset = frozenset(),
) -> dict:
    unknown_names = acknowledged_unknown - ALL_SURFACES
    if unknown_names:
        raise ValueError(
            f"acknowledged_unknown names surface(s) not in the registry: {sorted(unknown_names)} "
            f"-- typo? see cxr_shortcuts.data.leakage.ALL_SURFACES"
        )

    checked: dict[str, str] = {}

    if patient_manifest is not None:
        assert_no_patient_leakage(patient_manifest)
        checked["classifier_patient_split"] = "checked_clean"

    if discriminator_manifest is not None:
        assert_no_discriminator_arm_leakage(discriminator_manifest)
        checked["discriminator_patient_split"] = "checked_clean"

    if donor_manifest is not None:
        assert_no_donor_leakage(donor_manifest)
        checked["donor_train_eval_donor_id_split"] = "checked_clean"

    if donor_manifest_with_patient_ids is not None:
        assert_no_donor_patient_leakage(donor_manifest_with_patient_ids)
        checked["donor_train_eval_patient_split"] = "checked_clean"

        if classifier_patient_split is not None:
            assert_eval_donors_outside_training_split(donor_manifest_with_patient_ids, classifier_patient_split)
            checked["donor_eval_outside_training_split"] = "checked_clean"

    if donor_source_patient_ids is not None and audit_target_patient_ids is not None:
        assert_no_donor_source_audit_target_overlap(donor_source_patient_ids, audit_target_patient_ids)
        checked["donor_source_vs_audit_target_patient_overlap"] = "checked_clean"

    if gate_subset_locked is not None:
        if not gate_subset_locked:
            raise AssertionError(
                "gate_subset_locked=False -- the locked gate subset has been touched "
                "during development; this violates the pre-registration discipline in "
                "HANDOFF.md's 'Stage 0 Gate -- Pre-registered' section"
            )
        checked["gate_subset_locked"] = "checked_clean"

    if dev_slice_patient_ids is not None and eval_arm_patient_ids is not None:
        assert_dev_slice_disjoint_from_eval_arms(dev_slice_patient_ids, eval_arm_patient_ids)
        checked["dev_slice_disjoint_from_eval_arms"] = "checked_clean"

    for surface in acknowledged_unknown:
        checked.setdefault(surface, "acknowledged_unknown")

    missing = ALL_SURFACES - checked.keys()
    if missing:
        raise ValueError(
            f"leakage preflight: {len(missing)} registered surface(s) neither checked nor "
            f"acknowledged: {sorted(missing)} -- pass the real artifact to check it for real, "
            f"or add it to acknowledged_unknown if this run genuinely can't check it yet"
        )

    return checked
