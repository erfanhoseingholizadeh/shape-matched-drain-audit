# The interventional audit loop: scores a film before and after an edit.
# Run: library module, imported by the scripts

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from cxr_shortcuts.data.leakage import assert_no_leakage
from cxr_shortcuts.edits.drain.control_edit import apply_control_edit
from cxr_shortcuts.edits.drain.encoding_calibration import (
    donor_background_level as compute_donor_background_level,
)
from cxr_shortcuts.edits.drain.geometry import (
    NoValidPlacementError,
    placement_lung_side,
    valid_placement_mask,
)
from cxr_shortcuts.edits.drain.insertion import insert_drain
from cxr_shortcuts.eval.interventional import (
    mean_absolute_perturbation,
    mean_probability_shift,
    mean_ssim_perturbation,
    shortcut_induced_fpr,
)
from cxr_shortcuts.eval.schema import ResultRow, append_result, config_hash, require_git_sha

MIN_STRATUM_N = 20
DROP_RATE_FLAG_ABS_THRESHOLD = 0.05
DROP_RATE_FLAG_RATIO_THRESHOLD = 2.0

CONTROL_VALIDITY_RATIO_THRESHOLD = 10.0

PERTURBATION_MAD_FLOOR = 1.0

class DegenerateInsertionError(RuntimeError):
    pass

class TargetShapeMismatchError(ValueError):
    pass

def _validated_per_image_shapes(
    image: np.ndarray, left: np.ndarray, right: np.ndarray, control_source_image: np.ndarray
) -> np.ndarray:
    if left.shape != right.shape or left.shape != image.shape or control_source_image.shape != image.shape:
        raise TargetShapeMismatchError(
            f"image/left/right lung mask/control_source_image shapes must all match: "
            f"image={image.shape}, left={left.shape}, right={right.shape}, "
            f"control_source_image={control_source_image.shape}"
        )
    return left | right

@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, images: torch.Tensor) -> np.ndarray:
    model.eval()
    params = list(model.parameters())
    if params:
        images = images.to(params[0].device)
    logits = model(images).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()

def _stratum_counts(dropped: Sequence[bool], attempted: Sequence[bool], covariate: Sequence[bool]) -> tuple[int, int]:
    total = sum(1 for a, c in zip(attempted, covariate) if a and c)
    n_dropped = sum(1 for d, a, c in zip(dropped, attempted, covariate) if a and c and d)
    return n_dropped, total

def _stratified_drop_rows(
    prefix: str,
    dropped: Sequence[bool],
    attempted: Sequence[bool],
    left_feasible: Sequence[bool],
    right_feasible: Sequence[bool],
    small_area: Sequence[bool],
) -> tuple[list[tuple[str, float]], dict[str, list[tuple[int, float]]]]:
    strata = {
        "left_lung_feasible": ("laterality", left_feasible),
        "right_lung_feasible": ("laterality", right_feasible),
        "small_lung_area": ("lung_area", small_area),
        "large_lung_area": ("lung_area", [not s for s in small_area]),
    }
    rows: list[tuple[str, float]] = []
    rates_by_axis: dict[str, list[tuple[int, float]]] = {"laterality": [], "lung_area": []}
    for stratum_name, (axis, covariate) in strata.items():
        n_dropped, total = _stratum_counts(dropped, attempted, covariate)
        rate = (n_dropped / total) if total else 0.0
        rows.append((f"{prefix}_dropped_count_{stratum_name}", float(n_dropped)))
        rows.append((f"{prefix}_total_count_{stratum_name}", float(total)))
        rows.append((f"{prefix}_drop_rate_{stratum_name}", float(rate)))
        rates_by_axis[axis].append((total, rate))
    return rows, rates_by_axis

def _axis_imbalance_flag(rate_pairs: Sequence[tuple[int, float]]) -> bool:
    evaluable = [rate for n, rate in rate_pairs if n >= MIN_STRATUM_N]
    if any(rate > DROP_RATE_FLAG_ABS_THRESHOLD for rate in evaluable):
        return True
    if len(evaluable) == 2:
        lo, hi = min(evaluable), max(evaluable)
        if hi > 0 and (lo == 0 or hi / lo > DROP_RATE_FLAG_RATIO_THRESHOLD):
            return True
    return False

def _placement_side_imbalance_flag(n_left: int, n_right: int) -> bool:
    total = n_left + n_right
    if total < MIN_STRATUM_N:
        return False
    lo, hi = min(n_left, n_right), max(n_left, n_right)
    if hi == 0:
        return False
    return lo == 0 or hi / lo > DROP_RATE_FLAG_RATIO_THRESHOLD

def run_stage0_audit(
    model: torch.nn.Module,
    target_images: Sequence[np.ndarray],
    left_lung_masks: Sequence[np.ndarray],
    right_lung_masks: Sequence[np.ndarray],
    donor_patch: np.ndarray,
    donor_mask: np.ndarray,
    control_source_image: np.ndarray,
    to_model_input: Callable[[np.ndarray], torch.Tensor],
    rng: np.random.Generator,
    results_path: Path,
    run_id: str,
    seed: int,
    backbone: str,
    dataset: str,
    operating_point_threshold: float = 0.5,
    donor_patient_id: Optional[str] = None,
    target_patient_ids: Optional[Sequence[str]] = None,
    control_source_patient_id: Optional[str] = None,
    blend_mode: str = "log_additive",
    donor_background_level: Optional[float] = None,
) -> dict:

    if donor_patch.shape != donor_mask.shape:
        raise ValueError("donor_patch and donor_mask must share the same shape")

    if not donor_mask.any():
        raise ValueError("donor_mask has zero True pixels -- no drain content to insert")

    if blend_mode == "log_additive" and donor_background_level is None:
        donor_background_level = compute_donor_background_level(donor_patch, donor_mask)

    if target_patient_ids is not None and len(target_patient_ids) != len(target_images):
        raise ValueError(
            f"target_patient_ids has {len(target_patient_ids)} entries but target_images has "
            f"{len(target_images)} -- they must be parallel sequences"
        )

    assert_no_leakage(
        acknowledged_unknown=frozenset(
            {
                "classifier_patient_split",
                "donor_train_eval_donor_id_split",
                "donor_train_eval_patient_split",
                "donor_source_vs_audit_target_patient_overlap",
                "gate_subset_locked",
                "donor_eval_outside_training_split",
                "donor_pilot_flagged",
                "screener_patient_split",
                "screener_audit_target_overlap",
                "discriminator_patient_split",
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
    )

    n = len(target_images)
    shape_ok = [False] * n
    lung_areas = [0] * n
    left_feasible = [False] * n
    right_feasible = [False] * n
    for i, (image, left, right) in enumerate(zip(target_images, left_lung_masks, right_lung_masks)):
        try:
            union = _validated_per_image_shapes(image, left, right, control_source_image)
        except TargetShapeMismatchError:
            continue
        shape_ok[i] = True
        lung_areas[i] = int(union.sum())
        left_feasible[i] = bool(valid_placement_mask(donor_mask, left).any())
        right_feasible[i] = bool(valid_placement_mask(donor_mask, right).any())

    n_dropped_shape_mismatch = sum(1 for ok in shape_ok if not ok)
    usable_areas = [area for area, ok in zip(lung_areas, shape_ok) if ok]
    area_median = float(np.median(usable_areas)) if usable_areas else 0.0
    small_area = [area < area_median for area in lung_areas]

    kept_images = []
    kept_insertion_images = []
    kept_control_images = []
    kept_realized_sides = []
    kept_placement_masks = []
    insertion_dropped = [False] * n

    insertion_attempted = list(shape_ok)
    control_attempted = [False] * n
    control_dropped = [False] * n
    n_relaxed_intensity = 0
    n_relaxed_position = 0
    for i, (image, left, right) in enumerate(zip(target_images, left_lung_masks, right_lung_masks)):
        if not shape_ok[i]:
            continue
        try:
            insertion = insert_drain(
                image,
                left | right,
                donor_patch,
                donor_mask,
                rng,
                donor_patient_id=donor_patient_id,
                target_patient_id=(target_patient_ids[i] if target_patient_ids is not None else None),
                control_source_patient_id=control_source_patient_id,
                blend_mode=blend_mode,
                donor_background_level=donor_background_level,
            )
        except NoValidPlacementError:
            insertion_dropped[i] = True
            continue

        control_attempted[i] = True

        realized_side = placement_lung_side(donor_mask, insertion.offset, left, right)
        if realized_side is None:
            control_dropped[i] = True
            continue

        control = apply_control_edit(image, control_source_image, insertion, blend_mode=blend_mode)

        if "position" in control.relaxed:
            n_relaxed_position += 1
        elif "intensity" in control.relaxed:
            n_relaxed_intensity += 1
        kept_images.append(image)
        kept_insertion_images.append(insertion.image)
        kept_control_images.append(control.image)
        kept_realized_sides.append(realized_side)

        kept_placement_masks.append(insertion.placement_mask)

    n_dropped_insertion = sum(insertion_dropped)
    n_dropped_control = sum(control_dropped)

    if not kept_images:
        raise RuntimeError(
            f"all {len(target_images)} target images were dropped ({n_dropped_shape_mismatch} at "
            f"shape validation, {n_dropped_insertion} at insertion placement, {n_dropped_control} "
            f"at control placement) -- nothing left to measure"
        )

    probs_before = predict_probabilities(model, torch.stack([to_model_input(img) for img in kept_images]))
    probs_after_insertion = predict_probabilities(
        model, torch.stack([to_model_input(img) for img in kept_insertion_images])
    )
    probs_after_control = predict_probabilities(
        model, torch.stack([to_model_input(img) for img in kept_control_images])
    )

    insertion_shift = mean_probability_shift(probs_before, probs_after_insertion)
    control_shift = mean_probability_shift(probs_before, probs_after_control)
    insertion_fpr = shortcut_induced_fpr(probs_before, probs_after_insertion, operating_point_threshold)

    insertion_perturbation_mad = mean_absolute_perturbation(kept_images, kept_insertion_images, kept_placement_masks)
    control_perturbation_mad = mean_absolute_perturbation(kept_images, kept_control_images, kept_placement_masks)
    insertion_perturbation_ssim_delta = mean_ssim_perturbation(kept_images, kept_insertion_images)
    control_perturbation_ssim_delta = mean_ssim_perturbation(kept_images, kept_control_images)

    perturbation_mad_ratio = (
        float("inf") if control_perturbation_mad == 0 else insertion_perturbation_mad / control_perturbation_mad
    )
    perturbation_ssim_ratio = (
        float("inf")
        if control_perturbation_ssim_delta == 0
        else insertion_perturbation_ssim_delta / control_perturbation_ssim_delta
    )
    control_validity_flag = (
        perturbation_mad_ratio > CONTROL_VALIDITY_RATIO_THRESHOLD
        or perturbation_ssim_ratio > CONTROL_VALIDITY_RATIO_THRESHOLD
    )

    if insertion_perturbation_mad < PERTURBATION_MAD_FLOOR:
        raise DegenerateInsertionError(
            f"mean insertion perturbation ({insertion_perturbation_mad:.4f}) is below "
            f"PERTURBATION_MAD_FLOOR ({PERTURBATION_MAD_FLOOR}) -- the edit pipeline is not "
            f"rendering visible content under blend_mode={blend_mode!r} across the kept "
            f"population of {len(kept_images)} images. This halts the run rather than reporting "
            f"a Δp that would be indistinguishable from 'no edit happened'."
        )
    control_perturbation_floor_flag = control_perturbation_mad < PERTURBATION_MAD_FLOOR

    insertion_perturbation_floor_flag = False

    insertion_rows, insertion_rates = _stratified_drop_rows(
        "insertion_placement", insertion_dropped, insertion_attempted, left_feasible, right_feasible, small_area
    )
    control_rows, control_rates = _stratified_drop_rows(
        "control_placement", control_dropped, control_attempted, left_feasible, right_feasible, small_area
    )
    stratification_flag = any(
        _axis_imbalance_flag(rates["laterality"]) or _axis_imbalance_flag(rates["lung_area"])
        for rates in (insertion_rates, control_rates)
    )

    n_kept = len(kept_images)
    n_realized_left = sum(1 for s in kept_realized_sides if s == "left")
    n_realized_right = sum(1 for s in kept_realized_sides if s == "right")
    side_imbalance_flag = _placement_side_imbalance_flag(n_realized_left, n_realized_right)

    config = {"backbone": backbone, "dataset": dataset, "seed": seed}
    resolved_config_hash = config_hash(config)
    resolved_git_sha = require_git_sha()
    for metric_name, value in (
        ("insertion_probability_shift", insertion_shift),
        ("control_probability_shift", control_shift),
        ("shortcut_induced_fpr", insertion_fpr),
        ("insertion_perturbation_mad", insertion_perturbation_mad),
        ("control_perturbation_mad", control_perturbation_mad),
        ("perturbation_mad_ratio", perturbation_mad_ratio),
        ("insertion_perturbation_ssim_delta", insertion_perturbation_ssim_delta),
        ("control_perturbation_ssim_delta", control_perturbation_ssim_delta),
        ("perturbation_ssim_ratio", perturbation_ssim_ratio),
        ("control_validity_flag", float(control_validity_flag)),
        ("insertion_perturbation_floor_flag", float(insertion_perturbation_floor_flag)),
        ("control_perturbation_floor_flag", float(control_perturbation_floor_flag)),
        ("shape_mismatch_dropped_count", float(n_dropped_shape_mismatch)),
        ("insertion_placement_dropped_count", float(n_dropped_insertion)),
        ("control_placement_dropped_count", float(n_dropped_control)),
        ("control_placement_relaxed_intensity_count", float(n_relaxed_intensity)),
        ("control_placement_relaxed_position_count", float(n_relaxed_position)),
        *insertion_rows,
        *control_rows,
        ("lung_area_median_split_pixels", area_median),
        ("placement_drop_stratification_flag", float(stratification_flag)),
        ("realized_placement_side_count_left", float(n_realized_left)),
        ("realized_placement_side_count_right", float(n_realized_right)),
        ("realized_placement_side_rate_left", n_realized_left / n_kept),
        ("realized_placement_side_rate_right", n_realized_right / n_kept),
        ("placement_side_imbalance_flag", float(side_imbalance_flag)),
    ):
        append_result(
            ResultRow(
                run_id=run_id,
                method="erm",
                seed=seed,
                backbone=backbone,
                dataset=dataset,
                split="test",
                metric=metric_name,
                value=value,
                config_hash=resolved_config_hash,
                git_sha=resolved_git_sha,
                timestamp="",
            ),
            results_path,
        )

    return {
        "insertion_shift": insertion_shift,
        "control_shift": control_shift,
        "insertion_fpr": insertion_fpr,
        "insertion_perturbation_mad": insertion_perturbation_mad,
        "control_perturbation_mad": control_perturbation_mad,
        "perturbation_mad_ratio": perturbation_mad_ratio,
        "insertion_perturbation_ssim_delta": insertion_perturbation_ssim_delta,
        "control_perturbation_ssim_delta": control_perturbation_ssim_delta,
        "perturbation_ssim_ratio": perturbation_ssim_ratio,
        "control_validity_flag": float(control_validity_flag),
        "insertion_perturbation_floor_flag": float(insertion_perturbation_floor_flag),
        "control_perturbation_floor_flag": float(control_perturbation_floor_flag),
        "n_kept": n_kept,
        "n_dropped_shape_mismatch": n_dropped_shape_mismatch,
        "n_dropped_insertion": n_dropped_insertion,
        "n_dropped_control": n_dropped_control,
        "n_relaxed_intensity": n_relaxed_intensity,
        "n_relaxed_position": n_relaxed_position,
        "lung_area_median_split_pixels": area_median,
        "placement_drop_stratification_flag": float(stratification_flag),
        "n_realized_left": n_realized_left,
        "n_realized_right": n_realized_right,
        "placement_side_imbalance_flag": float(side_imbalance_flag),
    }
