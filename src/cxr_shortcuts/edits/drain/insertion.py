# Inserts a patch into a film, including the log-domain blend used for the tube and the tissue control.
# Run: library module, imported by the scripts

from __future__ import annotations

import dataclasses
from typing import Optional

import cv2
import numpy as np

from cxr_shortcuts.edits.drain.geometry import place_mask, sample_valid_placement

class DonorTargetPatientOverlapError(ValueError):
    pass

@dataclasses.dataclass(frozen=True)
class EditResult:
    image: np.ndarray
    placement_mask: np.ndarray
    offset: tuple[int, int]
    local_mask: np.ndarray

    relaxed: tuple[str, ...] = ()
    n_candidates: Optional[int] = None

def _check_patient_overlap(**patient_ids: Optional[str]) -> None:
    seen: dict[str, str] = {}
    for role, patient_id in patient_ids.items():
        if patient_id is None:
            continue
        if patient_id in seen:
            raise DonorTargetPatientOverlapError(
                f"{role} patient {patient_id!r} matches the {seen[patient_id]} patient -- this pairing "
                f"should have been excluded by upstream pool construction (see "
                f"docs/leakage_register.md #3 and data/leakage.py::assert_no_donor_source_audit_target_overlap)"
            )
        seen[patient_id] = role

def insert_drain(
    target_image: np.ndarray,
    lung_mask: np.ndarray,
    donor_patch: np.ndarray,
    donor_mask: np.ndarray,
    rng: np.random.Generator,
    pathology_mask: Optional[np.ndarray] = None,
    max_tries: int = 200,
    donor_patient_id: Optional[str] = None,
    target_patient_id: Optional[str] = None,
    control_source_patient_id: Optional[str] = None,
    blend_mode: str = "poisson",
    donor_background_level: Optional[float] = None,
) -> EditResult:
    _check_shapes(target_image, lung_mask, donor_patch, donor_mask)
    _check_patient_overlap(donor=donor_patient_id, target=target_patient_id, control_source=control_source_patient_id)
    if blend_mode == "log_additive" and donor_background_level is None:
        raise ValueError("donor_background_level is required when blend_mode='log_additive'")
    offset = sample_valid_placement(donor_mask, lung_mask, rng, pathology_mask=pathology_mask, max_tries=max_tries)
    if blend_mode == "poisson":
        blended = poisson_blend_patch(target_image, donor_patch, donor_mask, offset)
    elif blend_mode == "log_additive":
        blended = log_additive_blend_patch(target_image, donor_patch, donor_mask, offset, donor_background_level)
    else:
        raise ValueError(f"unknown blend_mode {blend_mode!r}, expected 'poisson' or 'log_additive'")
    placement_mask = place_mask(donor_mask, target_image.shape, offset)
    return EditResult(image=blended, placement_mask=placement_mask, offset=offset, local_mask=donor_mask)

def hard_paste_patch(
    target_image: np.ndarray,
    donor_patch: np.ndarray,
    local_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    row_off, col_off = offset
    h, w = local_mask.shape
    result = target_image.copy()
    region = result[row_off : row_off + h, col_off : col_off + w]
    result[row_off : row_off + h, col_off : col_off + w] = np.where(local_mask, donor_patch, region)
    return result

def log_additive_blend_patch(
    target_image: np.ndarray,
    donor_patch: np.ndarray,
    local_mask: np.ndarray,
    offset: tuple[int, int],
    background_level: float,
    eps: float = 1.0,
) -> np.ndarray:
    row_off, col_off = offset
    h, w = local_mask.shape
    target_region = target_image[row_off : row_off + h, col_off : col_off + w].astype(np.float64)
    donor_region = donor_patch.astype(np.float64)

    excess_log = np.log(donor_region + eps) - np.log(background_level + eps)
    blended_log = np.log(target_region + eps) + excess_log

    blended_region = np.round(np.clip(np.exp(blended_log) - eps, 0, 255))

    result = target_image.copy()
    region = result[row_off : row_off + h, col_off : col_off + w]
    result[row_off : row_off + h, col_off : col_off + w] = np.where(local_mask, blended_region, region)
    return result.astype(np.uint8)

def poisson_blend_patch(
    target_image: np.ndarray,
    donor_patch: np.ndarray,
    local_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    src = hard_paste_patch(target_image, donor_patch, local_mask, offset)
    placement_mask = place_mask(local_mask, target_image.shape, offset)

    ys, xs = np.nonzero(placement_mask)
    center = (int(xs.mean()), int(ys.mean()))

    src_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    dst_bgr = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
    mask_u8 = placement_mask.astype(np.uint8) * 255

    try:
        blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_u8, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return src

    return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2GRAY)

def _check_shapes(
    target_image: np.ndarray,
    lung_mask: np.ndarray,
    donor_patch: np.ndarray,
    donor_mask: np.ndarray,
) -> None:
    if target_image.shape != lung_mask.shape:
        raise ValueError("target_image and lung_mask must share the same shape")
    if donor_patch.shape != donor_mask.shape:
        raise ValueError("donor_patch and donor_mask must share the same shape")
