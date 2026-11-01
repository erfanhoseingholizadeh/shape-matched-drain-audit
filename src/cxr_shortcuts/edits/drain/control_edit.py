# The tissue control: the same footprint and place as the tube, filled with another patient's pixels.
# Run: library module, imported by the scripts

from __future__ import annotations

import numpy as np

from cxr_shortcuts.edits.drain.insertion import EditResult, log_additive_blend_patch, poisson_blend_patch

def apply_control_edit(
    target_image: np.ndarray,
    control_source_image: np.ndarray,
    insertion_result: EditResult,
    blend_mode: str = "log_additive",
) -> EditResult:
    if control_source_image.shape != target_image.shape:
        raise ValueError("control_source_image and target_image must share the same shape")

    row_off, col_off = insertion_result.offset
    h, w = insertion_result.local_mask.shape
    control_patch = control_source_image[row_off : row_off + h, col_off : col_off + w]

    if blend_mode == "poisson":
        blended = poisson_blend_patch(target_image, control_patch, insertion_result.local_mask, insertion_result.offset)
    elif blend_mode == "log_additive":
        control_background_level = float(control_patch.mean())
        blended = log_additive_blend_patch(
            target_image,
            control_patch,
            insertion_result.local_mask,
            insertion_result.offset,
            control_background_level,
        )
    else:
        raise ValueError(f"unknown blend_mode {blend_mode!r}, expected 'poisson' or 'log_additive'")
    return EditResult(
        image=blended,
        placement_mask=insertion_result.placement_mask,
        offset=insertion_result.offset,
        local_mask=insertion_result.local_mask,
    )
