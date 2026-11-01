# Settles the pixel encoding assumed by the log-domain blend.
# Run: library module, imported by the scripts

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from cxr_shortcuts.data.image_io import read_image_gray
from cxr_shortcuts.edits.drain.donor_library import resolve_donor_paths
from cxr_shortcuts.edits.drain.insertion import insert_drain
from cxr_shortcuts.eval.interventional import mean_absolute_perturbation, mean_ssim_perturbation

DEFAULT_N_BINS = 8

def local_background_pairs(
    patch: np.ndarray,
    mask: np.ndarray,
    erosion_px: int = 0,
    guard_band_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if erosion_px < 0 or guard_band_px < 0:
        raise ValueError(f"erosion_px/guard_band_px must be >= 0, got {erosion_px}, {guard_band_px}")

    if not mask.any():
        raise ValueError("mask has no True pixels -- nothing to sample")
    if mask.all():
        raise ValueError(
            "mask has no False pixels -- no local background in this crop to pair against "
            "(the donor's tight bounding-box crop happens to be entirely mask)"
        )

    interior_mask = binary_erosion(mask, iterations=erosion_px) if erosion_px > 0 else mask
    if not interior_mask.any():
        raise ValueError(
            f"eroding the mask by {erosion_px}px left no interior drain pixels -- this donor's mask is too "
            f"thin to sample with this much erosion"
        )
    dilated_mask = binary_dilation(mask, iterations=guard_band_px) if guard_band_px > 0 else mask
    if dilated_mask.all():
        raise ValueError(
            f"dilating the mask by {guard_band_px}px guard band leaves no background pixels anywhere in "
            f"this crop -- this donor's crop is too small/tight for this much guard band"
        )

    _, indices = distance_transform_edt(dilated_mask, return_indices=True)
    rows, cols = np.nonzero(interior_mask)
    bg_rows = indices[0][rows, cols]
    bg_cols = indices[1][rows, cols]

    drain_values = patch[rows, cols].astype(np.float64)
    background_values = patch[bg_rows, bg_cols].astype(np.float64)
    return drain_values, background_values

def donor_background_level(
    patch: np.ndarray, mask: np.ndarray, erosion_px: int = 0, guard_band_px: int = 0
) -> float:
    _, background_values = local_background_pairs(patch, mask, erosion_px=erosion_px, guard_band_px=guard_band_px)
    return float(np.median(background_values))

def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    slope, _intercept = np.polyfit(x, y, 1)
    return float(slope)

def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = values.mean()
    if mean == 0:
        return float("inf")
    return float(values.std() / abs(mean))

@dataclasses.dataclass(frozen=True)
class EncodingFlatnessReport:

    n_pairs: int
    n_bins_used: int
    cv_ratio: float
    cv_difference: float
    slope_ratio: float
    slope_difference: float
    flatter_relationship: str

def encoding_flatness_report(
    drain_values: np.ndarray,
    background_values: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> EncodingFlatnessReport:
    drain_values = np.asarray(drain_values, dtype=np.float64)
    background_values = np.asarray(background_values, dtype=np.float64)
    if drain_values.shape != background_values.shape:
        raise ValueError(
            f"drain_values and background_values must be parallel arrays, got shapes "
            f"{drain_values.shape} and {background_values.shape}"
        )
    n = drain_values.shape[0]
    if n == 0:
        raise ValueError("no (drain, background) pairs to diagnose")

    ratio = drain_values / background_values
    difference = drain_values - background_values

    edges = np.unique(np.quantile(background_values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError(
            f"background intensity has too little spread ({len(edges)} distinct quantile edges) "
            f"to bin meaningfully -- need at least 2 usable bins"
        )
    bin_idx = np.clip(np.digitize(background_values, edges[1:-1]), 0, len(edges) - 2)

    bin_centers, ratio_means, diff_means = [], [], []
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        if not sel.any():
            continue
        bin_centers.append(float(background_values[sel].mean()))
        ratio_means.append(float(ratio[sel].mean()))
        diff_means.append(float(difference[sel].mean()))

    if len(bin_centers) < 2:
        raise ValueError(f"only {len(bin_centers)} non-empty bin(s) after binning -- need at least 2")

    bin_centers_arr = np.array(bin_centers)
    ratio_means_arr = np.array(ratio_means)
    diff_means_arr = np.array(diff_means)

    cv_ratio = _coefficient_of_variation(ratio_means_arr)
    cv_difference = _coefficient_of_variation(diff_means_arr)

    return EncodingFlatnessReport(
        n_pairs=n,
        n_bins_used=len(bin_centers),
        cv_ratio=cv_ratio,
        cv_difference=cv_difference,
        slope_ratio=_fit_slope(bin_centers_arr, ratio_means_arr),
        slope_difference=_fit_slope(bin_centers_arr, diff_means_arr),
        flatter_relationship="ratio" if cv_ratio < cv_difference else "difference",
    )

def _load_donor_patch_and_mask(donor_record: Mapping) -> tuple[np.ndarray, np.ndarray]:
    patch = read_image_gray(donor_record["crop_path"])
    mask = np.load(donor_record["mask_path"])
    return patch, mask

def donor_library_encoding_report(
    records: pd.DataFrame,
    library_root,
    n_bins: int = DEFAULT_N_BINS,
    erosion_px: int = 0,
    guard_band_px: int = 0,
) -> dict:
    resolved = resolve_donor_paths(records, library_root)
    if len(resolved) == 0:
        raise ValueError("donor records table is empty -- no encoding report to compute")

    all_drain, all_background = [], []
    per_donor_background_level: dict[str, float] = {}
    per_donor_flatness_reports: dict[str, EncodingFlatnessReport] = {}
    donors_with_insufficient_spread: list[str] = []
    for _, row in resolved.iterrows():
        donor_id = str(row["donor_id"])
        patch, mask = _load_donor_patch_and_mask(row)
        drain_values, background_values = local_background_pairs(
            patch, mask, erosion_px=erosion_px, guard_band_px=guard_band_px
        )
        all_drain.append(drain_values)
        all_background.append(background_values)
        per_donor_background_level[donor_id] = float(np.median(background_values))
        try:
            per_donor_flatness_reports[donor_id] = encoding_flatness_report(
                drain_values, background_values, n_bins=n_bins
            )
        except ValueError:
            donors_with_insufficient_spread.append(donor_id)

    pooled_drain = np.concatenate(all_drain)
    pooled_background = np.concatenate(all_background)
    pooled_flatness = encoding_flatness_report(pooled_drain, pooled_background, n_bins=n_bins)

    per_donor_cv_ratio_values = [r.cv_ratio for r in per_donor_flatness_reports.values()]
    per_donor_cv_difference_values = [r.cv_difference for r in per_donor_flatness_reports.values()]
    vote_counts = {"ratio": 0, "difference": 0}
    for r in per_donor_flatness_reports.values():
        vote_counts[r.flatter_relationship] += 1

    return {
        "n_donors": len(resolved),
        "pooled_flatness_report": pooled_flatness,
        "per_donor_flatness_reports": per_donor_flatness_reports,
        "donors_with_insufficient_spread": donors_with_insufficient_spread,
        "per_donor_cv_ratio_values": per_donor_cv_ratio_values,
        "per_donor_cv_difference_values": per_donor_cv_difference_values,
        "per_donor_flatter_relationship_vote_counts": vote_counts,
        "donor_background_level": per_donor_background_level,
    }

def guard_band_sensitivity_report(
    records: pd.DataFrame,
    library_root,
    guard_band_widths_px: tuple[int, ...] = (2, 4, 8),
    erosion_px: int = 1,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    per_width_verdict: dict[int, str] = {}
    per_width_report: dict[int, dict] = {}
    widths_that_raised: dict[int, str] = {}
    for width in guard_band_widths_px:
        try:
            report = donor_library_encoding_report(
                records, library_root, n_bins=n_bins, erosion_px=erosion_px, guard_band_px=width
            )
        except ValueError as exc:
            widths_that_raised[width] = str(exc)
            continue
        per_width_report[width] = report
        per_width_verdict[width] = report["pooled_flatness_report"].flatter_relationship

    if not per_width_verdict:
        raise ValueError(
            f"every requested guard-band width {guard_band_widths_px} raised -- nothing to report "
            f"stability about; see widths_that_raised: {widths_that_raised}"
        )

    verdict_stable_across_widths = len(set(per_width_verdict.values())) == 1

    return {
        "per_width_verdict": per_width_verdict,
        "per_width_report": per_width_report,
        "widths_that_raised": widths_that_raised,
        "verdict_stable_across_widths": verdict_stable_across_widths,
    }

@dataclasses.dataclass(frozen=True)
class BlendModeDivergence:

    offset: tuple[int, int]
    mad: float
    ssim_delta: float

def blend_mode_divergence(
    target_image: np.ndarray,
    lung_mask: np.ndarray,
    donor_patch: np.ndarray,
    donor_mask: np.ndarray,
    background_level: Optional[float] = None,
    seed: int = 0,
) -> BlendModeDivergence:
    if background_level is None:
        background_level = donor_background_level(donor_patch, donor_mask)

    poisson_result = insert_drain(
        target_image, lung_mask, donor_patch, donor_mask, np.random.default_rng(seed), blend_mode="poisson"
    )
    log_additive_result = insert_drain(
        target_image,
        lung_mask,
        donor_patch,
        donor_mask,
        np.random.default_rng(seed),
        blend_mode="log_additive",
        donor_background_level=background_level,
    )
    if poisson_result.offset != log_additive_result.offset:
        raise ValueError(
            f"placement offsets diverged between blend modes ({poisson_result.offset} vs "
            f"{log_additive_result.offset}) despite identical seeds -- blend_mode must not affect "
            f"placement search, so this comparison is only meaningful with a shared offset"
        )

    mad = mean_absolute_perturbation(
        [poisson_result.image], [log_additive_result.image], [poisson_result.placement_mask]
    )
    ssim_delta = mean_ssim_perturbation([poisson_result.image], [log_additive_result.image])
    return BlendModeDivergence(offset=poisson_result.offset, mad=mad, ssim_delta=ssim_delta)
