# Shared geometry primitives for the edit pipeline.
# Run: library module, imported by the scripts

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

class NoValidPlacementError(RuntimeError):
    pass

class NoMatchedPlacementError(NoValidPlacementError):
    pass

def place_mask(local_mask: np.ndarray, image_shape: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    row_off, col_off = offset
    h, w = local_mask.shape
    if (
        row_off < 0
        or col_off < 0
        or row_off + h > image_shape[0]
        or col_off + w > image_shape[1]
    ):
        raise ValueError(
            f"placement out of bounds: offset={offset}, local shape={local_mask.shape}, "
            f"image shape={image_shape}"
        )
    canvas = np.zeros(image_shape, dtype=bool)
    canvas[row_off : row_off + h, col_off : col_off + w] = local_mask
    return canvas

def fits_inside_lung(placed_mask: np.ndarray, lung_mask: np.ndarray) -> bool:
    return bool(np.array_equal(placed_mask & lung_mask, placed_mask))

def overlaps(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool((mask_a & mask_b).any())

def sample_valid_placement(
    local_mask: np.ndarray,
    lung_mask: np.ndarray,
    rng: np.random.Generator,
    pathology_mask: np.ndarray | None = None,
    max_tries: int = 200,
) -> tuple[int, int]:
    h, w = local_mask.shape
    img_h, img_w = lung_mask.shape
    if h > img_h or w > img_w:
        raise NoValidPlacementError(
            f"local mask {local_mask.shape} cannot fit inside image {lung_mask.shape}"
        )

    max_row_off = img_h - h
    max_col_off = img_w - w
    for _ in range(max_tries):
        row_off = int(rng.integers(0, max_row_off + 1))
        col_off = int(rng.integers(0, max_col_off + 1))
        placed = place_mask(local_mask, lung_mask.shape, (row_off, col_off))
        if not fits_inside_lung(placed, lung_mask):
            continue
        if pathology_mask is not None and overlaps(placed, pathology_mask):
            continue
        return row_off, col_off

    raise NoValidPlacementError(
        f"no valid placement found for a {local_mask.shape} mask within the lung "
        f"field after {max_tries} tries"
    )

def valid_placement_mask(
    local_mask: np.ndarray,
    region_mask: np.ndarray,
    pathology_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = local_mask.shape
    img_h, img_w = region_mask.shape
    if h > img_h or w > img_w:
        return np.zeros((max(img_h - h + 1, 0), max(img_w - w + 1, 0)), dtype=bool)

    kernel = local_mask.astype(np.float64)
    target = float(local_mask.sum())
    overlap = fftconvolve(region_mask.astype(np.float64), kernel[::-1, ::-1], mode="valid")
    fits = np.isclose(overlap, target, atol=0.5)

    if pathology_mask is not None:
        pathology_overlap = fftconvolve(pathology_mask.astype(np.float64), kernel[::-1, ::-1], mode="valid")
        fits &= np.isclose(pathology_overlap, 0.0, atol=0.5)

    return fits

def placement_lung_side(
    local_mask: np.ndarray,
    offset: tuple[int, int],
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
) -> Optional[str]:
    placed = place_mask(local_mask, left_lung_mask.shape, offset)
    if fits_inside_lung(placed, left_lung_mask):
        return "left"
    if fits_inside_lung(placed, right_lung_mask):
        return "right"
    return None

def medial_lateral_axis(lung_mask: np.ndarray, image_width: int) -> tuple[int, int]:
    cols = np.nonzero(lung_mask.any(axis=0))[0]
    min_col, max_col = int(cols.min()), int(cols.max())
    bbox_center = (min_col + max_col) / 2
    if bbox_center < image_width / 2:
        return min_col, max_col
    return max_col, min_col

def local_mean_intensity(image: np.ndarray, local_mask: np.ndarray, offset: tuple[int, int]) -> float:
    row_off, col_off = offset
    h, w = local_mask.shape
    region = image[row_off : row_off + h, col_off : col_off + w]
    return float(region[local_mask].mean())

def assert_matched_footprint_area(reference_mask: np.ndarray, candidate_mask: np.ndarray, tolerance: float = 0.05) -> None:
    ref_area = int(reference_mask.sum())
    cand_area = int(candidate_mask.sum())
    if ref_area == 0:
        if cand_area != 0:
            raise ValueError("reference mask has zero area but candidate mask does not")
        return
    relative_diff = abs(cand_area - ref_area) / ref_area
    if relative_diff > tolerance:
        raise ValueError(
            f"control mask area ({cand_area}px) diverges from donor mask area "
            f"({ref_area}px) by {relative_diff:.1%}, exceeding tolerance {tolerance:.1%}"
        )

@dataclasses.dataclass(frozen=True)
class PlacementStats:
    side: str
    medial_lateral_position: float
    local_intensity: float

@dataclasses.dataclass(frozen=True)
class MatchedPlacement:
    offset: tuple[int, int]
    relaxed: tuple[str, ...]
    n_candidates: int

def compute_placement_stats(
    image: np.ndarray,
    local_mask: np.ndarray,
    offset: tuple[int, int],
    side: str,
    side_lung_mask: np.ndarray,
) -> PlacementStats:
    _, cols = np.nonzero(local_mask)
    centroid_col = offset[1] + cols.mean()
    lateral_col, medial_col = medial_lateral_axis(side_lung_mask, image.shape[1])
    span = medial_col - lateral_col
    position = 0.5 if span == 0 else (centroid_col - lateral_col) / span
    intensity = local_mean_intensity(image, local_mask, offset)
    return PlacementStats(side=side, medial_lateral_position=float(position), local_intensity=float(intensity))

def sample_matched_placement(
    local_mask: np.ndarray,
    reference_stats: PlacementStats,
    left_lung_mask: np.ndarray,
    right_lung_mask: np.ndarray,
    target_image: np.ndarray,
    rng: np.random.Generator,
    pathology_mask: Optional[np.ndarray] = None,
    position_tol: float = 0.2,
    intensity_tol: float = 20.0,
) -> MatchedPlacement:
    if target_image.shape != left_lung_mask.shape or target_image.shape != right_lung_mask.shape:
        raise ValueError("target_image, left_lung_mask, and right_lung_mask must share the same shape")

    side_lung_mask = left_lung_mask if reference_stats.side == "left" else right_lung_mask
    valid = valid_placement_mask(local_mask, side_lung_mask, pathology_mask)
    if not valid.any():
        raise NoMatchedPlacementError(
            f"no offset fits inside the {reference_stats.side} lung field at all, "
            "before any placement-matching constraint"
        )

    cols_local = np.nonzero(local_mask)[1]
    centroid_col_local = cols_local.mean()

    row_idx, col_idx = np.indices(valid.shape)
    centroid_cols = col_idx + centroid_col_local

    lateral_col, medial_col = medial_lateral_axis(side_lung_mask, target_image.shape[1])
    span = medial_col - lateral_col
    positions = np.full(valid.shape, 0.5) if span == 0 else (centroid_cols - lateral_col) / span

    kernel = local_mask.astype(np.float64)
    target_sum = float(local_mask.sum())
    intensity_sums = fftconvolve(target_image.astype(np.float64), kernel[::-1, ::-1], mode="valid")
    intensities = intensity_sums / target_sum

    position_close = np.abs(positions - reference_stats.medial_lateral_position) <= position_tol
    intensity_close = np.abs(intensities - reference_stats.local_intensity) <= intensity_tol

    for relax in ((), ("intensity",), ("intensity", "position")):
        mask = valid.copy()
        if "position" not in relax:
            mask &= position_close
        if "intensity" not in relax:
            mask &= intensity_close
        candidates = np.argwhere(mask)
        if len(candidates):
            chosen = candidates[int(rng.integers(len(candidates)))]
            return MatchedPlacement(
                offset=(int(chosen[0]), int(chosen[1])),
                relaxed=relax,
                n_candidates=int(len(candidates)),
            )

    raise NoMatchedPlacementError(
        f"no offset available inside the {reference_stats.side} lung field even after "
        "relaxing intensity and medial-lateral position matching -- laterality itself "
        "is never relaxed"
    )
