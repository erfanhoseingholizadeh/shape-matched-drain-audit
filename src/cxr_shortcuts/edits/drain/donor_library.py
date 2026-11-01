# Donor records (crop, mask, source id, split) and the checks on them.
# Run: library module, imported by the scripts

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

REQUIRED_DONOR_METADATA_COLUMNS = (
    "laterality",
    "insertion_angle_deg",
)

REQUIRED_DONOR_MASK_METRIC_COLUMNS = (
    "donor_mask_min_width_px",
    "donor_mask_max_inscribed_diameter_px",
)

REQUIRED_DONOR_COLUMNS = (
    "donor_id",
    "source_image_id",
    "crop_path",
    "mask_path",
    "source_image_width",
    "source_image_height",
    "annotation_started_at",
    "annotation_saved_at",
) + REQUIRED_DONOR_METADATA_COLUMNS + REQUIRED_DONOR_MASK_METRIC_COLUMNS

def donor_mask_width_metrics(mask: np.ndarray) -> dict:
    if not mask.any():
        raise ValueError("mask has no True pixels -- nothing to measure")
    if mask.all():
        raise ValueError(
            "mask has no False pixels -- distance_transform_edt has no real boundary to measure "
            "distance to (the donor's tight bounding-box crop happens to be entirely mask)"
        )
    distance = distance_transform_edt(mask)
    max_inscribed_diameter_px = float(2.0 * distance.max())
    skeleton = skeletonize(mask)
    if not skeleton.any():
        skeleton = mask
    min_width_px = float(2.0 * distance[skeleton].min())
    return {
        "donor_mask_min_width_px": min_width_px,
        "donor_mask_max_inscribed_diameter_px": max_inscribed_diameter_px,
    }

def mask_principal_axis_angle_deg(mask: np.ndarray) -> float:
    if int(mask.sum()) < 2:
        raise ValueError("mask has fewer than 2 True pixels -- no axis to fit")
    rows, cols = np.nonzero(mask)
    coords = np.stack([rows, cols], axis=1).astype(float)
    coords -= coords.mean(axis=0)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    d_row, d_col = eigenvectors[:, int(np.argmax(eigenvalues))]
    return float(math.degrees(math.atan2(abs(d_col), abs(d_row))))

def donor_laterality_from_mask(mask: np.ndarray, crop_offset_col: int, source_image_width: int) -> str:
    local_cols = np.nonzero(mask)[1]
    absolute_centroid_col = crop_offset_col + float(local_cols.mean())
    midline = source_image_width / 2
    return "right" if absolute_centroid_col < midline else "left"

def validate_donor_columns(records: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_DONOR_COLUMNS if c not in records.columns]
    if missing:
        raise ValueError(f"donor records missing required columns: {missing}")

_BLANK_STRING_FIELDS = ("donor_id", "source_image_id", "crop_path", "mask_path")
_POSITIVE_NUMERIC_FIELDS = (
    "source_image_width",
    "source_image_height",
) + REQUIRED_DONOR_MASK_METRIC_COLUMNS
_REAL_NUMERIC_FIELDS = ("insertion_angle_deg",) + _POSITIVE_NUMERIC_FIELDS
_TIMESTAMP_FIELDS = ("annotation_started_at", "annotation_saved_at")

def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""

def validate_donor_record(record: Mapping) -> None:
    missing_keys = [c for c in REQUIRED_DONOR_COLUMNS if c not in record]
    if missing_keys:
        raise ValueError(f"donor record missing required fields: {missing_keys}")

    for field in _BLANK_STRING_FIELDS:
        if _is_missing(record[field]):
            raise ValueError(f"donor record field {field!r} is missing (blank/None/NaN)")

    for field in _TIMESTAMP_FIELDS:
        value = record[field]
        if _is_missing(value):
            raise ValueError(f"donor record field {field!r} is missing (blank/None/NaN)")
        try:
            datetime.fromisoformat(str(value))
        except ValueError:
            raise ValueError(f"donor record field {field!r} must be an ISO 8601 timestamp, got {value!r}")

    laterality = record["laterality"]
    if laterality not in ("left", "right"):
        raise ValueError(f"donor record field 'laterality' must be 'left' or 'right', got {laterality!r}")

    for field in _REAL_NUMERIC_FIELDS:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"donor record field {field!r} must be a real number, got {value!r}")

    for field in _POSITIVE_NUMERIC_FIELDS:
        value = record[field]
        if value <= 0:
            raise ValueError(f"donor record field {field!r} must be positive, got {value!r}")

def assign_donor_splits(
    donor_ids: Sequence[str],
    *,
    n_train: int = 200,
    n_eval: int,
    seed: int = 0,
) -> dict[str, str]:
    unique_ids = sorted(set(donor_ids))
    if len(unique_ids) < n_train + n_eval:
        raise ValueError(
            f"need at least {n_train + n_eval} donors (n_train={n_train} + "
            f"n_eval={n_eval}), got {len(unique_ids)}"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    assignment: dict[str, str] = {}
    for did in shuffled[:n_train]:
        assignment[did] = "train"
    for did in shuffled[n_train : n_train + n_eval]:
        assignment[did] = "eval"
    for did in shuffled[n_train + n_eval :]:
        assignment[did] = "unused"
    return assignment

def build_donor_manifest(
    records: pd.DataFrame,
    *,
    n_train: int = 200,
    n_eval: int,
    seed: int = 0,
) -> pd.DataFrame:
    validate_donor_columns(records)
    assignment = assign_donor_splits(records["donor_id"], n_train=n_train, n_eval=n_eval, seed=seed)
    manifest = records.copy()
    manifest["split"] = manifest["donor_id"].map(assignment)
    return manifest

def donor_laterality_distribution(records: pd.DataFrame) -> dict:
    if len(records) == 0:
        raise ValueError("donor records table is empty -- no laterality distribution to compute")
    n_left = int((records["laterality"] == "left").sum())
    n_right = int((records["laterality"] == "right").sum())
    total = n_left + n_right
    return {
        "n_left": n_left,
        "n_right": n_right,
        "fraction_left": n_left / total,
        "fraction_right": n_right / total,
    }

def donor_annotation_time_report(records: pd.DataFrame) -> dict:
    if len(records) == 0:
        raise ValueError("donor records table is empty -- no annotation-time report to compute")
    started = pd.to_datetime(records["annotation_started_at"])
    saved = pd.to_datetime(records["annotation_saved_at"])
    seconds = (saved - started).dt.total_seconds()
    return {
        "n": int(len(seconds)),
        "median_seconds": float(seconds.median()),
        "p25_seconds": float(seconds.quantile(0.25)),
        "p75_seconds": float(seconds.quantile(0.75)),
        "min_seconds": float(seconds.min()),
        "max_seconds": float(seconds.max()),
    }

def relativize_donor_path(path, root) -> str:
    root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        raise ValueError(
            f"donor path {resolved} is not inside the donor-library root {root} -- "
            f"crop_path/mask_path must live under the same directory as donor_records.csv "
            f"so they can be stored as portable relative paths"
        )

def resolve_donor_paths(records: pd.DataFrame, root) -> pd.DataFrame:
    root = Path(root)
    resolved = records.copy()
    resolved["crop_path"] = resolved["crop_path"].apply(lambda p: str(root / p))
    resolved["mask_path"] = resolved["mask_path"].apply(lambda p: str(root / p))
    return resolved

_NEATX_LOCATION_TO_LATERALITY = {"L": "left", "R": "right"}

def _annotator_agreement(merged: pd.DataFrame, location_col: str) -> dict:
    is_both = merged[location_col] == "BOTH"
    comparable = merged[merged[location_col].isin(_NEATX_LOCATION_TO_LATERALITY)]
    normalized = comparable[location_col].map(_NEATX_LOCATION_TO_LATERALITY)
    n_comparable = len(comparable)
    n_agree = int((normalized == comparable["laterality"]).sum())
    return {
        "n_comparable": n_comparable,
        "n_agree": n_agree,
        "rate": (n_agree / n_comparable) if n_comparable else None,
        "n_excluded_both": int(is_both.sum()),
    }

def donor_laterality_neatx_agreement(
    donor_records: pd.DataFrame,
    neatx_locations: pd.DataFrame,
    source_image_id_col: str = "source_image_id",
    neatx_image_id_col: str = "image_id",
) -> dict:
    merged = donor_records.merge(
        neatx_locations, left_on=source_image_id_col, right_on=neatx_image_id_col, how="left"
    )

    agreement_a1 = _annotator_agreement(merged, "Drain_Location_a1")
    agreement_a2 = _annotator_agreement(merged, "Drain_Location_a2")

    annotators_agree = merged["Drain_Location_a1"] == merged["Drain_Location_a2"]
    confident = _annotator_agreement(merged[annotators_agree], "Drain_Location_a1")

    return {
        "n_donors": len(donor_records),
        "n_matched": int(merged[neatx_image_id_col].notna().sum()),
        "agreement_vs_a1": agreement_a1,
        "agreement_vs_a2": agreement_a2,
        "agreement_vs_confident_consensus": {
            "n_comparable": confident["n_comparable"],
            "n_agree": confident["n_agree"],
            "rate": confident["rate"],
        },
    }

def assert_donor_matches_target_resolution(donor_record: Mapping, target_shape: tuple[int, int]) -> None:
    recorded = (int(donor_record["source_image_height"]), int(donor_record["source_image_width"]))
    if recorded != tuple(target_shape):
        raise ValueError(
            f"donor resolution mismatch: donor {donor_record['donor_id']!r} was recorded at "
            f"{recorded} (height, width), but the target image is {tuple(target_shape)} -- "
            f"a donor patch's pixel dimensions are only meaningful against a target image at "
            f"the same native resolution it was cropped from"
        )

def assert_no_donor_leakage(manifest: pd.DataFrame) -> None:
    train_ids = set(manifest.loc[manifest["split"] == "train", "donor_id"])
    eval_ids = set(manifest.loc[manifest["split"] == "eval", "donor_id"])
    overlap = train_ids & eval_ids
    if overlap:
        raise AssertionError(f"donor leakage between train/eval splits: {sorted(overlap)}")
