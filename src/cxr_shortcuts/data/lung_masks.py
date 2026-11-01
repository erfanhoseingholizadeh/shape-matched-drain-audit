# Loader for the CheXmask lung masks (run-length encoded) and their quality scores.
# Run: library module, imported by the scripts

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from cxr_shortcuts.data.reserved_guard import assert_not_reserved

DEFAULT_RCA_COLUMN = "Dice RCA (Mean)"
RCA_MAX_COLUMN = "Dice RCA (Max)"

def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    if not rle or not rle.strip() or rle.strip() == "-1":
        return np.zeros(shape, dtype=bool)

    tokens = [int(t) for t in rle.split()]
    starts, lengths = tokens[0::2], tokens[1::2]

    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for start, length in zip(starts, lengths):
        flat[start - 1 : start - 1 + length] = True
    return flat.reshape(shape)

def encode_rle(mask: np.ndarray) -> str:
    flat = mask.astype(int).reshape(-1)
    padded = np.concatenate([[0], flat, [0]])
    boundaries = np.where(padded[1:] != padded[:-1])[0] + 1
    boundaries[1::2] -= boundaries[::2]
    return " ".join(str(x) for x in boundaries)

def combine_lung_mask(left_lung: np.ndarray, right_lung: np.ndarray) -> np.ndarray:
    return left_lung | right_lung

def load_chexmask_lung_masks(
    df: pd.DataFrame,
    image_id_col: str = "Image Index",
    left_lung_col: str = "Left Lung",
    right_lung_col: str = "Right Lung",
    height_col: str = "Height",
    width_col: str = "Width",
) -> Mapping[str, np.ndarray]:
    assert_not_reserved(df[image_id_col], "load_chexmask_lung_masks")
    masks = {}
    for _, row in df.iterrows():
        shape = (int(row[height_col]), int(row[width_col]))
        left = decode_rle(row[left_lung_col], shape)
        right = decode_rle(row[right_lung_col], shape)
        masks[row[image_id_col]] = combine_lung_mask(left, right)
    return masks

def load_chexmask_lateral_masks(
    df: pd.DataFrame,
    image_id_col: str = "Image Index",
    left_lung_col: str = "Left Lung",
    right_lung_col: str = "Right Lung",
    height_col: str = "Height",
    width_col: str = "Width",
) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    assert_not_reserved(df[image_id_col], "load_chexmask_lateral_masks")
    masks = {}
    for _, row in df.iterrows():
        shape = (int(row[height_col]), int(row[width_col]))
        left = decode_rle(row[left_lung_col], shape)
        right = decode_rle(row[right_lung_col], shape)
        masks[row[image_id_col]] = (left, right)
    return masks

def filter_by_rca_quality(
    df: pd.DataFrame,
    min_rca: float,
    column: str = DEFAULT_RCA_COLUMN,
) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"RCA column {column!r} not found in dataframe columns: {list(df.columns)}")
    return df[df[column] >= min_rca].copy()

def rca_filter_report(
    candidate_ids: Sequence[str],
    rca_df: pd.DataFrame,
    min_rca: float,
    image_id_col: str = "Image Index",
    column: str = DEFAULT_RCA_COLUMN,
) -> dict:
    if column not in rca_df.columns:
        raise ValueError(f"RCA column {column!r} not found in dataframe columns: {list(rca_df.columns)}")

    candidate_ids = list(candidate_ids)
    scored = rca_df.set_index(image_id_col)[column]
    scored = scored[~scored.index.duplicated(keep="first")]

    present_ids = [cid for cid in candidate_ids if cid in scored.index]
    no_row_ids = [cid for cid in candidate_ids if cid not in scored.index]
    eligible_ids = [cid for cid in present_ids if scored.loc[cid] >= min_rca]
    below_threshold_ids = [cid for cid in present_ids if scored.loc[cid] < min_rca]

    return {
        "total_candidates": len(candidate_ids),
        "no_chexmask_row": len(no_row_ids),
        "below_threshold": len(below_threshold_ids),
        "eligible": len(eligible_ids),
        "eligible_ids": eligible_ids,
        "threshold": min_rca,
        "column": column,
    }

def rca_distribution_report(
    candidate_ids: Sequence[str],
    rca_df: pd.DataFrame,
    thresholds: Sequence[float],
    image_id_col: str = "Image Index",
    column: str = DEFAULT_RCA_COLUMN,
) -> dict:
    matched = rca_df[rca_df[image_id_col].isin(set(candidate_ids))]
    if matched.empty:
        raise ValueError("no candidate_ids have a matching CheXmask row")

    scores = matched[column]
    return {
        "n_candidates": len(candidate_ids),
        "n_matched": len(matched),
        "median": float(scores.median()),
        "q25": float(scores.quantile(0.25)),
        "q75": float(scores.quantile(0.75)),
        "survival": {
            thr: {"count": int((scores >= thr).sum()), "rate": float((scores >= thr).mean())}
            for thr in thresholds
        },
    }

def sample_rca_bands(
    rca_df: pd.DataFrame,
    candidate_ids: Sequence[str],
    band_edges: Sequence[float],
    n_per_band: int,
    rng: np.random.Generator,
    image_id_col: str = "Image Index",
    column: str = DEFAULT_RCA_COLUMN,
) -> pd.DataFrame:
    matched = rca_df[rca_df[image_id_col].isin(set(candidate_ids))]

    rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = matched[(matched[column] >= lo) & (matched[column] < hi)]
        n = min(n_per_band, len(band))
        if n == 0:
            continue
        sampled = band.sample(n=n, random_state=rng.integers(0, 2**32 - 1))
        for _, row in sampled.iterrows():
            rows.append({"image_id": row[image_id_col], "rca_score": row[column], "band": f"[{lo:.2f},{hi:.2f})"})

    return pd.DataFrame(rows, columns=["image_id", "rca_score", "band"])

def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    if image.shape != mask.shape:
        raise ValueError(f"image and mask must share shape, got {image.shape} vs {mask.shape}")

    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
    overlay_color = np.array(color, dtype=np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * overlay_color
    return rgb.clip(0, 255).astype(np.uint8)

def export_mask_overlay_samples(
    samples: pd.DataFrame,
    images: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in samples.iterrows():
        image_id = row["image_id"]
        overlay = overlay_mask_on_image(images[image_id], masks[image_id])
        overlay_path = output_dir / f"{image_id}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        rows.append({"image_id": image_id, "band": row["band"], "overlay_path": str(overlay_path)})

    manifest = pd.DataFrame(rows, columns=["image_id", "band", "overlay_path"])
    manifest.to_csv(output_dir / "rca_band_review_manifest.csv", index=False)
    return manifest
