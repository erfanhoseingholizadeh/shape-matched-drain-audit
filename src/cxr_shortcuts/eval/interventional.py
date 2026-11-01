# Interventional metrics: the change in predicted probability when the edit is applied.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.metrics import structural_similarity

from cxr_shortcuts.eval.observational import compute_auc

def mean_probability_shift(probs_before: Sequence[float], probs_after: Sequence[float]) -> float:
    before = np.asarray(probs_before, dtype=float)
    after = np.asarray(probs_after, dtype=float)
    return float((after - before).mean())

def shortcut_induced_fpr(
    probs_before: Sequence[float],
    probs_after: Sequence[float],
    threshold: float,
) -> float:
    before = np.asarray(probs_before, dtype=float)
    after = np.asarray(probs_after, dtype=float)

    true_negative_before = before < threshold
    if not true_negative_before.any():
        raise ValueError("no images were below threshold before the edit")

    flipped = true_negative_before & (after >= threshold)
    return float(flipped.sum() / true_negative_before.sum())

def mean_absolute_perturbation(
    before_images: Sequence[np.ndarray],
    after_images: Sequence[np.ndarray],
    placement_masks: Sequence[np.ndarray],
) -> float:
    values = []
    for before, after, mask in zip(before_images, after_images, placement_masks):
        diff = np.abs(after.astype(np.float64) - before.astype(np.float64))
        values.append(float(diff[mask].mean()))
    return float(np.mean(values))

def mean_ssim_perturbation(
    before_images: Sequence[np.ndarray],
    after_images: Sequence[np.ndarray],
) -> float:
    values = []
    for before, after in zip(before_images, after_images):
        score = structural_similarity(before, after, data_range=255)
        values.append(1.0 - float(score))
    return float(np.mean(values))

def delta_auc_under_intervention(
    labels: Sequence[int],
    scores_before: Sequence[float],
    scores_after: Sequence[float],
) -> float:
    return compute_auc(labels, scores_after) - compute_auc(labels, scores_before)
