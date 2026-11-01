# The panel-level statistic of the realism discriminators.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices

SeedScores = tuple[np.ndarray, np.ndarray]
PerDetectorSeedScores = dict[str, Sequence[SeedScores]]

def _detector_mean_auc(seed_scores: Sequence[SeedScores], indices: np.ndarray) -> float:
    aucs = []
    for real_scores, edited_scores in seed_scores:
        real_scores = np.asarray(real_scores)
        edited_scores = np.asarray(edited_scores)
        labels = np.concatenate([np.zeros(len(indices)), np.ones(len(indices))])
        scores = np.concatenate([real_scores[indices], edited_scores[indices]])
        aucs.append(roc_auc_score(labels, scores))
    return float(np.mean(aucs))

def _panel_statistic(per_detector_seed_scores: PerDetectorSeedScores, indices: np.ndarray) -> float:
    return max(
        _detector_mean_auc(seed_scores, indices) for seed_scores in per_detector_seed_scores.values()
    )

def _per_detector_means(per_detector_seed_scores: PerDetectorSeedScores, indices: np.ndarray) -> dict[str, float]:
    return {
        name: _detector_mean_auc(seed_scores, indices) for name, seed_scores in per_detector_seed_scores.items()
    }

def panel_point_estimate(per_detector_seed_scores: PerDetectorSeedScores) -> tuple[float, dict[str, float]]:
    first_seed_scores = next(iter(per_detector_seed_scores.values()))[0]
    n = len(first_seed_scores[0])
    indices = np.arange(n)

    per_detector = _per_detector_means(per_detector_seed_scores, indices)
    panel_statistic = _panel_statistic(per_detector_seed_scores, indices)
    return panel_statistic, per_detector

def panel_gate_ci(
    cluster_ids: Sequence,
    per_detector_seed_scores: PerDetectorSeedScores,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:

    def statistic_fn(indices: np.ndarray) -> float:
        return _panel_statistic(per_detector_seed_scores, indices)

    return cluster_bootstrap_ci_indices(cluster_ids, statistic_fn, n_resamples, ci, seed)
