# Percentile and patient-clustered bootstrap intervals.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_ci_indices(
    n: int,
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        estimates[i] = statistic_fn(indices)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(estimates, [alpha, 1 - alpha])
    return float(lo), float(hi)

def cluster_bootstrap_ci_indices(
    cluster_ids: Sequence,
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    cluster_ids = np.asarray(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled_clusters = unique_clusters[rng.integers(0, n_clusters, size=n_clusters)]
        indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        estimates[i] = statistic_fn(indices)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(estimates, [alpha, 1 - alpha])
    return float(lo), float(hi)

def gate_criterion_ci(
    donor_cluster_ids: Sequence,
    patient_cluster_ids: Sequence,
    statistic_fn: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    donor_ci = cluster_bootstrap_ci_indices(donor_cluster_ids, statistic_fn, n_resamples, ci, seed)
    patient_ci = cluster_bootstrap_ci_indices(patient_cluster_ids, statistic_fn, n_resamples, ci, seed)

    donor_width = donor_ci[1] - donor_ci[0]
    patient_width = patient_ci[1] - patient_ci[0]
    return donor_ci if donor_width >= patient_width else patient_ci

def cluster_bootstrap_auc_ci(
    cluster_ids: Sequence,
    real_scores: np.ndarray,
    edited_scores: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    real_scores = np.asarray(real_scores)
    edited_scores = np.asarray(edited_scores)

    def statistic_fn(indices: np.ndarray) -> float:
        labels = np.concatenate([np.zeros(len(indices)), np.ones(len(indices))])
        scores = np.concatenate([real_scores[indices], edited_scores[indices]])
        return float(roc_auc_score(labels, scores))

    return cluster_bootstrap_ci_indices(cluster_ids, statistic_fn, n_resamples, ci, seed)

def cluster_bootstrap_auc_difference_ci(
    cluster_ids: Sequence,
    real_scores_a: np.ndarray,
    edited_scores_a: np.ndarray,
    real_scores_b: np.ndarray,
    edited_scores_b: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    real_scores_a = np.asarray(real_scores_a)
    edited_scores_a = np.asarray(edited_scores_a)
    real_scores_b = np.asarray(real_scores_b)
    edited_scores_b = np.asarray(edited_scores_b)

    def statistic_fn(indices: np.ndarray) -> float:
        labels = np.concatenate([np.zeros(len(indices)), np.ones(len(indices))])
        scores_a = np.concatenate([real_scores_a[indices], edited_scores_a[indices]])
        scores_b = np.concatenate([real_scores_b[indices], edited_scores_b[indices]])
        auc_a = roc_auc_score(labels, scores_a)
        auc_b = roc_auc_score(labels, scores_b)
        return float(auc_b - auc_a)

    return cluster_bootstrap_ci_indices(cluster_ids, statistic_fn, n_resamples, ci, seed)
