# Observational metrics: AUC, drain-stratified AUC and the subgroup gap.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return float(roc_auc_score(labels, scores))

def stratified_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
) -> dict[str, float]:
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    groups_arr = np.asarray(groups)
    return {
        group: compute_auc(labels_arr[groups_arr == group], scores_arr[groups_arr == group])
        for group in sorted(set(groups_arr.tolist()))
    }

def subgroup_gap(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    group_a: str,
    group_b: str,
) -> float:
    per_group = stratified_auc(labels, scores, groups)
    return per_group[group_a] - per_group[group_b]
