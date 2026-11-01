# Expected calibration error and Brier score.
# Run: library module, imported by the scripts

from __future__ import annotations

import numpy as np

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    if n == 0:
        raise ValueError("expected_calibration_error requires at least one prediction")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_indices = np.clip(np.digitize(probs, bin_edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_indices == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_prob = float(probs[in_bin].mean())
        mean_label = float(labels[in_bin].mean())
        ece += (count / n) * abs(mean_prob - mean_label)
    return float(ece)
