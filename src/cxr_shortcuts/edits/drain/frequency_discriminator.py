# A frequency-domain real-versus-edited detector, one member of the realism panel.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_RADIAL_BINS = 32
N_ANGULAR_SECTORS = 8

def _hann_window_2d(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w))

def _radial_and_angular_bin_indices(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dy = yy - cy
    dx = xx - cx
    radius = np.sqrt(dy**2 + dx**2)
    max_radius = min(cy, cx)

    radial_bin = np.clip((radius / max_radius * N_RADIAL_BINS).astype(int), 0, N_RADIAL_BINS - 1)

    angle = np.arctan2(dy, dx)
    angular_sector = np.clip(
        ((angle + np.pi) / (2 * np.pi) * N_ANGULAR_SECTORS).astype(int), 0, N_ANGULAR_SECTORS - 1
    )
    return radial_bin, angular_sector, max_radius

def radial_power_spectrum_features(image: np.ndarray, n_radial_bins: int = N_RADIAL_BINS) -> np.ndarray:
    windowed = image.astype(np.float64) * _hann_window_2d(image.shape)
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    log_magnitude = np.log1p(np.abs(spectrum))

    radial_bin, angular_sector, max_radius = _radial_and_angular_bin_indices(image.shape)

    radial_features = np.array(
        [log_magnitude[radial_bin == b].mean() if np.any(radial_bin == b) else 0.0 for b in range(n_radial_bins)]
    )

    outer = radial_bin >= (n_radial_bins // 2)
    angular_features = np.array(
        [
            log_magnitude[outer & (angular_sector == s)].mean() if np.any(outer & (angular_sector == s)) else 0.0
            for s in range(N_ANGULAR_SECTORS)
        ]
    )

    return np.concatenate([radial_features, angular_features])

def _features_for_images(images: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([radial_power_spectrum_features(image) for image in images])

def train_frequency_discriminator(
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    seed: int = 0,
) -> Pipeline:
    features = np.concatenate([_features_for_images(real_images), _features_for_images(edited_images)])
    labels = np.concatenate([np.zeros(len(real_images)), np.ones(len(edited_images))])

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classify", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(features, labels)
    return model

def predict_scores_frequency(model: Pipeline, images: Sequence[np.ndarray]) -> np.ndarray:
    features = _features_for_images(images)
    return model.predict_proba(features)[:, 1]

def frequency_discriminator_auc(
    model: Pipeline, real_images: Sequence[np.ndarray], edited_images: Sequence[np.ndarray]
) -> float:
    scores = np.concatenate([predict_scores_frequency(model, real_images), predict_scores_frequency(model, edited_images)])
    labels = np.concatenate([np.zeros(len(real_images)), np.ones(len(edited_images))])
    return float(roc_auc_score(labels, scores))
