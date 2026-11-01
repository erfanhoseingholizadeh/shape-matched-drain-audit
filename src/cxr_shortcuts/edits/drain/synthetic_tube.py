# The parametric synthetic tube: a curved band with a marker stripe, generated at native resolution.
# Run: library module, imported by the scripts

from __future__ import annotations

import math

import cv2
import numpy as np

from cxr_shortcuts.edits.drain.annotate_tool import crop_to_mask_bbox

def generate_synthetic_tube_patch(
    length_px: float,
    caliber_px: float,
    rng: np.random.Generator,
    curvature: float = 0.25,
    fill_intensity: int = 190,
    stripe_intensity: int = 235,
    stripe_width_px: float = 2.0,
    background_level: float = 40.0,
    chord_orientation_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if caliber_px <= 0:
        raise ValueError(f"caliber_px must be positive, got {caliber_px!r}")

    caliber_int = max(1, int(round(caliber_px)))
    canvas_size = int(length_px + caliber_px * 4)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    center = np.array([canvas_size / 2.0, canvas_size / 2.0])
    orientation_rad = math.radians(chord_orientation_deg)
    direction = np.array([math.cos(orientation_rad), math.sin(orientation_rad)])
    perpendicular = np.array([direction[1], -direction[0]])
    p0 = center - (length_px / 2.0) * direction
    p2 = center + (length_px / 2.0) * direction
    sign = 1.0 if rng.random() < 0.5 else -1.0
    bend = sign * curvature * length_px * (0.9 + 0.2 * rng.random())
    p1 = center + bend * perpendicular

    n_samples = 200
    t = np.linspace(0.0, 1.0, n_samples)[:, None]
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    points_xy = curve[:, ::-1].astype(np.int32)

    for i in range(n_samples - 1):
        cv2.line(canvas, tuple(points_xy[i]), tuple(points_xy[i + 1]), color=1, thickness=caliber_int)

    mask_full = canvas.astype(bool)
    if not mask_full.any():
        raise ValueError("generated tube mask is empty -- length_px/caliber_px too small for this canvas")

    fill_full = np.where(mask_full, fill_intensity, background_level).astype(np.float64)

    stripe_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    stripe_width_int = max(1, int(round(stripe_width_px)))
    for i in range(n_samples - 1):
        cv2.line(stripe_canvas, tuple(points_xy[i]), tuple(points_xy[i + 1]), color=1, thickness=stripe_width_int)
    stripe_mask_full = stripe_canvas.astype(bool) & mask_full
    fill_full[stripe_mask_full] = stripe_intensity

    patch_full = np.clip(fill_full, 0, 255).astype(np.uint8)
    patch, mask, _offset = crop_to_mask_bbox(patch_full, mask_full)
    return patch, mask, background_level
