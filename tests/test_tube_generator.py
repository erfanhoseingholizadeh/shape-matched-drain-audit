# Tests of the synthetic tube generator and the log-domain blend on a synthetic image.
# Run: python3 -m pytest tests/test_tube_generator.py
import numpy as np

from cxr_shortcuts.edits.drain.insertion import log_additive_blend_patch
from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch


def make(seed=0, **kw):
    return generate_synthetic_tube_patch(length_px=kw.pop("length_px", 160), caliber_px=kw.pop("caliber_px", 12), rng=np.random.default_rng(seed), **kw)


def test_patch_mask_and_background_have_matching_shapes_and_fill():
    patch, mask, bg = make(fill_intensity=190, stripe_intensity=235, background_level=40.0)
    assert patch.shape == mask.shape and patch.dtype == np.uint8 and mask.dtype == bool
    assert mask.any() and bg == 40.0
    assert (patch[~mask] == 40).all()
    assert set(np.unique(patch[mask])) <= {190, 235}


def test_same_seed_gives_the_same_tube_and_a_different_seed_a_different_one():
    a, b = make(seed=3), make(seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = make(seed=4)
    assert a[1].shape != c[1].shape or not np.array_equal(a[1], c[1])


def test_the_tube_is_about_as_thick_as_asked():
    _, mask, _ = make(caliber_px=14, length_px=200, curvature=0.0, chord_orientation_deg=90.0)
    rows = mask.sum(axis=0)
    assert 12 <= np.median(rows[rows > 0]) <= 16


def test_a_non_positive_caliber_is_refused():
    import pytest
    with pytest.raises(ValueError):
        make(caliber_px=0)


def test_blending_changes_only_the_masked_pixels_of_a_synthetic_film():
    film = np.full((300, 300), 120, dtype=np.uint8)
    patch, mask, bg = make(seed=1)
    out = log_additive_blend_patch(film, patch, mask, (40, 50), bg)
    h, w = mask.shape
    window = out[40:40 + h, 50:50 + w]
    assert (window[mask] > 120).all()
    assert (window[~mask] == 120).all()
    outside = out.copy()
    outside[40:40 + h, 50:50 + w] = 120
    assert (outside == 120).all()
