# Tests of the patient-clustered bootstrap on a toy array.
# Run: python3 -m pytest tests/test_bootstrap.py
import numpy as np

from analysis import patient_bootstrap as bt


def toy():
    patients = np.repeat(np.arange(20), 3)
    values = np.tile([1.0, 2.0, 3.0], 20) + np.repeat(np.linspace(-1, 1, 20), 3)
    return patients, values


def test_draw_rows_sum_to_the_number_of_patients_and_are_reproducible():
    w = bt.draw(20, seed=7, b=200)
    assert w.shape == (200, 20) and (w.sum(axis=1) == 20).all()
    assert np.array_equal(w, bt.draw(20, seed=7, b=200))
    assert not np.array_equal(w, bt.draw(20, seed=8, b=200))


def test_the_interval_contains_the_point_estimate_and_the_point_is_the_plain_mean():
    patients, values = toy()
    _, inverse = bt.patient_index(patients)
    w = bt.draw(20, seed=1, b=500)
    r = bt.mean_with_interval(values, inverse, w)
    assert abs(r["mean"] - values.mean()) < 1e-12
    assert r["ci95"][0] < r["mean"] < r["ci95"][1]
    assert r["n_units"] == 60


def test_a_constant_array_has_a_zero_width_interval():
    patients = np.repeat(np.arange(10), 2)
    _, inverse = bt.patient_index(patients)
    r = bt.mean_with_interval(np.full(20, 0.5), inverse, bt.draw(10, seed=2, b=100))
    assert r["ci95"] == [0.5, 0.5]
