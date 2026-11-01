# The patient-clustered percentile bootstrap shared by artifact_summary.py and probability_scale.py.
# Run: library module, imported by the scripts

from __future__ import annotations

import numpy as np

B = 10_000

def patient_index(patient_ids) -> tuple[np.ndarray, np.ndarray]:
    patients, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
    return patients, inverse

def draw(n_patients: int, seed: int, b: int = B) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_patients, size=(b, n_patients))
    w = np.zeros((b, n_patients), dtype=np.int64)
    for r in range(b):
        w[r] = np.bincount(idx[r], minlength=n_patients)
    return w

def per_patient_sums(values: np.ndarray, inverse: np.ndarray, n_patients: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    s = np.zeros(n_patients)
    n = np.zeros(n_patients)
    np.add.at(s, inverse[ok], v[ok])
    np.add.at(n, inverse[ok], 1.0)
    return s, n

def boot_mean(values: np.ndarray, inverse: np.ndarray, w: np.ndarray) -> np.ndarray:
    s, n = per_patient_sums(values, inverse, w.shape[1])
    den = w @ n
    return (w @ s) / np.where(den > 0, den, np.nan)

def interval(samples: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]

def mean_with_interval(values: np.ndarray, inverse: np.ndarray, w: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    point = float(np.nanmean(v))
    return {"mean": point, "ci95": interval(boot_mean(v, inverse, w)), "n_units": int(np.isfinite(v).sum())}

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")

def boot_spearman(x: np.ndarray, y: np.ndarray, inverse: np.ndarray, seed: int, b: int = B) -> np.ndarray:
    from scipy.stats import rankdata
    n_p = int(inverse.max()) + 1
    members = [np.flatnonzero(inverse == p) for p in range(n_p)]
    rng = np.random.default_rng(seed)
    out = np.empty(b)
    for r in range(b):
        drawn = rng.integers(0, n_p, size=n_p)
        sel = np.concatenate([members[p] for p in drawn])
        rx, ry = rankdata(x[sel]), rankdata(y[sel])
        rx, ry = rx - rx.mean(), ry - ry.mean()
        d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        out[r] = (rx * ry).sum() / d if d > 0 else np.nan
    return out

def boot_spearman_pairs(m: np.ndarray, pairs: list[tuple[int, int]], inverse: np.ndarray, seed: int, b: int = B) -> np.ndarray:
    from scipy.stats import rankdata
    n_p = int(inverse.max()) + 1
    members = [np.flatnonzero(inverse == p) for p in range(n_p)]
    rng = np.random.default_rng(seed)
    out = np.empty((b, len(pairs)))
    ia = np.array([i for i, _ in pairs])
    ib = np.array([j for _, j in pairs])
    for r in range(b):
        drawn = rng.integers(0, n_p, size=n_p)
        sel = np.concatenate([members[p] for p in drawn])
        rk = rankdata(m[sel], axis=0)
        rk = rk - rk.mean(axis=0)
        num = (rk[:, ia] * rk[:, ib]).sum(axis=0)
        den = np.sqrt((rk[:, ia] ** 2).sum(axis=0) * (rk[:, ib] ** 2).sum(axis=0))
        out[r] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out
