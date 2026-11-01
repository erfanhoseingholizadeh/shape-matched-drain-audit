# Computes per-target pixel statistics of the four edit arms (spread, seam, frequency content) on the exploratory development-half targets.
# Run: python3 scoring/artifact_stats.py stats --composite-dir DIR --masks-csv CSV --out-dir DIR [--commit SHA] [--smoke-n N] [--only-present]

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

NATIVE = 1024
MODEL = 224
BAND_R_NATIVE = 2
BAND_R_MODEL = 1
RING_INNER = 6
RING_OUTER = 30
RING_MIN_PIXELS = 200
FFT_N = 512
HF_LOW_BIN = 28
HF_HIGH_BIN = 56
CROP_PAD = 4
EXPECTED_DEV_HALF_TARGETS = 77
EXPECTED_MANIFEST_DIGEST = "cccb2fc8d3058f5ae1e5124dbd61294e2142454b343092fe433d275170727e64"
COLUMNS = ["image_id", "patient_id", "footprint", "arm", "donor", "n_footprint", "n_band_native", "n_band_model", "n_ring", "ring_ok",
           "seam_native", "seam_model", "b1_mean_diff", "b2_sd_diff", "b3_w1", "hf_share"]

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def disk_element(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r

def band(footprint: np.ndarray, r: int) -> np.ndarray:
    from scipy import ndimage as ndi
    se = disk_element(r)
    return ndi.binary_dilation(footprint, structure=se) & ~ndi.binary_erosion(footprint, structure=se)

def sobel_magnitude(a: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    a = a.astype(np.float64)
    return np.hypot(ndi.sobel(a, axis=1, mode="nearest"), ndi.sobel(a, axis=0, mode="nearest")) / 8.0

def seam_statistic(arm: np.ndarray, band_mask: np.ndarray) -> float:
    ys, xs = np.where(band_mask)
    y0, y1 = max(int(ys.min()) - CROP_PAD, 0), min(int(ys.max()) + 1 + CROP_PAD, arm.shape[0])
    x0, x1 = max(int(xs.min()) - CROP_PAD, 0), min(int(xs.max()) + 1 + CROP_PAD, arm.shape[1])
    g = sobel_magnitude(arm[y0:y1, x0:x1])
    return float(g[band_mask[y0:y1, x0:x1]].mean())

def to_model_size(arm: np.ndarray) -> np.ndarray:
    from PIL import Image
    return np.array(Image.fromarray(arm).resize((MODEL, MODEL), Image.BILINEAR))

def mask_to_model_size(mask: np.ndarray) -> np.ndarray:
    from PIL import Image
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((MODEL, MODEL), Image.BILINEAR)) >= 128

def ring_mask(footprint: np.ndarray, lung: np.ndarray, both_footprints: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(~footprint)
    return lung & (d > RING_INNER) & (d <= RING_OUTER) & ~both_footprints

def _radial_bins(n: int = FFT_N) -> np.ndarray:
    idx = np.fft.fftfreq(n, d=1.0 / n)
    return np.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2)

_K = None

def hf_share(arm: np.ndarray, footprint: np.ndarray) -> float:
    global _K
    if _K is None:
        _K = _radial_bins()
    ys, xs = np.where(footprint)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    assert h <= FFT_N and w <= FFT_N, f"the footprint's bounding box {h} by {w} does not fit the {FFT_N}-point array"
    sub = arm[y0:y1, x0:x1].astype(np.float64)
    f = footprint[y0:y1, x0:x1]
    r = np.zeros((FFT_N, FFT_N))
    r[:h, :w] = np.where(f, sub - sub[f].mean(), 0.0)
    p = np.abs(np.fft.fft2(r)) ** 2
    total = p[(_K > 0) & (_K < HF_HIGH_BIN)].sum()
    if total <= 0.0:
        return float("nan")
    return float(p[(_K >= HF_LOW_BIN) & (_K < HF_HIGH_BIN)].sum() / total)

def intensity_statistics(arm: np.ndarray, before: np.ndarray, footprint: np.ndarray, ring: np.ndarray) -> tuple[float, float, float]:
    from scipy.stats import wasserstein_distance
    x = arm[footprint].astype(np.float64)
    y = before[ring].astype(np.float64)
    return float(x.mean() - y.mean()), float(x.std() - y.std()), float(wasserstein_distance(x, y))

def footprint_context(footprint: np.ndarray, lung: np.ndarray, both: np.ndarray) -> dict:
    fm = mask_to_model_size(footprint)
    ring = ring_mask(footprint, lung, both)
    return {"F": footprint, "band_native": band(footprint, BAND_R_NATIVE), "F_model": fm, "band_model": band(fm, BAND_R_MODEL), "ring": ring,
            "ring_ok": bool(ring.sum() >= RING_MIN_PIXELS)}

def arm_row(arm: np.ndarray, before: np.ndarray, ctx: dict) -> dict:
    nan = float("nan")
    seam_n = seam_statistic(arm, ctx["band_native"])
    seam_m = seam_statistic(to_model_size(arm), ctx["band_model"])
    b1 = b2 = b3 = nan
    if ctx["ring_ok"]:
        b1, b2, b3 = intensity_statistics(arm, before, ctx["F"], ctx["ring"])
    return {"n_footprint": int(ctx["F"].sum()), "n_band_native": int(ctx["band_native"].sum()), "n_band_model": int(ctx["band_model"].sum()), "n_ring": int(ctx["ring"].sum()),
            "ring_ok": int(ctx["ring_ok"]), "seam_native": seam_n, "seam_model": seam_m, "b1_mean_diff": b1, "b2_sd_diff": b2, "b3_w1": b3, "hf_share": hf_share(arm, ctx["F"])}

def target_rows(image_id: str, patient_id: str, z, lung: np.ndarray) -> list[dict]:
    before = z["before"]
    masks = {"tube": z["tube_mask"].astype(bool), "disk": z["disk_mask"].astype(bool)}
    both = masks["tube"] | masks["disk"]
    arms = {"tube": (z["tube"], z["tube_tissue"]), "disk": (z["disk"], z["disk_tissue"])}
    rows = []
    for fp, (edit, tissue) in arms.items():
        ctx = footprint_context(masks[fp], lung, both)
        base = {"image_id": image_id, "patient_id": patient_id, "footprint": fp}
        rows.append({**base, "arm": "before", "donor": -1, **arm_row(before, before, ctx)})
        rows.append({**base, "arm": "edit", "donor": -1, **arm_row(edit, before, ctx)})
        for d in range(tissue.shape[0]):
            rows.append({**base, "arm": "tissue", "donor": d, **arm_row(tissue[d], before, ctx)})
    return rows

def run_stats(args) -> None:
    import pandas as pd
    from cxr_shortcuts.data.lung_masks import load_chexmask_lung_masks
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved, reserved_ids_path
    from cxr_shortcuts.eval.population_log import ids_digest, log_population

    comp = Path(args.composite_dir)
    manifest = json.loads((comp / "composite_manifest.json").read_text())
    assert manifest["manifest_sha256"] == EXPECTED_MANIFEST_DIGEST, "the composite manifest is not the positive-audit stage one"
    dev = sorted(i for i, m in manifest["meta"].items() if m["stratum"] == "dev_half")
    assert len(dev) == EXPECTED_DEV_HALF_TARGETS, f"{len(dev)} development-half targets, expected {EXPECTED_DEV_HALF_TARGETS}"
    ids_full_digest = ids_digest(dev)
    assert_not_reserved(dev, "edit_artifacts targets")
    ids = dev
    if args.only_present:
        ids = [i for i in ids if (comp / manifest["meta"][i]["file"]).exists()]
    if args.smoke_n:
        ids = ids[: args.smoke_n]
    assert ids, "no target to process"
    log(f"{len(ids)} of {len(dev)} development-half targets; reserved-id file {reserved_ids_path()}")
    masks_csv = Path(args.masks_csv)
    if args.expect_masks_sha:
        assert sha256_file(masks_csv) == args.expect_masks_sha, "the CheXmask rows file is not the one the positive-audit stage composite job read"
    rca = pd.read_csv(masks_csv)
    sub = rca[rca["Image Index"].isin(ids)]
    assert sub["Image Index"].nunique() == len(ids), f"CheXmask rows found for {sub['Image Index'].nunique()} of {len(ids)} targets"
    lungs = load_chexmask_lung_masks(sub)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for n, iid in enumerate(ids):
        meta = manifest["meta"][iid]
        f = comp / meta["file"]
        assert sha256_file(f) == manifest["file_hashes"][meta["file"]], f"{meta['file']}: sha256 differs from the composite manifest"
        with np.load(f) as z:
            assert z["before"].shape == (NATIVE, NATIVE) and lungs[iid].shape == (NATIVE, NATIVE)
            rows += target_rows(iid, str(meta["patient_id"]), z, lungs[iid])
        if (n + 1) % 10 == 0 or n + 1 == len(ids):
            log(f"{n + 1}/{len(ids)} targets, {time.time() - t0:.0f} s")
    with open(out / "artifact_stats_per_target.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log_population(out, "dev_half_targets_images", ids, kind="image")
    import scipy
    import PIL
    side = {"commit": args.commit, "script_sha256": sha256_file(Path(__file__)), "n_targets": len(ids), "n_targets_full": len(dev), "dev_half_ids_digest": ids_full_digest,
            "composite_manifest_sha256": manifest["manifest_sha256"], "composite_manifest_file_sha256": sha256_file(comp / "composite_manifest.json"),
            "masks_csv_sha256": sha256_file(masks_csv), "reserved_ids_sha256": sha256_file(reserved_ids_path()), "n_rows": len(rows),
            "ring_not_ok": {fp: sorted({r["image_id"] for r in rows if r["footprint"] == fp and not r["ring_ok"]}) for fp in ("tube", "disk")},
            "nan_hf_rows": sum(1 for r in rows if r["hf_share"] != r["hf_share"]),
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "PIL": PIL.__version__, "pandas": pd.__version__}, "smoke_n": args.smoke_n, "only_present": bool(args.only_present),
            "constants": {"band_r_native": BAND_R_NATIVE, "band_r_model": BAND_R_MODEL, "ring": [RING_INNER, RING_OUTER], "ring_min_pixels": RING_MIN_PIXELS, "fft_n": FFT_N, "hf_bins": [HF_LOW_BIN, HF_HIGH_BIN]}}
    (out / "artifact_stats_sidecar.json").write_text(json.dumps(side, indent=1, sort_keys=True))
    log(f"wrote {len(rows)} rows for {len(ids)} targets")

def main() -> None:
    ap = argparse.ArgumentParser(description='per-target artifact statistics of the four edit arms on the exploratory development-half targets.')
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--composite-dir", required=True)
    p.add_argument("--masks-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--commit", default="")
    p.add_argument("--expect-masks-sha", default="")
    p.add_argument("--smoke-n", type=int, default=0)
    p.add_argument("--only-present", action="store_true")
    args = ap.parse_args()
    run_stats(args)

if __name__ == "__main__":
    main()
