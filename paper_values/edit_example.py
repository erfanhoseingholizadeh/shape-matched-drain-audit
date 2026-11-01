# Selects the example target of Fig. 1 by the committed rule and checks the persisted composite against its manifest.
# Run: python3 paper_values/edit_example.py select | render

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
INSPECT = REPO_ROOT / "review/positive_audit/inspect"
DEV = REPO_ROOT / "review/positive_audit/populations/dev_half_targets_images.ids.txt"
OUT = REPO_ROOT / "paper/figures/edit_example.pdf"
EXPECTED_SHA256 = "a470e7a1330b07afcc1c7a7132c0241f58b9818a8978e396c9486bb3e5c010b3"
RULE = "first by ascending image id among the exploratory run's persisted inspection composites whose target is in the development half"

def chosen_id() -> str:
    dev = {ln.strip().removesuffix(".png") for ln in DEV.read_text().splitlines() if ln.strip()}
    ids = sorted(p.stem for p in INSPECT.glob("*.npz") if p.stem in dev)
    assert ids, "no inspection composite belongs to the development half"
    return ids[0]

def render() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import hashlib
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved

    iid = chosen_id()
    assert_not_reserved([f"{iid}.png"], "edit_example.render")
    assert hashlib.sha256((INSPECT / f"{iid}.npz").read_bytes()).hexdigest() == EXPECTED_SHA256, "the inspection composite is not the one the rule selected when this script was committed"
    with np.load(INSPECT / f"{iid}.npz") as z:
        arr = {"before": z["before"].astype(float), "tube": z["tube"].astype(float), "tube_tissue": z["tube_tissue"][0].astype(float),
               "disk": z["disk"].astype(float), "disk_tissue": z["disk_tissue"][0].astype(float)}
    keys = [("before", "unedited"), ("tube", "rendered tube"), ("tube_tissue", "tube-shaped tissue"), ("disk", "rendered disk"), ("disk_tissue", "disk-shaped tissue")]
    fig, axes = plt.subplots(2, 5, figsize=(12.2 / 2.54, 12.2 / 2.54 * 0.5), constrained_layout=True)
    vmax = max(1.0, float(np.percentile(np.abs(arr["tube"] - arr["before"]), 99.9)))
    for j, (k, title) in enumerate(keys):
        axes[0, j].imshow(arr[k], cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(title, fontsize=5.2)
        if k == "before":
            axes[1, j].axis("off")
            cax = axes[1, j].inset_axes([0.08, 0.5, 0.84, 0.07])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="magma"), cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=4.5, length=2)
            cb.set_label("|difference|, gray levels", fontsize=5)
        else:
            axes[1, j].imshow(np.abs(arr[k] - arr["before"]), cmap="magma", vmin=0, vmax=vmax)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    fig.savefig(OUT, metadata={"CreationDate": None, "Creator": None, "Producer": None, "Title": None, "Author": None}, dpi=200)
    plt.close(fig)
    return OUT

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "select":
        print(chosen_id())
    elif cmd == "render":
        print("wrote", render())
    else:
        sys.exit("one worked example of the edit on an exploratory development-half positive target: unedited film, rendered tube,\ntube-shaped tissue, rendered disk, disk-shaped tissue, with absolute difference maps.\n\nSELECTION RULE (fixed and committed before the figure is rendered; it keys on population membership only and on no score, outcome or geometry):\n    among the composites persisted for visual inspection by the exploratory run (review/positive_audit/inspect/*.npz), keep those whose target image is in the\n    development half's exploratory target list (review/positive_audit/populations/dev_half_targets_images.ids.txt); the example is the first by ascending image id.\nThe inspection composites were exported by the exploratory run for a visual check and a montage of them was viewed then; the rule does not claim the\nchosen target was unseen, only that the choice among them is blind to every score. The tissue arms use the first of the eight donors. One example\ncannot show the distribution of placements, and the caption says so.\n\nThe figure holds one NIH ChestX-ray14 image (public dataset) with edits drawn on it; it is committed under the owner's ruling that already covers paper/figures/edit_example.pdf, which this render replaces. No held-out image is read.\n\nUsage: python3 paper_values/edit_example.py select | render\n")
