# Renders the five-panel strip of the edit on one target, from the persisted composite.
# Run: python3 paper_values/isbi_edit_strip.py render | sizes

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_values import edit_example as p73

OUT = ROOT / "paper_isbi" / "figures" / "edit_example_isbi.pdf"
TEXTWIDTH_MM = 178.0
STRIP_FRACTION = 0.685
WIDTH_CM = STRIP_FRACTION * TEXTWIDTH_MM / 10.0
HEIGHT_CM = 3.2
FONT_PT = 7.0
CONTOUR_COLOR, CONTOUR_LW = "#00e5ff", 0.5
WINDOW = 256
TITLES = {"before": "unedited", "tube": "rendered\ntube", "tube_tissue": "tube-shaped\ntissue", "disk": "rendered\ndisk", "disk_tissue": "disk-shaped\ntissue"}

def crop_origin(tube_mask: np.ndarray, disk_mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(tube_mask | disk_mask)
    cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    y0, x0 = cy - WINDOW // 2, cx - WINDOW // 2
    h, w = tube_mask.shape
    assert 0 <= y0 and y0 + WINDOW <= h and 0 <= x0 and x0 + WINDOW <= w, "the window leaves the film"
    assert y0 <= ys.min() and ys.max() < y0 + WINDOW and x0 <= xs.min() and xs.max() < x0 + WINDOW, "the window does not hold both shapes"
    return y0, x0

def render() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved

    iid = p73.chosen_id()
    assert_not_reserved([f"{iid}.png"], "isbi_edit_strip.render")
    path = p73.INSPECT / f"{iid}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == p73.EXPECTED_SHA256, "the inspection composite is not the one the rule selected when edit_example.py was committed"
    with np.load(path) as z:
        arr = {"before": z["before"].astype(float), "tube": z["tube"].astype(float), "tube_tissue": z["tube_tissue"][0].astype(float),
               "disk": z["disk"].astype(float), "disk_tissue": z["disk_tissue"][0].astype(float)}
        y0, x0 = crop_origin(z["tube_mask"], z["disk_mask"])
        masks = {"tube": z["tube_mask"].astype(float)[y0:y0 + WINDOW, x0:x0 + WINDOW], "disk": z["disk_mask"].astype(float)[y0:y0 + WINDOW, x0:x0 + WINDOW]}
    arr = {k: v[y0:y0 + WINDOW, x0:x0 + WINDOW] for k, v in arr.items()}
    keys = ["before", "tube", "tube_tissue", "disk", "disk_tissue"]

    outlines = {"before": ("tube", "disk"), "tube": ("tube",), "tube_tissue": ("tube",), "disk": ("disk",), "disk_tissue": ("disk",)}
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54), constrained_layout=True)
    for j, k in enumerate(keys):
        axes[j].imshow(arr[k], cmap="gray", vmin=0, vmax=255)
        for m in outlines[k]:
            axes[j].contour(masks[m], levels=[0.5], colors=CONTOUR_COLOR, linewidths=CONTOUR_LW)
        axes[j].set_title(TITLES[k], fontsize=FONT_PT, linespacing=1.0)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None, "Creator": None, "Producer": None, "Title": None, "Author": None}, dpi=200)
    plt.close(fig)
    return OUT

def sizes() -> None:
    import subprocess
    import re
    out = subprocess.run(["pdftotext", "-bbox", str(OUT), "-"], capture_output=True, text=True, check=True).stdout
    heights = sorted({round(float(m.group(2)) - float(m.group(1)), 2) for m in re.finditer(r'yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)"', out)} or [])
    page = re.search(r'<page width="([\d.]+)" height="([\d.]+)"', out)
    print(f"labels set at {FONT_PT} pt; page {page.group(1)} x {page.group(2)} pt ({float(page.group(1)) / 72 * 25.4:.2f} mm wide); placed width {STRIP_FRACTION} x {TEXTWIDTH_MM} mm = {STRIP_FRACTION * TEXTWIDTH_MM:.2f} mm")
    print("distinct word-box heights (pt):", heights)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "render":
        print("wrote", render())
    elif cmd == "sizes":
        sizes()
    else:
        sys.exit("the four-arm strip re-rendered for the ISBI paper with embedded TrueType (Type 42) fonts and every label at least 7 pt at printed size.\n\nSame image and selection rule as paper/figures/edit_example.pdf (paper_values/edit_example.py): the first exploratory development-half inspection composite by ascending id,\nread through the reserved-image guard and the sha256 check of that script. Earlier version: two rows by five panels, absolute-difference maps below, a colorbar in the empty first panel of row two.\nLater version: the difference-map row and its colorbar are dropped, so the strip is one row of\nfive panels, and the height is set for one row; the images, their order, the 7 pt labels and the width (0.685 of the text width, factor 1.000) are unchanged.\nWhat changes: matplotlib's pdf.fonttype is 42 (the old file carried Type 3 fonts), the labels are 7 pt (the old ones were 4.5 to 5.2 pt) and the panel titles break onto two lines to fit\nthe panel width, and the figure's native width is the width main.tex places it at (0.685 of the 178 mm text width), so the placed factor is 1.000 and the printed type size is the size set here.\n\nCurrent version: the strip shows zoomed crops of the same target, the same five arms and the same donor. THE CROP RULE (fixed before any crop was rendered): one window for every panel, a square of WINDOW = 256 pixels (a quarter of the film's width) cut from the 1024-pixel arrays and centred on the centre of the bounding box of the union of the\nrendered tube's and the rendered disk's masks (integer centre, rounded down). The rule keys on the placement alone. The masks' extents were read as numbers before the rule was fixed (tube 199 by 60 pixels, disk 66 by 66, union rows 203 to 401 and\ncolumns 648 to 719), to confirm that a quarter-width window holds both shapes with at least 28 pixels of margin; no crop had been viewed. The caption states the rule in words and says that a montage of the persisted composites had been viewed.\nThe image, the selection of the target, the donor, the order of the panels, the Type 42 fonts, the 7 pt labels, the width (0.685 of the text width, factor 1.000) and the float's height are unchanged.\n\npaper/figures/edit_example.pdf is not touched. The file written here, paper_isbi/figures/edit_example_isbi.pdf, shows the same one NIH ChestX-ray14 image under the owner's ruling for that figure.\n\nUsage: python3 paper_values/isbi_edit_strip.py render | sizes\n")
