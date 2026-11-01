# Draws the audit-design overview (the construction of the four arms, the primary contrast and the three descriptive contrasts with the held-out
# DenseNet-121 values) as a schematic, from the recorded held-out analysis and the training configuration; it uses no image.
# Run: python3 paper_values/isbi_design_figure.py [--check]    default: write the figure and print its size; --check exits 1 if the committed PDF differs from a fresh render or its numbers differ from numbers.tex.

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paper_values import generate_tex_figures as gtf

OUT = ROOT / "paper_isbi" / "figures_pending" / "design_overview.pdf"
W = 86.0 / 25.4 * 72.0
FONT = 7.0
LINE = 8.4
TEAL, BOXFILL, BOXEDGE = "#107878", "#dce9f7", "#6e8fc2"
ORANGE, GOLD, BLUE, GREEN, PEACH = "#d9600f", "#e3a008", "#2a7fc0", "#1a9878", "#fce9dc"
INK, FILM_EDGE = "#222222", "#1b1b1b"
RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FONT, "font.family": "DejaVu Sans", "hatch.linewidth": 0.5}

def values() -> dict[str, str]:
    pooled = gtf._gate("dn")["originals"]["pooled"]
    cfg = yaml.safe_load((ROOT / "configs" / "densenet121_original.yaml").read_text())
    size = cfg["data"]["image_size"]
    return {"q-gate-dn-rshape-m": f"{pooled['rendered_shape']['mean']:.3f}", "q-gate-dn-tshape-m": f"{pooled['tissue_shape']['mean']:.3f}",
            "q-gate-dn-inter-m": f"{pooled['interaction']['mean']:.3f}", "d-input-res": str(int(size))}

class Canvas:

    def __init__(self, height: float):
        self.h = height
        self.fig = plt.figure(figsize=(W / 72.0, height / 72.0))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.texts: list[tuple[object, tuple[float, float, float, float] | None, str]] = []

    def text(self, x, y, s, ha="left", va="center", color=INK, weight="normal", style="normal", rotation=0, within=None, label=""):
        t = self.ax.text(x, y, s, ha=ha, va=va, color=color, fontweight=weight, fontstyle=style, rotation=rotation, fontsize=FONT, linespacing=1.0)
        self.texts.append((t, within, label or s))
        return t

    def box(self, x, y, w, h, fc=BOXFILL, ec=BOXEDGE, lw=0.8, ls="-", r=2.5, z=1):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
        self.ax.add_patch(p)
        return p

    def arrow(self, p0, p1, color=TEAL, lw=0.9, rad=0.0):
        self.ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=3))

    def check(self) -> None:
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        boxes = []
        for t, within, label in self.texts:
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            if x0 < -0.01 or x1 > W + 0.01 or y0 < -0.01 or y1 > self.h + 0.01:
                raise SystemExit(f"design figure: the text {label!r} leaves the figure ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}) of {W:.1f} by {self.h:.1f}")
            if within is not None and not (x0 >= within[0] - 0.01 and x1 <= within[2] + 0.01 and y0 >= within[1] - 0.01 and y1 <= within[3] + 0.01):
                raise SystemExit(f"design figure: the text {label!r} leaves its box {within} ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
            if t.get_fontsize() != FONT:
                raise SystemExit(f"design figure: the text {label!r} is set at {t.get_fontsize()} pt, not {FONT}")
            boxes.append((x0, y0, x1, y1, label))
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if a[0] < b[2] - 0.2 and b[0] < a[2] - 0.2 and a[1] < b[3] - 0.9 and b[1] < a[3] - 0.9:
                    raise SystemExit(f"design figure: the texts {a[4]!r} and {b[4]!r} overlap")

def tube_polygon(x, y, T, side=1.0, half_width=0.036, n=24):
    t = np.linspace(0, 1, n)
    cx = x + T * (0.26 + 0.17 * t + 0.05 * np.sin(np.pi * t))
    cy = y + T * (0.27 + 0.36 * t)
    dx, dy = np.gradient(cx), np.gradient(cy)
    nrm = np.hypot(dx, dy)
    nx, ny = -dy / nrm, dx / nrm
    hw = T * half_width
    left = np.c_[cx + nx * hw, cy + ny * hw]
    right = np.c_[cx - nx * hw, cy - ny * hw]
    return np.vstack([left, right[::-1]])

def draw_film(cv: Canvas, x: float, y: float, T: float, kind: str) -> None:
    ax = cv.ax
    e = 2.6
    ax.add_patch(Polygon([(x, y), (x + e, y - e), (x + T + e, y - e), (x + T, y)], fc="#303030", ec=FILM_EDGE, lw=0.3, zorder=2))
    ax.add_patch(Polygon([(x + T, y), (x + T + e, y - e), (x + T + e, y + T - e), (x + T, y + T)], fc="#101010", ec=FILM_EDGE, lw=0.3, zorder=2))
    ax.add_patch(Rectangle((x, y), T, T, fc="#050505", ec=FILM_EDGE, lw=0.3, zorder=2))
    ax.add_patch(Ellipse((x + T * 0.5, y + T * 0.52), T * 0.86, T * 0.98, fc="#a9a9a9", ec="none", zorder=3))
    lungs = []
    for cxl in (0.335, 0.665):
        lung = Ellipse((x + T * cxl, y + T * 0.46), T * 0.22, T * 0.47, fc="#2c2c2c", ec="none", zorder=4)
        ax.add_patch(lung)
        lungs.append(lung)
        for fy in (0.31, 0.43, 0.55):
            ln, = ax.plot([x + T * (cxl - 0.12), x + T * (cxl + 0.12)], [y + T * fy] * 2, color="#5a5a5a", lw=0.35, zorder=4.2, solid_capstyle="butt")
            ln.set_clip_path(lung)
    ax.add_patch(Ellipse((x + T * 0.63, y + T * 0.62), T * 0.2, T * 0.16, fc="#7c7c7c", ec="none", zorder=4.4))
    ax.add_patch(Rectangle((x + T * 0.455, y + T * 0.14), T * 0.09, T * 0.8, fc="#d3d3d3", ec="none", zorder=4.6))
    ax.add_patch(Rectangle((x + T * 0.22, y + T * 0.205), T * 0.56, T * 0.035, fc="#dcdcdc", ec="none", zorder=4.6))
    if kind in ("tube_rendered", "tube_tissue"):
        poly = tube_polygon(x, y, T)
        if kind == "tube_rendered":
            ax.add_patch(Polygon(poly, fc="#d6d6d6", ec=ORANGE, lw=0.9, zorder=5))
        else:
            ax.add_patch(Polygon(poly, fc="none", ec=BLUE, lw=0.9, hatch="//////", zorder=5))
    else:
        c = (x + T * 0.335, y + T * 0.5)
        if kind == "disk_rendered":
            ax.add_patch(Circle(c, T * 0.1, fc="#d6d6d6", ec=GOLD, lw=1.0, zorder=5))
        else:
            ax.add_patch(Circle(c, T * 0.1, fc="none", ec=GREEN, lw=1.0, hatch="//////", zorder=5))

def render(out_dir: Path | None = None, report: bool = False) -> Path:
    v = values()
    T = 29.0
    gap = 5.5
    bh = 2 * LINE + 3.6

    y1, y2 = 0.5, 0.5 + bh + gap
    flow_bottom = y2 + bh
    body = flow_bottom + 7.5
    head_y = body + 3.6
    row1 = body + 12.4
    row2 = row1 + T + 7.0
    leg_y = row2 + T + 5.0
    right_bottom = body + 0.5 + (4 * LINE + 3.6) + 3.5 + LINE * (2.5 + 3 + 0.35) + 1.5 + 2 * LINE
    H = round(max(leg_y + 2 * LINE + 1.0, right_bottom), 1) + 0.5
    with plt.rc_context(RC):
        cv = Canvas(H)
        three = 72.0
        gx = (W - 1.0 - 3 * three) / 2
        steps = [("target film\nand lung mask", 0.5, y1, three), ("parametric tube", 0.5 + three + gx, y1, three), ("donor tissue in the\nsame footprint", 0.5 + 2 * (three + gx), y1, three),
                 (f"log-domain blend at\nnative resolution", W - 0.5 - 108.0, y2, 108.0), (f"resized to {v['d-input-res']} only at\nthe classifier input", 0.5, y2, 112.0)]
        for s, x, y, w in steps:
            cv.box(x, y, w, bh)
            cv.text(x + w / 2, y + bh / 2, s, ha="center", within=(x, y, x + w, y + bh), label=s)
        cv.arrow((0.5 + three + 1.0, y1 + bh / 2), (0.5 + three + gx - 1.0, y1 + bh / 2))
        cv.arrow((0.5 + 2 * three + gx + 1.0, y1 + bh / 2), (0.5 + 2 * (three + gx) - 1.0, y1 + bh / 2))
        x3 = 0.5 + 2 * (three + gx) + three / 2
        cv.arrow((x3, y1 + bh + 0.8), (x3, y2 - 0.8))
        cv.arrow((W - 0.5 - 108.0 - 1.0, y2 + bh / 2), (0.5 + 112.0 + 1.0, y2 + bh / 2))

        lx = 11.0
        xs = [lx, lx + T + 8.0]
        for c, s in zip(xs, ("rendered", "tissue")):
            cv.text(c + T / 2, head_y, s, ha="center", label=s)
        for r_y, s in ((row1, "tube"), (row2, "disk")):
            cv.text(4.5, r_y + T / 2, s, ha="center", rotation=90, label=f"row {s}")
        kinds = {(0, 0): "tube_rendered", (0, 1): "tube_tissue", (1, 0): "disk_rendered", (1, 1): "disk_tissue"}
        for (ri, ci), k in kinds.items():
            draw_film(cv, xs[ci], (row1, row2)[ri], T, k)
        cv.ax.add_patch(FancyBboxPatch((lx - 2.2, row1 - 4.0), 2 * T + 8.0 + 4.4 + 2.6, T + 4.0 + 3.0, boxstyle="round,pad=0,rounding_size=3", fc="none", ec=ORANGE, lw=1.0, ls=(0, (3.2, 2.0)), zorder=6))
        cv.text(0.5, leg_y + 3.0, "solid = rendered content", label="legend 1")
        cv.text(0.5, leg_y + 3.0 + LINE, "hatched = donor tissue", label="legend 2")

        rx = xs[1] + T + 14.0
        pw = W - 0.5 - rx
        top = body + 0.5
        cv.ax.add_patch(FancyBboxPatch((rx, top), pw, 4 * LINE + 3.6, boxstyle="round,pad=0,rounding_size=2.5", fc=PEACH, ec=ORANGE, lw=0.8, zorder=1))
        px = rx + 3.0
        lines = [("Primary contrast (dashed box):", ORANGE, "bold", "normal"), ("rendered tube minus tube-shaped tissue", INK, "normal", "normal"),
                 ("Estimand: the rendering's content", INK, "normal", "normal"), ("beyond same-shape tissue", INK, "normal", "normal")]
        for i, (s, col, wt, st) in enumerate(lines):
            cv.text(px, top + 1.8 + LINE * (i + 0.5), s, color=col, weight=wt, style=st, within=(rx, top, rx + pw, top + 4 * LINE + 3.6), label=s)
        y = top + 4 * LINE + 3.6 + 3.5
        cv.text(rx + 1.0, y + LINE * 0.5, "Descriptive (no margin, no verdict),", weight="bold", color=INK, label="descriptive header")
        cv.text(rx + 1.0, y + LINE * 1.5, "held-out DenseNet-121, logits:", label="descriptive units")
        for i, (s, val) in enumerate((("rendered tube minus rendered disk", v["q-gate-dn-rshape-m"]), ("tube minus disk tissue", v["q-gate-dn-tshape-m"]), ("interaction", v["q-gate-dn-inter-m"]))):
            yy = y + LINE * (2.5 + i + 0.35)
            cv.text(rx + 1.0, yy, s, label=s)
            cv.text(W - 0.5, yy, val, ha="right", label=f"value {val}")
        yy = y + LINE * (2.5 + 3 + 0.35) + 1.5
        cv.text(rx + 1.0, yy + LINE * 0.5, "the choice of control alone", color=ORANGE, style="italic", label="verdict line 1")
        cv.text(rx + 1.0, yy + LINE * 1.5, "would change the verdict", color=ORANGE, style="italic", label="verdict line 2")
        cv.check()
        out_dir = out_dir or OUT.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / OUT.name
        cv.fig.savefig(out, metadata=gtf.META)
        plt.close(cv.fig)
    if report:
        print(f"figure {W:.2f} by {H:.1f} pt = {W / 72 * 25.4:.1f} by {H / 72 * 25.4:.1f} mm = {W / 72:.3f} by {H / 72:.3f} in; flow {flow_bottom:.1f} pt, body from {body:.1f} pt; values {v}")
    return out

def smallest_label_pt(pdf: Path) -> float:
    out = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    hs = [float(m.group(2)) - float(m.group(1)) for m in re.finditer(r'yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)"', out)]
    return min(hs) / 1.164

def macro_problems() -> list[str]:
    nt = ROOT / "paper_isbi" / "numbers.tex"
    if not nt.exists():
        return []
    defs = dict(re.findall(r"\\defval\{([^}]+)\}\{([^}]*)\}", nt.read_text()))
    return [f"{k}: the figure prints {v!r}, numbers.tex has {defs.get(k)!r}" for k, v in values().items() if defs.get(k) != v]

def main() -> int:
    if "--check" in sys.argv:
        bad = macro_problems()
        if bad:
            print("the figure's numbers differ from the paper's macros:", *bad, sep="\n  ")
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            fresh = render(Path(tmp))
            if not OUT.exists() or OUT.read_bytes() != fresh.read_bytes():
                print(f"{OUT.relative_to(ROOT)} differs from a fresh run of paper_values/isbi_design_figure.py")
                return 1
        print("design_overview.pdf is current")
        return 0
    out = render(report=True)
    print("wrote", out, f"(smallest label {smallest_label_pt(out):.2f} pt)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
