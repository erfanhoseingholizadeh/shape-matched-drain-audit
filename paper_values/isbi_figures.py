# Draws the held-out contrasts, the planted checkpoints and the originals figure from the recorded analysis outputs.
# Run: python3 paper_values/isbi_figures.py [--check]    --check exits 1 if the committed PDF differs from a fresh render.

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paper_values import generate_tex_figures as gtf

OUT = ROOT / "paper_isbi" / "figures"
WIDTH_IN = 178.0 / 25.4
HEIGHT_IN = 1.35
ARCH = (("dn", "densenet121"), ("rn", "resnet50"))

MARK = {"densenet121": "o", "resnet50": "s"}
LINE = {"densenet121": "-", "resnet50": "--"}
RC = {"pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}

def gate_and_dose(out_dir: Path = OUT) -> Path:
    ro = {"dn": json.loads((ROOT / "review/planted_intermediate/outputs/readout.json").read_text())["points"]}
    lor = {k: v["log_odds_ratio"] for k, v in ro["dn"].items() if v["log_odds_ratio"] is not None}
    with plt.rc_context(RC):
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(WIDTH_IN, HEIGHT_IN), gridspec_kw={"width_ratios": [1.3, 0.95, 0.62]})

        labels = [("tube_rendered", "rendered\ntube"), ("tube_tissue", "tissue\ntube"), ("disk_rendered", "rendered\ndisk"), ("disk_tissue", "tissue\ndisk"), ("primary", "rendered minus\ntissue tube")]
        xs = [0, 1, 2, 3, 4.5]
        for off, (arch, name) in zip((-0.13, 0.13), ARCH):
            g = gtf._gate(arch)["originals"]
            margin = g["verdict"]["margin"]
            for x, (key, _) in zip(xs, labels):
                blk = g["pooled"][key]
                m, (lo, hi) = blk["mean"], blk["ci95"]
                ax_a.errorbar([x + off], [m], yerr=[[m - lo], [hi - m]], fmt=MARK[name], color=gtf.COLORS[name], markersize=3.6, capsize=2, linewidth=0.9, zorder=4, label=gtf.LABEL[name] if x == 0 else None)
            ax_a.axhspan(-margin, margin, xmin=0.77, xmax=1.0, color=gtf.COLORS[name], alpha=0.13, linewidth=0)
        ax_a.axhline(0, color=gtf.GREY, linewidth=0.6, linestyle="--")
        ax_a.set_xticks(xs)
        ax_a.set_xticklabels([t for _, t in labels])
        ax_a.set_ylabel("arm minus\nunedited (Δlogit)")
        ax_a.set_title("(a) Arms and primary contrast", loc="left")
        ax_a.spines[["top", "right"]].set_visible(False)

        for arch, name in ARCH:
            g = gtf._gate(arch)
            col = gtf.COLORS[name]
            margin, invert = g["originals"]["verdict"]["margin"], g["originals"]["verdict"]["invert_constant"]
            ks = [k for k in ("zero", "weak", "mid30", "mid50", "strong") if k in g["planted"]["strengths"]]
            x = [lor[k] + (-0.08 if arch == "dn" else 0.08) for k in ks]
            m = [g["planted"]["strengths"][k]["primary"]["mean"] for k in ks]
            c = [g["planted"]["strengths"][k]["primary"]["ci95"] for k in ks]
            ax_b.errorbar(x, m, yerr=[[a - ci[0] for a, ci in zip(m, c)], [ci[1] - a for a, ci in zip(m, c)]], fmt=MARK[name] + LINE[name], color=col, markersize=3.6, capsize=2, linewidth=0.8, zorder=4, label=gtf.LABEL[name])
            ax_b.axhspan(-margin, margin, color=col, alpha=0.13, linewidth=0)
            ax_b.axhline(invert, color=col, linewidth=0.8, linestyle=":")
        ax_b.set_yscale("symlog", linthresh=0.2)
        ax_b.set_yticks([-0.1, 0, 0.1, 0.5, 1, 2, 5])
        ax_b.set_yticklabels(["−0.1", "0", "0.1", "0.5", "1", "2", "5"])
        ax_b.set_xlabel("planting log odds ratio")
        ax_b.set_ylabel("tube minus tissue\ntube (Δlogit)")
        ax_b.set_title("(b) Planted controls", loc="left")
        ax_b.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
        ax_b.spines[["top", "right"]].set_visible(False)

        for off, (arch, name) in zip((-0.14, 0.14), ARCH):
            g = gtf._gate(arch)["originals"]
            margin = g["verdict"]["margin"]
            pcs = g["per_checkpoint_descriptive"]
            for i in range(5):
                blk = pcs[f"{name}_original_seed{i}"]["primary"]
                m, (lo, hi) = blk["mean"], blk["ci90"]
                ax_c.errorbar([i + off], [m], yerr=[[m - lo], [hi - m]], fmt=MARK[name], color=gtf.COLORS[name], markersize=3.6, capsize=2, linewidth=0.9, zorder=4, label=gtf.LABEL[name] if i == 0 else None)
            ax_c.axhspan(-margin, margin, color=gtf.COLORS[name], alpha=0.13, linewidth=0)
        ax_c.axhline(0, color=gtf.GREY, linewidth=0.6, linestyle="--")
        ax_c.set_xticks(range(5))
        ax_c.set_xticklabels([str(i) for i in range(5)])
        ax_c.set_xlabel("seed")
        ax_c.set_ylabel("primary contrast\n(Δlogit)")
        ax_c.set_title("(c) Checkpoints", loc="left")
        ax_c.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.4, w_pad=1.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "gate_and_dose.pdf"
        fig.savefig(out, metadata=gtf.META)
        plt.close(fig)
    return out

FIGURES = (gate_and_dose,)

def main() -> int:
    if "--check" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = gate_and_dose(Path(tmp))
            committed = OUT / fresh.name
            if not committed.exists() or committed.read_bytes() != fresh.read_bytes():
                print(f"paper_isbi/figures/{fresh.name} differs from a fresh run of paper_values/isbi_figures.py")
                return 1
        print("gate_and_dose.pdf is current")
        return 0
    print("wrote", gate_and_dose())
    return 0

if __name__ == "__main__":
    sys.exit(main())
