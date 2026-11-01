# Draws the data figures as vector PDF; its helpers are also used by isbi_figures.py.
# Run: python3 paper_values/generate_tex_figures.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
WIDTH_IN = 12.2 / 2.54
COLORS = {"densenet121": "#2166ac", "resnet50": "#b2182b"}
LABEL = {"densenet121": "DenseNet-121", "resnet50": "ResNet-50"}
META = {"CreationDate": None, "Creator": None, "Producer": None, "Title": None, "Author": None}
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                            "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})

def within_patient() -> Path:
    files = {
        "densenet121": ROOT / "review/densenet121_remeasure/outputs/within_patient_contrast.json",
        "resnet50": ROOT / "review/resnet50_remeasure/outputs/within_patient_contrast_resnet50.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.5))
    for ax, (arch, path) in zip(axes, files.items()):
        d = json.loads(path.read_text())["full_val"]["per_seed"]
        n_patients = len(d[sorted(d, key=int)[0]]["logit"]["distribution"])
        for seed_s in sorted(d, key=int):
            seed = int(seed_s)
            dist = np.array(d[seed_s]["logit"]["distribution"])
            jitter = (np.random.default_rng(seed).random(len(dist)) - 0.5) * 0.5
            ax.scatter(np.full(len(dist), seed) + jitter, dist, s=5, alpha=0.45, color=COLORS[arch], linewidths=0)
            mean, lo, hi = d[seed_s]["logit"]["mean_paired_diff"], d[seed_s]["logit"]["ci_low"], d[seed_s]["logit"]["ci_high"]
            ax.errorbar([seed], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D", color="black", markersize=3, capsize=2, zorder=5, linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_xticks(range(5))
        ax.set_xlabel("seed")
        ax.set_ylabel(r"paired $\Delta$logit")
        ax.set_title(f"{LABEL[arch]}, {n_patients} patients")
    fig.tight_layout()
    out = OUT / "within_patient.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

INK = "#222222"
GREY = "#8a8a8a"
BLUE, ORANGE = "#2166ac", "#c8641c"
PF = ROOT / "review/power_floor/outputs/power_floor.json"
DELTA = {"densenet121": ROOT / "review/densenet121_remeasure/outputs/delta_p_pilot_full.json",
         "resnet50": ROOT / "review/resnet50_remeasure/outputs/delta_p_pilot_resnet50.json"}

def _gate(arch: str) -> dict:
    return json.loads((ROOT / f"review/heldout/confirmatory/analysis_{'densenet121' if arch == 'dn' else 'resnet50'}.json").read_text())

def dose_response() -> Path:
    ro = {"dn": json.loads((ROOT / "review/planted_intermediate/outputs/readout.json").read_text())["points"],
          "rn": json.loads((ROOT / "review/heldout/part4/resnet50_readout.json").read_text())["points"]}
    lor = {k: v["log_odds_ratio"] for k, v in ro["dn"].items() if v["log_odds_ratio"] is not None}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), sharey=True)
    for ax, arch in zip(axes, ("dn", "rn")):
        col = COLORS["densenet121" if arch == "dn" else "resnet50"]
        g = _gate(arch)
        margin, invert = g["originals"]["verdict"]["margin"], g["originals"]["verdict"]["invert_constant"]
        ax.set_yscale("symlog", linthresh=0.2)
        ks = [k for k in ("zero", "weak", "mid30", "mid50", "strong") if k in ro[arch]]
        x = [lor[k] for k in ks]
        gap = [ro[arch][k]["audit"]["gap"] for k in ks]
        ci = [ro[arch][k]["audit"]["ci"] for k in ks]
        ax.errorbar(x, gap, yerr=[[a - c[0] for a, c in zip(gap, ci)], [c[1] - a for a, c in zip(gap, ci)]], fmt="o-", color=col, markersize=3.5, capsize=2, linewidth=0.8, zorder=4, label="negative targets")
        gk = [k for k in ks if k in g["planted"]["strengths"]]
        gx = [lor[k] + 0.09 for k in gk]
        gm = [g["planted"]["strengths"][k]["primary"]["mean"] for k in gk]
        gc = [g["planted"]["strengths"][k]["primary"]["ci95"] for k in gk]
        ax.errorbar(gx, gm, yerr=[[a - c[0] for a, c in zip(gm, gc)], [c[1] - a for a, c in zip(gm, gc)]], fmt="s", mfc="white", color=INK, markersize=3.5, capsize=2, linewidth=0.8, zorder=5, label="gate targets")
        ax.axhspan(-margin, margin, color=GREY, alpha=0.25, linewidth=0)
        ax.axhline(invert, color=ORANGE, linewidth=0.8, linestyle=":")
        ax.text(4.5, invert * 1.12, "INVERT constant", fontsize=5.5, color=ORANGE, va="bottom", ha="right")
        ax.text(4.5, margin * 1.2, "margin (shaded)", fontsize=5.5, color=INK, va="bottom", ha="right")
        ax.set_yticks([-0.1, 0, 0.1, 0.5, 1, 2, 5])
        ax.set_yticklabels(["\u22120.1", "0", "0.1", "0.5", "1", "2", "5"])
        ax.set_xlabel("planting log odds ratio")
        ax.set_title(LABEL["densenet121" if arch == "dn" else "resnet50"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"insertion $-$ control ($\Delta$logit, symlog)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    out = OUT / "dose_response.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def factorial_gate() -> Path:
    expl = json.loads((ROOT / "review/positive_audit/positive_analysis.json").read_text())
    labels = [("tube_rendered", "rendered\ntube"), ("tube_tissue", "tissue\ntube"), ("disk_rendered", "rendered\ndisk"), ("disk_tissue", "tissue\ndisk"), ("primary", "tube \u2212\ntissue tube")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), sharey=True)
    for ax, arch in zip(axes, ("dn", "rn")):
        name = "densenet121" if arch == "dn" else "resnet50"
        g = _gate(arch)["originals"]
        gp, ep = g["pooled"], expl[name]["pooled"]
        margin = g["verdict"]["margin"]
        xs = [0, 1, 2, 3, 4.6]
        for x, (key, _) in zip(xs, labels):
            for dx, blk, filled in ((-0.13, gp[key], True), (0.13, ep[key], False)):
                m, (lo, hi) = blk["mean"], blk["ci95"]
                ax.errorbar([x + dx], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=COLORS[name], mfc=COLORS[name] if filled else "white", markersize=3.5, capsize=2, linewidth=0.9, zorder=4)
        ax.axhspan(-margin, margin, xmin=0.8, xmax=1.0, color=GREY, alpha=0.3, linewidth=0)
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([t for _, t in labels], fontsize=5.0)
        ax.set_title(LABEL[name])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"arm $-$ unedited ($\Delta$logit)")
    axes[0].plot([], [], "o", color=INK, markersize=3.5, label="gate (confirmatory)")
    axes[0].plot([], [], "o", mfc="white", color=INK, markersize=3.5, label="exploratory run")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    out = OUT / "factorial.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def gate_per_checkpoint() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.5), sharey=True)
    for ax, arch in zip(axes, ("dn", "rn")):
        name = "densenet121" if arch == "dn" else "resnet50"
        g = _gate(arch)["originals"]
        margin = g["verdict"]["margin"]
        for i, lab in enumerate(g["labels"]):
            b = g["per_checkpoint_descriptive"][lab]["primary"]
            ax.errorbar([i], [b["mean"]], yerr=[[b["mean"] - b["ci90"][0]], [b["ci90"][1] - b["mean"]]], fmt="o", color=COLORS[name], markersize=3.5, capsize=2, linewidth=0.9, zorder=4)
        pm = g["pooled"]["primary"]
        ax.errorbar([5.5], [pm["mean"]], yerr=[[pm["mean"] - pm["ci90"][0]], [pm["ci90"][1] - pm["mean"]]], fmt="D", color="black", markersize=4, capsize=2, linewidth=1.1, zorder=5)
        lo, hi = g["recipe_level_secondary"]["t95"]
        ax.errorbar([6.6], [g["recipe_level_secondary"]["mean"]], yerr=[[g["recipe_level_secondary"]["mean"] - lo], [hi - g["recipe_level_secondary"]["mean"]]], fmt="^", mfc="white", color="black", markersize=4, capsize=2, linewidth=1.1, zorder=5)
        ax.axhspan(-margin, margin, color=GREY, alpha=0.25, linewidth=0)
        ax.axhline(0, color=GREY, linewidth=0.6, linestyle="--")
        ax.set_xticks(list(range(5)) + [5.5, 6.6])
        ax.set_xticklabels([str(i) for i in range(5)] + ["pooled\n(90%)", "recipe\n(95%)"], fontsize=6)
        ax.set_xlabel("original checkpoint (seed)")
        ax.set_title(LABEL[name])
        ax.set_xlim(-0.6, 7.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"rendered tube $-$ tube-shaped tissue ($\Delta$logit)")
    fig.tight_layout()
    out = OUT / "gate_checkpoints.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def population_flow() -> Path:
    sc = json.loads((ROOT / "review/dataset_splits/split_counts.json").read_text())
    exp = json.loads((ROOT / "review/positive_audit/populations/index.json").read_text())
    gate_idx = json.loads((ROOT / "review/heldout/confirmatory/populations/index.json").read_text())
    n = lambda x: f"{x:,}"
    n_sel = json.loads((ROOT / "review/positive_audit/select_half_provenance.json").read_text())["S_int"]
    n_rint = json.loads((ROOT / "review/clean_population/outputs/clean_population.json").read_text())["never_selected_half_R_int"]["patients"]
    rows = [
        ("Train split", f"{n(sc['train_patients'])} patients", ["F", "", "", "", "", "", ""], ["no", "no", "no"]),
        ("Validation: selection half", f"{n(n_sel)} patients", ["", "S", "e", "", "", "", ""], ["no", "no", "no"]),
        ("Validation: never-selected half", f"{n(n_rint)} patients", ["", "", "E", "", "T D", "", ""], ["yes", "part", "part"]),
        ("Test: development half", f"{n(sc['test_gate_subset_dev_patients'])} patients", ["", "", "", "T", "T D", "T D", ""], ["yes", "yes", "yes"]),
        ("Test: gate half", f"{n(sc['test_gate_subset_gate_patients'])} patients", ["", "", "", "", "", "", "T D"], ["yes", "yes", "yes"]),
    ]
    steps = ["train", "select", "validity,\nobserv.", "negative\naudit,\ncontrols", "explor.\npositives", "same-\ndesign\n(dev)", "gate\nread\n(once)"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(-1.1, 7.2); ax.axis("off")
    x0, cw = 2.9, 0.92
    for j, st in enumerate(steps):
        ax.text(x0 + cw * (j + 0.5), 6.1, st, ha="center", va="center", fontsize=5.0)
    ax.annotate("", xy=(x0 + cw * 7, 6.95), xytext=(x0, 6.95), arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": INK})
    ax.text(x0, 7.12, "order of runs", fontsize=5.5, va="bottom")
    ax.vlines(x0 + cw * 5, 0.1, 5.5, color=ORANGE, linewidth=0.9, linestyle="--")
    ax.text(x0 + cw * 5, 0.0, "protocol frozen", fontsize=5.3, color=ORANGE, ha="center", va="top")
    hx = x0 + cw * 7 + 0.3
    for k, hdr in enumerate(("orig.", "plant.\nDN", "plant.\nRN")):
        ax.text(hx + 0.98 * k + 0.35, 6.1, hdr, ha="center", va="center", fontsize=5.0)
    ax.text(hx + 0.98 + 0.35, 6.8, "held out from", fontsize=5.5, ha="center", va="bottom")
    for i, (name, size, cells, held) in enumerate(rows):
        y = 5.0 - i * 1.0
        ax.text(0.0, y + 0.12, name, fontsize=5.8, va="center")
        ax.text(0.0, y - 0.28, size, fontsize=5.0, va="center", color=GREY)
        ax.hlines(y - 0.5, 0, hx + 2.9, color=GREY, linewidth=0.3)
        for j, c in enumerate(cells):
            if c:
                ax.text(x0 + cw * (j + 0.5), y, c, ha="center", va="center", fontsize=6.5, color=COLORS["densenet121"], fontweight="bold")
        for k, h in enumerate(held):
            sym = {"yes": "\u25a0", "part": "\u25d0", "no": "\u25a1"}[h]
            ax.text(hx + 0.98 * k + 0.35, y, sym, ha="center", va="center", fontsize=7)
    ax.text(0.0, -0.45, "F fit   S select an epoch   E evaluate   e within-patient contrast only   T target of an edit   D donor of tissue", fontsize=5.0, va="center")
    ax.text(0.0, -0.85, "\u25a0 held out   \u25d0 partly (the planted checkpoints' selection half overlaps half of the never-selected half)   \u25a1 not held out", fontsize=5.0, va="center")
    fig.tight_layout()
    out = OUT / "population_flow.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

PURPLE = "#7b3294"
R83 = ROOT / "review/replication_512_readout"
ARCHS_ = (("densenet121", "dn"), ("resnet50", "rn"))

def _band(ax, margin, color=GREY, alpha=0.22):
    ax.axhspan(-margin, margin, color=color, alpha=alpha, linewidth=0, zorder=0)
    ax.axhline(0, color=INK, linewidth=0.5, zorder=1)

def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)

def e5_resolution() -> Path:
    rd = json.loads((R83 / "e5/readout.json").read_text())
    ih = json.loads((ROOT / "review/replication_512/e5_in_hand.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), sharey=True)
    for ax, (arch, _) in zip(axes, ARCHS_):
        x = rd[arch]
        orig = x["sets"]["orig_valid"]
        m512, m224 = x["margin_512"], orig["pooled"]["margin_224"]
        _band(ax, m512)
        for s in (m224, -m224):
            ax.axhline(s, color=INK, linewidth=0.6, linestyle=":", zorder=1)
        e4 = {int(k): v for k, v in ih[arch]["e4_matched_seeds_5_to_9"]["per_checkpoint"].items()}
        e5 = {int(l.rsplit("seed", 1)[1]): v for l, v in orig["per_checkpoint_primary"].items()}
        ctl = {int(k): v["control_224_micro32"] for k, v in x["control_224"]["per_seed"].items()}
        for s in sorted(e5):
            ys = [e4[s]] + ([ctl[s]] if s in ctl else []) + [e5[s]]
            ax.plot([s] * len(ys), ys, color=GREY, linewidth=0.6, zorder=2)
        ax.scatter(sorted(e4), [e4[s] for s in sorted(e4)], s=16, marker="o", color=BLUE, zorder=4, label="224 pixels, batch of 128", linewidths=0)
        ax.scatter(sorted(ctl), [ctl[s] for s in sorted(ctl)], s=16, marker="s", color=ORANGE, zorder=4, label="224 pixels, micro-batches of 32", linewidths=0)
        ax.scatter(sorted(e5), [e5[s] for s in sorted(e5)], s=20, marker="^", color=PURPLE, zorder=4, label="512 pixels, micro-batches of 32", linewidths=0)
        p = orig["pooled"]
        ax.errorbar([10.2], [p["mean"]], yerr=[[p["mean"] - p["ci90"][0]], [p["ci90"][1] - p["mean"]]], fmt="^", color=PURPLE, markersize=5, capsize=2, linewidth=1.0, zorder=5)
        ax.text(10.2, p["ci90"][1] + 0.03, "pooled", fontsize=5.5, ha="center", va="bottom", color=INK)
        ax.set_xticks(sorted(e5))
        ax.set_xlim(4.5, 10.9)
        ax.set_xlabel("seed")
        ax.set_title(LABEL[arch])
        _clean(ax)
    axes[0].set_ylabel(r"primary contrast ($\Delta$logit)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=6, handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = OUT / "e5_resolution.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def e3_ladder() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.7), sharey=True)
    for ax, (arch, _) in zip(axes, ARCHS_):
        d = json.loads(ROOT.joinpath(f"review/further_readout/e3/e3_{arch}.json").read_text())
        margin = d["e3"]["margin"]
        delta = {r["setting"]: r.get("realized_delta") for r in d["ladder"]}
        rungs = [r for r in ("zero", "weak", "e3lo", "e3hi", "mid30", "mid50", "strong") if r in d["e3"]["settings"]]
        ax.set_yscale("symlog", linthresh=0.2)
        _band(ax, margin)
        for i, r in enumerate(rungs):
            s = d["e3"]["settings"][r]["raw"]
            seeds = list(d["e3"]["settings"][r]["per_seed_raw"].values())
            seeds = [v["mean"] if isinstance(v, dict) else v for v in seeds]
            ax.scatter([i + 0.18] * len(seeds), seeds, s=5, color=GREY, linewidths=0, zorder=3)
            eq = s["verdict"] == "EQUIVALENT"
            ax.errorbar([i], [s["mean"]], yerr=[[s["mean"] - s["ci90"][0]], [s["ci90"][1] - s["mean"]]], fmt="o", mfc=COLORS[arch] if eq else "white", mec=COLORS[arch], color=COLORS[arch], markersize=4.5, capsize=2, linewidth=0.9, zorder=5)
        ax.set_xticks(range(len(rungs)))
        ax.set_xticklabels([(f"{r}\n{delta[r]:+.3f}" if delta.get(r) is not None else r).replace("-", "−") for r in rungs], fontsize=5.5)
        ax.set_yticks([-0.1, 0, 0.1, 0.5, 1, 2, 5])
        ax.set_yticklabels(["−0.1", "0", "0.1", "0.5", "1", "2", "5"])
        ax.set_xlabel("rung and realized probability difference", fontsize=6.5)
        ax.set_title(LABEL[arch])
        _clean(ax)
    axes[0].set_ylabel(r"primary contrast ($\Delta$logit, symlog)")
    axes[1].scatter([], [], s=18, marker="o", facecolor="black", edgecolor="black", label="EQUIVALENT")
    axes[1].scatter([], [], s=18, marker="o", facecolor="white", edgecolor="black", label="EFFECT")
    axes[1].legend(frameon=False, loc="upper left", fontsize=5.5, handletextpad=0.2)
    fig.tight_layout()
    out = OUT / "e3_ladder.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def e4_seeds() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), sharey=True)
    for ax, (arch, _) in zip(axes, ARCHS_):
        e4 = json.loads((ROOT / f"review/further_readout/e4/e4_{arch}.json").read_text())["e4"]
        v = e4["valid_only"]
        per = e4["per_checkpoint_primary"]
        fam = {0: sorted(k for k in per if "_original_" in k), 1: sorted(k for k in per if k.startswith("e4_"))}
        _band(ax, e4["margin"])
        rng = np.random.default_rng(0)
        for x0, keys in fam.items():
            ys = [per[k] for k in keys]
            ax.scatter(x0 + (rng.random(len(ys)) - 0.5) * 0.35, ys, s=9, color=COLORS[arch], alpha=0.7, linewidths=0, zorder=3)
        for x0, p in ((0, v["family_pooled"]["originals"]), (1, v["family_pooled"]["e4"]), (2.1, v["combined_pool"])):
            ax.errorbar([x0 + (0.32 if x0 < 2 else 0)], [p["mean"]], yerr=[[p["mean"] - p["ci90"][0]], [p["ci90"][1] - p["mean"]]], fmt="D", color=INK, markersize=3.5, capsize=2, linewidth=0.9, zorder=5)
        ax.set_xticks([0, 1, 2.1])
        ax.set_xticklabels([f"original\n({len(fam[0])} seeds)", f"further\n({len(fam[1])} seeds)", f"all {len(per)},\npooled"], fontsize=6)
        ax.set_xlim(-0.6, 2.7)
        ax.set_title(LABEL[arch])
        _clean(ax)
    axes[0].set_ylabel(r"primary contrast ($\Delta$logit)")
    fig.tight_layout()
    out = OUT / "e4_seeds.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def e5_manip() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.5), sharey=True)
    for ax, (arch, _) in zip(axes, ARCHS_):
        m = json.loads((R83 / f"e5_manip512/{arch}/manipulation_readout.json").read_text())
        ax.axhline(0.5, color=INK, linewidth=0.5, linestyle="--", zorder=1)
        pos, ticks, labels = 0.0, [], []
        for g in ("zero", "strong"):
            v = m["groups"][g]["valid_only"]
            for sd in v["seeds"]:
                r = v["per_seed"][str(sd)]
                for dx, key, col, mk in ((-0.13, "auc_unedited", BLUE, "o"), (0.13, "auc_edited", ORANGE, "s")):
                    a = r[key]
                    ax.errorbar([pos + dx], [a["auc"]], yerr=[[a["auc"] - a["ci_low"]], [a["ci_high"] - a["auc"]]], fmt=mk, color=col, markersize=3.8, capsize=1.5, linewidth=0.8, zorder=4)
                ax.plot([pos - 0.13, pos + 0.13], [r["auc_unedited"]["auc"], r["auc_edited"]["auc"]], color=GREY, linewidth=0.6, zorder=2)
                ticks.append(pos)
                labels.append(str(sd))
                pos += 1.0
            ax.text(pos - 2.0, -0.12, g, ha="center", va="top", fontsize=6.5, color=INK)
            pos += 0.8
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("planted seed", labelpad=12)
        ax.set_title(LABEL[arch])
        _clean(ax)
    axes[0].set_ylabel("AUC on the conflict set")
    axes[0].errorbar([], [], yerr=[], fmt="o", color=BLUE, markersize=3.8, label="unedited images")
    axes[0].errorbar([], [], yerr=[], fmt="s", color=ORANGE, markersize=3.8, label="edited images")
    axes[0].legend(frameon=False, loc="lower left", fontsize=5.5, handletextpad=0.2)
    fig.tight_layout()
    out = OUT / "e5_manip.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

def device_check() -> Path:
    d = json.loads((R83 / "device_check_result.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), sharex=True)
    for ax, (arch, _) in zip(axes, ARCHS_):
        row, yt, yl = 0, [], []
        for grp, gname in (("originals", "original"), ("zero", "zero strength")):
            g = d["arch"][arch]["groups"][grp]
            thr = g["verdict"]["threshold"]
            n = len(g["checkpoints"])
            ax.axvspan(-thr, thr, ymin=0, ymax=1, color=GREY, alpha=0.0)
            ax.fill_betweenx([row - 0.5, row + n - 0.5], -thr, thr, color=GREY, alpha=0.22, linewidth=0, zorder=0)
            for i, (name, c) in enumerate(g["checkpoints"].items()):
                ax.scatter([c["difference_gpu_minus_cpu"]], [row + i], s=12, color=COLORS[arch], zorder=4, linewidths=0)
                yt.append(row + i)
                yl.append(name.rsplit("_", 1)[1])
            ax.text(thr * 1.04, row + (n - 1) / 2, gname, fontsize=5.5, va="center", ha="left", color=INK)
            row += n + 1
        ax.axvline(0, color=INK, linewidth=0.5, zorder=1)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel(r"GPU $-$ CPU primary contrast ($\Delta$logit)")
        ax.set_title(LABEL[arch])
        _clean(ax)
    fig.tight_layout()
    out = OUT / "device_check.pdf"
    fig.savefig(out, metadata=META)
    plt.close(fig)
    return out

FIGURES = (e5_resolution, e3_ladder, e4_seeds, e5_manip, device_check, within_patient, dose_response, factorial_gate, gate_per_checkpoint, population_flow)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        print("wrote", f())
    return 0

if __name__ == "__main__":
    sys.exit(main())
