# Writes the verdict table from the recorded intervals and checks that the registered rule recovers each label.
# Run: python3 paper_values/isbi_tables.py [--check]    --check exits 1 if paper_isbi/tables/verdicts.tex differs from a fresh render.

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis import positive_analysis as pa

OUT = ROOT / "paper_isbi" / "tables"
SHORT = {"dn": "DNet", "rn": "RNet"}
DIR = {"positive": "Pos", "negative": "Neg", "not established": "NE"}

def _analysis(name: str) -> dict:
    return json.loads((ROOT / f"review/heldout/confirmatory/analysis_{name}.json").read_text())

def _columns(primary: dict, margin: float, invert: float) -> tuple[str, str, dict]:
    v = pa.verdict(primary, margin, invert)
    inside = v["equivalent_90_inside_margin"]
    lo95, hi95 = primary["ci95"]
    direction = "positive" if lo95 > 0 else ("negative" if hi95 < 0 else "not established")
    has_direction = direction != "not established"

    label = ("EQUIVALENT" if inside else ("EFFECT" if has_direction else "INCONCLUSIVE"))
    flag = bool(inside and has_direction)
    assert (label, flag) == (v["label"], v["effect_inside_margin_flag"]), (label, flag, v)
    return ("yes" if inside else "no"), direction, v

def render_gate() -> str:
    rows = []
    for arch, name, tag in (("dn", "densenet121", "DenseNet-121"), ("rn", "resnet50", "ResNet-50")):
        a = _analysis(name)
        v0 = a["originals"]["verdict"]
        inside, direction, _ = _columns(a["originals"]["pooled"]["primary"], v0["margin"], v0["invert_constant"])
        h95 = "\\val{q-gate-dn-primary-h95-d4}" if arch == "dn" else "\\val{q-gate-rn-primary-h95}"
        rows.append((f"{SHORT[arch]}, Orig.", f"\\val{{q-gate-{arch}-primary-m}}", f"[\\val{{q-gate-{arch}-primary-l90}},\\val{{q-gate-{arch}-primary-h90}}]",
                     f"[\\val{{q-gate-{arch}-primary-l95}},{h95}]", f"$\\pm$\\val{{q-margin-{arch}}}", inside, direction))
    dn = _analysis("densenet121")
    weak = dn["planted"]["strengths"]["weak"]["primary"]
    inside, direction, _ = _columns(weak, dn["originals"]["verdict"]["margin"], dn["originals"]["verdict"]["invert_constant"])

    rows.append(("DNet, Weak", "\\val{q-gate-dn-pl-weak-m-s}", "[\\val{i91-pl-dn-weak-l90},\\val{i91-pl-dn-weak-h90}]", "[\\val{q-gate-dn-pl-weak-l95},\\val{q-gate-dn-pl-weak-h95}]",
                 "$\\pm$\\val{q-margin-dn}", inside, direction))
    for arch, name, tag in (("dn", "densenet121", "DenseNet-121"), ("rn", "resnet50", "ResNet-50")):
        a = _analysis(name)
        s = a["planted"]["strengths"]["strong"]["primary"]
        inside, direction, v = _columns(s, a["originals"]["verdict"]["margin"], a["originals"]["verdict"]["invert_constant"])
        assert v["mean_over_invert_constant"] >= 1 and direction == "positive", (arch, v)
        rows.append((f"{SHORT[arch]}, Strong", f"\\val{{q-gate-{arch}-pl-strong-m}}", f"[\\val{{i91-pl-{arch}-strong-l90}},\\val{{i91-pl-{arch}-strong-h90}}]",
                     f"[\\val{{q-gate-{arch}-pl-strong-l95}},\\val{{q-gate-{arch}-pl-strong-h95}}]", f"bar $\\geq$\\val{{q-invert-{arch}}}", inside, direction))
    fmt = lambda rs: "\n\n".join(f"{r[0]}\n& {r[1]} & {r[2]} - {r[3]} & {r[4]} & {'Y' if r[5] == 'yes' else 'N'} / {DIR[r[6]]} \\\\" for r in rs) + "\n"
    return (
        "\\begin{table}[t]\n"
        "\\caption{Pooled tube--tissue logit contrast on held-out targets\n"
        "(\\val{q-gate-dn-n-targets} images, \\val{q-gate-dn-n-patients} patients). CIs are 90\\% and 95\\%.}\n"
        "\\label{tab:verdicts}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\n"
        "Checkpoint & Contrast & 90\\% CI - 95\\% CI & Margin & Verdict \\\\\n"
        "\\midrule\n"
        + fmt(rows[:2]) +
        "\\midrule\n"
        + fmt(rows[2:]) +
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\n"
        "\\vspace{2pt}\n"
        "\\parbox{\\columnwidth}{\\scriptsize\n"
        "DNet: DenseNet-121; RNet: ResNet-50. Y/N: within/outside equivalence margin.\n"
        "NE: direction not established; Neg/Pos: negative/positive direction.\n"
        "}\n"
        "\\end{table}\n"
    )

def render_all() -> dict[str, str]:
    return {"verdicts": render_gate()}

def main() -> int:
    texts = render_all()
    if "--check" in sys.argv:
        bad = [n for n, t in texts.items() if not (OUT / f"{n}.tex").exists() or (OUT / f"{n}.tex").read_text() != t]
        if bad:
            print(f"paper_isbi/tables/{bad} differ from a fresh run of paper_values/isbi_tables.py")
            return 1
        print("tables are current")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in texts.items():
        (OUT / f"{n}.tex").write_text(t)
        print("wrote", OUT / f"{n}.tex")
    return 0

if __name__ == "__main__":
    sys.exit(main())
