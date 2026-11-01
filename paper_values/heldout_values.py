# Reads the recorded analysis outputs and writes the flat JSON of values (contrasts, intervals, margins, counts) the paper cites.
# Run: python3 paper_values/heldout_values.py [--write-sources] [--all]

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "review/heldout_values/outputs/manuscript_values.json"
SOURCES = REPO_ROOT / "paper/sources.json"
R = REPO_ROOT / "review"
ARCH = {"dn": "densenet121", "rn": "resnet50"}
INSTR = {"dn": "phaseA-densenet", "rn": "phaseA-resnet50"}
CONTRASTS = {"primary": "primary", "rt": "tube_rendered", "tt": "tube_tissue", "rd": "disk_rendered", "dt": "disk_tissue",
             "tshape": "tissue_shape", "rshape": "rendered_shape", "cdisks": "content_on_disks", "inter": "interaction"}

def _j(p: Path):
    return json.loads(p.read_text())

def _interval(block: dict, level: int):
    key = f"ci{level}"
    return block[key] if key in block else (block["ci"] if level == 95 and "ci" in block else None)

def build() -> tuple[dict, dict]:
    v: dict = {}
    meta: dict = {}

    def put(id_, val, decimals=3, kind="f"):
        v[id_] = float(val)
        meta[id_] = (decimals, kind)

    gate = {a: _j(R / f"heldout/confirmatory/analysis_{n}.json") for a, n in ARCH.items()}
    expl = _j(R / "positive_audit/positive_analysis.json")
    neg = _j(R / "originals_twobytwo/twobytwo/twobytwo_analysis.json")

    def block_stats(prefix, blk, with90=True):
        put(f"{prefix}-m", blk["mean"])
        if "ci90" in blk and with90:
            put(f"{prefix}-l90", blk["ci90"][0]); put(f"{prefix}-h90", blk["ci90"][1])
        ci95 = blk.get("ci95") or blk.get("ci")
        if ci95:
            put(f"{prefix}-l95", ci95[0]); put(f"{prefix}-h95", ci95[1])

    for a in ARCH:

        g = gate[a]["originals"]["pooled"]
        d = gate[a]["dev_same_design_comparator"]["originals"]["pooled"]
        for pop, pooled in (("gate", g), ("dev", d)):
            for short, long in CONTRASTS.items():
                if long in pooled:
                    block_stats(f"q-{pop}-{a}-{short}", pooled[long])

        e = expl[ARCH[a]]["pooled"]
        for short, long in (("primary", "primary"), ("rt", "tube_rendered"), ("tt", "tube_tissue"), ("rd", "disk_rendered"), ("dt", "disk_tissue"),
                            ("cdisks", "content_on_disks"), ("inter", "interaction")):
            block_stats(f"q-expl-{a}-{short}", e[long])
        put(f"q-expl-{a}-rshape-m", e["tube_rendered"]["mean"] - e["disk_rendered"]["mean"])
        put(f"q-expl-{a}-tshape-m", e["tube_tissue"]["mean"] - e["disk_tissue"]["mean"])
        for src in ("dev_half", "R_int"):
            block_stats(f"q-expl-{a}-{src}-primary", expl[ARCH[a]]["by_source"][src]["primary"])
        put(f"q-expl-{a}-n-targets", e["_n_targets"], 0, "i"); put(f"q-expl-{a}-n-patients", e["_n_patients"], 0, "i")

        n = neg[ARCH[a]]
        cells = n["cells"]
        for short, long in (("rt", "tube_rendered"), ("tt", "tube_tissue"), ("rd", "disk_rendered"), ("dt", "disk_tissue")):
            block_stats(f"q-neg-{a}-{short}", {"mean": cells[long]["mean"], "ci95": cells[long]["ci"]})
        dec = n["decomposition"]
        for short, long in (("primary", "content_effect_on_tubes"), ("cdisks", "content_effect_on_disks"), ("inter", "interaction")):
            block_stats(f"q-neg-{a}-{short}", {"mean": dec[long]["mean"], "ci95": dec[long]["ci"]})
        put(f"q-neg-{a}-rshape-m", cells["tube_rendered"]["mean"] - cells["disk_rendered"]["mean"])
        put(f"q-neg-{a}-tshape-m", cells["tube_tissue"]["mean"] - cells["disk_tissue"]["mean"])
        put(f"q-neg-{a}-n-images", n["n_images"], 0, "i"); put(f"q-neg-{a}-n-patients", n["n_patients"], 0, "i")

        ga = gate[a]
        put(f"q-margin-{a}", ga["originals"]["verdict"]["margin"], 4 if a == "rn" else 3)
        put(f"q-invert-{a}", ga["originals"]["verdict"]["invert_constant"], 4 if a == "rn" else 2)
        put(f"q-gate-{a}-margin-sens", ga["margin_sensitivity_recomputed_on_gate_positives_baselines"]["_mean_tighter"], 3)
        put(f"q-gate-{a}-n-targets", g["_n_targets"], 0, "i"); put(f"q-gate-{a}-n-patients", g["_n_patients"], 0, "i")
        put(f"q-dev-{a}-n-targets", d["_n_targets"], 0, "i"); put(f"q-dev-{a}-n-patients", d["_n_patients"], 0, "i")

        rl = ga["originals"]["recipe_level_secondary"]
        put(f"q-gate-{a}-recipe-l95", rl["t95"][0]); put(f"q-gate-{a}-recipe-h95", rl["t95"][1])
        put(f"q-gate-{a}-recipe-var", rl["variance_across_checkpoints"], 4)
        put(f"q-gate-{a}-recipe-sd", rl["variance_across_checkpoints"] ** 0.5, 3)
        rld = ga["dev_same_design_comparator"]["originals"]["recipe_level_secondary"]
        put(f"q-dev-{a}-recipe-l95", rld["t95"][0]); put(f"q-dev-{a}-recipe-h95", rld["t95"][1])

        for i, lab in enumerate(ga["originals"]["labels"]):
            cp = ga["originals"]["per_checkpoint_descriptive"][lab]["primary"]
            put(f"q-gate-{a}-ck{i}-primary-m", cp["mean"])
            put(f"q-gate-{a}-ck{i}-primary-l90", cp["ci90"][0]); put(f"q-gate-{a}-ck{i}-primary-h90", cp["ci90"][1])
        pcm = [ga["originals"]["per_checkpoint_descriptive"][lab]["primary"]["mean"] for lab in ga["originals"]["labels"]]
        put(f"q-gate-{a}-ck-min", min(pcm)); put(f"q-gate-{a}-ck-max", max(pcm))
        put(f"q-gate-{a}-ck-n-neg", sum(1 for x in pcm if x < 0), 0, "i")
        put(f"q-gate-{a}-ck-n-inside", sum(1 for x in pcm if abs(x) < ga["originals"]["verdict"]["margin"]), 0, "i")

        for tag, key in (("de", "drain_labelled_elsewhere"), ("nde", "not_drain_labelled_elsewhere"), ("ap", "view_AP"), ("pa", "view_PA")):
            sb = ga["strata"][key]
            block_stats(f"q-gate-{a}-{tag}-primary", sb["primary"])
            put(f"q-gate-{a}-{tag}-n-targets", sb["_n_targets"], 0, "i"); put(f"q-gate-{a}-{tag}-n-patients", sb["_n_patients"], 0, "i")
            if a == "dn":
                pass

        gd = ga["dev_same_design_comparator"]["gate_minus_dev_same_design_primary"]
        put(f"q-gd-{a}-m", gd["gate_minus_other"]); put(f"q-gd-{a}-l95", gd["ci95"][0]); put(f"q-gd-{a}-h95", gd["ci95"][1])

        for src in ("dev_half", "R_int"):
            b = ga["beside_positive_audit_exploratory"][src]
            put(f"q-gx-{a}-{src}-est", b["positive_audit_estimate"])
            put(f"q-gx-{a}-{src}-diff", b["gate_minus_other"]); put(f"q-gx-{a}-{src}-l95", b["ci95"][0]); put(f"q-gx-{a}-{src}-h95", b["ci95"][1])

        vc = ga["originals"]["variance_components"]
        put(f"q-gate-{a}-sd-patient", vc["sigma_a2"] ** 0.5, 3); put(f"q-gate-{a}-sd-within", vc["sigma_w2"] ** 0.5, 3)

        for view in ("AP", "PA"):
            st = ga["donor_stats"][view]
            put(f"q-gate-donor-{view.lower()}-pool", st["pool_patients"], 0, "c"); put(f"q-gate-donor-{view.lower()}-slots", st["slots"], 0, "c")

        for k, blk in ga["planted"]["strengths"].items():
            block_stats(f"q-gate-{a}-pl-{k}", blk["primary"])

        if a == "dn":
            ro = _j(R / "planted_intermediate/outputs/readout.json")["points"]
        else:
            ro = _j(R / "heldout/part4/resnet50_readout.json")["points"]
        for k, pt in ro.items():
            ab = pt["audit"]
            put(f"q-neg-{a}-pl-{k}-m", ab["gap"]); put(f"q-neg-{a}-pl-{k}-l95", ab["ci"][0]); put(f"q-neg-{a}-pl-{k}-h95", ab["ci"][1])
            put(f"q-neg-{a}-pl-{k}-manip", pt["manip_effect"].get("mean_of_per_seed", pt["manip_effect"].get("mean", float("nan"))), 3)
    ro_dn = _j(R / "planted_intermediate/outputs/readout.json")["points"]
    for k, pt in ro_dn.items():
        if k != "original":
            put(f"q-lor-{k}", pt["log_odds_ratio"], 2)

    for pop in ("gate", "dev", "expl", "neg"):
        for a in ARCH:
            put(f"q-{pop}-{a}-rdt-m", v[f"q-{pop}-{a}-rt-m"] - v[f"q-{pop}-{a}-dt-m"])
    geo = [v[f"q-{p}-{a}-rshape-m"] for p in ("neg", "expl", "gate") for a in ARCH]
    put("q-geo-min", min(geo)); put("q-geo-max", max(geo))
    tg = [v[f"q-{p}-{a}-tshape-m"] for p in ("neg", "expl", "gate") for a in ARCH]
    put("q-tgeo-min", min(tg), 3); put("q-tgeo-max", max(tg), 3)

    put("q-gate-donor-images", _j(R / "heldout/confirmatory/populations/index.json")["gate_donor_images"]["n"], 0, "c")
    put("q-gate-donor-patients", _j(R / "heldout/confirmatory/populations/index.json")["gate_donor_patients"]["n"], 0, "c")
    pidx = _j(R / "positive_audit/populations/index.json")
    put("q-expl-dev-targets", pidx["dev_half_targets_images"]["n"], 0, "i"); put("q-expl-dev-patients", pidx["dev_half_targets_patients"]["n"], 0, "i")
    put("q-expl-rint-targets", pidx["R_int_C_targets_images"]["n"] + pidx["R_int_Q_targets_images"]["n"], 0, "i")
    put("q-expl-rint-patients", pidx["R_int_C_targets_patients"]["n"] + pidx["R_int_Q_targets_patients"]["n"], 0, "i")
    cnt = _j(R / "feasibility/counts.json")
    put("q-neatx-test-rows", cnt["neatx_by_split"]["test"], 0, "c")
    put("q-gate-eligible-present-images", cnt["halves"]["gate"]["s2_after_rca_filter"]["drain_present"]["images"], 0, "i")
    prep = _j(R / "heldout/confirmatory/prepare_record.json")
    put("q-gate-n-drain-elsewhere", prep["n_drain_labelled_elsewhere"], 0, "i")
    put("q-zero", 0.0, 1)
    return v, meta

def fmt(id_: str, value: float, meta: dict) -> str:
    dec, kind = meta[id_]
    if id_.endswith("-d4"):
        dec = 4
    plus = id_.endswith("-s")
    if kind == "i":
        return f"{int(round(value))}"
    if kind == "c":
        return f"{int(round(value)):,}"
    s = f"{value:+.{dec}f}" if plus else f"{value:.{dec}f}"
    return s

def lookup(vals: dict, id_: str):
    for suf in ("-s", "-d4"):
        if id_.endswith(suf) and id_[: -len(suf)] in vals:
            return id_[: -len(suf)]
    return id_

def used_ids() -> set[str]:
    ids: set[str] = set()
    for sub in ("sections", "supp", "tables"):
        for f in (REPO_ROOT / "paper" / sub).glob("*.tex"):
            ids |= set(re.findall(r"\\val\{(q-[^}]+)\}", f.read_text()))
    return ids

COMPARISONS = [
    {"id": "cmp-q-gate-dn-ck1-above-margin", "instrument": "phaseA-densenet", "note": "DenseNet-121 checkpoint 1's gate contrast exceeds the margin.", "left": "q-gate-dn-ck1-primary-m", "right": "q-margin-dn", "relation": "gt"},
    {"id": "cmp-q-neg-dn-primary-inside-margin", "instrument": "phaseA-densenet", "note": "The upper end of DenseNet-121's 95% interval on the 289 negative targets is below the margin.", "left": "q-neg-dn-primary-h95", "right": "q-margin-dn", "relation": "lt"},
]
for _a, _n in (("dn", "DenseNet-121"), ("rn", "ResNet-50")):
    COMPARISONS += [
        {"id": f"cmp-q-strong-above-invert-{_a}", "instrument": INSTR[_a], "note": f"{_n}: the strong planted checkpoints' pooled primary contrast on the gate targets is above the INVERT constant.", "left": f"q-gate-{_a}-pl-strong-m", "right": f"q-invert-{_a}", "relation": "gt"},
        {"id": f"cmp-q-mid30-above-margin-{_a}", "instrument": INSTR[_a], "note": f"{_n}: the first intermediate setting's gate contrast is above the margin.", "left": f"q-gate-{_a}-pl-mid30-m", "right": f"q-margin-{_a}", "relation": "gt"},
        {"id": f"cmp-q-rshape-above-margin-{_a}", "instrument": INSTR[_a], "note": f"{_n}: the rendered tube minus rendered disk on the gate is above the margin.", "left": f"q-gate-{_a}-rshape-m", "right": f"q-margin-{_a}", "relation": "gt"},
    ]

for _a, _n in (("dn", "DenseNet-121"), ("rn", "ResNet-50")):
    COMPARISONS.append({"id": f"cmp-q-{_a}-90-below-zero", "instrument": INSTR[_a], "note": f"{_n}: the pooled 90% interval of the primary contrast on the gate targets lies below zero (upper end below zero).", "left": f"q-gate-{_a}-primary-h90", "right": "q-zero", "relation": "lt"})

def note_for(id_: str) -> str:
    return f"Value {id_}, read at full precision from its recorded output by paper_values/heldout_values.py."

def write_sources(vals: dict, meta: dict, ids: set[str]) -> int:
    entries = json.loads(SOURCES.read_text())
    by_id = {e["id"]: i for i, e in enumerate(entries)}
    n = 0
    for id_ in sorted(ids):
        base = lookup(vals, id_)
        if base not in vals:
            print("  UNRESOLVED:", id_)
            continue
        arch = re.search(r"-(dn|rn)-", id_ if id_.startswith("q-") else "")
        inst = INSTR[arch.group(1)] if arch else "phaseA-both"
        tol = 0.5 * 10 ** (-meta[base][0]) + 1e-9
        entry = {"id": id_, "value": fmt(id_, vals[base], {id_: meta[base]}),
                 "instrument": inst, "kind": "verified", "script": "paper_values/heldout_values.py",
                 "artifact": "review/heldout_values/outputs/manuscript_values.json", "path": base, "tolerance": tol, "note": note_for(id_)}
        if id_ in by_id:
            entries[by_id[id_]] = entry
        else:
            entries.append(entry)
            by_id[id_] = len(entries) - 1
        n += 1
    for c in COMPARISONS:
        entry = {"kind": "comparison", **c}
        if c["id"] in by_id:
            entries[by_id[c["id"]]] = entry
        else:
            entries.append(entry)
            by_id[c["id"]] = len(entries) - 1
    SOURCES.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return n

def main(argv: list) -> int:
    vals, meta = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(vals, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(vals)} keys)")
    if "--write-sources" in argv:
        ids = set(vals) if "--all" in argv else used_ids() | {x for c in COMPARISONS for x in (c["left"], c["right"]) if x in vals}
        n = write_sources(vals, meta, ids)
        print(f"upserted {n} verified sources into paper/sources.json")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
