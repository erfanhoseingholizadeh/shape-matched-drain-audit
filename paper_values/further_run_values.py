# Writes the values the paper cites for the near-margin plantings, the further seeds and the 512-pixel replication.
# Run: python3 paper_values/further_run_values.py [--write-sources]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "review"
OUT = R / "replication_512_readout/manuscript_values.json"
SOURCES = ROOT / "paper/sources.json"
ARCHS = ("densenet121", "resnet50")
AK = {"densenet121": "dn", "resnet50": "rn"}
INSTR = {"densenet121": "phaseA-densenet", "resnet50": "phaseA-resnet50", "both": "phaseA-both"}
RUNGS = ("zero", "weak", "e3lo", "e3hi", "mid30", "mid50", "strong")
E2_FAMS = {"original": "orig", "e4": "e4", "planted_strong224": "pstrong", "planted_zero224": "pzero"}

VALS: dict = {}
CMPS: dict = {}

def put(i: str, x, dec: int, note: str, instrument: str = "phaseA-both") -> None:
    if i in VALS:
        raise ValueError(f"duplicate id {i}")
    VALS[i] = (x if isinstance(x, str) else float(x), dec, note, instrument)

def cmp_lt(i: str, left: str, right: str, note: str, instrument: str) -> None:
    CMPS[i] = {"kind": "comparison", "id": i, "instrument": instrument, "note": note, "left": left, "right": right, "relation": "lt"}

def js(p: str):
    return json.loads((R / p).read_text())

def triple(prefix: str, point: float, ci: list, dec: int, note: str, instrument: str = "phaseA-both") -> None:
    put(f"{prefix}-m", point, dec, note, instrument)
    put(f"{prefix}-l95", ci[0], dec, note + " (lower end of the 95% interval)", instrument)
    put(f"{prefix}-h95", ci[1], dec, note + " (upper end of the 95% interval)", instrument)

def build() -> None:
    VALS.clear()
    CMPS.clear()
    a = js("replication_512/e2/audit_fidelity.json")
    ins = INSTR["both"]
    put("p83-n-targets", a["targets"], 0, "The exploratory audit targets of the generative comparison (and of the planting ladder and the further seeds).", ins)
    put("p83-n-patients", a["patients"], 0, "Patients of those targets.", ins)

    for arch in ARCHS:
        d = js(f"further_readout/e3/e3_{arch}.json")
        k = AK[arch]
        sets = d["e3"]["settings"]
        delta = {r["setting"]: r for r in d["ladder"]}
        for rung in RUNGS:
            if rung not in sets:
                continue
            r = sets[rung]["raw"]
            put(f"p83-e3-{k}-{rung}-delta", delta[rung]["realized_delta"], 3, f"Realized planting delta of the {rung} rung ({arch}).", INSTR[arch])
            put(f"p83-e3-{k}-{rung}-raw", r["mean"], 3, f"Pooled primary contrast of the {rung} rung ({arch}, raw).", INSTR[arch])
            put(f"p83-e3-{k}-{rung}-verdict", r["verdict"] + (f", {r['sign']}" if r["verdict"] == "EFFECT" else ""), 0, f"Registered verdict of the {rung} rung ({arch}), with the sign of an EFFECT.", INSTR[arch])
            put(f"p83-e3-{k}-{rung}-raw-l90", r["ci90"][0], 3, f"Lower end of the 90% interval, {rung} ({arch}).", INSTR[arch])
            put(f"p83-e3-{k}-{rung}-raw-h90", r["ci90"][1], 3, f"Upper end of the 90% interval, {rung} ({arch}).", INSTR[arch])
            if rung != "zero":
                pz = sets[rung]["planted_minus_zero"]
                triple(f"p83-e3-{k}-{rung}-mz", pz["mean"], pz["ci95"], 3, f"Planted minus zero, {rung} ({arch}).", INSTR[arch])

    for arch in ARCHS:
        e4 = js(f"further_readout/e4/e4_{arch}.json")["e4"]["valid_only"]
        k = AK[arch]
        fp = e4["family_pooled"]
        for fam, key in (("orig", "originals"), ("e4", "e4")):
            put(f"p83-e4-{k}-{fam}-m", fp[key]["mean"], 3, f"Family-pooled primary contrast, {arch}, {key} (134 targets).", INSTR[arch])
            put(f"p83-e4-{k}-{fam}-l90", fp[key]["ci90"][0], 3, f"Lower end of the 90% interval, {arch} {key}.", INSTR[arch])
            put(f"p83-e4-{k}-{fam}-h90", fp[key]["ci90"][1], 3, f"Upper end of the 90% interval, {arch} {key}.", INSTR[arch])
        triple(f"p83-e4-{k}-famdiff", e4["family_difference_e4_minus_originals"]["mean"], e4["family_difference_e4_minus_originals"]["ci95"], 3, f"E4 family minus originals, {arch}.", INSTR[arch])
        cp = e4["combined_pool"]
        put(f"p83-e4-{k}-comb-m", cp["mean"], 3, f"Pooled primary contrast over the fifteen checkpoints, {arch}.", INSTR[arch])
        put(f"p83-e4-{k}-comb-l90", cp["ci90"][0], 3, f"Lower end of the 90% interval over the fifteen checkpoints, {arch}.", INSTR[arch])
        put(f"p83-e4-{k}-comb-h90", cp["ci90"][1], 3, f"Upper end of the 90% interval over the fifteen checkpoints, {arch}.", INSTR[arch])
        rl = e4["recipe_level"]
        put(f"p83-e4-{k}-recipe-l90", rl["t90"][0], 3, f"Recipe-level t interval (90%), lower end, {arch}.", INSTR[arch])
        put(f"p83-e4-{k}-recipe-h90", rl["t90"][1], 3, f"Recipe-level t interval (90%), upper end, {arch}.", INSTR[arch])
        put(f"p83-e4-{k}-neg", e4["counts"]["negative_point_estimates"], 0, f"Checkpoints of the fifteen with a negative point estimate, {arch}.", INSTR[arch])
        put(f"p83-e4-{k}-above", e4["counts"]["above_margin"], 0, f"Checkpoints of the fifteen above the margin, {arch}.", INSTR[arch])
    build_e5()

def build_e5() -> None:
    ins = INSTR["both"]
    vp = R / "replication_512/e5/validity/validity.json"
    if not vp.exists():
        return
    v = json.loads(vp.read_text())
    ck = v["checkpoints"]
    for arch in ARCHS:
        k = AK[arch]
        for fam, tag in (("orig", "orig"), ("zero", "zero"), ("strong", "strong")):
            rows = [c for name, c in ck.items() if name.startswith(f"{arch}/{arch}_{fam}_seed")]
            put(f"p83-e5v-{k}-{tag}-n", len(rows), 0, f"E5 checkpoints scored for validity at 512 pixels, {arch} {fam}.", INSTR[arch])
            put(f"p83-e5v-{k}-{tag}-nvalid", sum(bool(c["valid"]) for c in rows), 0, f"E5 checkpoints that clear both validity clauses at 512 pixels on R_str, {arch} {fam}.", INSTR[arch])
        rows = [c for name, c in ck.items() if name.startswith(f"{arch}/")]
        put(f"p83-e5v-{k}-auc-min", min(c["auc"] for c in rows), 3, f"Lowest clause (a) AUC over the {arch} E5 checkpoints at 512 pixels.", INSTR[arch])
        put(f"p83-e5v-{k}-auc-max", max(c["auc"] for c in rows), 3, f"Highest clause (a) AUC over the {arch} E5 checkpoints at 512 pixels.", INSTR[arch])
        put(f"p83-e5v-{k}-medp-min", min(c["median_neg_p"] for c in rows), 3, f"Lowest clause (b) median negative probability over the {arch} E5 checkpoints at 512 pixels.", INSTR[arch])
    assert all(c["valid"] for c in ck.values()), "the prose says every E5 checkpoint is valid; a failing checkpoint needs its own prose"
    sz = {json.loads((R / f"replication_512/e5/audit/size_{a}.json").read_text())["image_size"] for a in ARCHS}
    assert sz == {512}, sz
    put("p83-e5-size", 512, 0, "Input size of the E5 classifiers in pixels (the scorer's size file for both architectures).", ins)
    put("p83-e5-n-train", len(ck), 0, "Classifiers trained at 512 pixels for E5 (ten originals, six zero-strength and six strong-strength planted).", ins)
    cfg = json.loads((ROOT / "configs/e5_microbatch.json").read_text())
    assert cfg["densenet121"] == cfg["resnet50"], "one micro-batch for both architectures"
    put("p83-e5-micro", cfg["densenet121"], 0, "Micro-batch of the 512-pixel trainings (the registered memory rule, one value for both architectures).", ins)
    put("p83-e5v-n-patients", v["report_half"]["patients"], 0, "Patients of the report half R_str on which E5 validity is evaluated.", ins)
    put("p83-e5v-n-images", v["report_half"]["images"], 0, "Images of the report half R_str.", ins)
    cp = R / "replication_512_readout/e5ctl_validity224/validity.json"
    if cp.exists():
        cv = json.loads(cp.read_text())["checkpoints"]
        assert all(c["valid"] for c in cv.values()), "the prose says every control checkpoint is valid"
        put("p83-e5-n-ctl", len(cv), 0, "224-pixel micro-batch control classifiers trained (seeds 5 to 8, both architectures).", ins)
        for arch in ARCHS:
            rows = [c for name, c in cv.items() if name.startswith(f"{arch}/")]
            put(f"p83-e5v-{AK[arch]}-ctl-n", len(rows), 0, f"224-pixel micro-batch control checkpoints scored for validity, {arch}.", INSTR[arch])
            put(f"p83-e5v-{AK[arch]}-ctl-nvalid", sum(bool(c["valid"]) for c in rows), 0, f"Control checkpoints that clear both validity clauses on R_str, {arch}.", INSTR[arch])
    mp = R / "replication_512_readout/e5/margins.json"
    if mp.exists():
        m = json.loads(mp.read_text())["archs"]
        for arch in ARCHS:
            r = m[arch]
            put(f"p83-e5-{AK[arch]}-margin512", r["margin_512"], 3, f"512-pixel margin (uniform logit shift for a mean probability change of 0.02), {arch}, pooled over the valid E5 originals.", INSTR[arch])
            put(f"p83-e5-{AK[arch]}-invert512", r["invert_512"], 3, f"512-pixel INVERT constant (same shift at 0.10), {arch}.", INSTR[arch])
            put(f"p83-e5-{AK[arch]}-margin-n", r["pooled"]["n_negatives"], 0, f"Pooled baseline negative logits behind the 512 margin, {arch}.", INSTR[arch])
    build_e5_readout()
    assert_prose_flags()
    build_e5_manip()

def build_e5_readout() -> None:
    rp = R / "replication_512_readout/e5/readout.json"
    if not rp.exists():
        return
    rd = json.loads(rp.read_text())

    for a_ in ARCHS:
        assert rd[a_]["sets"]["orig_valid"]["pooled"]["invert_met_512"] is False, "prose: the INVERT rule is met in neither architecture"
        assert rd[a_]["sets"]["strong_valid"]["planted_minus_zero"]["relation_to_margin"] == "wholly_above_margin", "prose: strong minus zero lies wholly above the 512 margin in both"
        assert rd[a_]["sets"]["orig_valid"]["pooled"]["effect_95_excludes_zero"] and rd[a_]["sets"]["orig_valid"]["pooled"]["ci90"][1] < 0, "prose: both 90% intervals lie below zero"
    assert rd["resnet50"]["sets"]["orig_valid"]["pooled"]["verdict_512_margin"] == rd["resnet50"]["sets"]["orig_valid"]["pooled"]["verdict_224_margin"] == "EFFECT", "prose: ResNet-50 reads EFFECT under both margins"
    assert rd["densenet121"]["sets"]["orig_valid"]["pooled"]["verdict_512_margin"] == "EFFECT" and rd["densenet121"]["sets"]["orig_valid"]["pooled"]["verdict_224_margin"] == "EQUIVALENT", "prose: DenseNet-121 reads EFFECT at 512 and EQUIVALENT at 224"
    from analysis import positive_analysis as _P71
    assert _P71.verdict(rd["densenet121"]["sets"]["orig_valid"]["pooled"], _P71.MARGIN["densenet121"], _P71.INVERT["densenet121"])["effect_inside_margin_flag"] is True, "prose: DenseNet-121's originals read EQUIVALENT with the flag under the 224 margin"
    assert rd["densenet121"]["control_224"]["attribution_to_resolution_allowed"] is False and rd["resnet50"]["control_224"]["attribution_to_resolution_allowed"] is True, "prose: the registered rule allows an attribution to resolution in ResNet-50 only"
    assert all(rd[a_]["control_224"]["n_seeds"] == 4 for a_ in ARCHS), "prose: four control seeds"
    for arch in ARCHS:
        k, x, ins = AK[arch], rd[arch], INSTR[arch]
        for fam in ("orig", "zero", "strong"):
            s = x["sets"][f"{fam}_valid"]
            pl = s["pooled"]
            nm = {"orig": "original", "zero": "zero-strength", "strong": "strong-strength"}[fam]
            put(f"p83-e5-{k}-{fam}-m", pl["mean"], 3, f"512 pixels, {arch}, {nm} checkpoints: pooled primary contrast (valid checkpoints).", ins)
            for lvl, key in (("90", "ci90"), ("95", "ci95")):
                put(f"p83-e5-{k}-{fam}-l{lvl}", pl[key][0], 3, f"Lower end of the {lvl}% interval, 512 pixels, {arch} {nm}.", ins)
                put(f"p83-e5-{k}-{fam}-h{lvl}", pl[key][1], 3, f"Upper end of the {lvl}% interval, 512 pixels, {arch} {nm}.", ins)
            put(f"p83-e5-{k}-{fam}-v512", pl["verdict_512_margin"], 0, f"Registered verdict under the 512 margin, {arch} {nm}.", ins)
            put(f"p83-e5-{k}-{fam}-v224", pl["verdict_224_margin"], 0, f"Registered verdict under the registered 224 margin (sensitivity), {arch} {nm}.", ins)

            from analysis import positive_analysis as P71
            put(f"p83-e5-{k}-{fam}-f512", "yes" if P71.verdict(pl, pl["margin_512"], P71.INVERT[arch])["effect_inside_margin_flag"] else "no", 0, f"Whether the verdict under the 512 margin carries the flag 'effect inside the margin', {arch} {nm}.", ins)
            put(f"p83-e5-{k}-{fam}-f224", "yes" if P71.verdict(pl, P71.MARGIN[arch], P71.INVERT[arch])["effect_inside_margin_flag"] else "no", 0, f"Whether the verdict under the registered 224 margin carries the flag 'effect inside the margin', {arch} {nm}.", ins)
            put(f"p83-e5-{k}-{fam}-sign", pl["sign"], 0, f"Sign of the pooled contrast, 512 pixels, {arch} {nm}.", ins)
            put(f"p83-e5-{k}-{fam}-differ", "yes" if pl["verdicts_differ_between_margins"] else "no", 0, f"Whether the verdict differs between the 512 and the 224 margin, {arch} {nm}.", ins)
        cmp_lt(f"cmp-p83-e5-{k}-orig-90-below-zero", f"p83-e5-{k}-orig-h90", "q-zero", f"{arch}: the pooled 90% interval of the 512-pixel originals' primary contrast lies below zero (upper end below zero).", ins)
        o = x["sets"]["orig_valid"]
        put(f"p83-e5-{k}-orig-neg", o["checkpoints_negative"], 0, f"Original checkpoints (of five) with a negative primary contrast at 512 pixels, {arch}.", ins)
        put(f"p83-e5-{k}-orig-above512", o["checkpoints_above_512_margin"], 0, f"Original checkpoints above the 512 margin, {arch}.", ins)
        put(f"p83-e5-{k}-orig-above224", o["checkpoints_above_224_margin"], 0, f"Original checkpoints above the 224 margin, {arch}.", ins)
        put(f"p83-e5-{k}-orig-invert", "yes" if o["pooled"]["invert_met_512"] else "no", 0, f"Whether the INVERT rule is met at 512 pixels, {arch}.", ins)
        for s_, v in o["per_checkpoint_primary"].items():
            put(f"p83-e5-{k}-orig-s{s_.rsplit('seed', 1)[1]}", v, 3, f"Primary contrast of the 512-pixel original checkpoint of seed {s_.rsplit('seed', 1)[1]}, {arch}.", ins)
        put(f"p83-e5-{k}-orig-min", min(o["per_checkpoint_primary"].values()), 3, f"Lowest primary contrast among the five 512-pixel original checkpoints, {arch}.", ins)
        put(f"p83-e5-{k}-orig-max", max(o["per_checkpoint_primary"].values()), 3, f"Highest primary contrast among the five 512-pixel original checkpoints, {arch}.", ins)
        mz = x["sets"]["strong_valid"]["planted_minus_zero"]
        triple(f"p83-e5-{k}-mz", mz["mean"], mz["ci95"], 3, f"Strong minus zero planting at 512 pixels, {arch}.", ins)
        put(f"p83-e5-{k}-mz-rel", mz["relation_to_margin"].replace("_", " "), 0, f"Relation of the strong-minus-zero interval to the 512 margin, {arch}.", ins)
        rl = x["recipe_level_originals"]["at_512_margin"]
        put(f"p83-e5-{k}-rec-m", rl["mean"], 3, f"Recipe-level mean over the five 512-pixel originals, {arch}.", ins)
        put(f"p83-e5-{k}-rec-sd", rl["sd"], 3, f"Between-checkpoint SD of the five 512-pixel originals, {arch}.", ins)
        for lvl, key in (("90", "ci90"), ("95", "ci95")):
            put(f"p83-e5-{k}-rec-l{lvl}", rl[key][0], 3, f"Recipe-level t interval ({lvl}%, 4 degrees of freedom), lower end, {arch}.", ins)
            put(f"p83-e5-{k}-rec-h{lvl}", rl[key][1], 3, f"Recipe-level t interval ({lvl}%), upper end, {arch}.", ins)
        put(f"p83-e5-{k}-rec-inside", "yes" if rl["ci90_inside_margin"] else "no", 0, f"Whether the recipe-level 90% interval is inside the 512 margin, {arch}.", ins)
        for key, tag, nm in (("content_main", "content", "content main effect"), ("shape_main", "shape", "shape main effect"), ("interaction", "inter", "interaction")):
            triple(f"p83-e5-{k}-tbt-{tag}", x["two_by_two_originals"][key]["mean"], x["two_by_two_originals"][key]["ci95"], 3, f"Two-by-two {nm} (shape by content) on the 512-pixel originals, {arch}.", ins)
        c = x["compare_224"]
        for s_, v in c["matched_seed_differences_512_minus_224_e4"].items():
            put(f"p83-e5-{k}-md-s{s_}", v, 3, f"512-pixel original minus the 224 further-seed checkpoint of seed {s_}, {arch}.", ins)
        t_ = c["matched_difference_t_interval"]
        put(f"p83-e5-{k}-md-m", t_["mean"], 3, f"Mean matched-seed difference (512 minus 224), {arch}.", ins)
        for lvl, key in (("90", "ci90"), ("95", "ci95")):
            put(f"p83-e5-{k}-md-l{lvl}", t_[key][0], 3, f"Matched-seed difference, t interval ({lvl}%, 4 degrees of freedom), lower end, {arch}.", ins)
            put(f"p83-e5-{k}-md-h{lvl}", t_[key][1], 3, f"Matched-seed difference, t interval ({lvl}%), upper end, {arch}.", ins)
        put(f"p83-e5-{k}-md-det", c["detectable_difference_95"], 3, f"Size of a difference between two five-checkpoint means that cannot be told from seed noise, {arch}.", ins)
        put(f"p83-e5-{k}-e4m", c["e4_matched_mean_224"], 3, f"Mean primary of the 224 further seeds 5 to 9, {arch}.", ins)
        put(f"p83-e5-{k}-fam-minus-e4", c["minus_e4_ten_mean"], 3, f"512 original family mean minus the mean of the ten further seeds at 224, {arch}.", ins)
        put(f"p83-e5-{k}-fam-minus-gate", c["minus_five_224_originals_selected_on_S_int_mean"], 3, f"512 original family mean minus the mean of the five gate-family originals at 224 (different selection half), {arch}.", ins)
        put(f"p83-e5-{k}-pz-d", c["planted_zero_512_minus_224"], 3, f"Zero-strength planted family at 512 minus 224, {arch}.", ins)
        put(f"p83-e5-{k}-ps-d", c["planted_strong_512_minus_224"], 3, f"Strong-strength planted family at 512 minus 224, {arch}.", ins)
        ct = x["control_224"]
        for s_ in ct["seeds"]:
            r_ = ct["per_seed"][str(s_)]
            put(f"p83-e5-{k}-ctl-e4-s{s_}", r_["e4_224_batch128"], 3, f"Primary contrast of the 224 further-seed checkpoint (batch 128 in one piece) of seed {s_}, {arch}.", ins)
            put(f"p83-e5-{k}-ctl-e5-s{s_}", r_["e5_512_micro32"], 3, f"Primary contrast of the 512-pixel original of seed {s_} (micro-batch 32), {arch}.", ins)
            put(f"p83-e5-{k}-ctl-tot-s{s_}", r_["total_512_minus_224"], 3, f"Total 512-minus-224 difference of seed {s_} (512 original minus 224 further seed), {arch}.", ins)
            put(f"p83-e5-{k}-ctl-mb-s{s_}", r_["micro_batching_effect"], 3, f"Micro-batching effect (224 control minus 224 further seed), seed {s_}, {arch}.", ins)
            put(f"p83-e5-{k}-ctl-rs-s{s_}", r_["resolution_effect_at_fixed_micro_batch"], 3, f"Resolution effect at fixed micro-batch (512 original minus 224 control), seed {s_}, {arch}.", ins)
        for key, tag, nm in (("micro_batching_effect", "mb", "micro-batching effect"), ("resolution_effect_at_fixed_micro_batch", "rs", "resolution effect"), ("total_512_minus_224", "tot", "total 512-minus-224 difference")):
            put(f"p83-e5-{k}-ctl-{tag}-m", ct["mean_over_seeds"][key], 3, f"Mean over the control's seeds of the {nm}, {arch}.", ins)
        put(f"p83-e5-{k}-ctl-limit", ct["seed_noise_limit"], 3, f"Seed-noise limit of the control for {ct['n_seeds']} seeds, {arch}.", ins)
        put(f"p83-e5-{k}-ctl-n", ct["n_seeds"], 0, f"Seeds in the control comparison, {arch}.", ins)
        put(f"p83-e5-{k}-ctl-allowed", "yes" if ct["attribution_to_resolution_allowed"] else "no", 0, f"Whether the registered sentence rule allows attributing the difference to resolution, {arch}.", ins)
        for lab, v in ct["per_checkpoint_primary"].items():
            put(f"p83-e5-{k}-ctl-prim-s{lab.rsplit('seed', 1)[1]}", v, 3, f"Primary contrast of the 224 micro-batch control checkpoint of seed {lab.rsplit('seed', 1)[1]}, {arch}.", ins)

def assert_prose_flags() -> None:
    from analysis import positive_analysis as P71

    def flag(arch: str, ci90: list, ci95: list, margin: float, mean: float) -> bool:
        return bool(P71.verdict({"ci90": ci90, "ci95": ci95, "mean": mean}, margin, P71.INVERT[arch])["effect_inside_margin_flag"])
    want = {("densenet121", "e3lo"): True, ("densenet121", "weak"): False, ("resnet50", "zero"): True, ("resnet50", "weak"): False}
    for arch in ARCHS:
        e3 = js(f"further_readout/e3/e3_{arch}.json")["e3"]
        for (a, s), w in want.items():
            if a == arch:
                r = e3["settings"][s]["raw"]
                assert flag(arch, r["ci90"], r["ci95"], e3["margin"], r["mean"]) is w, f"prose: the flag of {arch} {s} in S13 is {w}"
    e4 = js("further_readout/e4/e4_resnet50.json")["e4"]
    r = e4["valid_only"]["family_pooled"]["e4"]
    assert flag("resnet50", r["ci90"], r["ci95"], e4["margin"], r["mean"]) is True, "prose: ResNet-50's further seeds read EQUIVALENT with the flag in 3.7"

def build_e5_manip() -> None:
    for arch in ARCHS:
        mp = R / f"replication_512_readout/e5_manip512/{arch}/manipulation_readout.json"
        if not mp.exists():
            continue
        m, k, ins = json.loads(mp.read_text()), AK[arch], INSTR[arch]
        d = m["decision"]
        assert d["manipulation_check_passes_at_strong"] is True and d["variants_agree_at_strong"] is True and d["all_checkpoint_version_passes_at_strong"] is True, "prose: the check passes at strong in both architectures, the variants agree and so does the all-checkpoint version"
        put(f"p83-e5-{k}-manip-pass", "yes" if d["manipulation_check_passes_at_strong"] else "no", 0, f"Whether the manipulation check passes at strong at 512 pixels, {arch}.", ins)
        put(f"p83-e5-{k}-manip-agree", "yes" if d["variants_agree_at_strong"] else "no", 0, f"Whether the pooled and the per-seed variants agree at strong, {arch}.", ins)
        for g, tag in (("strong", "strong"), ("zero", "zero")):
            v = m["groups"][g]["valid_only"]
            pc = v["pooled_concatenated"]
            for e, et in (("edited", "ed"), ("unedited", "un")):
                triple(f"p83-e5-{k}-manip-{tag}-{et}", pc[e]["auc"], [pc[e]["ci_low"], pc[e]["ci_high"]], 3, f"Conflict-set AUC at 512 pixels, {arch}, {g} planting, {e} images (pooled over the valid seeds).", ins)
            put(f"p83-e5-{k}-manip-{tag}-eff", v["pooled_effect_unedited_minus_edited"], 4, f"Fall in conflict-set AUC from unedited to edited images, {g} planting, {arch}.", ins)
            put(f"p83-e5-{k}-manip-{tag}-n", len(v["seeds"]), 0, f"Valid seeds in the manipulation check, {g} planting, {arch}.", ins)

def display(x, dec: int) -> str:
    if isinstance(x, str):
        return x
    s = f"{x:.{dec}f}"
    if dec == 0:
        s = f"{int(round(x)):,}"
    return s

def main(argv: list) -> int:
    build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({k: v[0] for k, v in VALS.items()}, indent=1) + "\n")
    print(len(VALS), "values written to", OUT.relative_to(ROOT))
    if "--write-sources" in argv:
        s = json.loads(SOURCES.read_text())
        by = {e["id"]: i for i, e in enumerate(s)}
        for i, (x, dec, note, instrument) in VALS.items():
            e = {"id": i, "value": display(x, dec), "instrument": instrument, "kind": "verified", "script": "paper_values/further_run_values.py", "artifact": "review/replication_512_readout/manuscript_values.json", "path": i,
                 "tolerance": (0.5 if dec == 0 else 5.1 * 10 ** (-dec - 1)) if not isinstance(x, str) else 0, "note": note + " Read from the recorded output by paper_values/further_run_values.py."}
            if i in by:
                s[by[i]] = e
            else:
                s.append(e)
        for i, e in CMPS.items():
            if i in by:
                s[by[i]] = e
            else:
                s.append(e)
        SOURCES.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        print("sources upserted:", len(VALS), "comparisons:", len(CMPS))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
