# Tests of the registered verdict rule on fixed numbers.
# Run: python3 -m pytest tests/test_decision_rule.py
from analysis.heldout_analysis import verdict


def primary(mean, ci90, ci95):
    return {"mean": mean, "ci90": ci90, "ci95": ci95}


def test_equivalent_when_the_90_interval_is_inside_the_margin():
    v = verdict(primary(0.01, [-0.05, 0.06], [-0.07, 0.08]), margin=0.1, invert=0.4)
    assert v["label"] == "EQUIVALENT" and not v["effect_inside_margin_flag"]
    assert abs(v["mean_over_invert_constant"] - 0.025) < 1e-12


def test_effect_inside_the_margin_is_flagged():
    v = verdict(primary(-0.06, [-0.09, -0.03], [-0.095, -0.01]), margin=0.1, invert=0.4)
    assert v["label"] == "EQUIVALENT" and v["effect_inside_margin_flag"]
    assert v["effect_sign"].startswith("negative")


def test_effect_when_the_95_interval_excludes_zero_and_the_90_interval_leaves_the_margin():
    v = verdict(primary(0.3, [0.2, 0.4], [0.18, 0.42]), margin=0.1, invert=0.4)
    assert v["label"] == "EFFECT" and v["effect_sign"].startswith("positive")


def test_inconclusive_when_neither_holds():
    v = verdict(primary(0.05, [-0.12, 0.2], [-0.15, 0.25]), margin=0.1, invert=0.4)
    assert v["label"] == "INCONCLUSIVE" and v["effect_sign"] is None
