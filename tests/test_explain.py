import numpy as np
import pytest

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.explain.counterfactuals import CounterfactualGenerator
from credit_xai.explain.shap_explainer import ShapExplainer, stability
from credit_xai.features import MODEL_FEATURES


@pytest.fixture(scope="module")
def explainer(credit_model):
    return ShapExplainer(credit_model)


@pytest.fixture(scope="module")
def rejected_row(credit_model):
    test = load_processed("test")
    p = credit_model.prob_default(test)
    return test.loc[[p.idxmax()]]


def test_shap_additivity(explainer, rejected_row):
    """base + somma dei contributi = log-odds del modello (TreeSHAP è esatto)."""
    ex = explainer.explain_one(rejected_row)
    total = ex["base_value"] + sum(c["shap"] for c in ex["contributions"])
    p = ex["prob_default"]
    assert total == pytest.approx(np.log(p / (1 - p)), abs=1e-2)


def test_contributions_cover_all_features(explainer, rejected_row):
    ex = explainer.explain_one(rejected_row)
    assert {c["feature"] for c in ex["contributions"]} == set(MODEL_FEATURES)
    shap_abs = [abs(c["shap"]) for c in ex["contributions"]]
    assert shap_abs == sorted(shap_abs, reverse=True)


def test_direction_matches_sign(explainer, rejected_row):
    for c in explainer.explain_one(rejected_row)["contributions"]:
        expected = "aumenta il rischio" if c["shap"] > 0 else "riduce il rischio"
        assert c["direction"] == expected


def test_stability_range(explainer, rejected_row):
    assert 0 <= stability(explainer, rejected_row, n=10) <= 1


def test_counterfactuals_are_valid(credit_model, rejected_row):
    gen = CounterfactualGenerator(credit_model, load_processed("train"))
    cfs = gen.generate(rejected_row)
    for cf in cfs:
        assert cf["prob_default"] < credit_model.threshold
        for change in cf["changes"]:
            assert change["feature"] in config.ACTIONABLE_FEATURES
            if change["feature"] == "checking_status":
                assert change["to"] not in config.IMPLAUSIBLE_TARGETS["checking_status"]
