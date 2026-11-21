import json

import numpy as np

from credit_xai import config
from credit_xai.data import load_processed
from credit_xai.features import MODEL_FEATURES, build_preprocessor, get_X_y
from credit_xai.train import choose_threshold, evaluate, expected_cost


def test_sensitive_attributes_not_in_model():
    for attr in config.EXCLUDED_FROM_MODEL:
        assert attr not in MODEL_FEATURES


def test_preprocessor_output(project):
    X, _ = get_X_y(load_processed("train"))
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_expected_cost_weights_false_negatives_more():
    y = np.array([1, 0])
    assert expected_cost(y, np.array([0, 0])) == config.COST_FALSE_NEGATIVE / 2
    assert expected_cost(y, np.array([1, 1])) == config.COST_FALSE_POSITIVE / 2


def test_choose_threshold_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    assert 0.05 <= choose_threshold(y, proba) <= 0.95


def test_evaluate_keys():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.4]), 0.5)
    assert {"auc", "f1", "expected_cost", "confusion"} <= set(m)


def test_training_artifacts(project, credit_model):
    metrics = json.loads(config.METRICS_FILE.read_text())
    assert {"logistic_regression", "xgboost"} <= set(metrics)
    assert 0 < credit_model.threshold < 1
    X, _ = get_X_y(load_processed("test"))
    p = credit_model.prob_default(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert (credit_model.approve(X) == (p < credit_model.threshold)).all()
