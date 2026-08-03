"""Tests for src/models.py -- the ladder itself, independent of any data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Product, Sum, WhiteKernel
from sklearn.pipeline import Pipeline

from src import models
from src.seeds import RANDOM_SEED


def test_ladder_order_and_membership():
    names = [r.name for r in models.build_ladder()]
    assert names == ["mean", "ridge", "gp", "random_forest"]
    assert models.BASELINE == "mean"
    assert models.RF_MUST_BEAT == "ridge"


def test_random_forest_can_be_excluded():
    names = [r.name for r in models.build_ladder(include_random_forest=False)]
    assert "random_forest" not in names


def test_only_gp_advertises_uncertainty():
    with_unc = {r.name for r in models.build_ladder() if r.gives_uncertainty}
    assert with_unc == {"gp"}


def test_gp_kernel_is_constant_times_rbf_plus_white():
    kernel = models.make_gp_kernel()
    assert isinstance(kernel, Sum)
    assert isinstance(kernel.k1, Product)      # ConstantKernel * RBF
    assert isinstance(kernel.k2, WhiteKernel)  # + white noise


def test_every_rung_standardises_except_the_dummy():
    for rung in models.build_ladder():
        if rung.name == "mean":
            continue
        assert isinstance(rung.estimator, Pipeline)
        assert "scale" in dict(rung.estimator.steps)


def test_seeds_are_wired_through():
    rungs = {r.name: r for r in models.build_ladder(seed=RANDOM_SEED)}
    assert rungs["gp"].estimator["model"].random_state == RANDOM_SEED
    assert rungs["random_forest"].estimator["model"].random_state == RANDOM_SEED


def test_predict_with_std_returns_sigma_for_gp_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.05, size=8)

    rungs = {r.name: r for r in models.build_ladder()}

    gp = rungs["gp"].estimator.fit(X, y)
    mean, std = models.predict_with_std(gp, X[:3])
    assert mean.shape == (3,)
    assert std is not None and std.shape == (3,) and (std >= 0).all()

    ridge = rungs["ridge"].estimator.fit(X, y)
    mean, std = models.predict_with_std(ridge, X[:3])
    assert mean.shape == (3,)
    assert std is None


def test_ladder_is_reproducible_across_builds():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    y = rng.normal(size=9)

    def fit_predict():
        rungs = {r.name: r for r in models.build_ladder()}
        return {n: r.estimator.fit(X, y).predict(X) for n, r in rungs.items()}

    a, b = fit_predict(), fit_predict()
    for name in a:
        np.testing.assert_allclose(a[name], b[name], err_msg=name)


@pytest.mark.parametrize("name", ["mean", "ridge", "gp", "random_forest"])
def test_every_rung_fits_and_predicts(name):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=10)
    rung = {r.name: r for r in models.build_ladder()}[name]
    preds = rung.estimator.fit(X, y).predict(X)
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()
