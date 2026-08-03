#!/usr/bin/env python3
"""The model ladder.

Deliberately a ladder, climbed in order of how much structure each model
assumes:

1. ``mean``          -- predicts the training mean. The thing every other model
                        must beat before it has earned any attention.
2. ``ridge``         -- linear, regularised. The simplest model that can use
                        the features at all.
3. ``gp``            -- Gaussian process, ConstantKernel * RBF + WhiteKernel.
                        The only rung that carries a calibrated predictive
                        variance, which is what an acquisition function needs.
4. ``random_forest`` -- optional. Reported only if it beats ridge by more than
                        the fold noise; otherwise the results table says so.

Every estimator is a pipeline with standardisation, because the GP length scale
and the ridge penalty are both scale-dependent. Every estimator that has a
random component takes the project seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.seeds import RANDOM_SEED

# Ridge penalty. Fixed rather than tuned by an inner search: the training set is
# far too small for nested CV, and the project brief specifies plain LOO. A
# fixed, documented alpha is honest; a tuned one would leak the test fold.
RIDGE_ALPHA = 1.0

# Baseline rung. Everything is compared against this.
BASELINE = "mean"

# Rung that random_forest must beat to be reported at all.
RF_MUST_BEAT = "ridge"


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder."""

    name: str
    estimator: Any
    gives_uncertainty: bool
    note: str


def make_gp_kernel():
    """ConstantKernel * RBF + WhiteKernel.

    The constant carries the signal amplitude, the RBF the correlation length
    in feature space, and the white noise term the irreducible scatter -- which
    is what lets the GP report an uncertainty that does not collapse to zero at
    the training points.
    """
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )


def build_ladder(include_random_forest: bool = True,
                 seed: int = RANDOM_SEED) -> list[Rung]:
    """Construct the ladder in order. Returns a list of Rung."""
    rungs = [
        Rung(
            name="mean",
            estimator=DummyRegressor(strategy="mean"),
            gives_uncertainty=False,
            note="predicts the training mean; the bar every model must clear",
        ),
        Rung(
            name="ridge",
            estimator=Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=RIDGE_ALPHA, random_state=seed)),
            ]),
            gives_uncertainty=False,
            note=f"linear, alpha={RIDGE_ALPHA} fixed (no inner CV at this n)",
        ),
        Rung(
            name="gp",
            estimator=Pipeline([
                ("scale", StandardScaler()),
                ("model", GaussianProcessRegressor(
                    kernel=make_gp_kernel(),
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=seed,
                )),
            ]),
            gives_uncertainty=True,
            note="ConstantKernel * RBF + WhiteKernel; carries predictive sigma",
        ),
    ]

    if include_random_forest:
        rungs.append(
            Rung(
                name="random_forest",
                estimator=Pipeline([
                    ("scale", StandardScaler()),
                    ("model", RandomForestRegressor(
                        n_estimators=500,
                        random_state=seed,
                    )),
                ]),
                gives_uncertainty=False,
                note=f"reported only if it beats {RF_MUST_BEAT} beyond fold noise",
            )
        )

    return rungs


def predict_with_std(estimator: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict, returning (mean, std) where the estimator supports std."""
    model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    if isinstance(model, GaussianProcessRegressor):
        if isinstance(estimator, Pipeline):
            Xt = estimator[:-1].transform(X)
        else:
            Xt = X
        mean, std = model.predict(Xt, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    return np.asarray(estimator.predict(X), dtype=float), None
