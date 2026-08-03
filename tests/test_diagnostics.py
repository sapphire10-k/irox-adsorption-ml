"""Smoke tests for src/calibration.py and src/learning_curve.py.

Both run against the bundled 10-row fixture, so they exercise the real code
paths without touching the project's own data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import calibration, evaluate, learning_curve
from src.seeds import RANDOM_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_paths() -> dict[str, Path]:
    return {
        "dataset": FIXTURES / "dataset_fixture.csv",
        "features": FIXTURES / "features_fixture.csv",
        "h2": FIXTURES / "h2_reference_fixture.json",
    }


@pytest.fixture
def xy(fixture_paths):
    dataset = pd.read_csv(fixture_paths["dataset"])
    features = pd.read_csv(fixture_paths["features"])
    e_clean = evaluate.get_clean_reference(dataset)
    e_ref_h = evaluate.load_h2_reference(fixture_paths["h2"])
    targets = evaluate.compute_targets(dataset, e_clean, e_ref_h)
    train = targets[targets["role"] == "train"][["config_id", "E_ads_eV"]]
    return evaluate.align(features, train.reset_index(drop=True))


# --------------------------------------------------------------------------
# calibration: maths
# --------------------------------------------------------------------------

def test_gaussian_two_sided_known_values():
    assert calibration.gaussian_two_sided(1.0) == pytest.approx(0.6827, abs=1e-4)
    assert calibration.gaussian_two_sided(1.959964) == pytest.approx(0.95, abs=1e-4)
    assert calibration.gaussian_two_sided(2.575829) == pytest.approx(0.99, abs=1e-4)


def test_wilson_interval_brackets_the_estimate():
    lo, hi = calibration.wilson_interval(2, 5)
    assert lo < 0.4 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_is_wide_at_small_n():
    narrow = calibration.wilson_interval(50, 100)
    wide = calibration.wilson_interval(2, 4)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_wilson_interval_handles_zero_n():
    lo, hi = calibration.wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_coverage_tolerance_is_one_over_n():
    assert calibration.coverage_tolerance(5) == pytest.approx(0.2)
    assert calibration.coverage_tolerance(0) == 1.0


# --------------------------------------------------------------------------
# calibration: on the fixture
# --------------------------------------------------------------------------

def test_gp_loo_predictions_return_positive_sigmas(xy):
    X, y, _, _ = xy
    preds, sigmas = calibration.gp_loo_predictions(X, y)
    assert preds.shape == y.shape
    assert sigmas.shape == y.shape
    assert (sigmas > 0).all()
    assert np.isfinite(preds).all() and np.isfinite(sigmas).all()


def test_calibration_table_is_monotone_in_nominal_level(xy):
    X, y, _, _ = xy
    preds, sigmas = calibration.gp_loo_predictions(X, y)
    table = calibration.calibration_table(y, preds, sigmas)

    assert list(table["nominal_label"]) == [l for l, _ in calibration.NOMINAL_LEVELS]
    # A wider interval can never contain fewer points.
    assert (table["inside"].diff().dropna() >= 0).all()
    assert (table["n"] == len(y)).all()
    assert (table["empirical_coverage"] <= 1.0).all()


def test_calibration_summary_reports_but_never_applies_rescaling(xy):
    X, y, _, _ = xy
    preds, sigmas = calibration.gp_loo_predictions(X, y)
    summary = calibration.calibration_summary(y, preds, sigmas)

    assert "variance_rescale_factor_NOT_APPLIED" in summary
    assert summary["variance_rescale_factor_NOT_APPLIED"] == pytest.approx(summary["rms_z"])

    # Applying the factor would be the bug this guards against: the sigmas the
    # function was handed must be untouched.
    preds2, sigmas2 = calibration.gp_loo_predictions(X, y)
    np.testing.assert_allclose(sigmas, sigmas2)


def test_verdict_names_the_direction_of_miscalibration():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    preds = np.array([1.0, -1.0, 1.0, -1.0])
    sigmas = np.array([0.01, 0.01, 0.01, 0.01])   # absurdly confident
    table = calibration.calibration_table(y, preds, sigmas)
    summary = calibration.calibration_summary(y, preds, sigmas)
    text = "\n".join(calibration.verdict(table, summary))

    assert "OVERCONFIDENT" in text
    assert "NOT been applied" in text


def test_verdict_detects_underconfidence():
    y = np.zeros(6)
    preds = np.zeros(6)
    sigmas = np.full(6, 10.0)     # absurdly wide
    table = calibration.calibration_table(y, preds, sigmas)
    summary = calibration.calibration_summary(y, preds, sigmas)
    text = "\n".join(calibration.verdict(table, summary))
    assert "overstates its error" in text


def test_calibration_is_deterministic(xy):
    X, y, _, _ = xy
    a = calibration.gp_loo_predictions(X, y)
    b = calibration.gp_loo_predictions(X, y)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_calibration_cli_writes_figure_and_csv(fixture_paths, tmp_path):
    fig = tmp_path / "calibration.png"
    out = tmp_path / "calibration.csv"
    proc = subprocess.run(
        [sys.executable, "src/calibration.py",
         "--dataset", str(fixture_paths["dataset"]),
         "--features", str(fixture_paths["features"]),
         "--h2-reference", str(fixture_paths["h2"]),
         "--figure", str(fig), "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert fig.exists() and fig.stat().st_size > 5000
    assert out.exists()

    table = pd.read_csv(out)
    assert len(table) == len(calibration.NOMINAL_LEVELS)
    assert (table["seed"] == RANDOM_SEED).all()
    assert "68%" in proc.stdout and "95%" in proc.stdout
    assert "NOT been applied" in proc.stdout


# --------------------------------------------------------------------------
# learning curve
# --------------------------------------------------------------------------

def test_subset_indices_are_exhaustive_when_few_combinations():
    subsets = learning_curve.subset_indices(n=5, m=4, n_orderings=25)
    assert len(subsets) == 5                       # C(5,4)
    assert len({tuple(s) for s in subsets}) == 5   # all distinct


def test_subset_indices_are_distinct_and_right_size():
    subsets = learning_curve.subset_indices(n=10, m=4, n_orderings=12)
    assert len(subsets) == 12
    assert all(len(s) == 4 for s in subsets)
    assert len({tuple(sorted(s)) for s in subsets}) == 12


def test_subset_indices_are_reproducible():
    a = learning_curve.subset_indices(10, 5, 8)
    b = learning_curve.subset_indices(10, 5, 8)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


def test_curve_covers_every_size_and_model(xy):
    X, y, _, _ = xy
    table = learning_curve.curve(X, y, n_orderings=4)

    assert set(table["n_train"]) == set(range(evaluate.MIN_TRAIN_ROWS, len(y) + 1))
    assert set(table["model"]) == {"mean", "ridge", "gp"}
    assert (table["mae_mean_eV"] >= 0).all()
    assert (table["mae_min_eV"] <= table["mae_max_eV"]).all()
    assert (table["n_subsets"] >= 1).all()


def test_curve_flags_numerical_instability_rather_than_hiding_it(xy):
    """A blown-up fold must be flagged, not silently dropped or repaired."""
    X, y, _, _ = xy
    table = learning_curve.curve(X, y, n_orderings=3)
    assert "unstable" in table.columns
    assert "unstable_threshold_eV" in table.columns
    assert "p_over_n_train" in table.columns
    # Whatever the verdict, flagged rows stay in the table.
    assert len(table) == len(table.dropna(subset=["mae_mean_eV"]))


def test_unstable_threshold_scales_with_the_target_span(xy):
    X, y, _, _ = xy
    table = learning_curve.curve(X, y, n_orderings=3)
    span = float(y.max() - y.min())
    expected = learning_curve.UNSTABLE_FACTOR * span
    assert table["unstable_threshold_eV"].to_numpy() == pytest.approx(expected)


def test_curve_is_deterministic(xy):
    X, y, _, _ = xy
    a = learning_curve.curve(X, y, n_orderings=4)
    b = learning_curve.curve(X, y, n_orderings=4)
    pd.testing.assert_frame_equal(a, b)


def test_verdict_says_when_nothing_beats_the_baseline():
    """Constructed so every model is at best equal to the mean predictor."""
    table = pd.DataFrame([
        {"model": "mean", "n_train": 4, "mae_mean_eV": 0.50, "mae_std_eV": 0.10,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.75},
        {"model": "gp", "n_train": 4, "mae_mean_eV": 0.60, "mae_std_eV": 0.10,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.75},
        {"model": "mean", "n_train": 5, "mae_mean_eV": 0.48, "mae_std_eV": 0.00,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.60},
        {"model": "gp", "n_train": 5, "mae_mean_eV": 0.65, "mae_std_eV": 0.00,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.60},
    ])
    text = "\n".join(learning_curve.data_limited_verdict(table, floor=0.2))
    assert "NO model beats the mean baseline" in text
    assert "got WORSE" in text


def test_verdict_reports_improvement_with_the_right_sign():
    table = pd.DataFrame([
        {"model": "mean", "n_train": 4, "mae_mean_eV": 0.50, "mae_std_eV": 0.01,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.75},
        {"model": "gp", "n_train": 4, "mae_mean_eV": 0.40, "mae_std_eV": 0.01,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.75},
        {"model": "mean", "n_train": 5, "mae_mean_eV": 0.50, "mae_std_eV": 0.00,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.60},
        {"model": "gp", "n_train": 5, "mae_mean_eV": 0.20, "mae_std_eV": 0.00,
         "unstable": False, "n_features": 3, "p_over_n_train": 0.60},
    ])
    text = "\n".join(learning_curve.data_limited_verdict(table, floor=0.05))
    assert "improved by 0.2000 eV" in text
    assert "Beats the mean baseline" in text


def test_verdict_explains_the_cause_of_instability():
    table = pd.DataFrame([
        {"model": "mean", "n_train": 3, "mae_mean_eV": 0.5, "mae_std_eV": 0.1,
         "unstable": False, "n_features": 25, "p_over_n_train": 8.3},
        {"model": "ridge", "n_train": 3, "mae_mean_eV": 5e8, "mae_std_eV": 1e9,
         "unstable": True, "n_features": 25, "p_over_n_train": 8.3},
    ])
    text = "\n".join(learning_curve.data_limited_verdict(table, floor=0.2))
    assert "NUMERICALLY UNSTABLE" in text
    assert "25 features" in text


def test_learning_curve_cli_writes_figure_and_csv(fixture_paths, tmp_path):
    fig = tmp_path / "learning_curve.png"
    out = tmp_path / "learning_curve.csv"
    proc = subprocess.run(
        [sys.executable, "src/learning_curve.py",
         "--dataset", str(fixture_paths["dataset"]),
         "--features", str(fixture_paths["features"]),
         "--h2-reference", str(fixture_paths["h2"]),
         "--orderings", "4",
         "--figure", str(fig), "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert fig.exists() and fig.stat().st_size > 5000
    assert out.exists()

    table = pd.read_csv(out)
    assert (table["seed"] == RANDOM_SEED).all()
    assert "Data-limited regime" in proc.stdout
    assert "noise floor" in proc.stdout


def test_both_figures_are_regenerable_byte_for_byte(fixture_paths, tmp_path):
    """One command each, and the same command twice gives the same figure."""
    def run(script: str, tag: str, extra: list[str]) -> bytes:
        fig = tmp_path / f"{tag}.png"
        proc = subprocess.run(
            [sys.executable, f"src/{script}",
             "--dataset", str(fixture_paths["dataset"]),
             "--features", str(fixture_paths["features"]),
             "--h2-reference", str(fixture_paths["h2"]),
             "--figure", str(fig), "--out", str(tmp_path / f"{tag}.csv"), *extra],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return fig.read_bytes()

    assert run("calibration.py", "cal_a", []) == run("calibration.py", "cal_b", [])
    assert (run("learning_curve.py", "lc_a", ["--orderings", "4"])
            == run("learning_curve.py", "lc_b", ["--orderings", "4"]))
