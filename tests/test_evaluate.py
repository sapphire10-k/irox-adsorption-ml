"""Tests for src/evaluate.py.

Two things matter here: the whole ladder runs end to end on the bundled
fixture, and the evaluator refuses to proceed on misaligned features/targets
rather than silently dropping or reordering rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import evaluate
from src.evaluate import AlignmentError

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
# references and targets
# --------------------------------------------------------------------------

def test_clean_reference_is_the_reference_row(fixture_paths):
    dataset = pd.read_csv(fixture_paths["dataset"])
    assert evaluate.get_clean_reference(dataset) == pytest.approx(-10375.723335)


def test_missing_reference_row_is_an_error(fixture_paths):
    dataset = pd.read_csv(fixture_paths["dataset"])
    with pytest.raises(ValueError, match="No role=='reference' row"):
        evaluate.get_clean_reference(dataset[dataset["role"] != "reference"])


def test_duplicate_reference_rows_are_an_error(fixture_paths):
    dataset = pd.read_csv(fixture_paths["dataset"])
    doubled = pd.concat([dataset, dataset[dataset["role"] == "reference"]])
    with pytest.raises(ValueError, match="Expected exactly one"):
        evaluate.get_clean_reference(doubled)


def test_target_follows_the_stated_definition(fixture_paths):
    dataset = pd.read_csv(fixture_paths["dataset"])
    e_clean = evaluate.get_clean_reference(dataset)
    e_ref_h = evaluate.load_h2_reference(fixture_paths["h2"])
    out = evaluate.compute_targets(dataset, e_clean, e_ref_h)

    row = out[out["config_id"] == "slab_H_o20_ready"].iloc[0]
    expected = row["energy_eV"] - e_clean - row["n_adsorbates"] * e_ref_h
    assert row["E_ads_eV"] == pytest.approx(expected)

    # The reference row itself has n=0, so its own E_ads is exactly zero.
    ref = out[out["role"] == "reference"].iloc[0]
    assert ref["E_ads_eV"] == pytest.approx(0.0)


def test_missing_h2_cache_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="script-generated"):
        evaluate.load_h2_reference(tmp_path / "absent.json")


def test_h2_cache_without_the_field_is_an_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"E_H2_eV": -22.4}))
    with pytest.raises(ValueError, match="E_ref_H_eV"):
        evaluate.load_h2_reference(bad)


# --------------------------------------------------------------------------
# alignment refusal
# --------------------------------------------------------------------------

def _targets(fixture_paths) -> pd.DataFrame:
    dataset = pd.read_csv(fixture_paths["dataset"])
    e_clean = evaluate.get_clean_reference(dataset)
    e_ref_h = evaluate.load_h2_reference(fixture_paths["h2"])
    out = evaluate.compute_targets(dataset, e_clean, e_ref_h)
    return out[out["role"] == "train"][["config_id", "E_ads_eV"]].reset_index(drop=True)


def test_refuses_features_missing_a_training_config(fixture_paths):
    features = pd.read_csv(fixture_paths["features"]).iloc[1:]
    with pytest.raises(AlignmentError, match="no feature row"):
        evaluate.align(features, _targets(fixture_paths))


def test_refuses_extra_feature_rows(fixture_paths):
    features = pd.read_csv(fixture_paths["features"])
    extra = features.iloc[[0]].copy()
    extra["config_id"] = "slab_H_o999_ready"
    with pytest.raises(AlignmentError, match="no matching target"):
        evaluate.align(pd.concat([features, extra]), _targets(fixture_paths))


def test_refuses_duplicate_config_ids(fixture_paths):
    features = pd.read_csv(fixture_paths["features"])
    with pytest.raises(AlignmentError, match="duplicate config_id"):
        evaluate.align(pd.concat([features, features.iloc[[0]]]),
                       _targets(fixture_paths))


def test_refuses_features_without_config_id(fixture_paths):
    features = pd.read_csv(fixture_paths["features"]).drop(columns=["config_id"])
    with pytest.raises(AlignmentError, match="no 'config_id' column"):
        evaluate.align(features, _targets(fixture_paths))


def test_refuses_non_finite_features(fixture_paths):
    features = pd.read_csv(fixture_paths["features"])
    features.loc[0, "f_coordination"] = np.nan
    with pytest.raises(AlignmentError, match="non-finite values"):
        evaluate.align(features, _targets(fixture_paths))


def test_refuses_when_no_numeric_features(fixture_paths):
    features = pd.read_csv(fixture_paths["features"])[["config_id"]]
    with pytest.raises(AlignmentError, match="no numeric feature columns"):
        evaluate.align(features, _targets(fixture_paths))


def test_alignment_is_order_independent(fixture_paths):
    features = pd.read_csv(fixture_paths["features"])
    targets = _targets(fixture_paths)
    X1, y1, ids1, _ = evaluate.align(features, targets)
    X2, y2, ids2, _ = evaluate.align(features.iloc[::-1].reset_index(drop=True),
                                     targets.iloc[::-1].reset_index(drop=True))
    assert ids1 == ids2
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(y1, y2)


# --------------------------------------------------------------------------
# full-ladder smoke test
# --------------------------------------------------------------------------

def test_full_ladder_runs_on_the_fixture(xy):
    X, y, config_ids, feature_names = xy
    assert X.shape == (10, 3)
    assert len(y) == 10

    table = evaluate.evaluate_ladder(X, y)
    assert list(table["model"]) == ["mean", "ridge", "gp", "random_forest"]
    assert table["mae_eV"].notna().all()
    assert (table["mae_eV"] >= 0).all()
    assert (table["n_train_rows"] == 10).all()


def test_baseline_row_has_zero_gain_against_itself(xy):
    X, y, _, _ = xy
    table = evaluate.evaluate_ladder(X, y)
    base = table[table["model"] == "mean"].iloc[0]
    assert base["mae_gain_vs_baseline_eV"] == pytest.approx(0.0)
    assert not base["beats_baseline"]


def test_gp_reports_calibration_and_others_do_not(xy):
    X, y, _, _ = xy
    table = evaluate.evaluate_ladder(X, y).set_index("model")
    assert np.isfinite(table.loc["gp", "mean_pred_sigma_eV"])
    assert np.isfinite(table.loc["gp", "mean_abs_z"])
    assert 0.0 <= table.loc["gp", "coverage_95"] <= 1.0
    assert pd.isna(table.loc["ridge", "mean_abs_z"])


def test_random_forest_reporting_is_gated_on_beating_ridge(xy):
    X, y, _, _ = xy
    table = evaluate.evaluate_ladder(X, y).set_index("model")
    clears = table.loc["random_forest", "rf_beats_ridge"]
    assert table.loc["random_forest", "reported"] == clears


def test_ladder_is_deterministic(xy):
    X, y, _, _ = xy
    a = evaluate.evaluate_ladder(X, y)
    b = evaluate.evaluate_ladder(X, y)
    pd.testing.assert_frame_equal(a, b)


def test_ladder_refuses_too_few_rows(xy):
    X, y, _, _ = xy
    with pytest.raises(ValueError, match="at least 3 training rows"):
        evaluate.evaluate_ladder(X[:2], y[:2])


# --------------------------------------------------------------------------
# noise floor + report
# --------------------------------------------------------------------------

def test_noise_floor_takes_the_larger_estimate():
    floor = evaluate.noise_floor_with_empirical(-10375.723335)
    assert floor["tolerance_implied_eV"] == pytest.approx(0.005)
    assert floor["empirical_eV"] == pytest.approx(0.234265, abs=1e-5)
    assert floor["operative_eV"] == pytest.approx(floor["empirical_eV"])


def test_report_states_caution_and_floor(xy):
    X, y, config_ids, feature_names = xy
    table = evaluate.evaluate_ladder(X, y)
    floor = evaluate.noise_floor_with_empirical(-10375.723335)
    text = evaluate.format_table(table, floor, config_ids, feature_names)

    assert "leave-one-out" in text.lower()
    assert "operative floor" in text
    assert "VERDICT:" in text
    assert "CAUTION: n = 10" in text
    # The restraint clause must appear one way or the other.
    assert "random_forest" in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_writes_model_ladder_csv(fixture_paths, tmp_path):
    out = tmp_path / "results" / "model_ladder.csv"
    proc = subprocess.run(
        [sys.executable, "src/evaluate.py",
         "--dataset", str(fixture_paths["dataset"]),
         "--features", str(fixture_paths["features"]),
         "--h2-reference", str(fixture_paths["h2"]),
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()

    table = pd.read_csv(out)
    assert set(table["model"]) == {"mean", "ridge", "gp", "random_forest"}
    assert (table["seed"] == 0).all()
    assert "VERDICT:" in proc.stdout


def test_cli_refuses_misaligned_features(fixture_paths, tmp_path):
    trimmed = tmp_path / "trimmed.csv"
    pd.read_csv(fixture_paths["features"]).iloc[1:].to_csv(trimmed, index=False)

    proc = subprocess.run(
        [sys.executable, "src/evaluate.py",
         "--dataset", str(fixture_paths["dataset"]),
         "--features", str(trimmed),
         "--h2-reference", str(fixture_paths["h2"]),
         "--out", str(tmp_path / "x.csv")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "no feature row" in proc.stderr
