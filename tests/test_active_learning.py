"""Smoke tests for src/feature_subset.py and src/active_learning.py.

Everything runs against the bundled 10-row fixture. Random repeats are kept
small so the suite stays fast; the code paths are the same ones the real run
uses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import active_learning, evaluate, feature_subset
from src.seeds import RANDOM_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

REPEATS = 2   # keep the smoke test quick
SMALL_N = 5   # rows used for the fixed-holdout paths: C(n,2) x pool GP fits


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


@pytest.fixture
def xy_small(xy):
    """A 5-row slice, matching the real training-set size."""
    X, y, ids, names = xy
    return X[:SMALL_N], y[:SMALL_N], ids[:SMALL_N], names


@pytest.fixture
def pool_dataset(fixture_paths, tmp_path) -> Path:
    """The fixture dataset with three rows relabelled as pending candidates."""
    dataset = pd.read_csv(fixture_paths["dataset"])
    train_ids = dataset[dataset["role"] == "train"]["config_id"].tolist()
    # 3 pending candidates, and the training set trimmed to SMALL_N rows so the
    # fixed-holdout sweep stays a smoke test rather than a full run.
    dataset.loc[dataset["config_id"].isin(train_ids[-3:]), "role"] = "candidate"
    dataset.loc[dataset["config_id"].isin(train_ids[-3:]), "triage_bin"] = "RESTART_FIRE"
    surplus = train_ids[SMALL_N:-3]
    dataset.loc[dataset["config_id"].isin(surplus), "role"] = "excluded"
    out = tmp_path / "dataset_with_pool.csv"
    dataset.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------------
# feature subset: the stated decision
# --------------------------------------------------------------------------

def test_model_features_are_the_four_stated_ones():
    assert feature_subset.MODEL_FEATURES == (
        "gcn_mean", "nn1_dist_A", "n_bridge_sites", "n_adsorbates",
    )


def test_every_model_feature_has_a_recorded_rationale():
    documented = {name for name, _ in feature_subset.RATIONALE}
    assert documented == set(feature_subset.MODEL_FEATURES)
    for _, why in feature_subset.RATIONALE:
        assert len(why) > 40


def test_subset_frame_keeps_only_the_chosen_columns():
    full = pd.DataFrame({
        "config_id": ["b", "a"],
        "gcn_mean": [1.0, 2.0], "nn1_dist_A": [2.4, 2.4],
        "n_bridge_sites": [1, 2], "n_adsorbates": [1, 2],
        "z_site_A": [26.0, 26.0], "selection_note": ["x", "y"],
    })
    out = feature_subset.subset_frame(full)
    assert list(out.columns) == ["config_id", *feature_subset.MODEL_FEATURES]
    assert list(out["config_id"]) == ["a", "b"]      # deterministic order


def test_subset_frame_refuses_a_missing_feature():
    with pytest.raises(KeyError, match="missing modelling feature"):
        feature_subset.subset_frame(pd.DataFrame({"config_id": ["a"], "gcn_mean": [1.0]}))


def test_rationale_markdown_mentions_that_it_precedes_comparison():
    md = feature_subset.rationale_markdown()
    assert "before" in md
    for name in feature_subset.MODEL_FEATURES:
        assert f"`{name}`" in md


# --------------------------------------------------------------------------
# live recommendation
# --------------------------------------------------------------------------

def test_ranking_is_sorted_by_sigma_and_numbered(xy):
    X, y, _, _ = xy
    meta = pd.DataFrame({"config_id": [f"cand_{i}" for i in range(4)],
                         "coverage": ["1H", "2H", "1H", "2H"]})
    X_pool = X[:4] + 0.05
    ranking = active_learning.rank_candidates(X, y, X_pool, meta)

    assert list(ranking["rank"]) == [1, 2, 3, 4]
    sigmas = ranking["predictive_sigma_eV"].to_numpy()
    assert (np.diff(sigmas) <= 1e-12).all()
    assert "cumulative_cost_minutes" in ranking.columns
    assert ranking["cumulative_cost_minutes"].iloc[0] == pytest.approx(
        active_learning.MINUTES_PER_OPTIMISATION
    )


def test_degeneracy_detected_when_every_sigma_is_equal():
    ranking = pd.DataFrame({
        "predictive_sigma_eV": [0.615, 0.615, 0.6150001],
        "predicted_E_ads_eV": [0.734, 0.734, 0.734],
    })
    d = active_learning.ranking_degeneracy(ranking)
    assert d["degenerate"]
    assert d["sigma_relative_span"] < active_learning.DEGENERACY_FRACTION


def test_degeneracy_not_flagged_when_sigmas_differ():
    ranking = pd.DataFrame({
        "predictive_sigma_eV": [0.9, 0.5, 0.2],
        "predicted_E_ads_eV": [1.0, 0.5, 0.1],
    })
    assert not active_learning.ranking_degeneracy(ranking)["degenerate"]


# --------------------------------------------------------------------------
# retrospective simulation (confounded variant)
# --------------------------------------------------------------------------

def test_simulate_acquisition_covers_both_policies(xy_small):
    X, y, _, _ = xy_small
    runs = active_learning.simulate_acquisition(X, y, n_random_repeats=REPEATS)
    assert set(runs["policy"]) == {"max_uncertainty", "random"}
    assert runs["n_acquired"].min() == active_learning.START_SIZE
    # The last point is never scored, because nothing would be held out.
    assert runs["n_acquired"].max() == len(y) - 1
    assert runs["held_out_mae_eV"].notna().all()


def test_simulate_acquisition_refuses_too_few_points(xy):
    X, y, _, _ = xy
    with pytest.raises(ValueError, match="at least"):
        active_learning.simulate_acquisition(X[:3], y[:3])


def test_first_step_is_identical_under_both_policies(xy_small):
    """No choice has been made yet, so the two policies must agree exactly."""
    X, y, _, _ = xy_small
    runs = active_learning.simulate_acquisition(X, y, n_random_repeats=REPEATS)
    first = runs[runs["n_acquired"] == active_learning.START_SIZE]
    summary = first.groupby("policy")["held_out_mae_eV"].mean()
    assert summary["max_uncertainty"] == pytest.approx(summary["random"])


# --------------------------------------------------------------------------
# unconfounded fixed-holdout variant
# --------------------------------------------------------------------------

def test_fixed_holdout_scores_both_policies_on_the_same_points(xy_small):
    X, y, _, _ = xy_small
    runs = active_learning.simulate_fixed_holdout(X, y, n_random_repeats=REPEATS)
    assert set(runs["policy"]) == {"max_uncertainty", "random"}
    assert "pool_size" in runs.columns
    assert (runs["pool_size"] == len(y) - active_learning.HOLDOUT_SIZE).all()
    assert runs["fixed_holdout_mae_eV"].notna().all()


def test_exhausted_pool_step_is_marked_vacuous(xy_small):
    """When both policies have taken the whole pool they are the same model."""
    X, y, _, _ = xy_small
    runs = active_learning.simulate_fixed_holdout(X, y, n_random_repeats=REPEATS)
    paired = active_learning.paired_comparison_fixed(runs, noise_floor_eV=0.234)

    pool_size = len(y) - active_learning.HOLDOUT_SIZE
    at_full = paired[paired["n_acquired"] == pool_size].iloc[0]
    assert at_full["vacuous_pool_exhausted"]
    assert not at_full["uncertainty_beats_random"]


def test_gain_below_the_noise_floor_is_not_called_a_win(xy_small):
    X, y, _, _ = xy_small
    runs = active_learning.simulate_fixed_holdout(X, y, n_random_repeats=REPEATS)
    tiny_floor = active_learning.paired_comparison_fixed(runs, noise_floor_eV=0.0)
    huge_floor = active_learning.paired_comparison_fixed(runs, noise_floor_eV=1e6)
    # With an enormous floor nothing can qualify as a real win.
    assert not huge_floor["uncertainty_beats_random"].any()
    assert (huge_floor["gain_exceeds_noise_floor"] == False).all()  # noqa: E712
    assert len(tiny_floor) == len(huge_floor)


def test_fixed_holdout_is_deterministic(xy_small):
    X, y, _, _ = xy_small
    a = active_learning.simulate_fixed_holdout(X, y, n_random_repeats=REPEATS)
    b = active_learning.simulate_fixed_holdout(X, y, n_random_repeats=REPEATS)
    pd.testing.assert_frame_equal(a, b)


def test_verdict_states_the_finding_when_uncertainty_does_not_win():
    paired = pd.DataFrame([
        {"n_acquired": 3, "n_starts": 10, "mean_gain_eV": 0.01,
         "gain_se_eV": 0.04, "starts_where_uncertainty_wins": 5,
         "starts_where_random_wins": 5, "ties": 0,
         "uncertainty_beats_random": False},
    ])
    text = "\n".join(active_learning.acquisition_verdict(paired, n_train=5))
    assert "does NOT beat random" in text
    assert "Five points do not support a surrogate yet" in text
    assert "rerun as the" in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_writes_figure_and_both_tables(fixture_paths, pool_dataset, tmp_path):
    fig = tmp_path / "active_learning.png"
    ranking_out = tmp_path / "candidate_ranking.csv"
    policy_out = tmp_path / "acquisition_policy.csv"
    proc = subprocess.run(
        [sys.executable, "src/active_learning.py",
         "--dataset", str(pool_dataset),
         "--features", str(fixture_paths["features"]),
         "--h2-reference", str(fixture_paths["h2"]),
         "--random-repeats", str(REPEATS),
         "--figure", str(fig),
         "--ranking-out", str(ranking_out),
         "--policy-out", str(policy_out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert fig.exists() and fig.stat().st_size > 5000

    ranking = pd.read_csv(ranking_out)
    assert len(ranking) == 3
    assert (ranking["seed"] == RANDOM_SEED).all()
    assert "gp_rms_z_under_loo" in ranking.columns

    policy = pd.read_csv(policy_out)
    assert set(policy["evaluation"]) == {
        "not_yet_acquired_pool_CONFOUNDED", "fixed_holdout_UNCONFOUNDED",
    }

    # The report must carry both the queue and the honesty framing.
    assert "LIVE RECOMMENDATION" in proc.stdout
    assert "UNCONFOUNDED RE-CHECK" in proc.stdout
    assert "23 min" in proc.stdout


def test_cli_is_reproducible(fixture_paths, pool_dataset, tmp_path):
    def run(tag: str) -> bytes:
        out = tmp_path / f"ranking_{tag}.csv"
        proc = subprocess.run(
            [sys.executable, "src/active_learning.py",
             "--dataset", str(pool_dataset),
             "--features", str(fixture_paths["features"]),
             "--h2-reference", str(fixture_paths["h2"]),
             "--random-repeats", str(REPEATS),
             "--figure", str(tmp_path / f"fig_{tag}.png"),
             "--ranking-out", str(out),
             "--policy-out", str(tmp_path / f"policy_{tag}.csv")],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return out.read_bytes()

    assert run("a") == run("b")
