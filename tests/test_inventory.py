"""Tests for src/inventory.py, run against the synthetic tree in conftest.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import inventory
from src.seeds import RANDOM_SEED, set_seeds

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(fake_project) -> pd.DataFrame:
    triage = inventory.load_triage(fake_project["triage_summary"], fake_project["jobs"])
    return inventory.build([fake_project["runs"]], triage)


def _row(df: pd.DataFrame, config_id: str) -> pd.Series:
    matches = df[df["config_id"] == config_id]
    assert len(matches) == 1, f"expected exactly one row for {config_id}, got {len(matches)}"
    return matches.iloc[0]


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------

def test_seed_is_fixed_and_returned():
    assert RANDOM_SEED == 0
    assert set_seeds() == 0


# --------------------------------------------------------------------------
# name parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "base_name, expected",
    [
        ("slab_clean_ready", ("clean", "clean", 0)),
        ("slab_H_o21_ready", ("o21", "1H", 1)),
        ("slab_H_o88_ready", ("o88", "1H", 1)),
        # Still parsed correctly even though it is out of scope downstream.
        ("slab_H_hydrated_o21_ready", ("o21", "1H", 1)),
        ("slab_2H_o21_o46_ready", ("o21+o46", "2H", 2)),
        ("slab_mystery", ("unknown", "unknown", 0)),
    ],
)
def test_parse_base_name(base_name, expected):
    assert inventory.parse_base_name(base_name) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [("GFN2-xTB", "xTB"), ("gfn1-xtb", "xTB"), ("Quantum ESPRESSO", "QE"),
     ("PBE/pw.x", "QE"), (None, "unknown"), ("", "unknown"), ("other", "unknown")],
)
def test_classify_method(raw, expected):
    assert inventory.classify_method(raw) == expected


# --------------------------------------------------------------------------
# one row per configuration
# --------------------------------------------------------------------------

def test_one_row_per_configuration(dataset):
    assert dataset["config_id"].is_unique
    assert list(dataset.columns) == inventory.COLUMNS


def test_expected_configurations_present(dataset):
    assert set(dataset["config_id"]) == {
        "slab_H_o20_ready",           # untriaged, has runs
        "slab_H_o21_ready",           # OK
        "slab_H_o22_ready",           # RESTART_FIRE
        "slab_H_o44_ready",           # RELABEL_SITE
        "slab_H_o88_ready",  # PENDING, metadata only
        "slab_2H_o21_o46_ready",      # built, never run, never triaged
    }


def test_bare_directory_metadata_is_not_a_configuration(dataset):
    assert "o20" not in set(dataset["config_id"])


# --------------------------------------------------------------------------
# training set vs candidate pool
# --------------------------------------------------------------------------

def test_only_allowlisted_converged_bins_are_training_data(dataset):
    train = dataset[dataset["role"] == "train"]
    assert set(train["config_id"]) == {"slab_H_o21_ready", "slab_H_o44_ready"}
    assert train["converged"].all()
    assert set(train["triage_bin"]) <= inventory.TRAIN_BINS


def test_relabel_site_trains_but_under_its_final_site(dataset):
    # The triage contract: the run is sound, only its label moved.
    row = _row(dataset, "slab_H_o44_ready")
    assert row["converged"]
    assert row["triage_bin"] == "RELABEL_SITE"
    assert row["role"] == "train"
    assert row["site_id"] == "o45"
    assert row["site_intended"] == "o44"


@pytest.mark.parametrize("bin_name", ["RESTART_FIRE", "RESTART_MORE_STEPS",
                                      "DISCARD_PHYSICAL", "SETUP_ERROR",
                                      "PENDING", "NOT_TRIAGED"])
def test_non_allowlisted_bins_never_train(bin_name):
    assert bin_name not in inventory.TRAIN_BINS


def test_unconverged_row_is_kept_as_candidate(dataset):
    row = _row(dataset, "slab_H_o22_ready")
    assert row["role"] == "candidate"
    assert not row["converged"]
    assert row["triage_bin"] == "RESTART_FIRE"
    # Kept, not dropped: it still carries its energy.
    assert row["energy_eV"] == pytest.approx(-10391.1)


def test_pending_row_is_kept_as_candidate_with_no_energy(dataset):
    row = _row(dataset, "slab_H_o88_ready")
    assert row["role"] == "candidate"
    assert row["status"] == "PENDING"
    assert row["triage_bin"] == "PENDING"
    assert pd.isna(row["energy_eV"])


def test_built_but_never_run_pair_is_a_pending_candidate(dataset):
    row = _row(dataset, "slab_2H_o21_o46_ready")
    assert row["coverage"] == "2H"
    assert row["n_adsorbates"] == 2
    assert row["role"] == "candidate"
    assert row["status"] == "PENDING"
    assert not row["in_triage"]
    assert pd.isna(row["energy_eV"])


def test_untriaged_converged_run_is_never_training_data(dataset):
    """The o20 case: a converged run with no triage record stays a candidate."""
    row = _row(dataset, "slab_H_o20_ready")
    assert not row["in_triage"]
    assert row["triage_bin"] == "NOT_TRIAGED"
    assert row["role"] == "candidate"
    assert row["n_runs_found"] == 2


# --------------------------------------------------------------------------
# site intent
# --------------------------------------------------------------------------

def test_site_intended_is_never_overwritten(dataset):
    row = _row(dataset, "slab_H_o44_ready")
    assert row["site_intended"] == "o44"
    assert row["site_id"] == "o45"
    assert row["site_relabelled"]


def test_site_id_follows_intent_when_no_relabel(dataset):
    row = _row(dataset, "slab_H_o21_ready")
    assert row["site_id"] == "o21"
    assert row["site_intended"] == "o21"
    assert not row["site_relabelled"]


def test_untriaged_site_falls_back_to_parsed_name(dataset):
    row = _row(dataset, "slab_H_o20_ready")
    assert row["site_id"] == "o20"
    assert row["site_intended"] == "o20"
    assert not row["site_relabelled"]


# --------------------------------------------------------------------------
# energy provenance
# --------------------------------------------------------------------------

def test_triage_job_dir_selects_the_energy(dataset):
    """o21 has two runs; triage points at run_b, so run_b's energy must win."""
    row = _row(dataset, "slab_H_o21_ready")
    assert row["energy_source"] == "triage_job_dir"
    assert row["energy_eV"] == pytest.approx(-10387.5)
    assert row["n_runs_found"] == 2


def test_untriaged_multi_run_energy_is_flagged_ambiguous(dataset):
    row = _row(dataset, "slab_H_o20_ready")
    assert row["energy_source"] == "ambiguous_multi_run"


def test_config_hash_carried_through(dataset):
    row = _row(dataset, "slab_H_o21_ready")
    assert row["config_hash"] == "abc123def456"


def test_provenance_records_both_sources(dataset):
    row = _row(dataset, "slab_H_o21_ready")
    assert "run=" in row["provenance"]
    assert "triage=" in row["provenance"]


def test_geometry_path_resolves_for_run(dataset):
    row = _row(dataset, "slab_H_o21_ready")
    assert row["geometry_path"].endswith("slab_H_o21_ready_final.traj")
    assert Path(row["geometry_path"]).exists()


def test_geometry_path_resolves_for_pending_pair(dataset):
    row = _row(dataset, "slab_2H_o21_o46_ready")
    assert row["geometry_path"].endswith("slab_2H_o21_o46_ready.traj")
    assert Path(row["geometry_path"]).exists()


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_build_is_deterministic(fake_project):
    triage = inventory.load_triage(fake_project["triage_summary"], fake_project["jobs"])
    a = inventory.build([fake_project["runs"]], triage)
    b = inventory.build([fake_project["runs"]], triage)
    pd.testing.assert_frame_equal(a, b)


def test_same_root_twice_does_not_duplicate(fake_project):
    triage = inventory.load_triage(fake_project["triage_summary"], fake_project["jobs"])
    once = inventory.build([fake_project["runs"]], triage)
    twice = inventory.build([fake_project["runs"], fake_project["runs"] / "run_a"], triage)
    assert len(once) == len(twice)
    assert once["config_id"].is_unique and twice["config_id"].is_unique


# --------------------------------------------------------------------------
# triage loading
# --------------------------------------------------------------------------

def test_load_triage_merges_job_dir_from_per_job_records(fake_project):
    triage = inventory.load_triage(fake_project["triage_summary"], fake_project["jobs"])
    assert "job_dir" in triage["slab_H_o21_ready"]
    assert triage["slab_H_o21_ready"]["job_dir"].endswith("run_b/results")


def test_load_triage_without_jobs_dir(fake_project, tmp_path):
    triage = inventory.load_triage(fake_project["triage_summary"], tmp_path / "nope")
    assert triage["slab_H_o21_ready"]["triage_bin"] == "OK"
    assert triage["slab_H_o21_ready"].get("job_dir") is None


def test_load_triage_rejects_csv_without_job_id(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="job_id"):
        inventory.load_triage(bad, None)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def test_summary_reports_training_count_and_honesty_caveat(dataset, fake_project):
    text = inventory.summarise(dataset, [fake_project["runs"]],
                               fake_project["triage_summary"])
    assert "TRAINING POINTS AVAILABLE TODAY : 2" in text
    assert "NOT adsorption energy" in text
    assert "slab_H_o20_ready" in text          # flagged as not triaged
    assert "intended o44 -> final o45" in text  # relabel surfaced


def test_summary_handles_empty_frame(fake_project):
    empty = inventory.build([], {})
    text = inventory.summarise(empty, [], fake_project["triage_summary"])
    assert "No configurations found." in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_writes_dataset(fake_project, tmp_path):
    out = tmp_path / "data" / "dataset.csv"
    proc = subprocess.run(
        [sys.executable, "src/inventory.py",
         "--root", str(fake_project["runs"]),
         "--triage-summary", str(fake_project["triage_summary"]),
         "--triage-jobs", str(fake_project["jobs"]),
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    df = pd.read_csv(out)
    assert len(df) == 6
    assert list(df.columns) == inventory.COLUMNS


def test_cli_requires_triage_summary(fake_project):
    proc = subprocess.run(
        [sys.executable, "src/inventory.py", "--root", str(fake_project["runs"])],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "--triage-summary" in proc.stderr


def test_cli_requires_a_root(fake_project, tmp_path):
    proc = subprocess.run(
        [sys.executable, "src/inventory.py",
         "--triage-summary", str(fake_project["triage_summary"]),
         "--out", str(tmp_path / "x.csv")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "no roots given" in proc.stderr


def test_cli_fails_loudly_on_missing_root(fake_project, tmp_path):
    proc = subprocess.run(
        [sys.executable, "src/inventory.py",
         "--root", str(tmp_path / "nope"),
         "--triage-summary", str(fake_project["triage_summary"])],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "root not found" in proc.stderr
