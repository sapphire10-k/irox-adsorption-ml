"""Hydrated configurations are out of scope.

They are accidental carryover from an earlier tool migration. Their rows stay
in the dataset for provenance, but role=='excluded' must keep them out of
training, out of the candidate pool, and out of the feature table.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import featurise, inventory

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_id", [
    "slab_H_hydrated_o21_ready",
    "slab_2H_hydrated_o21_o46_ready",
    "slab_H_HYDRATED_o44_ready",
])
def test_hydrated_names_are_excluded(config_id):
    assert inventory.is_excluded(config_id)


@pytest.mark.parametrize("config_id", [
    "slab_H_o21_ready", "slab_2H_o21_o46_ready", "slab_clean_ready_r1",
])
def test_in_scope_names_are_not_excluded(config_id):
    assert not inventory.is_excluded(config_id)


def test_excluded_role_beats_every_other_role(tmp_path):
    """Even a converged, triaged-OK hydrated run must come out excluded."""
    import json

    runs = tmp_path / "runs" / "results"
    runs.mkdir(parents=True)
    cfg = "slab_H_hydrated_o21_ready"
    (runs / f"{cfg}_results.json").write_text(json.dumps({
        "base_name": cfg, "method": "GFN2-xTB", "e_final": -10390.0,
        "converged": True, "fmax_final": 0.04,
    }))

    triage = {cfg: {
        "job_id": cfg, "status": "RUN", "site_intended": "o21",
        "site_final": "o21", "converged": True, "triage_bin": "OK",
        "config_hash": "deadbeef",
    }}

    df = inventory.build([tmp_path / "runs"], triage)
    row = df[df["config_id"] == cfg].iloc[0]
    assert row["role"] == "excluded"
    # The row survives: provenance is not discarded.
    assert row["energy_eV"] == pytest.approx(-10390.0)
    assert row["triage_bin"] == "OK"


def test_featurise_skips_excluded_rows_by_default(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([
        {"config_id": "slab_H_o21_ready", "role": "train"},
        {"config_id": "slab_H_hydrated_o21_ready", "role": "excluded"},
    ]).to_csv(dataset, index=False)

    out = tmp_path / "features.csv"
    proc = subprocess.run(
        [sys.executable, "src/featurise.py",
         "--dataset", str(dataset), "--root", str(tmp_path),
         "--roots-file", "", "--manifest", "", "--out", str(out),
         "--dictionary-out", str(tmp_path / "d.csv"),
         "--dictionary-md", str(tmp_path / "d.md")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "out of scope        : 1" in proc.stdout
    # Neither config has geometry here, so the table is empty either way; what
    # matters is that the excluded row was never even looked for.
    assert "slab_H_hydrated_o21_ready" not in proc.stdout.split("out of scope")[1]


# --------------------------------------------------------------------------
# the real dataset
# --------------------------------------------------------------------------

def _real(name: str) -> Path:
    return REPO_ROOT / "data" / name


@pytest.mark.skipif(not _real("dataset.csv").exists(), reason="dataset not built")
def test_real_dataset_excludes_every_hydrated_config():
    df = pd.read_csv(_real("dataset.csv"))
    hydrated = df[df["config_id"].str.contains("hydrated", case=False)]
    assert len(hydrated) > 0
    assert (hydrated["role"] == "excluded").all()


@pytest.mark.skipif(not _real("features.csv").exists(), reason="features not built")
def test_no_excluded_config_reaches_the_feature_table():
    dataset = pd.read_csv(_real("dataset.csv"))
    features = pd.read_csv(_real("features.csv"))
    excluded = set(dataset[dataset["role"] == "excluded"]["config_id"])
    assert excluded
    assert not (excluded & set(features["config_id"]))


@pytest.mark.skipif(not _real("features.csv").exists(), reason="features not built")
def test_name_based_and_geometry_based_detection_agree():
    """Nothing carrying water should survive as an in-scope configuration."""
    features = pd.read_csv(_real("features.csv"))
    assert (features["n_water"] == 0).all()
