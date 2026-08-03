"""The README is generated, so a stale or hand-edited README is a test failure.

These tests also assert the section order the project requires, and that the
numbers quoted in the prose agree with the committed results tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src import build_readme
from src.feature_subset import MODEL_FEATURES
from src.featurise import FEATURE_DICTIONARY

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def rendered() -> str:
    return build_readme.build(REPO_ROOT)


@pytest.fixture(scope="module")
def committed() -> str:
    return README.read_text()


def test_committed_readme_matches_the_generator(rendered, committed):
    """If this fails, run: python src/build_readme.py"""
    assert committed == rendered, (
        "README.md is stale or was hand-edited. Regenerate with "
        "'python src/build_readme.py'."
    )


def test_section_order_is_the_required_one(committed):
    headings = re.findall(r"^## (.+)$", committed, flags=re.MULTILINE)
    assert headings == [
        "The question",
        "Limitations and judgment",
        "Data provenance",
        "Methods",
        "Results",
        "Figures",
        "Feature dictionary",
        "Reproducing every number",
        "How this repository was built",
    ]


def test_limitations_precedes_results(committed):
    assert committed.index("## Limitations and judgment") < committed.index("## Results")


def test_ci_badge_is_present_and_points_at_the_workflow(committed):
    assert committed.lstrip().startswith("[![CI]")
    assert "actions/workflows/ci.yml/badge.svg" in committed
    assert (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists()


# --------------------------------------------------------------------------
# the numbers in the prose must match the tables
# --------------------------------------------------------------------------

def test_training_set_size_matches_the_dataset(committed):
    dataset = pd.read_csv(REPO_ROOT / "data" / "dataset.csv")
    n_train = int((dataset["role"] == "train").sum())
    assert f"{n_train} are converged and usable as training data today" in committed


def test_coverage_counts_are_internally_consistent(committed):
    m = re.search(r"There are (\d+) adsorption\nconfigurations in scope "
                  r"\((\d+) single-H, (\d+) two-adsorbate pairs\)", committed)
    assert m, "in-scope configuration sentence not found"
    total, n_1h, n_2h = (int(g) for g in m.groups())
    assert total == n_1h + n_2h


def test_ladder_numbers_match_the_results_table(committed):
    ladder = pd.read_csv(REPO_ROOT / "results" / "model_ladder.csv")
    for _, row in ladder.iterrows():
        assert f"| `{row['model']}`" in committed
        assert f"{row['mae_eV']:.4f}" in committed


def test_calibration_counts_match(committed):
    calib = pd.read_csv(REPO_ROOT / "results" / "calibration.csv")
    for label in ("68%", "95%"):
        row = calib.loc[calib["nominal_label"] == label].iloc[0]
        assert f"{int(row['inside'])}/{int(row['n'])}" in committed


def test_noise_floor_matches(committed):
    ladder = pd.read_csv(REPO_ROOT / "results" / "model_ladder.csv")
    assert f"{float(ladder['noise_operative_eV'].iloc[0]):.4f} eV**" in committed


def test_geometry_audit_numbers_are_quoted(committed):
    geom = pd.read_csv(REPO_ROOT / "results" / "geometry_audit.csv")
    slab = geom.iloc[1]
    assert f"{float(slab['ir_o_shortest_A']):.3f} A" in committed
    assert f"{int(slab['n_oo_pairs_below_2A'])} O-O contacts" in committed


def test_h2_reference_is_quoted_from_the_cache(committed):
    import json
    h2 = json.loads((REPO_ROOT / "data" / "h2_reference.json").read_text())
    assert f"{float(h2['E_ref_H_eV']):.6f}" in committed


# --------------------------------------------------------------------------
# required content
# --------------------------------------------------------------------------

def test_feature_dictionary_is_complete(committed):
    for name, _, _ in FEATURE_DICTIONARY:
        assert f"`{name}`" in committed


def test_modelling_subset_is_named(committed):
    for name in MODEL_FEATURES:
        assert f"`{name}`" in committed


def test_provenance_names_the_upstream_pipeline_and_triage(committed):
    assert "IrOx_VQE" in committed
    assert "triage" in committed.lower()


def test_states_where_ml_fits_and_where_it_does_not(committed):
    assert "Prioritisation and uncertainty" in committed
    assert "replacing the physics" in committed


def test_small_dataset_is_framed_as_a_design_choice(committed):
    assert "small by design" in committed
    assert "not apologised for" in committed


def test_no_rescaling_is_promised_in_the_prose(committed):
    assert "reported and never" in committed and "applied" in committed


def test_geometry_concern_is_declared_unfixed(committed):
    assert "has **not** been changed here" in committed
    assert "upstream" in committed


def test_agent_first_line_is_present(committed):
    assert "Built agent-first" in committed
    assert "validated by the test suite" in committed


def test_reproduction_covers_every_generating_script(committed):
    for script in ("inventory.py", "h2_reference.py", "featurise.py",
                   "feature_subset.py", "evaluate.py", "calibration.py",
                   "learning_curve.py", "active_learning.py",
                   "geometry_audit.py", "build_readme.py"):
        assert script in committed, script
