"""Tests for src/featurise.py.

The two that matter most:

* ``test_build_is_reproducible_in_one_command`` -- one command, same bytes.
* the no-leakage block -- a fixture whose pre- and post-relaxation geometries
  differ substantially, proving the featuriser reads the input structure and
  never the relaxed one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import write

from src import featurise
from src.featurise import LeakageError

REPO_ROOT = Path(__file__).resolve().parents[1]

# Displacement applied to make the "relaxed" fixture differ from the input.
RELAXED_SHIFT_A = 0.9


def _slab(oh_distance: float = 1.0, shift: float = 0.0) -> Atoms:
    """A small non-periodic IrO2-like slab with one H on a bridging oxygen.

    Two Ir below one bridging O, so the binding oxygen has Ir coordination 2.
    ``shift`` displaces every heavy atom, standing in for relaxation.
    """
    positions = [
        (0.0, 0.0, 0.0), (2.4, 0.0, 0.0),          # Ir, Ir
        (4.8, 0.0, 0.0), (7.2, 0.0, 0.0),          # Ir, Ir
        (1.2, 0.0, 2.08),                          # O  -- bridging, index 4
        (6.0, 0.0, 2.08),                          # O  -- bridging, index 5
        (3.6, 2.0, 0.0),                           # Ir, index 6
        (1.2, 2.0, 2.08),                          # O, index 7
    ]
    symbols = ["Ir", "Ir", "Ir", "Ir", "O", "O", "Ir", "O"]
    positions = [(x, y, z + shift * (0.3 if i % 2 else 0.7))
                 for i, (x, y, z) in enumerate(positions)]
    # H sits above the bridging O at index 4.
    ox, oy, oz = positions[4]
    positions.append((ox, oy, oz + oh_distance))
    symbols.append("H")

    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=(20.0, 20.0, 20.0), pbc=False)
    return atoms


@pytest.fixture
def geometry_tree(tmp_path: Path) -> dict:
    """A run tree holding BOTH a pre-relaxation input and a relaxed output.

    The two geometries differ by well over an angstrom, so any feature read
    from the wrong file is unmistakable.
    """
    root = tmp_path / "runs"
    cfg = "slab_H_o21_ready"

    structures = root / "job" / "structures"
    results = root / "job" / "results"
    structures.mkdir(parents=True)
    results.mkdir(parents=True)

    pre = _slab(oh_distance=1.0, shift=0.0)
    post = _slab(oh_distance=0.97, shift=RELAXED_SHIFT_A)

    write(structures / f"{cfg}.traj", pre)
    write(results / f"{cfg}_final.traj", post)          # relaxed -- denylisted
    write(results / f"{cfg}_relaxed.xyz", post)         # relaxed -- denylisted
    write(results / f"{cfg}_opt.traj", [pre, post])     # frame 0 == pre

    (results / f"metadata_{cfg}.json").write_text(json.dumps({
        "base_name": cfg,
        "structure_file": f"outputs/job/structures/{cfg}.traj",
    }))

    return {"root": root, "config_id": cfg, "pre": pre, "post": post}


# --------------------------------------------------------------------------
# no leakage
# --------------------------------------------------------------------------

def test_relaxed_artefacts_are_recognised():
    assert featurise.is_relaxed_artefact("slab_H_o21_ready_final.traj")
    assert featurise.is_relaxed_artefact("slab_H_o21_ready_relaxed.xyz")
    assert not featurise.is_relaxed_artefact("slab_H_o21_ready.traj")
    assert not featurise.is_relaxed_artefact("slab_H_o21_ready_opt.traj")


def test_loading_a_relaxed_file_raises(geometry_tree):
    final = geometry_tree["root"] / "job" / "results" / f"{geometry_tree['config_id']}_final.traj"
    assert final.exists()
    with pytest.raises(LeakageError, match="relaxed geometry"):
        featurise.load_geometry(final)


def test_loading_a_non_zero_frame_raises(geometry_tree):
    opt = geometry_tree["root"] / "job" / "results" / f"{geometry_tree['config_id']}_opt.traj"
    with pytest.raises(LeakageError, match="only frame 0"):
        featurise.load_geometry(opt, frame=-1)


def test_resolver_prefers_the_built_input(geometry_tree):
    path, source, frame = featurise.find_pre_relaxation_geometry(
        geometry_tree["config_id"], [geometry_tree["root"]]
    )
    assert source == "built_input"
    assert frame == 0
    assert path.parent.name == "structures"
    assert not featurise.is_relaxed_artefact(path)


def test_features_match_the_pre_relaxation_geometry_not_the_relaxed_one(geometry_tree):
    """The no-leakage test: features must equal those of the input structure."""
    frame, skipped = featurise.build([geometry_tree["config_id"]], [geometry_tree["root"]])
    assert not skipped
    row = frame.iloc[0]

    from_pre = featurise.featurise_atoms(geometry_tree["pre"])
    from_post = featurise.featurise_atoms(geometry_tree["post"])

    # The two geometries really are different, or this test proves nothing.
    assert from_pre["z_site_A"] != pytest.approx(from_post["z_site_A"])
    assert from_pre["d_OH_mean_A"] != pytest.approx(from_post["d_OH_mean_A"])

    for key in ("z_site_A", "d_OH_mean_A", "nn1_dist_A", "cn_mean"):
        assert row[key] == pytest.approx(from_pre[key]), key
        assert row[key] != pytest.approx(from_post[key]), key


def test_no_relaxed_file_is_ever_referenced(geometry_tree):
    frame, _ = featurise.build([geometry_tree["config_id"]], [geometry_tree["root"]])
    for path in frame["geometry_file"]:
        assert not featurise.is_relaxed_artefact(path)


def test_opt_traj_fallback_uses_frame_zero(tmp_path):
    """With no built input available, frame 0 of the trajectory is used."""
    cfg = "slab_H_o99_ready"
    results = tmp_path / "job" / "results"
    results.mkdir(parents=True)
    pre, post = _slab(1.0, 0.0), _slab(0.97, RELAXED_SHIFT_A)
    write(results / f"{cfg}_opt.traj", [pre, post])

    path, source, frame = featurise.find_pre_relaxation_geometry(cfg, [tmp_path])
    assert source == "opt_traj_frame0"
    assert frame == 0

    built, _ = featurise.build([cfg], [tmp_path])
    assert built.iloc[0]["z_site_A"] == pytest.approx(
        featurise.featurise_atoms(pre)["z_site_A"]
    )


def test_dataset_geometry_path_column_is_not_used(geometry_tree, tmp_path):
    """dataset.csv points at relaxed output; featurise must ignore it."""
    dataset = tmp_path / "dataset.csv"
    relaxed = geometry_tree["root"] / "job" / "results" / f"{geometry_tree['config_id']}_final.traj"
    pd.DataFrame([{
        "config_id": geometry_tree["config_id"], "role": "train",
        "geometry_path": str(relaxed),
    }]).to_csv(dataset, index=False)

    out = tmp_path / "features.csv"
    proc = subprocess.run(
        [sys.executable, "src/featurise.py",
         "--dataset", str(dataset), "--root", str(geometry_tree["root"]),
         "--roots-file", "", "--manifest", "", "--out", str(out),
         "--dictionary-out", str(tmp_path / "d.csv"),
         "--dictionary-md", str(tmp_path / "d.md")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    built = pd.read_csv(out)
    assert not featurise.is_relaxed_artefact(built.iloc[0]["geometry_file"])
    assert built.iloc[0]["geometry_source"] == "built_input"


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------

def test_binding_atom_is_the_oxygen_under_the_hydrogen():
    atoms = _slab()
    assert featurise.binding_atoms(atoms) == [4]


def test_bridging_oxygen_has_coordination_two():
    atoms = _slab()
    cn = featurise.coordination_numbers(atoms)
    assert cn[4] == pytest.approx(2.0)
    assert featurise.site_type_from_cn(cn[4]) == "O_bridge"


def test_site_type_classification():
    assert featurise.site_type_from_cn(1) == "O_terminal"
    assert featurise.site_type_from_cn(2) == "O_bridge"
    assert featurise.site_type_from_cn(3) == "O_3f"
    assert featurise.site_type_from_cn(5) == "O_high"
    assert featurise.site_type_from_cn(float("nan")) == "unknown"


def test_gcn_uses_the_documented_normaliser():
    atoms = _slab()
    cn = featurise.coordination_numbers(atoms)
    gcn = featurise.generalised_cn(atoms, cn, 4)
    neighbours = [0, 1]  # the two Ir under the bridging O
    expected = sum(cn[j] for j in neighbours) / featurise.CN_MAX_REF
    assert gcn == pytest.approx(expected)


def test_hydrogen_is_excluded_from_coordination():
    atoms = _slab(oh_distance=1.0)
    cn = featurise.coordination_numbers(atoms)
    close = _slab(oh_distance=0.8)
    assert cn[4] == pytest.approx(featurise.coordination_numbers(close)[4])


def test_two_adsorbate_pair_geometry():
    atoms = _slab()
    ox, oy, oz = atoms.positions[5]
    atoms.append(Atoms("H", positions=[(ox, oy, oz + 1.0)])[0])

    row = featurise.featurise_atoms(atoms)
    assert row["n_adsorbates"] == 2
    assert row["is_2H"] == 1
    expected = float(np.linalg.norm(atoms.positions[5] - atoms.positions[4]))
    assert row["pair_separation_A"] == pytest.approx(expected)
    assert row["pair_hh_separation_A"] > 0
    assert -1.0 <= row["pair_cos_theta_z"] <= 1.0


def test_single_adsorbate_pair_features_are_zero_with_indicator():
    row = featurise.featurise_atoms(_slab())
    assert row["is_2H"] == 0
    for key in ("pair_separation_A", "pair_hh_separation_A",
                "pair_cos_theta_z", "pair_inplane_angle_deg", "pair_same_row"):
        assert row[key] == 0


def test_coadsorbed_water_is_not_counted_as_adsorbed_hydrogen():
    """The hydrated variant is H* + H2O; only the H* is an adsorbate."""
    atoms = _slab()                       # one H* on the bridging O at index 4
    # Add a water molecule floating above the surface, bonded to no Ir.
    water_o = (3.6, 0.0, 8.0)
    atoms += Atoms("O", positions=[water_o])
    atoms += Atoms("H", positions=[(water_o[0] + 0.76, water_o[1], water_o[2] + 0.59)])
    atoms += Atoms("H", positions=[(water_o[0] - 0.76, water_o[1], water_o[2] + 0.59)])

    assert (atoms.get_chemical_symbols().count("H")) == 3
    row = featurise.featurise_atoms(atoms)
    assert row["n_adsorbates"] == 1        # not 3
    assert row["n_water"] == 1
    assert row["is_2H"] == 0

    # The water oxygen is not a lattice oxygen.
    mask = featurise.lattice_oxygen_mask(atoms)
    assert mask[4]
    assert not mask[8]


def test_water_hydrogens_do_not_pollute_the_oh_distance():
    plain = featurise.featurise_atoms(_slab())
    atoms = _slab()
    water_o = (3.6, 0.0, 8.0)
    atoms += Atoms("O", positions=[water_o])
    atoms += Atoms("H", positions=[(water_o[0] + 0.76, water_o[1], water_o[2] + 0.59)])
    hydrated = featurise.featurise_atoms(atoms)
    assert hydrated["d_OH_mean_A"] == pytest.approx(plain["d_OH_mean_A"])


def test_clean_slab_has_no_binding_site():
    atoms = _slab()
    del atoms[[a.index for a in atoms if a.symbol == "H"]]
    row = featurise.featurise_atoms(atoms)
    assert row["n_adsorbates"] == 0
    assert row["cn_mean"] == 0.0


def test_every_feature_is_finite():
    row = featurise.featurise_atoms(_slab())
    for name in featurise.feature_columns():
        assert np.isfinite(row[name]), name


# --------------------------------------------------------------------------
# dictionary
# --------------------------------------------------------------------------

def test_dictionary_covers_every_emitted_feature(geometry_tree):
    frame, _ = featurise.build([geometry_tree["config_id"]], [geometry_tree["root"]])
    emitted = set(frame.columns) - {"config_id", *featurise.METADATA_COLUMNS}
    assert emitted == set(featurise.feature_columns())


def test_dictionary_entries_have_definition_and_units():
    for name, definition, units in featurise.FEATURE_DICTIONARY:
        assert name and definition and units
        assert len(definition) > 10, name


def test_dictionary_markdown_is_a_table():
    md = featurise.dictionary_markdown()
    assert md.startswith("| feature | definition | units |")
    for name, _, _ in featurise.FEATURE_DICTIONARY:
        assert f"`{name}`" in md


# --------------------------------------------------------------------------
# reproducibility, in one command
# --------------------------------------------------------------------------

def test_build_is_reproducible_in_one_command(geometry_tree, tmp_path):
    def run(tag: str) -> str:
        out = tmp_path / f"features_{tag}.csv"
        dataset = tmp_path / "dataset.csv"
        pd.DataFrame([{"config_id": geometry_tree["config_id"], "role": "train"}]
                     ).to_csv(dataset, index=False)
        proc = subprocess.run(
            [sys.executable, "src/featurise.py",
             "--dataset", str(dataset), "--root", str(geometry_tree["root"]),
             "--roots-file", "", "--manifest", "", "--out", str(out),
             "--dictionary-out", str(tmp_path / f"d_{tag}.csv"),
             "--dictionary-md", str(tmp_path / f"d_{tag}.md")],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return out.read_text()

    assert run("a") == run("b")


def test_build_is_deterministic_in_process(geometry_tree):
    a, _ = featurise.build([geometry_tree["config_id"]], [geometry_tree["root"]])
    b, _ = featurise.build([geometry_tree["config_id"]], [geometry_tree["root"]])
    pd.testing.assert_frame_equal(a, b)


def test_missing_geometry_is_skipped_not_invented(geometry_tree):
    frame, skipped = featurise.build(
        [geometry_tree["config_id"], "slab_H_o404_ready"], [geometry_tree["root"]]
    )
    assert len(frame) == 1
    assert skipped[0]["config_id"] == "slab_H_o404_ready"
    assert "no pre-relaxation geometry" in skipped[0]["reason"]
