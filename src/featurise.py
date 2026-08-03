#!/usr/bin/env python3
"""Build data/features.csv from PRE-RELAXATION geometries only.

Why pre-relaxation only
-----------------------
The point of the surrogate is to rank configurations that have *not* been
optimised yet. Candidates in the pool have no relaxed geometry at all, so any
descriptor read off a relaxed structure would be unavailable at prediction time
and would leak the answer into the training set. This module therefore refuses
to open relaxed artefacts. Concretely:

* ``*_final.traj``, ``*_final.xyz``, ``*_relaxed.xyz`` are on a hard denylist.
* An optimisation trajectory ``*_opt.traj`` may be used, but only frame 0,
  which is the input geometry by construction.
* The preferred source is the built input under ``structures/``.

``data/dataset.csv`` is used for the configuration list and bookkeeping only.
Its ``geometry_path`` column points at relaxed output for completed runs, so it
is deliberately NOT used to locate geometry here.

Coordination conventions
------------------------
Coordination is counted over Ir-O pairs within a fixed cutoff. The as-built
slab has a uniform Ir-O first shell at 2.402 A and a clean gap to the next
shell at 3.2 A, so the cutoff is insensitive anywhere in ~2.5-3.1 A.

The generalised coordination number follows Calle-Vallejo:

    GCN(i) = sum_{j in nn(i)} CN(j) / CN_max

with ``CN_max`` the ideal octahedral Ir coordination in rutile (6). Raw CN is
emitted alongside so nothing is hidden behind the normalisation.

Caveat carried into the feature dictionary: these slabs are non-periodic
(``pbc=False``), so atoms near the lateral edges have genuinely lower
coordination than an interior site would. ``lateral_margin_A`` is emitted so
that edge effects are visible rather than silently baked into CN and GCN.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from src.seeds import RANDOM_SEED, set_seeds

# --- conventions, all documented in the feature dictionary ----------------

CN_CUTOFF_A = 2.6      # Ir-O first-shell cutoff
CN_MAX_REF = 6.0       # ideal rutile Ir octahedral coordination (GCN normaliser)
H_BOND_MAX_A = 1.3     # max H-to-heavy distance for binding-atom assignment

# Filenames that hold relaxed geometry. Never opened by this module.
RELAXED_PATTERNS = (
    re.compile(r"_final\.(traj|xyz)$", re.IGNORECASE),
    re.compile(r"_relaxed\.(traj|xyz)$", re.IGNORECASE),
)


class LeakageError(RuntimeError):
    """Raised when a relaxed geometry is about to be used as a feature source."""


def is_relaxed_artefact(path: Path | str) -> bool:
    """True if the filename identifies a post-relaxation structure."""
    name = Path(path).name
    return any(p.search(name) for p in RELAXED_PATTERNS)


def assert_pre_relaxation(path: Path) -> None:
    """Refuse relaxed artefacts, loudly."""
    if is_relaxed_artefact(path):
        raise LeakageError(
            f"{path} is a relaxed geometry. Features must come from the "
            "pre-relaxation structure, because candidates have no relaxed "
            "geometry and using one would leak the optimisation result."
        )


# --- locating the pre-relaxation geometry ---------------------------------

def find_pre_relaxation_geometry(config_id: str, roots: list[Path]
                                 ) -> tuple[Path, str, int] | None:
    """Locate the input geometry for a configuration.

    Returns (path, source_kind, frame_index) or None. Preference order:

    1. ``**/structures/<config_id>.traj``  -- the built input
    2. the ``structure_file`` named in ``metadata_<config_id>.json``
    3. ``**/<config_id>_opt.traj`` frame 0 -- the optimiser's starting frame
    """
    # 1. built input under structures/
    for root in roots:
        hits = sorted(root.rglob(f"structures/{config_id}.traj"),
                      key=lambda p: len(str(p)))
        for hit in hits:
            if not is_relaxed_artefact(hit):
                return hit, "built_input", 0

    # 2. structure_file recorded in metadata
    for root in roots:
        for meta_path in sorted(root.rglob(f"metadata_{config_id}.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            named = meta.get("structure_file")
            if not isinstance(named, str) or not named:
                continue
            base = Path(named).name
            if is_relaxed_artefact(base):
                continue
            for r2 in roots:
                found = sorted(r2.rglob(base), key=lambda p: len(str(p)))
                if found:
                    return found[0], "metadata_structure_file", 0

    # 3. first frame of the optimisation trajectory
    for root in roots:
        hits = sorted(root.rglob(f"{config_id}_opt.traj"), key=lambda p: len(str(p)))
        if hits:
            return hits[0], "opt_traj_frame0", 0

    return None


def load_geometry(path: Path, frame: int = 0) -> Atoms:
    """Read a pre-relaxation geometry. Refuses relaxed files and negative frames."""
    assert_pre_relaxation(path)
    if frame != 0:
        raise LeakageError(
            f"refusing to read frame {frame} of {path}; only frame 0 is the "
            "pre-relaxation geometry"
        )
    return read(str(path), index=0)


# --- geometry helpers ------------------------------------------------------

def lattice_oxygen_mask(atoms: Atoms, cutoff: float = CN_CUTOFF_A) -> np.ndarray:
    """True for oxygens belonging to the slab lattice, i.e. bonded to Ir.

    A co-adsorbed water molecule contributes an oxygen with no Ir neighbour.
    Its hydrogens are not H* adsorbates and must not be counted as such.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    is_ir = symbols == "Ir"

    mask = np.zeros(len(atoms), dtype=bool)
    for i in np.where(symbols == "O")[0]:
        d = np.linalg.norm(positions - positions[i], axis=1)
        mask[i] = bool(((d > 0) & (d < cutoff) & is_ir).any())
    return mask


def binding_pairs(atoms: Atoms) -> list[tuple[int, int]]:
    """(hydrogen index, lattice atom index) for every H* adsorbate.

    Hydrogens belonging to a co-adsorbed molecule (water) are excluded: they
    bind to an oxygen that is not part of the slab lattice.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    h_idx = np.where(symbols == "H")[0]
    heavy = np.where(symbols != "H")[0]
    lattice = lattice_oxygen_mask(atoms) | (symbols == "Ir")

    pairs: list[tuple[int, int]] = []
    for h in h_idx:
        d = np.linalg.norm(positions[heavy] - positions[h], axis=1)
        j = int(np.argmin(d))
        if d[j] <= H_BOND_MAX_A and lattice[heavy[j]]:
            pairs.append((int(h), int(heavy[j])))
    return pairs


def binding_atoms(atoms: Atoms) -> list[int]:
    """Indices of the lattice atoms carrying an H* adsorbate."""
    return [b for _, b in binding_pairs(atoms)]


def count_water(atoms: Atoms) -> int:
    """Number of co-adsorbed molecular oxygens carrying at least one H."""
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    lattice = lattice_oxygen_mask(atoms)
    h_idx = np.where(symbols == "H")[0]

    count = 0
    for i in np.where(symbols == "O")[0]:
        if lattice[i]:
            continue
        if h_idx.size and (np.linalg.norm(positions[h_idx] - positions[i], axis=1)
                           <= H_BOND_MAX_A).any():
            count += 1
    return count


def coordination_numbers(atoms: Atoms, cutoff: float = CN_CUTOFF_A) -> np.ndarray:
    """Ir-O coordination number for every atom. H is excluded from counting."""
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    n = len(atoms)

    heavy_mask = symbols != "H"
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

    # Only count Ir-O contacts: this is an oxide, Ir-Ir and O-O are not bonds.
    is_ir = symbols == "Ir"
    is_o = symbols == "O"
    pair_ok = (is_ir[:, None] & is_o[None, :]) | (is_o[:, None] & is_ir[None, :])

    close = (d > 0) & (d < cutoff) & pair_ok & heavy_mask[None, :]
    cn = close.sum(axis=1).astype(float)
    cn[~heavy_mask] = np.nan
    return cn


def generalised_cn(atoms: Atoms, cn: np.ndarray, index: int,
                   cutoff: float = CN_CUTOFF_A,
                   cn_max: float = CN_MAX_REF) -> float:
    """GCN(i) = sum over neighbours j of CN(j) / cn_max."""
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    d = np.linalg.norm(positions - positions[index], axis=1)

    is_ir = symbols == "Ir"
    is_o = symbols == "O"
    if symbols[index] == "O":
        partner = is_ir
    elif symbols[index] == "Ir":
        partner = is_o
    else:
        return float("nan")

    neighbours = np.where((d > 0) & (d < cutoff) & partner)[0]
    if neighbours.size == 0:
        return 0.0
    return float(np.nansum(cn[neighbours]) / cn_max)


def site_type_from_cn(cn_value: float) -> str:
    """Classify an oxygen site by its Ir coordination."""
    if not np.isfinite(cn_value):
        return "unknown"
    n = int(round(cn_value))
    if n <= 1:
        return "O_terminal"
    if n == 2:
        return "O_bridge"
    if n == 3:
        return "O_3f"
    return "O_high"


def lateral_margin(atoms: Atoms, index: int) -> float:
    """Distance from an atom to the nearest lateral edge of the slab's extent.

    These slabs are non-periodic, so this is how far a site sits from the
    artificial boundary where coordination starts to be truncated.
    """
    positions = atoms.get_positions()
    x, y = positions[index, 0], positions[index, 1]
    xs, ys = positions[:, 0], positions[:, 1]
    return float(min(x - xs.min(), xs.max() - x, y - ys.min(), ys.max() - y))


# --- the feature table -----------------------------------------------------

def featurise_atoms(atoms: Atoms) -> dict[str, Any]:
    """Compute every descriptor for one pre-relaxation structure."""
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    cn = coordination_numbers(atoms)
    pairs = binding_pairs(atoms)
    bound = [b for _, b in pairs]
    n_ads = len(bound)

    row: dict[str, Any] = {
        "n_adsorbates": n_ads,
        "is_2H": int(n_ads == 2),
        "n_water": count_water(atoms),
        "n_atoms": len(atoms),
    }

    if n_ads == 0:
        # Clean slab: no binding site. Emitted with zeros and n_adsorbates = 0.
        row.update({
            "cn_mean": 0.0, "cn_min": 0.0, "cn_max": 0.0,
            "gcn_mean": 0.0, "gcn_min": 0.0, "gcn_max": 0.0,
            "n_bridge_sites": 0, "n_3f_sites": 0,
            "d_OH_mean_A": 0.0,
            "nn1_dist_A": 0.0, "nn2_dist_A": 0.0, "nn3_dist_A": 0.0,
            "frac_Ir_in_shell": 0.0,
            "z_site_A": 0.0, "z_below_top_A": 0.0, "lateral_margin_A": 0.0,
            "pair_separation_A": 0.0, "pair_hh_separation_A": 0.0,
            "pair_cos_theta_z": 0.0, "pair_inplane_angle_deg": 0.0,
            "pair_same_row": 0,
        })
        return row

    cns = np.array([cn[i] for i in bound], dtype=float)
    gcns = np.array([generalised_cn(atoms, cn, i) for i in bound], dtype=float)
    types = [site_type_from_cn(c) for c in cns]

    row.update({
        "cn_mean": float(cns.mean()),
        "cn_min": float(cns.min()),
        "cn_max": float(cns.max()),
        "gcn_mean": float(gcns.mean()),
        "gcn_min": float(gcns.min()),
        "gcn_max": float(gcns.max()),
        "n_bridge_sites": int(sum(t == "O_bridge" for t in types)),
        "n_3f_sites": int(sum(t == "O_3f" for t in types)),
    })

    # H-to-binding-atom distances, using the matched H* pairs so that a
    # co-adsorbed water's hydrogens never enter the average.
    d_oh = [float(np.linalg.norm(positions[h] - positions[b])) for h, b in pairs]
    row["d_OH_mean_A"] = float(np.mean(d_oh))

    # Neighbour shell of the binding atom(s): distances and element identity
    nn_dists, ir_fraction = [], []
    for b in bound:
        d = np.linalg.norm(positions - positions[b], axis=1)
        order = np.argsort(d)
        heavy_order = [j for j in order if j != b and symbols[j] != "H"]
        nn_dists.append([float(d[j]) for j in heavy_order[:3]])
        shell = [j for j in heavy_order if d[j] < CN_CUTOFF_A]
        ir_fraction.append(
            float(np.mean([symbols[j] == "Ir" for j in shell])) if shell else 0.0
        )
    nn_arr = np.array(nn_dists, dtype=float)
    row["nn1_dist_A"] = float(nn_arr[:, 0].mean())
    row["nn2_dist_A"] = float(nn_arr[:, 1].mean())
    row["nn3_dist_A"] = float(nn_arr[:, 2].mean())
    row["frac_Ir_in_shell"] = float(np.mean(ir_fraction))

    # Height of the site
    z_top = float(positions[symbols != "H", 2].max())
    z_sites = np.array([positions[b, 2] for b in bound], dtype=float)
    row["z_site_A"] = float(z_sites.mean())
    row["z_below_top_A"] = float(z_top - z_sites.mean())
    row["lateral_margin_A"] = float(np.mean([lateral_margin(atoms, b) for b in bound]))

    # Pair geometry -- meaningful only for 2H. Filled with 0.0 otherwise, with
    # is_2H acting as the indicator so the zeros are never mistaken for data.
    if n_ads == 2:
        (h_a, a), (h_b, b) = pairs
        vec = positions[b] - positions[a]
        sep = float(np.linalg.norm(vec))
        row["pair_separation_A"] = sep
        row["pair_hh_separation_A"] = float(
            np.linalg.norm(positions[h_b] - positions[h_a])
        )
        row["pair_cos_theta_z"] = float(vec[2] / sep) if sep > 0 else 0.0
        row["pair_inplane_angle_deg"] = float(
            np.degrees(np.arctan2(abs(vec[1]), abs(vec[0])))
        )
        # Same row along the bridging-oxygen direction: the pair vector lies
        # essentially along one lateral axis.
        row["pair_same_row"] = int(abs(vec[1]) < 0.5 or abs(vec[0]) < 0.5)
    else:
        row.update({
            "pair_separation_A": 0.0, "pair_hh_separation_A": 0.0,
            "pair_cos_theta_z": 0.0, "pair_inplane_angle_deg": 0.0,
            "pair_same_row": 0,
        })

    return row


FEATURE_DICTIONARY: list[tuple[str, str, str]] = [
    ("n_adsorbates", "Number of adsorbed H atoms, from the pre-relaxation geometry", "count"),
    ("is_2H", "1 for two-adsorbate configurations, 0 otherwise; indicator for the pair block", "0/1"),
    ("n_water", "Co-adsorbed water molecules, i.e. oxygens carrying H but bonded to no Ir; their hydrogens are excluded from n_adsorbates", "count"),
    ("n_atoms", "Total atoms in the configuration", "count"),
    ("cn_mean", f"Mean Ir-O coordination number of the binding oxygen(s), cutoff {CN_CUTOFF_A} A", "count"),
    ("cn_min", "Minimum coordination number across binding oxygens", "count"),
    ("cn_max", "Maximum coordination number across binding oxygens", "count"),
    ("gcn_mean", f"Mean generalised coordination number, sum_j CN(j)/{CN_MAX_REF:.0f} (Calle-Vallejo)", "dimensionless"),
    ("gcn_min", "Minimum generalised coordination number across binding oxygens", "dimensionless"),
    ("gcn_max", "Maximum generalised coordination number across binding oxygens", "dimensionless"),
    ("n_bridge_sites", "Binding oxygens with Ir coordination 2 (bridging)", "count"),
    ("n_3f_sites", "Binding oxygens with Ir coordination 3 (three-fold)", "count"),
    ("d_OH_mean_A", "Mean H to binding-oxygen distance in the built geometry", "angstrom"),
    ("nn1_dist_A", "Mean distance from binding oxygen to its 1st nearest heavy atom", "angstrom"),
    ("nn2_dist_A", "Mean distance from binding oxygen to its 2nd nearest heavy atom", "angstrom"),
    ("nn3_dist_A", "Mean distance from binding oxygen to its 3rd nearest heavy atom", "angstrom"),
    ("frac_Ir_in_shell", f"Fraction of atoms within {CN_CUTOFF_A} A of the binding oxygen that are Ir", "fraction"),
    ("z_site_A", "Mean z coordinate of the binding oxygen(s)", "angstrom"),
    ("z_below_top_A", "Depth of the binding oxygen below the topmost slab atom", "angstrom"),
    ("lateral_margin_A", "Distance from the binding site to the nearest lateral edge of the non-periodic slab; low values mean coordination is truncated by the finite cell", "angstrom"),
    ("pair_separation_A", "Distance between the two binding oxygens (2H only, else 0)", "angstrom"),
    ("pair_hh_separation_A", "Distance between the two H atoms (2H only, else 0)", "angstrom"),
    ("pair_cos_theta_z", "Cosine of the angle between the O-O pair vector and the surface normal (2H only, else 0)", "dimensionless"),
    ("pair_inplane_angle_deg", "In-plane angle of the O-O pair vector from the x axis (2H only, else 0)", "degrees"),
    ("pair_same_row", "1 if the pair vector lies along a lateral axis, i.e. same bridging-oxygen row (2H only, else 0)", "0/1"),
]

# Metadata columns emitted alongside the features. These are NEVER features:
# selection_note is free text recording the surface-chemistry reasoning behind a
# site, and nothing downstream may fit on it.
METADATA_COLUMNS = ["selection_note", "geometry_source", "geometry_file"]

# Column holding the free-text note in the orbital manifest spreadsheet.
MANIFEST_SHEET = "Orbital_Manifest"
MANIFEST_NOTE_COLUMN = "selection note"


def load_selection_notes(manifest: Path,
                         sheet: str = MANIFEST_SHEET) -> dict[str, str]:
    """Map config_id -> selection_note from the orbital manifest.

    The manifest is keyed by site and state rather than by configuration id, so
    the mapping is reconstructed here:

        state 'clean'                -> every slab_clean_ready* configuration
        state 'H_star', target_O = N -> slab_H_o{N}_ready
        a site labelled 'Hydrated'   -> slab_H_hydrated_o{N}_ready

    Configurations absent from the manifest simply get no note.
    """
    frame = pd.read_excel(manifest, sheet_name=sheet)
    if MANIFEST_NOTE_COLUMN not in frame.columns:
        raise ValueError(
            f"{manifest} sheet '{sheet}' has no '{MANIFEST_NOTE_COLUMN}' column"
        )

    notes: dict[str, str] = {}
    for _, row in frame.iterrows():
        note = row[MANIFEST_NOTE_COLUMN]
        if not isinstance(note, str) or not note.strip():
            continue
        note = note.strip()

        site = str(row.get("site", "") or "")
        state = str(row.get("state", "") or "")
        target = row.get("target_O_index")

        if "hydrated" in site.lower() or "hydrated" in state.lower():
            found = re.search(r"o?(\d+)", site, re.IGNORECASE)
            if found:
                notes[f"slab_H_hydrated_o{int(found.group(1))}_ready"] = note
            continue

        if state.strip().lower() == "clean" or site.strip().lower() == "clean":
            notes["slab_clean_ready"] = note
            notes["slab_clean_ready_r1"] = note
            continue

        if pd.notna(target):
            notes[f"slab_H_o{int(target)}_ready"] = note

    return notes


def feature_columns() -> list[str]:
    """Feature column names, in dictionary order."""
    return [name for name, _, _ in FEATURE_DICTIONARY]


def build(config_ids: list[str], roots: list[Path],
          selection_notes: dict[str, str] | None = None
          ) -> tuple[pd.DataFrame, list[dict]]:
    """Featurise every configuration. Returns (features, skipped)."""
    rows, skipped = [], []

    for config_id in sorted(config_ids):
        located = find_pre_relaxation_geometry(config_id, roots)
        if located is None:
            skipped.append({"config_id": config_id, "reason": "no pre-relaxation geometry found"})
            continue
        path, source, frame = located
        try:
            atoms = load_geometry(path, frame)
        except (LeakageError, OSError, IndexError) as exc:
            skipped.append({"config_id": config_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        row = {"config_id": config_id}
        row.update(featurise_atoms(atoms))
        # Metadata, never a feature.
        row["selection_note"] = (selection_notes or {}).get(config_id, "")
        row["geometry_source"] = source
        row["geometry_file"] = str(path)
        rows.append(row)

    columns = ["config_id", *feature_columns(), *METADATA_COLUMNS]
    frame_out = pd.DataFrame(rows, columns=columns)
    if not frame_out.empty:
        frame_out = frame_out.sort_values("config_id", kind="mergesort").reset_index(drop=True)
    return frame_out, skipped


def dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DICTIONARY, columns=["name", "definition", "units"])


def dictionary_markdown() -> str:
    """Markdown table for the README. Generated, never hand-edited."""
    lines = ["| feature | definition | units |", "| --- | --- | --- |"]
    for name, definition, units in FEATURE_DICTIONARY:
        lines.append(f"| `{name}` | {definition} | {units} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description="Featurise pre-relaxation geometries.")
    ap.add_argument("--dataset", default="data/dataset.csv",
                    help="Configuration list. geometry_path is NOT used.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--roots-file", default="data/inventory_roots.txt")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--dictionary-out", default="data/feature_dictionary.csv")
    ap.add_argument("--dictionary-md", default="data/feature_dictionary.md")
    ap.add_argument("--manifest",
                    default="/Users/manishamalhotra/Documents/IrO2_Simulation_Data/orbital_manifest.xlsx",
                    help="Orbital manifest spreadsheet supplying selection_note. "
                         "Pass an empty string to skip.")
    ap.add_argument("--role", default=None,
                    help="Only featurise this dataset role (default: all in scope).")
    ap.add_argument("--include-excluded", action="store_true",
                    help="Also featurise role=='excluded' rows. Off by default: "
                         "excluded configurations are out of scope and nothing "
                         "downstream may consume their features.")
    args = ap.parse_args()

    raw_roots = list(args.root)
    if args.roots_file:
        for line in Path(args.roots_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw_roots.append(line)
    roots = [Path(r).expanduser() for r in raw_roots]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise FileNotFoundError(f"root(s) not found: {missing}")
    if not roots:
        ap.error("no roots given")

    dataset = pd.read_csv(args.dataset)

    n_excluded = 0
    if "role" in dataset.columns and not args.include_excluded:
        n_excluded = int((dataset["role"] == "excluded").sum())
        dataset = dataset[dataset["role"] != "excluded"]
    if args.role:
        dataset = dataset[dataset["role"] == args.role]
    config_ids = dataset["config_id"].tolist()

    selection_notes: dict[str, str] = {}
    if args.manifest:
        manifest = Path(args.manifest).expanduser()
        if not manifest.exists():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        selection_notes = load_selection_notes(manifest)

    features, skipped = build(config_ids, roots, selection_notes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    dictionary_frame().to_csv(args.dictionary_out, index=False)
    Path(args.dictionary_md).write_text(dictionary_markdown())

    print("=" * 70)
    print("Featurisation -- PRE-RELAXATION geometries only")
    print("=" * 70)
    print(f"seed                : {RANDOM_SEED}")
    print(f"configurations in   : {len(config_ids)}")
    print(f"featurised          : {len(features)}")
    print(f"skipped             : {len(skipped)}")
    print(f"out of scope        : {n_excluded} (role=excluded, not featurised)")
    print(f"features per config : {len(feature_columns())}")
    print(f"selection notes     : {len(selection_notes)} in manifest, "
          f"{int((features['selection_note'] != '').sum()) if not features.empty else 0} matched (metadata, not a feature)")
    print()
    if not features.empty:
        print("geometry source breakdown:")
        for source, n in features["geometry_source"].value_counts().items():
            print(f"    {source:<28s} {n}")
        print()
        by_role = dataset.set_index("config_id")["role"]
        roles = features["config_id"].map(by_role).value_counts()
        print("by dataset role:")
        for role, n in roles.items():
            print(f"    {role:<28s} {n}")
        print()
        print("coverage:")
        for cov, n in features["n_adsorbates"].value_counts().sort_index().items():
            print(f"    {int(cov)}H{'':<26s} {n}".replace("0H", "clean"))
        print()
    if skipped:
        print("skipped configurations:")
        for item in skipped[:20]:
            print(f"    {item['config_id']}: {item['reason']}")
        if len(skipped) > 20:
            print(f"    ... +{len(skipped) - 20} more")
        print()
    print(f"wrote {out}")
    print(f"wrote {args.dictionary_out}")
    print(f"wrote {args.dictionary_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()
