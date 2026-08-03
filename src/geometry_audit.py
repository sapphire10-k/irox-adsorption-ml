#!/usr/bin/env python3
"""Audit the input slab geometry against reference rutile IrO2.

Every descriptor in this project is read off the slabs produced by the upstream
pipeline. This script measures those slabs and compares them with a rutile IrO2
crystal built from published lattice parameters, so that any structural
disagreement is a recorded number rather than an impression.

It needs the real geometry files, so it is run locally and its output
(``results/geometry_audit.csv``) is committed. ``src/build_readme.py`` reads that
CSV, which is why the README can be regenerated in CI without the run folders.

This script only measures. It does not modify any structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.spacegroup import crystal

from src.featurise import find_pre_relaxation_geometry, load_geometry
from src.seeds import RANDOM_SEED, set_seeds

# Rutile IrO2, space group P4_2/mnm (136). Lattice parameters and the oxygen
# internal coordinate u from the crystallographic literature.
RUTILE_A = 4.4983
RUTILE_C = 3.1544
RUTILE_U = 0.3058

# Any O-O contact below this is chemically implausible in an oxide: the O-O bond
# in a peroxide is about 1.49 A, and the shortest O-O in rutile IrO2 is 2.47 A.
OO_IMPLAUSIBLE_A = 2.0


def reference_rutile() -> Atoms:
    """Rutile IrO2 from published parameters."""
    return crystal(
        ("Ir", "O"),
        basis=[(0.0, 0.0, 0.0), (RUTILE_U, RUTILE_U, 0.0)],
        spacegroup=136,
        cellpar=[RUTILE_A, RUTILE_A, RUTILE_C, 90, 90, 90],
    )


def measure(atoms: Atoms, periodic: bool) -> dict[str, float]:
    """Ir-O and O-O statistics for one structure."""
    symbols = np.array(atoms.get_chemical_symbols())
    ir = np.where(symbols == "Ir")[0]
    ox = np.where(symbols == "O")[0]

    if periodic:
        full = atoms.get_all_distances(mic=True)
    else:
        pos = atoms.get_positions()
        full = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    d_iro = full[np.ix_(ir, ox)]
    d_oo = full[np.ix_(ox, ox)]
    oo_offdiag = d_oo[~np.eye(len(ox), dtype=bool)]

    n_close_pairs = int((oo_offdiag < OO_IMPLAUSIBLE_A).sum() // 2)

    return {
        "n_Ir": len(ir),
        "n_O": len(ox),
        "O_per_Ir": len(ox) / len(ir) if len(ir) else float("nan"),
        "ir_o_shortest_A": float(d_iro.min()),
        "ir_o_shell_spread_A": float(
            np.sort(d_iro.ravel())[:len(ir) * 4].max() - d_iro.min()
        ),
        "o_o_shortest_A": float(oo_offdiag.min()),
        "n_oo_pairs_below_2A": n_close_pairs,
    }


def audit(config_id: str, roots: list[Path]) -> pd.DataFrame:
    """Compare one project structure with reference rutile."""
    located = find_pre_relaxation_geometry(config_id, roots)
    if located is None:
        raise FileNotFoundError(
            f"no pre-relaxation geometry found for {config_id} under {roots}"
        )
    path, source, frame = located
    slab = load_geometry(path, frame)

    rows = [
        {"structure": "reference_rutile_IrO2", "source": "ase.spacegroup.crystal",
         "detail": f"a={RUTILE_A}, c={RUTILE_C}, u={RUTILE_U}",
         **measure(reference_rutile(), periodic=True)},
        {"structure": config_id, "source": source, "detail": str(path),
         **measure(slab, periodic=False)},
    ]
    frame_out = pd.DataFrame(rows)
    frame_out["oo_implausible_threshold_A"] = OO_IMPLAUSIBLE_A
    frame_out["seed"] = RANDOM_SEED
    return frame_out


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-id", default="slab_clean_ready",
                    help="Structure to audit. The clean slab by default, so the "
                         "result is a property of the lattice, not of an adsorbate.")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--roots-file", default="data/inventory_roots.txt")
    ap.add_argument("--out", default="results/geometry_audit.csv")
    args = ap.parse_args()

    raw = list(args.root)
    if args.roots_file:
        for line in Path(args.roots_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.append(line)
    roots = [Path(r).expanduser() for r in raw]
    missing = [r for r in roots if not r.exists()]
    if missing:
        raise FileNotFoundError(f"root(s) not found: {missing}")

    table = audit(args.config_id, roots)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    ref = table.iloc[0]
    slab = table.iloc[1]

    print("=" * 76)
    print("Geometry audit -- input slab versus reference rutile IrO2")
    print("=" * 76)
    print(f"{'quantity':<32}{'reference':>14}{'this slab':>14}")
    print("-" * 76)
    for key, label in (
        ("ir_o_shortest_A", "shortest Ir-O, A"),
        ("ir_o_shell_spread_A", "Ir-O first-shell spread, A"),
        ("o_o_shortest_A", "shortest O-O, A"),
        ("n_oo_pairs_below_2A", f"O-O pairs < {OO_IMPLAUSIBLE_A} A"),
        ("O_per_Ir", "O per Ir"),
    ):
        print(f"{label:<32}{ref[key]:>14.3f}{slab[key]:>14.3f}")
    print()

    ratio = slab["ir_o_shortest_A"] / ref["ir_o_shortest_A"]
    print(f"Shortest Ir-O is {ratio:.2f}x the rutile value "
          f"({100 * (ratio - 1):+.0f}%).")
    if slab["n_oo_pairs_below_2A"] > 0:
        print(f"{int(slab['n_oo_pairs_below_2A'])} O-O contacts below "
              f"{OO_IMPLAUSIBLE_A} A. The shortest O-O in rutile IrO2 is "
              f"{ref['o_o_shortest_A']:.2f} A;")
        print("an O-O separation near 1.5 A is a peroxide bond length, not an "
              "oxide lattice spacing.")
    print()
    print("This script measures only. Nothing is corrected here.")
    print(f"wrote {out}")
    print("=" * 76)


if __name__ == "__main__":
    main()
