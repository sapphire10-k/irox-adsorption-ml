#!/usr/bin/env python3
"""Compute and cache the hydrogen reference E_ref_H = 1/2 E(H2) at GFN2-xTB.

The target definition for this project is

    E_ads = E(slab + nH) - E_clean - n * E_ref_H,   E_ref_H = 1/2 E(H2)

so E_ref_H must be produced by a script under the same frozen settings as every
other energy, not typed in by hand. The molecule is relaxed in a large cell with
the same calculator and the same force target as the slab runs.

The result is cached to JSON. Re-running reads the cache unless --force is
given, so downstream scripts are cheap and deterministic.

Requires the ``xtb`` Python calculator. If it is not installed this script
fails loudly rather than substituting a literature value.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from src.seeds import RANDOM_SEED, set_seeds

# Frozen settings. These mirror the slab optimisations.
METHOD = "GFN2-xTB"
FMAX = 0.05
MAX_STEPS = 200
CELL_A = 20.0          # cubic cell edge, Angstrom -- large enough to isolate H2
INITIAL_HH_DISTANCE = 0.74  # Angstrom, near the known equilibrium

DEFAULT_CACHE = Path("data/h2_reference.json")


def compute_h2_reference(fmax: float = FMAX, cell_a: float = CELL_A,
                         max_steps: int = MAX_STEPS) -> dict:
    """Relax H2 in a large cell at GFN2-xTB and return the reference record."""
    # Imported here so that --help and the cache path work without xtb present.
    from ase import Atoms
    from ase.optimize import BFGS

    try:
        from xtb.ase.calculator import XTB
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ImportError(
            "The xtb Python calculator is not installed, so E_ref_H cannot be "
            "computed under the frozen config. Install it (conda install -c "
            "conda-forge xtb-python) and re-run. This script will not "
            "substitute a literature value."
        ) from exc

    import ase

    atoms = Atoms(
        "H2",
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, INITIAL_HH_DISTANCE)],
        cell=(cell_a, cell_a, cell_a),
        pbc=False,
    )
    atoms.center()
    atoms.calc = XTB(method=METHOD)

    opt = BFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))

    e_h2 = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(((forces ** 2).sum(axis=1) ** 0.5).max())
    d_final = float(atoms.get_distance(0, 1))

    if not converged:
        raise RuntimeError(
            f"H2 relaxation did not reach fmax={fmax} in {max_steps} steps "
            f"(fmax_final={fmax_final:.4f}). Refusing to cache an unconverged "
            "reference."
        )

    return {
        "E_H2_eV": e_h2,
        "E_ref_H_eV": e_h2 / 2.0,
        "method": METHOD,
        "fmax_target": fmax,
        "fmax_final": fmax_final,
        "converged": converged,
        "n_steps": int(opt.get_number_of_steps()),
        "cell_a_A": cell_a,
        "hh_distance_A": d_final,
        "initial_hh_distance_A": INITIAL_HH_DISTANCE,
        "pbc": False,
        "seed": RANDOM_SEED,
        "ase_version": ase.__version__,
        "python": platform.python_version(),
    }


def load_or_compute(cache: Path = DEFAULT_CACHE, force: bool = False,
                    **kwargs) -> dict:
    """Return the cached reference, computing and caching it if needed."""
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    record = compute_h2_reference(**kwargs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache JSON path.")
    ap.add_argument("--force", action="store_true", help="Recompute even if cached.")
    ap.add_argument("--fmax", type=float, default=FMAX)
    ap.add_argument("--cell", type=float, default=CELL_A)
    args = ap.parse_args()

    record = load_or_compute(Path(args.cache), force=args.force,
                             fmax=args.fmax, cell_a=args.cell)

    print(f"E(H2)    = {record['E_H2_eV']:.6f} eV")
    print(f"E_ref_H  = {record['E_ref_H_eV']:.6f} eV   (= 1/2 E(H2))")
    print(f"method   = {record['method']}, fmax_final = {record['fmax_final']:.5f}")
    print(f"d(H-H)   = {record['hh_distance_A']:.4f} A")
    print(f"cached   -> {args.cache}")


if __name__ == "__main__":
    main()
