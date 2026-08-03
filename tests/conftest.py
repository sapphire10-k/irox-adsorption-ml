"""Fixtures: a synthetic run tree plus a synthetic triage report.

The fake data deliberately reproduces the awkward cases in the real project:

* a configuration triaged OK          -> training data
* a configuration triaged RESTART_*   -> unconverged, candidate pool
* a configuration triaged PENDING     -> never optimised, candidate pool
* a configuration with a converged run but NO triage record (the o20 case)
* a configuration triaged RELABEL_SITE, where site_final != site_intended
* a two-adsorbate pair built but never run (metadata only)
* a configuration with several run folders, disambiguated by triage job_dir
* an untriaged configuration with several run folders (ambiguous)
* bare directory-level metadata that is not a configuration at all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_HASH = "abc123def456"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _result(base_name: str, energy: float, converged: bool = True) -> dict:
    return {
        "base_name": base_name,
        "method": "GFN2-xTB",
        "e_final": energy,
        "fmax_final": 0.04 if converged else 0.28,
        "converged": converged,
        "n_steps": 150 if converged else 200,
        "structure_file": f"outputs/batch/{base_name}.traj",
        "final_traj": f"outputs/results/{base_name}_final.traj",
    }


@pytest.fixture
def fake_project(tmp_path: Path) -> dict[str, Path]:
    """Build a fake run tree + triage report. Returns paths to both."""
    runs = tmp_path / "runs"
    triage_out = tmp_path / "triage_out"
    jobs = triage_out / "jobs"

    # ---- run folders -------------------------------------------------------
    # o21: triaged OK, two run folders; triage points at run_b.
    _write_json(runs / "run_a" / "results" / "slab_H_o21_ready_results.json",
                _result("slab_H_o21_ready", -10380.0))
    _write_json(runs / "run_b" / "results" / "slab_H_o21_ready_results.json",
                _result("slab_H_o21_ready", -10387.5))

    # o22: triaged RESTART_FIRE, unconverged.
    _write_json(runs / "run_b" / "results" / "slab_H_o22_ready_results.json",
                _result("slab_H_o22_ready", -10391.1, converged=False))

    # o44: triaged RELABEL_SITE -- H settled on o45.
    _write_json(runs / "run_b" / "results" / "slab_H_o44_ready_results.json",
                _result("slab_H_o44_ready", -10389.2))

    # o20: converged run, but NO triage record anywhere. Two run folders, so
    # the energy choice is ambiguous as well as untriaged.
    _write_json(runs / "run_a" / "results" / "slab_H_o20_ready_results.json",
                _result("slab_H_o20_ready", -10386.0))
    _write_json(runs / "run_b" / "results" / "slab_H_o20_ready_results.json",
                _result("slab_H_o20_ready", -10387.6))

    # A geometry file that really exists, for path resolution.
    geom = runs / "run_b" / "results" / "slab_H_o21_ready_final.traj"
    geom.parent.mkdir(parents=True, exist_ok=True)
    geom.write_bytes(b"fake traj")

    # ---- built-but-never-run configurations --------------------------------
    # A 2H pair: metadata + structure, no results.
    _write_json(runs / "pairs" / "o21_o46" / "results"
                / "metadata_slab_2H_o21_o46_ready.json",
                {"base_name": "slab_2H_o21_o46_ready",
                 "occupied_o_indices": [21, 46],
                 "structure_file": "outputs/pairs/o21_o46/slab_2H_o21_o46_ready.traj"})
    pair_traj = runs / "pairs" / "o21_o46" / "structures" / "slab_2H_o21_o46_ready.traj"
    pair_traj.parent.mkdir(parents=True, exist_ok=True)
    pair_traj.write_bytes(b"fake pair traj")

    # A plain pending configuration: metadata only, triaged PENDING.
    _write_json(runs / "pending" / "results"
                / "metadata_slab_H_o88_ready.json",
                {"base_name": "slab_H_o88_ready"})

    # Bare directory-level metadata -- NOT a configuration.
    _write_json(runs / "run_a" / "results" / "metadata_o20.json", {"target_o_index": 20})

    # ---- triage report -----------------------------------------------------
    rows = [
        # job_id, status, intended, final, converged, bin, reason, fmax
        ("slab_H_o21_ready", "RUN", "o21", "o21", True, "OK",
         "converged at intended site", 0.0448),
        ("slab_H_o22_ready", "RUN", "o22", "o22", False, "RESTART_FIRE",
         "oscillation or plateau without progress; switch BFGS to FIRE", 0.1094),
        ("slab_H_o44_ready", "RUN", "o44", "o45", True, "RELABEL_SITE",
         "H settled on a different site than intended", 0.0492),
        ("slab_H_o88_ready", "PENDING", "o88", "", None, "PENDING",
         "structures and metadata exist, never optimised", None),
    ]
    header = ("job_id,status,site_intended,site_final,converged,triage_bin,"
              "triage_reason,fmax_final,config_hash\n")
    body = "".join(
        f'{j},{s},{si},{sf},{"" if c is None else c},{b},"{r}",'
        f'{"" if f is None else f},{CONFIG_HASH}\n'
        for j, s, si, sf, c, b, r, f in rows
    )
    triage_out.mkdir(parents=True, exist_ok=True)
    (triage_out / "triage_summary.csv").write_text(header + body)

    # Per-job records carry job_dir, pinning which run folder was judged.
    _write_json(jobs / "slab_H_o21_ready.json",
                {"job_id": "slab_H_o21_ready",
                 "job_dir": str(runs / "run_b" / "results"),
                 "config_hash": CONFIG_HASH})
    _write_json(jobs / "slab_H_o22_ready.json",
                {"job_id": "slab_H_o22_ready",
                 "job_dir": str(runs / "run_b" / "results"),
                 "config_hash": CONFIG_HASH})
    _write_json(jobs / "slab_H_o44_ready.json",
                {"job_id": "slab_H_o44_ready",
                 "job_dir": str(runs / "run_b" / "results"),
                 "config_hash": CONFIG_HASH})
    _write_json(jobs / "slab_H_o88_ready.json",
                {"job_id": "slab_H_o88_ready",
                 "job_dir": str(runs / "pending" / "results"),
                 "config_hash": CONFIG_HASH})

    return {
        "runs": runs,
        "triage_summary": triage_out / "triage_summary.csv",
        "jobs": jobs,
    }
