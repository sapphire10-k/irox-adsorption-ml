#!/usr/bin/env python3
"""Build data/dataset.csv: one row per adsorption configuration.

Two sources are joined on configuration id:

1. **Run folders** -- the ``*_results.json`` / ``metadata_*.json`` artefacts the
   optimisation stage writes. These supply energies and geometry paths.
2. **The triage tool's summary** (``triage_summary.csv``, plus the per-job
   ``jobs/*.json`` records beside it). These supply the convergence verdict,
   the triage bin, and ``site_final``.

Rules encoded here, all following from the project ground rules:

* **Triage is authoritative on convergence.** Where a configuration has a
  triage record, its verdict wins over whatever the run's own results JSON
  claims. The triage tool exists to characterise runs before they reach the ML
  dataset, so its answer is the one that counts.
* **Untriaged means candidate, never training data.** A configuration with no
  triage record gets ``in_triage=False`` and ``role="candidate"`` even if a
  results JSON reports convergence, because it has not been checked against the
  frozen optimiser config.
* **``site_intended`` is never overwritten.** The ML label is ``site_id``, taken
  from ``site_final`` where triage supplies one; the intended site is carried
  alongside verbatim, and ``site_relabelled`` flags any disagreement.
* **Nothing is dropped.** Pending and unconverged configurations are kept and
  flagged as the active-learning candidate pool.
* ``energy_eV`` is a **total system energy**, not an adsorption energy. Forming
  an adsorption energy needs a converged clean-slab reference at the same level
  of theory; this script does not attempt the conversion.

Usage
-----
    python src/inventory.py --roots-file data/inventory_roots.txt \
        --triage-summary <path>/triage_summary.csv --out data/dataset.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.seeds import RANDOM_SEED, set_seeds

COLUMNS = [
    "config_id",
    "site_id",
    "site_intended",
    "site_relabelled",
    "coverage",
    "n_adsorbates",
    "method",
    "energy_eV",
    "converged",
    "triage_bin",
    "triage_reason",
    "status",
    "in_triage",
    "role",
    "config_hash",
    "fmax_final",
    "n_runs_found",
    "energy_source",
    "geometry_path",
    "provenance",
]

# Only base names following the pipeline's convention are treated as
# configurations; bare directory-level metadata (e.g. "o20") is not a config.
_CONFIG_NAME = re.compile(r"^slab_")

_XTB_PAT = re.compile(r"xtb|gfn", re.IGNORECASE)
_QE_PAT = re.compile(r"espresso|quantum|\bqe\b|pw\.x|pbe", re.IGNORECASE)

# Triage bins whose converged runs are allowed into the training set.
# RELABEL_SITE qualifies because the triage tool's contract is that the final
# site becomes the ML label while the intended site is retained as metadata --
# the run is good, only its label moved. Every other bin (RESTART_*,
# DISCARD_PHYSICAL, SETUP_ERROR, PENDING, NOT_TRIAGED) stays in the candidate
# pool. An allowlist, so a new bin is excluded until someone decides otherwise.
TRAIN_BINS = frozenset({"OK", "RELABEL_SITE"})

# Configurations that are out of scope for this study. Hydrated variants are
# accidental carryover from an earlier tool migration. Their rows are kept so
# the provenance survives, but role="excluded" means nothing downstream may
# touch them: not training, not the candidate pool, not active-learning ranking.
EXCLUDED_NAME_PATTERNS = (re.compile(r"hydrated", re.IGNORECASE),)


def is_excluded(config_id: str) -> bool:
    """True if a configuration is out of scope for the study."""
    return any(p.search(config_id) for p in EXCLUDED_NAME_PATTERNS)


def classify_method(raw: str | None) -> str:
    """Map a free-text method string onto 'xTB', 'QE', or 'unknown'."""
    if not raw:
        return "unknown"
    if _XTB_PAT.search(raw):
        return "xTB"
    if _QE_PAT.search(raw):
        return "QE"
    return "unknown"


def parse_base_name(base_name: str) -> tuple[str, str, int]:
    """Derive (site_id, coverage, n_adsorbates) from a pipeline base name.

        slab_clean_ready           -> ("clean", "clean", 0)
        slab_H_o21_ready           -> ("o21", "1H", 1)
        slab_H_hydrated_o21_ready  -> ("o21", "1H", 1)
        slab_2H_o21_o46_ready      -> ("o21+o46", "2H", 2)
    """
    sites = re.findall(r"o(\d+)", base_name)
    site_id = "+".join(f"o{s}" for s in sites) if sites else "clean"

    if re.search(r"clean", base_name, re.IGNORECASE) and not sites:
        return "clean", "clean", 0

    m = re.search(r"slab_(\d*)H(?:_|$)", base_name)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        if sites and len(sites) != n:
            n = len(sites)
    elif sites:
        n = len(sites)
    else:
        return "unknown", "unknown", 0

    return site_id, f"{n}H", n


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None if unreadable or not an object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _clean(value: Any) -> Any:
    """Normalise pandas/JSON missing values to None."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def load_triage(summary_csv: Path, jobs_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Load triage verdicts keyed by job_id, merging in per-job records.

    The per-job JSON carries ``job_dir``, which pins exactly which run folder
    produced the verdict. Without it, a configuration with several run folders
    is ambiguous.
    """
    triage: dict[str, dict[str, Any]] = {}

    frame = pd.read_csv(summary_csv)
    if "job_id" not in frame.columns:
        raise ValueError(f"{summary_csv} has no 'job_id' column")

    for record in frame.to_dict(orient="records"):
        job_id = str(record["job_id"])
        triage[job_id] = {k: _clean(v) for k, v in record.items()}

    if jobs_dir is None:
        jobs_dir = summary_csv.parent / "jobs"
    if jobs_dir.is_dir():
        for path in sorted(jobs_dir.glob("*.json")):
            record = load_json(path)
            if record is None:
                continue
            job_id = str(record.get("job_id") or path.stem)
            merged = triage.setdefault(job_id, {"job_id": job_id})
            for key, value in record.items():
                # The summary CSV is the primary source; per-job records fill
                # gaps (notably job_dir) without overriding it.
                if merged.get(key) is None:
                    merged[key] = _clean(value)

    return triage


def index_runs(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Map config_id -> list of run records found under roots."""
    runs: dict[str, list[dict[str, Any]]] = {}
    seen: set[Path] = set()

    for root in roots:
        for path in sorted(root.rglob("*_results.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            result = load_json(path)
            if result is None:
                continue

            base = str(result.get("base_name") or path.stem.replace("_results", ""))
            if not _CONFIG_NAME.match(base):
                continue

            energy = result.get("e_final", result.get("e_init"))
            runs.setdefault(base, []).append(
                {
                    "path": resolved,
                    "dir": resolved.parent,
                    "energy": float(energy) if isinstance(energy, (int, float)) else None,
                    "converged": bool(result.get("converged", False)),
                    "fmax_final": _clean(result.get("fmax_final")),
                    "method": classify_method(result.get("method")),
                    "result": result,
                }
            )
    return runs


def index_metadata(roots: list[Path]) -> dict[str, Path]:
    """Map config_id -> a metadata file, for configurations that were built."""
    metas: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.rglob("metadata_*.json")):
            data = load_json(path)
            base = None
            if data is not None:
                base = data.get("base_name")
            if not base:
                base = path.stem[len("metadata_") :]
            base = str(base)
            if _CONFIG_NAME.match(base):
                metas.setdefault(base, path)
    return metas


def resolve_geometry(run: dict[str, Any] | None, roots: list[Path],
                     meta_path: Path | None) -> str:
    """Best-effort geometry path: the relaxed structure, else the input structure."""
    if run is not None:
        result = run["result"]
        for key in ("final_traj", "structure_file", "input_structure", "final_xyz"):
            cand = result.get(key)
            if not isinstance(cand, str) or not cand:
                continue
            direct = Path(cand)
            if direct.exists():
                return str(direct.resolve())
            sibling = run["dir"] / direct.name
            if sibling.exists():
                return str(sibling.resolve())
            for root in roots:
                hits = sorted(root.rglob(direct.name), key=lambda p: len(str(p)))
                if hits:
                    return str(hits[0].resolve())

    # Pending configurations: fall back to the built .traj beside the metadata.
    if meta_path is not None:
        data = load_json(meta_path) or {}
        cand = data.get("structure_file")
        name = Path(cand).name if isinstance(cand, str) and cand else None
        if name:
            for root in roots:
                hits = sorted(root.rglob(name), key=lambda p: len(str(p)))
                if hits:
                    return str(hits[0].resolve())
    return ""


def pick_run(config_id: str, runs: list[dict[str, Any]],
             triage: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    """Choose which run supplies the energy, and say how the choice was made."""
    with_energy = [r for r in runs if r["energy"] is not None]
    if not with_energy:
        return None, "none"

    # Preferred: the exact run folder the triage record points at.
    job_dir = triage.get("job_dir") if triage else None
    if job_dir:
        target = Path(str(job_dir)).resolve()
        for run in with_energy:
            if run["dir"] == target or target in run["dir"].parents \
                    or run["dir"] in target.parents:
                return run, "triage_job_dir"

    if len(with_energy) == 1:
        return with_energy[0], "sole_run"

    # Several runs and no triage pointer. Deterministic: converged first, then
    # lowest energy, then path. Flagged so the ambiguity stays visible.
    ordered = sorted(
        with_energy, key=lambda r: (not r["converged"], r["energy"], str(r["path"]))
    )
    return ordered[0], "ambiguous_multi_run"


def build(roots: list[Path], triage: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Join run folders and triage verdicts into one row per configuration."""
    runs = index_runs(roots)
    metas = index_metadata(roots)

    config_ids = sorted(set(runs) | set(metas) | set(triage))
    records: list[dict[str, Any]] = []

    for config_id in config_ids:
        if not _CONFIG_NAME.match(config_id):
            continue

        tri = triage.get(config_id)
        in_triage = tri is not None
        tri = tri or {}

        cfg_runs = runs.get(config_id, [])
        run, energy_source = pick_run(config_id, cfg_runs, tri if in_triage else None)

        parsed_site, coverage, n_ads = parse_base_name(config_id)

        # site_intended is preserved verbatim; the ML label follows site_final.
        site_intended = _clean(tri.get("site_intended")) or parsed_site
        site_final = _clean(tri.get("site_final"))
        site_id = site_final or site_intended
        site_relabelled = bool(site_final and site_final != site_intended)

        # Triage is authoritative on convergence where it has an opinion.
        if in_triage:
            converged = _clean(tri.get("converged"))
            converged = bool(converged) if converged is not None else False
            triage_bin = _clean(tri.get("triage_bin")) or "UNKNOWN"
            status = _clean(tri.get("status")) or "UNKNOWN"
        else:
            # Untriaged: report the run's own convergence flag honestly. The
            # role rule below is what keeps the row out of the training set.
            converged = bool(run["converged"]) if run is not None else False
            triage_bin = "NOT_TRIAGED"
            status = "RUN" if run is not None else "PENDING"

        # A converged clean slab is the E_clean reference. It is never training
        # data and never a candidate for optimisation: it is the zero point.
        if is_excluded(config_id):
            role = "excluded"
        elif coverage == "clean" and converged:
            role = "reference"
        elif in_triage and triage_bin in TRAIN_BINS and converged:
            role = "train"
        else:
            role = "candidate"

        provenance_bits = []
        if run is not None:
            provenance_bits.append(f"run={run['path']}")
        if in_triage:
            provenance_bits.append(f"triage={config_id}")
        if not provenance_bits:
            provenance_bits.append(f"metadata={metas.get(config_id)}")

        records.append(
            {
                "config_id": config_id,
                "site_id": site_id,
                "site_intended": site_intended,
                "site_relabelled": site_relabelled,
                "coverage": coverage,
                "n_adsorbates": n_ads,
                "method": run["method"] if run is not None else "",
                "energy_eV": run["energy"] if run is not None else np.nan,
                "converged": converged,
                "triage_bin": triage_bin,
                "triage_reason": _clean(tri.get("triage_reason")) or "",
                "status": status,
                "in_triage": in_triage,
                "role": role,
                "config_hash": _clean(tri.get("config_hash")) or "",
                "fmax_final": _clean(tri.get("fmax_final"))
                if in_triage else (run["fmax_final"] if run is not None else None),
                "n_runs_found": len(cfg_runs),
                "energy_source": energy_source,
                "geometry_path": resolve_geometry(run, roots, metas.get(config_id)),
                "provenance": "; ".join(str(b) for b in provenance_bits),
            }
        )

    return pd.DataFrame(records, columns=COLUMNS).sort_values(
        "config_id", kind="mergesort"
    ).reset_index(drop=True)


def summarise(df: pd.DataFrame, roots: list[Path], summary_csv: Path) -> str:
    """Render the summary. Every number here is computed from df."""
    lines: list[str] = []
    add = lines.append

    add("=" * 70)
    add("IrO2(110) H adsorption -- configuration inventory")
    add("=" * 70)
    add(f"seed              : {RANDOM_SEED}")
    add(f"triage summary    : {summary_csv}")
    add("run folder roots  :")
    for root in roots:
        add(f"    {root}")
    add("")

    if df.empty:
        add("No configurations found.")
        return "\n".join(lines)

    train = df[df["role"] == "train"]
    cand = df[df["role"] == "candidate"]

    add(f"configurations (rows)          : {len(df)}")
    add(f"  training set (converged)     : {len(train)}")
    add(f"  reference rows               : {int((df['role'] == 'reference').sum())}")
    add(f"  candidate pool               : {len(cand)}")
    add("")

    add("coverage breakdown (all rows):")
    for cov, n in df["coverage"].value_counts().sort_index().items():
        n_train = int((train["coverage"] == cov).sum())
        add(f"    {cov:<8s} {n:>3d}   (train: {n_train})")
    add("")

    add("triage bin breakdown:")
    for bin_name, n in df["triage_bin"].value_counts().items():
        add(f"    {bin_name:<20s} {n}")
    add("")

    add("status breakdown:")
    for status, n in df["status"].value_counts().items():
        add(f"    {status:<20s} {n}")
    add("")

    add("method breakdown (rows with an energy):")
    with_e = df[df["energy_eV"].notna()]
    for method, n in with_e["method"].value_counts().items():
        add(f"    {method:<20s} {n}")
    add(f"    (rows with no energy yet : {int(df['energy_eV'].isna().sum())})")
    add("")

    add(f"configurations not in triage   : {int((~df['in_triage']).sum())}")
    not_triaged = df[~df["in_triage"]]["config_id"].tolist()
    for cid in not_triaged[:20]:
        add(f"    {cid}")
    if len(not_triaged) > 20:
        add(f"    ... +{len(not_triaged) - 20} more")
    add("")

    ambiguous = df[df["energy_source"] == "ambiguous_multi_run"]
    if not ambiguous.empty:
        add(f"energy chosen from several untriaged runs : {len(ambiguous)}")
        for cid in ambiguous["config_id"]:
            n = int(df.loc[df["config_id"] == cid, "n_runs_found"].iloc[0])
            add(f"    {cid} ({n} runs found)")
        add("")

    if df["site_relabelled"].any():
        add("site relabelled by triage (label follows site_final):")
        for _, row in df[df["site_relabelled"]].iterrows():
            add(f"    {row['config_id']}: intended {row['site_intended']} "
                f"-> final {row['site_id']}")
        add("")

    add("energy range, eV (total system energy, NOT adsorption energy):")
    if with_e.empty:
        add("    no energies available")
    else:
        add(f"    all rows with energy : {with_e['energy_eV'].min():.6f} .. "
            f"{with_e['energy_eV'].max():.6f}")
        if not train.empty:
            te = train["energy_eV"].dropna()
            if not te.empty:
                add(f"    training rows        : {te.min():.6f} .. {te.max():.6f}"
                    f"  (span {te.max() - te.min():.6f})")
    add("")

    add(f"TRAINING POINTS AVAILABLE TODAY : {len(train)}")
    add("=" * 70)
    return "\n".join(lines)


def main() -> None:
    set_seeds()

    ap = argparse.ArgumentParser(description="Inventory IrO2(110) H adsorption configurations.")
    ap.add_argument("--root", action="append", default=[],
                    help="Run folder root. Repeat for several.")
    ap.add_argument("--roots-file",
                    help="File with one root per line ('#' comments ignored).")
    ap.add_argument("--triage-summary", required=True,
                    help="Path to the triage tool's triage_summary.csv.")
    ap.add_argument("--triage-jobs",
                    help="Directory of per-job triage JSON records. "
                         "Defaults to 'jobs/' beside the summary CSV.")
    ap.add_argument("--out", default="data/dataset.csv", help="Output CSV path.")
    ap.add_argument("--quiet", action="store_true", help="Suppress the summary.")
    args = ap.parse_args()

    raw_roots = list(args.root)
    if args.roots_file:
        for line in Path(args.roots_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw_roots.append(line)
    if not raw_roots:
        ap.error("no roots given: pass --root and/or --roots-file")

    roots = []
    for raw in raw_roots:
        root = Path(raw).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"root not found: {root}")
        roots.append(root)

    summary_csv = Path(args.triage_summary).expanduser()
    if not summary_csv.exists():
        raise FileNotFoundError(f"triage summary not found: {summary_csv}")
    jobs_dir = Path(args.triage_jobs).expanduser() if args.triage_jobs else None

    triage = load_triage(summary_csv, jobs_dir)
    df = build(roots, triage)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    if not args.quiet:
        print(summarise(df, roots, summary_csv))
        print(f"\nwrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
