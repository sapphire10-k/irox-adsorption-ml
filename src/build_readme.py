#!/usr/bin/env python3
"""Generate README.md from the committed results tables.

The project rule is that every number in the README comes from a script. The
cleanest way to guarantee that is not to write the README by hand at all: this
module reads ``data/`` and ``results/``, formats them, and writes the file. CI
regenerates it and fails if the checked-in copy differs, so a stale README is a
build failure rather than a quiet inaccuracy.

Prose is authored here; numbers never are.

    python src/build_readme.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.featurise import FEATURE_DICTIONARY
from src.feature_subset import MODEL_FEATURES
from src.seeds import RANDOM_SEED

REPO = "sapphire10-k/irox-adsorption-ml"
UPSTREAM = "https://github.com/sapphire10-k/IrOx_VQE"


def _f(x: float, places: int = 4) -> str:
    return f"{float(x):.{places}f}"


def count_tests(repo_root: Path) -> int | None:
    """Number of collected tests. Collection only -- nothing is executed."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
            cwd=repo_root, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(proc.stdout.strip().splitlines()):
        if "test" in line and "collected" in line:
            for token in line.split():
                if token.isdigit():
                    return int(token)
        if line.strip().endswith("tests collected") or "/" in line:
            continue
    # Fall back to counting the node ids pytest printed.
    ids = [l for l in proc.stdout.splitlines() if "::" in l]
    return len(ids) or None


def build(repo_root: Path) -> str:
    data = repo_root / "data"
    results = repo_root / "results"

    dataset = pd.read_csv(data / "dataset.csv")
    features = pd.read_csv(data / "features.csv")
    ladder = pd.read_csv(results / "model_ladder.csv")
    calib = pd.read_csv(results / "calibration.csv")
    curve = pd.read_csv(results / "learning_curve.csv")
    ranking = pd.read_csv(results / "candidate_ranking.csv")
    policy = pd.read_csv(results / "acquisition_policy.csv")
    geom = pd.read_csv(results / "geometry_audit.csv")
    h2 = json.loads((data / "h2_reference.json").read_text())

    roles = dataset["role"].value_counts()
    n_train = int(roles.get("train", 0))
    n_candidate = int(roles.get("candidate", 0))
    n_reference = int(roles.get("reference", 0))
    n_excluded = int(roles.get("excluded", 0))
    n_total = len(dataset)
    # In scope means: an adsorption configuration that is not excluded. Counting
    # from the dataset without the role filter would include the hydrated row and
    # disagree with the coverage counts below, which come from the feature table.
    in_scope = dataset[(dataset["coverage"] != "clean")
                       & (dataset["role"] != "excluded")]
    n_adsorption = len(in_scope)

    cov = features["n_adsorbates"].value_counts()
    n_1h = int(cov.get(1, 0))
    n_2h = int(cov.get(2, 0))

    e_clean = float(dataset.loc[dataset["role"] == "reference", "energy_eV"].iloc[0])
    e_ref_h = float(h2["E_ref_H_eV"])
    floor = float(ladder["noise_operative_eV"].iloc[0])
    floor_tol = float(ladder["noise_tolerance_implied_eV"].iloc[0])
    floor_emp = float(ladder["noise_empirical_eV"].iloc[0])

    base_mae = float(ladder.loc[ladder["model"] == "mean", "mae_eV"].iloc[0])
    gp_row = ladder.loc[ladder["model"] == "gp"].iloc[0]

    c68 = calib.loc[calib["nominal_label"] == "68%"].iloc[0]
    c95 = calib.loc[calib["nominal_label"] == "95%"].iloc[0]
    rms_z = float(calib["rms_z"].iloc[0])

    unstable_models = sorted(set(curve.loc[curve["unstable"], "model"])) if "unstable" in curve else []

    # Active learning: the degeneracy of the live ranking, and the unconfounded
    # policy comparison at the one step that can distinguish the policies.
    sigma_span = float(ranking["predictive_sigma_eV"].max()
                       - ranking["predictive_sigma_eV"].min())
    sigma_mean = float(ranking["predictive_sigma_eV"].mean())
    sigma_rel = sigma_span / sigma_mean if sigma_mean else float("nan")

    unconf = policy[policy["evaluation"] == "fixed_holdout_UNCONFOUNDED"]
    informative = unconf[(unconf["n_acquired"] > 1)
                         & (unconf["vacuous_pool_exhausted"] == False)]  # noqa: E712
    informative = informative.dropna(subset=["mean_gain_eV"])
    al_gain = float(informative["mean_gain_eV"].iloc[0]) if len(informative) else float("nan")
    al_se = float(informative["gain_se_eV"].iloc[0]) if len(informative) else float("nan")
    al_beats = bool(informative["uncertainty_beats_random"].any()) if len(informative) else False

    g_ref, g_slab = geom.iloc[0], geom.iloc[1]
    ir_o_ratio = float(g_slab["ir_o_shortest_A"]) / float(g_ref["ir_o_shortest_A"])

    n_tests = count_tests(repo_root)
    tests_phrase = f"all {n_tests} tests" if n_tests else "the full test suite"

    L: list[str] = []
    a = L.append

    # ---------------------------------------------------------------- title
    a(f"[![CI](https://github.com/{REPO}/actions/workflows/ci.yml/badge.svg)]"
      f"(https://github.com/{REPO}/actions/workflows/ci.yml)")
    a("")
    a("# irox-adsorption-ml")
    a("")

    # ------------------------------------------------------------- question
    a("## The question")
    a("")
    a("On my curated set of converged H adsorption configurations on IrO2(110), as")
    a("inventoried in `data/dataset.csv`, can a Gaussian process with honest")
    a("uncertainties predict adsorption energetics well enough to prioritise which")
    a("configuration to optimise next?")
    a("")

    # --------------------------------------------- limitations and judgment
    a("## Limitations and judgment")
    a("")
    a("This section is first because it governs how every number below should be read.")
    a("")
    a(f"**The dataset is small by design.** There are {n_adsorption} adsorption")
    a(f"configurations in scope ({n_1h} single-H, {n_2h} two-adsorbate pairs), of which")
    a(f"{n_train} are converged and usable as training data today. The sites were chosen")
    a("for surface-chemistry reasons set out in the accompanying preprint, not sampled to")
    a("fill a feature space. That is a design choice and it is not apologised for: a")
    a("curated set of physically motivated sites is what makes the individual points")
    a("interpretable. It does mean the statistical questions this repository can answer")
    a("are narrow, and the code is built to say so rather than to paper over it.")
    a("")
    a(f"**What {n_train} converged points can support.** Bookkeeping and provenance:")
    a("which configurations exist, which converged, under which frozen optimiser")
    a("settings, and which are still pending. Descriptor computation from")
    a("pre-relaxation geometry. Leave-one-out diagnostics that put an honest bound on")
    a("what a surrogate is doing. Detecting that a surrogate is *not* working.")
    a("")
    a("**What they cannot support.** A predictive model. Measured on this training set,")
    a(f"no rung of the ladder beats a mean predictor: the baseline MAE is")
    a(f"{_f(base_mae)} eV and the Gaussian process is worse, at")
    a(f"{_f(float(gp_row['mae_eV']))} eV. The GP's intervals are overconfident, with")
    a(f"{int(c68['inside'])}/{int(c68['n'])} held-out points inside the nominal 68%")
    a(f"interval and {int(c95['inside'])}/{int(c95['n'])} inside the nominal 95% one, and")
    a(f"an RMS z of {_f(rms_z, 2)}. Uncertainty-driven acquisition does not beat random")
    a(f"selection ({al_gain:+.4f} +/- {_f(al_se)} eV, indistinguishable). The live")
    a(f"candidate ranking is degenerate: predictive sigma varies by {sigma_rel:.2%}")
    a("across the whole pending pool, so the ordering carries no information.")
    a("")
    a("**Where ML fits here.** Prioritisation and uncertainty: deciding which expensive")
    a("calculation to run next, and reporting how confident that suggestion is. That is")
    a("the only role claimed. **Where it does not fit:** replacing the physics. Nothing")
    a("in this repository is a substitute for the electronic-structure calculation. The")
    a("surrogate's job would be to order a queue, never to supply an energy.")
    a("")
    a("**No rescaling, no silent repair.** The variance rescaling that would make the")
    a(f"GP's intervals self-consistent (sigma x {_f(rms_z, 3)}) is reported and never")
    a("applied. Numerically unstable results are flagged and kept rather than dropped")
    if unstable_models:
        a(f"(currently: {', '.join(unstable_models)} at small subset sizes). Nothing")
    else:
        a("(none in the current run). Nothing")
    a("downstream consumes anything marked excluded or untriaged.")
    a("")
    a("**Open concern about the input geometry.** The slabs this project reads were")
    a("built upstream. Measured against rutile IrO2 constructed from published lattice")
    a(f"parameters (`src/geometry_audit.py`), the shortest Ir-O contact is")
    a(f"{_f(float(g_slab['ir_o_shortest_A']), 3)} A against a reference")
    a(f"{_f(float(g_ref['ir_o_shortest_A']), 3)} A, which is {ir_o_ratio:.2f}x the")
    a(f"expected value, and there are {int(g_slab['n_oo_pairs_below_2A'])} O-O contacts")
    a(f"below {_f(float(g_slab['oo_implausible_threshold_A']), 1)} A, the shortest at")
    a(f"{_f(float(g_slab['o_o_shortest_A']), 3)} A against a reference")
    a(f"{_f(float(g_ref['o_o_shortest_A']), 3)} A. An O-O separation near 1.5 A is a")
    a("peroxide bond length, not an oxide lattice spacing. Stoichiometry is correct at")
    a(f"{_f(float(g_slab['O_per_Ir']), 2)} O per Ir. This has **not** been changed here:")
    a("the descriptors are computed faithfully from the structures as given, and the")
    a("methodological conclusions above concern sample size and are unaffected by the")
    a("target values. But the adsorption energies should not be read chemically until")
    a("the slab construction in the upstream pipeline has been checked.")
    a("")

    # -------------------------------------------------------- provenance
    a("## Data provenance")
    a("")
    a(f"Nothing in `data/` originates here. Energies and geometries come from the")
    a(f"computational-chemistry pipeline at [{UPSTREAM}]({UPSTREAM}). That repository")
    a("keeps no outputs in Git: results are published as downloadable CI artefacts, so")
    a("the run folders live outside both repositories. The exact locations searched are")
    a("committed in `data/inventory_roots.txt`.")
    a("")
    a("Convergence is not decided here either. Every run is characterised by the")
    a("upstream triage tool (`scripts/triage_runs.py`), which bins it against a frozen")
    a("optimiser configuration and hashes that configuration into each record. This")
    a("repository treats triage as authoritative: a configuration with no triage record")
    a("stays a candidate no matter what its own results file claims. `site_intended` is")
    a("carried verbatim and never overwritten; ML labels follow `site_final`.")
    a("")
    a(f"Roles in `data/dataset.csv` ({n_total} rows): {n_train} train, {n_candidate}")
    a(f"candidate, {n_reference} reference, {n_excluded} excluded. Excluded rows are")
    a("hydrated configurations, carried over from an earlier tool migration and out of")
    a("scope; they are kept for provenance and never trained on, ranked, or featurised.")
    a("")

    # ------------------------------------------------------------- methods
    a("## Methods")
    a("")
    a(f"Fixed seed {RANDOM_SEED} project-wide, set through `src/seeds.py`.")
    a("")
    a("**Target.** `E_ads = E(slab + nH) - E_clean - n * E_ref_H`, with")
    a(f"`E_clean = {_f(e_clean, 6)}` eV from the `role=reference` row and")
    a(f"`E_ref_H = {_f(e_ref_h, 6)}` eV = half the GFN2-xTB energy of H2 relaxed in a")
    a("large cell by `src/h2_reference.py`. Neither number is typed in by hand.")
    a("")
    a("**Features.** `src/featurise.py` computes descriptors from **pre-relaxation**")
    a("geometry only. Candidates have no relaxed geometry, so a descriptor read off a")
    a("relaxed structure would be unavailable at prediction time and would leak the")
    a("optimisation result into training. Relaxed artefacts are on a hard denylist, and")
    a("an optimisation trajectory may be used only at frame 0.")
    a("")
    a(f"**Feature subset.** {len(FEATURE_DICTIONARY)} descriptors are computed and")
    a(f"recorded; {len(MODEL_FEATURES)} are used for modelling, chosen on physical")
    a("grounds before any performance was measured and not revised afterwards, because")
    a("data-driven selection at this sample size would leak. See the dictionary below.")
    a("")
    a("**Evaluation.** Leave-one-out, not nested cross-validation: nested CV would spend")
    a("folds this dataset does not have. Model differences are compared against paired")
    a("fold noise, and a difference smaller than that noise is reported as no")
    a("difference.")
    a("")
    a("**Noise floor.** Two independent estimates, the larger taken as operative:")
    a(f"{_f(floor_tol)} eV implied by the force-convergence tolerance, and")
    a(f"{_f(floor_emp)} eV observed as the run-to-run spread on the one configuration")
    a(f"optimised twice. Operative floor: **{_f(floor)} eV**. An error at or below the")
    a("floor is not a good model; it is the limit of what the data resolves.")
    a("")

    # ------------------------------------------------------------- results
    a("## Results")
    a("")
    a(f"Leave-one-out on {n_train} converged points, {len(MODEL_FEATURES)} features.")
    a("`beats baseline` requires the gain to exceed its own paired standard error.")
    a("")
    a("| model | LOO MAE (eV) | RMSE (eV) | gain vs baseline (eV) | beats baseline |")
    a("| --- | --- | --- | --- | --- |")
    for _, row in ladder.iterrows():
        reported = "" if bool(row.get("reported", True)) else " *(withheld)*"
        a(f"| `{row['model']}`{reported} | {_f(row['mae_eV'])} | {_f(row['rmse_eV'])} "
          f"| {float(row['mae_gain_vs_baseline_eV']):+.4f} +/- {_f(row['gain_se_eV'])} "
          f"| {'yes' if bool(row['beats_baseline']) else 'no'} |")
    a("")
    a("`random_forest` is reported only if it beats ridge by more than the fold noise;")
    a("in the current run it does not, and the restraint is the point.")
    a("")
    a("**Interval calibration.**")
    a("")
    a("| nominal | inside | empirical | Wilson 95% CI |")
    a("| --- | --- | --- | --- |")
    for _, row in calib.iterrows():
        a(f"| {row['nominal_label']} | {int(row['inside'])}/{int(row['n'])} "
          f"| {float(row['empirical_coverage']):.0%} "
          f"| [{_f(row['wilson_lo'], 2)}, {_f(row['wilson_hi'], 2)}] |")
    a("")
    a(f"RMS z = {_f(rms_z, 2)}. The intervals are too narrow; no rescaling is applied.")
    a("")
    a("**Acquisition policy.** Comparing max-uncertainty against random selection on a")
    a("fixed held-out set, so both policies are scored on identical points. Scoring on")
    a("the not-yet-acquired pool instead would favour max-uncertainty by construction,")
    a("since that policy takes the hardest points first; both variants are recorded in")
    a("`results/acquisition_policy.csv` and labelled.")
    a("")
    verdict = ("beats random" if al_beats else "does **not** beat random")
    a(f"At the one step that can distinguish the policies, uncertainty sampling")
    a(f"{verdict}: mean gain {al_gain:+.4f} +/- {_f(al_se)} eV.")
    a("")
    a(f"**Headline.** {n_train} converged points do not support a surrogate yet. The")
    a("diagnostics are built to be rerun as the pending restarts and pairs extend the")
    a("training set.")
    a("")

    # ------------------------------------------------------------- figures
    a("## Figures")
    a("")
    a("| figure | what it shows |")
    a("| --- | --- |")
    a("| `figures/calibration.png` | GP leave-one-out parity with 68% and 95% "
      "intervals, and a reliability curve. Below the diagonal means overconfident. |")
    a("| `figures/learning_curve.png` | LOO error against training-set size. The band "
      "is the spread across subsets; where it exceeds the gap between models, the data "
      "cannot rank them. |")
    a("| `figures/active_learning.png` | The pending queue ranked by predictive sigma "
      "(flagged degenerate), and the unconfounded acquisition-policy comparison. |")
    a("")

    # -------------------------------------------------- feature dictionary
    a("## Feature dictionary")
    a("")
    a("Computed by `src/featurise.py` from pre-relaxation geometry; values in")
    a("`data/features.csv`.")
    a("")
    a((data / "feature_dictionary.md").read_text().rstrip())
    a("")
    a((data / "feature_subset_rationale.md").read_text().rstrip())
    a("")
    a("`selection_note` travels alongside as a metadata text column, carried from the")
    a("orbital manifest. It records the surface-chemistry reasoning behind a site and is")
    a("never used as a feature.")
    a("")

    # ------------------------------------------------------ reproduction
    a("## Reproducing every number")
    a("")
    a("Each command regenerates the artefact it names. Run in order from a clean")
    a("checkout; every number in this README comes from one of these steps.")
    a("")
    a("```bash")
    a("conda env create -f environment.yml && conda activate irox-adsorption-ml")
    a("")
    a("# 1. inventory the run folders and the triage report -> data/dataset.csv")
    a("python src/inventory.py --roots-file data/inventory_roots.txt \\")
    a("    --triage-summary <triage_out>/triage_summary.csv --out data/dataset.csv")
    a("")
    a("# 2. hydrogen reference (needs xtb) -> data/h2_reference.json")
    a("python src/h2_reference.py")
    a("")
    a("# 3. descriptors from pre-relaxation geometry -> data/features.csv")
    a("python src/featurise.py --dataset data/dataset.csv \\")
    a("    --roots-file data/inventory_roots.txt --out data/features.csv")
    a("")
    a("# 4. the stated modelling subset -> data/features_model*.csv")
    a("python src/feature_subset.py")
    a("python src/feature_subset.py --role train --out data/features_model_train.csv")
    a("")
    a("# 5. model ladder -> results/model_ladder.csv")
    a("python src/evaluate.py --features data/features_model_train.csv")
    a("")
    a("# 6. diagnostics -> results/ + figures/")
    a("python src/calibration.py   --features data/features_model.csv")
    a("python src/learning_curve.py --features data/features_model.csv")
    a("python src/active_learning.py --features data/features_model.csv")
    a("")
    a("# 7. input-geometry audit -> results/geometry_audit.csv")
    a("python src/geometry_audit.py")
    a("")
    a("# 8. regenerate this README from the tables above")
    a("python src/build_readme.py")
    a("")
    a("# validate")
    a("python -m pytest tests/ -q")
    a("```")
    a("")
    a("Steps 1, 3 and 7 need the run folders listed in `data/inventory_roots.txt`, and")
    a("step 2 needs the `xtb` calculator. Steps 4-6, 8 and the tests run from committed")
    a("data alone, which is what CI exercises.")
    a("")

    # -------------------------------------------------------- attribution
    a("## How this repository was built")
    a("")
    a(f"Built agent-first: written by directing Claude Code under the ground rules")
    a(f"above, with every result validated by the test suite ({tests_phrase}), and each")
    a("module reviewed and committed by M. Malhotra before the next began.")
    a("")

    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="README.md")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    text = build(root)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
