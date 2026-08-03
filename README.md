[![CI](https://github.com/sapphire10-k/irox-adsorption-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sapphire10-k/irox-adsorption-ml/actions/workflows/ci.yml)

# irox-adsorption-ml

## The question

On my curated set of converged H adsorption configurations on IrO2(110), as
inventoried in `data/dataset.csv`, can a Gaussian process with honest
uncertainties predict adsorption energetics well enough to prioritise which
configuration to optimise next?

## Limitations and judgment

This section is first because it governs how every number below should be read.

**The dataset is small by design.** There are 26 adsorption
configurations in scope (9 single-H, 17 two-adsorbate pairs), of which
5 are converged and usable as training data today. The sites were chosen
for surface-chemistry reasons set out in the accompanying preprint, not sampled to
fill a feature space. That is a design choice and it is not apologised for: a
curated set of physically motivated sites is what makes the individual points
interpretable. It does mean the statistical questions this repository can answer
are narrow, and the code is built to say so rather than to paper over it.

**What 5 converged points can support.** Bookkeeping and provenance:
which configurations exist, which converged, under which frozen optimiser
settings, and which are still pending. Descriptor computation from
pre-relaxation geometry. Leave-one-out diagnostics that put an honest bound on
what a surrogate is doing. Detecting that a surrogate is *not* working.

**What they cannot support.** A predictive model. Measured on this training set,
no rung of the ladder beats a mean predictor: the baseline MAE is
0.6345 eV and the Gaussian process is worse, at
0.8320 eV. The GP's intervals are overconfident, with
2/5 held-out points inside the nominal 68%
interval and 2/5 inside the nominal 95% one, and
an RMS z of 4.32. Uncertainty-driven acquisition does not beat random
selection (+0.0127 +/- 0.0354 eV, indistinguishable). The live
candidate ranking is degenerate: predictive sigma varies by 0.00%
across the whole pending pool, so the ordering carries no information.

**Where ML fits here.** Prioritisation and uncertainty: deciding which expensive
calculation to run next, and reporting how confident that suggestion is. That is
the only role claimed. **Where it does not fit:** replacing the physics. Nothing
in this repository is a substitute for the electronic-structure calculation. The
surrogate's job would be to order a queue, never to supply an energy.

**No rescaling, no silent repair.** The variance rescaling that would make the
GP's intervals self-consistent (sigma x 4.323) is reported and never
applied. Numerically unstable results are flagged and kept rather than dropped
(none in the current run). Nothing
downstream consumes anything marked excluded or untriaged.

**Open concern about the input geometry.** The slabs this project reads were
built upstream. Measured against rutile IrO2 constructed from published lattice
parameters (`src/geometry_audit.py`), the shortest Ir-O contact is
2.402 A against a reference
1.945 A, which is 1.23x the
expected value, and there are 48 O-O contacts
below 2.0 A, the shortest at
1.517 A against a reference
2.471 A. An O-O separation near 1.5 A is a
peroxide bond length, not an oxide lattice spacing. Stoichiometry is correct at
2.00 O per Ir. This has **not** been changed here:
the descriptors are computed faithfully from the structures as given, and the
methodological conclusions above concern sample size and are unaffected by the
target values. But the adsorption energies should not be read chemically until
the slab construction in the upstream pipeline has been checked.

## Data provenance

Nothing in `data/` originates here. Energies and geometries come from the
computational-chemistry pipeline at [https://github.com/sapphire10-k/IrOx_VQE](https://github.com/sapphire10-k/IrOx_VQE). That repository
keeps no outputs in Git: results are published as downloadable CI artefacts, so
the run folders live outside both repositories. The exact locations searched are
committed in `data/inventory_roots.txt`.

Convergence is not decided here either. Every run is characterised by the
upstream triage tool (`scripts/triage_runs.py`), which bins it against a frozen
optimiser configuration and hashes that configuration into each record. This
repository treats triage as authoritative: a configuration with no triage record
stays a candidate no matter what its own results file claims. `site_intended` is
carried verbatim and never overwritten; ML labels follow `site_final`.

Roles in `data/dataset.csv` (29 rows): 5 train, 22
candidate, 1 reference, 1 excluded. Excluded rows are
hydrated configurations, carried over from an earlier tool migration and out of
scope; they are kept for provenance and never trained on, ranked, or featurised.

## Methods

Fixed seed 0 project-wide, set through `src/seeds.py`.

**Target.** `E_ads = E(slab + nH) - E_clean - n * E_ref_H`, with
`E_clean = -10375.723335` eV from the `role=reference` row and
`E_ref_H = -13.370124` eV = half the GFN2-xTB energy of H2 relaxed in a
large cell by `src/h2_reference.py`. Neither number is typed in by hand.

**Features.** `src/featurise.py` computes descriptors from **pre-relaxation**
geometry only. Candidates have no relaxed geometry, so a descriptor read off a
relaxed structure would be unavailable at prediction time and would leak the
optimisation result into training. Relaxed artefacts are on a hard denylist, and
an optimisation trajectory may be used only at frame 0.

**Feature subset.** 25 descriptors are computed and
recorded; 4 are used for modelling, chosen on physical
grounds before any performance was measured and not revised afterwards, because
data-driven selection at this sample size would leak. See the dictionary below.

**Evaluation.** Leave-one-out, not nested cross-validation: nested CV would spend
folds this dataset does not have. Model differences are compared against paired
fold noise, and a difference smaller than that noise is reported as no
difference.

**Noise floor.** Two independent estimates, the larger taken as operative:
0.0050 eV implied by the force-convergence tolerance, and
0.2343 eV observed as the run-to-run spread on the one configuration
optimised twice. Operative floor: **0.2343 eV**. An error at or below the
floor is not a good model; it is the limit of what the data resolves.

## Results

Leave-one-out on 5 converged points, 4 features.
`beats baseline` requires the gain to exceed its own paired standard error.

| model | LOO MAE (eV) | RMSE (eV) | gain vs baseline (eV) | beats baseline |
| --- | --- | --- | --- | --- |
| `mean` | 0.6345 | 0.7679 | +0.0000 +/- 0.0000 | no |
| `ridge` | 0.7285 | 0.8686 | -0.0940 +/- 0.0956 | no |
| `gp` | 0.8320 | 0.9937 | -0.1975 +/- 0.1593 | no |
| `random_forest` *(withheld)* | 0.6517 | 0.7401 | -0.0172 +/- 0.0843 | no |

`random_forest` is reported only if it beats ridge by more than the fold noise;
in the current run it does not, and the restraint is the point.

**Interval calibration.**

| nominal | inside | empirical | Wilson 95% CI |
| --- | --- | --- | --- |
| 50% | 2/5 | 40% | [0.12, 0.77] |
| 68% | 2/5 | 40% | [0.12, 0.77] |
| 80% | 2/5 | 40% | [0.12, 0.77] |
| 90% | 2/5 | 40% | [0.12, 0.77] |
| 95% | 2/5 | 40% | [0.12, 0.77] |
| 99% | 3/5 | 60% | [0.23, 0.88] |

RMS z = 4.32. The intervals are too narrow; no rescaling is applied.

**Acquisition policy.** Comparing max-uncertainty against random selection on a
fixed held-out set, so both policies are scored on identical points. Scoring on
the not-yet-acquired pool instead would favour max-uncertainty by construction,
since that policy takes the hardest points first; both variants are recorded in
`results/acquisition_policy.csv` and labelled.

At the one step that can distinguish the policies, uncertainty sampling
does **not** beat random: mean gain +0.0127 +/- 0.0354 eV.

**Headline.** 5 converged points do not support a surrogate yet. The
diagnostics are built to be rerun as the pending restarts and pairs extend the
training set.

## Figures

| figure | what it shows |
| --- | --- |
| `figures/calibration.png` | GP leave-one-out parity with 68% and 95% intervals, and a reliability curve. Below the diagonal means overconfident. |
| `figures/learning_curve.png` | LOO error against training-set size. The band is the spread across subsets; where it exceeds the gap between models, the data cannot rank them. |
| `figures/active_learning.png` | The pending queue ranked by predictive sigma (flagged degenerate), and the unconfounded acquisition-policy comparison. |

## Feature dictionary

Computed by `src/featurise.py` from pre-relaxation geometry; values in
`data/features.csv`.

| feature | definition | units |
| --- | --- | --- |
| `n_adsorbates` | Number of adsorbed H atoms, from the pre-relaxation geometry | count |
| `is_2H` | 1 for two-adsorbate configurations, 0 otherwise; indicator for the pair block | 0/1 |
| `n_water` | Co-adsorbed water molecules, i.e. oxygens carrying H but bonded to no Ir; their hydrogens are excluded from n_adsorbates | count |
| `n_atoms` | Total atoms in the configuration | count |
| `cn_mean` | Mean Ir-O coordination number of the binding oxygen(s), cutoff 2.6 A | count |
| `cn_min` | Minimum coordination number across binding oxygens | count |
| `cn_max` | Maximum coordination number across binding oxygens | count |
| `gcn_mean` | Mean generalised coordination number, sum_j CN(j)/6 (Calle-Vallejo) | dimensionless |
| `gcn_min` | Minimum generalised coordination number across binding oxygens | dimensionless |
| `gcn_max` | Maximum generalised coordination number across binding oxygens | dimensionless |
| `n_bridge_sites` | Binding oxygens with Ir coordination 2 (bridging) | count |
| `n_3f_sites` | Binding oxygens with Ir coordination 3 (three-fold) | count |
| `d_OH_mean_A` | Mean H to binding-oxygen distance in the built geometry | angstrom |
| `nn1_dist_A` | Mean distance from binding oxygen to its 1st nearest heavy atom | angstrom |
| `nn2_dist_A` | Mean distance from binding oxygen to its 2nd nearest heavy atom | angstrom |
| `nn3_dist_A` | Mean distance from binding oxygen to its 3rd nearest heavy atom | angstrom |
| `frac_Ir_in_shell` | Fraction of atoms within 2.6 A of the binding oxygen that are Ir | fraction |
| `z_site_A` | Mean z coordinate of the binding oxygen(s) | angstrom |
| `z_below_top_A` | Depth of the binding oxygen below the topmost slab atom | angstrom |
| `lateral_margin_A` | Distance from the binding site to the nearest lateral edge of the non-periodic slab; low values mean coordination is truncated by the finite cell | angstrom |
| `pair_separation_A` | Distance between the two binding oxygens (2H only, else 0) | angstrom |
| `pair_hh_separation_A` | Distance between the two H atoms (2H only, else 0) | angstrom |
| `pair_cos_theta_z` | Cosine of the angle between the O-O pair vector and the surface normal (2H only, else 0) | dimensionless |
| `pair_inplane_angle_deg` | In-plane angle of the O-O pair vector from the x axis (2H only, else 0) | degrees |
| `pair_same_row` | 1 if the pair vector lies along a lateral axis, i.e. same bridging-oxygen row (2H only, else 0) | 0/1 |

### Modelling feature subset

The full dictionary above records all descriptors computed. 4 of them are used for modelling, chosen on physical grounds **before** any performance comparison and not revised in response to results (data-driven selection at this sample size would leak).

| feature | why this one |
| --- | --- |
| `gcn_mean` | Generalised coordination number of the binding oxygen. Coordination-based descriptors are the standard proxy for adsorption strength on oxide surfaces: under-coordinated sites bind more strongly. |
| `nn1_dist_A` | Nearest-neighbour distance from the binding oxygen. Captures the local bond-length environment independently of the coordination count. |
| `n_bridge_sites` | Site type: how many binding oxygens are bridging (Ir coordination 2) rather than three-fold or terminal. Bridging versus in-plane oxygen is the primary site distinction on rutile (110). |
| `n_adsorbates` | Coverage, 1H versus 2H. Two adsorbates interact laterally, which is a different energetic regime from an isolated adsorbate. Constant across the present all-1H training set and kept deliberately, so that it is already in place when a pair converges. |

`selection_note` travels alongside as a metadata text column, carried from the
orbital manifest. It records the surface-chemistry reasoning behind a site and is
never used as a feature.

## Reproducing every number

Each command regenerates the artefact it names. Run in order from a clean
checkout; every number in this README comes from one of these steps.

```bash
conda env create -f environment.yml && conda activate irox-adsorption-ml

# 1. inventory the run folders and the triage report -> data/dataset.csv
python src/inventory.py --roots-file data/inventory_roots.txt \
    --triage-summary <triage_out>/triage_summary.csv --out data/dataset.csv

# 2. hydrogen reference (needs xtb) -> data/h2_reference.json
python src/h2_reference.py

# 3. descriptors from pre-relaxation geometry -> data/features.csv
python src/featurise.py --dataset data/dataset.csv \
    --roots-file data/inventory_roots.txt --out data/features.csv

# 4. the stated modelling subset -> data/features_model*.csv
python src/feature_subset.py
python src/feature_subset.py --role train --out data/features_model_train.csv

# 5. model ladder -> results/model_ladder.csv
python src/evaluate.py --features data/features_model_train.csv

# 6. diagnostics -> results/ + figures/
python src/calibration.py   --features data/features_model.csv
python src/learning_curve.py --features data/features_model.csv
python src/active_learning.py --features data/features_model.csv

# 7. input-geometry audit -> results/geometry_audit.csv
python src/geometry_audit.py

# 8. regenerate this README from the tables above
python src/build_readme.py

# validate
python -m pytest tests/ -q
```

Steps 1, 3 and 7 need the run folders listed in `data/inventory_roots.txt`, and
step 2 needs the `xtb` calculator. Steps 4-6, 8 and the tests run from committed
data alone, which is what CI exercises.

## How this repository was built

Built agent-first: written by directing Claude Code under the ground rules
above, with every result validated by the test suite (all 182 tests), and each
module reviewed and committed by M. Malhotra before the next began.

