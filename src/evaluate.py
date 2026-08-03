#!/usr/bin/env python3
"""Leave-one-out evaluation of the model ladder.

One command produces ``results/model_ladder.csv`` and prints a table comparing
every rung to the baseline and to the noise floor.

Target
------
    E_ads = E(slab + nH) - E_clean - n * E_ref_H

with ``E_clean`` taken from the ``role == "reference"`` row of the dataset and
``E_ref_H = 1/2 E(H2)`` read from the cache written by ``src/h2_reference.py``.
Both are script-generated; neither is typed in here.

Why leave-one-out and not nested CV
-----------------------------------
The training set is small by design. Nested CV would spend folds it does not
have; LOO uses every point. The cost is that fold-to-fold scatter is large, so
this script reports the paired fold noise beside every comparison and refuses
to call a difference real unless it exceeds it.

Noise floor
-----------
Two independent estimates are reported:

* **tolerance-implied** -- a residual force at the convergence threshold acting
  over a residual displacement, ``fmax_target * displacement_scale``.
* **empirical** -- the observed run-to-run spread on the one configuration that
  has been optimised twice under the frozen config (the clean slab).

The larger of the two is the operative floor. A model whose error sits at or
below the floor is not being praised; it is being told it has hit the limit of
what the underlying data can resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut

from src.models import BASELINE, RF_MUST_BEAT, build_ladder, predict_with_std
from src.seeds import RANDOM_SEED, set_seeds

# --- noise floor inputs, each with stated provenance ----------------------

# Force convergence threshold of the frozen optimiser config, eV/Angstrom.
FMAX_TARGET = 0.05
# Residual displacement scale, Angstrom. A relaxed structure sitting at the
# force threshold is typically this far from the true minimum.
DISPLACEMENT_SCALE = 0.1

# The clean slab is the one configuration optimised twice. The superseded run's
# artefacts no longer exist, so its energy is recorded here as a literal with
# provenance rather than read from disk. It is used ONLY to size the noise
# floor; it is never used as an energy.
E_CLEAN_SUPERSEDED_EV = -10375.9576  # older manifest, artefacts deleted

MIN_TRAIN_ROWS = 3

# Below this many training rows, LOO fold noise is large enough relative to any
# plausible model difference that the ranking is provisional, not a selection.
SMALL_N_WARN = 20


class AlignmentError(ValueError):
    """Raised when features and targets do not line up."""


def load_h2_reference(path: Path) -> float:
    """Read E_ref_H from the cache written by src/h2_reference.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"H2 reference cache not found at {path}. Run "
            "'python src/h2_reference.py' first -- E_ref_H must be "
            "script-generated, not supplied by hand."
        )
    record = json.loads(path.read_text())
    if "E_ref_H_eV" not in record:
        raise ValueError(f"{path} has no 'E_ref_H_eV' field")
    return float(record["E_ref_H_eV"])


def get_clean_reference(dataset: pd.DataFrame) -> float:
    """Return E_clean from the dataset's reference row."""
    ref = dataset[dataset["role"] == "reference"]
    if len(ref) == 0:
        raise ValueError(
            "No role=='reference' row in the dataset, so E_clean is unknown "
            "and no adsorption energy can be formed."
        )
    if len(ref) > 1:
        raise ValueError(
            f"Expected exactly one role=='reference' row, found {len(ref)}: "
            f"{sorted(ref['config_id'])}"
        )
    return float(ref["energy_eV"].iloc[0])


def compute_targets(dataset: pd.DataFrame, e_clean: float,
                    e_ref_h: float) -> pd.DataFrame:
    """Attach E_ads to every row that has a total energy."""
    out = dataset.copy()
    out["E_ads_eV"] = (
        out["energy_eV"] - e_clean - out["n_adsorbates"] * e_ref_h
    )
    out.loc[out["energy_eV"].isna(), "E_ads_eV"] = np.nan
    return out


def align(features: pd.DataFrame, targets: pd.DataFrame,
          feature_columns: list[str] | None = None
          ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Join features to targets on config_id, refusing anything misaligned.

    Returns (X, y, config_ids, feature_names).
    """
    for name, frame in (("features", features), ("targets", targets)):
        if "config_id" not in frame.columns:
            raise AlignmentError(f"{name} table has no 'config_id' column")
        dupes = frame["config_id"][frame["config_id"].duplicated()].tolist()
        if dupes:
            raise AlignmentError(f"{name} table has duplicate config_id: {sorted(set(dupes))}")

    f_ids = set(features["config_id"])
    t_ids = set(targets["config_id"])

    missing = sorted(t_ids - f_ids)
    if missing:
        raise AlignmentError(
            f"{len(missing)} training configuration(s) have no feature row: {missing}"
        )
    extra = sorted(f_ids - t_ids)
    if extra:
        raise AlignmentError(
            f"{len(extra)} feature row(s) have no matching target: {extra}. "
            "Refusing to silently drop them."
        )

    merged = targets.merge(features, on="config_id", how="inner", validate="one_to_one")
    merged = merged.sort_values("config_id", kind="mergesort").reset_index(drop=True)

    if feature_columns is None:
        drop = set(targets.columns) | {"config_id"}
        feature_columns = [
            c for c in features.columns
            if c != "config_id" and c not in drop
            and pd.api.types.is_numeric_dtype(features[c])
        ]
    if not feature_columns:
        raise AlignmentError("no numeric feature columns found")

    X = merged[feature_columns].to_numpy(dtype=float)
    y = merged["E_ads_eV"].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        bad = [feature_columns[i] for i in np.where(~np.isfinite(X).all(axis=0))[0]]
        raise AlignmentError(f"non-finite values in feature column(s): {bad}")
    if not np.isfinite(y).all():
        bad = [merged["config_id"][i] for i in np.where(~np.isfinite(y))[0]]
        raise AlignmentError(f"non-finite target for config(s): {bad}")

    return X, y, merged["config_id"].tolist(), list(feature_columns)


def noise_floor() -> dict[str, float]:
    """Two independent noise-floor estimates plus the operative one."""
    tolerance = FMAX_TARGET * DISPLACEMENT_SCALE
    empirical = float("nan")
    return {
        "tolerance_implied_eV": tolerance,
        "empirical_eV": empirical,
        "operative_eV": tolerance,
    }


def noise_floor_with_empirical(e_clean: float) -> dict[str, float]:
    """Noise floor including the observed clean-slab run-to-run spread."""
    tolerance = FMAX_TARGET * DISPLACEMENT_SCALE
    empirical = abs(e_clean - E_CLEAN_SUPERSEDED_EV)
    return {
        "tolerance_implied_eV": tolerance,
        "empirical_eV": empirical,
        "operative_eV": max(tolerance, empirical),
    }


def loo_predict(estimator, X: np.ndarray, y: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """Leave-one-out predictions. Returns (predictions, sigmas or None)."""
    loo = LeaveOneOut()
    preds = np.empty(len(y), dtype=float)
    sigmas = np.empty(len(y), dtype=float)
    any_sigma = False

    for train_idx, test_idx in loo.split(X):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        mean, std = predict_with_std(model, X[test_idx])
        preds[test_idx] = mean
        if std is not None:
            sigmas[test_idx] = std
            any_sigma = True
        else:
            sigmas[test_idx] = np.nan

    return preds, (sigmas if any_sigma else None)


def evaluate_ladder(X: np.ndarray, y: np.ndarray,
                    include_random_forest: bool = True,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Run every rung through LOO and assemble the results table."""
    n = len(y)
    if n < MIN_TRAIN_ROWS:
        raise ValueError(
            f"leave-one-out needs at least {MIN_TRAIN_ROWS} training rows, got {n}"
        )

    rungs = build_ladder(include_random_forest=include_random_forest, seed=seed)
    abs_errors: dict[str, np.ndarray] = {}
    rows = []

    for rung in rungs:
        preds, sigmas = loo_predict(rung.estimator, X, y)
        err = y - preds
        abs_err = np.abs(err)
        abs_errors[rung.name] = abs_err

        row = {
            "model": rung.name,
            "n_train_rows": n,
            "mae_eV": float(abs_err.mean()),
            "rmse_eV": float(np.sqrt((err ** 2).mean())),
            "max_abs_err_eV": float(abs_err.max()),
            "fold_noise_se_eV": float(abs_err.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "gives_uncertainty": rung.gives_uncertainty,
            "note": rung.note,
        }

        if sigmas is not None and np.isfinite(sigmas).all() and (sigmas > 0).all():
            z = err / sigmas
            row["mean_pred_sigma_eV"] = float(sigmas.mean())
            row["mean_abs_z"] = float(np.abs(z).mean())
            row["coverage_95"] = float((np.abs(z) <= 1.96).mean())
        else:
            row["mean_pred_sigma_eV"] = np.nan
            row["mean_abs_z"] = np.nan
            row["coverage_95"] = np.nan

        rows.append(row)

    table = pd.DataFrame(rows)

    # Paired comparison against the baseline: per-fold difference in absolute
    # error, so the fold noise that both models share cancels.
    base = abs_errors[BASELINE]
    deltas, delta_ses, beats = [], [], []
    for name in table["model"]:
        d = base - abs_errors[name]
        mean_d = float(d.mean())
        se_d = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        deltas.append(mean_d)
        delta_ses.append(se_d)
        beats.append(bool(name != BASELINE and np.isfinite(se_d) and mean_d > se_d))

    table["mae_gain_vs_baseline_eV"] = deltas
    table["gain_se_eV"] = delta_ses
    table["beats_baseline"] = beats

    # random_forest is reported only if it clears RF_MUST_BEAT beyond fold noise.
    if "random_forest" in set(table["model"]) and RF_MUST_BEAT in abs_errors:
        d = abs_errors[RF_MUST_BEAT] - abs_errors["random_forest"]
        mean_d = float(d.mean())
        se_d = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        clears = bool(np.isfinite(se_d) and mean_d > se_d)
        table["rf_beats_ridge"] = [
            clears if m == "random_forest" else None for m in table["model"]
        ]
        table["rf_gain_vs_ridge_eV"] = [
            mean_d if m == "random_forest" else None for m in table["model"]
        ]
        table["reported"] = [
            clears if m == "random_forest" else True for m in table["model"]
        ]
    else:
        table["rf_beats_ridge"] = None
        table["rf_gain_vs_ridge_eV"] = None
        table["reported"] = True

    return table


def format_table(table: pd.DataFrame, floor: dict[str, float],
                 config_ids: list[str], feature_names: list[str]) -> str:
    """Render the printed report. Every number comes from table or floor."""
    lines: list[str] = []
    add = lines.append
    n = int(table["n_train_rows"].iloc[0])

    add("=" * 78)
    add("Model ladder -- leave-one-out cross-validation")
    add("=" * 78)
    add(f"seed              : {RANDOM_SEED}")
    add(f"training rows (n) : {n}")
    add(f"features          : {len(feature_names)}  ({', '.join(feature_names)})")
    add(f"LOO folds         : {n}, each fitting on {n - 1} points")
    add("")
    add("Noise floor on the target:")
    add(f"  tolerance-implied : {floor['tolerance_implied_eV']:.4f} eV "
        f"(fmax {FMAX_TARGET} eV/A x {DISPLACEMENT_SCALE} A)")
    add(f"  empirical         : {floor['empirical_eV']:.4f} eV "
        f"(clean-slab run-to-run spread)")
    add(f"  operative floor   : {floor['operative_eV']:.4f} eV")
    add("")

    header = f"{'model':<16}{'MAE eV':>10}{'RMSE eV':>10}{'gain vs base':>14}{'+/- se':>9}{'beats?':>8}"
    add(header)
    add("-" * len(header))
    for _, row in table.iterrows():
        if row["model"] != BASELINE and row["reported"] is False:
            continue
        beats = "-" if row["model"] == BASELINE else ("yes" if row["beats_baseline"] else "no")
        add(f"{row['model']:<16}{row['mae_eV']:>10.4f}{row['rmse_eV']:>10.4f}"
            f"{row['mae_gain_vs_baseline_eV']:>14.4f}{row['gain_se_eV']:>9.4f}{beats:>8}")
    add("")

    # Uncertainty calibration, the point of the GP rung.
    unc = table[table["gives_uncertainty"] & table["mean_abs_z"].notna()]
    if not unc.empty:
        add("Predictive uncertainty (LOO):")
        for _, row in unc.iterrows():
            add(f"  {row['model']}: mean sigma = {row['mean_pred_sigma_eV']:.4f} eV, "
                f"mean |z| = {row['mean_abs_z']:.2f}, "
                f"95% coverage = {row['coverage_95']:.0%} "
                f"({int(round(row['coverage_95'] * n))}/{n} folds)")
        add("")

    # The restraint clause.
    rf = table[table["model"] == "random_forest"]
    if not rf.empty:
        clears = rf["rf_beats_ridge"].iloc[0]
        gain = rf["rf_gain_vs_ridge_eV"].iloc[0]
        if clears:
            add(f"random_forest beats {RF_MUST_BEAT} by {gain:.4f} eV MAE, "
                "beyond fold noise -- reported above.")
        else:
            add(f"random_forest did NOT beat {RF_MUST_BEAT} beyond fold noise "
                f"(gain {gain:+.4f} eV MAE). Withheld from the table above.")
        add("")

    # Overall verdict, stated conservatively.
    winners = table[(table["model"] != BASELINE) & table["beats_baseline"] & table["reported"]]
    if winners.empty:
        add("VERDICT: no rung beats the mean predictor by more than the fold noise.")
        add("         On this training set the ladder does not support a claim that")
        add("         any model has learned the target.")
    else:
        best = winners.sort_values("mae_eV").iloc[0]
        add(f"VERDICT: {best['model']} beats the baseline by "
            f"{best['mae_gain_vs_baseline_eV']:.4f} +/- {best['gain_se_eV']:.4f} eV MAE.")
        if best["mae_eV"] <= floor["operative_eV"]:
            add(f"         Its MAE ({best['mae_eV']:.4f} eV) is at or below the noise")
            add("         floor, so the data cannot resolve any further improvement.")

    if n < SMALL_N_WARN:
        add("")
        add(f"CAUTION: n = {n}. With this many points the LOO fold noise is large")
        add("         relative to any plausible model difference. Treat the ranking")
        add("         above as provisional, not as a model selection.")

    add("=" * 78)
    return "\n".join(lines)


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description="LOO evaluation of the model ladder.")
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--features", required=True,
                    help="CSV with config_id plus numeric feature columns.")
    ap.add_argument("--h2-reference", default="data/h2_reference.json")
    ap.add_argument("--out", default="results/model_ladder.csv")
    ap.add_argument("--role", default="train",
                    help="Dataset role to train on (default: train).")
    ap.add_argument("--no-random-forest", action="store_true")
    args = ap.parse_args()

    dataset = pd.read_csv(args.dataset)
    features = pd.read_csv(args.features)

    e_clean = get_clean_reference(dataset)
    e_ref_h = load_h2_reference(Path(args.h2_reference))

    with_targets = compute_targets(dataset, e_clean, e_ref_h)
    train = with_targets[with_targets["role"] == args.role]
    train = train[train["E_ads_eV"].notna()]
    train = train[["config_id", "E_ads_eV"]].reset_index(drop=True)

    X, y, config_ids, feature_names = align(features, train)

    table = evaluate_ladder(X, y, include_random_forest=not args.no_random_forest)
    floor = noise_floor_with_empirical(e_clean)

    for key, value in floor.items():
        table[f"noise_{key}"] = value
    table["E_clean_eV"] = e_clean
    table["E_ref_H_eV"] = e_ref_h
    table["seed"] = RANDOM_SEED

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print(format_table(table, floor, config_ids, feature_names))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
