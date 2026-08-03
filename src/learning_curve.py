#!/usr/bin/env python3
"""Where does the data-limited regime bite?

For each training-set size m, draw m configurations, run leave-one-out inside
that subset, and record the error. Repeat over several random orderings so that
the spread caused by *which* points you happen to have is visible next to the
difference between models.

Reading the figure
------------------
Two things matter more than the trend:

* **The band.** It is the spread across orderings at fixed m. Where the band is
  wider than the gap between two models, the data cannot tell those models
  apart, no matter how suggestive the means look.
* **The floor.** The horizontal line is the operative noise floor on the target.
  A curve at or below it has stopped measuring model quality and started
  measuring optimiser scatter.

Subsets are drawn from random orderings with fixed seeds, so the whole figure is
reproducible. At m close to n the number of *distinct* subsets is small, and the
count is reported rather than left implicit.

One command:

    python src/learning_curve.py --features data/features.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # deterministic, headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import (MIN_TRAIN_ROWS, align, compute_targets,
                          get_clean_reference, load_h2_reference, loo_predict,
                          noise_floor_with_empirical)
from src.models import BASELINE, build_ladder
from src.seeds import RANDOM_SEED, set_seeds

DEFAULT_N_ORDERINGS = 25

# A fold MAE this many times the full target span cannot be a statement about
# model quality; it is numerical blow-up. With p features and n_train points,
# p >> n leaves near-constant columns whose standardisation divides by almost
# zero. Such points are flagged, reported, and kept in the CSV -- never
# quietly dropped or repaired, because the cause is the feature-to-sample
# ratio and that is a modelling decision, not a plotting one.
UNSTABLE_FACTOR = 100.0


def subset_indices(n: int, m: int, n_orderings: int,
                   seed: int = RANDOM_SEED) -> list[np.ndarray]:
    """Distinct index subsets of size m, drawn from seeded random orderings.

    Deduplicated so that a repeated draw is not counted twice as evidence. If
    the number of possible subsets is smaller than n_orderings, every subset is
    enumerated instead and the result is exhaustive.
    """
    total = comb(n, m)
    if total <= n_orderings:
        return [np.array(c, dtype=int) for c in combinations(range(n), m)]

    rng = np.random.default_rng(seed + m)  # m in the seed: reproducible per size
    seen: set[tuple[int, ...]] = set()
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < n_orderings and attempts < 50 * n_orderings:
        attempts += 1
        order = rng.permutation(n)[:m]
        key = tuple(sorted(order.tolist()))
        if key in seen:
            continue
        seen.add(key)
        out.append(np.array(key, dtype=int))
    return out


def curve(X: np.ndarray, y: np.ndarray, n_orderings: int = DEFAULT_N_ORDERINGS,
          include_random_forest: bool = False,
          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """LOO MAE for every model at every training-set size."""
    n = len(y)
    span = float(y.max() - y.min()) or 1.0
    unstable_above = UNSTABLE_FACTOR * span
    rungs = build_ladder(include_random_forest=include_random_forest, seed=seed)
    rows = []

    for m in range(MIN_TRAIN_ROWS, n + 1):
        subsets = subset_indices(n, m, n_orderings, seed=seed)
        exhaustive = comb(n, m) <= n_orderings
        for rung in rungs:
            maes = []
            for idx in subsets:
                preds, _ = loo_predict(rung.estimator, X[idx], y[idx])
                maes.append(float(np.abs(y[idx] - preds).mean()))
            maes_arr = np.array(maes, dtype=float)
            rows.append({
                "model": rung.name,
                "n_train": m,
                "n_subsets": len(subsets),
                "n_possible_subsets": comb(n, m),
                "exhaustive": exhaustive,
                "mae_mean_eV": float(maes_arr.mean()),
                "mae_std_eV": float(maes_arr.std(ddof=1)) if len(maes_arr) > 1 else 0.0,
                "mae_min_eV": float(maes_arr.min()),
                "mae_max_eV": float(maes_arr.max()),
                "mae_p25_eV": float(np.percentile(maes_arr, 25)),
                "mae_p75_eV": float(np.percentile(maes_arr, 75)),
                "n_features": int(X.shape[1]),
                "p_over_n_train": float(X.shape[1]) / m,
                "unstable": bool(maes_arr.mean() > unstable_above),
                "unstable_threshold_eV": unstable_above,
            })

    return pd.DataFrame(rows)


def data_limited_verdict(table: pd.DataFrame, floor: float) -> list[str]:
    """State plainly where the spread swamps the model differences."""
    lines: list[str] = []
    sizes = sorted(table["n_train"].unique())
    models = [m for m in table["model"].unique() if m != BASELINE]

    bad = table[table["unstable"]]
    if not bad.empty:
        for name, grp in bad.groupby("model"):
            where = ", ".join(str(int(v)) for v in sorted(grp["n_train"]))
            ratio = grp["p_over_n_train"].max()
            lines.append(
                f"  {name} is NUMERICALLY UNSTABLE at n_train = {where} "
                f"(MAE far above the target span)."
            )
            lines.append(
                f"  Cause: {int(grp['n_features'].iloc[0])} features against "
                f"as few as {int(grp['n_train'].min())} training points "
                f"(p/n up to {ratio:.0f});"
            )
            lines.append(
                "  standardising a near-constant column divides by almost "
                "zero. Those numbers say nothing"
            )
            lines.append(
                "  about model quality and are excluded from the trend "
                "reading below, but kept in the CSV."
            )
        lines.append("")

    table = table[~table["unstable"]]
    if table.empty:
        lines.append("  Every model/size combination was unstable; no trend "
                     "can be read.")
        return lines
    sizes = sorted(table["n_train"].unique())
    models = [m for m in table["model"].unique() if m != BASELINE]

    indistinguishable = []
    for m in sizes:
        at_m = table[table["n_train"] == m].set_index("model")
        band = float(at_m["mae_std_eV"].max())
        gaps = [abs(at_m.loc[a, "mae_mean_eV"] - at_m.loc[BASELINE, "mae_mean_eV"])
                for a in models if a in at_m.index]
        if gaps and band >= max(gaps):
            indistinguishable.append(m)

    if indistinguishable:
        lines.append(
            f"  At n_train = {', '.join(str(m) for m in indistinguishable)} the "
            "spread across subsets is at least as large as every"
        )
        lines.append(
            f"  model's gap to the {BASELINE} baseline. At those sizes the data "
            "cannot rank the models."
        )
    else:
        lines.append(
            "  At every size tested, at least one model's gap to the baseline "
            "exceeds the subset spread."
        )

    largest = table[table["n_train"] == sizes[-1]].set_index("model")
    below_floor = [m for m in largest.index if largest.loc[m, "mae_mean_eV"] <= floor]
    if below_floor:
        lines.append(
            f"  At the full training set, {', '.join(below_floor)} sit at or "
            f"below the {floor:.3f} eV noise floor,"
        )
        lines.append(
            "  so their remaining error is not distinguishable from optimiser "
            "scatter."
        )

    # Trend at the top end. The yardstick is the spread at the second-largest
    # size, because the largest size has only one possible subset and therefore
    # a spread of exactly zero -- which would make any change look significant.
    if len(sizes) >= 2:
        candidates = largest["mae_mean_eV"].drop(index=[BASELINE], errors="ignore")
        if not candidates.empty:
            best = candidates.idxmin()
            prev = table[(table["n_train"] == sizes[-2]) & (table["model"] == best)]
            if not prev.empty:
                prev_mae = float(prev["mae_mean_eV"].iloc[0])
                curr_mae = float(largest.loc[best, "mae_mean_eV"])
                change = curr_mae - prev_mae          # negative = improved
                band = float(prev["mae_std_eV"].iloc[0])

                if change < 0:
                    direction = f"improved by {abs(change):.4f} eV"
                elif change > 0:
                    direction = f"got WORSE by {change:.4f} eV"
                else:
                    direction = "did not change"
                lines.append(
                    f"  {best}: from n={sizes[-2]} to n={sizes[-1]} it {direction}."
                )
                if abs(change) < band:
                    lines.append(
                        f"  That is smaller than the subset spread at n={sizes[-2]} "
                        f"({band:.4f} eV), so the trend is not resolved."
                    )

        # The comparison that actually matters: does anything beat the baseline?
        base_mae = float(largest.loc[BASELINE, "mae_mean_eV"])
        beaten = [m for m in candidates.index if largest.loc[m, "mae_mean_eV"] < base_mae]
        if not beaten:
            lines.append(
                f"  At the full training set NO model beats the {BASELINE} "
                f"baseline ({base_mae:.4f} eV MAE)."
            )
            lines.append(
                "  On this data the ladder does not support a claim that any "
                "model has learned the target."
            )
        else:
            lines.append(
                f"  Beats the {BASELINE} baseline at the full training set: "
                f"{', '.join(beaten)}."
            )

    return lines


def make_figure(table: pd.DataFrame, floor: dict[str, float], out: Path) -> None:
    """MAE vs training-set size, one line per model, band = subset spread."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colours = {"mean": "#7f8c8d", "ridge": "#e67e22",
               "gp": "#1f3d63", "random_forest": "#8e44ad"}

    n_unstable = int(table["unstable"].sum())
    plotted = table[~table["unstable"]]

    for name, grp in plotted.groupby("model", sort=False):
        grp = grp.sort_values("n_train")
        colour = colours.get(name, None)
        ax.plot(grp["n_train"], grp["mae_mean_eV"], "-o", lw=1.9, ms=5,
                color=colour, label=name, zorder=3)
        ax.fill_between(grp["n_train"],
                        grp["mae_mean_eV"] - grp["mae_std_eV"],
                        grp["mae_mean_eV"] + grp["mae_std_eV"],
                        color=colour, alpha=0.16, zorder=1)

    ax.axhline(floor["operative_eV"], color="#c0392b", ls="--", lw=1.4,
               label=f"noise floor {floor['operative_eV']:.3f} eV", zorder=2)

    sizes = sorted(table["n_train"].unique())
    ax.set_xticks(sizes)
    ax.set_xlabel("training-set size (LOO run within each subset)")
    ax.set_ylabel("LOO MAE (eV)")
    ax.set_title("Learning curve: band = spread across random subsets")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    n_sub = table.groupby("n_train")["n_subsets"].max()
    caption = "subsets per size: " + ", ".join(
        f"n={int(k)}:{int(v)}" for k, v in n_sub.items()
    )
    if n_unstable:
        omitted = sorted(set(table[table["unstable"]]["model"]))
        caption += (f"  |  omitted as numerically unstable (p >> n): "
                    f"{', '.join(omitted)}")
    fig.text(0.5, -0.02, f"seed {RANDOM_SEED}  |  {caption}",
             ha="center", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description="LOO learning curve over growing subsets.")
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--h2-reference", default="data/h2_reference.json")
    ap.add_argument("--role", default="train")
    ap.add_argument("--orderings", type=int, default=DEFAULT_N_ORDERINGS,
                    help="Random orderings per size (capped by the number of "
                         "distinct subsets, which is then exhaustive).")
    ap.add_argument("--random-forest", action="store_true",
                    help="Include the random_forest rung.")
    ap.add_argument("--figure", default="figures/learning_curve.png")
    ap.add_argument("--out", default="results/learning_curve.csv")
    args = ap.parse_args()

    dataset = pd.read_csv(args.dataset)
    features = pd.read_csv(args.features)

    e_clean = get_clean_reference(dataset)
    e_ref_h = load_h2_reference(Path(args.h2_reference))
    targets = compute_targets(dataset, e_clean, e_ref_h)
    train = targets[(targets["role"] == args.role) & targets["E_ads_eV"].notna()]
    train = train[["config_id", "E_ads_eV"]].reset_index(drop=True)

    # Features cover the whole in-scope dataset; restrict to the rows that have
    # a target before aligning, so evaluate.align's strict one-to-one guard
    # stays intact rather than being loosened.
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    wanted = set(train["config_id"])
    subset = features[features["config_id"].isin(wanted)][["config_id", *numeric]]
    X, y, config_ids, feature_names = align(subset, train)

    table = curve(X, y, n_orderings=args.orderings,
                  include_random_forest=args.random_forest)
    floor = noise_floor_with_empirical(e_clean)

    table["seed"] = RANDOM_SEED
    table["noise_floor_eV"] = floor["operative_eV"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    make_figure(table, floor, Path(args.figure))

    width = 78
    print("=" * width)
    print("Learning curve -- LOO error over growing training subsets")
    print("=" * width)
    print(f"seed              : {RANDOM_SEED}")
    print(f"full training set : {len(y)}")
    print(f"features          : {len(feature_names)}")
    print(f"noise floor       : {floor['operative_eV']:.4f} eV (operative)")
    print()
    print(f"{'model':<15}{'n_train':>8}{'subsets':>9}{'MAE mean':>11}"
          f"{'MAE sd':>9}{'MAE min':>9}{'MAE max':>9}")
    print("-" * width)
    for _, row in table.iterrows():
        flag = "" if row["exhaustive"] else "*"
        print(f"{row['model']:<15}{int(row['n_train']):>8}"
              f"{int(row['n_subsets']):>8}{flag:<1}"
              f"{row['mae_mean_eV']:>11.4f}{row['mae_std_eV']:>9.4f}"
              f"{row['mae_min_eV']:>9.4f}{row['mae_max_eV']:>9.4f}")
    print()
    print("* sampled from random orderings; unmarked sizes enumerate every "
          "distinct subset.")
    print()
    print("Data-limited regime:")
    for line in data_limited_verdict(table, floor["operative_eV"]):
        print(line)
    print()
    print(f"wrote {args.figure}")
    print(f"wrote {out}")
    print("=" * width)


if __name__ == "__main__":
    main()
