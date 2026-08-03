#!/usr/bin/env python3
"""Which pending configuration should be optimised next?

Each optimisation costs about 23 minutes, so the ordering of the queue is the
decision this module supports. Two outputs, deliberately separated:

1. **The live recommendation.** Fit the GP on the converged points and rank the
   pending candidate pool by predictive uncertainty. This is what an
   uncertainty-driven acquisition function would choose.

2. **The retrospective check.** On converged points only, replay acquisition
   from small starting sets under two policies -- pick-by-max-uncertainty and
   pick-at-random -- and compare held-out error against points acquired. This
   asks whether the ranking in (1) is worth trusting. If uncertainty does not
   beat random at this sample size, that is the finding, and it is reported as
   such rather than smoothed over.

Two caveats that the numbers cannot express on their own, both printed at run
time:

* The GP's intervals are measurably overconfident under leave-one-out (see
  ``src/calibration.py``). The uncertainty ranking is therefore an *ordering*
  signal at best; the sigma values are not trustworthy magnitudes.
* Every 2H candidate is out of distribution in coverage: no 2H configuration has
  converged, so ``n_adsorbates`` is constant in training and the kernel has no
  length scale for it. High sigma on the pairs reflects that absence, not a
  chemical insight.

One command:

    python src/active_learning.py --features data/features_model.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # deterministic, headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import calibration_summary, gp_loo_predictions
from src.evaluate import (align, compute_targets, get_clean_reference,
                          load_h2_reference, noise_floor_with_empirical)
from src.models import build_ladder, predict_with_std
from src.seeds import RANDOM_SEED, set_seeds

# Wall-clock cost of one optimisation, from the existing run records.
MINUTES_PER_OPTIMISATION = 23.0

# Acquisition simulation: start from every subset of this size, then add one at
# a time. Starting sets are enumerated exhaustively, so no sampling is involved
# in the starting condition.
START_SIZE = 2

# Random-policy repeats per starting set.
N_RANDOM_REPEATS = 20

# Size of the fixed held-out set in the unconfounded variant of the policy
# check (see simulate_fixed_holdout).
HOLDOUT_SIZE = 2

# If the spread of predictive sigma across the pool is below this fraction of
# its mean, the ranking is not ordering anything: it is ordering numerical
# noise. Reported as degenerate rather than presented as a recommendation.
DEGENERACY_FRACTION = 0.01


def gp_estimator(seed: int = RANDOM_SEED):
    """The GP rung of the ladder, unmodified."""
    return {r.name: r for r in build_ladder(seed=seed)}["gp"].estimator


# --------------------------------------------------------------------------
# 1. live recommendation
# --------------------------------------------------------------------------

def rank_candidates(X_train: np.ndarray, y_train: np.ndarray,
                    X_pool: np.ndarray, pool_meta: pd.DataFrame,
                    seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Rank the pending pool by GP predictive uncertainty, descending."""
    model = gp_estimator(seed).fit(X_train, y_train)
    mean, sigma = predict_with_std(model, X_pool)
    if sigma is None:
        raise RuntimeError("the GP rung returned no predictive sigma")

    out = pool_meta.copy().reset_index(drop=True)
    out["predicted_E_ads_eV"] = mean
    out["predictive_sigma_eV"] = sigma
    # Deterministic ordering: sigma descending, then config_id for ties.
    out = out.sort_values(["predictive_sigma_eV", "config_id"],
                          ascending=[False, True], kind="mergesort")
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["cumulative_cost_minutes"] = out["rank"] * MINUTES_PER_OPTIMISATION
    return out


def ranking_degeneracy(ranking: pd.DataFrame) -> dict[str, float | bool]:
    """Is the uncertainty ranking actually discriminating between candidates?

    A GP whose length scale has collapsed reverts to its prior everywhere, so
    every candidate receives the same mean and the same sigma. The ranking then
    looks like a recommendation while carrying no information at all. This is
    the check that catches it.
    """
    sigma = ranking["predictive_sigma_eV"].to_numpy(dtype=float)
    mean = ranking["predicted_E_ads_eV"].to_numpy(dtype=float)
    sigma_span = float(sigma.max() - sigma.min())
    mean_span = float(mean.max() - mean.min())
    rel = sigma_span / float(sigma.mean()) if sigma.mean() else 0.0
    return {
        "sigma_mean_eV": float(sigma.mean()),
        "sigma_span_eV": sigma_span,
        "sigma_relative_span": rel,
        "prediction_span_eV": mean_span,
        "degenerate": bool(rel < DEGENERACY_FRACTION),
    }


# --------------------------------------------------------------------------
# 2. retrospective acquisition simulation
# --------------------------------------------------------------------------

def _held_out_mae(X: np.ndarray, y: np.ndarray, acquired: list[int],
                  seed: int) -> float:
    """MAE on the not-yet-acquired points, from a GP fit on the acquired ones."""
    remaining = [i for i in range(len(y)) if i not in acquired]
    if not remaining:
        return float("nan")
    model = gp_estimator(seed).fit(X[acquired], y[acquired])
    preds = model.predict(X[remaining])
    return float(np.abs(y[remaining] - preds).mean())


def _acquire_by_uncertainty(X: np.ndarray, y: np.ndarray,
                            acquired: list[int], seed: int) -> int:
    """Index of the remaining point with the largest predictive sigma."""
    remaining = [i for i in range(len(y)) if i not in acquired]
    model = gp_estimator(seed).fit(X[acquired], y[acquired])
    _, sigma = predict_with_std(model, X[remaining])
    if sigma is None:
        raise RuntimeError("the GP rung returned no predictive sigma")
    # Ties broken by lowest index, so the policy is deterministic.
    return int(remaining[int(np.argmax(sigma))])


def simulate_acquisition(X: np.ndarray, y: np.ndarray,
                         start_size: int = START_SIZE,
                         n_random_repeats: int = N_RANDOM_REPEATS,
                         seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Replay both policies from every starting subset. One row per measurement.

    Error is MAE on the points not yet acquired, so it is genuinely held out.
    The final step is omitted because at that point nothing is held out.
    """
    n = len(y)
    if n < start_size + 2:
        raise ValueError(
            f"need at least {start_size + 2} points to compare policies, got {n}"
        )

    rows = []
    for start in combinations(range(n), start_size):
        start_list = list(start)
        start_key = "+".join(str(i) for i in start_list)

        # --- max-uncertainty policy: deterministic given the start ---
        acquired = list(start_list)
        rows.append({"policy": "max_uncertainty", "start": start_key, "repeat": 0,
                     "n_acquired": len(acquired),
                     "held_out_mae_eV": _held_out_mae(X, y, acquired, seed)})
        while len(acquired) < n - 1:
            acquired.append(_acquire_by_uncertainty(X, y, acquired, seed))
            rows.append({"policy": "max_uncertainty", "start": start_key, "repeat": 0,
                         "n_acquired": len(acquired),
                         "held_out_mae_eV": _held_out_mae(X, y, acquired, seed)})

        # --- random policy: repeated with fixed, start-specific seeds ---
        for repeat in range(n_random_repeats):
            rng = np.random.default_rng(
                seed + 1000 * (repeat + 1) + sum(start_list)
            )
            acquired = list(start_list)
            rows.append({"policy": "random", "start": start_key, "repeat": repeat,
                         "n_acquired": len(acquired),
                         "held_out_mae_eV": _held_out_mae(X, y, acquired, seed)})
            while len(acquired) < n - 1:
                remaining = [i for i in range(n) if i not in acquired]
                acquired.append(int(rng.choice(remaining)))
                rows.append({"policy": "random", "start": start_key,
                             "repeat": repeat, "n_acquired": len(acquired),
                             "held_out_mae_eV": _held_out_mae(X, y, acquired, seed)})

    return pd.DataFrame(rows)


def simulate_fixed_holdout(X: np.ndarray, y: np.ndarray,
                           holdout_size: int = HOLDOUT_SIZE,
                           n_random_repeats: int = N_RANDOM_REPEATS,
                           seed: int = RANDOM_SEED) -> pd.DataFrame:
    """The unconfounded variant of the policy comparison.

    ``simulate_acquisition`` scores each policy on the points it has NOT yet
    acquired. That comparison is structurally biased in favour of
    max-uncertainty: the policy deliberately takes the hardest points first, so
    what it leaves behind is the easiest to predict. The apparent win can be an
    artefact of the evaluation, not a benefit of the policy.

    Here the held-out set is fixed in advance and never acquired by either
    policy, so both are scored on exactly the same points. Every holdout split
    is enumerated, so nothing is sampled except the random policy itself.
    """
    n = len(y)
    if n < holdout_size + 2:
        raise ValueError(
            f"need at least {holdout_size + 2} points, got {n}"
        )

    rows = []
    for holdout in combinations(range(n), holdout_size):
        held = list(holdout)
        pool = [i for i in range(n) if i not in held]
        key = "+".join(str(i) for i in held)

        def score(acquired: list[int]) -> float:
            model = gp_estimator(seed).fit(X[acquired], y[acquired])
            preds = model.predict(X[held])
            return float(np.abs(y[held] - preds).mean())

        # --- max-uncertainty, deterministic ---
        for first in pool:
            acquired = [first]
            rows.append({"policy": "max_uncertainty", "holdout": key,
                         "first": first, "repeat": 0,
                         "n_acquired": 1, "pool_size": len(pool),
                         "fixed_holdout_mae_eV": score(acquired)})
            while len(acquired) < len(pool):
                rest = [i for i in pool if i not in acquired]
                model = gp_estimator(seed).fit(X[acquired], y[acquired])
                _, sigma = predict_with_std(model, X[rest])
                acquired.append(int(rest[int(np.argmax(sigma))]))
                rows.append({"policy": "max_uncertainty", "holdout": key,
                             "first": first, "repeat": 0,
                             "n_acquired": len(acquired),
                             "pool_size": len(pool),
                             "fixed_holdout_mae_eV": score(acquired)})

            # --- random, from the same first point ---
            for repeat in range(n_random_repeats):
                rng = np.random.default_rng(
                    seed + 7919 * (repeat + 1) + 31 * first + sum(held)
                )
                acquired = [first]
                rows.append({"policy": "random", "holdout": key, "first": first,
                             "repeat": repeat, "n_acquired": 1,
                             "pool_size": len(pool),
                             "fixed_holdout_mae_eV": score(acquired)})
                while len(acquired) < len(pool):
                    rest = [i for i in pool if i not in acquired]
                    acquired.append(int(rng.choice(rest)))
                    rows.append({"policy": "random", "holdout": key,
                                 "first": first, "repeat": repeat,
                                 "n_acquired": len(acquired),
                                 "pool_size": len(pool),
                                 "fixed_holdout_mae_eV": score(acquired)})

    return pd.DataFrame(rows)


def paired_comparison_fixed(runs: pd.DataFrame,
                            noise_floor_eV: float = 0.0) -> pd.DataFrame:
    """Paired uncertainty-vs-random on the fixed-holdout runs.

    Pairing is on (holdout split, first point), so both policies start from
    identical information and are scored on identical points.
    """
    rows = []
    for n_acq in sorted(runs["n_acquired"].unique()):
        at_n = runs[runs["n_acquired"] == n_acq]
        diffs = []
        for _, grp in at_n.groupby(["holdout", "first"]):
            unc = grp[grp["policy"] == "max_uncertainty"]["fixed_holdout_mae_eV"]
            rnd = grp[grp["policy"] == "random"]["fixed_holdout_mae_eV"]
            if unc.empty or rnd.empty:
                continue
            diffs.append(float(rnd.mean() - unc.iloc[0]))
        if not diffs:
            continue
        arr = np.array(diffs, dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        pool_size = int(at_n["pool_size"].iloc[0])
        # When every pool point has been acquired, both policies hold identical
        # training sets: the step cannot distinguish them and any difference is
        # fit-ordering noise. Marked vacuous rather than reported as a result.
        vacuous = bool(n_acq >= pool_size)
        real = bool(arr.mean() > se and se > 0 and not vacuous
                    and abs(arr.mean()) > noise_floor_eV)
        rows.append({
            "n_acquired": int(n_acq),
            "pool_size": pool_size,
            "n_pairs": len(arr),
            "mean_gain_eV": float(arr.mean()),
            "gain_se_eV": se,
            "pairs_where_uncertainty_wins": int((arr > 0).sum()),
            "pairs_where_random_wins": int((arr < 0).sum()),
            "vacuous_pool_exhausted": vacuous,
            "gain_exceeds_noise_floor": bool(abs(arr.mean()) > noise_floor_eV),
            "uncertainty_beats_random": real,
        })
    return pd.DataFrame(rows)


def summarise_acquisition(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of held-out MAE per policy and acquisition count."""
    grouped = runs.groupby(["policy", "n_acquired"])["held_out_mae_eV"]
    out = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    out = out.rename(columns={"count": "n_runs", "mean": "mae_mean_eV",
                              "std": "mae_std_eV", "min": "mae_min_eV",
                              "max": "mae_max_eV"})
    out["mae_std_eV"] = out["mae_std_eV"].fillna(0.0)
    out["mae_se_eV"] = out["mae_std_eV"] / np.sqrt(out["n_runs"].clip(lower=1))
    return out


def summarise_fixed(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of fixed-holdout MAE per policy and acquisition count."""
    g = runs.groupby(["policy", "n_acquired"])["fixed_holdout_mae_eV"]
    out = g.agg(["count", "mean", "std"]).reset_index()
    out = out.rename(columns={"count": "n_runs", "mean": "mae_mean_eV",
                              "std": "mae_std_eV"})
    out["mae_std_eV"] = out["mae_std_eV"].fillna(0.0)
    return out


def paired_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    """Per starting set, uncertainty vs the mean of random. Paired, so the
    difficulty of a particular starting set cancels."""
    rows = []
    for n_acq in sorted(runs["n_acquired"].unique()):
        at_n = runs[runs["n_acquired"] == n_acq]
        diffs = []
        for start, grp in at_n.groupby("start"):
            unc = grp[grp["policy"] == "max_uncertainty"]["held_out_mae_eV"]
            rnd = grp[grp["policy"] == "random"]["held_out_mae_eV"]
            if unc.empty or rnd.empty:
                continue
            # positive = uncertainty is better (lower error) than random
            diffs.append(float(rnd.mean() - unc.iloc[0]))
        if not diffs:
            continue
        arr = np.array(diffs, dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        rows.append({
            "n_acquired": int(n_acq),
            "n_starts": len(arr),
            "mean_gain_eV": float(arr.mean()),
            "gain_se_eV": se,
            "starts_where_uncertainty_wins": int((arr > 0).sum()),
            "starts_where_random_wins": int((arr < 0).sum()),
            "ties": int((arr == 0).sum()),
            # A gain is only called real if it clears its own standard error.
            "uncertainty_beats_random": bool(arr.mean() > se and se > 0),
        })
    return pd.DataFrame(rows)


def acquisition_verdict(paired: pd.DataFrame, n_train: int) -> list[str]:
    """State plainly whether uncertainty sampling earned its keep."""
    lines: list[str] = []
    if paired.empty:
        return ["  No comparable acquisition steps."]

    # The first step is identical under both policies (same starting set, no
    # choice made yet), so it carries no information about the policies.
    informative = paired[paired["n_acquired"] > START_SIZE]
    if informative.empty:
        return ["  No acquisition step distinguishes the policies at this n."]

    any_real = False
    for _, row in informative.iterrows():
        n_acq = int(row["n_acquired"])
        gain, se = row["mean_gain_eV"], row["gain_se_eV"]
        wins, losses = int(row["starts_where_uncertainty_wins"]), int(row["starts_where_random_wins"])
        if row["uncertainty_beats_random"]:
            any_real = True
            call = "uncertainty BEATS random"
        elif gain < -se and se > 0:
            call = "random beats uncertainty"
        else:
            call = "indistinguishable"
        lines.append(
            f"  {n_acq} acquired: mean gain {gain:+.4f} +/- {se:.4f} eV "
            f"({wins} starts to {losses}) -- {call}"
        )

    lines.append("")
    if any_real:
        lines.append(
            "  Uncertainty sampling beats random at at least one step by more "
            "than its own standard error."
        )
    else:
        lines.append(
            "  FINDING: uncertainty sampling does NOT beat random selection at "
            f"n = {n_train}."
        )
        lines.append(
            "  Five points do not support a surrogate yet, so they do not "
            "support an acquisition"
        )
        lines.append(
            "  function built on that surrogate either. These diagnostics will "
            "be rerun as the"
        )
        lines.append(
            "  restarts and 2H pairs extend the training set."
        )
    return lines


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def make_figure(ranking: pd.DataFrame, summary: pd.DataFrame,
                paired: pd.DataFrame, floor: dict[str, float],
                n_train: int, out: Path, top_n: int = 12,
                degeneracy: dict | None = None) -> None:
    """One figure: the live queue on the left, the policy check on the right."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- left: candidate ranking by predictive sigma ---
    top = ranking.head(top_n).iloc[::-1]
    colours = ["#8e44ad" if c == "2H" else "#1f77b4" for c in top["coverage"]]
    ax1.barh(np.arange(len(top)), top["predictive_sigma_eV"], color=colours)
    ax1.set_yticks(np.arange(len(top)))
    ax1.set_yticklabels(
        [c.replace("slab_", "").replace("_ready", "") for c in top["config_id"]],
        fontsize=8,
    )
    ax1.set_xlabel(r"GP predictive $\sigma$ (eV)")
    ax1.set_title(f"Pending queue by uncertainty (top {len(top)} of {len(ranking)})",
                  fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#8e44ad"),
               plt.Rectangle((0, 0), 1, 1, color="#1f77b4")]
    ax1.legend(handles, ["2H (out of distribution)", "1H"],
               fontsize=8, loc="lower right")
    ax1.grid(alpha=0.25, axis="x")
    if degeneracy and degeneracy["degenerate"]:
        ax1.text(0.5, 0.5, "DEGENERATE\nall sigma equal to within "
                 f"{degeneracy['sigma_relative_span']:.2%}\n"
                 "ranking carries no information",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=11, color="#c0392b", weight="bold",
                 bbox=dict(facecolor="white", edgecolor="#c0392b", alpha=0.93))

    # --- right: acquisition policies ---
    style = {"max_uncertainty": ("#c0392b", "-o"),
             "random": ("#7f8c8d", "--s")}
    for policy, grp in summary.groupby("policy"):
        grp = grp.sort_values("n_acquired")
        colour, marker = style.get(policy, ("#333333", "-o"))
        ax2.plot(grp["n_acquired"], grp["mae_mean_eV"], marker, lw=1.9, ms=5,
                 color=colour, label=policy, zorder=3)
        ax2.fill_between(grp["n_acquired"],
                         grp["mae_mean_eV"] - grp["mae_std_eV"],
                         grp["mae_mean_eV"] + grp["mae_std_eV"],
                         color=colour, alpha=0.15, zorder=1)

    ax2.axhline(floor["operative_eV"], color="#27ae60", ls=":", lw=1.5,
                label=f"noise floor {floor['operative_eV']:.3f} eV", zorder=2)
    ax2.set_xticks(sorted(summary["n_acquired"].unique()))
    ax2.set_xlabel("points acquired")
    ax2.set_ylabel("MAE on the fixed held-out set (eV)")
    ax2.set_title("Acquisition policy, unconfounded (fixed holdout)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_ylim(bottom=0)

    if not paired.empty:
        informative = paired[paired["n_acquired"] > 1]
        if not informative.empty and not informative["uncertainty_beats_random"].any():
            ax2.text(0.5, 0.06,
                     "uncertainty does not beat random at this n",
                     transform=ax2.transAxes, ha="center", fontsize=9,
                     color="#c0392b",
                     bbox=dict(facecolor="white", edgecolor="#c0392b", alpha=0.9))

    fig.suptitle(
        f"Active learning on {n_train} converged points  |  seed {RANDOM_SEED}  |  "
        f"~{MINUTES_PER_OPTIMISATION:.0f} min per optimisation",
        fontsize=9, y=1.00,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description="Rank the pending queue; check the policy.")
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--features", default="data/features_model.csv")
    ap.add_argument("--h2-reference", default="data/h2_reference.json")
    ap.add_argument("--train-role", default="train")
    ap.add_argument("--pool-role", default="candidate")
    ap.add_argument("--figure", default="figures/active_learning.png")
    ap.add_argument("--ranking-out", default="results/candidate_ranking.csv")
    ap.add_argument("--policy-out", default="results/acquisition_policy.csv")
    ap.add_argument("--random-repeats", type=int, default=N_RANDOM_REPEATS)
    args = ap.parse_args()

    dataset = pd.read_csv(args.dataset)
    features = pd.read_csv(args.features)

    e_clean = get_clean_reference(dataset)
    e_ref_h = load_h2_reference(Path(args.h2_reference))
    targets = compute_targets(dataset, e_clean, e_ref_h)
    floor = noise_floor_with_empirical(e_clean)

    # --- training set ---
    train_rows = targets[(targets["role"] == args.train_role)
                         & targets["E_ads_eV"].notna()]
    train = train_rows[["config_id", "E_ads_eV"]].reset_index(drop=True)
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    train_features = features[features["config_id"].isin(set(train["config_id"]))]
    X_train, y_train, train_ids, feature_names = align(
        train_features[["config_id", *numeric]], train
    )

    # --- pending pool: adsorption configurations only ---
    pool = targets[targets["role"] == args.pool_role].copy()
    n_pool_all = len(pool)
    pool = pool[pool["n_adsorbates"] >= 1]
    n_clean_dropped = n_pool_all - len(pool)

    pool_meta = pool[["config_id", "coverage", "n_adsorbates", "triage_bin",
                      "status", "site_id"]].reset_index(drop=True)
    pool_features = features[features["config_id"].isin(set(pool_meta["config_id"]))]
    pool_features = pool_features.sort_values("config_id", kind="mergesort")
    pool_meta = pool_meta.sort_values("config_id", kind="mergesort").reset_index(drop=True)
    missing = set(pool_meta["config_id"]) - set(pool_features["config_id"])
    if missing:
        raise ValueError(f"pending configurations without features: {sorted(missing)}")
    X_pool = pool_features[numeric].to_numpy(dtype=float)

    ranking = rank_candidates(X_train, y_train, X_pool, pool_meta)
    degeneracy = ranking_degeneracy(ranking)

    # --- retrospective policy check ---
    runs = simulate_acquisition(X_train, y_train,
                                n_random_repeats=args.random_repeats)
    summary = summarise_acquisition(runs)
    paired = paired_comparison(runs)

    fixed_runs = simulate_fixed_holdout(X_train, y_train,
                                        n_random_repeats=args.random_repeats)
    fixed_summary = summarise_fixed(fixed_runs)
    fixed_paired = paired_comparison_fixed(fixed_runs,
                                          floor["operative_eV"])

    # --- calibration context, measured not assumed ---
    loo_preds, loo_sigmas = gp_loo_predictions(X_train, y_train)
    cal = calibration_summary(y_train, loo_preds, loo_sigmas)

    for frame in (ranking, summary, paired):
        frame["seed"] = RANDOM_SEED
    ranking["gp_rms_z_under_loo"] = cal["rms_z"]

    Path(args.ranking_out).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.ranking_out, index=False)
    for frame in (fixed_summary, fixed_paired):
        frame["seed"] = RANDOM_SEED
    combined = summary.merge(paired, on="n_acquired", how="outer",
                            suffixes=("", "_paired"))
    combined["evaluation"] = "not_yet_acquired_pool_CONFOUNDED"
    fixed_combined = fixed_summary.merge(fixed_paired, on="n_acquired",
                                        how="outer", suffixes=("", "_paired"))
    fixed_combined["evaluation"] = "fixed_holdout_UNCONFOUNDED"
    pd.concat([combined, fixed_combined], ignore_index=True) \
      .to_csv(args.policy_out, index=False)

    make_figure(ranking, fixed_summary, fixed_paired, floor, len(y_train),
                Path(args.figure), degeneracy=degeneracy)

    # ------------------------------------------------------------------
    width = 92
    print("=" * width)
    print("Active learning -- which pending configuration to optimise next")
    print("=" * width)
    print(f"seed                : {RANDOM_SEED}")
    print(f"converged training  : {len(y_train)} points, {len(feature_names)} features "
          f"({', '.join(feature_names)})")
    print(f"pending pool        : {len(ranking)} adsorption configurations")
    if n_clean_dropped:
        print(f"                      ({n_clean_dropped} clean-slab row(s) excluded: "
              "not an adsorption configuration)")
    print(f"cost per run        : ~{MINUTES_PER_OPTIMISATION:.0f} min")
    print()

    print("-" * width)
    print("1. LIVE RECOMMENDATION -- pending pool ranked by GP predictive uncertainty")
    print("-" * width)
    header = (f"{'#':>3} {'configuration':<26}{'cov':>5}{'triage_bin':>21}"
              f"{'pred E_ads':>12}{'sigma':>9}")
    print(header)
    print("-" * width)
    for _, row in ranking.iterrows():
        print(f"{int(row['rank']):>3} "
              f"{row['config_id'].replace('slab_', '').replace('_ready', ''):<26}"
              f"{row['coverage']:>5}{row['triage_bin']:>21}"
              f"{row['predicted_E_ads_eV']:>12.3f}{row['predictive_sigma_eV']:>9.3f}")
    print()
    if degeneracy["degenerate"]:
        print("!" * width)
        print("THIS RANKING IS DEGENERATE AND MUST NOT BE USED TO CHOOSE.")
        print(f"  Predictive sigma spans only {degeneracy['sigma_span_eV']:.2e} eV "
              f"about a mean of {degeneracy['sigma_mean_eV']:.3f} eV")
        print(f"  ({degeneracy['sigma_relative_span']:.3%} relative spread); the "
              f"predictions span {degeneracy['prediction_span_eV']:.2e} eV.")
        print("  The GP has reverted to its prior for every candidate: the kernel")
        print("  found no usable length scale in 5 training points, so no candidate")
        print("  is distinguishable from any other. The order above is numerical")
        print("  noise, not a preference. Choose the next run on scientific grounds.")
        print("!" * width)
        print()
    print("Read this ranking with two caveats:")
    print(f"  * The GP is overconfident under LOO (RMS z = {cal['rms_z']:.2f}); "
          "sigma is an ordering")
    print("    signal at best, not a trustworthy magnitude.")
    n_2h = int((ranking["coverage"] == "2H").sum())
    if n_2h:
        print(f"  * All {n_2h} 2H candidates are out of distribution in coverage: no 2H "
              "configuration has")
        print("    converged, so the kernel has no length scale for n_adsorbates. Their "
              "high sigma")
        print("    reflects that absence, not chemistry.")
    print()

    print("-" * width)
    print("2. RETROSPECTIVE CHECK -- does max-uncertainty beat random?")
    print("-" * width)
    print(f"starting sets       : all {len(list(combinations(range(len(y_train)), START_SIZE)))} "
          f"subsets of size {START_SIZE} (exhaustive)")
    print(f"random repeats      : {args.random_repeats} per starting set")
    print(f"error measured on   : the not-yet-acquired points (genuinely held out)")
    print()
    print(f"{'policy':<18}{'acquired':>9}{'runs':>7}{'MAE mean':>11}{'MAE sd':>9}"
          f"{'MAE min':>9}{'MAE max':>9}")
    print("-" * width)
    for _, row in summary.sort_values(["n_acquired", "policy"]).iterrows():
        print(f"{row['policy']:<18}{int(row['n_acquired']):>9}{int(row['n_runs']):>7}"
              f"{row['mae_mean_eV']:>11.4f}{row['mae_std_eV']:>9.4f}"
              f"{row['mae_min_eV']:>9.4f}{row['mae_max_eV']:>9.4f}")
    print()
    print("Verdict:")
    for line in acquisition_verdict(paired, len(y_train)):
        print(line)
    print()
    print("-" * width)
    print("2b. UNCONFOUNDED RE-CHECK -- fixed held-out set")
    print("-" * width)
    print("The comparison above scores each policy on the points it has NOT yet")
    print("acquired. That favours max-uncertainty by construction: it takes the")
    print("hardest points first, so what it leaves behind is the easiest to predict.")
    print("Here the held-out set is fixed in advance and never acquired, so both")
    print(f"policies are scored on identical points ({HOLDOUT_SIZE} held out, every "
          "split enumerated).")
    print()
    print(f"{'policy':<18}{'acquired':>9}{'runs':>7}{'MAE mean':>11}{'MAE sd':>9}")
    print("-" * width)
    for _, row in fixed_summary.sort_values(["n_acquired", "policy"]).iterrows():
        print(f"{row['policy']:<18}{int(row['n_acquired']):>9}"
              f"{int(row['n_runs']):>7}{row['mae_mean_eV']:>11.4f}"
              f"{row['mae_std_eV']:>9.4f}")
    print()
    print("Verdict on the unconfounded comparison:")
    for _, row in fixed_paired[fixed_paired["n_acquired"] > 1].iterrows():
        if row["vacuous_pool_exhausted"]:
            call = ("VACUOUS: whole pool acquired, both policies hold identical "
                    "sets")
        elif row["uncertainty_beats_random"]:
            call = "uncertainty BEATS random"
        elif not row["gain_exceeds_noise_floor"]:
            call = "indistinguishable (gain below the noise floor)"
        elif row["mean_gain_eV"] < -row["gain_se_eV"] and row["gain_se_eV"] > 0:
            call = "random beats uncertainty"
        else:
            call = "indistinguishable"
        print(f"  {int(row['n_acquired'])} acquired: mean gain "
              f"{row['mean_gain_eV']:+.4f} +/- {row['gain_se_eV']:.4f} eV "
              f"({int(row['pairs_where_uncertainty_wins'])} pairs to "
              f"{int(row['pairs_where_random_wins'])}) -- {call}")

    informative = fixed_paired[(fixed_paired["n_acquired"] > 1)
                               & ~fixed_paired["vacuous_pool_exhausted"]]
    if informative.empty:
        print("  No step distinguishes the policies at this n.")
    else:
        if not informative["uncertainty_beats_random"].any():
            print()
            print("  FINDING: with the evaluation bias removed, uncertainty sampling")
            print(f"  does NOT beat random selection at n = {len(y_train)}. Five points do")
            print("  not support a surrogate yet, so they do not support an acquisition")
            print("  function built on one. These diagnostics will be rerun as the")
            print("  restarts and 2H pairs extend the training set.")
    print()
    last = int(summary["n_acquired"].max())
    print(f"Note: at {last} acquired, only {len(y_train) - last} point(s) remain held "
          "out, so that column")
    print("      rests on very few measurements. The first step is identical under both")
    print("      policies by construction.")
    print()
    print(f"wrote {args.figure}")
    print(f"wrote {args.ranking_out}")
    print(f"wrote {args.policy_out}")
    print("=" * width)


if __name__ == "__main__":
    main()
