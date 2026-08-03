#!/usr/bin/env python3
"""Are the GP's prediction intervals honest?

Under leave-one-out, every point is predicted by a model that never saw it, and
the GP reports a predictive sigma alongside each mean. This script asks the only
question that matters for an acquisition function: **when the GP claims a 68%
interval, do 68% of held-out points actually fall inside it?**

Reporting rules
---------------
* Empirical coverage is reported as a raw count (k of n) beside the nominal
  level, with a Wilson interval, because at small n coverage can only be
  measured in steps of 1/n.
* Miscalibration is stated plainly. The variance rescaling factor that *would*
  bring the intervals into line is reported as a diagnostic and is **never
  applied**. A silently rescaled interval is a fabricated uncertainty.

One command:

    python src/calibration.py --features data/features.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # deterministic, headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import (align, compute_targets, get_clean_reference,
                          load_h2_reference, loo_predict,
                          noise_floor_with_empirical)
from src.models import build_ladder
from src.seeds import RANDOM_SEED, set_seeds

# Nominal levels reported in the table. z values are the two-sided Gaussian
# critical values; kept as literals so no scipy dependency is needed.
NOMINAL_LEVELS = [
    ("50%", 0.674490),
    ("68%", 1.000000),   # the conventional one-sigma interval, 68.27%
    ("80%", 1.281552),
    ("90%", 1.644854),
    ("95%", 1.959964),
    ("99%", 2.575829),
]

# Coverage within this margin of nominal is not called miscalibrated, because at
# small n the estimate cannot resolve better than about 1/n.
def coverage_tolerance(n: int) -> float:
    return 1.0 / n if n else 1.0


def gaussian_two_sided(z: float) -> float:
    """Nominal two-sided coverage of +/- z sigma for a standard normal."""
    return math.erf(z / math.sqrt(2.0))


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Honest at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def gp_loo_predictions(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_SEED
                       ) -> tuple[np.ndarray, np.ndarray]:
    """LOO mean and sigma from the GP rung of the ladder."""
    gp = {r.name: r for r in build_ladder(seed=seed)}["gp"].estimator
    preds, sigmas = loo_predict(gp, X, y)
    if sigmas is None:
        raise RuntimeError("the GP rung returned no predictive sigma")
    return preds, sigmas


def calibration_table(y: np.ndarray, preds: np.ndarray, sigmas: np.ndarray
                      ) -> pd.DataFrame:
    """Empirical vs nominal coverage at each reported level."""
    n = len(y)
    z_scores = (y - preds) / sigmas
    rows = []
    for label, z_crit in NOMINAL_LEVELS:
        inside = int((np.abs(z_scores) <= z_crit).sum())
        lo, hi = wilson_interval(inside, n)
        nominal = gaussian_two_sided(z_crit)
        rows.append({
            "nominal_label": label,
            "nominal_coverage": nominal,
            "z_critical": z_crit,
            "inside": inside,
            "n": n,
            "empirical_coverage": inside / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "gap": inside / n - nominal,
        })
    return pd.DataFrame(rows)


def calibration_summary(y: np.ndarray, preds: np.ndarray,
                        sigmas: np.ndarray) -> dict[str, float]:
    """Scalar diagnostics. The rescale factor is reported, never applied."""
    z = (y - preds) / sigmas
    rms_z = float(np.sqrt(np.mean(z ** 2)))
    return {
        "n": len(y),
        "mean_sigma_eV": float(sigmas.mean()),
        "min_sigma_eV": float(sigmas.min()),
        "max_sigma_eV": float(sigmas.max()),
        "rms_residual_eV": float(np.sqrt(np.mean((y - preds) ** 2))),
        "mean_abs_z": float(np.abs(z).mean()),
        "rms_z": rms_z,
        # sigma * this factor would give rms_z == 1. Diagnostic only.
        "variance_rescale_factor_NOT_APPLIED": rms_z,
    }


def verdict(table: pd.DataFrame, summary: dict[str, float]) -> list[str]:
    """Plain-language calibration verdict. No hedging, no rescaling."""
    n = int(summary["n"])
    tol = coverage_tolerance(n)
    lines: list[str] = []

    for label in ("68%", "95%"):
        row = table[table["nominal_label"] == label].iloc[0]
        emp, nom = row["empirical_coverage"], row["nominal_coverage"]
        gap = emp - nom
        if abs(gap) <= tol:
            direction = "consistent with nominal"
        elif gap < 0:
            direction = "OVERCONFIDENT: intervals are too narrow"
        else:
            direction = "underconfident: intervals are too wide"
        lines.append(
            f"  {label} interval: {int(row['inside'])}/{n} inside "
            f"({emp:.0%} vs {nom:.0%} nominal) -- {direction}"
        )

    rms_z = summary["rms_z"]
    if rms_z > 1.0:
        lines.append(
            f"  RMS z = {rms_z:.2f} > 1: residuals are larger than the GP's own "
            "sigma, i.e. it understates its error."
        )
    elif rms_z < 1.0:
        lines.append(
            f"  RMS z = {rms_z:.2f} < 1: residuals are smaller than sigma, i.e. "
            "it overstates its error."
        )
    else:
        lines.append(f"  RMS z = {rms_z:.2f}: sigma matches the residual scale.")

    lines.append(
        f"  A variance rescaling of sigma x {rms_z:.3f} would set RMS z to 1. "
        "It has NOT been applied."
    )
    if n < 20:
        lines.append(
            f"  Coverage at n = {n} can only be measured in steps of "
            f"{tol:.0%}; treat these fractions as coarse."
        )
    return lines


def make_figure(y: np.ndarray, preds: np.ndarray, sigmas: np.ndarray,
                config_ids: list[str], floor: dict[str, float],
                out: Path) -> None:
    """Two panels: parity with error bars, and a reliability curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # --- parity with 1-sigma and 2-sigma bars ---
    order = np.argsort(y)
    ax1.errorbar(y[order], preds[order], yerr=1.96 * sigmas[order],
                 fmt="none", ecolor="#b0c4de", elinewidth=6,
                 capsize=0, label="95% interval", zorder=1)
    ax1.errorbar(y[order], preds[order], yerr=sigmas[order],
                 fmt="none", ecolor="#4a7ebb", elinewidth=2.5,
                 capsize=0, label="68% interval", zorder=2)
    ax1.scatter(y[order], preds[order], s=42, color="#1f3d63",
                zorder=3, label="LOO prediction")

    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    pad = 0.08 * (lims[1] - lims[0] or 1.0)
    lims = [lims[0] - pad, lims[1] + pad]
    ax1.plot(lims, lims, "k--", lw=1, label="perfect")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(r"$E_{\mathrm{ads}}$, LOO truth (eV)")
    ax1.set_ylabel(r"$E_{\mathrm{ads}}$, GP prediction (eV)")
    ax1.set_title(f"GP leave-one-out parity (n = {len(y)})")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    # --- reliability curve ---
    z_grid = np.linspace(0.01, 3.0, 300)
    nominal = np.array([gaussian_two_sided(z) for z in z_grid])
    z_scores = np.abs((y - preds) / sigmas)
    empirical = np.array([(z_scores <= z).mean() for z in z_grid])

    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax2.step(nominal, empirical, where="post", lw=2, color="#1f3d63",
             label="observed")
    for label, z_crit in (("68%", 1.0), ("95%", 1.959964)):
        nom = gaussian_two_sided(z_crit)
        emp = float((z_scores <= z_crit).mean())
        ax2.scatter([nom], [emp], s=70, zorder=4,
                    color="#c0392b" if abs(emp - nom) > 1 / len(y) else "#27ae60")
        ax2.annotate(f"{label}: {emp:.0%}", (nom, emp),
                     textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("nominal coverage")
    ax2.set_ylabel("empirical coverage (fraction inside)")
    ax2.set_title("Reliability: below the diagonal = overconfident")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25)

    fig.suptitle(
        f"GP interval calibration under LOO  |  seed {RANDOM_SEED}  |  "
        f"noise floor {floor['operative_eV']:.3f} eV  |  no rescaling applied",
        fontsize=9, y=1.00,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description="GP interval calibration under LOO.")
    ap.add_argument("--dataset", default="data/dataset.csv")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--h2-reference", default="data/h2_reference.json")
    ap.add_argument("--role", default="train")
    ap.add_argument("--figure", default="figures/calibration.png")
    ap.add_argument("--out", default="results/calibration.csv")
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

    preds, sigmas = gp_loo_predictions(X, y)
    table = calibration_table(y, preds, sigmas)
    summary = calibration_summary(y, preds, sigmas)
    floor = noise_floor_with_empirical(e_clean)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enriched = table.copy()
    for key, value in summary.items():
        enriched[key] = value
    enriched["seed"] = RANDOM_SEED
    enriched.to_csv(out, index=False)

    make_figure(y, preds, sigmas, config_ids, floor, Path(args.figure))

    width = 74
    print("=" * width)
    print("GP prediction-interval calibration under leave-one-out")
    print("=" * width)
    print(f"seed              : {RANDOM_SEED}")
    print(f"held-out points   : {len(y)}")
    print(f"features          : {len(feature_names)}")
    print(f"noise floor       : {floor['operative_eV']:.4f} eV (operative)")
    print()
    print(f"{'nominal':>8}{'z':>8}{'inside':>10}{'empirical':>12}"
          f"{'Wilson 95% CI':>22}")
    print("-" * width)
    for _, row in table.iterrows():
        ci = f"[{row['wilson_lo']:.2f}, {row['wilson_hi']:.2f}]"
        print(f"{row['nominal_label']:>8}{row['z_critical']:>8.3f}"
              f"{int(row['inside']):>6}/{int(row['n']):<3}"
              f"{row['empirical_coverage']:>12.0%}{ci:>22}")
    print()
    print("Predictive sigma:")
    print(f"  mean {summary['mean_sigma_eV']:.4f} eV, "
          f"range {summary['min_sigma_eV']:.4f} .. {summary['max_sigma_eV']:.4f} eV")
    print(f"  RMS LOO residual {summary['rms_residual_eV']:.4f} eV")
    print()
    print("Verdict:")
    for line in verdict(table, summary):
        print(line)
    print()
    print(f"wrote {args.figure}")
    print(f"wrote {out}")
    print("=" * width)


if __name__ == "__main__":
    main()
