#!/usr/bin/env python3
"""The modelling feature subset: a stated decision, fixed before any comparison.

Why a subset exists
-------------------
``data/features.csv`` carries the full 25-descriptor dictionary and remains the
record of what was computed. It is not the modelling matrix. With five converged
training points, 25 features means p/n = 5, which produced two visible failures:
ridge blew up numerically (standardising near-constant columns divides by almost
zero) and the GP returned predictive sigmas an order of magnitude too small on
two of five held-out points.

How this subset was chosen
--------------------------
On **physical grounds, before any performance was measured**, and it is not
revised in response to the numbers. Data-driven selection at n = 5 would choose
features using the same points the model is then scored on, which is leakage.
Each entry earns its place from surface chemistry, not from a score:

* ``gcn_mean`` -- generalised coordination number of the binding oxygen. The
  standard coordination-based descriptor for adsorption strength on oxides:
  under-coordinated sites bind more strongly.
* ``nn1_dist_A`` -- nearest-neighbour distance from the binding oxygen. Local
  bond-length environment, independent of the coordination count.
* ``n_bridge_sites`` -- site type, as the number of binding oxygens that are
  bridging (Ir coordination 2) rather than three-fold or terminal. Bridging
  versus in-plane oxygen is the primary site distinction on rutile (110).
* ``n_adsorbates`` -- coverage, 1H versus 2H. Lateral interaction between two
  adsorbates is a different energetic regime from an isolated adsorbate.

Known consequences, stated rather than hidden
---------------------------------------------
* ``n_adsorbates`` is constant across the current all-1H training set, so it
  carries no information *yet*. It is kept because it will be the distinguishing
  feature the moment a 2H pair converges, and adding it later in response to
  results would be exactly the iteration this decision forbids.
* ``n_bridge_sites`` separates ``slab_H_o44_ready`` (an edge site whose
  coordination is truncated by the non-periodic cell) from the other four. That
  is a real geometric difference, not a chemical one; see ``lateral_margin_A``
  in the full dictionary.

This module writes ``data/features_model.csv``, containing only these columns.
Every downstream script is pointed at that file, so the subset is applied by
data rather than by editing the models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.seeds import RANDOM_SEED, set_seeds

# The modelling features, in a fixed order. Chosen on physical grounds before
# any performance comparison. DO NOT reorder, extend, or prune in response to
# results: that would turn a stated prior decision into data-driven selection.
MODEL_FEATURES: tuple[str, ...] = (
    "gcn_mean",
    "nn1_dist_A",
    "n_bridge_sites",
    "n_adsorbates",
)

RATIONALE: tuple[tuple[str, str], ...] = (
    ("gcn_mean",
     "Generalised coordination number of the binding oxygen. Coordination-based "
     "descriptors are the standard proxy for adsorption strength on oxide "
     "surfaces: under-coordinated sites bind more strongly."),
    ("nn1_dist_A",
     "Nearest-neighbour distance from the binding oxygen. Captures the local "
     "bond-length environment independently of the coordination count."),
    ("n_bridge_sites",
     "Site type: how many binding oxygens are bridging (Ir coordination 2) "
     "rather than three-fold or terminal. Bridging versus in-plane oxygen is "
     "the primary site distinction on rutile (110)."),
    ("n_adsorbates",
     "Coverage, 1H versus 2H. Two adsorbates interact laterally, which is a "
     "different energetic regime from an isolated adsorbate. Constant across "
     "the present all-1H training set and kept deliberately, so that it is "
     "already in place when a pair converges."),
)


def subset_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Return config_id plus the modelling features, in the fixed order."""
    missing = [c for c in MODEL_FEATURES if c not in features.columns]
    if missing:
        raise KeyError(
            f"features table is missing modelling feature(s): {missing}. "
            "Rebuild data/features.csv with src/featurise.py."
        )
    out = features[["config_id", *MODEL_FEATURES]].copy()
    return out.sort_values("config_id", kind="mergesort").reset_index(drop=True)


def rationale_markdown() -> str:
    """Markdown for the README. Generated, never hand-edited."""
    lines = [
        "### Modelling feature subset",
        "",
        f"The full dictionary above records all descriptors computed. "
        f"{len(MODEL_FEATURES)} of them are used for modelling, chosen on "
        "physical grounds **before** any performance comparison and not revised "
        "in response to results (data-driven selection at this sample size "
        "would leak).",
        "",
        "| feature | why this one |",
        "| --- | --- |",
    ]
    for name, why in RATIONALE:
        lines.append(f"| `{name}` | {why} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    set_seeds()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="data/features_model.csv")
    ap.add_argument("--rationale-md", default="data/feature_subset_rationale.md")
    ap.add_argument("--dataset", default="data/dataset.csv",
                    help="Used only with --role, to look roles up.")
    ap.add_argument("--role", default=None,
                    help="Restrict output rows to this dataset role. "
                         "src/evaluate.py requires a features table that "
                         "matches its targets one-to-one, so it is given a "
                         "role=train file; scripts that filter internally "
                         "get the unrestricted file.")
    args = ap.parse_args()

    features = pd.read_csv(args.features)
    subset = subset_frame(features)

    if args.role:
        dataset = pd.read_csv(args.dataset)
        keep = set(dataset[dataset["role"] == args.role]["config_id"])
        subset = subset[subset["config_id"].isin(keep)].reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False)
    Path(args.rationale_md).write_text(rationale_markdown())

    print("=" * 70)
    print("Modelling feature subset -- stated decision, fixed in advance")
    print("=" * 70)
    print(f"seed                : {RANDOM_SEED}")
    print(f"full dictionary     : {len([c for c in features.columns])} columns in "
          f"{args.features}")
    print(f"modelling features  : {len(MODEL_FEATURES)}")
    for name in MODEL_FEATURES:
        n_unique = subset[name].nunique()
        note = "  (constant across these rows)" if n_unique == 1 else ""
        print(f"    {name:<18s} unique values: {n_unique}{note}")
    print(f"role filter         : {args.role or '(none)'}")
    print(f"rows                : {len(subset)}")
    print()
    print("Chosen on physical grounds before any performance comparison.")
    print("Not to be revised in response to results.")
    print()
    print(f"wrote {out}")
    print(f"wrote {args.rationale_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()
