"""Single seeding entry point. Every script in this repo calls set_seeds() first.

Keeping this in one place means the project-wide seed is auditable from one line
rather than scattered across scripts.
"""

from __future__ import annotations

import os
import random

import numpy as np

RANDOM_SEED = 0


def set_seeds(seed: int = RANDOM_SEED) -> int:
    """Seed every source of randomness this project uses. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
