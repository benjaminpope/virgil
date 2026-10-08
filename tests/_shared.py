"""Constants and helpers several test modules used to copy."""

from pathlib import Path

import jax
import numpy as onp

PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])

CALIBRATED_VISIBILITY = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)

_H, _C, _K = 6.62607015e-34, 299792458.0, 1.380649e-23


def planck(wavel, temperature):
    """Unnormalized Planck B_λ, written from scratch."""
    return wavel**-5.0 / onp.expm1(_H * _C / (wavel * _K * temperature))


def float_leaves(grads):
    """The floating-point leaves of a gradient tree."""
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads)
        if hasattr(leaf, "dtype") and onp.issubdtype(leaf.dtype, onp.floating)
    ]
