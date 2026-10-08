"""AMIGO mixed-DISCO data products.

An AMIGO reduction stores, per filter, a record of DISCO coefficients (linear
combinations of log-amplitudes and phases whose errors are independent) with
the operators that map model visibilities to them. :func:`load_oi_data` reads
such a product into [`OIData`][virgil.oidata.OIData] objects;
``OIData(record)`` also accepts a single record directly.
"""

from pathlib import Path

import jax.numpy as np
import numpy as onp

from ._geometry import find_uv_grid


MIXED_DISCO_REQUIRED_FIELDS = (
    "u",
    "v",
    "wavelength_m",
    "disco_coefficients",
    "disco_sigma",
    "disco_logamp_model_operator",
    "disco_phase_model_operator",
)


def is_mixed_disco_record(data):
    """Whether a dictionary is an AMIGO mixed-DISCO record."""
    return "disco_coefficients" in data


def validate_mixed_disco_record(record):
    """Raise if an AMIGO mixed-DISCO record is incomplete or inconsistent."""
    missing = [
        name for name in MIXED_DISCO_REQUIRED_FIELDS if name not in record
    ]
    if missing:
        raise KeyError(
            f"AMIGO mixed-DISCO record is missing fields: {missing}"
        )

    u = onp.asarray(record["u"], dtype=float)
    v = onp.asarray(record["v"], dtype=float)
    coefficients = onp.asarray(record["disco_coefficients"], dtype=float)
    sigma = onp.asarray(record["disco_sigma"], dtype=float)
    logamp = onp.asarray(record["disco_logamp_model_operator"], dtype=float)
    phase = onp.asarray(record["disco_phase_model_operator"], dtype=float)

    if u.ndim != 1 or v.shape != u.shape:
        raise ValueError("AMIGO DISCO u and v must be matching vectors.")
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError(
            "AMIGO DISCO coefficients must be a non-empty vector."
        )
    if sigma.shape != coefficients.shape:
        raise ValueError(
            "AMIGO DISCO coefficients and sigma must have matching shapes."
        )
    expected_operator_shape = (sigma.size, u.size)
    if (
        logamp.shape != expected_operator_shape
        or phase.shape != expected_operator_shape
    ):
        raise ValueError(
            "AMIGO DISCO model operators have inconsistent shapes."
        )
    if not all(
        onp.all(onp.isfinite(value))
        for value in (u, v, coefficients, sigma, logamp, phase)
    ):
        raise ValueError("AMIGO DISCO arrays must be finite.")
    if onp.any(sigma <= 0.0):
        raise ValueError("AMIGO DISCO sigma values must be positive.")

    tolerance = 1e-12 * max(float(onp.max(sigma)), 1.0)
    if onp.any(onp.diff(sigma) < -tolerance):
        raise ValueError(
            "AMIGO DISCO modes must be ordered by increasing uncertainty."
        )

    if "disco_covariance" in record:
        _check_diagonal_covariance(record["disco_covariance"], sigma)


def _check_diagonal_covariance(covariance, sigma):
    """DISCO errors are independent by construction; check a stored matrix.

    Products need not store the covariance: ``disco_sigma`` holds all of
    it. Older products that do are checked for consistency.
    """
    covariance = onp.asarray(covariance, dtype=float)
    if covariance.shape != (sigma.size, sigma.size):
        raise ValueError("AMIGO DISCO covariance has inconsistent shape.")
    if not onp.all(onp.isfinite(covariance)):
        raise ValueError("AMIGO DISCO arrays must be finite.")
    covariance_scale = max(
        float(onp.max(onp.abs(onp.diag(covariance)))),
        onp.finfo(covariance.dtype).tiny,
    )
    off_diagonal = covariance - onp.diag(onp.diag(covariance))
    relative_off_diagonal = float(
        onp.max(onp.abs(off_diagonal)) / covariance_scale
    )
    if relative_off_diagonal > 1e-10:
        raise ValueError(
            "AMIGO DISCO covariance is not diagonal: relative "
            f"off-diagonal {relative_off_diagonal:.3g}."
        )
    if not onp.allclose(onp.diag(covariance), sigma**2, rtol=1e-8, atol=0.0):
        raise ValueError(
            "AMIGO DISCO covariance diagonal does not match sigma squared."
        )


def mixed_disco_fields(record):
    """Validate a mixed-DISCO record and return the OIData field values."""
    validate_mixed_disco_record(record)
    u = -onp.asarray(record["u"], dtype=float)
    v = -onp.asarray(record["v"], dtype=float)
    return {
        "u": np.asarray(u),
        "v": np.asarray(v),
        "wavel": np.asarray(record["wavelength_m"], dtype=float),
        "vis": np.asarray(record["disco_coefficients"], dtype=float),
        "d_vis": np.asarray(record["disco_sigma"], dtype=float),
        "phi": np.empty((0,), dtype=float),
        "d_phi": np.empty((0,), dtype=float),
        "i_cps1": None,
        "i_cps2": None,
        "i_cps3": None,
        "vis_mat": np.asarray(
            record["disco_logamp_model_operator"], dtype=float
        ),
        "phi_mat": np.asarray(
            record["disco_phase_model_operator"], dtype=float
        ),
        "vis_index": None,
        "phi_index": None,
        "uv_grid": find_uv_grid(u, v),
        "cp_noise": None,
        "observable_kind": "mixed_log_complex",
        "vis_mode": "logamp",
        "v2_flag": False,
        "cp_flag": False,
    }


def load_oi_data(path, filter_name=None):
    """Load one or all filters from an AMIGO mixed-DISCO NumPy product.

    The file is a pickled dictionary loaded with ``allow_pickle=True``, which
    can run arbitrary code: only load files you trust.

    Parameters
    ----------
    path : str or os.PathLike
        ``.npy`` file holding a ``{filter_name: record}`` dictionary.
    filter_name : str, optional
        Filter to load. By default every filter is loaded.

    Returns
    -------
    OIData or dict[str, OIData]
        The chosen filter, or a dictionary of all filters.
    """
    from .oidata import OIData

    records = onp.load(Path(path), allow_pickle=True).item()
    if not isinstance(records, dict):
        raise TypeError(
            "Expected a filter-keyed dictionary in the NumPy file."
        )
    if filter_name is not None:
        return OIData(records[filter_name])
    return {name: OIData(record) for name, record in records.items()}
