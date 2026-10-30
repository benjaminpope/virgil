"""Small helpers shared across virgil: unit constants, traced-value
checks, and the rules for naming flux parameters."""

import dataclasses
import math

import jax
import jax.numpy as np
import numpy as onp


# === CONSTANTS ===

rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad
dtor = np.pi / 180.0  # convert deg to rad
# FWHM of a Gaussian over its standard deviation, 2 sqrt(2 ln 2)
FWHM_PER_SIGMA = float(2.0 * onp.sqrt(2.0 * onp.log(2.0)))


def _clean_float(value):
    return value if math.isfinite(value) else None


def clean_json(value):
    """``value`` with NaN and infinities replaced by ``None`` (strict JSON)."""
    if isinstance(value, float):
        return _clean_float(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (onp.ndarray, onp.generic)) or hasattr(
        value, "tolist"
    ):
        return clean_json(onp.asarray(value).tolist())
    return value


def _unit_sum(image):
    """``image`` scaled to sum to one."""
    return image / np.sum(image)


def check_part_name(name, owner, kind, example, clashes_with):
    """Reject a part name that is not an identifier, starts with ``_`` or clashes with ``owner``'s attributes."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"{kind[0].upper()}{kind[1:]} name {name!r} must be a valid "
            "Python identifier, so "
            f"that it can be used in parameter paths such as {example!r}."
        )
    fields = {f.name for f in dataclasses.fields(owner)}
    if name.startswith("_") or name in fields or hasattr(owner, name):
        raise ValueError(
            f"'{name}' cannot be a {kind} name because it clashes "
            f"with {clashes_with}; choose another name."
        )


def wrap_phase(phase):
    """``phase`` (radians) wrapped to ``[-π, π)``."""
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


# === TRACED VALUES ===


def group_indices(labels, indices=None):
    """Ascending ``indices`` (default all) of each distinct label, by label.

    One stable sort instead of a pass over ``labels`` per label.
    """
    labels = onp.asarray(labels).reshape(-1)
    indices = onp.arange(labels.size) if indices is None else indices
    order = onp.argsort(labels, kind="stable")
    cuts = onp.flatnonzero(onp.diff(labels[order])) + 1
    return onp.split(indices[order], cuts) if labels.size else []


def concrete(value):
    """``value`` as a NumPy array, or ``None`` when traced (so checks skip under ``jit``)."""
    try:
        return onp.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


# === ERROR INFLATION ===


def inflate_errors(
    sigma,
    reference=None,
    absolute=None,
    relative=None,
    scale=None,
    combine="quadrature",
):
    """Uncertainties scaled, with absolute and relative terms combined.

    The one rule behind both ``likelihood.inflated_errors`` (quadrature,
    relative to the model) and ``OIData.with_error_floor`` (a maximum,
    relative to the data).

    Parameters
    ----------
    sigma : array-like
        The uncertainties.
    reference : array-like, optional
        What ``relative`` is a fraction of (the model or the data).
    absolute, relative, scale : float, optional
        An absolute term, a term ``relative * |reference|``, and a factor
        applied to ``sigma`` first. ``None`` leaves a term out.
    combine : {"quadrature", "max"}, optional
        Add the terms in quadrature (default), or take the largest of
        ``sigma`` and the terms (a floor).
    """
    sigma = np.asarray(sigma)
    if scale is not None:
        sigma = scale * sigma
    terms = []
    if absolute is not None:
        terms.append(np.broadcast_to(np.asarray(absolute), sigma.shape))
    if relative is not None:
        terms.append(relative * np.abs(np.asarray(reference)))
    if combine == "quadrature":
        for term in terms:
            sigma = np.hypot(sigma, term)
        return sigma
    if combine == "max":
        for term in terms:
            sigma = np.maximum(sigma, term)
        return sigma
    raise ValueError(
        f"combine must be 'quadrature' or 'max', not {combine!r}."
    )


# === PER-DATASET MODELS ===


def _reference(model):
    """The model regularizers act on; raises for a model per dataset.

    A regularizer has no dataset to belong to, so a list of models would
    silently regularize only the first.
    """
    if isinstance(model, (list, tuple)):
        raise ValueError(
            "Regularizers cannot be used with a model function that returns "
            "one model per dataset: they would act on the first only. Put the "
            "penalty in the priors, or fit with a single model."
        )
    return model


def _per_dataset(model, n_data):
    """One model per dataset: ``model`` repeated, or its list checked."""
    if not isinstance(model, (list, tuple)):
        return [model] * n_data
    if len(model) != n_data:
        raise ValueError(
            f"The model function returned {len(model)} models for "
            f"{n_data} datasets."
        )
    return list(model)


# === FLUX PARAMETERS ===
#
# A parameter is a flux when the last part of its name (or zodiax path) is
# ``flux`` (``flux``, ``comp.flux``, ``comp.disk.flux``), or when it is a flux
# spectrum's reference ratio (``comp.flux.ratio``). Fluxes are relative
# to a reference component (usually the primary star at flux 1), so for a
# companion ``flux`` is its companion/primary flux ratio.


def is_flux_param(name):
    """Whether ``name`` ends in ``flux`` or is a flux spectrum's ``flux.ratio``."""
    parts = str(name).split(".")
    return parts[-1] == "flux" or parts[-2:] == ["flux", "ratio"]


def resolve_flux_param(keys, flux_param=None):
    """The key among ``keys`` that is the flux: ``flux_param``, else the one flux key (raises if none or several)."""
    keys = list(keys)
    if flux_param is not None:
        if flux_param not in keys:
            raise ValueError(
                f"flux_param {flux_param!r} is not one of the keys {keys}."
            )
        return flux_param
    candidates = [key for key in keys if is_flux_param(key)]
    if len(candidates) != 1:
        found = f"several ({candidates})" if candidates else "none"
        raise ValueError(
            f"Could not tell which of {keys} is the flux: {found} end in "
            "'flux'. Pass flux_param=<key>."
        )
    return candidates[0]


def _named_part(self, name, kind):
    """``__getattr__`` of a model with named ``parts``: the part ``name``."""
    try:
        names = object.__getattribute__(self, "names")
        parts = object.__getattribute__(self, "parts")
    except AttributeError:
        raise AttributeError(name) from None
    if name in names:
        return parts[names.index(name)]
    raise AttributeError(
        f"{type(self).__name__} has no {kind} or attribute '{name}'; its "
        f"{kind}s are {list(names)}."
    )
