"""Simulated observations of a scene, and bias tests across instruments.

[`simulate`][virgil.simulate.simulate] observes a scene with the sampling,
errors and times of a template: real data from one instrument, or the
synthetic coverage of [`virgil.coverage`][virgil.coverage]. It answers
"what would instrument B measure for the scene instrument A sees?", and,
with ``shift_days``, "what would the same coverage give at other epochs?".
[`bias_test`][virgil.simulate.bias_test] fits a model to many noise draws
and reports the spread of the fitted parameters, which shows biases from a
model that is simpler than the scene (e.g. two point sources fitted to a
star with extended emission) and how well the coverage constrains a
parameter.
"""

import copy

import jax
import numpy as onp

from .fitting import fit


__all__ = ["bias_test", "simulate"]


def simulate(scene, template, key=None, noise_scale=1.0, shift_days=None):
    """Observe ``scene`` with the sampling, errors and times of ``template``.

    Parameters
    ----------
    scene : SourceModel
        The scene. A time-dependent scene (one with
        [`Attached`][virgil.models.Attached] components) is evaluated at
        each sample's own time, so the template needs times.
    template : OIData
        The observation to copy: its uv samples, wavelengths, observables,
        closure-phase correlations and projections, and errors.
    key : jax.random.PRNGKey, optional
        For Gaussian noise drawn with the template's errors (closure phases
        correlated as in the template); noiseless without it.
    noise_scale : float, optional
        Multiplies the noise (not the stored errors).
    shift_days : float, optional
        Move every sample's time by this many days, e.g. to see what the
        same coverage would give at a later epoch of an orbit.

    Returns
    -------
    OIData
        The template with its observables replaced by the scene's.
    """
    data = template
    if shift_days is not None:
        if data.dt is None:
            raise ValueError("shift_days needs a template with times.")
        data = _shift_times(data, shift_days)
    return data.with_model(scene, key=key, noise_scale=noise_scale)


def _shift_times(data, days):
    """``data`` with every time moved by ``days``.

    The float64 ``t_ref`` moves and the small float32 offsets ``dt`` are
    kept, so even a shift of decades keeps the samples' times to well under
    a second. ``t_ref`` is a static field, so it is set on a copy.
    """
    shifted = copy.copy(data)
    object.__setattr__(shifted, "t_ref", data.t_ref + float(days))
    return shifted


def bias_test(scene, template, model, priors, n, key, **fit_kwargs):
    """Fit ``model`` to ``n`` noisy simulations of ``scene``.

    Parameters
    ----------
    scene : SourceModel
        The truth, simulated with [`simulate`][virgil.simulate.simulate].
    template : OIData
        The observation to simulate.
    model, priors
        The model fitted to each simulation and its priors, as in
        [`fit`][virgil.fitting.fit]. The model may differ from the scene
        (that is the point of a bias test).
    n : int
        Number of noise draws.
    key : jax.random.PRNGKey
        Seed for the draws.
    **fit_kwargs
        Passed to [`fit`][virgil.fitting.fit] (e.g. ``init``, ``method``).

    Returns
    -------
    dict
        For each fitted path, the ``n`` fitted values as an array; and
        ``chi2_red``, the reduced χ² of each fit. Compare their means and
        spreads with the truth.
    """
    results = {}
    for draw in jax.random.split(key, n):
        result = fit(
            model, priors, simulate(scene, template, key=draw), **fit_kwargs
        )
        for path, value in result.values.items():
            results.setdefault(path, []).append(onp.asarray(value))
        results.setdefault("chi2_red", []).append(result.info["chi2_red"])
    return {path: onp.array(values) for path, values in results.items()}
