"""NUTS on one packed unconstrained vector, for small likelihood-term models.

[`numpyro_model`][virgil.likelihood.numpyro_model] samples every prior at its own
site, so each leapfrog step carries a pytree of small leaves through NumPyro's
loop. For a model of a few parameters (an orbit fit to astrometry) that
bookkeeping, not the physics, is most of the step time.
[`flat_nuts`][virgil.flat_nuts.flat_nuts] packs the unconstrained coordinates of all
priors into one vector and runs blackjax's window adaptation and NUTS on it, with the
same posterior: the log density is the priors' log densities plus the log-determinants of
the transforms plus the likelihood terms, as in `numpyro_model`.

Source: Hoffman and Gelman (2014), NUTS; adaptation as in Stan, via blackjax.
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as np
from numpyro.distributions.transforms import biject_to

from .angles import is_angle_vector, vector_angle, vector_site

__all__ = ["flat_nuts", "flat_nuts_sampler", "packed_posterior"]


def packed_posterior(priors, likelihoods):
    """The packed log density and its coordinate maps.

    Parameters
    ----------
    priors : dict
        Path to prior, scalar priors and [`AngleVector`][virgil.angles.AngleVector]
        (a 2-vector, sampled at ``<path>_vec``), as for
        [`numpyro_model`][virgil.likelihood.numpyro_model].
    likelihoods : sequence
        Terms with a ``loglike(values)`` method, e.g. ``PositionData.term(orbit_fn)``.

    Returns
    -------
    logdensity : callable
        Packed unconstrained vector to the log posterior density.
    unpack : callable
        Packed vector to the dict of sites a ``numpyro_model`` run reports: each path's
        constrained value (angles in degrees) and each angle vector at ``<path>_vec``.
    pack : callable
        Dict of those sites (angle paths may be given as ``<path>_vec``) to the vector.
    size : int
        Length of the packed vector.
    """
    paths = list(priors)
    sizes = [2 if is_angle_vector(priors[p]) else 1 for p in paths]
    maps = {
        p: None if is_angle_vector(priors[p]) else biject_to(priors[p].support)
        for p in paths
    }
    cuts = [sum(sizes[:i]) for i in range(len(paths))]

    def _split(u):
        for p, i, n in zip(paths, cuts, sizes):
            yield p, (u[i : i + n] if n > 1 else u[i])

    def unpack(u):
        out = {}
        for p, ui in _split(u):
            if maps[p] is None:
                out[vector_site(p)] = ui
                out[p] = vector_angle(ui)
            else:
                out[p] = maps[p](ui)
        return out

    def logdensity(u):
        total = 0.0
        values = {}
        for p, ui in _split(u):
            prior = priors[p]
            if maps[p] is None:
                total = total + prior.log_prob(ui)
                values[p] = vector_angle(ui)
            else:
                x = maps[p](ui)
                total = (
                    total
                    + prior.log_prob(x)
                    + maps[p].log_abs_det_jacobian(ui, x)
                )
                values[p] = x
        for term in likelihoods:
            total = total + term.loglike(values)
        return total

    def pack(values):
        parts = []
        for p in paths:
            if maps[p] is None:
                parts.append(np.asarray(values[vector_site(p)], float))
            else:
                parts.append(
                    np.atleast_1d(maps[p].inv(np.asarray(values[p], float)))
                )
        return np.concatenate(parts)

    return logdensity, unpack, pack, sum(sizes)


def flat_nuts(priors, likelihoods, *, seed=0, init=None, **options):
    """Run [`flat_nuts_sampler`][virgil.flat_nuts.flat_nuts_sampler] once.

    Takes its options and returns ``(samples, info)``; see there. To time or repeat
    fits without recompiling, keep the sampler and call it with new seeds.
    """
    return flat_nuts_sampler(priors, likelihoods, **options)(seed, init)


def flat_nuts_sampler(
    priors,
    likelihoods,
    *,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    target_accept=0.8,
    max_tree_depth=10,
    dense_mass=True,
):
    """Build a blackjax NUTS sampler, with window adaptation, on the packed posterior.

    The chains run in one vectorized program, compiled on the first call and reused
    after. Same posterior as ``MCMC(NUTS(numpyro_model(...)))``; the sites are reported
    under the same names.

    Parameters
    ----------
    priors, likelihoods
        As for [`packed_posterior`][virgil.flat_nuts.packed_posterior].
    num_warmup, num_samples, num_chains : int, optional
        As for NUTS.
    target_accept : float, optional
        The adaptation's target acceptance rate.
    max_tree_depth : int, optional
        Trajectory doublings.
    dense_mass : bool, optional
        Dense (default) or diagonal mass matrix.

    Returns
    -------
    sample : callable
        ``sample(seed=0, init=None)`` returns ``(samples, info)``. ``init`` is one dict
        of starting site values, or one per chain, keyed as the sampled sites (``<path>``
        for scalars, ``<path>_vec`` for angle vectors); the default draws from the
        priors. ``samples`` maps site name to an array of shape
        ``(num_chains, num_samples, ...)``; ``info`` has ``num_steps`` and ``diverging``
        of that shape, and the adapted ``step_size`` per chain.
    """
    logdensity, unpack, pack, _ = packed_posterior(priors, likelihoods)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=not dense_mass,
        target_acceptance_rate=target_accept,
        max_num_doublings=max_tree_depth,
    )

    def one_chain(k, u):
        k_warm, k_run = jax.random.split(k)
        (state, params), _ = warmup.run(k_warm, u, num_steps=num_warmup)
        kernel = blackjax.nuts(
            logdensity, **params
        ).step  # params carry max_num_doublings

        def body(state, k):
            state, info = kernel(k, state)
            return state, (
                state.position,
                info.num_integration_steps,
                info.is_divergent,
            )

        _, (draws, steps, div) = jax.lax.scan(
            body, state, jax.random.split(k_run, num_samples)
        )
        return draws, steps, div, params["step_size"]

    run = jax.jit(jax.vmap(one_chain))
    unpack_all = jax.jit(jax.vmap(jax.vmap(unpack)))

    def sample(seed=0, init=None):
        key, k_init = jax.random.split(jax.random.PRNGKey(seed))
        if init is None:
            keys = jax.random.split(k_init, num_chains * len(priors)).reshape(
                num_chains, len(priors), -1
            )
            starts = [
                {
                    (vector_site(p) if is_angle_vector(pr) else p): pr.sample(
                        keys[c, i]
                    )
                    for i, (p, pr) in enumerate(priors.items())
                }
                for c in range(num_chains)
            ]
        elif isinstance(init, dict):
            starts = [init] * num_chains
        else:
            starts = list(init)
        u0 = np.stack([pack(s) for s in starts])
        draws, steps, div, step_size = run(
            jax.random.split(key, num_chains), u0
        )
        info = {"num_steps": steps, "diverging": div, "step_size": step_size}
        return unpack_all(draws), info

    return sample
