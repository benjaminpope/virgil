"""Pretuning of the SMC mutation with one parameter population per label group.

Buchholz, Chopin & Jacob (2021) keep a population of HMC parameters, one per
particle, noise it each stage and resample it by the expected squared jumping
distance (ESJD) of one trial step. Here a particle's parameters are drawn only
from the trials of the particles in its own group, so each group's step size
and trajectory length follow that group's geometry rather than the average of
a cloud whose groups (a broad prior share, narrow modes) differ.

Source: Buchholz, Chopin & Jacob 2021 (see ``references/provenance.toml``).
"""

import jax
import jax.numpy as np
from blackjax.smc.base import update_and_take_last
from blackjax.smc.from_mcmc import unshared_parameters_and_step_fn


def group_pretune(mcmc_init, mcmc_step, group, alpha, sigma, n, measure_of):
    """A ``pretune(key, state, logposterior)`` for BlackJAX's ``pretuning``.

    ``group`` ``(K,)`` maps a label ``particle["c"]`` to its group; ``sigma``
    is the noise scale of each tuned parameter (``step_size``,
    ``num_integration_steps``); ``measure_of(state)`` is the ESJD factory."""

    def pretune(key, state, logposterior):
        unshared, shared_step = unshared_parameters_and_step_fn(
            state.parameter_override, mcmc_step
        )
        one_step, _ = update_and_take_last(
            mcmc_init, logposterior, shared_step, 1, n
        )
        particles = state.sampler_state.particles
        moved, info = one_step(jax.random.split(key, n), particles, unshared)
        gain = np.ravel(
            measure_of(state)(particles, moved, info.acceptance_rate)
        )
        kn, kr = jax.random.split(key)
        g = group[particles["c"]]
        # Row i draws from the trials of the particles in i's group.
        logits = np.where(
            g[:, None] == g[None, :], np.log(alpha + gain)[None, :], -np.inf
        )
        pick = jax.random.categorical(kr, logits, axis=1)
        out = dict(state.parameter_override)
        for name, s in sigma.items():
            trial = out[name] + s * jax.random.normal(kn, (n,))
            out[name] = trial[pick]
        out["step_size"] = np.abs(out["step_size"])
        out["num_integration_steps"] = np.maximum(
            np.abs(np.round(out["num_integration_steps"])).astype(int), 1
        )
        return out

    return pretune
