"""Simulated data and expected shapes shared by the OrbitPipeline test files."""

import jax
import numpy as onp

import virgil.pipeline.orbit as orbit_module
from tests._shared import vlti_night
from virgil.models import OrbitalBinary, PointSource
from virgil.orbits import KeplerOrbit
from virgil.pipeline import OrbitPipeline
from virgil.simulate import simulate

T_REF = 60000.0
TRUTH = dict(
    period=12.38,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
# Two seasons: three nights, a gap of 238 d, three more.
SEASONS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])


def _night(mjd, rotation):
    return vlti_night(mjd, 3, rotation, d_phi=0.02, stations=True)


def _simulate(times, truth=None, flux=FLUX, seed=3):
    """One simulated night per time, for the orbit ``truth``."""
    if flux is None:
        scene = PointSource()
    else:
        orbit = KeplerOrbit(**(truth or TRUTH), t_ref=T_REF)
        scene = OrbitalBinary(orbit, flux)
    key = jax.random.PRNGKey(seed)
    nights = {}
    with jax.enable_x64(True):
        for k, mjd in enumerate(times):
            night = _night(mjd, 7.0 * k)
            nights[f"n{k}"] = simulate(
                scene, night, key=jax.random.fold_in(key, k)
            )
    return nights


SUMMARY_KEYS = {
    "schema": None,
    "pipeline": None,
    "stability": None,
    "status": None,
    "worst_check": None,
    "data": set(orbit_module._DATA_KEYS),
    "nights": None,
    "chi2": {"n_independent", "best_orbit_reduced", "nights"},
    "search": set(orbit_module._SEARCH_KEYS),
    "modes": set(orbit_module._MODES_KEYS),
    "orbit": {"n", "verdict", "source", "elements"},
    "posterior": set(orbit_module._POSTERIOR_KEYS),
    "warnings": None,
    "checks": None,
}
CHECKS = [
    "chi2",
    "night_flags",
    "labels_free",
    "bands_failed",
    "alias_ambiguity",
    "winner_flags",
    "tempering",
    "primary_resolved",
    "no_orbit",
    "period_bound",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
    "mode_converged",
]


# 2-5 minutes on a CI runner, so each golden carries a 15-minute timeout
# in place of the 5-minute hang detector of pyproject.toml.
# The search keeps the engine's default effort (that is the point: the default
# run must work); only the sampler is small. Do not add search settings here
# to speed up CI. A blind run takes 2-5 minutes, so each golden test carries a
# 15-minute timeout in place of the 5-minute hang detector of pyproject.toml.
GOLDEN = {
    "num_warmup": 200,
    "num_samples": 150,
    "num_chains": 2,
    "quicklook_execute": False,
}


def _blind(nights, out, **settings):
    pipeline = OrbitPipeline(nights, output=out, **{**GOLDEN, **settings})
    return pipeline, pipeline.run()
