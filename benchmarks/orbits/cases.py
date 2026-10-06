"""Simulated multi-epoch binaries for the orbit benchmarks.

A ``Case`` is a manifest (seeds and parameters, no data); ``build`` turns it
into a ``System``: simulated ``Epochs``, the true scene, priors, the position
grid and trial periods. The adversarial cases of design/automatic_orbits.md
section 9.3 that are cheap to generate are here; the others raise
``NotImplementedError`` with their number.

Reuses the layout of tests/test_epochs.py: four telescopes, six baselines,
four closure triangles, one wavelength, V and closure phase per night.
"""

import dataclasses
import math

import jax
import numpy as onp
import numpyro.distributions as dist

from virgil.epochs import Epochs
from virgil.models import BinaryModelCartesian, OrbitalBinary
from virgil.oidata import OIData, cp_indices
from virgil.orbits import KeplerOrbit
from virgil.simulate import simulate

T_REF = 60500.0
TRUTH = dict(
    period=700.0,
    dt_peri=-120.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=20.0,
)
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
OBS_PER_FRAME = len(PAIRS) + len(TRIANGLES)  # V and closure phases
NAMES = tuple(TRUTH) + ("flux",)

PRIORS = {
    "period": dist.LogUniform(300.0, 3000.0),
    "dt_peri": dist.Uniform(-1500.0, 1500.0),
    "ecc": dist.Uniform(0.0, 0.9),
    "inc": dist.Uniform(0.0, 180.0),
    "omega": dist.Uniform(-180.0, 360.0),
    "Omega": dist.Uniform(-180.0, 360.0),
    "a_mas": dist.LogUniform(2.0, 100.0),
    "flux": dist.LogUniform(1e-3, 1.0),
}

CASES = {
    "A1": "clean: many epochs, s = 1",
    "A2": "period alias: epochs near multiples of P/2",
    "A5": "sparse epochs: 3, 4 or 5 epochs",
    "A6": "inflated errors: noise s = 3-5 times the quoted errors",
    "A7": "near-equal flux: f = 0.8-1.0",
    "A9": "grid edge: companion beyond the grid edge in some epochs",
    "A10": "non-detection: some epochs have no companion",
}


@dataclasses.dataclass(frozen=True)
class Case:
    """A manifest: everything needed to regenerate a system."""

    case: str = "A1"
    seed: int = 0
    n_epochs: int = 6
    n_obs: int = 100  # observables per dataset (rounded to whole frames)
    grid_size: int = 33  # points per position axis
    n_flux: int = 3
    n_candidates: int = 40
    n_phase: int = 24
    n_periods: int = 20
    truth: tuple = tuple(TRUTH.items())
    flux: float = 0.1
    d_vis: float = 0.01
    d_phi: float = 0.02  # degrees (the data carry phi_unit="deg")

    def manifest(self):
        d = dataclasses.asdict(self)
        d["truth"] = dict(self.truth)
        return d

    @property
    def frames(self):
        return max(1, math.ceil(self.n_obs / OBS_PER_FRAME))

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


@dataclasses.dataclass
class System:
    """A built case: data plus everything the steps need."""

    case: Case
    epochs: Epochs
    truth_values: dict
    scene_fn: object
    priors: dict
    grid: dict
    periods: onp.ndarray
    noise_scale: float
    t_ref: float = T_REF

    @property
    def n_obs_total(self):
        return int(sum(d.vis.size + d.phi.size for d in self.epochs.data))

    def bucket(self):
        """The shape bucket: systems sharing it should share compiles."""
        return (
            len(self.epochs),
            self.case.frames,
            self.case.grid_size,
            self.case.n_flux,
        )


def scene(period, dt_peri, ecc, inc, omega, Omega, a_mas, flux):
    orbit = KeplerOrbit(
        period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref=T_REF
    )
    return OrbitalBinary(orbit, flux)


def start_values(orbit, flux):
    """``start_values`` for ``start_from_positions`` (Python floats)."""
    period = float(orbit.period)
    dt_peri = float(orbit.dt_peri)
    return {
        "period": period,
        "dt_peri": dt_peri - period * onp.round(dt_peri / period),
        "ecc": float(onp.clip(orbit.ecc, 0.01, 0.85)),
        "inc": float(onp.clip(orbit.inc, 1.0, 179.0)),
        "omega": float(onp.mod(orbit.omega, 360.0)),
        "Omega": float(onp.mod(orbit.Omega, 360.0)),
        "a_mas": float(orbit.a_mas),
        "flux": flux,
    }


def night(mjd, frames=1, rotation=0.0, d_vis=0.01, d_phi=0.5):
    """VLTI-like data: ``frames`` four-telescope snapshots.

    Frames are at most 1 h apart and the whole night is at most 6 h and 60
    degrees of rotation, so many frames mean many exposures, not a long arc.
    """
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    step_h = min(1.0, 6.0 / frames)
    angles = onp.deg2rad(rotation + (60.0 / frames) * onp.arange(frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )

    def shift(i):
        return onp.concatenate([i + k * n for k in range(frames)])

    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 2.2e-6,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, d_vis),
            "phi": onp.zeros(len(i1) * frames),
            "d_phi": onp.full(len(i1) * frames, d_phi),
            "phi_unit": "deg",
            "i_cps1": shift(i1),
            "i_cps2": shift(i2),
            "i_cps3": shift(i3),
            "mjd": onp.repeat(mjd + step_h * onp.arange(frames) / 24.0, n),
        }
    )


def _epoch_times(c, truth):
    """Days from ``T_REF`` of each epoch."""
    n, period = c.n_epochs, truth["period"]
    rng = onp.random.default_rng(c.seed)
    if c.case == "A2":  # near multiples of P/2, with a small jitter
        k = onp.arange(n)
        return k * period / 2.0 + rng.normal(0.0, 0.02 * period, n) - 300.0
    if c.case == "A5":
        n = min(n, 5)
    return (
        onp.sort(rng.uniform(-300.0, 400.0, n))
        if n > 6
        else onp.linspace(-300.0, 400.0, n)
    )


def build(c):
    """Build the system of manifest ``c`` (float64 inside)."""
    if c.case not in CASES:
        raise NotImplementedError(
            f"Case {c.case} is not generated yet; known: {sorted(CASES)}"
        )
    truth = dict(c.truth)
    flux = c.flux
    noise_scale = 1.0
    if c.case == "A6":
        noise_scale = 3.0 + 2.0 * onp.random.default_rng(c.seed).random()
    if c.case == "A7":
        flux = 0.8 + 0.2 * onp.random.default_rng(c.seed).random()
    times = _epoch_times(c, truth)
    n = len(times)
    nights = [
        night(
            T_REF + t,
            rotation=30.0 * k,
            frames=c.frames,
            d_vis=c.d_vis,
            d_phi=c.d_phi,
        )
        for k, t in enumerate(times)
    ]
    keys = jax.random.split(jax.random.PRNGKey(c.seed), n)
    with jax.enable_x64(True):
        names = [f"e{k}" for k in range(n)]
        snaps = Epochs(dict(zip(names, nights))).snapshots(
            scene(**truth, flux=flux)
        )
        if c.case == "A10":  # the odd epochs have no detectable companion
            snaps = [
                BinaryModelCartesian(s.dra, s.ddec, 1e-6) if k % 2 else s
                for k, s in enumerate(snaps)
            ]
        observed = [
            simulate(s, t, k, noise_scale=noise_scale)
            for s, t, k in zip(snaps, nights, keys)
        ]
        epochs = Epochs(dict(zip(names, observed)))
    half = 15.0 if c.case == "A9" else 30.0  # A9: the grid cuts the orbit
    axis = onp.linspace(-half, half, c.grid_size)
    grid = {
        "dra": axis,
        "ddec": axis,
        "flux": onp.geomspace(0.05, 1.0, c.n_flux)
        if c.case == "A7"
        else onp.geomspace(0.05, 0.2, c.n_flux),
    }
    return System(
        case=c,
        epochs=epochs,
        truth_values={**truth, "flux": flux},
        scene_fn=scene,
        priors=PRIORS,
        grid=grid,
        periods=onp.geomspace(400.0, 2000.0, c.n_periods),
        noise_scale=float(noise_scale),
    )
