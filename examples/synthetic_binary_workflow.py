from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer import MCMC, NUTS

from virgil.inference import fisher_projection, hessian_matrix
from virgil.grid_fit import likelihood_grid
from virgil.likelihood import loglike
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData, closure_phases, cp_indices
from virgil.oifits import write_oifits


@dataclass(frozen=True)
class RecoverySummary:
    truth: dict[str, float]
    grid_estimate: dict[str, float]
    hmc_median: dict[str, float]
    hmc_std: dict[str, float]
    fisher_hmc_median: dict[str, float]
    fisher_hmc_std: dict[str, float]
    noise_settings: dict[str, float]
    output_file: str


def _array_geometry() -> tuple[
    jnp.ndarray, jnp.ndarray, np.ndarray, np.ndarray
]:
    station_xy = np.array(
        [
            [0.0, 0.0],
            [3.2, 0.2],
            [1.4, 2.6],
            [-1.1, 1.8],
        ]
    )

    baseline_pairs = np.array(
        [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
        ],
        dtype=int,
    )

    triangles = np.array(
        [
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ],
        dtype=int,
    )

    ucoord = []
    vcoord = []
    for a, b in baseline_pairs:
        pa = station_xy[a - 1]
        pb = station_xy[b - 1]
        delta = pb - pa
        ucoord.append(delta[0])
        vcoord.append(delta[1])

    return jnp.array(ucoord), jnp.array(vcoord), baseline_pairs, triangles


def _triangle_uv(
    station_xy: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u1, v1, u2, v2 = [], [], [], []
    for a, b, c in triangles:
        pa = station_xy[a - 1]
        pb = station_xy[b - 1]
        pc = station_xy[c - 1]
        d1 = pb - pa
        d2 = pc - pb
        u1.append(d1[0])
        v1.append(d1[1])
        u2.append(d2[0])
        v2.append(d2[1])
    return np.array(u1), np.array(v1), np.array(u2), np.array(v2)


def _build_synthetic_oifits_dict(
    seed: int = 4,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Build a synthetic OIFITS-like dictionary for docs and tests.

    Notes
    -----
    ``virgil.oidata`` computes phases internally in radians. OIFITS phase
    channels in this helper are written in degrees (``VISPHI``, ``T3PHI``) so
    they match the standard OIFITS convention expected by downstream readers.
    """
    rng = np.random.default_rng(seed)
    wavel = 4.8e-6

    ucoord, vcoord, baseline_pairs, triangles = _array_geometry()
    station_xy = np.array(
        [
            [0.0, 0.0],
            [3.2, 0.2],
            [1.4, 2.6],
            [-1.1, 1.8],
        ]
    )
    u1, v1, u2, v2 = _triangle_uv(station_xy, triangles)

    truth = {"dra": 110.0, "ddec": -70.0, "flux": 3.2e-3}
    model = BinaryModelCartesian(**truth)
    cvis = model.model(ucoord, vcoord, jnp.array([wavel]))

    visamp = jnp.abs(cvis)
    visphi = jnp.rad2deg(jnp.angle(cvis))
    vis2 = visamp**2

    i1, i2, i3 = cp_indices(baseline_pairs, triangles)
    cp = jnp.rad2deg(closure_phases(cvis, i1, i2, i3))

    visamp_scale = jnp.maximum(jnp.median(visamp), 1e-6)
    visphi_scale = jnp.maximum(jnp.median(jnp.abs(visphi)), 5.0)
    vis2_scale = jnp.maximum(jnp.median(vis2), 1e-6)
    cp_scale = jnp.maximum(jnp.median(jnp.abs(cp)), 5.0)

    noise_settings = {
        "visamp_err_frac": 0.002,
        "visphi_err_frac": 0.004,
        "vis2_err_frac": 0.001,
        "cp_err_frac": 0.004,
    }

    visamp_err = (
        noise_settings["visamp_err_frac"]
        * visamp_scale
        * jnp.ones_like(visamp)
    )
    visphi_err = (
        noise_settings["visphi_err_frac"]
        * visphi_scale
        * jnp.ones_like(visphi)
    )
    vis2_err = (
        noise_settings["vis2_err_frac"] * vis2_scale * jnp.ones_like(vis2)
    )
    cp_err = noise_settings["cp_err_frac"] * cp_scale * jnp.ones_like(cp)

    visamp_obs = np.array(
        visamp + visamp_err * jnp.array(rng.normal(size=visamp.shape))
    )
    visphi_obs = np.array(
        visphi + visphi_err * jnp.array(rng.normal(size=visphi.shape))
    )
    vis2_obs = np.array(
        vis2 + vis2_err * jnp.array(rng.normal(size=vis2.shape))
    )
    cp_obs = np.array(cp + cp_err * jnp.array(rng.normal(size=cp.shape)))

    n_bl = len(baseline_pairs)
    n_cp = len(triangles)

    dic = {
        "info": {
            "TARGET": "UNKNOWN",
            "OBJECT": "UNKNOWN",
            "INSTRUME": "SYNTH",
            "MASK": "SYNTH_MASK",
            "ARRNAME": "SYNTH_MASK",
            "FILT": "F480M",
            "DATE-OBS": "2000-01-01",
            "TELESCOP": "SIM",
            "OBSERVER": "virgil-docs",
            "INSMODE": "NRM",
            "PA": 0.0,
            "MJD": 61000.0,
            "PSCALE": 65.0,
            "ISZ": 81,
            "STAXY": station_xy,
            "CTRS_EQT": station_xy.copy(),
        },
        "OI_WAVELENGTH": {
            "EFF_WAVE": wavel,
            "EFF_BAND": 0.3e-6,
        },
        "OI_VIS": {
            "TARGET_ID": 1,
            "TIME": 0.0,
            "MJD": 61000.0,
            "INT_TIME": 1.0,
            "VISAMP": visamp_obs,
            "VISAMPERR": np.array(visamp_err),
            "VISPHI": visphi_obs,
            "VISPHIERR": np.array(visphi_err),
            "UCOORD": np.array(ucoord),
            "VCOORD": np.array(vcoord),
            "STA_INDEX": baseline_pairs,
            "FLAG": np.zeros(n_bl, dtype=bool),
        },
        "OI_VIS2": {
            "TARGET_ID": 1,
            "TIME": 0.0,
            "MJD": 61000.0,
            "INT_TIME": 1.0,
            "VIS2DATA": vis2_obs,
            "VIS2ERR": np.array(vis2_err),
            "UCOORD": np.array(ucoord),
            "VCOORD": np.array(vcoord),
            "STA_INDEX": baseline_pairs,
            "FLAG": np.zeros(n_bl, dtype=bool),
        },
        "OI_T3": {
            "TARGET_ID": 1,
            "TIME": 0.0,
            "MJD": 61000.0,
            "INT_TIME": 1.0,
            "T3AMP": np.ones(n_cp),
            "T3AMPERR": np.ones(n_cp),
            "T3PHI": cp_obs,
            "T3PHIERR": np.array(cp_err),
            "U1COORD": u1,
            "V1COORD": v1,
            "U2COORD": u2,
            "V2COORD": v2,
            "STA_INDEX": triangles,
            "FLAG": np.zeros(n_cp, dtype=bool),
        },
    }
    return dic, truth, noise_settings


def _recover_grid(oidata: OIData) -> dict[str, float]:
    samples = {
        "dra": jnp.linspace(-220.0, 220.0, 41),
        "ddec": jnp.linspace(-220.0, 220.0, 41),
        "flux": 10 ** jnp.linspace(-4.5, -1.5, 36),
    }
    ll = likelihood_grid(BinaryModelCartesian, oidata, samples)
    best = jnp.unravel_index(jnp.argmax(ll), ll.shape)
    return {
        "dra": float(samples["dra"][best[0]]),
        "ddec": float(samples["ddec"][best[1]]),
        "flux": float(samples["flux"][best[2]]),
    }


def _recover_hmc(
    oidata: OIData,
    seed: int = 2026,
    init: dict[str, float] | None = None,
    num_warmup: int = 800,
    num_samples: int = 2000,
) -> tuple[dict[str, float], dict[str, float]]:
    params = ["dra", "ddec", "flux"]

    def model_hmc(data_obj: OIData):
        dra = numpyro.sample("dra", dist.Uniform(-250.0, 250.0))
        ddec = numpyro.sample("ddec", dist.Uniform(-250.0, 250.0))
        log10_flux = numpyro.sample("log10_flux", dist.Uniform(-6.0, -1.0))
        flux = 10.0**log10_flux
        numpyro.factor(
            "loglike",
            loglike([dra, ddec, flux], params, data_obj, BinaryModelCartesian),
        )

    if init is None:
        init_values = {
            "dra": 0.0,
            "ddec": 0.0,
            "log10_flux": -3.0,
        }
    else:
        init_values = {
            "dra": float(init["dra"]),
            "ddec": float(init["ddec"]),
            "log10_flux": float(np.log10(max(init["flux"], 1e-12))),
        }

    kernel = NUTS(model_hmc, init_strategy=init_to_value(values=init_values))
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data_obj=oidata)
    s = mcmc.get_samples()

    flux_samples = 10.0 ** s["log10_flux"]

    median = {
        "dra": float(jnp.median(s["dra"])),
        "ddec": float(jnp.median(s["ddec"])),
        "flux": float(jnp.median(flux_samples)),
    }
    std = {
        "dra": float(jnp.std(s["dra"])),
        "ddec": float(jnp.std(s["ddec"])),
        "flux": float(jnp.std(flux_samples)),
    }
    return median, std


def _recover_hmc_fisher(
    oidata: OIData,
    init: dict[str, float],
    seed: int = 2027,
    num_warmup: int = 800,
    num_samples: int = 2000,
) -> tuple[dict[str, float], dict[str, float]]:
    params = ["dra", "ddec", "flux"]

    x0_dict = {
        "dra": float(init["dra"]),
        "ddec": float(init["ddec"]),
        "log10_flux": float(np.log10(max(init["flux"], 1e-12))),
    }
    x0, unravel = ravel_pytree(x0_dict)

    def objective(x):
        xdict = unravel(x)
        flux = 10.0 ** xdict["log10_flux"]
        values = jnp.array([xdict["dra"], xdict["ddec"], flux])
        return -loglike(values, params, oidata, BinaryModelCartesian)

    fmat = hessian_matrix(objective, x0)
    proj = fisher_projection(fmat)

    def model_hmc(data_obj: OIData):
        u = numpyro.sample(
            "u",
            dist.Normal(0.0, 1.0).expand([x0.shape[0]]).to_event(1),
        )
        log_q_u = dist.Normal(0.0, 1.0).log_prob(u).sum()
        x = x0 + jnp.dot(proj, u)
        xdict = unravel(x)

        dra = xdict["dra"]
        ddec = xdict["ddec"]
        log10_flux = xdict["log10_flux"]
        flux = 10.0**log10_flux

        numpyro.deterministic("dra", dra)
        numpyro.deterministic("ddec", ddec)
        numpyro.deterministic("flux", flux)
        log_prior_x = (
            dist.Uniform(-250.0, 250.0).log_prob(dra)
            + dist.Uniform(-250.0, 250.0).log_prob(ddec)
            + dist.Uniform(-6.0, -1.0).log_prob(log10_flux)
        )
        numpyro.factor("prior_correction", log_prior_x - log_q_u)
        numpyro.factor(
            "loglike",
            loglike([dra, ddec, flux], params, data_obj, BinaryModelCartesian),
        )

    kernel = NUTS(model_hmc)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data_obj=oidata)
    s = mcmc.get_samples()

    median = {
        "dra": float(jnp.median(s["dra"])),
        "ddec": float(jnp.median(s["ddec"])),
        "flux": float(jnp.median(s["flux"])),
    }
    std = {
        "dra": float(jnp.std(s["dra"])),
        "ddec": float(jnp.std(s["ddec"])),
        "flux": float(jnp.std(s["flux"])),
    }
    return median, std


def run_synthetic_binary_demo(
    output_path: str | Path = "docs/generated/synthetic_binary.oifits",
    num_warmup: int = 800,
    num_samples: int = 2000,
) -> RecoverySummary:
    """Generate synthetic OIFITS data and run grid/HMC recovery checks.

    ``num_warmup`` and ``num_samples`` set the length of both HMC runs.

    The generated file stores OIFITS phase channels in degrees while
    ``OIData`` normalizes loaded phases to internal radians.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dic, truth, noise_settings = _build_synthetic_oifits_dict()
    write_oifits(dic, output_path)

    # OIData reads the OIFITS file directly (with astropy).
    oidata = OIData(output_path)

    grid_est = _recover_grid(oidata)
    chain = {"num_warmup": num_warmup, "num_samples": num_samples}
    hmc_median, hmc_std = _recover_hmc(oidata, init=grid_est, **chain)
    fisher_hmc_median, fisher_hmc_std = _recover_hmc_fisher(
        oidata, grid_est, **chain
    )

    return RecoverySummary(
        truth=truth,
        grid_estimate=grid_est,
        hmc_median=hmc_median,
        hmc_std=hmc_std,
        fisher_hmc_median=fisher_hmc_median,
        fisher_hmc_std=fisher_hmc_std,
        noise_settings=noise_settings,
        output_file=str(output_path),
    )


def within_two_sigma(summary: RecoverySummary) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in ("dra", "ddec", "flux"):
        sigma = max(summary.hmc_std[key], 1e-10)
        checks[key] = (
            abs(summary.hmc_median[key] - summary.truth[key]) <= 2.0 * sigma
        )
    return checks


def fisher_within_three_sigma(summary: RecoverySummary) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in ("dra", "ddec", "flux"):
        sigma = max(summary.fisher_hmc_std[key], 1e-10)
        checks[key] = (
            abs(summary.fisher_hmc_median[key] - summary.truth[key])
            <= 3.0 * sigma
        )
    return checks


if __name__ == "__main__":
    result = run_synthetic_binary_demo()
    checks = within_two_sigma(result)
    fisher_checks = fisher_within_three_sigma(result)
    print("Saved synthetic OIFITS:", result.output_file)
    print("Truth:", result.truth)
    print("Grid estimate:", result.grid_estimate)
    print("HMC median:", result.hmc_median)
    print("HMC std:", result.hmc_std)
    print("Fisher-HMC median:", result.fisher_hmc_median)
    print("Fisher-HMC std:", result.fisher_hmc_std)
    print("2σ checks:", checks)
    print("Fisher 3σ checks:", fisher_checks)
