"""linear_flux_grid: closed-form linearised flux map (fouriever ``lincmap``)."""

import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.grid_fit import (
    Gaussian,
    LinearFluxGrid,
    LogUniform,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    linear_flux_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from virgil.likelihood import build_model, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from tests._test_data import oidata

TRUE_POS = (250.0, 150.0)


def _simulate(flux, noise_scale, seed=1):
    """Binary data on the nu Hor baselines, with errors scaled by noise_scale."""
    cvis = BinaryModelCartesian(*TRUE_POS, flux).model(
        oidata.u, oidata.v, oidata.wavel
    )
    rng = onp.random.default_rng(seed)
    d_vis, d_phi = oidata.d_vis * noise_scale, oidata.d_phi * noise_scale
    return OIData(
        {
            "u": oidata.u,
            "v": oidata.v,
            "wavel": oidata.wavel,
            "vis": oidata.to_vis(cvis)
            + rng.standard_normal(oidata.vis.shape) * d_vis,
            "d_vis": d_vis,
            "phi": oidata.to_phases(cvis)
            + rng.standard_normal(oidata.phi.shape) * d_phi,
            "d_phi": d_phi,
            "i_cps1": oidata.i_cps1,
            "i_cps2": oidata.i_cps2,
            "i_cps3": oidata.i_cps3,
            "v2_flag": oidata.v2_flag,
            "cp_flag": oidata.cp_flag,
        }
    )


def _grid():
    # The flux axis is ignored by linear_flux_grid; it only names the flux.
    return {
        "dra": np.array([TRUE_POS[0], 100.0, -300.0]),
        "ddec": np.array([TRUE_POS[1], -200.0, 50.0]),
        "flux": np.array([1e-3]),
    }


@pytest.fixture(scope="module")
def faint():
    data = _simulate(1e-3, noise_scale=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optimized = optimized_flux_grid(data, BinaryModelCartesian, _grid())
    laplace = laplace_flux_uncertainty_grid(
        data, BinaryModelCartesian, _grid(), flux=optimized
    )
    return data, optimized, laplace


def test_faint_companion_matches_optimizer_and_laplace(faint):
    data, optimized, laplace = faint
    flux, error, snr = linear_flux_grid(data, BinaryModelCartesian, _grid())[
        :3
    ]
    assert flux.shape == error.shape == snr.shape == (3, 3)
    # Measured: at the true pixel f_hat is 0.9977e-3 against 0.9992e-3
    # (0.15% apart, from O(f^2) terms), and sigma_f agrees to 0.3%.
    assert onp.isclose(flux[0, 0], optimized[0, 0], rtol=0.01)
    assert onp.isclose(flux[0, 0], 1e-3, rtol=0.05)
    assert onp.allclose(error, laplace, rtol=0.02)
    # Off-source pixels (f_hat ~ 4e-4 from the companion's leakage) agree
    # to well under a sigma_f (3e-6).
    assert onp.allclose(flux, optimized, atol=1e-5)
    assert onp.allclose(snr, flux / error)


def test_snr_peaks_at_true_position(faint):
    data = faint[0]
    fine = {
        "dra": TRUE_POS[0] + np.linspace(-30.0, 30.0, 13),
        "ddec": TRUE_POS[1] + np.linspace(-30.0, 30.0, 13),
        "flux": np.array([1e-3]),
    }
    _, _, snr = linear_flux_grid(data, BinaryModelCartesian, fine)[:3]
    assert np.unravel_index(np.argmax(snr), snr.shape) == (6, 6)


def test_bright_companion_is_biased():
    # Documented limitation: for f ~ 0.3 the linearisation underestimates
    # the flux (measured 0.20 for a true 0.30, with the optimizer at 0.30),
    # so we only check the sign and the size of the bias, not agreement.
    data = _simulate(0.3, noise_scale=0.01)
    flux, _, _ = linear_flux_grid(data, BinaryModelCartesian, _grid())[:3]
    assert 0.1 < flux[0, 0] < 0.27


def test_gradient_matches_finite_difference():
    """The jvp g equals the fouriever-style finite difference at f0=1e-4.

    fouriever's ``lincmap`` (``util.clin``) uses g = model(f0)/f0 with
    f0 = 1e-4 and the same weighted least squares; here the exact
    derivative is compared with that finite difference of the whitened
    residuals, through sigma_f = (g.g)**-0.5. (fouriever itself is not
    importable in the test environment.)
    """
    data = _simulate(1e-3, noise_scale=0.1)
    _, error, _ = linear_flux_grid(data, BinaryModelCartesian, _grid())[:3]
    params = ("dra", "ddec", "flux")

    def resid(flux):
        values = [TRUE_POS[0], TRUE_POS[1], flux]
        return whitened_residuals(
            build_model(BinaryModelCartesian, params, values), data
        )

    g = (resid(1e-4) - resid(0.0)) / 1e-4
    assert onp.isclose(error[0, 0], 1.0 / np.sqrt(np.sum(g * g)), rtol=0.02)


def test_gauss_newton_fixes_bright_companion():
    data = _simulate(0.3, noise_scale=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optimized = optimized_flux_grid(data, BinaryModelCartesian, _grid())
    laplace = laplace_flux_uncertainty_grid(
        data, BinaryModelCartesian, _grid(), flux=optimized
    )
    flux0, _, _ = linear_flux_grid(data, BinaryModelCartesian, _grid())[:3]
    flux, error, snr = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), n_iter=3
    )[:3]
    # Measured at f = 0.3: n_iter=0 gives 0.198; n_iter=3 gives 0.29999995
    # against the optimizer's 0.29999986 (3e-7 relative), and sigma_f agrees
    # with the Laplace value to 2e-5 relative (5.4843e-7 vs 5.4844e-7).
    assert onp.isclose(flux[0, 0], optimized[0, 0], rtol=0.01)
    assert onp.isclose(error[0, 0], laplace[0, 0], rtol=0.05)
    assert onp.allclose(snr, flux / error)
    # n_iter=0 is the default, unchanged.
    f_def = linear_flux_grid(data, BinaryModelCartesian, _grid())[0]
    f_zero = linear_flux_grid(data, BinaryModelCartesian, _grid(), n_iter=0)[0]
    assert onp.array_equal(f_def, f_zero)


@pytest.mark.parametrize("noise_scale", [0.1, 30.0])
def test_prior_matches_numerical_marginalisation(noise_scale):
    """Closed form against quadrature over f, in the linear regime (f=1e-3).

    noise_scale=0.1 is a strong detection (log B ~ 5e4) and 30 a marginal
    one (log B of order 1), where the log-det term matters.
    """
    data = _simulate(1e-3, noise_scale=noise_scale)
    mean, sd = 2e-3, 3e-3
    out = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=Gaussian(mean, sd)
    )
    base = linear_flux_grid(data, BinaryModelCartesian, _grid())
    assert base.posterior_mean is None and base.log_bayes_factor is None
    assert onp.array_equal(out.flux, base.flux)
    params = ("dra", "ddec", "flux")
    for ij in [(0, 0), (1, 1)]:
        dra, ddec = float(_grid()["dra"][ij[0]]), float(_grid()["ddec"][ij[1]])

        def resid(f):
            return whitened_residuals(
                build_model(BinaryModelCartesian, params, [dra, ddec, f]),
                data,
            )

        # Linear model about f = 0 (n_iter=0): r(f) = r0 + f g, with g the
        # jvp (a float32 finite difference would be too noisy here).
        r0, g = jax.jvp(resid, (0.0,), (1.0,))
        r0, g = onp.asarray(r0, dtype=float), onp.asarray(g, dtype=float)
        centre = float(out.posterior_mean[ij])
        width = 10 * float(out.posterior_sd[ij])
        fgrid = onp.linspace(centre - width, centre + width, 20001)
        # log L(f) - log L(0), up to float rounding, in log space.
        dchi = onp.sum(
            (r0[None] + fgrid[:, None] * g[None]) ** 2, axis=1
        ) - onp.sum(r0**2)
        logprior = -0.5 * ((fgrid - mean) / sd) ** 2 - onp.log(
            onp.sqrt(2 * onp.pi) * sd
        )
        logint = -0.5 * dchi + logprior
        m = logint.max()
        w = onp.exp(logint - m)
        zint = onp.trapezoid(w, fgrid)
        log_z = m + onp.log(zint)
        pm = onp.trapezoid(fgrid * w, fgrid) / zint
        psd = onp.sqrt(onp.trapezoid(fgrid**2 * w, fgrid) / zint - pm**2)
        assert onp.isclose(out.log_bayes_factor[ij], log_z, rtol=1e-3)
        assert onp.isclose(out.posterior_mean[ij], pm, rtol=1e-3)
        assert onp.isclose(out.posterior_sd[ij], psd, rtol=1e-3)


def test_numpyro_priors_match_the_named_tuples():
    data = _simulate(1e-3, noise_scale=0.1)
    for new, old in [
        (dist.LogUniform(1e-5, 1e-1), LogUniform(1e-5, 1e-1)),
        (dist.Normal(2e-3, 3e-3), Gaussian(2e-3, 3e-3)),
    ]:
        res_new = linear_flux_grid(
            data, BinaryModelCartesian, _grid(), prior=new
        )
        res_old = linear_flux_grid(
            data, BinaryModelCartesian, _grid(), prior=old
        )
        assert type(res_new) is type(res_old)
        for a, b in zip(res_new, res_old):
            assert onp.array_equal(a, b, equal_nan=True)


def test_bare_tuple_prior_is_an_error():
    data = _simulate(1e-3, noise_scale=0.1)
    with pytest.raises(TypeError, match=r"bare \(mean, sd\) tuple"):
        linear_flux_grid(
            data, BinaryModelCartesian, _grid(), prior=(2e-3, 3e-3)
        )


def test_numpyro_prior_needs_scalar_parameters():
    data = _simulate(1e-3, noise_scale=0.1)
    with pytest.raises(ValueError, match="scalar"):
        linear_flux_grid(
            data,
            BinaryModelCartesian,
            _grid(),
            prior=dist.Normal(np.zeros(2), 1.0),
        )
    with pytest.raises(ValueError, match="0 < f_min < f_max"):
        linear_flux_grid(
            data,
            BinaryModelCartesian,
            _grid(),
            prior=dist.LogUniform(1e-2, 1e-3),
        )


def test_grid_tools_take_flux_param_and_batch_size_by_keyword():
    data = _simulate(1e-3, noise_scale=0.1)
    grid = _grid()
    for fn in (
        linear_flux_grid,
        laplace_flux_uncertainty_grid,
        optimized_flux_grid,
    ):
        with pytest.raises(TypeError):
            fn(data, BinaryModelCartesian, grid, None, "flux")
    with pytest.raises(TypeError):
        linear_flux_grid(data, BinaryModelCartesian, grid, "flux")
    with pytest.raises(TypeError):
        optimized_likelihood_grid(data, BinaryModelCartesian, grid, "flux")
    with pytest.raises(TypeError):
        likelihood_grid(data, BinaryModelCartesian, grid, 10**6)


def test_prior_validation():
    data = _simulate(1e-3, noise_scale=0.1)
    for bad in [
        LogUniform(1e-3, 1e-4),
        LogUniform(0.0, 1.0),
        LogUniform(1e-3, float("inf")),
        LogUniform(float("nan"), 1.0),
        Gaussian(0, 0),
        Gaussian(float("inf"), 1.0),
        Gaussian(0.0, float("inf")),
        Gaussian(float("nan"), 1.0),
    ]:
        with pytest.raises(ValueError):
            linear_flux_grid(data, BinaryModelCartesian, _grid(), prior=bad)
    with pytest.raises(TypeError):
        linear_flux_grid(data, BinaryModelCartesian, _grid(), prior=1.0)


def _brute_log_uniform(f_hat, sigma, f_min, f_max, n=2_000_001):
    """Evidence, mean, sd by trapezoid on a dense log grid (float64)."""
    u = onp.linspace(onp.log(f_min), onp.log(f_max), n)
    f = onp.exp(u)
    h = -((f - f_hat) ** 2 - f_hat**2) / (2 * sigma**2)
    m = h.max()
    w = onp.exp(h - m)
    z = onp.trapezoid(w, u)
    mean = onp.trapezoid(w * f, u) / z
    sd = onp.sqrt(onp.trapezoid(w * (f - mean) ** 2, u) / z)
    return m + onp.log(z) - onp.log(u[-1] - u[0]), mean, sd


@pytest.mark.parametrize("noise_scale", [0.1, 30.0])
def test_log_uniform_matches_brute_force(noise_scale):
    """Strong (log B ~ 5e4) and marginal (log B of order 1) detections."""
    data = _simulate(1e-3, noise_scale=noise_scale)
    f_min, f_max = 1e-6, 1.0
    out = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=LogUniform(f_min, f_max)
    )
    base = linear_flux_grid(data, BinaryModelCartesian, _grid())
    assert onp.array_equal(out.flux, base.flux)
    for ij in [(0, 0), (1, 1)]:
        f_hat = float(out.flux[ij])
        sigma = float(out.flux_error[ij])
        log_b, mean, sd = _brute_log_uniform(f_hat, sigma, f_min, f_max)
        assert onp.isclose(out.log_bayes_factor[ij], log_b, rtol=1e-4)
        assert onp.isclose(out.posterior_mean[ij], mean, rtol=1e-4)
        assert onp.isclose(out.posterior_sd[ij], sd, rtol=1e-4)


def test_log_uniform_occam_factor():
    """Widening the bounds shifts log B by -Delta ln ln(f_max / f_min)."""
    data = _simulate(1e-3, noise_scale=0.1)
    b1 = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=LogUniform(1e-6, 1.0)
    )
    b2 = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=LogUniform(1e-8, 1e2)
    )
    expected = -onp.log(onp.log(1e10) / onp.log(1e6))
    shift = onp.asarray(b2.log_bayes_factor - b1.log_bayes_factor)
    assert onp.isclose(shift[0, 0], expected, atol=1e-3)
    assert onp.allclose(b1.posterior_mean, b2.posterior_mean, rtol=1e-4)


def test_log_uniform_estimate_outside_bounds_is_finite():
    """A peak far above f_max or below f_min still gives a finite log B."""
    data = _simulate(1e-3, noise_scale=0.1)
    for bounds in [(1e-9, 1e-7), (1e-1, 1.0)]:
        out = linear_flux_grid(
            data, BinaryModelCartesian, _grid(), prior=LogUniform(*bounds)
        )
        assert onp.all(onp.isfinite(out.log_bayes_factor))
        assert onp.all(out.posterior_mean >= bounds[0] * 0.999)
        assert onp.all(out.posterior_mean <= bounds[1] * 1.001)


@pytest.mark.parametrize(
    "prior", [None, Gaussian(2e-3, 3e-3), LogUniform(1e-6, 1.0)]
)
def test_return_type_does_not_depend_on_prior(prior):
    data = _simulate(1e-3, noise_scale=0.1)
    res = linear_flux_grid(data, BinaryModelCartesian, _grid(), prior=prior)
    assert isinstance(res, LinearFluxGrid)
    assert res._fields == (
        "flux",
        "flux_error",
        "snr",
        "posterior_mean",
        "posterior_sd",
        "log_bayes_factor",
    )
    flux, error, snr = res[:3]
    assert onp.allclose(snr, flux / error)
    assert (res.log_bayes_factor is None) == (prior is None)


def test_log_uniform_float32_edge_cases():
    """Very wide bounds (f_max/f_min overflows float32) and f_hat above f_max."""
    data = _simulate(1e-3, noise_scale=0.1)
    wide = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=LogUniform(1e-30, 1e10)
    )
    assert onp.all(onp.isfinite(wide.log_bayes_factor))
    # f_hat (1e-3) far above f_max: compare with brute force from f_hat, sigma.
    out = linear_flux_grid(
        data, BinaryModelCartesian, _grid(), prior=LogUniform(1e-9, 1e-7)
    )
    f_hat, sigma = float(out.flux[0, 0]), float(out.flux_error[0, 0])
    log_b, _, _ = _brute_log_uniform(f_hat, sigma, 1e-9, 1e-7)
    assert onp.isclose(out.log_bayes_factor[0, 0], log_b, rtol=1e-4)
