"""Chromatic companion searches: a WR star with emission lines plus an OB star.

The Wolf-Rayet primary has a continuum plus a Gaussian emission line; the OB
companion has a continuum only, so the flux ratio drops through the line. The
grids, limits and detection statistics take ``comp.flux.ratio`` as the flux
axis, and the differential phases and spectra enter beside V2 and closure
phases.
"""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import (
    GaussianLine,
    OIData,
    PointSource,
    PowerLaw,
    Scaled,
    Sum,
    System,
)
from virgil.coverage import vlti_oidata
from virgil.detection import (
    DetectionMC,
    detection_statistics,
    injection_recovery,
)
from virgil.grid_fit import likelihood_grid, linear_flux_grid
from virgil.spectra import reference_flux
from virgil.limits import absil_limits, injection_limits
from tests._shared import limit_significances

BRG = 2.1661e-6
W0 = BRG
FWHM = 1.5e-9
WAVES = onp.linspace(BRG - 3e-9, BRG + 3e-9, 9)
CONTINUUM = [
    (WAVES[0] - 1e-10, WAVES[1] + 1e-10),
    (WAVES[-2] - 1e-10, WAVES[-1] + 1e-10),
]
LINE = [(WAVES[3] - 1e-10, WAVES[5] + 1e-10)]

TRUTH = {"dra": 8.0, "ddec": -5.0}
RATIO = 0.08


def _primary():
    return PointSource(
        flux=Sum(
            continuum=PowerLaw(1.0, wavel0=W0),
            brg=GaussianLine(3.0, BRG, FWHM),
        )
    )


def _scene(flux, dra=TRUTH["dra"], ddec=TRUTH["ddec"]):
    return System(
        star=_primary(), comp=PointSource(dra=dra, ddec=ddec, flux=flux)
    )


def _power(ratio):
    return PowerLaw(ratio, wavel0=W0)


def _scaled(ratio):
    return Scaled(PowerLaw(1.0, wavel0=W0), ratio)


def _template(blocks=("v2", "t3", "visphi", "flux"), sigma_phi=0.01, unit=1.0):
    """GRAVITY AT-like snapshots with zero data, in dictionary form."""
    base = vlti_oidata(
        hour_angles_h=(-1.0, 1.0),
        wavelengths_m=WAVES,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    n = base.u.size
    record = {
        "u": onp.asarray(base.u),
        "v": onp.asarray(base.v),
        "wavel": onp.asarray(base.wavel),
        "frame": onp.asarray(base.frame),
        "stations": onp.asarray(base.stations),
        "vis": onp.ones(n),
        "d_vis": onp.full(n, 0.01),
        "phi": onp.zeros(base.phi.size),
        "d_phi": onp.full(base.phi.size, onp.radians(0.5)),
        "i_cps1": onp.asarray(base.i_cps1),
        "i_cps2": onp.asarray(base.i_cps2),
        "i_cps3": onp.asarray(base.i_cps3),
    }
    if "v2" not in blocks:
        record["vis"] = onp.full(n, onp.nan)
    if "t3" not in blocks:
        record["i_cps1"] = record["i_cps2"] = record["i_cps3"] = None
        record["phi"] = record["d_phi"] = None
        record["cp_flag"] = False
    if "visphi" in blocks:
        record["visphi"] = {
            "sample": onp.arange(n),
            "value": onp.zeros(n),
            "error": onp.full(n, sigma_phi),
        }
    if "flux" in blocks:
        rows = onp.repeat(onp.arange(n // WAVES.size), WAVES.size)
        record["flux"] = {
            "wavel": onp.asarray(base.wavel),
            "value": onp.ones(n),
            "error": onp.full(n, 0.05 * unit),
            "row": rows,
        }
    return record


def _data(
    truth,
    blocks=("v2", "t3", "visphi", "flux"),
    sigma_phi=0.01,
    key=1,
    unit=1.0,
):
    """``unit`` is the spectrum's continuum level (1, or e.g. raw ADU)."""
    data = OIData(_template(blocks, sigma_phi, unit))
    if "flux" in blocks:
        data = data.with_flux_scale(scale=(unit, 0.5 * unit))
    if "visphi" in blocks:
        data = data.with_continuum(CONTINUUM, lines=LINE)
    return data.with_model(truth, key=jax.random.PRNGKey(key))


@pytest.fixture(scope="module")
def wr_data():
    """V2, T3, differential phases and a flux spectrum of the WR + OB pair."""
    return _data(_scene(_power(RATIO)))


COMPANIONS = {"power": _power, "scaled": _scaled}


def _grid(ratios):
    return {
        "comp.dra": onp.array([-8.0, 0.0, 8.0]),
        "comp.ddec": onp.array([-5.0, 0.0, 5.0]),
        "comp.flux.ratio": onp.asarray(ratios),
    }


def test_scaled_is_the_template_times_a_ratio():
    template = Sum(
        continuum=PowerLaw(0.5, index=-2.0, wavel0=W0),
        line=GaussianLine(1.0, BRG, FWHM),
    )
    scaled = Scaled(template, 0.1)
    wavel = onp.linspace(BRG - 4e-9, BRG + 4e-9, 17)
    # The template's shape, normalized to ratio at its reference wavelength
    # (here template() != 1), so that ratio is a flux ratio.
    assert float(template()) != pytest.approx(1.0, rel=0.1)
    onp.testing.assert_allclose(
        scaled(wavel), 0.1 * template(wavel) / template(), rtol=1e-6
    )
    assert float(scaled()) == pytest.approx(0.1, rel=1e-6)
    assert float(reference_flux(scaled)) == pytest.approx(0.1, rel=1e-6)
    assert bool(scaled.is_physical())
    with pytest.raises(ValueError, match="negative"):
        Scaled(template, -0.1)
    with pytest.raises(TypeError, match="Spectrum"):
        Scaled(0.5, 0.1)


def test_scaled_holds_the_template_fixed_and_ratio_is_a_flux_param():
    from virgil._utils import is_flux_param

    scene = _scene(Scaled(PowerLaw(1.0, index=-2.0, wavel0=W0), 0.1))
    assert is_flux_param("comp.flux.ratio")

    def total(ratio, index, amplitude):
        spectrum = Scaled(PowerLaw(amplitude, index, W0), ratio)
        return spectrum(np.array([BRG * 1.01]))[0]

    d_ratio, d_index, d_amp = jax.grad(total, argnums=(0, 1, 2))(
        0.1, -2.0, 1.0
    )
    assert float(d_ratio) > 0.0
    assert float(d_index) == 0.0 and float(d_amp) == 0.0
    # The ratio resolves as a path of the scene.
    assert float(scene.get("comp.flux.ratio")) == pytest.approx(0.1)


@pytest.mark.parametrize("companion", COMPANIONS)
def test_likelihood_and_linear_flux_grids_recover_the_chromatic_companion(
    wr_data, companion
):
    template = _scene(COMPANIONS[companion](0.05))
    grid = _grid([0.04, 0.08, 0.16])
    loglike = onp.asarray(likelihood_grid(template, data=wr_data, grid=grid))
    best = onp.unravel_index(onp.argmax(loglike), loglike.shape)
    assert [float(grid[k][i]) for k, i in zip(grid, best)] == [8.0, -5.0, 0.08]

    result = linear_flux_grid(template, wr_data, grid, n_iter=3)
    snr = onp.asarray(result.snr)
    i, j = onp.unravel_index(onp.nanargmax(snr), snr.shape)
    assert (grid["comp.dra"][i], grid["comp.ddec"][j]) == (8.0, -5.0)
    flux = float(result.flux[i, j])
    assert flux == pytest.approx(RATIO, rel=0.1)
    # And within 3 sigma_flux (= flux / snr), like the other statistical checks.
    assert abs(flux - RATIO) < 3 * flux / snr[i, j]
    assert snr[i, j] > 20.0


def test_limits_are_finite_and_sensible_for_the_chromatic_companion(wr_data):
    positions = {
        "comp.dra": onp.array([8.0, -6.0]),
        "comp.ddec": onp.array([-5.0]),
        "comp.flux.ratio": onp.array([0.01]),
    }
    limits = {}
    for companion, make in COMPANIONS.items():
        template = _scene(make(0.05))
        for name, function in (
            ("injection", injection_limits),
            ("absil", absil_limits),
        ):
            limit = onp.asarray(function(template, wr_data, positions, 3.0))
            assert limit.shape == (2, 1)
            assert onp.all(onp.isfinite(limit))
            # The limit solves its own definition, evaluated through the
            # public likelihood (not a bound: it fails if the flux is off).
            onp.testing.assert_allclose(
                limit_significances(template, wr_data, positions, limit, name),
                3.0,
                atol=1e-2,
            )
            limits[companion, name] = limit
    # Both spectra are the same model, so the same limits.
    for name in ("injection", "absil"):
        onp.testing.assert_allclose(
            limits["power", name], limits["scaled", name], rtol=1e-3
        )
    # A 3-sigma limit is far below the companion that is in the data, at
    # the empty position (-6, -5).
    assert limits["power", "injection"][1, 0] < 0.05


@pytest.mark.slow
def test_detection_statistics_and_mc_detect_the_chromatic_companion(wr_data):
    grid = _grid(onp.geomspace(0.02, 0.3, 6))
    template = _scene(_scaled(0.05))
    stats = detection_statistics(template, wr_data, grid)
    assert (
        float(stats["comp.dra"]) == 8.0 and float(stats["comp.ddec"]) == -5.0
    )
    assert float(stats["comp.flux.ratio"]) == pytest.approx(RATIO, rel=0.3)
    assert float(stats["delta_chi2"]) > 100.0
    assert float(stats["max_snr"]) > 10.0

    injections = {
        "comp.dra": onp.array([8.0]),
        "comp.ddec": onp.array([-5.0]),
        "comp.flux.ratio": onp.array([RATIO]),
    }
    mc = injection_recovery(
        template,
        _scene(0.0),
        OIData(_template())
        .with_flux_scale(scale=(1.0, 0.5))
        .with_continuum(CONTINUUM, lines=LINE),
        grid,
        0,
        n_null=2,
        injections=injections,
        chunk_size=2,
        progress=False,
    )
    assert isinstance(mc, DetectionMC)
    assert float(mc.injected["best_dra"][0]) == 8.0
    assert float(mc.injected["best_ddec"][0]) == -5.0
    assert float(mc.injected["best_flux"][0]) == pytest.approx(RATIO, rel=0.5)
    assert mc.injected["delta_chi2"][0] > 10.0 * mc.null["delta_chi2"].max()


def test_differential_phase_alone_detects_what_a_grey_fit_misses():
    # 0.1 mas is a fiftieth of the 4 mas resolution of the 130 m baselines:
    # V2 and closure phases change by a fraction of their errors (the
    # differential phases have smaller errors, 0.005 rad, than the closure
    # phases, 0.0087 rad, and more phases per sample). The
    # photocentre moves with the line (the flux ratio drops from 0.2 to
    # about 0.05 across it), so the differential phase sees it.
    sep = 0.1
    truth = _scene(_power(0.2), dra=sep, ddec=0.0)
    differential = _data(truth, ("visphi",), sigma_phi=0.005, key=2)
    grey = _data(truth, ("v2", "t3"), key=2)
    axis = onp.array([-sep, 0.0, sep])
    ratios = onp.geomspace(0.05, 0.5, 8)
    chromatic = detection_statistics(
        _scene(_power(0.1)),
        differential,
        {
            "comp.dra": axis,
            "comp.ddec": onp.zeros(1),
            "comp.flux.ratio": ratios,
        },
    )
    constant = detection_statistics(
        _scene(0.1),
        grey,
        {"comp.dra": axis, "comp.ddec": onp.zeros(1), "comp.flux": ratios},
    )
    assert float(chromatic["comp.dra"]) == pytest.approx(sep, rel=1e-5)
    assert float(chromatic["delta_chi2"]) > 30.0
    assert float(constant["delta_chi2"]) < 9.0


ADU = 1e5


def _flux_limit(data):
    positions = {
        "comp.dra": onp.array([0.2]),
        "comp.ddec": onp.array([0.0]),
        "comp.flux.ratio": onp.array([0.01]),
    }
    template = _scene(_power(0.05))
    return float(injection_limits(template, data, positions, 3.0)[0, 0])


def _null_flux_data(unit, mu):
    """Null-scene data whose spectrum has grey scale ~``unit``, then the
    scale prior restated with mean ``mu`` (the data are not regenerated)."""
    blocks = ("v2", "t3", "flux")
    data = _data(_scene(_power(1e-9)), blocks, unit=unit)
    return data.with_flux_scale(scale=(mu, 0.5 * mu))


@pytest.mark.parametrize("mu", [ADU, ADU / 2, 2 * ADU, ADU / 3])
def test_oiflux_injection_limit_ignores_the_stated_scale_prior(mu):
    """Audit finding 724#12.3b: injected at the data's grey scale, not at mu.

    `FluxSpectrum.predict` returns ``mu t``, so adding its change of ``t`` at
    ``mu`` scaled the injection by ``mu / k`` (limits ~3x too loose at
    ``mu = k / 3``, clipped to the search bound below). The limit must not
    depend on the stated prior mean, nor on the data's units.
    """
    reference = _flux_limit(_null_flux_data(1.0, 1.0))
    assert _flux_limit(_null_flux_data(ADU, mu)) == pytest.approx(
        reference, rel=1e-3
    )


def test_oiflux_tightens_the_injection_limit():
    """The spectrum tightens an unresolved companion's limit (null data)."""
    blocks = ("v2", "t3")
    without = _flux_limit(_data(_scene(_power(1e-9)), blocks))
    with_flux = _flux_limit(_null_flux_data(ADU, ADU))
    assert with_flux < 0.8 * without
