"""Chromatic scenes: spectra, resolved flux, several files, likelihood options."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import (
    BlackBody,
    GaussianDisk,
    ModulatedGaussianRim,
    OIData,
    PointSource,
    PowerLaw,
    Resolved,
    System,
    UniformDisk,
    write_oifits,
)
from virgil._utils import is_flux_param, resolve_flux_param
from virgil.spectra import Tabulated
from virgil.likelihood import (
    joint_loglike,
    model_loglike,
    numpyro_model,
)
from tests.test_oifits import TRUTH, _tables

WAVES = onp.array([1.55e-6, 1.65e-6, 1.75e-6])


def _scene(flux_s=PowerLaw(0.05, -1.0, 1.65e-6)):
    return System(
        primary=UniformDisk(0.6, flux=PowerLaw(0.6, -4.0, 1.65e-6)),
        secondary=PointSource(flux=flux_s, dra=3.0, ddec=-1.0),
        rim=GaussianDisk(4.0, flux=PowerLaw(0.3, 2.0, 1.65e-6)),
        background=Resolved(flux=PowerLaw(0.05, 1.0, 1.65e-6)),
    )


def test_power_law_at_index_zero_is_achromatic():
    u, v = onp.array([30.0, -20.0]), onp.array([10.0, 40.0])
    chromatic = System(
        star=PointSource(flux=PowerLaw(1.0, 0.0)),
        comp=PointSource(flux=PowerLaw(0.1, 0.0), dra=5.0),
    )
    plain = System(star=PointSource(), comp=PointSource(flux=0.1, dra=5.0))
    for wavel in WAVES:
        assert np.allclose(
            chromatic.model(u, v, wavel), plain.model(u, v, wavel)
        )


def test_chromatic_system_matches_sparco_formula():
    u, v = onp.array([30.0, -20.0, 5.0]), onp.array([10.0, 40.0, -60.0])
    scene = _scene()
    for wavel in WAVES:
        ratio = wavel / 1.65e-6
        weights = {
            "primary": 0.6 * ratio**-4.0,
            "secondary": 0.05 * ratio**-1.0,
            "rim": 0.3 * ratio**2.0,
            "background": 0.05 * ratio**1.0,
        }
        numerator = sum(
            weights[name] * scene.components[name].model(u, v, wavel)
            for name in ("primary", "secondary", "rim")
        )
        expected = numerator / sum(weights.values())
        assert np.allclose(scene.model(u, v, wavel), expected, atol=1e-6)


def test_spectrum_parameters_are_paths():
    scene = _scene().set("secondary.flux.index", 3.0)
    assert float(scene.secondary.flux.index) == 3.0
    assert is_flux_param("secondary.flux.ratio")
    assert not is_flux_param("secondary.flux.index")
    assert (
        resolve_flux_param(["secondary.dra", "secondary.flux.ratio"])
        == "secondary.flux.ratio"
    )
    with pytest.raises(ValueError, match="negative"):
        PowerLaw(-0.1, 1.0)


def test_resolved_flux_only_lowers_the_visibilities():
    u, v = onp.array([0.0, 30.0]), onp.array([0.0, 10.0])
    scene = System(star=PointSource(), background=Resolved(flux=0.25))
    cvis = scene.model(u, v, 1.65e-6)
    assert np.allclose(cvis, np.array([1.0, 1.0 / 1.25]))
    image = scene.render(npix=16, fov_mas=10.0)
    assert np.isclose(float(image.sum()), 1.0)


def test_several_files_read_as_one(tmp_path):
    paths = [
        write_oifits(_tables(waves=(1.6e-6, 1.7e-6)), tmp_path / "a.fits"),
        write_oifits(_tables(waves=(1.55e-6,)), tmp_path / "b.fits"),
    ]
    joined = OIData(paths)
    singles = [OIData(path) for path in paths]
    assert joined.u.size == sum(single.u.size for single in singles)
    assert joined.phi.size == sum(single.phi.size for single in singles)
    assert joined.wavel.shape == joined.u.shape
    assert np.allclose(
        model_loglike(TRUTH, joined),
        joint_loglike(None, lambda params, index: TRUTH, singles),
        rtol=1e-6,
    )


def _toon_loglike(model, data_obj, vis_error_rel, phi_error):
    # The hand-written likelihood this option replaces (iras08_modelling).
    model_data = data_obj.model(model)
    data, errors = data_obj.flatten_data()
    n_vis = data_obj.vis.size
    errors_vis = np.hypot(errors[:n_vis], vis_error_rel * model_data[:n_vis])
    errors_phi = np.hypot(errors[n_vis:], phi_error)
    errors = np.concatenate([errors_vis, errors_phi])
    # Closure-phase residuals Δ enter as sin Δ, plus the periodic penalty
    # 2 sin²(Δ/2)/σ, as in virgil.likelihood.whitened_residuals (the
    # original used Δ). The original also treated the four closure phases
    # of each frame and channel as independent, counting them 4/3 times;
    # they are whitened as correlated groups instead (OIData.cp_noise; see
    # test_closure). The penalty rows add nothing to the normalization.
    resid = model_data - data
    phase, phase_errors = data_obj.cp_noise.whiten(
        np.sin(resid[n_vis:]), errors_phi
    )
    penalty = 2.0 * np.sin(0.5 * resid[n_vis:]) ** 2 / errors_phi
    whitened = np.concatenate([resid[:n_vis] / errors_vis, phase])
    errors = np.concatenate([errors_vis, phase_errors])
    return (
        -0.5 * np.sum(whitened**2)
        - 0.5 * np.sum(penalty**2)
        - np.sum(np.log(errors))
        - whitened.size / 2 * np.log(2 * np.pi)
    )


def test_error_inflation_matches_hand_written_likelihood(tmp_path):
    data = OIData(write_oifits(_tables(waves=WAVES), tmp_path / "x.fits"))
    model = _scene()
    expected = _toon_loglike(model, data, 0.02, 0.01)
    got = model_loglike(model, data, vis_error_rel=0.02, phi_error=0.01)
    assert np.allclose(got, expected, rtol=1e-6)
    assert not np.allclose(got, model_loglike(model, data), rtol=1e-3)


def test_unphysical_models_are_rejected_under_jit(tmp_path):
    data = OIData(write_oifits(_tables(waves=WAVES), tmp_path / "x.fits"))

    def rim_scene(rim_flux, az_amp):
        return System(
            star=PointSource(),
            rim=ModulatedGaussianRim(
                10.0,
                2.0,
                30.0,
                20.0,
                az_amps=az_amp,
                az_pas=0.0,
                flux=rim_flux,
            ),
        )

    @jax.jit
    def logl(rim_flux, az_amp):
        return model_loglike(
            rim_scene(rim_flux, az_amp), data, reject_unphysical=True
        )

    assert np.isfinite(logl(0.3, 0.5))
    assert logl(-0.1, 0.5) == -np.inf  # negative (e.g. remainder) flux
    assert logl(0.3, 1.5) == -np.inf  # rim brightness goes negative


def test_flux_ratio_priors_must_be_non_negative():
    with pytest.raises(ValueError, match="negative"):
        numpyro_model(
            _scene(), {"secondary.flux.ratio": dist.Normal(0.0, 1.0)}, None
        )


def test_render_under_jit():
    rim = lambda diam: ModulatedGaussianRim(
        diam, 2.0, 30.0, 40.0, az_amps=0.5, az_pas=60.0
    )
    render = jax.jit(lambda diam: rim(diam).render(npix=32, fov_mas=30.0))
    assert np.allclose(render(12.0), rim(12.0).render(npix=32, fov_mas=30.0))


def test_resolved_alone_cannot_be_rendered():
    with pytest.raises(ValueError, match="no image"):
        Resolved(flux=0.2).render(npix=16, fov_mas=10.0)


def test_several_files_need_a_target_name(tmp_path):
    paths = [
        write_oifits(_tables(waves=(1.6e-6,)), tmp_path / f"{name}.fits")
        for name in "ab"
    ]
    with pytest.raises(TypeError, match="by name"):
        OIData(paths, target=1)


def test_zero_total_flux_is_unphysical(tmp_path):
    data = OIData(write_oifits(_tables(waves=WAVES), tmp_path / "x.fits"))
    scene = System(star=PointSource(), background=Resolved(flux=0.1))

    @jax.jit
    def logl(star_flux, bkg_flux):
        dark = scene.set(
            ["star.flux", "background.flux"], [star_flux, bkg_flux]
        )
        return model_loglike(dark, data, reject_unphysical=True)

    assert np.isfinite(logl(1.0, 0.1))
    assert logl(0.0, 0.0) == -np.inf


def test_power_law_reference_wavelength_must_be_positive():
    with pytest.raises(ValueError, match="wavel0"):
        PowerLaw(0.1, index=1.5, wavel0=0.0)
    traced = jax.jit(
        lambda w0: PointSource(flux=PowerLaw(0.1, 1.5, w0)).is_physical()
    )
    assert bool(traced(1.65e-6))
    assert not bool(traced(-1.65e-6))


# Planck's law written out independently of virgil.spectra.
_H, _C, _K = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def _planck(wavel, temperature):
    return wavel**-5.0 / onp.expm1(_H * _C / (wavel * _K * temperature))


@pytest.mark.parametrize("temperature", [1120.0, 2400.0, 7250.0])
def test_black_body_is_a_planck_spectrum_normalized_at_wavel0(temperature):
    spectrum = BlackBody(0.3, temperature, 1.65e-6)
    expected = (
        0.3 * _planck(WAVES, temperature) / _planck(1.65e-6, temperature)
    )
    assert onp.allclose(onp.asarray(spectrum(WAVES)), expected, rtol=1e-5)
    assert float(spectrum(None)) == pytest.approx(0.3)


def test_cold_black_body_does_not_overflow():
    cold = BlackBody(0.3, 50.0)
    assert float(cold(1.65e-6)) == pytest.approx(0.3)
    values = onp.asarray(cold(WAVES))
    assert onp.all(onp.isfinite(values)) and onp.all(values >= 0.0)


def test_hot_black_body_tends_to_rayleigh_jeans():
    hot = BlackBody(1.0, 1e7)(WAVES)
    assert onp.allclose(hot, PowerLaw(1.0, -4.0)(WAVES), rtol=1e-3)


def test_black_body_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        BlackBody(0.1, 0.0)
    assert not bool(
        BlackBody(0.1, 1000.0).set("temperature", -5.0).is_physical()
    )


def test_sparco_fractions_and_temperatures_match_the_published_formula():
    # Hillen et al. (2016), eq. 1: V = Σ f_i Λ_i V_i / Σ f_i Λ_i, with
    # Σ f_i = 1 and every Λ_i = 1 at 1.65 µm. Relative ratios that are not
    # normalized give the same visibilities.
    fractions = {"pri": 0.597, "sec": 0.039, "ring": 0.209, "back": 0.155}
    temperatures = {"sec": 4000.0, "ring": 1120.0, "back": 2400.0}
    rim = ModulatedGaussianRim(14.15, 3.2, 19.0, 6.0, az_amps=0.4, az_pas=60.0)
    scale = 2.0  # unnormalized ratios
    scene = System(
        pri=PointSource(flux=PowerLaw(scale * fractions["pri"], -4.0)),
        sec=PointSource(
            flux=BlackBody(scale * fractions["sec"], temperatures["sec"]),
            dra=0.67,
            ddec=0.45,
        ),
        ring=rim.set("flux", BlackBody(scale * fractions["ring"], 1120.0)),
        back=Resolved(flux=BlackBody(scale * fractions["back"], 2400.0)),
    )
    u, v = onp.array([30.0, -20.0, 5.0]), onp.array([10.0, 40.0, -60.0])
    for wavel in WAVES:
        lam = {"pri": (wavel / 1.65e-6) ** -4.0}
        for name, temperature in temperatures.items():
            lam[name] = _planck(wavel, temperature) / _planck(
                1.65e-6, temperature
            )
        unit = {
            "pri": 1.0,
            "sec": scene.sec.set("flux", 1.0).model(u, v, wavel),
            "ring": rim.model(u, v, wavel),
            "back": 0.0,
        }
        expected = sum(
            fractions[n] * lam[n] * unit[n] for n in fractions
        ) / sum(fractions[n] * lam[n] for n in fractions)
        assert np.allclose(scene.model(u, v, wavel), expected, atol=1e-5)


def test_each_channel_of_a_multichannel_dataset_gets_its_own_weights():
    # V² and closure phases of a SPARCO scene observed in several channels at
    # once equal those of the scene observed one channel at a time.
    from virgil.coverage import vlti_oidata

    scene = _scene(flux_s=BlackBody(0.05, 3000.0))
    together = vlti_oidata(hour_angles_h=(0.0, 2.0), wavelengths_m=WAVES)
    joint = onp.asarray(together.model(scene))
    n_vis = together.vis.size
    for k, wavel in enumerate(WAVES):
        single = vlti_oidata(hour_angles_h=(0.0, 2.0), wavelengths_m=[wavel])
        model = onp.asarray(single.model(scene))
        nv = single.vis.size
        assert onp.allclose(
            joint[:n_vis][k :: len(WAVES)], model[:nv], atol=1e-6
        )
        assert onp.allclose(
            joint[n_vis:][k :: len(WAVES)], model[nv:], atol=1e-5
        )


def test_temperatures_have_gradients_and_are_not_fluxes():
    scene = _scene(flux_s=BlackBody(0.05, 3000.0))
    u, v = onp.array([30.0, -20.0]), onp.array([10.0, 40.0])

    def v2(temperature):
        model = scene.set("secondary.flux.temperature", temperature)
        return np.sum(np.abs(model.model(u, v, 1.55e-6)) ** 2)

    assert onp.isfinite(float(jax.grad(v2)(3000.0)))
    assert float(jax.grad(v2)(3000.0)) != 0.0
    assert not is_flux_param("secondary.flux.temperature")


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_tabulated_interpolates_between_channels():
    spectrum = Tabulated([0.2, 0.4, 0.1], WAVES)
    assert np.allclose(spectrum(WAVES), np.array([0.2, 0.4, 0.1]))
    assert np.isclose(spectrum(1.60e-6), 0.3)
    assert np.isclose(spectrum(1.0e-6), 0.2)  # constant beyond the ends
    assert np.isclose(spectrum(), np.mean(np.array([0.2, 0.4, 0.1])))


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_reference_flux_is_each_spectrums_own_reference():
    from virgil.spectra import reference_flux

    nodes = Tabulated([0.2, 0.4, 0.1], WAVES)
    assert reference_flux(nodes).shape == ()
    assert np.isclose(
        reference_flux(nodes), np.mean(np.array([0.2, 0.4, 0.1]))
    )
    assert np.isclose(reference_flux(PowerLaw(0.3, 1.0)), 0.3)
    assert np.isclose(reference_flux(BlackBody(0.3, 3000.0)), 0.3)
    assert reference_flux(0.25) == 0.25


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_negative_tabulated_node_is_rejected_despite_positive_mean():
    # Tabulated rejects negatives itself, so inject one past its constructor.
    bad = eqx.tree_at(
        lambda s: s.ratio,
        Tabulated([0.2, 0.4, 0.1], WAVES),
        np.array([0.5, -0.1, 0.4]),
    )
    assert float(bad()) > 0.0
    with pytest.raises(ValueError, match="must be non-negative"):
        PointSource(flux=bad)


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_tabulated_rejects_bad_tables():
    with pytest.raises(ValueError, match="non-negative"):
        Tabulated([0.2, -0.1, 0.1], WAVES)
    with pytest.raises(ValueError, match="increasing"):
        Tabulated([0.2, 0.1, 0.1], WAVES[::-1])
    with pytest.raises(ValueError, match="finite and non-negative"):
        Tabulated([0.2, onp.nan, 0.1], WAVES)
    with pytest.raises(ValueError, match="positive"):
        Tabulated([0.2, 0.1, 0.1], onp.array([-1.0e-6, 1.6e-6, 1.7e-6]))
    with pytest.raises(ValueError, match="finite"):
        Tabulated([0.2, 0.1, 0.1], onp.array([1.5e-6, onp.nan, 1.7e-6]))
    with pytest.raises(ValueError, match="non-empty"):
        Tabulated([], [])


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_tabulated_is_physical_checks_traced_tables():
    good = Tabulated([0.2, 0.4, 0.1], WAVES)
    assert bool(good.is_physical())
    assert not bool(
        good.set("ratio", np.array([0.2, -0.4, 0.1])).is_physical()
    )
    assert not bool(good.set("wavel", WAVES[::-1]).is_physical())
    with pytest.raises(ValueError, match="same length"):
        Tabulated([0.2, 0.1], WAVES)


@pytest.mark.filterwarnings("ignore:Tabulated is deprecated:FutureWarning")
def test_tabulated_flux_per_channel_matches_achromatic_scenes():
    u, v = onp.array([30.0, -20.0]), onp.array([10.0, 40.0])
    ratios = onp.array([0.05, 0.3, 0.1])
    chromatic = System(
        star=PointSource(),
        comp=PointSource(flux=Tabulated(ratios, WAVES), dra=5.0),
    )
    for wavel, ratio in zip(WAVES, ratios):
        plain = System(star=PointSource(), comp=PointSource(ratio, dra=5.0))
        assert np.allclose(
            chromatic.model(u, v, wavel), plain.model(u, v, wavel)
        )


@pytest.mark.skipif(
    jax.config.jax_enable_x64,
    reason="the overflow is specific to float32; x64 is on globally",
)
def test_blackbody_ratio_finite_when_representable_in_float32():
    # x0 - x = 90 overflows exp() in float32, but the whole ratio is
    # exp(~78.5), which is representable: it must come out finite.
    temperature, wavel0, wavel = 143.88, 1.0e-6, 10.0e-6
    got = BlackBody(1.0, temperature, wavel0)(np.float32(wavel))
    assert got.dtype == np.float32
    x, x0 = (
        1.438776877e-2 / (wavel * temperature),
        1.438776877e-2 / (wavel0 * temperature),
    )
    expected = onp.exp(
        5 * onp.log(wavel0 / wavel)
        + x0
        - x
        + onp.log(-onp.expm1(-x0))
        - onp.log(-onp.expm1(-x))
    )
    assert onp.isfinite(got)
    onp.testing.assert_allclose(float(got), expected, rtol=1e-4)
