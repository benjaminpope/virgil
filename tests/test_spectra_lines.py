"""Stage 6a spectra: lines, node spectra, sums, totals and wavelength cuts."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import (
    GaussianDisk,
    GaussianLine,
    LorentzianLine,
    Nodes,
    OIData,
    PointSource,
    PowerLaw,
    Sum,
    System,
)
from virgil.likelihood import model_loglike
from virgil.oifits import build_hdulist, read_oifits
from virgil.spectra import Tabulated
from tests.test_oifits import TRUTH, _tables

BRG = 2.1661e-6
FWHM = 1.0e-9


@pytest.mark.parametrize(
    "line, integral",
    [
        (
            GaussianLine(0.4, BRG, FWHM),
            0.4 * FWHM * onp.sqrt(onp.pi / (4.0 * onp.log(2.0))),
        ),
        # The Lorentzian's slow wings need a wide grid to converge.
        (LorentzianLine(0.4, BRG, FWHM), 0.4 * onp.pi * FWHM / 2.0),
    ],
)
def test_a_line_has_its_flux_integral_and_centroid(line, integral):
    with jax.enable_x64(True):
        wavel = onp.linspace(BRG - 4000 * FWHM, BRG + 4000 * FWHM, 400001)
        flux = onp.asarray(line(np.asarray(wavel)))
        total = onp.trapezoid(flux, wavel)
        centroid = onp.trapezoid(flux * wavel, wavel) / total
    assert total == pytest.approx(integral, rel=2e-3)
    assert centroid == pytest.approx(BRG, abs=1e-3 * FWHM)
    # Half maximum at half the FWHM, and the reference flux at the centre.
    assert float(line(BRG + FWHM / 2)) == pytest.approx(0.2, rel=1e-4)
    assert float(line()) == pytest.approx(0.4, rel=1e-6)


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_node_spectra_interpolate_their_nodes(kind):
    nodes = onp.array([2.0e-6, 2.1e-6, 2.15e-6, 2.3e-6])
    values = onp.array([0.2, 0.5, 0.3, 0.4])
    spectrum = Nodes(values, nodes, kind=kind)
    assert onp.allclose(spectrum(nodes), values, rtol=1e-5)
    # Constant beyond the ends by default; the first node is the reference.
    assert onp.allclose(spectrum(onp.array([1.9e-6, 2.4e-6])), [0.2, 0.4])
    assert float(spectrum()) == pytest.approx(0.2)


def test_cubic_nodes_are_a_natural_spline():
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    nodes = onp.array([2.0, 2.1, 2.15, 2.3, 2.4]) * 1e-6
    values = onp.array([0.2, 0.5, 0.3, 0.4, 0.1])
    wavel = onp.linspace(2.0e-6, 2.4e-6, 97)
    reference = scipy_interpolate.CubicSpline(nodes, values, bc_type="natural")
    ours = Nodes(values, nodes, kind="cubic")(wavel)
    assert onp.allclose(ours, reference(wavel), atol=2e-5)


def test_an_excess_outside_its_window_is_the_outside_value():
    window = onp.array([2.160e-6, 2.166e-6, 2.172e-6])
    excess = Nodes([0.0, 0.5, 0.0], window, outside=0.0, wavel0=2.166e-6)
    assert onp.allclose(excess(onp.array([2.0e-6, 2.3e-6])), 0.0)
    assert float(excess(2.163e-6)) == pytest.approx(0.25, rel=1e-4)
    assert float(excess()) == pytest.approx(0.5)


def test_a_sum_allows_absorption_but_not_a_negative_total():
    flux = Sum(
        continuum=PowerLaw(1.0, index=-2.0, wavel0=2.2e-6),
        brg=GaussianLine(-0.3, BRG, FWHM),
    )
    continuum = 1.0 * (BRG / 2.2e-6) ** -2.0
    assert float(flux(BRG)) == pytest.approx(continuum - 0.3, rel=1e-5)
    assert float(flux()) == pytest.approx(1.0)  # at the Sum's wavel0
    assert bool(flux.is_physical())
    star = PointSource(flux=flux)  # accepted: the total is positive
    assert star.flux.brg.amplitude == pytest.approx(-0.3)

    too_deep = Sum(
        continuum=PowerLaw(1.0, wavel0=2.2e-6),
        brg=GaussianLine(-1.5, BRG, FWHM),
    )
    assert not bool(too_deep.is_physical())
    with pytest.raises(ValueError, match="non-negative"):
        PointSource(flux=too_deep)
    with pytest.raises(ValueError, match="non-negative"):
        PointSource(flux=GaussianLine(-0.3, BRG, FWHM))


def test_is_physical_sees_a_negative_total_under_jit():
    star = PointSource(
        flux=Sum(
            continuum=PowerLaw(1.0, wavel0=2.2e-6),
            brg=GaussianLine(-0.3, BRG, FWHM),
        )
    )

    @jax.jit
    def valid(depth):
        return star.set("flux.brg.amplitude", depth).is_physical()

    assert bool(valid(-0.5))
    assert not bool(valid(-1.5))


def test_a_line_in_a_component_changes_the_visibility_only_near_the_line():
    def scene(amplitude):
        return System(
            star=PointSource(flux=PowerLaw(1.0, wavel0=2.2e-6)),
            wind=GaussianDisk(
                3.0,
                flux=Sum(
                    continuum=PowerLaw(0.2, wavel0=2.2e-6),
                    brg=GaussianLine(amplitude, BRG, FWHM),
                ),
            ),
        )

    u, v = onp.array([60.0, 100.0]), onp.array([20.0, -40.0])
    far, near = 2.10e-6, BRG
    flat, lined = scene(0.0), scene(1.0)
    assert np.allclose(flat.model(u, v, far), lined.model(u, v, far))
    assert not np.allclose(flat.model(u, v, near), lined.model(u, v, near))

    def v2(amplitude):
        cvis = scene(amplitude).model(u, v, near)
        return np.sum(np.abs(cvis) ** 2)

    grad = float(jax.grad(v2)(1.0))
    assert onp.isfinite(grad) and grad != 0.0


def test_gradients_flow_through_cubic_nodes():
    nodes = onp.array([2.0e-6, 2.1e-6, 2.2e-6, 2.3e-6])

    def flux(values):
        return np.sum(Nodes(values, nodes, kind="cubic")(2.17e-6))

    grad = onp.asarray(jax.grad(flux)(np.array([0.2, 0.5, 0.3, 0.4])))
    assert onp.isfinite(grad).all()
    assert grad.sum() == pytest.approx(1.0, rel=1e-5)  # partition of unity


def test_total_spectrum_sums_the_components():
    scene = System(
        star=PointSource(flux=PowerLaw(1.0, index=-4.0, wavel0=2.2e-6)),
        wind=GaussianDisk(
            3.0,
            flux=Sum(
                continuum=PowerLaw(0.2, wavel0=2.2e-6),
                brg=GaussianLine(0.5, BRG, FWHM),
            ),
        ),
        comp=System(core=PointSource(), flux=0.1, dra=20.0),
    )
    wavel = onp.array([2.1e-6, BRG, 2.3e-6])
    expected = (
        (wavel / 2.2e-6) ** -4.0
        + 0.2
        + 0.5 * onp.exp(-4.0 * onp.log(2.0) * ((wavel - BRG) / FWHM) ** 2)
        + 0.1
    )
    assert onp.allclose(scene.total_spectrum(wavel), expected, rtol=1e-5)
    single = PointSource(flux=PowerLaw(0.3, index=1.0, wavel0=2.2e-6))
    assert onp.allclose(
        single.total_spectrum(wavel), 0.3 * wavel / 2.2e-6, rtol=1e-6
    )


def test_tabulated_is_deprecated_but_unchanged():
    nodes = onp.array([2.0e-6, 2.1e-6, 2.2e-6])
    with pytest.warns(DeprecationWarning, match="Nodes"):
        old = Tabulated([0.2, 0.4, 0.1], nodes)
    new = Nodes([0.2, 0.4, 0.1], nodes)
    wavel = onp.linspace(1.9e-6, 2.3e-6, 11)
    assert onp.allclose(old(wavel), new(wavel))
    assert float(old()) == pytest.approx(onp.mean([0.2, 0.4, 0.1]))


def test_invalid_spectra_are_rejected():
    nodes = onp.array([2.0e-6, 2.1e-6, 2.2e-6])
    with pytest.raises(ValueError, match="increasing"):
        Nodes([0.1, 0.2, 0.3], nodes[::-1])
    with pytest.raises(ValueError, match="3 nodes"):
        Nodes([0.1, 0.2], nodes[:2], kind="cubic")
    with pytest.raises(ValueError, match="kind"):
        Nodes([0.1, 0.2, 0.3], nodes, kind="quadratic")
    with pytest.raises(ValueError, match="fwhm"):
        GaussianLine(0.1, BRG, -1e-9)
    with pytest.raises(TypeError, match="Spectrum"):
        Sum(continuum=0.5)


def test_select_keeps_whole_triangles_in_a_wavelength_range():
    waves = (2.0e-6, 2.1e-6, 2.2e-6, 2.3e-6)
    data = OIData(read_oifits(build_hdulist(_tables(waves=waves))))
    middle = data.select(2.05e-6, 2.25e-6)
    assert onp.allclose(
        onp.unique(onp.asarray(middle.wavel)), [2.1e-6, 2.2e-6], rtol=1e-6
    )
    assert middle.n_independent == pytest.approx(data.n_independent / 2)
    upper = data.select(wavel_min=2.15e-6)
    lower = data.select(wavel_max=2.15e-6)
    noisy = data.with_model(TRUTH, jax.random.PRNGKey(0))
    whole = model_loglike(TRUTH, noisy)
    parts = model_loglike(TRUTH, noisy.select(wavel_min=2.15e-6)) + (
        model_loglike(TRUTH, noisy.select(wavel_max=2.15e-6))
    )
    assert parts == pytest.approx(whole, rel=1e-5)
    assert upper.n_independent + lower.n_independent == data.n_independent
    with pytest.raises(ValueError, match="No samples"):
        data.select(3.0e-6)
