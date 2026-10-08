"""Injection-method detection limits (Gallenne et al. 2015, section 3.2).

The data in ``tests/data/candid_injection_null.npz`` are a noise-only
observation of a 0.8 mas uniform disk with three telescopes, six hour
angles and five wavelengths: 90 V² (error 0.01) and 30 closure phases
(error 0.5 degrees), written to OIFITS by virgil-validation's simulator
(seed 8) and read back with ``OIData``. With three telescopes the closure
phases are uncorrelated, so CANDID and virgil count the same 120 data.

The pinned limits were computed with CANDID 1.1.0 (2023/07/20): its own
``_injectCompanionData`` and ``_chi2Func``, with its ``_nSigmas`` = 3
solved exactly at each position (Brent) rather than by CANDID's bracketing
and interpolation, with the disk diameter fixed at 0.8 mas.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as np
import numpy as onp
import pytest

from virgil import PointSource, System, UniformDisk
from virgil.likelihood import build_model
from virgil.limits import absil_limits, injection_limits
from virgil.oidata import OIData

DATA = Path(__file__).parent / "data" / "candid_injection_null.npz"

# (dra, ddec) in mas, and CANDID's 3-sigma injection limits there
# (companion/primary flux).
POSITIONS = [
    (-8.0, -7.0),
    (-3.0, 2.0),
    (5.0, 8.0),
    (11.0, -3.0),
    (2.5, -6.0),
    (-7.0, 7.0),
]
CANDID_INJECTION = [
    0.0032713687531664756,
    0.0027792984936711824,
    0.003640027371596345,
    0.0028913851751908542,
    0.0026747627313398826,
    0.003576278089613849,
]


@pytest.fixture(scope="module")
def data():
    arrays = dict(onp.load(DATA))
    return OIData({**arrays, "v2_flag": True, "cp_flag": True})


@pytest.fixture(scope="module")
def template():
    return System(star=UniformDisk(0.8), comp=PointSource(0.01, 0.0, 0.0))


def one_point(dra, ddec, fluxes=(0.01,)):
    return {
        "comp.dra": np.array([dra]),
        "comp.ddec": np.array([ddec]),
        "comp.flux": np.asarray(fluxes),
    }


def test_injection_limits_match_candid(data, template):
    """The limits agree with CANDID's criterion to 1e-4 (relative).

    The residual difference (about 1e-5) is float32 arithmetic here against
    CANDID's float64, magnified by the steepness of the significance near
    the root.
    """
    got = onp.array(
        [
            float(injection_limits(template, data, one_point(x, y), 3.0)[0, 0])
            for x, y in POSITIONS
        ]
    )
    assert got == pytest.approx(CANDID_INJECTION, rel=1e-4)


def test_injection_is_absil_on_data_reflected_about_the_null(data, template):
    """Injecting +signal is Absil's -signal on the reflected residuals.

    The null chi-squared of the injected data is |r + s|^2, Absil's is
    |r - s|^2, for residuals r and signal s. Reflecting the data about the
    null model (2 m0 - d, an odd map of the residuals, which also holds for
    the sine-based phase residuals) turns one into the other. Neither
    method is more sensitive in general: the cross term r.s has either
    sign, and averages to zero over noise realizations.
    """
    null = build_model(template, ("comp.flux",), [0.0])
    n_vis = data.vis.shape[0]
    m0 = data.model(null)
    reflected = eqx.tree_at(
        lambda d: (d.vis, d.phi),
        data,
        (2 * m0[:n_vis] - data.vis, 2 * m0[n_vis:] - data.phi),
    )
    fluxes = onp.logspace(-4, -1, 8)
    for x, y in POSITIONS[:3]:
        samples = one_point(x, y, fluxes)
        injection = injection_limits(template, data, samples, 3.0)
        absil = absil_limits(template, reflected, samples, 3.0)
        assert float(injection[0, 0]) == pytest.approx(
            float(absil[0, 0]), rel=2e-3
        )


def test_injection_limit_depends_on_the_chi2_ratio_only(data, template):
    """Scaling every error bar leaves the limit unchanged.

    The criterion is a ratio of chi-squared values, so only the data's
    scatter relative to the null model matters, not the quoted errors.
    """
    scaled = eqx.tree_at(
        lambda d: (d.d_vis, d.d_phi),
        data,
        (2 * data.d_vis, 2 * data.d_phi),
    )
    samples = one_point(5.0, 8.0)
    nominal = float(injection_limits(template, data, samples, 3.0)[0, 0])
    other = float(injection_limits(template, scaled, samples, 3.0)[0, 0])
    assert other == pytest.approx(nominal, rel=1e-3)


def test_injection_limits_grid_shape_and_bounds(data, template):
    samples = {
        "comp.dra": np.array([-6.0, 0.0, 6.0]),
        "comp.ddec": np.array([-4.0, 4.0]),
        "comp.flux": np.array([0.01]),
    }
    limits = injection_limits(template, data, samples, 3.0)
    assert limits.shape == (3, 2)
    assert onp.all(onp.isfinite(limits))
    assert onp.all((limits > 1e-6) & (limits < 1.0))

    # A bracket that cannot contain the limit clips, and warns.
    with pytest.warns(RuntimeWarning, match="clipped"):
        clipped = injection_limits(
            template, data, samples, 3.0, flux_bounds=(0.5, 1.0)
        )
    assert onp.allclose(clipped, 0.5)


@pytest.mark.parametrize(
    "bounds",
    [
        (1e-6, float("inf")),
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (1.0, 1e-6),
    ],
)
def test_injection_limits_rejects_invalid_flux_bounds(data, template, bounds):
    with pytest.raises(ValueError, match="flux_bounds"):
        injection_limits(
            template, data, one_point(5.0, 8.0), 3.0, flux_bounds=bounds
        )


def test_injection_limits_rejects_unreachable_sigma(data, template):
    with pytest.raises(ValueError, match="cannot be reached"):
        injection_limits(template, data, one_point(5.0, 8.0), 0.1)
