"""ArchimedeanSpiral against references that do not use its own code path.

The references are written out here in NumPy float64: a dense quadrature along
the arm (position, arc-length element, fade and the sky Gaussian from first
principles), the Fourier transforms of a uniform disk and of a uniform
segment, and the mirror, rotation and foreshortening of the geometry.
"""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from scipy.optimize import brentq
from scipy.special import j1

from virgil._geometry import image_coordinates
from virgil.fitting import fit
from virgil.models import (
    ArchimedeanSpiral,
    PointSource,
    System,
)
from virgil.inference import laplace_cov
from virgil.coverage import vlti_oidata
from virgil.spectra import PowerLaw

# Independent of virgil's own constant.
MAS2RAD = onp.pi / 180.0 / 3600.0 / 1000.0
WAVEL = 1.0e-6  # u, v below are in metres, so baselines are u / 1e-6 waves
SHAPES = {
    "face_on": dict(pitch=30.0, turns=1.5, width=3.0, pa=40.0),
    "inclined_clockwise_faded": dict(
        pitch=-30.0,
        turns=2.2,
        width=3.0,
        pa=40.0,
        inc=55.0,
        node=30.0,
        fade=0.7,
    ),
}


def _reference_visibility(
    u, v, pitch, turns, width, pa=0.0, inc=0.0, node=0.0, fade=0.0
):
    """Quadrature along the arm, 200001 points, with its own geometry."""
    theta = onp.linspace(0.0, 2.0 * onp.pi * turns, 200001)
    radius = abs(pitch) / (2.0 * onp.pi) * theta
    cos_i = onp.cos(onp.radians(inc))
    # The in-plane azimuth (from the major axis) whose projection has sky
    # position angle pa, found by root finding on the projected direction.
    target = onp.radians(pa - node)
    start = brentq(
        lambda a: onp.angle(
            onp.exp(
                1j * (onp.arctan2(onp.sin(a) * cos_i, onp.cos(a)) - target)
            )
        ),
        target - 1.4,
        target + 1.4,
    )
    azimuth = start + onp.sign(pitch) * theta
    xt, yt = radius * onp.sin(azimuth), radius * onp.cos(azimuth)
    arc = onp.hypot(onp.gradient(xt, theta), onp.gradient(yt, theta))
    weight = onp.exp(-fade * theta / (2.0 * onp.pi)) * arc
    weight[[0, -1]] *= 0.5
    weight /= weight.sum()
    n = onp.radians(node)
    x = xt * cos_i * onp.cos(n) + yt * onp.sin(n)
    y = -xt * cos_i * onp.sin(n) + yt * onp.cos(n)
    qx, qy = u / WAVEL * MAS2RAD, v / WAVEL * MAS2RAD
    arm = (
        onp.exp(-2j * onp.pi * (onp.outer(qx, x) + onp.outer(qy, y))) @ weight
    )
    # Sky covariance of a Gaussian that is round in the arm plane.
    a = onp.array([[onp.cos(n), onp.sin(n)], [-onp.sin(n), onp.cos(n)]]) @ (
        onp.diag([cos_i, 1.0])
    )
    sigma = width / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))
    q = onp.stack([qx, qy], 1)
    quad = onp.einsum("ni,ij,nj->n", q, sigma**2 * a @ a.T, q)
    return arm * onp.exp(-2.0 * onp.pi**2 * quad)


def _baselines(n=40, span=100.0, seed=0):
    rng = onp.random.default_rng(seed)
    return rng.uniform(-span, span, (2, n))


@pytest.mark.parametrize("shape", SHAPES)
def test_visibility_matches_quadrature_along_the_arm(shape):
    kwargs = SHAPES[shape]
    u, v = _baselines()  # up to 1.9 widths per cycle, longer than the docs ask
    with jax.enable_x64(True):
        spiral = ArchimedeanSpiral(**kwargs)
        got = onp.asarray(spiral.model(u, v, WAVEL))
    expected = _reference_visibility(u, v, **kwargs)
    # The chain is the trapezoidal rule in the arm angle; its error is the
    # Euler-Maclaurin end term (h^2/12) |g'| at each end, with g the unit-mass
    # weight density along the arc, |g'| <= 2 pi q / L, and h N = L, so
    # |dV| <= 2 pi q h / (6 N) with h the largest node spacing (at the end
    # of the arm, where ds/dtheta = a sqrt(1 + theta^2)).
    q_max = onp.hypot(u, v).max() / WAVEL * MAS2RAD
    big_theta = 2.0 * onp.pi * kwargs["turns"]
    h = (
        abs(kwargs["pitch"])
        / (2.0 * onp.pi)
        * onp.hypot(1.0, big_theta)
        * big_theta
        / (spiral.nodes - 1)
    )
    bound = 2.0 * onp.pi * q_max * h / (6.0 * spiral.nodes)
    assert onp.max(onp.abs(got - expected)) < bound


def test_visibility_agrees_with_the_rendered_image():
    # Image, DFT on a fine grid: 0.25 mas pixels sample the 3 mas FWHM blobs
    # 12 times, so pixel aliasing is below exp(-3.56 (12 - 1.9)^2).
    npix, fov = 640, 160.0
    kwargs = SHAPES["inclined_clockwise_faded"]
    u, v = _baselines(n=12)
    with jax.enable_x64(True):
        spiral = ArchimedeanSpiral(**kwargs)
        image = onp.asarray(spiral.render(npix=npix, fov_mas=fov)).ravel()
        got = onp.asarray(spiral.model(u, v, WAVEL))
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(npix, fov))
    phase = onp.exp(
        -2j * onp.pi * MAS2RAD * (onp.outer(u, xx) + onp.outer(v, yy)) / WAVEL
    )
    dft = phase @ image / image.sum()
    # Same chain on both sides, so only the pixel sampling and the finite
    # field of view separate them; both are far below float64 precision.
    assert onp.max(onp.abs(dft - got)) < 1e-9


def test_flux_is_one_and_the_zero_baseline_is_one():
    spiral = ArchimedeanSpiral(**SHAPES["inclined_clockwise_faded"])
    assert onp.isclose(
        float(spiral.render(npix=128, fov_mas=160.0).sum()), 1.0
    )
    assert onp.isclose(complex(spiral.model(0.0, 0.0, 1.6e-6)), 1.0)
    # In a System, with the star at the centre, V = (1 + f V_arm) / (1 + f).
    scene = System(star=PointSource(), arm=spiral.set("flux", 0.5))
    u, v = _baselines(n=5, span=20.0)
    expected = (1.0 + 0.5 * onp.asarray(spiral.model(u, v, WAVEL))) / 1.5
    assert onp.allclose(
        onp.asarray(scene.model(u, v, WAVEL)), expected, atol=1e-6
    )
    assert onp.isclose(complex(scene.model(0.0, 0.0, 1.6e-6)), 1.0)


def test_chromatic_flux_weights_the_component_like_any_other():
    spiral = ArchimedeanSpiral(**SHAPES["face_on"])
    scene = System(
        star=PointSource(),
        arm=spiral.set("flux", PowerLaw(0.4, index=2.0, wavel0=1.0e-6)),
    )
    u, v = _baselines(n=5, span=20.0)
    for wavel in (1.0e-6, 2.0e-6):
        flux = 0.4 * (wavel / 1.0e-6) ** 2.0
        expected = (1.0 + flux * onp.asarray(spiral.model(u, v, wavel))) / (
            1.0 + flux
        )
        got = onp.asarray(scene.model(u, v, wavel))
        assert onp.allclose(got, expected, atol=1e-6)


def test_tight_spiral_is_a_uniform_disk():
    # With the pitch far below the width and many turns, the arm (length
    # 2 pi r per pitch) tiles the plane at a uniform density: a uniform disk
    # of radius R = pitch * turns (equal area), blurred by the Gaussian. Its
    # last turn leaves a ragged edge, a shell of thickness at most pitch / 2
    # about the circle, holding at most perimeter * pitch / 2 / area =
    # pitch / R of the flux, and each unit of misplaced flux moves V by at
    # most 1 (the centre adds only O((pitch / R)^2)).
    pitch, turns, width = 0.4, 50.0, 3.0
    radius = pitch * turns
    u, v = _baselines(n=30, span=40.0)
    q = onp.hypot(u, v) / WAVEL * MAS2RAD

    def disk(radius):
        x = 2.0 * onp.pi * q * radius
        blur = onp.exp(-(onp.pi**2) * width**2 * q**2 / (4.0 * onp.log(2.0)))
        return 2.0 * j1(x) / x * blur

    with jax.enable_x64(True):
        got = onp.asarray(
            ArchimedeanSpiral(pitch, turns, width).model(u, v, WAVEL)
        )
    bound = pitch / radius
    assert onp.max(onp.abs(got - disk(radius))) < bound
    # The bound is not vacuous: a disk 20 percent larger is outside it.
    assert onp.max(onp.abs(got - disk(1.2 * radius))) > bound


def test_very_short_arm_is_a_straight_segment():
    # For turns -> 0 the arm is a radial segment of length L = pitch * turns
    # leaving the centre at pa, Gaussian blurred, whose transform is
    # exp(-i pi q.L) sinc(pi q.L). It departs from the line by at most
    # a theta^2 (a = pitch / 2 pi, theta the end angle), which moves the phase
    # of a baseline q by at most 2 pi q a theta^2.
    pitch, turns, width, pa = 15000.0, 0.001, 3.0, 35.0
    length = pitch * turns
    theta = 2.0 * onp.pi * turns
    deviation = pitch / (2.0 * onp.pi) * theta**2
    u, v = _baselines(n=30, span=8.0)
    qx, qy = u / WAVEL * MAS2RAD, v / WAVEL * MAS2RAD
    along = qx * onp.sin(onp.radians(pa)) + qy * onp.cos(onp.radians(pa))
    segment = onp.exp(-1j * onp.pi * along * length) * onp.sinc(along * length)
    blur = onp.exp(
        -(onp.pi**2) * width**2 * (qx**2 + qy**2) / (4.0 * onp.log(2.0))
    )
    with jax.enable_x64(True):
        got = onp.asarray(
            ArchimedeanSpiral(pitch, turns, width, pa=pa).model(u, v, WAVEL)
        )
    bound = 2.0 * onp.pi * onp.hypot(qx, qy).max() * deviation
    assert onp.max(onp.abs(got - segment * blur)) < bound
    # and the bound is not vacuous: the wrong direction is far outside it
    wrong = onp.exp(-1j * onp.pi * -along * length) * onp.sinc(along * length)
    assert onp.max(onp.abs(got - wrong * blur)) > 5.0 * bound


def _reflect(u, v, pa):
    """Baselines mirrored in the line at position angle ``pa``."""
    n = onp.array([onp.sin(onp.radians(pa)), onp.cos(onp.radians(pa))])
    q = onp.stack([u, v], 1)
    mirrored = 2.0 * (q @ n)[:, None] * n - q
    return mirrored[:, 0], mirrored[:, 1]


def test_flipping_the_winding_mirrors_the_image():
    # Mirror in the line through the centre at the start of the arm.
    u, v = _baselines(n=20, span=30.0)
    ur, vr = _reflect(u, v, 40.0)
    ccw = ArchimedeanSpiral(30.0, 1.5, 3.0, pa=40.0, fade=0.3)
    cw = ArchimedeanSpiral(-30.0, 1.5, 3.0, pa=40.0, fade=0.3)
    assert onp.allclose(
        onp.asarray(cw.model(u, v, WAVEL)),
        onp.asarray(ccw.model(ur, vr, WAVEL)),
        atol=2e-6,
    )
    # And the sense is as documented: a quarter turn of a positive pitch from
    # a start due North ends due East, at radius pitch / 4; a negative one
    # ends due West.
    for pitch, east in ((30.0, 7.5), (-30.0, -7.5)):
        arm = ArchimedeanSpiral(pitch, 0.25, 3.0, pa=0.0, nodes=50)
        x, y, _ = (onp.asarray(a) for a in arm.curve())
        assert onp.allclose([x[-1], y[-1]], [east, 0.0], atol=1e-4)


def test_render_puts_east_on_the_left_and_north_up():
    arm = ArchimedeanSpiral(30.0, 0.25, 3.0, pa=0.0)
    image = onp.asarray(arm.render(npix=128, fov_mas=60.0))
    rows, cols = onp.indices(image.shape)
    row, col = (image * rows).sum(), (image * cols).sum()
    # The arm runs from the centre to the East and a little North: left of
    # and above the centre pixel (row 0 is North, column 0 East).
    assert col < 63.5 and row < 63.5


@pytest.mark.parametrize("inc", [0.0, 50.0])
def test_rotating_pa_and_node_rotates_the_image(inc):
    phi = 70.0
    u, v = _baselines(n=20, span=30.0)
    c, s = onp.cos(onp.radians(phi)), onp.sin(onp.radians(phi))
    # Turning the sky by phi (position angles increase) maps a baseline the
    # same way: (x, y) -> (x c + y s, -x s + y c).
    ur, vr = c * u + s * v, -s * u + c * v
    base = ArchimedeanSpiral(30.0, 1.5, 3.0, pa=20.0, inc=inc, node=10.0)
    turned = ArchimedeanSpiral(
        30.0, 1.5, 3.0, pa=20.0 + phi, inc=inc, node=10.0 + phi
    )
    assert onp.allclose(
        onp.asarray(turned.model(ur, vr, WAVEL)),
        onp.asarray(base.model(u, v, WAVEL)),
        atol=2e-6,
    )


@pytest.mark.parametrize("pa", [0.0, 90.0])
def test_inclination_foreshortens_across_the_major_axis(pa):
    # With the node due North the East-West axis is compressed by cos(inc);
    # a start along either axis is the same in the plane and on the sky. A
    # compressed image f(x / c, y) has transform c F(c q_x, q_y).
    inc = 60.0
    u, v = _baselines(n=20, span=30.0)
    tilted = ArchimedeanSpiral(30.0, 1.5, 3.0, pa=pa, inc=inc, node=0.0)
    flat = ArchimedeanSpiral(30.0, 1.5, 3.0, pa=pa)
    assert onp.allclose(
        onp.asarray(tilted.model(u, v, WAVEL)),
        onp.asarray(flat.model(onp.cos(onp.radians(inc)) * u, v, WAVEL)),
        atol=2e-6,
    )


@pytest.mark.parametrize("pitch", [30.0, -30.0])
@pytest.mark.parametrize(
    ("pa", "inc", "node"),
    [(0.0, 0.0, 0.0), (130.0, 65.0, 20.0), (250.0, 40.0, 100.0)],
)
def test_arm_leaves_the_centre_at_pa_on_the_sky(pitch, pa, inc, node):
    arm = ArchimedeanSpiral(
        pitch, 1e-4, 3.0, pa=pa, inc=inc, node=node, nodes=3
    )
    x, y, _ = (onp.asarray(a) for a in arm.curve())
    assert onp.isclose(x[0], 0.0) and onp.isclose(y[0], 0.0)
    offset = onp.degrees(onp.arctan2(x[1], y[1])) - pa
    assert abs((offset + 180.0) % 360.0 - 180.0) < 0.1


PATHS = ["arm.pitch", "arm.turns", "arm.width", "arm.pa", "arm.inc"] + [
    "arm.node",
    "arm.fade",
    "arm.flux",
    "arm.dra",
    "arm.ddec",
]


def _scene():
    arm = ArchimedeanSpiral(
        30.0, 1.5, 3.0, pa=40.0, inc=35.0, node=20.0, fade=0.4, flux=0.6
    )
    return System(star=PointSource(), arm=arm)


def test_gradients_are_finite_for_every_parameter():
    scene = _scene()
    u, v = np.array([0.0, 2.0e7, -5.0e7]), np.array([0.0, 3.0e7, 1.0e7])

    def loss(values):
        vis = scene.set(PATHS, values).model(u, v, 1.0)
        return np.sum(vis.real * 0.3 + vis.imag * 0.7)

    grads = jax.grad(loss)([scene.get(p) for p in PATHS])
    assert all(onp.isfinite(float(g)) for g in grads)
    # each parameter matters to this complex-valued loss
    assert all(float(g) != 0.0 for g in grads)


def test_gradients_match_finite_differences():
    scene = _scene()
    u, v = np.array([1.0e7, 2.0e7, -5.0e7]), np.array([4.0e6, 3.0e7, 1.0e7])
    with jax.enable_x64(True):

        def loss(values):
            vis = scene.set(PATHS, values).model(u, v, 1.0)
            return np.sum(vis.real * 0.3 + vis.imag * 0.7)

        x0 = [np.asarray(scene.get(p), dtype=float) for p in PATHS]
        grads = jax.grad(loss)(x0)
        for i, g in enumerate(grads):
            step = 1e-5 * max(1.0, abs(float(x0[i])))
            up = [x + step * (j == i) for j, x in enumerate(x0)]
            down = [x - step * (j == i) for j, x in enumerate(x0)]
            fd = (loss(up) - loss(down)) / (2.0 * step)
            assert onp.isclose(float(g), float(fd), rtol=1e-4, atol=1e-9), (
                PATHS[i]
            )


@pytest.mark.slow
def test_fit_recovers_pitch_turns_and_pa_within_their_uncertainty():
    # Simulated V^2 and closure phases of a star plus an arm on the four-UT
    # baselines, fitted from a perturbed start. The truth must lie within
    # 3 sigma of the fit, sigma from the Laplace covariance at the truth.
    paths = ["arm.pitch", "arm.turns", "arm.pa"]
    truth_values = [8.0, 1.5, 60.0]
    truth = System(
        star=PointSource(),
        arm=ArchimedeanSpiral(
            *truth_values[:2], 2.0, pa=60.0, fade=0.3, flux=0.5
        ),
    )
    data = vlti_oidata(wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 3))
    sim = data.with_model(truth, key=jax.random.PRNGKey(2))
    start = truth.set(paths, [7.0, 1.35, 80.0])
    priors = {
        "arm.pitch": dist.Uniform(2.0, 20.0),
        "arm.turns": dist.Uniform(0.5, 3.0),
        "arm.pa": dist.Uniform(0.0, 180.0),
    }
    result = fit(start, priors, sim, method="lm")
    sigma = onp.sqrt(
        onp.diag(
            onp.asarray(
                laplace_cov(onp.array(truth_values), paths, truth, sim)
            )
        )
    )
    found = onp.array([float(result.values[p]) for p in paths])
    assert onp.all(onp.abs(found - truth_values) < 3.0 * sigma)
    # and it moved: the start was several sigma off in each
    assert onp.all(
        onp.abs(onp.array([7.0, 1.35, 80.0]) - truth_values) > 3.0 * sigma
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pitch", 0.0),
        ("turns", 0.0),
        ("turns", -1.0),
        ("width", 0.0),
        ("fade", -0.1),
        ("inc", 90.0),
    ],
)
def test_invalid_shapes_are_rejected(name, value):
    kwargs = {"pitch": 30.0, "turns": 1.5, "width": 3.0} | {name: value}
    with pytest.raises(ValueError, match=name):
        ArchimedeanSpiral(**kwargs)


def test_nodes_are_validated_and_needed_when_traced():
    with pytest.raises(ValueError, match="nodes"):
        ArchimedeanSpiral(30.0, 1.5, 3.0, nodes=1)
    good = ArchimedeanSpiral(30.0, 1.5, 3.0)
    assert bool(good.is_physical())
    assert not bool(good.set("pitch", np.array(0.0)).is_physical())
    with pytest.raises(ValueError, match="nodes"):
        jax.jit(
            lambda p: ArchimedeanSpiral(p, 1.5, 3.0).model(1e6, 0.0, 1e-6)
        )(30.0)
    traced = jax.jit(
        lambda p: ArchimedeanSpiral(p, 1.5, 3.0, nodes=500).model(
            1e6, 0.0, 1e-6
        )
    )
    assert onp.isfinite(complex(traced(30.0)))
