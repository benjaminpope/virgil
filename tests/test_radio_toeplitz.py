"""virgil.radio_toeplitz: the dirty-beam χ² against the direct radio term."""

import math
import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import (
    CalibrationNuisance,
    ComplexVisData,
    Image,
    PointSource,
    System,
    fit,
)
from virgil import _nufft_spread as spread_grid
from virgil._geometry import set_ft_defaults
from virgil._utils import mas2rad
from ._compiles import count_compiles
from virgil.fitting import (
    NoLeastSquaresForm,
    gauss_newton_diag,
    gauss_newton_mass,
)


@pytest.fixture(autouse=True)
def float64():
    """Tight tolerances need float64; opt in locally, as AGENTS.md asks."""
    with jax.enable_x64(True):
        yield


PIXEL = 2.0  # mas
DDID_OBS = (0, 0, 1, 1)  # 2 observations x 2 ddids
CENTRES = onp.array([230e9, 232e9, 240e9, 242e9])


def _uv(n, reach, rng):
    """Points in a disc reaching ``reach`` of the pixel Nyquist limit."""
    q = reach / (2.0 * mas2rad * PIXEL) * onp.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * onp.pi, n)
    return q * onp.cos(phi), q * onp.sin(phi)


def _data(n, npix, seed=0, reach=0.45, alpha=0.7):
    """Visibilities of a random image with gains, an offset and noise;
    channels spread ±1% around each ddid's centre."""
    rng = onp.random.default_rng(seed)
    ddid = rng.integers(0, len(DDID_OBS), n)
    obs = onp.asarray(DDID_OBS)[ddid]
    freq = CENTRES[ddid] * (1.0 + rng.uniform(-0.01, 0.01, n))
    u, v = _uv(n, reach, rng)
    weight = rng.uniform(0.5, 2.0, n) * 1e2
    blank = ComplexVisData.from_arrays(
        u, v, onp.zeros(n, complex), weight, freq, ddid=ddid + 5, obs=obs + 2
    )
    nuis = CalibrationNuisance(blank, alpha=alpha)
    truth = Image(rng.normal(size=(npix, npix)), PIXEL, ft="dft")
    values = _values(nuis, flux=0.9, gain=[1.15], offset=[[0.4, -0.7]])
    model = onp.asarray(blank.model(truth, values, nuis))
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    return ComplexVisData.from_arrays(
        u,
        v,
        model + noise / onp.sqrt(weight),
        weight,
        freq,
        ddid=ddid + 5,
        obs=obs + 2,
    )


def _values(nuis, **sites):
    values = {k: np.asarray(x) for k, x in nuis.defaults().items()}
    for name, x in sites.items():
        if nuis._site(name) in values:
            values[nuis._site(name)] = np.asarray(x, dtype=float)
    return values


def _scene(npix, rotation=0.0, **kw):
    def scene(values):
        return Image(
            values["eta"], PIXEL, rotation_deg=rotation, ft="dft", **kw
        )

    template = Image(
        np.zeros((npix, npix)), PIXEL, rotation_deg=rotation, **kw
    )
    return scene, template


def _losses(tdata, scene, nuis, **term_kw):
    """Toeplitz loss (less its constant) and direct loss, as functions of
    the values."""
    term = tdata.term(scene, nuis, **term_kw)
    direct = tdata.direct_term(scene, nuis, **term_kw)

    def toeplitz(values):
        chi2, log_norm = term.chi2_and_log_norm(values)
        return 0.5 * chi2 + log_norm

    def reference(values):
        r, log_norm = direct.residuals_and_log_norm(values)
        return 0.5 * np.sum(r**2) + log_norm

    return term, toeplitz, reference


def _loss_difference(data, mask, scene, nuis, v1, v2, sigma=None):
    """The direct loss difference L(v2) - L(v1), compensated: per-point
    differences of |r|², summed with ``math.fsum``, at a σ common to both."""
    w = onp.asarray(data.weight)
    vis = onp.asarray(data.vis)
    m1 = onp.asarray(data.model(scene(v1), v1, nuis))
    m2 = onp.asarray(data.model(scene(v2), v2, nuis))
    r1 = (vis - m1) * onp.sqrt(w)
    diff = onp.real(
        (m1 - m2) * onp.sqrt(w) * onp.conj(2.0 * r1 + (m1 - m2) * onp.sqrt(w))
    )
    ddid = onp.asarray(data.ddid)
    out = 0.0
    for d in range(len(DDID_OBS)):
        sel = mask & (ddid == d)
        if not sel.any():
            continue
        d_chi2 = math.fsum(diff[sel])
        if nuis.sigmas == "profile" and sigma is None:
            chi2_1 = math.fsum(onp.abs(r1[sel]) ** 2)
            out += sel.sum() * math.log1p(d_chi2 / chi2_1)
        else:
            s = 1.0 if sigma is None else float(sigma[d])
            out += 0.5 * d_chi2 / s**2
    return out


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _check_against_direct(data, npix, sigmas, offsets, rotation, seed, **kw):
    """Loss, loss difference and gradient (every site) of the Toeplitz term
    against the direct term; returns the largest relative differences."""
    rng = onp.random.default_rng(seed)
    mask = rng.uniform(size=len(data)) > 0.1
    nuis = CalibrationNuisance(
        data, sigmas=sigmas, offsets=offsets, alpha=0.7, blur_fwhm_mas=3.0
    )
    scene, template = _scene(npix, rotation)
    tdata = data.toeplitz(
        template,
        nuis,
        mask=mask,
        offsets="spread" if offsets else "none",
        **kw,
    )
    term, toeplitz, reference = _losses(tdata, scene, nuis)
    # the bound from the prior, not from the precompute under test
    bound = 0.0
    if offsets:
        bound = float(np.max(nuis.priors()["cal.offset"].high))
        angle = onp.deg2rad(rotation)
        spread = abs(onp.cos(angle)) + abs(onp.sin(angle))
        assert tdata.max_shift_pix * PIXEL == pytest.approx(
            bound * spread, rel=1e-12
        )
    v1 = _values(
        nuis,
        flux=0.8,
        gain=[1.1],
        # one sub-pixel offset, one at the bound of the precompute
        offset=[[0.37, -0.999 * bound]],
        sigma=[1.2, 0.9, 1.05, 1.3],
    )
    v1["eta"] = np.asarray(rng.normal(size=(npix, npix)))
    v2 = dict(v1, eta=v1["eta"] + 0.05 * rng.normal(size=(npix, npix)))

    loss, grad = jax.value_and_grad(toeplitz)(v1)
    loss_d, grad_d = jax.value_and_grad(reference)(v1)
    sigma = v1["cal.sigma"] if sigmas is True else None
    exact = _loss_difference(data, mask, scene, nuis, v1, v2, sigma)
    return dict(
        loss=abs(float(loss + term.loss_constant - loss_d))
        / abs(float(loss_d)),
        difference=abs(float(toeplitz(v2) - toeplitz(v1)) - exact)
        / abs(exact),
        direct_difference=abs(float(reference(v2) - reference(v1)) - exact)
        / abs(exact),
        **{f"grad {k}": _relative(grad[k], grad_d[k]) for k in grad_d},
    )


@pytest.mark.validates(
    "virgil.radio_toeplitz.ToeplitzVisData",
    roots=["self-consistency"],
)
@pytest.mark.parametrize(
    ("sigmas", "offsets"),
    # fast: each σ mode once, with and without offsets; the rest slow
    [
        ("profile", True),
        (False, False),
        (True, False),
        pytest.param(False, True, marks=pytest.mark.slow),
        pytest.param(True, True, marks=pytest.mark.slow),
        pytest.param("profile", False, marks=pytest.mark.slow),
    ],
)
def test_matches_direct_term(sigmas, offsets):
    npix = 15 if offsets else 16  # odd and even grids
    data = _data(1500, npix, seed=1)
    out = _check_against_direct(data, npix, sigmas, offsets, 25.0, seed=2)
    assert out["loss"] < 1e-12
    assert out["difference"] < 1e-10
    assert all(out[k] < 1e-10 for k in out if k.startswith("grad"))


@pytest.mark.slow
@pytest.mark.validates(
    "virgil.radio_toeplitz.ToeplitzVisData",
    roots=["self-consistency"],
)
def test_matches_direct_term_at_scale():
    """64² image, 1e5 points, 2 obs x 2 ddids, gains, sub-pixel offsets and
    offsets at the bound, taper, mask, channel spread, fixed α."""
    data = _data(100_000, 64, seed=3)
    for sigmas in (False, True, "profile"):
        out = _check_against_direct(data, 64, sigmas, True, 33.0, seed=4)
        assert out["loss"] < 1e-12, (sigmas, out)
        assert out["difference"] < 1e-10, (sigmas, out)
        assert all(out[k] < 1e-10 for k in out if k.startswith("grad")), (
            sigmas,
            out,
        )


@pytest.mark.slow
def test_integer_lags_are_exact_past_nyquist():
    """Without free offsets the beam and dirty image sit at integer lags,
    exact for any baseline length (here 1.4 of the pixel Nyquist limit)."""
    data = _data(800, 12, seed=5, reach=1.4)
    nuis = CalibrationNuisance(data, offsets=False, sigmas=False, alpha=0.7)
    scene, template = _scene(12)
    tdata = data.toeplitz(template, nuis)
    assert tdata.nyquist_fraction > 1.0
    term, toeplitz, reference = _losses(tdata, scene, nuis)
    values = _values(nuis, gain=[1.2])
    values["eta"] = np.asarray(onp.random.default_rng(6).normal(size=(12, 12)))
    loss, grad = jax.value_and_grad(toeplitz)(values)
    loss_d, grad_d = jax.value_and_grad(reference)(values)
    assert float(loss + term.loss_constant) == pytest.approx(
        float(loss_d), rel=1e-12
    )
    for k in grad_d:
        assert _relative(grad[k], grad_d[k]) < 1e-10


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("smoothed", marks=pytest.mark.slow),
        pytest.param("system", marks=pytest.mark.slow),
        "half_pixel",
    ],
)
def test_scene_variants(kind):
    data = _data(1000, 12, seed=7)
    nuis = CalibrationNuisance(data, sigmas="profile", alpha=0.7)
    if kind == "smoothed":
        scene, template = _scene(12, 10.0, blur_fwhm_mas=2.5)
    elif (
        kind == "half_pixel"
    ):  # a fixed image offset, as MPoL's map convention
        scene, template = _scene(12, 10.0, dra=-PIXEL / 2, ddec=-PIXEL / 2)
    else:
        image_scene, template = _scene(12, 10.0)

        def scene(values):
            return System(disk=image_scene(values))

    tdata = data.toeplitz(template, nuis, offsets="spread")
    term, toeplitz, reference = _losses(tdata, scene, nuis)
    values = _values(nuis, gain=[0.9], offset=[[1.1, 0.3]])
    values["eta"] = np.asarray(onp.random.default_rng(8).normal(size=(12, 12)))
    loss, grad = jax.value_and_grad(toeplitz)(values)
    loss_d, grad_d = jax.value_and_grad(reference)(values)
    assert float(loss + term.loss_constant) == pytest.approx(
        float(loss_d), rel=1e-12
    )
    for k in grad_d:
        assert _relative(grad[k], grad_d[k]) < 1e-10


def test_results_match_direct():
    """``loglike``, ``derived``, ``chi2_by_ddid`` and ``crosscheck``, with a
    ddid fully masked (profiled ŝ = 1, no χ²)."""
    data = _data(1200, 10, seed=9)
    mask = onp.asarray(data.ddid) != 2
    scene, template = _scene(10)
    values = None
    for sigmas in ("profile", True):
        nuis = CalibrationNuisance(data, sigmas=sigmas, alpha=0.7)
        tdata = data.toeplitz(template, nuis, mask=mask, offsets="spread")
        values = _values(nuis, gain=[1.1], sigma=[1.1, 0.8, 1.0, 1.2])
        values["eta"] = np.asarray(
            onp.random.default_rng(10).normal(size=(10, 10))
        )
        term, direct = tdata.term(scene, nuis), tdata.direct_term(scene, nuis)
        assert float(term.loglike(values)) == pytest.approx(
            float(direct.loglike(values)), rel=1e-12
        )
        for k, x in direct.derived(values).items():
            onp.testing.assert_allclose(term.derived(values)[k], x, rtol=1e-12)
        chi2, n = tdata.chi2_by_ddid(scene, values, nuis)
        chi2_d, n_d = data.chi2_by_ddid(scene, values, nuis, mask)
        onp.testing.assert_array_equal(n, n_d)
        assert chi2[2] == 0.0 and n[2] == 0
        onp.testing.assert_allclose(chi2, chi2_d, rtol=1e-12)
        check = tdata.crosscheck(scene, values, nuis, chunk_size=256)
        assert check["max_rel"] < 1e-12  # the direct term in chunks
    # a held-out score at fixed σ
    nuis = CalibrationNuisance(data, sigmas="profile", alpha=0.7)
    tdata = data.toeplitz(template, nuis, mask=mask, offsets="spread")
    sigma = onp.array([1.3, 0.7, 1.0, 1.1])
    _, toeplitz, reference = _losses(tdata, scene, nuis, sigma=sigma)
    term = tdata.term(scene, nuis, sigma=sigma)
    assert float(toeplitz(values) + term.loss_constant) == pytest.approx(
        float(reference(values)), rel=1e-12
    )


def test_rebuild_matches_a_fresh_precompute_without_recompiling():
    from virgil.radio_toeplitz import _build

    data = _data(900, 10, seed=11)
    nuis = CalibrationNuisance(data, sigmas=False, alpha=0.7)
    _, template = _scene(10)
    tdata = data.toeplitz(template, nuis, offsets="spread")
    rng = onp.random.default_rng(12)
    vis = onp.asarray(data.vis) + 0.1 * rng.normal(size=len(data))
    weight = onp.asarray(data.weight) * rng.uniform(0.5, 1.5, len(data))
    compiled = _build._cache_size()
    rebuilt = tdata.rebuild(vis, weight)
    assert _build._cache_size() == compiled
    fresh = ComplexVisData.from_arrays(
        data.u, data.v, vis, weight, data.freq, ddid=data.ddid, obs=data.obs
    ).toeplitz(template, nuis, offsets="spread")
    for k in fresh.arrays:
        onp.testing.assert_allclose(
            rebuilt.arrays[k],
            fresh.arrays[k],
            rtol=1e-13,
            atol=1e-13 * float(np.max(np.abs(fresh.arrays[k]))),
        )
    onp.testing.assert_array_equal(rebuilt.data.vis, vis)
    weight[3] = 0.0  # a bootstrap's dropped point: mask it instead
    with pytest.raises(Exception, match="finite and positive"):
        tdata.rebuild(vis, weight)


def _fit_problem(npix=10, sigmas="profile"):
    data = _data(1200, npix, seed=13)
    nuis = CalibrationNuisance(data, sigmas=sigmas, alpha=0.7)
    scene, template = _scene(npix)
    tdata = data.toeplitz(template, nuis, offsets="spread")
    priors = nuis.priors()
    priors["eta"] = dist.Normal(0.0, 1.0)
    init = _values(nuis)
    init["eta"] = np.zeros((npix, npix))
    return tdata, nuis, scene, priors, init


def _fit_both(tdata, nuis, scene, priors, init, **kw):
    out = []
    for term in (tdata.term(scene, nuis), tdata.direct_term(scene, nuis)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.append(
                fit(
                    lambda **v: scene(v),
                    priors,
                    (),
                    likelihoods=[term],
                    init=dict(init),
                    **kw,
                )
            )
    return out


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
@pytest.mark.slow
def test_fit_reaches_the_direct_optimum():
    problem = _fit_problem()
    # Adam follows the same trajectory step for step.
    a, b = _fit_both(
        *problem, method="adam", max_steps=200, learning_rate=3e-2
    )
    for k in b.values:
        assert _relative(a.values[k], b.values[k]) < 1e-10
    # L-BFGS (the default without a residual vector) reaches the same optimum.
    a, b = _fit_both(*problem, max_steps=2000, gtol=1e-9)
    assert a.info["method"] == "lbfgs" and a.info["converged"]
    assert a.info["loss"] == pytest.approx(b.info["loss"], rel=1e-10)
    assert a.info["chi2"] == pytest.approx(b.info["chi2"], rel=1e-10)
    assert a.info["ndata"] == b.info["ndata"]
    for k in b.values:
        assert _relative(a.values[k], b.values[k]) < 1e-4
    # deliberate: only stationary quantities (loss, χ²) are held to 1e-10
    # across the two optimiser paths. The two terms sum in different orders, so
    # L-BFGS stops at points that differ at round-off; the profiled sigma
    # moves at first order in that offset (2.5e-8 on some runner CPUs, #528).
    onp.testing.assert_allclose(
        a.info["derived"]["cal.sigma"],
        b.info["derived"]["cal.sigma"],
        rtol=1e-6,
    )
    # At the same parameters the terms' sigma must agree tightly.
    tdata, nuis, scene, *_ = problem
    same = tdata.term(scene, nuis).derived(a.values)["cal.sigma"]
    direct = tdata.direct_term(scene, nuis).derived(a.values)["cal.sigma"]
    onp.testing.assert_allclose(same, direct, rtol=1e-12)


@pytest.mark.slow
def test_diag_preconditioner_uses_gauss_newton_curvature():
    tdata, nuis, scene, priors, init = _fit_problem(sigmas=False)
    values = dict(
        init, eta=np.asarray(onp.random.default_rng(14).normal(size=(10, 10)))
    )
    values["cal.offset"] = np.asarray([[0.4, -0.7]])

    def diagonal(term):
        return gauss_newton_diag(
            lambda **v: scene(v),
            priors,
            (),
            values,
            likelihoods=[term],
            n_probes=0,
        )

    toeplitz = diagonal(tdata.term(scene, nuis))
    direct = diagonal(tdata.direct_term(scene, nuis))
    # The loss is quadratic in the pixels and in G = flux·gain: the same JᵀJ.
    for k in ("eta", "cal.flux", "cal.gain"):
        assert _relative(toeplitz[k], direct[k]) < 1e-10
    # Offsets take the loss Hessian, near JᵀJ.
    assert onp.all(toeplitz["cal.offset"] > 0.0)
    kw = dict(method="lbfgs", max_steps=2000, gtol=1e-9)
    a, b = _fit_both(
        tdata, nuis, scene, priors, init, precondition="diag", **kw
    )
    plain, _ = _fit_both(tdata, nuis, scene, priors, init, **kw)
    assert a.info["converged"] and a.info["steps"] < 1.5 * b.info["steps"]
    assert a.info["steps"] < 0.5 * plain.info["steps"]
    assert a.info["loss"] == pytest.approx(b.info["loss"], rel=1e-10)


# -- refusals -------------------------------------------------------------


@pytest.fixture
def small():
    data = _data(300, 8, seed=15)
    nuis = CalibrationNuisance(data, alpha=0.7)
    scene, template = _scene(8)
    return data, nuis, scene, template


def test_refuses_what_it_cannot_do(small):
    data, nuis, scene, template = small
    with pytest.raises(ValueError, match="fixed spectral index"):
        data.toeplitz(template, CalibrationNuisance(data))
    gains = CalibrationNuisance(data, alpha=0.7)
    gains.antenna_gains = True
    with pytest.raises(ValueError, match="antenna gains"):
        data.toeplitz(template, gains)
    far = _data(300, 8, seed=15, reach=0.99)
    with pytest.raises(ValueError, match="Nyquist"):
        far.toeplitz(
            template, CalibrationNuisance(far, alpha=0.7), offsets="spread"
        )
    with pytest.raises(TypeError, match="Image"):
        data.toeplitz(PointSource(), nuis, offsets="spread")
    # fitted offsets are opt-in, and only with a nuisance that fits them
    with pytest.raises(ValueError, match="offsets='spread'"):
        data.toeplitz(template, nuis)
    with pytest.raises(ValueError, match="needs a nuisance that fits"):
        data.toeplitz(
            template,
            CalibrationNuisance(data, alpha=0.7, offsets=False),
            offsets="spread",
        )
    with pytest.raises(ValueError, match="'none' or 'spread'"):
        data.toeplitz(template, nuis, offsets=True)
    tdata = data.toeplitz(template, nuis, offsets="spread")
    with pytest.raises(ValueError, match="differs from the one"):
        tdata.term(
            scene, CalibrationNuisance(data, alpha=0.7, blur_fwhm_mas=5.0)
        )
    no_offsets = data.toeplitz(
        template, CalibrationNuisance(data, alpha=0.7, offsets=False)
    )
    with pytest.raises(ValueError, match="without"):
        no_offsets.term(scene, nuis)
    values = _values(nuis)
    values["eta"] = np.zeros((8, 8))
    for bad, error in [
        (lambda v: PointSource(), TypeError),
        (lambda v: System(a=scene(v), b=PointSource()), ValueError),
        (lambda v: Image(v["eta"], 1.5 * PIXEL), ValueError),
    ]:
        with pytest.raises(error):
            tdata.term(bad, nuis).chi2_and_log_norm(values)
    term = tdata.term(scene, nuis)
    with pytest.raises(TypeError, match="no residual vector"):
        term(values)


def test_runtime_checks(small):
    data, nuis, scene, template = small
    tdata = data.toeplitz(template, nuis, offsets="spread", max_offset_mas=2.0)
    values = _values(nuis, offset=[[0.0, 3.0]])
    values["eta"] = np.zeros((8, 8))
    loss = eqx_jit(lambda t, v: t.chi2_and_log_norm(v)[0])
    with pytest.raises(Exception, match="beyond the bound"):
        loss(tdata.term(scene, nuis), values)
    values = _values(nuis)
    values["eta"] = np.zeros((8, 8))
    moved = tdata.term(lambda v: Image(v["eta"], PIXEL, dra=1.0), nuis)
    with pytest.raises(Exception, match="template's offset"):
        loss(moved, values)


def eqx_jit(f):
    import equinox as eqx

    return eqx.filter_jit(f)


def test_fit_protocol_refusals(small):
    data, nuis, scene, template = small
    term = data.toeplitz(template, nuis, offsets="spread").term(scene, nuis)
    priors = nuis.priors()
    priors["eta"] = dist.Normal(0.0, 1.0)
    init = _values(nuis)
    init["eta"] = np.zeros((8, 8))
    model = lambda **v: scene(v)  # noqa: E731
    with pytest.raises(NoLeastSquaresForm):
        fit(model, priors, (), likelihoods=[term], init=init, method="lm")
    with pytest.raises(NoLeastSquaresForm):
        gauss_newton_mass(model, priors, (), init, likelihoods=[term])
    with pytest.raises(ValueError, match="float64"):
        fit(
            model,
            priors,
            (),
            likelihoods=[term],
            init=init,
            dtype="float32",
            max_steps=2,
        )
    # a few L-BFGS steps: the reported χ² and point count are the direct term's
    result = fit(model, priors, (), likelihoods=[term], init=init, max_steps=3)
    direct = data.toeplitz(template, nuis, offsets="spread").direct_term(
        scene, nuis
    )
    residuals, _ = direct.residuals_and_log_norm(result.values)
    assert result.info["method"] == "lbfgs"
    assert result.info["chi2"][0] == pytest.approx(
        float(np.sum(residuals**2)), rel=1e-10
    )
    assert result.info["ndata"] == [2 * len(data)]


def test_needs_float64(small):
    data, nuis, scene, template = small
    with jax.enable_x64(False), pytest.raises(ValueError, match="float64"):
        data.toeplitz(template, nuis, offsets="spread")


def test_fixed_alpha_nuisance(small):
    data, _, scene, _ = small
    fixed = CalibrationNuisance(data, alpha=0.7)
    fitted = CalibrationNuisance(data, alpha=True)
    assert "cal.alpha" not in fixed.priors() and fixed.alpha_fixed == 0.7
    values = _values(fitted)
    values["eta"] = np.zeros((8, 8))
    values["cal.alpha"] = np.asarray(0.7)
    onp.testing.assert_allclose(
        data.model(scene(values), values, fixed),
        data.model(scene(values), values, fitted),
        rtol=1e-14,
    )
    assert CalibrationNuisance(data, alpha=False).alpha_fixed == 0.0
    assert CalibrationNuisance(data, alpha=0).alpha_fixed == 0.0
    with pytest.raises(ValueError, match="not the integer 1"):
        CalibrationNuisance(data, alpha=1)  # once "fit α"


# -- the spread-grid adapter ---------------------------------------------


@pytest.mark.validates("virgil._nufft_spread.evaluate", roots=["mathematics"])
def test_spread_grid_against_nufftax_and_direct_sums():
    from nufftax import nufft2d1

    rng = onp.random.default_rng(16)
    n, npix = 3000, 16
    tx, ty = rng.uniform(-0.8, 0.8, (2, n)) * onp.pi
    c = rng.normal(size=n) + 1j * rng.normal(size=n)
    kp = spread_grid.kernel(1e-14)

    def direct(x, y):
        return onp.real(
            onp.einsum(
                "k,ky,kx->yx",
                c,
                onp.exp(1j * ty[:, None] * y),
                onp.exp(1j * tx[:, None] * x),
            )
        )

    # integer modes: nufftax's own type-1 transform
    m = onp.arange(-npix // 2, npix // 2)
    nf = spread_grid.grid_size(npix, kp)
    b = spread_grid.spread(
        np.asarray(tx), np.asarray(ty), np.asarray(c), nf, kp
    )
    ours = spread_grid.evaluate(b, m, 0.0, m, 0.0, kp)
    theirs = onp.real(
        nufft2d1(
            np.asarray(tx),
            np.asarray(ty),
            np.asarray(c),
            (npix, npix),
            eps=1e-14,
            isign=1,
        )
    )
    scale = onp.abs(theirs).max()
    assert onp.abs(ours - theirs).max() < 1e-13 * scale
    # any real position within M/2, kernel not wrapping
    nf = spread_grid.grid_size(npix + 26, kp)
    assert 0.8 < spread_grid.margin(nf, kp)
    b = spread_grid.spread(
        np.asarray(tx), np.asarray(ty), np.asarray(c), nf, kp
    )
    m = npix // 2 - 1 - onp.arange(npix)
    for fx, fy in [(0.5, 0.5), (0.8, -0.3), (6.5, -5.5)]:
        ours = spread_grid.evaluate(b, m, fx, m, fy, kp)
        assert onp.abs(ours - direct(m + fx, m + fy)).max() < 1e-13 * scale
    # and differentiable in the shift
    fx = 0.3

    def at(f):
        return spread_grid.evaluate(b, m, f, m, 0.0, kp)[3, 5]

    x = m[5] + fx
    expected = onp.sum(
        onp.real(c * 1j * tx * onp.exp(1j * (tx * x + ty * m[3])))
    )
    assert float(jax.grad(at)(fx)) == pytest.approx(expected, rel=1e-11)


def test_spread_grid_needs_nufftax_0_6_1(monkeypatch):
    import nufftax

    for version in ("0.6.0", "0.6.0rc1", "0.5"):
        monkeypatch.setattr(nufftax, "__version__", version)
        with pytest.raises(ImportError, match="0.6.1"):
            spread_grid.kernel(1e-12)
    for version in ("0.7rc1", "0.6.2a1", "1.0.0.dev3"):
        monkeypatch.setattr(nufftax, "__version__", version)
        spread_grid.kernel(1e-12)


# -- chi2="auto": the choice in fit ----------------------------------------


@pytest.fixture
def auto_always():
    """Make every problem here big enough for ``chi2="auto"``."""
    old = set_ft_defaults(toeplitz_min_points=1, toeplitz_points_per_pixel=0)
    yield
    set_ft_defaults(**old)


def _auto(npix=8, n=600, scene=None, **nuis_kw):
    data = _data(n, npix, seed=21)
    kw = dict(alpha=0.7, offsets=False, sigmas=True) | nuis_kw
    nuis = CalibrationNuisance(data, **kw)
    scene = scene or _scene(npix)[0]
    init = _values(nuis)
    init["eta"] = np.zeros((npix, npix))
    return data, nuis, scene, init


def test_auto_term_is_toeplitz_and_exact(auto_always):
    data, nuis, scene, init = _auto()
    term = data.term(scene, nuis)
    resolved = term.resolve(init, False)[0]
    assert type(resolved).__name__ == "_ToeplitzTerm"
    rng = onp.random.default_rng(3)
    values = dict(init, eta=np.asarray(rng.normal(size=(8, 8))))

    def direct(v):
        r, log_norm = term.residuals_and_log_norm(v)
        return 0.5 * np.sum(r**2) + log_norm

    def toeplitz(v):
        chi2, log_norm = resolved.chi2_and_log_norm(v)
        return 0.5 * (chi2 + resolved.chi2_constant) + log_norm

    assert float(toeplitz(values)) == pytest.approx(
        float(direct(values)), rel=1e-10
    )
    g1, g2 = jax.grad(toeplitz)(values), jax.grad(direct)(values)
    for k in g2:
        onp.testing.assert_allclose(g1[k], g2[k], rtol=1e-10, atol=1e-9)


def _two_part(values):
    return System(
        star=PointSource(), env=Image(values["eta"], PIXEL, ft="dft")
    )


@pytest.mark.parametrize(
    "case",
    ["least_squares", "alpha", "offsets", "scene", "dra", "float32", "few"],
)
def test_auto_falls_back_to_direct(case, auto_always):
    kw, scene, least_squares = {}, None, False
    if case == "alpha":
        kw = dict(alpha=True)
    elif case == "offsets":
        kw = dict(offsets=True)
    elif case == "scene":
        scene = _two_part
    elif case == "dra":
        scene = lambda v: Image(  # noqa: E731
            v["eta"], PIXEL, ft="dft", dra=v["cal.flux"]
        )
    least_squares = case == "least_squares"
    data, nuis, scene, init = _auto(scene=scene, **kw)
    if case == "float32":
        init = {k: x.astype(np.float32) for k, x in init.items()}
    if case == "few":
        set_ft_defaults(toeplitz_min_points=10**9)
    term = data.term(scene, nuis)
    assert term.resolve(init, least_squares)[0] is term


def test_chi2_overrides(auto_always):
    data, nuis, scene, init = _auto()
    direct = data.term(scene, nuis, chi2="direct")
    assert direct.resolve(init, False) == (direct, "direct")
    set_ft_defaults(toeplitz_min_points=10**9)  # too small for auto only
    forced = data.term(scene, nuis, chi2="toeplitz").resolve(init, False)[0]
    assert type(forced).__name__ == "_ToeplitzTerm"
    with pytest.raises(ValueError, match="least squares"):
        data.term(scene, nuis, chi2="toeplitz").resolve(init, True)
    with pytest.raises(ValueError, match="not one Image"):
        data.term(_two_part, nuis, chi2="toeplitz").resolve(init, False)[0]
    with pytest.raises(ValueError, match="chi2 must be"):
        data.term(scene, nuis, chi2="fast")


def test_toeplitz_refuses_antenna_gains(auto_always):
    base = _data(300, 8, seed=2)
    rng = onp.random.default_rng(0)
    data = ComplexVisData.from_arrays(
        base.u,
        base.v,
        base.vis,
        base.weight,
        base.freq,
        ddid=base.ddid,
        obs=base.obs,
        time=onp.sort(rng.uniform(0, 100, 300)),
        ant1=rng.integers(0, 3, 300),
        ant2=rng.integers(3, 6, 300),
    )
    nuis = CalibrationNuisance(
        data, alpha=0.7, offsets=False, antenna_gains=True, solint="int"
    )
    scene = _scene(8)[0]
    init = {"eta": np.zeros((8, 8)), "cal.flux": np.asarray(1.0)}
    for chi2 in ("auto", "toeplitz"):
        term = data.term(scene, nuis, chi2=chi2)
        if chi2 == "auto":
            assert term.resolve(init, False)[0] is term
        else:
            with pytest.raises(ValueError, match="antenna_gains"):
                term.resolve(init, False)[0]
    with pytest.raises(ValueError, match="antenna gains"):
        data.toeplitz(_scene(8)[1], nuis)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("masked", [False, True])
def test_fit_with_auto_matches_direct_and_compiles_once(masked, auto_always):
    npix = 6
    shifted = lambda v: System(  # noqa: E731
        env=Image(v["eta"], PIXEL, ft="dft", dra=0.3, ddec=-0.2), dra=0.4
    )
    data, nuis, scene, init = _auto(
        npix=npix, n=400, scene=shifted, sigmas="profile"
    )
    mask = onp.arange(400) % 5 != 0 if masked else None
    priors = nuis.priors()
    priors["eta"] = dist.Normal(0.0, 1.0)
    model = lambda **v: scene(v)  # noqa: E731
    results = {}
    for chi2 in ("auto", "direct"):
        term = data.term(scene, nuis, mask=mask, chi2=chi2)
        results[chi2] = fit(
            model,
            priors,
            (),
            likelihoods=[term],
            init=dict(init),
            method="lbfgs",
            max_steps=5,
        )
    assert results["auto"].info["likelihood_forms"] == ["toeplitz"]
    assert results["direct"].info["likelihood_forms"] == ["direct"]
    assert results["auto"].info["chi2"][0] == pytest.approx(
        results["direct"].info["chi2"][0], rel=1e-8
    )
    if masked:
        return
    # the hook is wired into fit: an explicit request meets the method check
    with pytest.raises(ValueError, match="least squares"):
        fit(
            model,
            priors,
            (),
            likelihoods=[data.term(scene, nuis, chi2="toeplitz")],
            init=dict(init),
            method="lm",
        )
    term = data.term(scene, nuis)
    kw = dict(likelihoods=[term], init=dict(init), method="lbfgs", max_steps=5)
    fit(model, priors, (), **kw)
    with count_compiles() as compiles:
        fit(model, priors, (), **kw)
    assert compiles == []


def test_auto_reports_why_it_fell_back(auto_always):
    data, nuis, scene, init = _auto(offsets=True)
    term, form = data.term(scene, nuis).resolve(init, False)
    assert form.startswith("direct (auto)") and "offsets" in form
    set_ft_defaults(toeplitz_max_bytes=1)
    data, nuis, scene, init = _auto()
    assert (
        "toeplitz_max_bytes" in data.term(scene, nuis).resolve(init, False)[1]
    )
    _, form = data.term(scene, nuis, chi2="toeplitz").resolve(init, False)
    assert form == "toeplitz"  # explicit requests are not capped


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_adam_with_auto_picks_toeplitz_and_matches_direct(auto_always):
    data, nuis, scene, init = _auto(npix=6, n=400)
    priors = nuis.priors()
    priors["eta"] = dist.Normal(0.0, 1.0)
    out = {}
    for chi2 in ("auto", "direct"):
        out[chi2] = fit(
            lambda **v: scene(v),
            priors,
            (),
            likelihoods=[data.term(scene, nuis, chi2=chi2)],
            init=dict(init),
            method="adam",
            max_steps=30,
        )
    assert out["auto"].info["likelihood_forms"] == ["toeplitz"]
    assert out["auto"].info["loss"] == pytest.approx(
        out["direct"].info["loss"], rel=1e-10
    )
    onp.testing.assert_allclose(
        out["auto"].info["loss_history"],
        out["direct"].info["loss_history"],
        rtol=1e-10,
    )


def test_old_taper_name_gives_identical_toeplitz_and_direct_losses():
    data = _data(300, 12, seed=7)
    scene, template = _scene(12)
    losses = []
    for kw in ({"blur_fwhm_mas": 3.0}, {"taper_fwhm_mas": 3.0}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            nuis = CalibrationNuisance(data, alpha=0.7, offsets=False, **kw)
        tdata = data.toeplitz(template, nuis)
        values = _values(nuis, flux=0.8)
        values["eta"] = np.asarray(
            onp.random.default_rng(1).normal(size=(12, 12))
        )
        term = tdata.term(scene, nuis)
        direct = tdata.direct_term(scene, nuis)
        r, _ = direct.residuals_and_log_norm(values)
        losses.append((term.chi2_and_log_norm(values)[0], np.sum(r**2)))
    assert losses[0][0] == losses[1][0]
    assert losses[0][1] == losses[1][1]


def _noise_free_term(seed=1, n=300):
    """A profiled term on noise-free data from a known image and gains, and
    the true values: at them C_d + Δ_d is the Toeplitz approximation error
    plus round-off, of either sign."""
    rng = onp.random.default_rng(seed)
    ddid = rng.integers(0, len(DDID_OBS), n)
    obs = onp.asarray(DDID_OBS)[ddid]
    freq = CENTRES[ddid] * (1.0 + rng.uniform(-0.01, 0.01, n))
    u, v = _uv(n, 0.45, rng)
    weight = rng.uniform(0.5, 2.0, n) * 1e2
    kw = dict(ddid=ddid + 5, obs=obs + 2)
    blank = ComplexVisData.from_arrays(
        u, v, onp.zeros(n, complex), weight, freq, **kw
    )
    nuis = CalibrationNuisance(
        blank, sigmas="profile", offsets=False, alpha=0.7
    )
    eta = rng.normal(size=(8, 8))
    values = _values(nuis, flux=0.9, gain=[1.15])
    values["eta"] = eta
    scene, template = _scene(8)
    model = onp.asarray(blank.model(scene(values), values, nuis))
    data = ComplexVisData.from_arrays(u, v, model, weight, freq, **kw)
    tdata = data.toeplitz(template, nuis)
    return tdata.term(scene, nuis), values, tdata.direct_term(scene, nuis)


def test_profiled_loss_floors_value_and_gradient_together():
    """Where C_d + Δ_d is round-off (the model reproduces noise-free data),
    the profiled loss and its gradient are finite and are those of
    ``N ln max(1 + Δ/C, eps)``: value and gradient floor at the same place.
    Above the floor the Toeplitz term matches the direct term."""
    term, values, _ = _noise_free_term()
    c_sum, count = term.arrays["c_sum"], term.arrays["count"]
    eps = np.finfo(c_sum.dtype).eps

    def loss(v):
        chi2, log_norm = term.chi2_and_log_norm(v)
        return 0.5 * chi2 + log_norm

    def floored(v):
        delta = term._deltas(term.curvature_params(v))
        return np.sum(count * np.log(np.maximum(1.0 + delta / c_sum, eps)))

    for scale in (1.0, 0.99, 1.01):
        v = {**values, "eta": values["eta"] * scale}
        val, grad = jax.value_and_grad(loss)(v)
        ref_val, ref_grad = jax.value_and_grad(floored)(v)
        assert onp.isfinite(float(val))
        assert float(val) == pytest.approx(float(ref_val), abs=1e-12)
        for k in grad:
            assert onp.all(onp.isfinite(grad[k]))
            onp.testing.assert_allclose(
                grad[k], ref_grad[k], rtol=1e-7, atol=1e-9
            )
    # the depth of the floor is bounded: N ln(eps) per ddid, not N ln(tiny)
    assert float(floored(values)) >= float(np.sum(count)) * np.log(eps)


def test_toeplitz_matches_the_direct_term_on_noise_free_data_above_the_floor():
    """On noise-free data a model
    well away from the truth leaves C + Δ far above round-off, and the
    Toeplitz loss and gradient equal the direct term's; the comparison shares
    none of the Toeplitz arithmetic."""
    term, values, direct = _noise_free_term()
    c_sum = term.arrays["c_sum"]
    for scale in (0.0, 0.5, 1.5):
        v = {**values, "eta": values["eta"] * scale}
        # above the floor: C + Δ is a finite fraction of C, not round-off
        delta = term._deltas(term.curvature_params(v))
        assert float(np.min((c_sum + delta) / c_sum)) > 1e-3
        t_val, t_grad = jax.value_and_grad(term.loglike)(v)
        d_val, d_grad = jax.value_and_grad(direct.loglike)(v)
        assert float(t_val) == pytest.approx(float(d_val), rel=1e-10)
        for k in d_grad:
            onp.testing.assert_allclose(
                t_grad[k], d_grad[k], rtol=1e-8, atol=1e-8
            )


@pytest.mark.xfail(
    strict=True,
    reason="max_shift_pix ignores nuis.priors(offset=...) since fa53dce: "
    "26.73 mas, not 50 (virgil#589)",
)
def test_custom_offset_prior_bounds_widen_toeplitz_precompute(small):
    data, nuis, scene, template = small
    wide = 50.0
    nuis.priors(offset=(-wide, wide))
    tdata = data.toeplitz(template, nuis, offsets="spread")
    assert tdata.max_shift_pix * PIXEL == pytest.approx(wide, rel=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="curvature_loss is 0.5*chi2 + log_norm since b9e7be9: 638.27 vs "
    "0.5*chi2 = 88.49 (virgil#590)",
)
def test_curvature_loss_is_half_chi2_without_log_norm(small):
    """If virgil#590 decides ``_ToeplitzTerm.curvature_loss`` should drop
    ``log_norm``, it returns ``0.5 * chi2``. Today it adds ``log_norm``."""
    data, _, scene, template = small
    nuis = CalibrationNuisance(data, sigmas=True, offsets=False, alpha=0.7)
    tdata = data.toeplitz(template, nuis)
    term = tdata.term(scene, nuis)
    values = _values(nuis, flux=0.8, gain=[1.1], sigma=[2.5, 2.5, 2.5, 2.5])
    values["eta"] = np.zeros((8, 8))
    params = term.curvature_params(values)
    chi2, log_norm = term._reduced(params)
    assert float(log_norm) != 0.0
    assert float(term.curvature_loss(params)) == pytest.approx(
        0.5 * float(chi2), rel=1e-15
    )
