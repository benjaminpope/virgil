"""LCurve.dof_discrepancy: Hutchinson G and the dof-corrected Morozov root."""

import types

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import imaging
from virgil.coverage import ami_grid_record
from virgil.imaging import (
    LCurve,
    MaxEntropy,
    _hutchinson_g,
    _lerp_root,
    image_priors,
    l_curve,
)
from virgil.models import GaussianDisk, LinearImage
from virgil.oidata import OIData

NPIX, SCALE = 6, 16.0
DATA = OIData(ami_grid_record(pitch_m=0.7))


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def test_hutchinson_g_matches_the_exact_trace():
    rng = onp.random.default_rng(1)
    n = 40
    a = rng.normal(size=(n, 12))
    k = a @ a.T
    h = onp.diag(rng.uniform(0.5, 2.0, n)) + 0.1 * k
    exact = onp.trace(k @ onp.linalg.inv(k + h))
    g, se = _hutchinson_g(
        lambda v: k @ v, lambda v: h @ v, onp.ones(n), nprobe=64, tol=1e-10
    )
    assert se > 0.0
    assert abs(g - exact) <= 3.0 * se


def test_hutchinson_g_respects_the_free_mask():
    n = 10
    k, h = onp.diag(onp.arange(1.0, n + 1)), onp.eye(n)
    mask = onp.arange(n) < 6
    g, se = _hutchinson_g(
        lambda v: k @ v, lambda v: h @ v, mask, nprobe=4, tol=1e-12
    )
    exact = sum(lam / (lam + 1.0) for lam in range(1, 7))
    assert g == pytest.approx(exact, rel=1e-9) and se < 1e-9


def test_indefinite_hessian_is_refused():
    with pytest.raises(ValueError, match="positive definite"):
        _hutchinson_g(lambda v: 0.0 * v, lambda v: -v, onp.ones(4), nprobe=2)


# A linear-Gaussian toy: eigenvalues lam of K, data components d_j, ridge
# H = w I: G(w) = Σ lam/(lam + w) and χ²(w) = Σ d² (w/(lam + w))².
LAM = onp.array([40.0, 10.0, 3.0, 1.0, 0.3, 0.1])
D2 = onp.array([30.0, 8.0, 5.0, 2.0, 1.0, 0.5])
N_REAL = 20.0


def _analytic(w):
    g = onp.sum(LAM / (LAM + w))
    chi2 = onp.sum(D2 * (w / (LAM + w)) ** 2) + 3.0
    return chi2, g


def _toy_curve(weights, monkeypatch, s2=1.0):
    def fake_point(result, regularizer, weight, data, s2_, n_real, probe, c):
        chi2, g = _analytic(weight)
        # G through the estimator itself, on diagonal K and H = w I.
        g_est, se = _hutchinson_g(
            lambda v: LAM * v,
            lambda v: weight * v,
            onp.ones(LAM.size),
            nprobe=8,
            tol=1e-12,
        )
        assert g_est == pytest.approx(g, rel=1e-9)
        return dict(
            result=types.SimpleNamespace(values=weight),
            weight=weight,
            chi2=chi2,
            g=g,
            se=0.05,
            penalty=1.0,
            m=chi2 / s2_ - (n_real - g),
        )

    monkeypatch.setattr(imaging, "_dof_point", fake_point)
    results = [
        types.SimpleNamespace(info={"ndata": [N_REAL]}, values=w)
        for w in weights
    ]
    return LCurve(
        np.asarray(weights),
        np.zeros(len(weights)),
        np.zeros((len(weights), 1)),
        np.zeros(len(weights)),
        results,
        lambda w, init: types.SimpleNamespace(values=w),
    )


def _analytic_root(s2=1.0):
    from scipy.optimize import brentq

    def m(lw):
        chi2, g = _analytic(onp.exp(lw))
        return chi2 / s2 - (N_REAL - g)

    return float(onp.exp(brentq(m, -12.0, 12.0, xtol=1e-14)))


def test_root_matches_the_analytic_dof_corrected_discrepancy(monkeypatch):
    weights = onp.logspace(3, -3, 25)
    out = _toy_curve(weights, monkeypatch).dof_discrepancy(
        None, MaxEntropy(1.0), noise_scale_sigma=0.01
    )
    assert out.weight == pytest.approx(_analytic_root(), rel=2e-2)
    assert out.accepted is True and abs(out.m) <= 3.0 * out.g_se
    assert out.n_real == N_REAL
    up, down = out.weight_band  # the roots at s² ± 2σ_s, from the sweep
    assert up == pytest.approx(_analytic_root(1.02), rel=5e-2)
    assert down == pytest.approx(_analytic_root(0.98), rel=5e-2)
    assert up != down


def test_secant_step_reduces_the_residual(monkeypatch):
    weights = onp.logspace(3, -3, 5)  # coarse: interpolation is rough
    curve = _toy_curve(weights, monkeypatch)
    plain = curve.dof_discrepancy(None, MaxEntropy(1.0), secant=False)
    out = curve.dof_discrepancy(None, MaxEntropy(1.0))
    assert plain.accepted is None and len(plain.trace) == 0
    first = out.trace[0]
    assert len(out.trace) == 2 and abs(out.trace[1][1]) < abs(first[1])
    assert abs(out.m) < abs(first[1])
    exact = _analytic_root()
    assert abs(onp.log(out.weight / exact)) < abs(
        onp.log(plain.weight / exact)
    )


def test_no_crossing_gives_none(monkeypatch):
    curve = _toy_curve(onp.logspace(4, 2, 4), monkeypatch)  # m > 0 throughout
    assert curve.dof_discrepancy(None, MaxEntropy(1.0)) is None


def test_lerp_root_interpolates_in_ln_w():
    i, w = _lerp_root(onp.array([100.0, 1.0]), onp.array([1.0, -1.0]))
    assert (i, w) == (0, pytest.approx(10.0))
    assert _lerp_root(onp.array([2.0, 1.0]), onp.array([1.0, 2.0])) is None


@pytest.fixture(scope="module")
def sweep():
    with jax.enable_x64(True):
        return _fit_sweep()


def _fit_sweep():
    truth = LinearImage(
        onp.sqrt(
            0.3
            * onp.asarray(
                GaussianDisk(sigma=20.0)._image(
                    *np.meshgrid(
                        (np.arange(NPIX) - 2.5) * SCALE,
                        (np.arange(NPIX) - 2.5) * SCALE,
                    ),
                    SCALE,
                )
            )
            / 1.0
        ),
        SCALE,
        parametrisation="sqrt",
        reference_flux=0.3,
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(3))
    start = LinearImage(
        np.sqrt(np.full((NPIX, NPIX), 0.3 / NPIX**2)),
        SCALE,
        parametrisation="sqrt",
        reference_flux=0.3,
    )
    reg = MaxEntropy(1.0)
    curve = l_curve(
        start,
        image_priors(start),
        data,
        reg,
        [1e1, 1e0],
        method="lbfgs",
        max_steps=150,
    )
    return curve, data, reg


def test_g_agrees_with_the_dense_trace_on_a_real_fit(sweep):
    curve, data, reg = sweep
    result = curve.results[1]
    weight = float(curve.weights[1])
    point = imaging._dof_point(
        result, reg, weight, data, 1.0, 1.0, (64, 1e-10, 2000, 0), None
    )
    model = result.model
    x = model.pixels
    b0 = x**2
    _, jac = imaging._residual_jacobian(model, data, "pixels")
    jb = jac / (2.0 * onp.asarray(x).ravel())
    k = jb.T @ jb
    weighted = MaxEntropy(weight)
    h = onp.asarray(
        jax.hessian(lambda b: weighted.value(model.set("pixels", np.sqrt(b))))(
            b0
        ).reshape(b0.size, b0.size)
    )
    exact = onp.trace(k @ onp.linalg.inv(k + h))
    assert abs(point["g"] - exact) <= 3.0 * point["se"] + 1e-6 * exact


def test_dof_discrepancy_end_to_end_and_existing_discrepancy_unchanged(sweep):
    curve, data, _ = sweep
    before = curve.discrepancy(target=1.0)
    out = curve.dof_discrepancy(data, MaxEntropy(1.0), nprobe=8)
    assert curve.discrepancy(target=1.0) == before
    if out is not None:
        assert out.weights.size == 2 and onp.isfinite(out.m)
        assert out.h is not None and onp.isfinite(out.h)
    assert curve.refit is not None


def test_existing_discrepancy_on_a_known_curve():
    w = np.logspace(3, -3, 13)
    t = np.log10(w)
    chi2 = 100.0 * (1.0 + 10.0 ** np.clip(t, 0, None))
    curve = LCurve(w, chi2, chi2 / 100.0, 1.0 + 0.0 * t, [])
    assert np.isclose(curve.discrepancy(target=11.0), 10.0, rtol=0.05)
    assert curve.discrepancy(target=0.5) is None
