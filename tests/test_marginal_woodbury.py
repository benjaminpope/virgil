"""The Woodbury scale marginal of gains and offsets against its oracles.

The oracle is the rank-one whitening that the surface used before
(``_whiten_with_log_norm`` at each scale, kept as ``_whiten_at``) and the
dense covariance ``GainModes.covariance``; values, gradients and Hessians are
compared to a fine quadrature of the oracle with no window or centring.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil._linear import (
    arrow_at,
    arrow_problem,
    whiten_blocks,
    whiten_cholesky,
)
from virgil._marginal import gauss_legendre
from virgil._precision import cast_tree, run_in
from virgil.aliases import _N_SITES
from virgil.coverage import VLTI_UTS, vlti_oidata
from virgil.epochs import _Surface, _table_floats, _N_NODES
from virgil.likelihood import _gain_jacobian, _prediction_and_errors
from virgil.likelihood import _whiten_with_log_norm
from virgil.models import BinaryModelCartesian
from virgil.observables import VisibilityAmplitude

TRUTH = BinaryModelCartesian(dra=6.0, ddec=-4.0, flux=0.3)
S_MAX = 10.0
CASES = ["gains_offsets", "gains_only", "extra", "von_mises", "no_modes"]


def _night(case, seed=0):
    """Three nights' worth of frames with gains (two spanning modes, which
    join every frame), closure offsets, a V² amplitude extra block or
    uncorrelated closure phases (three telescopes), by ``case``."""
    stations = VLTI_UTS[:3] if case == "von_mises" else VLTI_UTS
    data = vlti_oidata(
        stations=stations,
        hour_angles_h=(-1.0, 0.0, 1.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 3),
        sigma_v2=0.03,
        sigma_cp_deg=3.0,
    ).with_model(TRUTH, jax.random.PRNGKey(seed), noise_scale=2.0)
    n = int(onp.asarray(data.vis).size)
    modes = onp.random.default_rng(1).normal(size=(2, n))
    if case == "no_modes":  # no spanning columns
        modes = None
    data = data.with_gains(
        telescope=0.05, baseline=0.03, chromatic=0.01, modes=modes
    )
    if case != "von_mises" and case != "gains_only":
        data = data.with_closure_offsets(triangle=0.01)
    if case == "extra":
        block = VisibilityAmplitude(
            np.full(n, 0.6), np.full(n, 0.05), onp.arange(n, dtype=onp.int32)
        )
        data = eqx.tree_at(lambda d: d.extras, data, (block,))
    return data


def _whiten_at(surface, data, model):
    """The oracle ``at(log_s)``: each block's χ² and gains/offsets log
    normalizer by whitening the data with the errors times ``exp(log_s)``
    through the rank-one scans (the surface's former path)."""
    prediction, errors, values, gain_terms, offset_terms = (
        _prediction_and_errors(model, data, {})
    )
    sizes = onp.asarray(surface.sizes)
    index = {n: k for k, n in enumerate(surface.names)}
    labels = ["vis", "phi"] + [f"extra[{k}]" for k in range(len(sizes) - 2)]
    block_of = onp.array([index.get(n, 0) for n in labels])

    def at(log_s):
        scale = np.repeat(
            np.exp(log_s[block_of]),
            sizes,
            total_repeat_length=int(sizes.sum()),
        )
        w, _, norm = _whiten_with_log_norm(
            data,
            prediction,
            values,
            errors * scale,
            gain_terms,
            offset_terms,
            split_norm=True,
        )
        own = np.stack(
            [
                norm[0]
                if name == "vis"
                else norm[1]
                if name == "phi"
                else np.zeros((), norm.dtype)
                for name in surface.names
            ]
        )
        return surface.block_chi2(w), own

    return at


def _oracle_score(surface, data, model, dof, n=400):
    """log ∫ exp(log f) over ln s of each block, by an ``n``-node rule over
    the whole bounded range (no window), from the rank-one whitening."""
    at = _whiten_at(surface, data, model)
    bound, low = float(onp.log(surface.s_max)), surface._low
    x, w = gauss_legendre(n, np.asarray(0.0).dtype)
    t = 0.5 * (bound + low) + 0.5 * (bound - low) * x
    log_w = np.log(w) + np.log(0.5 * (bound - low))
    nb = len(surface.blocks)
    # blocks are independent: the integral of block k needs only its own t
    total = 0.0
    for k in range(nb):
        c, o = jax.vmap(lambda ti, k=k: _at_block(at, nb, k, ti))(t)
        log_f = surface._log_f(k, t, c, o, data)
        total = total + jax.nn.logsumexp(dof[k] * log_f + log_w)
    return total + surface.floor * nb * onp.log(2.0)


def _at_block(at, nb, k, t):
    chi2, own = at(np.zeros(nb).at[k].set(t))
    return chi2[k], own[k]


def _kappa(data, model, s2_min):
    """The condition number of ``s² I + ŨᵀŨ`` at the smallest scale."""
    prediction, errors, _, gain_terms, _ = _prediction_and_errors(
        model, data, {}
    )
    n_vis = int(onp.asarray(data.vis).size)
    e = errors[:n_vis]
    jac = _gain_jacobian(data, prediction[:n_vis]) / e
    gains = data.gains
    uut = gains.covariance(
        np.zeros(n_vis), jac * e, gains.widths_for(gain_terms)
    )
    top = float(np.linalg.eigvalsh(uut / e[:, None] / e[None, :])[-1])
    return (top + s2_min) / s2_min


def _log_s(surface, seed):
    rng = onp.random.default_rng(seed)
    return np.asarray(rng.uniform(-1.5, 1.5, len(surface.blocks)))


@pytest.mark.parametrize("case", CASES)
def test_chi2_and_normalizer_match_the_rank_one_whitening(case):
    with run_in("float64"):
        data = _night(case)
        surface = _Surface(data, S_MAX)
        args = _prediction_and_errors(TRUTH, data, {})
        at = surface._at(*args, data)
        old = _whiten_at(surface, data, TRUTH)
        for seed in range(3):
            log_s = _log_s(surface, seed)
            chi2, own = at(log_s)
            chi2_old, own_old = old(log_s)
            tol = 1e-9 * float(np.max(chi2_old))
            assert onp.allclose(chi2, chi2_old, rtol=1e-9, atol=tol)
            assert onp.allclose(own, own_old, rtol=1e-9, atol=1e-9)


def test_visibility_block_matches_the_dense_covariance():
    """χ² and ½ log det of ``s² D + U Uᵀ`` from ``GainModes.covariance``."""
    with run_in("float64"):
        data = _night("gains_offsets")
        surface = _Surface(data, S_MAX)
        prediction, errors, values, gain_terms, _ = _prediction_and_errors(
            TRUTH, data, {}
        )
        n = int(onp.asarray(data.vis).size)
        at = surface._at(prediction, errors, values, gain_terms, {}, data)
        gains = data.gains
        jac = _gain_jacobian(data, prediction[:n])
        r = onp.asarray(prediction[:n] - values[:n])
        for t in (-1.0, 0.0, 0.8):
            s = onp.exp(t)
            cov = onp.asarray(
                gains.covariance(
                    errors[:n] * s, jac, gains.widths_for(gain_terms)
                )
            )
            chi2, own = at(np.asarray([t, 0.0]))
            # the surface's norm is ½ log det(C) - Σ log(s σ)
            half_logdet = 0.5 * onp.linalg.slogdet(cov)[1]
            assert float(chi2[0]) == pytest.approx(
                r @ onp.linalg.solve(cov, r), rel=1e-9
            )
            assert float(own[0]) == pytest.approx(
                half_logdet - onp.sum(onp.log(errors[:n] * s)), rel=1e-9
            )


@pytest.mark.parametrize("case", CASES)
def test_analytic_derivatives_match_autodiff(case):
    with run_in("float64"):
        data = _night(case)
        surface = _Surface(data, S_MAX)
        at = surface._at(*_prediction_and_errors(TRUTH, data, {}), data)
        log_s = _log_s(surface, 0)
        _, _, d_chi2, d_own = at(log_s, True)
        ones = np.ones_like(log_s)
        _, (j_chi2, j_own) = jax.jvp(lambda t: at(t), (log_s,), (ones,))
        assert onp.allclose(d_chi2, j_chi2, rtol=1e-8)
        assert onp.allclose(d_own, j_own, rtol=1e-8, atol=1e-10)


def _score(surface, data, model_of, x, dof, oracle):
    model = model_of(x)
    if oracle:
        return _oracle_score(surface, data, model, dof)
    return surface.score(surface.chi2(model, data, None, dof), dof, data)


@pytest.mark.slow  # compiles the oracle's gradient and Hessian
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("inflated", [False, True])
def test_value_gradient_and_hessian_match_the_oracle(case, inflated):
    with run_in("float64"):
        data = _night(case)
        surface = _Surface(data, S_MAX)
        surface = surface.inflated if inflated else surface
        dof = np.asarray(surface.dof_vector({"vis": 0.5, "phi": 0.7}))
        dof = np.broadcast_to(dof, (len(surface.blocks),))

        def model_of(x):
            return BinaryModelCartesian(x[0], x[1], 0.3)

        x = np.asarray([6.1, -3.9])
        eps = onp.finfo(onp.float64).eps
        kappa = _kappa(data, TRUTH, onp.exp(2 * surface._low))
        q = float(
            np.sum(surface.plain_chi2(surface.chi2(TRUTH, data, None, dof)))
        )
        # the quadrature's own error (1e-9 at 32 nodes) plus rounding
        # scaled by the condition number and the size of χ²
        tol = 1e-7 + 8 * eps * kappa * q
        values = [
            _score(surface, data, model_of, x, dof, o) for o in (False, True)
        ]
        assert float(values[0]) == pytest.approx(float(values[1]), abs=tol)
        grads, hess = [], []
        for o in (False, True):
            f = lambda y, o=o: _score(surface, data, model_of, y, dof, o)  # noqa: E731
            grads.append(jax.grad(f)(x))
            hess.append(jax.hessian(f)(x))
        scale = max(1.0, float(np.max(np.abs(grads[1]))))
        assert onp.allclose(grads[0], grads[1], rtol=1e-6, atol=tol * scale)
        scale = max(1.0, float(np.max(np.abs(hess[1]))))
        assert onp.allclose(hess[0], hess[1], rtol=1e-5, atol=tol * scale)


@pytest.mark.parametrize("case", ["gains_offsets", "von_mises"])
def test_fixed_score_matches_the_oracle(case):
    with run_in("float64"):
        data = _night(case)
        surface = _Surface(data, S_MAX)
        dof = np.asarray([0.5, 0.7])
        log_s = np.asarray([0.3, -0.4])
        old = _whiten_at(surface, data, TRUTH)
        chi2, own = old(log_s)
        expect = sum(
            dof[k]
            * surface._log_f(k, log_s[k : k + 1], chi2[k], own[k], data)[0]
            for k in range(len(surface.blocks))
        )
        got = surface.fixed_score(TRUTH, data, None, dof, log_s)
        assert float(got) == pytest.approx(float(expect), rel=1e-10)


def test_gauss_newton_weights_use_the_plain_chi2_of_the_table():
    with run_in("float64"):
        data = _night("gains_offsets")
        surface = _Surface(data, S_MAX)
        dof = np.asarray([0.5, 0.7])
        table = surface.chi2(TRUTH, data, None, dof)
        old = _whiten_at(surface, data, TRUTH)(np.zeros(2))[0]
        assert onp.allclose(surface.plain_chi2(table), old, rtol=1e-9)
        w = surface.weights(table, dof, data)
        assert onp.allclose(w, dof * surface.nu / old, rtol=1e-9)
        r = onp.random.default_rng(0).normal(size=data.n_residuals)
        rows = surface.weighted_rows(np.asarray(r), dof, data)
        # the rows carry the weights of their own block χ²
        w_r = dof * surface.nu / onp.asarray(surface.block_chi2(np.asarray(r)))
        want = onp.concatenate(
            [
                onp.sqrt(w_r[k]) * r[a:b]
                for k, (_, a, b, _, _) in enumerate(surface.blocks)
            ]
        )
        assert onp.allclose(rows, want)


@pytest.mark.parametrize("case", ["gains_offsets", "extra", "collinear"])
@pytest.mark.parametrize("at_low_edge", [False, True])
def test_float32_matches_float64_on_the_same_data(case, at_low_edge):
    """The float32 path on cast float64 data stays within 1e-5 of float64
    (also at the window's low edge, where ``s² I + G`` is worst conditioned,
    and with a spanning mode near the per-frame span)."""
    with run_in("float64"):
        data = _night("gains_offsets" if case == "collinear" else case)
        if case == "collinear":
            n = int(onp.asarray(data.vis).size)
            modes = onp.ones((2, n))
            modes[1] += 1e-3 * onp.random.default_rng(2).normal(size=n)
            data = data.with_gains(
                telescope=0.05, baseline=0.03, chromatic=0.01, modes=modes
            )
        surface = _Surface(data, S_MAX)
        log_s = _log_s(surface, 3)
        if at_low_edge:
            log_s = np.full_like(log_s, surface._low)
        chi2_ref, own_ref = (
            onp.asarray(a) for a in _whiten_at(surface, data, TRUTH)(log_s)
        )
        data32 = cast_tree(data, "float32")
        model32 = cast_tree(TRUTH, "float32")
    with run_in("float32"):
        surface = _Surface(data32, S_MAX)
        args = _prediction_and_errors(model32, data32, {})
        chi2, own = (
            onp.asarray(a)
            for a in surface._at(*args, data32)(log_s.astype("float32"))
        )
    assert chi2.dtype == onp.float32 and onp.isfinite(chi2).all()
    # The Schur complement comes from Gram sums, so a spanning mode within
    # 1e-3 of the per-frame span loses digits in float32 (1e-3 in χ², 0.07 in
    # the normalizer; the rank-one path loses 1e-6); float64 is unaffected.
    rtol, atol = (5e-3, 0.2) if case == "collinear" else (1e-5, 1e-4)
    assert onp.allclose(chi2, chi2_ref, rtol=rtol, atol=0)
    assert onp.allclose(own, own_ref, rtol=1e-5, atol=atol)


def test_arrow_marginal_is_the_dense_cholesky_marginal():
    rng = onp.random.default_rng(0)
    x, u = rng.normal(size=40), rng.normal(size=(40, 5))
    with run_in("float64"):
        problem = arrow_problem(x, onp.arange(40)[None], u[None])
        for s in (0.2, 1.0, 3.0):
            chi2, norm, _ = arrow_at(problem, s**2)
            white, half = whiten_cholesky(x / s, u / s)
            assert float(chi2) == pytest.approx(
                float(white @ white), rel=1e-10
            )
            assert float(norm) == pytest.approx(float(half), rel=1e-10)


def test_the_memory_model_bounds_the_compiled_working_set():
    """``_table_floats`` counts the arrays of a gradient through the table:
    it must not undercount the compiler's temporaries, nor exceed them by an
    order of magnitude."""
    data = _night("gains_offsets")

    def f(x):
        model = BinaryModelCartesian(x[0], x[1], x[2])
        surface = _Surface(data, S_MAX)
        return surface.score(
            surface.chi2(model, data, None, 1.0), np.ones(2), data
        )

    xs = np.tile(np.asarray([6.2, -3.9, 0.31]), (4, 1))
    compiled = jax.jit(jax.vmap(jax.value_and_grad(f))).lower(xs).compile()
    temp = compiled.memory_analysis().temp_size_in_bytes
    model = 4 * 4 * _table_floats(data)  # float32, four points
    assert temp <= model < 20 * temp
    assert _N_NODES == 32


def test_the_hessian_memory_model_bounds_the_compiled_working_set():
    """``_hessian_groups`` budgets ``8 · _N_SITES · _table_floats`` bytes per
    night (float64, a Hessian over the orbit's 8 sites and the diameter)."""
    with run_in("float64"):
        data = _night("gains_offsets")

        def f(x):
            shift = 1e-3 * np.sum(x[3:])  # every coordinate reaches the table
            model = BinaryModelCartesian(x[0] + shift, x[1], x[2])
            surface = _Surface(data, S_MAX)
            return surface.score(
                surface.chi2(model, data, None, 1.0), np.ones(2), data
            )

        x = np.asarray([6.2, -3.9, 0.31] + [0.1] * (_N_SITES - 3))
        compiled = jax.jit(jax.hessian(f)).lower(x).compile()
        temp = compiled.memory_analysis().temp_size_in_bytes
        assert temp <= 8 * _N_SITES * _table_floats(data) < 20 * temp


def test_an_all_padding_block_adds_nothing():
    """A block whose rows are all padded contributes no χ² and no log det,
    as in ``whiten_blocks``."""
    rng = onp.random.default_rng(1)
    n, m, k = 24, 3, 2
    x_, span_ = rng.normal(size=n), rng.normal(size=(n, k))
    local = rng.normal(size=(3, 8, m))
    rows = onp.arange(n).reshape(3, 8)
    rows[1] = n  # all padding
    local[1] = rng.normal(size=(8, m))  # nonzero columns on padded rows
    rows_j = rows
    with run_in("float64"):
        x, span = np.asarray(x_), np.asarray(span_)
        rows_j, local_j = np.asarray(rows), np.asarray(local)
        for s in (0.5, 1.0, 2.0):
            problem = arrow_problem(x / s, rows_j, local_j / s, span / s)
            chi2, norm, _ = arrow_at(problem, 1.0)
            white, extra = whiten_blocks(
                x / s, rows_j, local_j / s, span / s
            )
            assert float(chi2) == pytest.approx(
                float(white @ white), rel=1e-9
            )
            assert float(norm) == pytest.approx(float(extra.sum()), rel=1e-9)
