"""The damped-Newton kernel of `virgil._gauss_newton`.

Everything here is tiny (2 to 6 coordinates, seconds on a laptop): linear and
Rosenbrock least squares, an iteratively reweighted (IRLS) marginal
objective that is not a sum of squares, boxes, and the `step`/`solve`
contracts (vmap, scan, resume, monotone Φ, stop codes). See
design/history/gauss_newton_plan.md, GN1.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.optimize import minimize

from virgil._gauss_newton import (
    CONVERGED,
    LIMIT,
    NON_FINITE,
    RUNNING,
    STALLED,
    State,
    curvature,
    init,
    solve,
    step,
)


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


class LeastSquares(eqx.Module):
    """Φ = ½|ρ|² with ρ = A z − b, a linear problem."""

    A: jax.Array
    b: jax.Array

    def rows(self, z):
        return self.A @ z - self.b

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Rosenbrock(eqx.Module):
    def rows(self, z):
        return np.stack([10.0 * (z[1] - z[0] ** 2), 1.0 - z[0]])

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Quadratic(eqx.Module):
    """Φ = ½ (z − c)ᵀ H (z − c) with H = LᵀL, curvature rows L (z − c)."""

    L: jax.Array
    c: jax.Array

    def rows(self, z):
        return self.L @ (z - self.c)

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class Marginal(eqx.Module):
    """Φ = Σ_b (ν_b/2) ln χ²_b for two blocks of a linear model.

    Not a sum of squares. The rows carry the IRLS weight ν_b/χ²_b, frozen,
    so JᵀJ is the Gauss–Newton curvature of Φ (the majorise–minimise form).
    """

    A: tuple
    y: tuple

    def _chi2(self, z):
        return [np.sum((A @ z - y) ** 2) for A, y in zip(self.A, self.y)]

    def loss(self, z):
        return sum(
            0.5 * y.size * np.log(c) for y, c in zip(self.y, self._chi2(z))
        )

    def rows(self, z):
        out = []
        for A, y, c in zip(self.A, self.y, self._chi2(z)):
            w = jax.lax.stop_gradient(y.size / c)
            out.append(np.sqrt(w) * (A @ z - y))
        return np.concatenate(out)

    def diag(self, z):
        return np.zeros_like(z)


class Ignores(eqx.Module):
    """Rows that ignore coordinate 1, with P = 0 there."""

    def rows(self, z):
        return z[:1] - 1.0

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


class MissingTerm(eqx.Module):
    """Φ has a ½ z₁² term that neither ``rows`` nor ``diag`` carries."""

    def rows(self, z):
        return z[:1] - 1.0

    def loss(self, z):
        return 0.5 * (z[0] - 1.0) ** 2 + 0.5 * z[1] ** 2

    def diag(self, z):
        return np.zeros_like(z)


class LogSum(eqx.Module):
    """Φ = Σ log z: not finite for z ≤ 0."""

    def rows(self, z):
        return z

    def loss(self, z):
        return np.sum(np.log(z))

    def diag(self, z):
        return np.zeros_like(z)


class NoRows(eqx.Module):
    """A pure-prior problem: no rows at all, P = 1."""

    def rows(self, z):
        return np.zeros((0,), z.dtype)

    def loss(self, z):
        return 0.5 * np.sum(z**2)

    def diag(self, z):
        return np.ones_like(z)


def _no_bounds(n):
    return np.full(n, -np.inf), np.full(n, np.inf)


def test_linear_least_squares_matches_lstsq_in_five_steps():
    rng = onp.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    b = rng.normal(size=30)
    state = solve(
        LeastSquares(np.asarray(A), np.asarray(b)),
        np.zeros(4),
        gtol=1e-10,
        max_steps=5,
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 5
    onp.testing.assert_allclose(
        state.z, onp.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10
    )


def test_rosenbrock_residuals_reach_the_minimum_in_fifty_steps():
    state = solve(
        Rosenbrock(), np.array([-1.2, 1.0]), gtol=1e-10, max_steps=50
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 50
    onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-8)


def _marginal_problem():
    rng = onp.random.default_rng(1)
    A = (rng.normal(size=(20, 3)), rng.normal(size=(15, 3)))
    z_true = onp.array([1.0, -2.0, 0.5])
    # the second block is five times noisier, so the weights are far from 1
    y = (
        A[0] @ z_true + 0.1 * rng.normal(size=20),
        A[1] @ z_true + 0.5 * rng.normal(size=15),
    )
    return Marginal(tuple(map(np.asarray, A)), tuple(map(np.asarray, y)))


def test_marginal_objective_with_irls_rows_matches_scipy_bfgs():
    problem = _marginal_problem()
    z0 = np.zeros(3)
    # (gtol=1e-12 is below what Φ resolves; the decrement test ends the run.)
    state = solve(problem, z0, gtol=1e-12, dtol=1e-12, max_steps=100)
    assert int(state.stop) == CONVERGED
    ref = minimize(
        lambda z: float(problem.loss(np.asarray(z))),
        onp.zeros(3),
        jac=lambda z: onp.asarray(jax.grad(problem.loss)(np.asarray(z))),
        method="BFGS",
        options={"gtol": 1e-13},
    )
    onp.testing.assert_allclose(state.z, ref.x, atol=1e-8)
    assert float(state.value) <= ref.fun + 1e-12


def test_curvature_is_jtj_plus_diag_and_handles_no_rows():
    problem = _marginal_problem()
    z = np.array([0.3, -1.0, 0.2])
    J = jax.jacfwd(problem.rows)(z)
    onp.testing.assert_allclose(curvature(problem, z), J.T @ J, atol=1e-12)
    F = curvature(NoRows(), np.ones(3))
    onp.testing.assert_allclose(F, onp.eye(3))
    state = solve(NoRows(), np.ones(3), gtol=1e-12, max_steps=20)
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, 0.0, atol=1e-10)


def test_bound_holds_the_coordinate_whose_minimum_is_outside_the_box():
    problem = Quadratic(np.diag(np.array([1.0, 2.0])), np.array([2.0, 0.5]))
    state = solve(
        problem,
        np.array([0.2, 0.2]),
        lower=0.0,
        upper=1.0,
        gtol=1e-10,
        max_steps=30,
    )
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 0.5], atol=1e-9)
    # held at the upper bound: the gradient points out of the box
    assert float(state.grad[0]) < 0


def test_coordinate_on_a_bound_with_inward_gradient_leaves_it():
    problem = Quadratic(np.eye(2), np.array([1.0, 1.0]))
    state = solve(
        problem,
        np.array([0.0, 0.0]),
        lower=np.array([0.0, 0.0]),
        upper=np.array([5.0, 5.0]),
        gtol=1e-10,
        max_steps=20,
    )
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-9)


def test_strongly_coupled_coordinates_with_one_at_a_bound_do_not_cycle():
    rho = 0.99
    H = onp.array([[1.0, rho], [rho, 1.0]])
    L = onp.linalg.cholesky(H).T
    c = onp.array([-1.0, 2.0])
    problem = Quadratic(np.asarray(L), np.asarray(c))
    state = solve(
        problem,
        np.array([0.5, 5.0]),
        lower=np.array([0.0, -np.inf]),
        upper=np.array([np.inf, np.inf]),
        gtol=1e-10,
        max_steps=20,
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 20
    # coordinate 0 held at its bound; coordinate 1 minimizes over the rest
    onp.testing.assert_allclose(state.z, [0.0, c[1] + rho * c[0]], atol=1e-9)
    assert float(state.grad[0]) > 0
    assert abs(float(state.grad[1])) < 1e-9


def _starts():
    rng = onp.random.default_rng(2)
    return np.asarray(rng.uniform(-2, 2, size=(8, 2)))


def test_vmap_over_starts_equals_a_loop():
    problem = Rosenbrock()
    z0s = _starts()

    def run(z0):
        return solve(problem, z0, gtol=1e-10, max_steps=100)

    batched = jax.vmap(run)(z0s)
    for k in range(8):
        one = run(z0s[k])
        onp.testing.assert_allclose(batched.z[k], one.z, atol=1e-12, rtol=0)
        onp.testing.assert_allclose(
            batched.value[k], one.value, atol=1e-12, rtol=0
        )
        assert int(batched.steps[k]) == int(one.steps)
        assert int(batched.stop[k]) == int(one.stop)


def test_scan_of_steps_equals_solve_with_the_same_limit():
    problem, z0 = Rosenbrock(), np.array([-1.2, 1.0])
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)

    def body(state, _):
        return step(
            problem, state, lower, upper, tol, tol, np.int32(20)
        ), state.value

    scanned, _ = jax.lax.scan(
        body, init(problem, z0, lower, upper), None, length=20
    )
    solved = solve(problem, z0, max_steps=20)
    assert int(solved.stop) == LIMIT
    for a, b in zip(jax.tree.leaves(scanned), jax.tree.leaves(solved)):
        onp.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_resuming_after_seven_steps_equals_an_uninterrupted_solve():
    problem, z0 = Rosenbrock(), np.array([-1.2, 1.0])
    first = solve(problem, z0, gtol=1e-10, max_steps=7)
    assert int(first.stop) == LIMIT
    assert int(first.steps) == 7
    resumed = solve(problem, first, gtol=1e-10, max_steps=93)
    whole = solve(problem, z0, gtol=1e-10, max_steps=100)
    assert int(resumed.stop) == int(whole.stop) == CONVERGED
    assert isinstance(resumed, State)
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole)):
        onp.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def test_loss_never_increases_over_a_run():
    problem = Rosenbrock()
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)

    def body(state, _):
        new = step(problem, state, lower, upper, tol, tol, np.int32(60))
        return new, new.value

    start = init(problem, np.array([-1.2, 1.0]), lower, upper)
    _, values = jax.lax.scan(body, start, None, length=60)
    values = onp.concatenate([[float(start.value)], onp.asarray(values)])
    assert onp.all(onp.diff(values) <= 0)
    assert values[-1] < values[0]


def test_step_is_the_identity_once_stopped():
    problem = Rosenbrock()
    lower, upper = _no_bounds(2)
    tol = np.asarray(0.0)
    done = solve(problem, np.array([-1.2, 1.0]), gtol=1e-10, max_steps=100)
    again = step(problem, done, lower, upper, tol, tol, np.int32(1000))
    for a, b in zip(jax.tree.leaves(again), jax.tree.leaves(done)):
        onp.testing.assert_array_equal(a, b)


def test_non_finite_start_stops_with_non_finite():
    state = solve(LogSum(), np.array([-1.0, 1.0]), max_steps=10)
    assert int(state.stop) == NON_FINITE
    assert int(state.steps) == 0


def test_a_direction_the_rows_ignore_gives_no_nan_and_converges_in_the_others():
    state = solve(Ignores(), np.array([3.0, 7.0]), gtol=1e-10, max_steps=20)
    assert int(state.stop) == CONVERGED
    assert onp.all(onp.isfinite(onp.asarray(state.z)))
    assert onp.all(onp.isfinite(onp.asarray(state.curvature)))
    onp.testing.assert_allclose(state.z, [1.0, 7.0], atol=1e-9)


def test_float32_rosenbrock_ends_converged_not_stalled():
    with jax.enable_x64(False):
        state = solve(Rosenbrock(), np.array([-1.2, 1.0], dtype=np.float32))
        assert state.z.dtype == np.float32
        assert int(state.stop) == CONVERGED
        assert int(state.stop) not in (STALLED, RUNNING)
        onp.testing.assert_allclose(state.z, [1.0, 1.0], atol=1e-3)


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_rounding_floor_converges_with_a_nonzero_residual(x64):
    """A decrement tolerance converges at the rounding floor of Φ, not LIMIT.

    Rosenbrock has zero residual at the optimum, so it can stop on g == 0.
    Noisy least squares cannot: Φ only changes by rounding noise there, and
    ``dtol`` is raised to that noise. With the decrement test off
    (``dtol=0``) the same run stalls and says so, whatever its decrement
    (585#6: it used to be promoted to CONVERGED).
    """
    rng = onp.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    b = rng.normal(size=30)
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        problem = LeastSquares(np.asarray(A, dtype), np.asarray(b, dtype))
        state = solve(problem, np.zeros(4, dtype), dtol=1e-12)
        assert state.z.dtype == dtype
        assert int(state.stop) == CONVERGED
        assert int(state.steps) < 200
        ref = onp.linalg.lstsq(A, b, rcond=None)[0]
        onp.testing.assert_allclose(state.z, ref, atol=1e-10 if x64 else 1e-4)
        off = solve(problem, np.zeros(4, dtype))
        assert int(off.stop) == STALLED
        assert float(off.lam) > 1e10
        # dtol=None is the rounding resolution of Φ, reachable by construction
        auto = solve(problem, np.zeros(4, dtype), dtol=None)
        assert int(auto.stop) == CONVERGED
        # it stops within the decrement Φ resolves of the minimum, no closer
        from virgil._gauss_newton import rounding_dtol

        phi_ref = problem.loss(np.asarray(ref, dtype))
        assert float(auto.value - phi_ref) <= 2 * float(
            rounding_dtol(auto.value)
        )


def test_rounding_dtol_is_ten_ulp_of_phi_with_a_one_nat_floor():
    from virgil._gauss_newton import rounding_dtol

    for dtype in (np.float32, np.float64):
        eps = float(np.finfo(dtype).eps)
        assert float(rounding_dtol(dtype(0.1))) == pytest.approx(10 * eps)
        assert float(rounding_dtol(dtype(-5000.0))) == pytest.approx(5e4 * eps)


def test_a_coordinate_just_inside_a_bound_is_held_not_clipped_and_rejected():
    """The case ε is for: z₀ sits 1e-11 inside its bound with an outward gradient.

    Coordinate 1 is already at its conditional optimum, but the unconstrained
    Newton step moves coordinate 0 outward; clipped, it is worse than the
    start. With ε = 0 the coordinate is free, and the run needs 17 steps (9
    rejected); with ε it is held and converges at once.
    """
    rho = 0.99
    H = onp.array([[1.0, rho], [rho, 1.0]])
    problem = Quadratic(
        np.asarray(onp.linalg.cholesky(H).T), np.array([-1.0, 2.0])
    )
    state = solve(
        problem,
        np.array([1e-11, 1.01]),
        lower=np.array([0.0, -np.inf]),
        gtol=1e-8,
        max_steps=20,
    )
    assert int(state.stop) == CONVERGED
    assert int(state.steps) <= 3
    assert int(state.rejects) <= 1
    onp.testing.assert_allclose(state.z, [1e-11, 1.01], atol=1e-9)


def test_a_coordinate_just_inside_a_bound_in_float32_uses_a_float32_epsilon():
    """Away from 0, 1e-10 is below float32 resolution: ε scales with the dtype."""
    rho = 0.99
    H = onp.array([[1.0, rho], [rho, 1.0]])
    with jax.enable_x64(False):
        f32 = np.float32
        problem = Quadratic(
            np.asarray(onp.linalg.cholesky(H).T, f32),
            np.asarray([0.0, 2.0], f32),
        )
        state = solve(
            problem,
            np.array([1.0000001, 1.01], f32),
            lower=np.array([1.0, -np.inf], f32),
            gtol=1e-4,
            max_steps=30,
        )
        assert int(state.stop) == CONVERGED
        assert int(state.steps) <= 3
        assert int(state.rejects) <= 1


def test_a_gradient_without_curvature_is_damped_to_the_right_answer():
    """A term of Φ missing from rows and diag (zero curvature in coordinate 1).

    A zero diagonal takes unit Marquardt scaling, so the damped step there is
    a gradient step: it is accepted once λ is large enough and the run
    reaches the true minimum [1, 0] instead of stalling (it used to stall:
    the old floor made the step ~1/(λ·1e-12) long). Never a wrong answer.
    """
    start = np.array([3.0, 7.0])
    state = solve(MissingTerm(), start, gtol=1e-10, max_steps=200)
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 0.0], atol=1e-8)
    assert float(state.value) <= float(MissingTerm().loss(start))


class TinyDiagonal(MissingTerm):
    """As `MissingTerm`, but a diagonal of 1e-30 stands for the missing term."""

    def diag(self, z):
        return np.asarray([0.0, 1e-30])


def test_a_tiny_but_nonzero_curvature_keeps_its_tiny_damping_and_stalls():
    """The unit scaling is for an exactly zero diagonal only (plain Marquardt,
    no Moré running max): the step in coordinate 1 is ~g/(1e-30·λ), every
    trial is rejected, and the run says STALLED rather than claim success."""
    state = solve(TinyDiagonal(), np.array([3.0, 7.0]), gtol=1e-10)
    assert int(state.stop) == STALLED
    assert onp.all(onp.isfinite(onp.asarray(state.z)))


class WrongGradient(eqx.Module):
    """Φ = ½(z₀−1)² + ½z₁², but the z₁ gradient has the wrong sign."""

    def rows(self, z):
        return z[:1] - 1.0

    def loss(self, z):
        @jax.custom_jvp
        def uphill(w):
            return 0.5 * w**2

        @uphill.defjvp
        def _(primals, tangents):
            (w,), (t,) = primals, tangents
            return 0.5 * w**2, -w * t

        return 0.5 * (z[0] - 1.0) ** 2 + uphill(z[1])

    def diag(self, z):
        return np.zeros_like(z)


def test_a_gradient_that_disagrees_with_phi_stalls_with_a_large_decrement():
    """Every trial along a wrong gradient is rejected: STALLED, never CONVERGED."""
    state = solve(WrongGradient(), np.array([3.0, 7.0]), dtol=1e-12)
    assert int(state.stop) == STALLED
    assert onp.all(onp.isfinite(onp.asarray(state.z)))


class Scaled(eqx.Module):
    """ρ = (a(z₀−1), b(z₁−1)): curvatures a², b² 18 orders of magnitude apart."""

    def rows(self, z):
        return np.stack([1e6 * (z[0] - 1.0), 1e-3 * (z[1] - 1.0)])

    def loss(self, z):
        return 0.5 * np.sum(self.rows(z) ** 2)

    def diag(self, z):
        return np.zeros_like(z)


def test_marquardt_scaling_is_per_parameter_for_mixed_units():
    """585#4: a global floor max(diag)·1e-12 over-damped the small-unit
    coordinate (floor 1 against its curvature 1e-6), so it crawled.
    Per-parameter scaling takes the Newton step in both coordinates."""
    state = solve(Scaled(), np.array([2.0, 5000.0]), dtol=1e-12, max_steps=12)
    assert int(state.stop) == CONVERGED
    onp.testing.assert_allclose(state.z, [1.0, 1.0], rtol=1e-6)
