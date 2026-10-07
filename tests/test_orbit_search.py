"""``score_orbits``: candidate orbits scored on all epochs with the flux
shared, error scales marginalized, gains profiled and extra terms added
(PR D1 of the automatic-orbits design).

Every dataset is tiny (one VLTI frame or a few, a few channels). Noise is
drawn with NumPy, so that the data do not depend on the JAX version.
"""

import jax
import numpy as onp
import pytest

from virgil._precision import cast_tree

from virgil.coverage import VLTI_UTS, vlti_oidata
from virgil.epochs import Epochs, marginal_loglike
from virgil.likelihood import whitened_residuals
from virgil.models import BinaryModelCartesian, OrbitalBinary
from virgil.orbit_search import SharedFlux, rank_scores, score_orbits
from virgil.orbits import (
    KeplerOrbit,
    RVData,
    ThieleInnesOrbit,
    _unit_orbit,
)
from virgil.simulate import simulate

pytest.importorskip("jaxoplanet")

T_REF = 60000.0
TRUTH = KeplerOrbit(300.0, 50.0, 0.3, 50.0, 60.0, 120.0, 8.0, t_ref=T_REF)
OTHER = KeplerOrbit(300.0, 120.0, 0.3, 50.0, 60.0, 40.0, 8.0, t_ref=T_REF)
NIGHTS = (60000.0, 60060.0, 60150.0)
WAVELENGTHS = (2.0e-6, 2.2e-6, 2.4e-6)


def _night(
    orbit,
    flux,
    night,
    seed,
    *,
    stations=VLTI_UTS[:3],
    hours=(0.0,),
    sigma_v2=0.02,
    sigma_cp_deg=2.0,
    gain=1.0,
):
    """A night of V² and closure phases, with noise at the quoted errors."""
    template = vlti_oidata(
        stations=stations,
        hour_angles_h=hours,
        wavelengths_m=onp.asarray(WAVELENGTHS),
        sigma_v2=sigma_v2,
        sigma_cp_deg=sigma_cp_deg,
        nights_mjd=[night],
    )
    with jax.enable_x64(True):
        model = simulate(OrbitalBinary(orbit, flux), template)
    rng = onp.random.default_rng(seed)
    vis, d_vis = onp.asarray(model.vis), onp.asarray(model.d_vis)
    phi, d_phi = onp.asarray(model.phi), onp.asarray(model.d_phi)
    vis = gain * vis + d_vis * rng.standard_normal(vis.shape)
    phi = phi + d_phi * rng.standard_normal(phi.shape)
    return model.set(["vis", "phi"], [vis, phi])


def _epochs(orbit=TRUTH, flux=0.4, nights=NIGHTS, **options):
    return Epochs(
        {
            f"n{k}": _night(orbit, flux, night, seed=k, **options)
            for k, night in enumerate(nights)
        }
    )


def _f64(tree):
    """``tree`` in float64 (call inside ``jax.enable_x64``)."""
    return cast_tree(tree, "float64")


def _snapshot(orbit, mjd, flux):
    dra, ddec, _ = _f64(orbit).relative(onp.float64(mjd))
    return BinaryModelCartesian(float(dra), float(ddec), flux)


def _summed_marginal(epochs, orbit, flux):
    """Σ marginal_loglike over datasets with one time each."""
    with jax.enable_x64(True):
        return sum(
            float(
                marginal_loglike(
                    _snapshot(orbit, onp.mean(d.mjd), flux), _f64(d)
                )
            )
            for d in epochs.data
        )


def test_equals_summed_marginal_loglike_at_fixed_flux():
    epochs = _epochs()
    scores = score_orbits(
        epochs, OrbitalBinary, [TRUTH, OTHER], shared=SharedFlux([0.4])
    )
    for k, orbit in enumerate((TRUTH, OTHER)):
        expect = _summed_marginal(epochs, orbit, 0.4)
        assert scores.score[k] == pytest.approx(expect, rel=1e-9, abs=1e-7)
        assert scores.profiled[k] == pytest.approx(expect, rel=1e-9)
    assert scores.score[0] > scores.score[1] + 10.0
    assert scores.flux[:, 0] == pytest.approx([0.4, 0.4])
    assert onp.all(onp.isnan(scores.flux_err))


def test_several_times_per_epoch_equal_the_split_with_a_tied_scale():
    # A 3-day orbit moves ~2 mas over the night, about λ/2B: sample-time
    # evaluation matters. The comparator splits the night into one dataset
    # per frame (one time each, so a snapshot there is exact) and ties the
    # error scale across the split: the χ² of each block adds over the
    # frames before m = -Σ_b (ν_b/2) ln χ²_b.
    fast = KeplerOrbit(3.0, 0.4, 0.1, 40.0, 30.0, 70.0, 6.0, t_ref=T_REF)
    night = _night(fast, 0.4, T_REF, seed=7, hours=(-2.0, 0.0, 2.0))
    epochs = Epochs({"night": night})
    got = score_orbits(epochs, OrbitalBinary, [fast], shared=SharedFlux([0.4]))
    parts = night.split_by_epoch(gap_days=0.01)
    assert len(parts) == 3
    chi2 = onp.zeros(2)
    nu = onp.zeros(2)
    with jax.enable_x64(True):
        for part in parts:
            (time,) = onp.unique(part.mjd)
            r = onp.asarray(
                whitened_residuals(_snapshot(fast, time, 0.4), _f64(part))
            )
            n_vis = onp.asarray(part.vis).size
            chi2 += [onp.sum(r[:n_vis] ** 2), onp.sum(r[n_vis:] ** 2)]
            nu += [n_vis, r.size - n_vis]
    expect = -0.5 * onp.sum(nu * onp.log(chi2))
    assert got.score[0] == pytest.approx(expect, rel=1e-9)
    # One snapshot at the mean time is measurably different.
    snapshot = _summed_marginal(epochs, fast, 0.4)
    assert abs(snapshot - expect) > 1.0


def _brute(epochs, orbit, log_f, beta=(0.0,), wavel0=None, s_max=None):
    """The summed marginal_loglike on a dense (ln f, β) grid (float64), one
    β at a time (to keep memory small)."""
    with jax.enable_x64(True):
        positions = [
            [float(x) for x in _f64(orbit).relative(onp.mean(d.mjd))[:2]]
            for d in epochs.data
        ]
        datasets = _f64(epochs.data)
        ratios = [
            onp.broadcast_to(onp.asarray(d.wavel, float), onp.shape(d.u))
            / (wavel0 or 1.0)
            for d in epochs.data
        ]

        @jax.jit
        def total(lf, b):
            out = 0.0
            for p, d, r in zip(positions, datasets, ratios):
                f = jax.numpy.exp(lf) * r**b
                out = out + marginal_loglike(
                    BinaryModelCartesian(*p, f), d, s_max=s_max
                )
            return out

        column = jax.jit(jax.vmap(total, (0, None)))
        log_f = onp.asarray(log_f)
        return onp.stack(
            [onp.asarray(column(log_f, b)) for b in onp.asarray(beta)], axis=1
        )


def _log_mean_exp(values, x, axis=-1):
    """log of the mean of exp(values) over the range of x (trapezoid)."""
    peak = onp.max(values)
    integral = onp.trapezoid(onp.exp(values - peak), x, axis=axis)
    return peak + onp.log(integral / (x[-1] - x[0]))


# A well-measured flux (σ_ln f ≈ 0.02) whose true value falls at, between
# and next to the default grid's points (1e-3 to 1, 16 log-spaced).
@pytest.mark.parametrize("flux", [0.215, 0.28, 0.33, 0.4, 0.46])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_flux_marginal_matches_quadrature_wherever_the_peak_falls(flux, dtype):
    epochs = _epochs(flux=flux, sigma_v2=0.1, sigma_cp_deg=10.0)
    scores = score_orbits(epochs, OrbitalBinary, [TRUTH], dtype=dtype)
    log_f = onp.linspace(onp.log(1e-3), 0.0, 20001)
    values = _brute(epochs, TRUTH, log_f)[:, 0]
    assert scores.score[0] == pytest.approx(
        _log_mean_exp(values, log_f), abs=0.05
    )
    assert not scores.fallback[0, 0]
    assert scores.profiled[0] == pytest.approx(values.max(), abs=1e-2)
    best = onp.exp(log_f[onp.argmax(values)])
    assert scores.flux[0, 0] == pytest.approx(best, rel=2e-3)
    sigma = scores.flux_err[0, 0] / scores.flux[0, 0]
    assert 0.005 < sigma < 0.2
    assert not scores.flux_at_edge[0, 0]


@pytest.mark.parametrize("flux", [0.97, 1.3])
def test_flux_marginal_near_the_reference_bound(flux):
    # f <= 1 in the reference band: a peak just inside the bound, and one
    # beyond it (the profile then sits on the bound with a slope).
    epochs = _epochs(flux=flux, sigma_v2=0.1, sigma_cp_deg=10.0)
    scores = score_orbits(epochs, OrbitalBinary, [TRUTH])
    log_f = onp.linspace(onp.log(1e-3), 0.0, 20001)
    values = _brute(epochs, TRUTH, log_f)[:, 0]
    assert scores.score[0] == pytest.approx(
        _log_mean_exp(values, log_f), abs=0.05
    )
    assert not scores.fallback[0, 0]
    assert scores.flux_at_edge[0, 0] == (flux > 1.0)


def test_flux_marginal_when_the_prior_dominates():
    # A faint companion in noisy data: the likelihood is nearly flat in
    # ln f over most of the prior.
    epochs = _epochs(flux=0.003, sigma_v2=0.3, sigma_cp_deg=30.0)
    for dtype in ("float64", "float32"):
        scores = score_orbits(epochs, OrbitalBinary, [TRUTH], dtype=dtype)
        log_f = onp.linspace(onp.log(1e-3), 0.0, 20001)
        values = _brute(epochs, TRUTH, log_f)[:, 0]
        assert values.max() - values.min() > 1.0  # not exactly flat
        assert scores.score[0] == pytest.approx(
            _log_mean_exp(values, log_f), abs=0.05
        )


def test_flux_and_slope_marginal_matches_quadrature():
    epochs = _epochs(flux=0.33, sigma_v2=0.1, sigma_cp_deg=10.0)
    wavel0 = 2.2e-6
    shared = SharedFlux(slope=(-3.0, 3.0, 7), wavel0=wavel0)
    scores = score_orbits(epochs, OrbitalBinary, [TRUTH], shared=shared)
    # Brute force about the peak; the rest of the prior is negligible.
    lf0 = onp.log(scores.flux[0, 0])
    log_f = onp.linspace(lf0 - 0.8, min(lf0 + 0.8, 0.0), 1601)
    beta = onp.linspace(-3.0, 3.0, 601)
    values = _brute(epochs, TRUTH, log_f, beta, wavel0)
    edges = (values[0], values[-1], values[:, 0], values[:, -1])
    assert values.max() - max(e.max() for e in edges) > 15.0
    inner = _log_mean_exp(values, beta, axis=1) + onp.log(6.0)
    expect = _log_mean_exp(inner, log_f) + onp.log(
        (log_f[-1] - log_f[0]) / (6.0 * -onp.log(1e-3))
    )
    assert scores.score[0] == pytest.approx(expect, abs=0.05)
    assert not scores.fallback[0, 0]
    assert 0.0 < scores.slope_err[0, 0] < 3.0


def test_rv_term_adds_its_loglike():
    epochs = _epochs()
    mjd = onp.array([60010.0, 60100.0, 60200.0])
    rv = RVData(mjd, [1.0, -2.0, 0.5], 0.5, t_ref=T_REF)
    term = rv.term(lambda v: (v["orbit"], 0.5, 0.0, 50.0))
    shared = SharedFlux((0.1, 1.0, 8))
    base = score_orbits(epochs, OrbitalBinary, [TRUTH, OTHER], shared=shared)
    with_rv = score_orbits(
        epochs,
        OrbitalBinary,
        [TRUTH, OTHER],
        shared=shared,
        terms=[term, lambda o: rv.loglike(o, 0.5, 0.0, 50.0, jitter=1.0)],
    )
    with jax.enable_x64(True):
        rv64 = _f64(rv)
        for k, orbit in enumerate(_f64((TRUTH, OTHER))):
            expect = [
                float(rv64.loglike(orbit, 0.5, 0.0, 50.0)),
                float(rv64.loglike(orbit, 0.5, 0.0, 50.0, jitter=1.0)),
            ]
            # The callable sees the float32 RVData: agreement to 1e-6.
            assert with_rv.terms[k] == pytest.approx(expect, rel=1e-6)
            assert with_rv.score[k] - base.score[k] == pytest.approx(
                sum(expect), rel=1e-6
            )


def test_independent_of_batch_size():
    epochs = _epochs()
    candidates = [TRUTH, OTHER, TRUTH]
    shared = SharedFlux((0.1, 1.0, 8))
    every = score_orbits(epochs, OrbitalBinary, candidates, shared=shared)
    batched = score_orbits(
        epochs, OrbitalBinary, candidates, shared=shared, batch_size=2
    )
    # Batching changes how XLA fuses the Newton steps: equal to rounding.
    onp.testing.assert_allclose(batched.score, every.score, rtol=1e-10)
    onp.testing.assert_allclose(batched.flux, every.flux, rtol=1e-6)


def test_float32_agrees_with_float64_and_ties_rank_by_index():
    epochs = _epochs()
    candidates = [OTHER, TRUTH, TRUTH]
    shared = SharedFlux((0.1, 1.0, 8))
    f64 = score_orbits(epochs, OrbitalBinary, candidates, shared=shared)
    f32 = score_orbits(
        epochs, OrbitalBinary, candidates, shared=shared, dtype="float32"
    )
    onp.testing.assert_allclose(f32.score, f64.score, rtol=1e-4)
    onp.testing.assert_allclose(f32.flux, f64.flux, rtol=1e-3)
    # The two copies of the truth tie: by index, in both precisions.
    assert f64.order().tolist() == [1, 2, 0]
    assert f32.order().tolist() == [1, 2, 0]
    assert rank_scores([1.0, 1.0 + 4e-4, onp.nan, 1.0 - 4e-4]).tolist() == [
        0,
        1,
        3,
        2,
    ]


def _through(t0, r0, t1, r1, period=300.0, dt_peri=50.0, ecc=0.3):
    """The orbit through positions r0 at t0 and r1 at t1 (Thiele–Innes)."""
    with jax.enable_x64(True):
        x, y = _unit_orbit(onp.array([t0, t1]) - T_REF, period, dt_peri, ecc)
    m = onp.stack([onp.asarray(x), onp.asarray(y)], -1)
    b, g = onp.linalg.solve(m, [r0[0], r1[0]])
    a, f = onp.linalg.solve(m, [r0[1], r1[1]])
    return ThieleInnesOrbit(
        period, dt_peri, ecc, a, b, f, g, t_ref=T_REF
    ).to_kepler()


def test_per_epoch_flux_decoy_loses_under_shared_flux():
    # A companion at -r with flux 1/f has the same V² and closure phases as
    # one at r with flux f. The decoy follows the truth on night 0 and its
    # mirror image on night 1: with a flux per night (and night 1 allowed
    # f > 1) it fits exactly as well as the truth, but no single flux fits
    # both nights. The grids are fixed and symmetric in ln f.
    nights = (60000.0, 60060.0)
    epochs = _epochs(flux=0.3, nights=nights)
    with jax.enable_x64(True):
        r = [onp.asarray(TRUTH.relative(t)[:2], float) for t in nights]
    decoy = _through(nights[0], r[0], nights[1], -r[1])
    per_night = SharedFlux(
        grid={"a": (0.1, 1.0, 11), "b": (0.1, 10.0, 21)},
        bands={"n0": "a", "n1": "b"},
    )
    loose = score_orbits(
        epochs, OrbitalBinary, [TRUTH, decoy], shared=per_night
    )
    assert loose.score[1] == pytest.approx(loose.score[0], abs=1e-3)
    assert loose.flux[1, 1] == pytest.approx(1.0 / loose.flux[0, 1], rel=1e-6)
    shared = score_orbits(
        epochs,
        OrbitalBinary,
        [TRUTH, decoy],
        shared=SharedFlux((0.1, 1.0, 11)),
    )
    assert shared.score[0] > shared.score[1] + 10.0
    assert shared.order().tolist() == [0, 1]


def test_budget_refuses_before_scoring():
    epochs = _epochs()
    shared = SharedFlux((0.1, 1.0, 8))
    cost = score_orbits(
        epochs, OrbitalBinary, [TRUTH, OTHER], shared=shared
    ).cost
    with pytest.raises(ValueError, match=f"max_evaluations={cost - 1}"):
        score_orbits(
            epochs,
            OrbitalBinary,
            [TRUTH, OTHER],
            shared=shared,
            max_evaluations=cost - 1,
        )
    within = score_orbits(
        epochs, OrbitalBinary, [TRUTH, OTHER], shared=shared,
        max_evaluations=cost,
    )  # fmt: skip
    assert within.cost == cost
    fixed = score_orbits(
        epochs, OrbitalBinary, [TRUTH, OTHER], shared=SharedFlux([0.4])
    )
    assert fixed.cost == 2 * 3


def test_reference_band_flux_is_bounded_by_one():
    with pytest.raises(ValueError, match="at most 1"):
        score_orbits(
            _epochs(), OrbitalBinary, [TRUTH], shared=SharedFlux((0.1, 2.0, 4))
        )


def _summed_bounded(epochs, orbit, flux, s_max):
    with jax.enable_x64(True):
        return sum(
            float(
                marginal_loglike(
                    _snapshot(orbit, onp.mean(d.mjd), flux),
                    _f64(d),
                    s_max=s_max,
                )
            )
            for d in epochs.data
        )


def test_s_max_bounds_a_small_dof_block_driven_to_zero_chi2():
    # The closure phases (one triangle, three channels: ν = 3) of one night
    # are noise-free at the true position: χ² → 0, and without s_max the
    # block's -(ν/2) ln χ² is unbounded (a spike). With s_max the score is
    # Σ marginal_loglike(s_max), and the block gains at most about
    # ν ln s_max over a noisy one.
    exact = _night(TRUTH, 0.4, NIGHTS[0], seed=0)
    with jax.enable_x64(True):
        model = simulate(_snapshot(TRUTH, NIGHTS[0], 0.4), _f64(exact))
    exact = exact.set("phi", onp.asarray(model.phi))
    epochs = Epochs({"n0": exact, "n1": _night(TRUTH, 0.4, NIGHTS[1], 1)})
    shared = SharedFlux([0.4])
    free = score_orbits(epochs, OrbitalBinary, [TRUTH], shared=shared)
    s_max = 10.0
    bounded = score_orbits(
        epochs, OrbitalBinary, [TRUTH], shared=shared, s_max=s_max
    )
    expect = _summed_bounded(epochs, TRUTH, 0.4, s_max)
    assert bounded.score[0] == pytest.approx(expect, rel=1e-6, abs=1e-5)
    noisy = _epochs(nights=NIGHTS[:2])
    reference = _summed_bounded(noisy, TRUTH, 0.4, s_max)
    assert bounded.score[0] < reference + 3 * onp.log(s_max) + 5.0
    assert free.score[0] > bounded.score[0] + 30.0


def test_bounded_flux_and_slope_marginal_matches_quadrature():
    # Kept small (one night, V² only, a coarse grid, 8 nodes per side):
    # differentiating the s_max integral inside the 2-D marginal compiles
    # to a large program.
    (night,) = _epochs(
        flux=0.33, nights=NIGHTS[:1], sigma_v2=0.1, sigma_cp_deg=10.0
    ).data
    epochs = Epochs({"n0": night.select(observables=("vis",))})
    wavel0, s_max = 2.2e-6, 10.0
    shared = SharedFlux(
        (0.01, 1.0, 5), slope=(-2.0, 2.0, 3), wavel0=wavel0, order=8, newton=4
    )
    scores = score_orbits(
        epochs, OrbitalBinary, [TRUTH], shared=shared, s_max=s_max
    )
    assert not scores.fallback[0, 0]
    # The whole prior: bounded scales leave the tails (a faint companion
    # fitted with large scales) only a few nats down.
    log_f = onp.linspace(onp.log(0.01), 0.0, 801)
    beta = onp.linspace(-2.0, 2.0, 201)
    values = _brute(epochs, TRUTH, log_f, beta, wavel0, s_max=s_max)
    expect = _log_mean_exp(_log_mean_exp(values, beta, axis=1), log_f)
    assert scores.score[0] == pytest.approx(expect, abs=0.05)


def test_profiled_gain_matches_a_fitted_v2_scale():
    # One gain per night on log|V|: for V² it scales the model V² by
    # (1 + 2γ), so profiling it is a linear least-squares fit of a V² scale,
    # and costs the V² block one degree of freedom.
    epochs = _epochs(gain=1.1)
    gained = Epochs(
        {
            name: d.with_gains(modes=onp.ones((1, onp.size(d.u))))
            for name, d in zip(epochs.dataset_names, epochs.data)
        }
    )
    with pytest.raises(NotImplementedError):
        marginal_loglike(BinaryModelCartesian(1.0, 1.0, 0.4), gained.data[0])
    got = score_orbits(
        gained, OrbitalBinary, [TRUTH], shared=SharedFlux([0.4])
    )
    expect = 0.0
    with jax.enable_x64(True):
        for d in _f64(epochs.data):
            model = onp.asarray(
                d.model(_snapshot(TRUTH, onp.mean(d.mjd), 0.4))
            )
            n_vis = onp.asarray(d.vis).size
            m, sigma = model[:n_vis], onp.asarray(d.d_vis)
            data = onp.asarray(d.vis)
            c = onp.sum(data * m / sigma**2) / onp.sum(m**2 / sigma**2)
            chi2_vis = onp.sum(((data - c * m) / sigma) ** 2)
            r = onp.asarray(
                whitened_residuals(_snapshot(TRUTH, onp.mean(d.mjd), 0.4), d)
            )
            chi2_phi = onp.sum(r[n_vis:] ** 2)
            expect += -0.5 * (n_vis - 1) * onp.log(chi2_vis)
            expect += -0.5 * (r.size - n_vis) * onp.log(chi2_phi)
    assert got.score[0] == pytest.approx(expect, rel=1e-8)
    plain = score_orbits(
        epochs, OrbitalBinary, [TRUTH], shared=SharedFlux([0.4])
    )
    assert got.score[0] > plain.score[0]


def test_profiled_closure_offsets_absorb_offsets_in_the_data():
    # Four telescopes (correlated closure phases): an offset per frame and
    # triangle added to the data barely changes the score once offsets
    # are profiled, but changes it a lot when they are not.
    def night(phi_shift):
        d = _night(
            TRUTH, 0.4, NIGHTS[0], seed=3, stations=VLTI_UTS, sigma_cp_deg=1.0
        )
        return d.set("phi", d.phi + phi_shift)

    clean = night(0.0)
    offsets = clean.with_closure_offsets(triangle=1.0).phase_offsets
    modes = onp.asarray(
        offsets.modes(clean.cp_noise, onp.asarray(clean.phi).size)
    )
    rng = onp.random.default_rng(0)
    shift = modes @ (0.1 * rng.standard_normal(modes.shape[1]))
    shared = SharedFlux([0.4])

    def scored(data, with_offsets):
        if with_offsets:
            data = data.with_closure_offsets(triangle=1.0)
        epochs = Epochs({"n0": data})
        return score_orbits(epochs, OrbitalBinary, [TRUTH], shared=shared)

    profiled = (
        scored(night(shift), True).score[0] - scored(clean, True).score[0]
    )
    plain = (
        scored(night(shift), False).score[0] - scored(clean, False).score[0]
    )
    assert abs(plain) > 5.0
    assert abs(profiled) < 0.1 * abs(plain)


def test_refuses_a_model_whose_companion_free_scene_moves():
    epochs = _epochs()

    def moving(orbit, flux):
        return OrbitalBinary(orbit, 0.2 + flux)  # flux 0 still has a companion

    with pytest.raises(ValueError, match="must not depend on the orbit"):
        score_orbits(epochs, moving, [TRUTH, OTHER])
