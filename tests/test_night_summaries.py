"""Night summaries: per-peak linear-motion fits at visibility level."""

import types

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import night_summaries as ns  # noqa: E402
from virgil._precision import cast_tree, run_in  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from tests._shared import vlti_night  # noqa: E402

T0 = 60000.0
# (dra, ddec) mas, (vra, vdec) mas/day, ln f: a few tenths of a resolution
# element of motion over a six-frame night.
TRUTH = onp.array([3.0, -2.0, 4.0, -3.0, onp.log(0.3)])


def _times(data):
    _, idx = onp.unique(onp.asarray(data.frame), return_inverse=True)
    return onp.bincount(idx, weights=onp.asarray(data.mjd)) / onp.bincount(idx)


def _night(frames=6, seed=0, d_vis=0.005, d_phi=0.02, post=None):
    """A night with the companion at ``TRUTH``; ``post`` edits the empty data
    (gains, offsets) before simulating."""
    data = vlti_night(T0, frames, 0.0, d_vis=d_vis, d_phi=d_phi, stations=True)
    if post is not None:
        data = post(data)
    t_ref = float(onp.mean(_times(data)))
    model = ns._model(np.asarray(TRUTH), t_ref, None)
    return simulate(model, data, key=jax.random.PRNGKey(seed))


def _positions(start):
    """A stub catalogue with one peak at ``start = (dra, ddec, flux)``."""
    start = onp.atleast_2d(start)
    peaks = types.SimpleNamespace(
        dra=start[:, 0], ddec=start[:, 1], flux=start[:, 2]
    )
    return types.SimpleNamespace(
        names=("n",), peaks=(peaks,), primary_diam=None
    )


def _fit(data, start=(3.2, -1.8, 0.2), **kw):
    return ns.night_summaries([data], _positions(start), **kw)[0]


def test_pulls_over_seeds_are_unit_normal():
    # (p, v, ln f) within the reported covariance, over many noise draws.
    pulls = []
    for seed in range(24):
        peak = _fit(_night(seed=seed)).peaks[0]
        dz = peak.mean - TRUTH
        pulls.append(dz / onp.sqrt(onp.diag(peak.cov)))
        # the position is at the night's reference time, so p is the truth's
        # only where the truth is referenced there too
    pulls = onp.array(pulls)
    assert abs(pulls.mean()) < 0.3
    assert 0.75 < onp.sqrt((pulls**2).mean()) < 1.3
    # the joint 5-D χ² has mean 5
    assert 3.5 < (pulls**2).sum(1).mean() < 6.5


def test_the_velocity_variance_scales_as_one_over_span_squared():
    # Stretching the frame times by k and fitting the same visibilities
    # reparameterizes v -> k v: the velocity covariance falls as 1/k².
    data = _night()
    base = _fit(data).peaks[0]
    k = 2.5
    wide = eqx.tree_at(lambda d: d.dt, data, k * onp.asarray(data.dt))
    peak = _fit(wide).peaks[0]
    assert onp.allclose(peak.mean[2:4] * k, base.mean[2:4], rtol=1e-3)
    assert onp.allclose(
        peak.cov[2:4, 2:4] * k**2, base.cov[2:4, 2:4], rtol=1e-3
    )
    # and the variance matches σ_p² / Σ (t_j - t̄)² to the geometry of the
    # baselines: within a factor of a few of the position's
    s = _fit(data)
    ratio = base.cov[2, 2] * s.t_ss / base.cov[0, 0]
    assert 0.2 < ratio < 5.0


def test_the_covariance_scales_with_the_error_scale():
    data = _night()
    base = _fit(data).peaks[0]
    big = _fit(data.with_error_scale(3.0)).peaks[0]
    assert onp.allclose(big.cov, 9.0 * base.cov, rtol=1e-3)
    # the stored Hessians rescale the same way, in closed form
    scaled = base.covariance(scale=3.0)
    assert (
        onp.abs(scaled - 9.0 * base.cov).max() < 1e-6 * onp.abs(scaled).max()
    )
    assert onp.isclose(base.log_scale(1.0), 0.0)


def test_a_night_with_offsets_and_spanning_gain_modes_is_fitted():
    def post(d):
        n = int(onp.asarray(d.vis).size)
        d = d.with_gains(telescope=0.03, modes=onp.ones((1, n)))
        return d.with_closure_offsets(triangle=0.02)

    data = _night(post=post)
    peak = _fit(data).peaks[0]
    pull = (peak.mean - TRUTH) / onp.sqrt(onp.diag(peak.cov))
    assert onp.all(onp.isfinite(peak.cov)) and onp.abs(pull).max() < 5.0


def test_a_single_exposure_gives_an_uninformative_velocity():
    data = _night(frames=1)
    summary = _fit(data)
    assert summary.t_ss == 0.0
    peak = summary.peaks[0]
    # the position is still measured, the velocity carries no weight
    assert onp.sqrt(peak.cov[0, 0]) < 0.1
    assert peak.cov[2, 2] > 1e8 * peak.cov[0, 0]
    assert onp.all(onp.isfinite(peak.cov))
    # the jitter's velocity share is infinite, not a crash
    assert onp.isinf(peak.covariance(jitter=0.1, t_ss=summary.t_ss)[2, 2])


def test_the_jitter_adds_in_quadrature_to_position_and_velocity():
    data = _night()
    s = _fit(data)
    peak = s.peaks[0]
    extra = peak.covariance(jitter=0.2, t_ss=s.t_ss) - peak.cov
    assert onp.allclose(
        onp.diag(extra), [0.04, 0.04, 0.04 / s.t_ss] * 1 + [0.04 / s.t_ss, 0.0]
    )


def test_float32_data_give_the_float64_summary():
    data = _night()
    with run_in("float32"):
        d32 = cast_tree(data, "float32")
        p32 = _fit(d32).peaks[0]
    p64 = _fit(data).peaks[0]
    assert p32.mean.dtype == onp.float64 and p32.cov.dtype == onp.float64
    sigma = onp.sqrt(onp.diag(p64.cov))
    assert onp.all(onp.abs(p32.mean - p64.mean) < 0.5 * sigma)


def test_peaks_converging_to_one_point_merge_and_a_wrong_one_ranks_last():
    data = _night()
    starts = [(3.2, -1.8, 0.2), (3.0, -2.2, 0.4), (-3.0, 2.0, 0.3)]
    peaks = _fit(data, start=starts).peaks
    assert len(peaks) <= 2
    assert peaks[0].loglike == max(p.loglike for p in peaks)


def test_the_keep_rule_drops_a_peak_the_outlier_term_dominates():
    data = _night()
    starts = [(3.2, -1.8, 0.2), (-3.0, 2.0, 0.3)]
    summaries = (_fit(data, start=starts),)
    best = summaries[0].peaks[0].loglike
    kept = ns.keep_peaks(summaries, best - 5.0)[0].peaks
    assert len(kept) >= 1 and kept[0].loglike == best
    assert len(ns.keep_peaks(summaries, best + 100.0)[0].peaks) == 0
