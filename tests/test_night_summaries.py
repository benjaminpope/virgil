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
    n, chi2 = base.n.sum(), base.chi2.sum()
    expected = -(n * onp.log(3.0) + 0.5 * chi2 * (1 / 9 - 1))
    assert onp.isclose(base.log_scale(3.0), expected)


def _gains_and_offsets(d):
    n = int(onp.asarray(d.vis).size)
    d = d.with_gains(telescope=0.03, modes=onp.ones((1, n)))
    return d.with_closure_offsets(triangle=0.02)


def test_a_night_with_offsets_and_spanning_gain_modes_is_fitted():
    peak = _fit(_night(post=_gains_and_offsets)).peaks[0]
    pull = (peak.mean - TRUTH) / onp.sqrt(onp.diag(peak.cov))
    assert onp.all(onp.isfinite(peak.cov)) and onp.abs(pull).max() < 5.0
    # a scale does not factor out of s² D + U Uᵀ: no closed-form rescaling
    assert not peak.rescalable
    assert onp.isclose(peak.log_scale(1.0), 0.0)
    with pytest.raises(NotImplementedError):
        peak.covariance(scale=3.0)


def test_a_single_exposure_gives_an_uninformative_velocity():
    summary = _fit(_night(frames=1))
    assert summary.t_ss == 0.0 and summary.n_frames == 1
    peak = summary.peaks[0]
    # the position is still measured, the velocity carries no weight
    assert onp.sqrt(peak.cov[0, 0]) < 0.1
    assert peak.cov[2, 2] > 1e8 * peak.cov[0, 0]
    assert onp.all(onp.isfinite(peak.cov))
    # the jitter's velocity share is infinite, not a crash
    assert onp.isinf(summary.covariance(peak, jitter=0.1)[2, 2])


def test_the_jitter_is_independent_per_frame():
    # a per-frame jitter gains σ_J²/N at the night's position (the mean of N
    # frames) and σ_J²/Σ(t_j − t̄)² in the velocity
    s = _fit(_night())
    peak = s.peaks[0]
    extra = s.covariance(peak, jitter=0.2) - peak.cov
    sj2 = 0.2**2
    assert s.n_frames == 6
    assert onp.allclose(
        onp.diag(extra),
        [sj2 / 6, sj2 / 6, sj2 / s.t_ss, sj2 / s.t_ss, 0.0],
    )
    assert onp.allclose(s.covariance(peak), peak.cov)


def test_the_null_is_profiled_like_the_peaks():
    # a contrast loss is fitted by the amplitude for the null and the peaks
    # alike, so the profile adds to both
    data = _night()
    lost = eqx.tree_at(lambda d: d.vis, data, 0.9 * onp.asarray(data.vis))
    plain, profiled = _fit(lost), _fit(lost, amp_range=(0.5, 1.0))
    assert profiled.loglike_null > plain.loglike_null + 5.0
    assert profiled.peaks[0].loglike > plain.peaks[0].loglike + 5.0


def test_float32_data_give_the_float64_summary():
    data = _night()
    with run_in("float32"):
        d32 = cast_tree(data, "float32")
        p32 = _fit(d32).peaks[0]
    p64 = _fit(data).peaks[0]
    assert p32.mean.dtype == onp.float64 and p32.cov.dtype == onp.float64
    sigma = onp.sqrt(onp.diag(p64.cov))
    assert onp.all(onp.abs(p32.mean - p64.mean) < 0.5 * sigma)


def test_peaks_converging_to_one_point_merge_to_the_best():
    data = _night()
    starts = [(3.2, -1.8, 0.2), (3.0, -2.2, 0.4)]
    both = _fit(data, start=starts).peaks
    singles = [_fit(data, start=s).peaks[0] for s in starts]
    sigma = onp.sqrt(onp.diag(singles[0].cov))
    near = [p for p in both if onp.all(onp.abs(p.mean - TRUTH) < 6 * sigma)]
    assert len(near) == 1  # both starts found the truth: one peak
    assert onp.isclose(near[0].loglike, max(p.loglike for p in singles))


def test_peaks_are_never_reported_on_a_bound():
    # a start far from any companion runs onto the window or flux bound
    data = _night()
    for start in [(25.0, 25.0, 0.3), (3.0, -2.0, 1e-6), (3.0, -2.0, 5.0)]:
        for p in _fit(data, start=start).peaks:
            assert 1e-4 * 1.0001 < onp.exp(p.mean[4]) < 1.0 - 1e-6


def test_the_keep_rule_threshold_is_derived_from_the_node_count():
    summary = _fit(_night())
    best = summary.peaks[0].loglike

    def kept(k, shift):
        return len(ns.keep_peaks((summary,) * k, best + shift)[0].peaks)

    # K = 1: ε_lo = sin²(π/4) = ½, so a peak stays iff L >= b + ln δ
    edge = -onp.log(ns.DELTA)
    assert kept(1, edge - 0.02) == 1 and kept(1, edge + 0.02) == 0
    # K = 3: n = 2, ε_lo = sin²(π/8): L >= b + ln[δ ε / (1 - ε)]
    eps = onp.sin(onp.pi / 8) ** 2
    edge = -onp.log(ns.DELTA * eps / (1 - eps))
    assert kept(3, edge - 0.02) == 1 and kept(3, edge + 0.02) == 0
