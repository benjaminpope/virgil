"""Period aliases of an undersampled orbit: bands, window and evidences."""

import json

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import aliases  # noqa: E402
from virgil.aliases import (  # noqa: E402
    AliasBandResult,
    AliasResult,
    alias_bands,
    fit_orbit_aliases,
    position_profile,
    spectral_window,
)
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import KeplerOrbit, PositionData  # noqa: E402
from virgil.simulate import simulate  # noqa: E402

T_REF = 60000.0
TRUTH = dict(
    period=12.1,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
EPOCHS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])


def _night(mjd, frames=3, rotation=0.0, d_phi=0.02):
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )
    shift = lambda i: onp.concatenate([i + k * n for k in range(frames)])  # noqa: E731
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 2.2e-6,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, 0.01),
            "phi": onp.zeros(len(i1) * frames),
            "d_phi": onp.full(len(i1) * frames, d_phi),
            "i_cps1": shift(i1),
            "i_cps2": shift(i2),
            "i_cps3": shift(i3),
            "mjd": onp.repeat(mjd + onp.arange(frames) / 24.0, n),
        }
    )


def test_alias_bands_tile_the_period_range():
    times = [0.0, 100.0, 413.89]
    bands = alias_bands(times, (10.0, 14.0))
    assert [b.n for b in bands] == list(range(30, 42))
    n34 = next(b for b in bands if b.n == 34)
    assert n34.p_lo < 12.137 < n34.p_hi
    for lo, hi in zip(bands[1:], bands[:-1]):  # contiguous in period
        assert onp.isclose(lo.p_hi, hi.p_lo)


def test_spectral_window_is_one_at_zero_and_returns_at_aliases():
    t = onp.array([0.0, 1.0, 2.0, 3.0])  # a unit grid: aliases at integers
    w = spectral_window(t, [0.0, 0.5, 1.0])
    assert onp.allclose(w[[0, 2]], 1.0) and w[1] < 0.2


def test_position_profile_has_a_chi2_per_band():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(EPOCHS)
    cov = onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1))
    pos = PositionData(EPOCHS, dra, ddec, cov, t_ref=T_REF)
    bands = alias_bands(EPOCHS, (11.0, 13.0))
    rows = position_profile(pos, bands, k=3.0, n_best=1)
    assert [r["n"] for r in rows] == [b.n for b in bands]
    assert all(onp.isfinite(r["chi2"]) for r in rows)


def _synthetic(period):
    truth = {**TRUTH, "period": period}
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    key = jax.random.PRNGKey(3)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    dra, ddec, _ = orbit.relative(EPOCHS)
    rng = onp.random.default_rng(1)
    pos = PositionData(
        EPOCHS,
        onp.asarray(dra) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.asarray(ddec) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1)),
        t_ref=T_REF,
    )
    return data, pos


def test_alias_bands_include_the_long_period_band():
    bands = alias_bands([0.0, 100.0], (50.0, 500.0))
    assert (
        bands[0].n == 0 and bands[0].p_lo == 200.0 and bands[0].p_hi == 500.0
    )


# 12.38 d sits mid-band N = 20; 12.1 d is 0.05 cycles from the N = 20/21 edge.
@pytest.mark.slow
@pytest.mark.parametrize("period", [12.38, 12.1])
def test_injected_alias_wins_in_closure_phases(period, tmp_path):
    data, pos = _synthetic(period)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=pos,
        t_ref=T_REF,
        n_candidates=40,
        n_refine=4,
        n_is=400,
        n_samples=50,
        k=3.0,
    )
    n_true = round(float(onp.ptp(result.times)) / period)
    assert result.best.n == n_true
    assert result.best.p > 0.9
    assert abs(result.best.best["period"] - period) < 0.05
    assert result.best.chi2_red < 3.0  # raw χ²/ν on the quoted errors
    assert (
        n_true in result.samples
        and len(result.samples[n_true]["period"]) == 50
    )
    s = result.samples[n_true]
    assert onp.all((s["Omega"] >= 0) & (s["Omega"] < 180))  # mirror folded
    row = result.table()[0]
    assert row["n"] == n_true and "flags" in row
    result.to_json(tmp_path / "bands.json", n_samples=5)
    json.loads(
        (tmp_path / "bands.json").read_text()
    )  # strict JSON: no Infinity


@pytest.mark.slow
def test_random_starts_and_s_max_without_positions():
    data, _ = _synthetic(12.38)
    result = fit_orbit_aliases(
        data,
        (12.1, 12.7),
        t_ref=T_REF,
        n_random=300,
        n_refine=4,
        n_is=200,
        n_samples=20,
        s_max=50.0,
    )
    # Random starts rarely find a narrow ridge in 8-D: only check that it runs.
    assert result.bands and all(b.flags is not None for b in result.bands)


# --- robustness: the failures of OzSTAR job 18296568 ----------------------------------


def _band(n, epochs, **kw):
    return AliasBandResult(
        n=n, p_lo=1.0, p_hi=2.0, modes=[], log_z=-1.0, log_z_is=-1.0, ess=10.0,
        flagged=False, epochs=epochs, **kw
    )  # fmt: skip


def test_table_scales_are_flat_floats_for_multi_block_epochs():
    # An epoch with V2 and closure-phase blocks has one scale per block (a list).
    epochs = [dict(scale=[1.0, 2.5]), dict(scale=[1.5])]
    res = AliasResult(0.0, onp.zeros(2), 1.0, [_band(3, epochs)], {})
    row = res.table()[0]
    assert row["scales"] == [1.0, 2.5, 1.5]
    assert row["epoch_scales"] == [[1.0, 2.5], [1.5]]
    assert f"{min(row['scales']):>6.2f}" and f"{max(row['scales']):>6.2f}"


def test_singular_hessian_is_flagged_not_raised():
    class Flat:
        def hessian(self, x):
            return -onp.diag([1.0, 1.0, 1e-20, 1.0, 1.0, 1.0, 1.0, 1.0])

    x = onp.zeros(8)
    rec, draw = aliases._mode_evidence(
        Flat(), x, -1.0, 10, onp.random.default_rng(0)
    )
    assert draw is None and rec["flags"] == ["hessian-singular"]
    assert rec["log_z_laplace"] == -onp.inf


def test_band_checkpoint_roundtrip_and_key(tmp_path):
    band = alias_bands([0.0, 100.0, 413.89], (10.0, 14.0))[0]
    key = dict(n_is=10, edges=[[1.0, 2.0]])
    done = (
        _band(band.n, [dict(scale=[1.0])]),
        (onp.ones((3, 2)), onp.zeros(3)),
    )
    assert aliases._load_band(tmp_path, band, key) is None
    aliases._save_band(tmp_path, band, key, done)
    json.loads((tmp_path / f"band_{band.n}.json").read_text())
    got = aliases._load_band(tmp_path, band, key)
    assert got[0].n == band.n and got[1][0].shape == (3, 2)
    assert aliases._load_band(tmp_path, band, {**key, "n_is": 11}) is None
    assert aliases._load_band(None, band, key) is None


@pytest.mark.slow
def test_one_failing_band_is_isolated_checkpointed_and_not_refit(
    monkeypatch, tmp_path
):
    data, _ = _synthetic(12.38)
    first = alias_bands(EPOCHS, (12.1, 12.7))[0].n
    calls = []
    real = aliases._Problem.set_band

    def set_band(self, band):
        calls.append(band.n)
        if band.n == first:
            raise RuntimeError("boom")
        return real(self, band)

    monkeypatch.setattr(aliases._Problem, "set_band", set_band)
    kw = dict(
        t_ref=T_REF,
        n_random=20,
        n_refine=1,
        max_steps=3,
        n_is=20,
        n_samples=5,
        checkpoint_dir=tmp_path,
    )
    seen = []
    result = fit_orbit_aliases(data, (12.1, 12.7), on_band=seen.append, **kw)
    assert len(seen) == len(result.bands) > 1
    bad = [b for b in result.bands if b.flags == ["band-failed"]]
    assert [b.n for b in bad] == [first] and "boom" in bad[0].error
    others = [b for b in result.bands if b.n != first]
    assert {p.name for p in tmp_path.glob("band_*.pkl")} == {
        f"band_{b.n}.pkl" for b in others
    }
    assert not (
        tmp_path / f"band_{first}.pkl"
    ).exists()  # a failed band is retried
    result.to_json(tmp_path / "bands.json")
    json.loads((tmp_path / "bands.json").read_text())
    # A rerun with the same arguments loads the good bands and refits only the failed one.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), **kw)
    assert calls == [first]
    # A changed setting (s_max) invalidates the checkpoints.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), s_max=50.0, **kw)
    assert len(calls) == len(result.bands)
