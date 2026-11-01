"""Tests for the simulated orbit ensemble harness."""

import importlib.util
from pathlib import Path

import jax
import numpy as onp
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "orbit_ensemble", ROOT / "scripts" / "orbit_ensemble.py"
)
oe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oe)
draw_population = oe.draw_population
run_one = oe.run_one


def test_population_draws_follow_the_specification():
    """Verify 2000 population draws follow distribution and timing rules."""
    rng = onp.random.default_rng(0)
    pop = draw_population(rng, 2000)
    assert len(pop) == 2000

    periods = onp.array([p["period"] for p in pop])
    assert onp.all(periods >= 2.0)
    assert onp.all(periods <= 60.0)
    ks_p = stats.kstest(
        onp.log(periods),
        "uniform",
        args=(onp.log(2.0), onp.log(60.0) - onp.log(2.0)),
    )
    assert ks_p.pvalue > 1e-3

    ecc = onp.array([p["ecc"] for p in pop])
    assert onp.all(ecc >= 0.0)
    assert onp.all(ecc <= 0.8)

    cos_i = onp.array([p["cos_i"] for p in pop])
    ks_cos = stats.kstest(cos_i, "uniform", args=(-1.0, 2.0))
    assert ks_cos.pvalue > 1e-3

    k_nights = onp.array([p["K"] for p in pop])
    assert onp.all(k_nights >= 5)
    assert onp.all(k_nights <= 20)

    for p in pop:
        times = p["times"]
        assert len(times) == p["K"]
        assert len(onp.unique(times)) == p["K"]
        if p["layout"] == "one_season":
            assert times[-1] - times[0] <= 30.0
        else:
            t1 = times[times < 100.0]
            t2 = times[times > 100.0]
            assert len(t1) >= 2
            assert len(t2) >= 2
            gap = t2[0] - t1[0]
            assert abs(gap - 365.25) <= 30.0


def test_run_one_returns_a_complete_row():
    """Verify run_one outputs a complete result row and normalized posterior."""
    rng = onp.random.default_rng(42)
    params = None
    while params is None:
        draws = draw_population(rng, 10)
        for d in draws:
            if d["K"] == 6 and d["layout"] == "two_seasons":
                params = d
                break

    key = jax.random.key(123)
    row = run_one(params, key, n_particles=64, max_seconds=900.0, pilot=False)

    required_keys = {
        "orbit_id",
        "period",
        "ecc",
        "cos_i",
        "inc_deg",
        "omega_deg",
        "Omega_deg",
        "t_peri",
        "a_mas",
        "sigma",
        "K",
        "layout",
        "span",
        "T",
        "times",
        "n_cells",
        "number_of_cells",
        "true_n",
        "outside_domain",
        "p_pos_true",
        "p_pos_true_n",
        "in_99",
        "in_99_set",
        "rank_true",
        "top_cell_n",
        "top_cell_p_lo",
        "top_cell_p_hi",
        "flat",
        "is_flat",
        "spread_ln_z",
        "wall_time",
        "peak_rss",
        "exception",
        "p_pos",
    }
    for k in required_keys:
        assert k in row, f"Missing key: {k}"

    assert not row["exception"]
    assert row["p_pos"].sum() == pytest.approx(1.0, abs=1e-6)
