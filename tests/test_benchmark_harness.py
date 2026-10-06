"""Smoke test of the orbit benchmark harness (benchmarks/orbits/).

One tiny case through the real pipeline steps, checking that the harness
writes valid rows with compile and run times separated and that the
recompile detector works. The benchmarks themselves run on OzSTAR; see
design/automatic_orbits.md section 9.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.pardir, "benchmarks", "orbits"),
)

import cases  # noqa: E402
import harness  # noqa: E402
import run_bench  # noqa: E402

TINY = cases.Case(**run_bench.SIZES["tiny"]["ref"])


def test_a_tiny_case_writes_valid_rows(tmp_path):
    out = tmp_path / "bench.jsonl"
    code = run_bench.main(
        [
            "--size", "tiny",
            "--steps", "epoch_positions", "starting_orbits",
            "--repeats", "2",
            "--systems", "2",
            "--out", str(out),
        ]
    )  # fmt: skip
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert code == 0  # no warm recompile, no recompile for a second system
    assert [r["step"] for r in rows] == ["epoch_positions", "starting_orbits"]
    for r in rows:
        assert r["status"] == "ok"
        assert r["run_s"] > 0 and r["first_s"] >= r["run_s"] * 0.5
        assert r["systems_per_hour"] == pytest.approx(3600 / r["run_s"])
        assert r["n_compiles_warm"] == 0
        assert r["compile_s"] >= 0 and r["peak_rss_mb"] > 0
        assert len(r["run_all_s"]) == 2
        assert r["n_epochs"] == 4 and r["jax_version"] == jax.__version__
    assert rows[0]["n_compiles_first"] > 0  # the grid kernel was compiled
    assert rows[0]["n_compiles_new_system"] == 0


def test_the_csv_format_has_one_row_per_measurement(tmp_path):
    out = tmp_path / "bench.csv"
    run_bench.main(
        ["--size", "tiny", "--steps", "starting_orbits", "--repeats", "1",
         "--format", "csv", "--out", str(out)]
    )  # fmt: skip
    lines = out.read_text().splitlines()
    assert len(lines) == 2 and "run_s" in lines[0].split(",")


def test_the_recompile_detector_sees_a_shape_change():
    f = jax.jit(lambda x: x * 2.0)
    f(jnp.ones(3))
    with harness.count_compiles() as same:
        f(jnp.ones(3)).block_until_ready()
    with harness.count_compiles() as new:
        f(jnp.ones(5)).block_until_ready()
    assert same["n"] == 0 and new["n"] >= 1
    with pytest.raises(harness.RecompileError, match="compiled"):
        harness.assert_no_compiles("f", f, jnp.ones(7))
    harness.assert_no_compiles("f", f, jnp.ones(7))  # now cached


def test_measure_separates_the_first_call_from_warm_calls():
    f = jax.jit(lambda x: jnp.sin(x).sum())
    x = jnp.arange(10.0)
    m = harness.measure(lambda: f(x), repeats=3)
    assert m.n_compiles_first == 1 and m.n_compiles_warm == 0
    assert m.first_s > m.run_s
    assert m.compile_s > 0 and len(m.run_all_s) == 3


def test_cases_are_reproducible_from_their_manifest():
    a, b = cases.build(TINY), cases.build(TINY)
    assert a.bucket() == b.bucket()
    assert onp.allclose(a.epochs.data[0].vis, b.epochs.data[0].vis)
    other = cases.build(TINY.replace(seed=1))
    assert not onp.allclose(a.epochs.data[0].vis, other.epochs.data[0].vis)
    assert json.loads(json.dumps(TINY.manifest()))["seed"] == 0
    # The adversarial cases change what they say they change.
    assert cases.build(TINY.replace(case="A6")).noise_scale > 3.0
    assert len(cases.build(TINY.replace(case="A5", n_epochs=8)).epochs) == 5
    with pytest.raises(NotImplementedError, match="A13"):
        cases.build(TINY.replace(case="A13"))
