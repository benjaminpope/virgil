#!/usr/bin/env python
"""Benchmark the orbit pipeline: timing, compile cost, memory, throughput.

Run as a script (never a notebook; design/automatic_orbits.md section 9.5):

    python -X faulthandler benchmarks/orbits/run_bench.py --size tiny
    python -X faulthandler benchmarks/orbits/run_bench.py --size full \\
        --suite baseline --suite compile --out out/bench.jsonl

Each (step, variant, configuration) is one row of ``bench.jsonl`` (or CSV
with ``--format csv``): wall time cold and warm, backend compile time and
count, peak host/device memory, systems per hour and a result-quality check
against the simulated truth. Steps are in ``STEPS``; later PRs add theirs
there (one function per step, nothing else changes).

Suites: ``baseline`` runs the steps over the scaling axes of the size
preset; ``compile`` adds the compile-cost checks (HLO size of the
likelihood against the number of epochs, and NUTS with each
``chain_method`` in a subprocess, recording the exit status).

Failing assertions (exit status 1, listed at the end):

* a warm call that compiles (a recompile) in a step marked ``bounded=True``
  (any step with ``--strict``);
* with ``--systems 2``, a second system of the same shape bucket that
  compiles in a step marked ``bounded=True``;
* with ``--max-hlo-exponent X``, HLO size growing faster than n_epochs**X.

Profiling: ``--profile DIR`` wraps extra warm calls of each selected step in
``jax.profiler.trace`` (``StepTraceAnnotation`` per call, compile outside
the trace); ``--xla-dump DIR`` also dumps optimized HLO as text. Summarize
with ``jax_trace_summary.py``.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time


def _early_env(argv):
    """XLA reads XLA_FLAGS when the backend starts: set it before jax."""
    for i, a in enumerate(argv):
        if a == "--xla-dump" and i + 1 < len(argv):
            d = os.path.abspath(argv[i + 1])
        elif a.startswith("--xla-dump="):
            d = os.path.abspath(a.split("=", 1)[1])
        else:
            continue
        os.makedirs(d, exist_ok=True)
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_dump_to={d} --xla_dump_hlo_as_text".strip()
        )


if __name__ == "__main__":
    _early_env(sys.argv)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax  # noqa: E402
import numpy as onp  # noqa: E402

import cases  # noqa: E402
import harness  # noqa: E402
import jaxprof  # noqa: E402
from virgil.epochs import (  # noqa: E402
    epoch_positions,
    rank_orbits,
    start_from_positions,
)
from virgil.fitting import fit  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.orbits import KeplerOrbit, starting_orbits  # noqa: E402

# ---------------------------------------------------------------------------
# Size presets: reference values, and the axes varied one at a time
# ---------------------------------------------------------------------------

SIZES = {
    # CI and laptop smoke: seconds
    "tiny": dict(
        ref=dict(
            n_epochs=4,
            n_obs=100,
            grid_size=33,
            n_flux=2,
            n_candidates=10,
            n_phase=8,
            n_periods=8,
        ),
        axes={},
        repeats=2,
    ),
    # a few minutes on a laptop, one process
    "small": dict(
        ref=dict(
            n_epochs=4,
            n_obs=100,
            grid_size=33,
            n_flux=3,
            n_candidates=40,
            n_phase=24,
            n_periods=20,
        ),
        axes=dict(
            n_epochs=[4, 6, 8],
            n_obs=[100, 1000],
            grid_size=[33, 65],
            n_candidates=[40, 200],
        ),
        repeats=3,
    ),
    # the design's scaling axes (section 9.1): OzSTAR only
    "full": dict(
        ref=dict(
            n_epochs=8,
            n_obs=1000,
            grid_size=128,
            n_flux=16,
            n_candidates=10_000,
            n_phase=36,
            n_periods=50,
        ),
        axes=dict(
            n_epochs=[4, 8, 16, 32],
            n_obs=[100, 1000, 10_000],
            grid_size=[64, 128, 256],
            n_flux=[16, 32],
            n_candidates=[10_000, 100_000, 1_000_000, 10_000_000],
        ),
        repeats=5,
    ),
}

# ---------------------------------------------------------------------------
# Steps. Each is ``prepare(system, ctx) -> run`` (zero-argument, timed) plus
# ``quality(result, system) -> dict``. ``ctx`` caches upstream results
# outside the timing. ``bounded``: a new system of the same shape bucket
# must not compile (design section 9.4). Recompiles are recorded for every
# step (n_compiles_warm, n_compiles_new_system); only bounded steps fail on
# them, or all with --strict. rank_orbits is not bounded on main yet: its
# jitted kernel is rebuilt on every call (a closure over data and scene_of),
# so it recompiles per call. That is the baseline PR C/D must remove, and
# then set bounded=True.
# ---------------------------------------------------------------------------


def _truth_positions(system):
    with jax.enable_x64(True):
        orbit = KeplerOrbit(
            **{k: system.truth_values[k] for k in cases.TRUTH},
            t_ref=system.t_ref,
        )
        dra, ddec, _ = orbit.relative(system.epochs.times)
    return onp.asarray(dra), onp.asarray(ddec)


def _positions(system, ctx):
    if "positions" not in ctx:
        ctx["positions"] = epoch_positions(system.epochs, system.grid)
    return ctx["positions"]


def _seeds(system, ctx):
    """Positions of every dataset as seeds (the benchmark times the step,
    not the decisiveness flags)."""
    if "seeds" not in ctx:
        ctx["seeds"] = _positions(system, ctx).positions(t_ref=system.t_ref)
    return ctx["seeds"]


def _candidates(system, ctx):
    if "candidates" not in ctx:
        ctx["candidates"] = starting_orbits(
            _seeds(system, ctx),
            system.periods,
            n_phase=system.case.n_phase,
            n_best=system.case.n_candidates,
        )
    return ctx["candidates"]


def _binary_of(flux):
    return lambda orbit: OrbitalBinary(orbit, flux)


def prep_epoch_positions(system, ctx, variant):
    return lambda: epoch_positions(system.epochs, system.grid)


def quality_epoch_positions(result, system):
    dra, ddec = _truth_positions(system)
    err = onp.hypot(result.dra - dra, result.ddec - ddec)
    return dict(
        max_position_error_mas=float(err.max()),
        n_true_peak=int((err < 1.0).sum()),
        min_gap=float(result.gap.min()),
    )


def prep_starting_orbits(system, ctx, variant):
    seeds = _seeds(system, ctx)
    c = system.case
    return lambda: starting_orbits(
        seeds,
        system.periods,
        n_phase=c.n_phase,
        n_best=c.n_candidates,
    )


def quality_starting_orbits(result, system):
    return dict(n_returned=len(result), best_chi2=float(result[0][1]))


def _prior_draws(system, n, seed=0):
    """``n`` orbits with a leading axis, drawn from the priors."""
    rng = onp.random.default_rng(seed)
    p = system.priors
    lo = lambda d: float(d.low)  # noqa: E731
    hi = lambda d: float(d.high)  # noqa: E731
    return KeplerOrbit(
        period=onp.exp(
            rng.uniform(onp.log(lo(p["period"])), onp.log(hi(p["period"])), n)
        ),
        dt_peri=rng.uniform(lo(p["dt_peri"]), hi(p["dt_peri"]), n),
        ecc=rng.uniform(0.0, 0.9, n),
        inc=rng.uniform(0.0, 180.0, n),
        omega=rng.uniform(-180.0, 360.0, n),
        Omega=rng.uniform(-180.0, 360.0, n),
        a_mas=onp.exp(
            rng.uniform(onp.log(lo(p["a_mas"])), onp.log(hi(p["a_mas"])), n)
        ),
        t_ref=system.t_ref,
    )


def prep_rank_orbits(system, ctx, variant):
    flux = system.truth_values["flux"]
    batch = 2048
    if variant == "prior_draws":
        orbits = _prior_draws(system, system.case.n_candidates)
        return lambda: rank_orbits(
            _binary_of(flux), system.epochs, orbits, batch_size=batch
        )
    cands = _candidates(system, ctx)
    orbits = [orbit for orbit, _ in cands]
    return lambda: rank_orbits(
        _binary_of(flux), system.epochs, orbits, batch_size=batch
    )


def quality_rank_orbits(result, system):
    """Do the candidates reach the truth's likelihood? The truth is not
    among them: ``best_minus_truth`` is the best candidate's log likelihood
    minus the truth's (near or above 0 is good), and ``truth_rank`` is how
    many candidates beat the truth."""
    with jax.enable_x64(True):
        exact = float(
            system.epochs.loglike(
                OrbitalBinary(
                    KeplerOrbit(
                        **{k: system.truth_values[k] for k in cases.TRUTH},
                        t_ref=system.t_ref,
                    ),
                    system.truth_values["flux"],
                )
            )
        )
    return dict(
        best_loglike=float(result.loglike[0]),
        truth_loglike=exact,
        best_minus_truth=float(result.loglike[0] - exact),
        truth_rank=int((result.loglike > exact).sum()),
        n_ranked=len(result),
    )


def prep_start_from_positions(system, ctx, variant):
    c = system.case
    return lambda: start_from_positions(
        system.scene_fn,
        system.priors,
        system.epochs,
        cases.start_values,
        grid=system.grid,
        periods=system.periods,
        t_ref=system.t_ref,
        n_phase=c.n_phase,
        n_candidates=c.n_candidates,
        n_refine=2,
        method="lm",
    )


def quality_start_from_positions(result, system):
    return dict(
        best_chi2_red=float(result.best.info["chi2_red"]),
        n_modes=len(result.modes()),
        n_fits=len(result.fits),
    )


def _truth_init(system):
    t = system.truth_values
    return {
        **t,
        "period": t["period"] * 1.02,
        "Omega": t["Omega"] + 2.0,
        "a_mas": t["a_mas"] * 0.98,
    }


def prep_local_fit(system, ctx, variant):
    model_fn = system.epochs.model_fn(system.scene_fn)
    init = _truth_init(system)
    return lambda: fit(
        model_fn, system.priors, system.epochs.data, init=init, method=variant
    )


def quality_local_fit(result, system):
    return dict(chi2_red=float(result.info["chi2_red"]))


def prep_nuts_mode(system, ctx, variant):
    """NUTS from a start at the truth; ``variant`` is the chain_method.
    Few draws: this times the sampler's per-step cost and compile."""
    from numpyro.infer import MCMC, NUTS

    from virgil.likelihood import chain_init_params, numpyro_model

    n_chains = 2
    with jax.enable_x64(True):
        posterior = numpyro_model(
            system.epochs.model_fn(system.scene_fn),
            system.priors,
            system.epochs.data,
        )
        init = chain_init_params(
            posterior,
            [system.truth_values] * n_chains,
            jax.random.PRNGKey(0),
        )

    def run():
        with jax.enable_x64(True):
            mcmc = MCMC(
                NUTS(posterior),
                num_warmup=20,
                num_samples=20,
                num_chains=n_chains,
                chain_method=variant,
                progress_bar=False,
            )
            mcmc.run(jax.random.PRNGKey(1), init_params=init)
            return mcmc.get_samples(group_by_chain=True)

    return run


def quality_nuts_mode(result, system):
    return dict(
        finite=bool(all(onp.all(onp.isfinite(v)) for v in result.values()))
    )


STEPS = {
    "epoch_positions": dict(
        prepare=prep_epoch_positions,
        quality=quality_epoch_positions,
        variants=["grid"],
        bounded=True,
    ),
    "starting_orbits": dict(
        prepare=prep_starting_orbits,
        quality=quality_starting_orbits,
        variants=["thiele_innes"],
        bounded=False,
    ),
    "rank_orbits": dict(
        prepare=prep_rank_orbits,
        quality=quality_rank_orbits,
        variants=["starting_orbits", "prior_draws"],
        bounded=False,
    ),
    "start_from_positions": dict(
        prepare=prep_start_from_positions,
        quality=quality_start_from_positions,
        variants=["pipeline"],
        bounded=False,
    ),
    "local_fit": dict(
        prepare=prep_local_fit,
        quality=quality_local_fit,
        variants=["lm"],
        bounded=False,
    ),
    "nuts_mode": dict(
        prepare=prep_nuts_mode,
        quality=quality_nuts_mode,
        variants=["vectorized", "sequential"],
        bounded=False,
    ),
}
# Steps in the default (baseline) run: NUTS only when asked.
DEFAULT_STEPS = [
    "epoch_positions",
    "starting_orbits",
    "rank_orbits",
    "start_from_positions",
    "local_fit",
]
SMOKE_STEPS = ["epoch_positions", "starting_orbits", "rank_orbits"]
# prior_draws scales to millions of candidates; the others do not need it.
VARIANT_AXES = {"prior_draws": ("n_candidates",)}

# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def platform():
    d = jax.devices()[0]
    return f"{d.platform}:{getattr(d, 'device_kind', '')}".strip(":")


def commit():
    pin = os.environ.get("PIN_COMMIT")
    if pin:
        return pin
    try:
        out = subprocess.run(
            ["git", "-C", HERE, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def base_row(suite, system):
    c = system.case
    return dict(
        suite=suite,
        case=c.case,
        seed=c.seed,
        n_epochs=len(system.epochs),
        n_obs_per_dataset=int(system.n_obs_total / len(system.epochs)),
        grid_size=c.grid_size,
        n_flux=c.n_flux,
        n_candidates=c.n_candidates,
        noise_scale=system.noise_scale,
        platform=platform(),
        jax_version=jax.__version__,
        commit=commit(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def run_step(
    suite, name, variant, system, repeats, profile, assert_bucket, strict=False
):
    """One row. Exceptions become ``status: error`` rows."""
    step = STEPS[name]
    row = base_row(suite, system)
    row.update(step=name, variant=variant, bounded=step["bounded"])
    failures = []
    try:
        ctx = {}
        run = step["prepare"](system, ctx, variant)
        trace = None
        if profile:
            tag = f"{name}-{variant}-{system.case.case}"
            trace = lambda: jaxprof.maybe_profile(tag)  # noqa: E731
        m = harness.measure(run, repeats=repeats, profile_trace=trace)
        row.update(m.row())
        row["status"] = "ok"
        for k, v in step["quality"](m.result, system).items():
            row[f"q_{k}"] = v
        if m.n_compiles_warm and (step["bounded"] or strict):
            failures.append(
                f"{name}/{variant}: {m.n_compiles_warm} compile(s) in warm "
                f"calls: {m.compile_names_warm}"
            )
        if assert_bucket:
            row.update(_new_system_compiles(system, name, variant))
            if (step["bounded"] or strict) and row["n_compiles_new_system"]:
                failures.append(
                    f"{name}/{variant}: {row['n_compiles_new_system']} "
                    "compile(s) for a second system of the same bucket"
                )
    except Exception as e:  # a crash is a failed benchmark, not a skipped one
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
        failures.append(f"{name}/{variant}: {row['error']}")
    return row, failures


def _new_system_compiles(system, name, variant):
    other = cases.build(system.case.replace(seed=system.case.seed + 1))
    assert other.bucket() == system.bucket()
    step = STEPS[name]
    run = step["prepare"](other, {}, variant)
    with harness.count_compiles() as c:
        harness.block(run())
    return dict(
        n_compiles_new_system=c["n"],
        compile_names_new_system=sorted(set(c["names"])),
    )


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def configurations(size, case_name, seed, only_axes=None):
    """The reference configuration, then each axis varied alone."""
    preset = SIZES[size]
    ref = dict(preset["ref"])
    out = [("ref", ref)]
    for axis, values in preset["axes"].items():
        if only_axes and axis not in only_axes:
            continue
        for v in values:
            if v != ref[axis]:
                out.append((axis, {**ref, axis: v}))
    return [
        (axis, cases.Case(case=case_name, seed=seed, **cfg))
        for axis, cfg in out
    ]


def sweep(args, suite="baseline"):
    rows, failures = [], []
    names = args.steps or (
        SMOKE_STEPS if args.size == "tiny" else DEFAULT_STEPS
    )
    repeats = args.repeats or SIZES[args.size]["repeats"]
    for case_name in args.case:
        for axis, case in configurations(
            args.size, case_name, args.seed, args.axis
        ):
            system = cases.build(case)
            for name in names:
                for variant in STEPS[name]["variants"]:
                    if (
                        axis != "ref"
                        and variant in VARIANT_AXES
                        and axis not in VARIANT_AXES[variant]
                    ):
                        continue
                    if (
                        axis == "n_candidates"
                        and case.n_candidates > 100_000
                        and variant != "prior_draws"
                    ):
                        continue  # a Thiele-Innes grid cannot return so many
                    row, fails = run_step(
                        suite,
                        name,
                        variant,
                        system,
                        repeats,
                        args.profile,
                        args.systems > 1,
                        args.strict,
                    )
                    row["axis"] = axis
                    rows.append(row)
                    failures += fails
                    _log(row)
    return rows, failures


def _log(row):
    print(
        f"[{row['case']}|{row['axis']}] {row['step']}/{row['variant']} "
        f"n_epochs={row['n_epochs']} "
        + (
            f"cold {row['first_s']:.3f}s compile {row['compile_s']:.3f}s "
            f"x{row['n_compiles_first']} | warm {row['run_s']:.4f}s "
            f"(compiles {row['n_compiles_warm']}) "
            f"rss {row['peak_rss_mb']:.0f}MB "
            f"{row['systems_per_hour']:.0f} systems/h"
            if row["status"] == "ok"
            else f"ERROR {row['error']}"
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Compile suite
# ---------------------------------------------------------------------------


def compile_suite(args):
    """HLO size of the multi-epoch likelihood against n_epochs, and NUTS
    with each chain_method in a subprocess."""
    rows, failures = [], []
    counts = [2, 4, 8] if args.size != "full" else [2, 4, 8, 16, 32]
    sizes = []
    ref = SIZES[args.size]["ref"]
    for n in counts:
        system = cases.build(
            cases.Case(case="A1", seed=args.seed, **{**ref, "n_epochs": n})
        )

        def loglike(theta, system=system):
            return system.epochs.loglike(system.scene_fn(**theta))

        with jax.enable_x64(True), harness.RssWatcher() as watch:
            theta = {
                k: jax.numpy.asarray(v)
                for k, v in system.truth_values.items()
            }
            n_hlo, secs = harness.hlo_size(loglike, theta)
        sizes.append(n_hlo)
        row = base_row("compile", system)
        row.update(
            step="loglike_hlo",
            variant="epochs.loglike",
            status="ok",
            hlo_instructions=n_hlo,
            lower_compile_s=secs,
            compile_peak_rss_mb=watch.peak,
        )
        rows.append(row)
        print(f"[compile] loglike n_epochs={n}: {n_hlo} HLO ops, {secs:.2f}s")
    exponent = harness.growth_exponent(counts, sizes)
    summary = dict(
        suite="compile",
        step="loglike_hlo",
        variant="growth_exponent",
        status="ok",
        hlo_growth_exponent=exponent,
        n_epochs_axis=counts,
        hlo_instructions_axis=sizes,
        platform=platform(),
        jax_version=jax.__version__,
        commit=commit(),
    )
    rows.append(summary)
    print(f"[compile] HLO size grows as n_epochs**{exponent:.2f}")
    if args.max_hlo_exponent is not None and exponent > args.max_hlo_exponent:
        failures.append(
            f"loglike HLO grows as n_epochs**{exponent:.2f} "
            f"(limit {args.max_hlo_exponent})"
        )
    for method in args.chain_methods:
        row, fails = _nuts_subprocess(args, method)
        rows.append(row)
        failures += fails
        print(f"[compile] nuts {method}: {row['exit']}")
    return rows, failures


def _nuts_subprocess(args, method):
    """One NUTS run in its own process under faulthandler: a crash is a
    failed benchmark. Parallel chains get four forced host devices."""
    env = dict(os.environ)
    if method == "parallel":
        flags = env.get("XLA_FLAGS", "")
        env["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count=4".strip()
        )
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        os.path.abspath(__file__),
        "--size",
        args.size,
        "--seed",
        str(args.seed),
        "--suite",
        "nuts-child",
        "--chain-method",
        method,
        "--out",
        os.devnull,
    ]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    code = p.returncode
    exit_text = (
        f"signal {signal.Signals(-code).name}" if code < 0 else f"exit {code}"
    )
    row = dict(
        suite="compile",
        step="nuts_mode",
        variant=method,
        chain_method=method,
        status="ok" if code == 0 else "crash",
        exit=exit_text,
        returncode=code,
        wall_s=wall,
        platform=platform(),
        jax_version=jax.__version__,
        commit=commit(),
    )
    fails = []
    if code != 0:
        row["stderr_tail"] = p.stderr[-800:]
        fails.append(f"nuts chain_method={method}: {exit_text}")
    return row, fails


def nuts_child(args):
    system = cases.build(
        cases.Case(case="A1", seed=args.seed, **SIZES[args.size]["ref"])
    )
    run = prep_nuts_mode(system, {}, args.chain_method)
    harness.block(run())


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def write_rows(rows, path, fmt):
    if path == os.devnull:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        if fmt == "jsonl":
            for r in rows:
                f.write(json.dumps(r, default=_json_default) + "\n")
        else:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        k: json.dumps(v, default=_json_default)
                        if isinstance(v, (list, dict))
                        else v
                        for k, v in r.items()
                    }
                )
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (onp.integer, onp.floating, onp.bool_)):
        return o.item()
    if isinstance(o, onp.ndarray):
        return o.tolist()
    return str(o)


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size", choices=sorted(SIZES), default="tiny")
    p.add_argument(
        "--suite",
        action="append",
        default=None,
        help="baseline (default) and/or compile",
    )
    p.add_argument(
        "--case",
        action="append",
        default=None,
        help=f"adversarial case(s): {sorted(cases.CASES)}",
    )
    p.add_argument("--steps", nargs="+", choices=sorted(STEPS))
    p.add_argument("--axis", nargs="+", help="vary only these axes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int)
    p.add_argument(
        "--systems",
        type=int,
        default=1,
        help="2: also check a second system of the bucket for recompiles",
    )
    p.add_argument("--out", default="bench.jsonl")
    p.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p.add_argument(
        "--profile",
        metavar="DIR",
        help="trace extra warm calls of each step into DIR",
    )
    p.add_argument("--xla-dump", metavar="DIR", help="dump optimized HLO text")
    p.add_argument(
        "--chain-methods",
        nargs="*",
        default=["sequential", "vectorized", "parallel"],
    )
    p.add_argument("--chain-method", default="vectorized")  # nuts-child
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail on recompiles in every step, not only the bounded ones",
    )
    p.add_argument("--max-hlo-exponent", type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    suites = args.suite or ["baseline"]
    args.case = args.case or ["A1"]
    if args.profile:
        os.environ["JAX_PROFILE_DIR"] = os.path.abspath(args.profile)
    if "nuts-child" in suites:
        nuts_child(args)
        return 0
    rows, failures = [], []
    for suite in suites:
        r, f = (compile_suite if suite == "compile" else sweep)(args)
        rows += r
        failures += f
    write_rows(rows, args.out, args.format)
    print(f"wrote {len(rows)} rows to {args.out}")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
