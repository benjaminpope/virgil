# Orbit benchmark harness

Times each step of the multi-epoch orbit pipeline on simulated binaries
(design: `automatic_orbits.md`, section 9). Heavy runs go to OzSTAR
(`ozstar_scripts/scripts/orbit_bench_baseline`); locally only `--size tiny`.

```bash
.venv/bin/python -X faulthandler benchmarks/orbits/run_bench.py --size tiny
```

- `cases.py`: manifests (`Case`: seeds and parameters) and `build`, the
  simulator; adversarial cases A1, A2, A5, A6, A7, A9, A10.
- `harness.py`: `measure` (cold call, warm median, backend compile time and
  count, RSS, device memory), `count_compiles`, `hlo_size`.
- `run_bench.py`: steps (`STEPS`), size presets, the scaling sweep, the
  compile suite, output (`bench.jsonl` or `--format csv`), assertions.
  New pipeline steps are added to `STEPS`.
- `jaxprof.py`: opt-in profiling (`--profile DIR`, `--xla-dump DIR`).

A recompile in a warm call is recorded for every step and fails the run for
steps marked `bounded`, or all of them with `--strict`. `rank_orbits` is
not bounded on main: it rebuilds its jitted kernel on every call.
