# Performance

## Small NUTS fits: `flat_nuts`

NumPyro's NUTS samples every prior at its own site, so each leapfrog step of a
model with a handful of parameters, such as a seven-parameter orbit fit to
astrometry, spends most of its time in per-site bookkeeping. Compare samplers
by the warm time per leapfrog step, not by a jitted gradient call (which is
dominated by JAX dispatch that NUTS never pays), with one chain.
[`flat_nuts_sampler`][virgil.flat_nuts.flat_nuts_sampler] packs the unconstrained
coordinates into one vector and runs blackjax NUTS on it, for the same posterior
and the same site names.

On an Apple M4 (one thread, float64, 4 vectorized chains, 1000 + 1000
iterations, target acceptance 0.9, tree depth 8), both samplers started from the
same four points (NumPyro through
[`chain_init_params`][virgil.likelihood.chain_init_params]), a warm fit,
warmup included, took 2.9-3.0 s in NumPyro's NUTS and 0.68-0.69 s in
`flat_nuts`. The post-warmup trees were of similar length (113,000-128,000 and
111,000 leapfrog steps in total), the minimum effective sample size was 1650-2630
and 3030-3050, so the minimum ESS per second rose from 550-900 to about 4400. A
fixed-length HMC step on the packed potential alone took 26 against 36 µs.
Two things differ from `numpyro_model`: `init` is in constrained site values
(NumPyro's `init_params` are unconstrained), and `blackjax.window_adaptation`
omits NumPyro's initial step-size search, so chains started from prior draws can
get stuck (a warning says so); give starts near the optimum, as the orbit
pipelines' grid search does.

## Faster first calls: the compilation cache

virgil is built on JAX, which compiles each program the first time it runs. A
first fit in a fresh Python process is dominated by that compile time, not by the
fit itself; later calls in the same process are fast. To keep the compiled
programs between processes, turn on JAX's persistent compilation cache once,
before the first fit:

```python
import virgil

virgil.enable_compilation_cache()  # ~/.cache/virgil/jax, or pass a path
```

The first process fills the cache; later ones read from it, which cut a cold
binary fit by a factor of 2 to 4 in our benchmarks. Results are unchanged.

JAX's default thresholds skip programs that compile quickly or are small, which
covers most of virgil's, so the helper sets
`jax_persistent_cache_min_compile_time_secs` and
`jax_persistent_cache_min_entry_size_bytes` to zero. The directory is
`$VIRGIL_COMPILATION_CACHE` if set. The cache is keyed on the program, JAX
version and hardware, so it is safe to share and to delete. It is off by default.

On a cluster, put it on node-local or per-user scratch storage rather than a
shared path that many jobs write at once.
