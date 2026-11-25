# Performance

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

## Timing small NUTS fits: time per step, not per gradient

For a model with a handful of parameters, such as a seven-parameter
astrometric orbit fit (`PositionData`; ten unconstrained numbers, because the
three angles are sampled as `AngleVector` 2-vectors), compare samplers by the
warm time per leapfrog step. With one chain (or `chain_method="sequential"`)
that is the post-warmup sampling time divided by the summed
`extra_fields=("num_steps",)`; `num_steps` covers the collected samples only,
so leave warmup out of the time too. With vectorized or parallel chains the
batched loop runs until the longest tree finishes, so this ratio comes out low
by roughly the number of chains (or devices): time one chain for a per-chain
figure. A single jitted `value_and_grad` call is the wrong figure. It is
dominated by JAX's dispatch of a small pytree, and NUTS never pays it, because
the gradient runs inside NumPyro's compiled `while_loop`.

On an Apple M4 (one thread, float64) the orbit gradient inside a compiled
loop, without the Kepler solve, is about 1.5 µs; the Kepler solve adds about
5 µs per step, and a fixed-length HMC step on the full orbit model takes about
35 µs. About 12–18 µs of that is NumPyro's leapfrog loop, which a Gaussian of
the same size pays too. The rest is the per-site transforms,
log-determinants and prior log-densities of the model wrapper. Recording the
angles as `numpyro.deterministic` sites costs nothing inside the sampler: the
potential sums only the `sample` sites' log-densities, so the wrapper adds
nothing to the gradient program (measured: 36 against 37 µs per step with and
without it). The vector-to-angle conversion itself is part of the likelihood
and is not free.

Sampling the ten unconstrained numbers as one packed vector, with the same
density, took the same step from 35.5 to 26.4 µs in a trial. That changes the
sample-site structure, so it is not in the default path.
