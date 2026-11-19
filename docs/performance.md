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
