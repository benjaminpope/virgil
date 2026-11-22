"""Opt-in persistent JAX compilation cache."""

import os
from pathlib import Path

import jax
from jax._src import compilation_cache

__provisional__ = ("enable_compilation_cache",)


def enable_compilation_cache(path=None):
    """Turn on JAX's persistent compilation cache, tuned for virgil.

    A first call in a fresh process spends most of its time compiling many
    small programs. With the cache on, a later process loads them from disk,
    cutting a cold fit severalfold. It is off unless you call this, and it
    does not change results.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    JAX's defaults skip programs that compile in under a second or are small,
    which is most of virgil's, so this sets both thresholds to zero.

    Parameters
    ----------
    path : str or path-like, optional
        Cache directory. Defaults to ``$VIRGIL_COMPILATION_CACHE`` if set,
        else ``~/.cache/virgil/jax``. It is created if missing.

    Returns
    -------
    pathlib.Path
        The cache directory.

    Notes
    -----
    JAX settles whether to use the cache at its first compile, so this resets
    that decision; programs compiled earlier in the process are not cached.
    """
    chosen = path or os.environ.get("VIRGIL_COMPILATION_CACHE")
    directory = Path(chosen or Path.home() / ".cache" / "virgil" / "jax")
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    compilation_cache.reset_cache()
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return directory
