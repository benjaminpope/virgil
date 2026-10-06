"""Opt-in JAX profiling helpers (the jax-profiling procedure).

Tracing is off unless ``JAX_PROFILE_DIR`` is set (or ``--profile DIR`` of
``run_bench.py``, which sets it). Compile outside the trace, call
``block_until_ready`` inside it, and give each iteration a
``StepTraceAnnotation``. Summarize a capture headlessly with
``jax_trace_summary.py`` (ozstar_scripts/scripts/).
"""

import contextlib
import json
import os
import socket
import time

import jax


def profile_dir(tag="run"):
    root = os.environ.get("JAX_PROFILE_DIR")
    if not root:
        return None
    job = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    rank = os.environ.get("SLURM_PROCID", "0")
    return os.path.join(root, f"{tag}_{job}", f"proc{rank}")


@contextlib.contextmanager
def maybe_profile(tag="run", python_tracer=False, host_level=2, perfetto=True):
    d = profile_dir(tag)
    if d is None:
        yield None
        return
    opts = jax.profiler.ProfileOptions()
    opts.python_tracer_level = int(python_tracer)
    opts.host_tracer_level = host_level
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "env.txt"), "w") as f:
        f.write(env_summary())
    with jax.profiler.trace(
        d, create_perfetto_trace=perfetto, profiler_options=opts
    ):
        yield d


def env_summary():
    keys = (
        "XLA_FLAGS",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_PLATFORMS",
        "SLURM_JOB_ID",
    )
    lines = [
        f"host={socket.gethostname()}",
        f"jax={jax.__version__}",
        f"devices={jax.devices()}",
    ]
    lines += [f"{k}={os.environ.get(k, '')}" for k in keys]
    return "\n".join(lines) + "\n"


def step(i, name="step"):
    return jax.profiler.StepTraceAnnotation(name, step_num=i)


def xla_dump_flags(directory, existing=None):
    """``XLA_FLAGS`` text that dumps optimized HLO as text to ``directory``.

    Set it in the environment *before* the first jax import (or before the
    compile of interest in a fresh process): XLA reads it at backend start.
    """
    flags = f"--xla_dump_to={directory} --xla_dump_hlo_as_text"
    return f"{existing} {flags}" if existing else flags
