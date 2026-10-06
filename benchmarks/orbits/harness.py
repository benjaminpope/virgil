"""Timing, memory and compile-count machinery for the orbit benchmarks.

``measure(fn)`` calls ``fn`` once cold (compile + run), then ``repeats``
times warm, and returns a ``Measurement``: wall times (``block_until_ready``
on everything returned), the XLA backend compile time and count of each
phase (from ``jax.monitoring``, so a step that recompiles on every call
shows it in ``n_compiles_warm``), host RSS and device memory.

``compile_s`` is reported two ways: ``compile_s`` is the backend compile
time seen by the monitor during the cold call, and ``first_minus_warm_s`` is
the cold wall time minus the warm median (which also includes tracing).
A step whose kernels were already compiled earlier in the process shows
``n_compiles_first == 0``: order the sweep, or use a fresh process, when a
true cold compile is wanted.
"""

import contextlib
import dataclasses
import logging
import os
import re
import resource
import statistics
import sys
import threading
import time

import jax
import numpy as onp


class RecompileError(AssertionError):
    """A step compiled when its design says it must not."""


class CompileClock:
    """Counts XLA backend compiles and sums their time, process-wide."""

    _instance = None

    def __init__(self):
        self.seconds = 0.0
        self.count = 0
        self.names = []
        jax.monitoring.register_event_duration_secs_listener(self._on_event)
        self._log = _CompileLog(self)
        logger = logging.getLogger("jax")
        logger.addHandler(self._log)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _on_event(self, event, duration, **kwargs):
        if event.endswith("backend_compile_duration"):
            self.seconds += duration
            self.count += 1

    def snapshot(self):
        return self.seconds, self.count, len(self.names)


class _CompileLog(logging.Handler):
    """Collects the names jax.log_compiles reports (best effort)."""

    def __init__(self, clock):
        super().__init__(level=logging.DEBUG)
        self.clock = clock

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = re.match(r"Compiling (\S+)", msg)
        if m:
            self.clock.names.append(m.group(1))


@contextlib.contextmanager
def count_compiles():
    """Yield a dict filled on exit with ``n`` compiles, ``seconds`` and
    the ``names`` of the compiled functions (when jax logs them).

    >>> with count_compiles() as c:
    ...     pass
    """
    clock = CompileClock.get()
    s0, n0, k0 = clock.snapshot()
    out = {}
    with jax.log_compiles(True):
        try:
            yield out
        finally:
            s1, n1, k1 = clock.snapshot()
            out.update(n=n1 - n0, seconds=s1 - s0, names=clock.names[k0:k1])


def assert_no_compiles(what, fn, *args, **kwargs):
    """Call ``fn`` and raise ``RecompileError`` if it compiles anything."""
    with count_compiles() as c:
        out = fn(*args, **kwargs)
        block(out)
    if c["n"]:
        raise RecompileError(
            f"{what} compiled {c['n']} time(s): {sorted(set(c['names']))}"
        )
    return out


def block(tree):
    """``block_until_ready`` on every array in a result, looking inside
    tuples, lists, dicts and dataclasses."""
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        for f in dataclasses.fields(tree):
            block(getattr(tree, f.name))
    elif isinstance(tree, dict):
        for v in tree.values():
            block(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            block(v)
    elif hasattr(tree, "block_until_ready"):
        tree.block_until_ready()
    else:
        for leaf in jax.tree_util.tree_leaves(tree):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
    return tree


def current_rss_mb():
    """Resident set size now, in MB (Linux /proc; elsewhere the peak)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except OSError:
        return peak_rss_mb()


def peak_rss_mb():
    """Process peak RSS in MB (``ru_maxrss`` is bytes on macOS, KB on
    Linux). It is monotone over the process: use ``RssWatcher`` for a step."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


class RssWatcher:
    """Samples current RSS from a thread, so the peak during one phase
    (e.g. the compile) is separate from the process peak."""

    def __init__(self, interval=0.02):
        self.interval, self.peak = interval, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, current_rss_mb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = current_rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, current_rss_mb())


def device_peak_mb():
    """Peak device memory in MB, or None where the backend gives none (CPU)."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return None if peak is None else peak / 1e6


@dataclasses.dataclass
class Measurement:
    first_s: float
    run_s: float  # warm median
    run_all_s: list
    compile_s: float  # backend compile time in the cold call
    first_minus_warm_s: float
    n_compiles_first: int
    n_compiles_warm: int
    compile_names_warm: list
    peak_rss_mb: float  # process peak after the step
    compile_peak_rss_mb: float  # sampled during the cold call
    peak_device_mb: object
    result: object = None

    def row(self):
        d = dataclasses.asdict(self)
        d.pop("result")
        d["systems_per_hour"] = 3600.0 / self.run_s if self.run_s > 0 else None
        return d


def measure(fn, repeats=5, profile_trace=None):
    """Cold call, then ``repeats`` warm calls of ``fn()``.

    ``profile_trace``: an optional context-manager factory (see
    ``run_bench.py --profile``) wrapped around extra warm calls, after the
    timed ones, so the trace holds no compile and does not bias the timings.
    """
    with count_compiles() as cold, RssWatcher() as watch:
        t0 = time.perf_counter()
        result = block(fn())
        first = time.perf_counter() - t0
    warm_times = []
    with count_compiles() as warm:
        for _ in range(repeats):
            t0 = time.perf_counter()
            block(fn())
            warm_times.append(time.perf_counter() - t0)
    if profile_trace is not None:
        with profile_trace():
            for i in range(3):
                with jax.profiler.StepTraceAnnotation("step", step_num=i):
                    block(fn())
    run = statistics.median(warm_times) if warm_times else first
    return Measurement(
        first_s=first,
        run_s=run,
        run_all_s=warm_times,
        compile_s=cold["seconds"],
        first_minus_warm_s=max(first - run, 0.0),
        n_compiles_first=cold["n"],
        n_compiles_warm=warm["n"],
        compile_names_warm=sorted(set(warm["names"])),
        peak_rss_mb=peak_rss_mb(),
        compile_peak_rss_mb=watch.peak,
        peak_device_mb=device_peak_mb(),
        result=result,
    )


def hlo_size(fn, *args):
    """Instruction count of the optimized HLO module of ``jax.jit(fn)``
    on ``args``, and the seconds ``lower().compile()`` took."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    seconds = time.perf_counter() - t0
    text = compiled.as_text()
    n = sum(1 for line in text.splitlines() if re.match(r"\s+\S+ = ", line))
    return n, seconds


def growth_exponent(xs, sizes):
    """Slope of log(size) against log(x): 1 is linear growth, near 0 is
    constant, logarithmic growth gives a small and falling slope."""
    lx, ls = (
        onp.log(onp.asarray(xs, float)),
        onp.log(onp.asarray(sizes, float)),
    )
    return float(onp.polyfit(lx, ls, 1)[0])
