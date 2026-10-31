"""Print the outlier component's b_k and its timings (design/orbit_mixture_v2.md
§1.2 and §6.3). Writes no files.

Run from the repository root:
    ~/.claude/bin/pyslot .venv/bin/python scripts/time_orbit_mixture.py
"""

import sys
import time
import warnings

import jax
import numpy as onp

sys.path.insert(0, ".")
from tests import test_aliases as ta  # noqa: E402
from tests import test_night_flags as tn  # noqa: E402
from virgil import aliases, epochs  # noqa: E402
from virgil._precision import cast_tree  # noqa: E402

A_MAX = 12.0  # mas, as in the design's table
S_MAX = 50.0  # the test fixture's bound
FRACTION = 0.25  # of the resolution: the production lattice step


def timed(call):
    start = time.perf_counter()
    out = call()
    return out, time.perf_counter() - start


def main():
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = [cast_tree(d, "float64") for d in tn._with_bad_night(3.0)]
        dof = tn._unit_dof(data)
        flux_range = (0.01, 1.0)
        res = epochs.Epochs(
            {f"n{k}": d for k, d in enumerate(data)}
        ).resolution_mas
        kw = dict(
            a_max=A_MAX, flux_range=flux_range, step=FRACTION * res,
            s_max=S_MAX,
        )  # fmt: skip

        def cold():
            epochs._BROAD_CACHE.clear()
            return epochs.broad_loglike(data, dof, **kw)

        b, t_cold = timed(cold)
        _, t_warm = timed(cold)
        print(
            f"b_k, a_max={A_MAX:g} mas, s_max={S_MAX:g}, step={FRACTION} res"
        )
        print("  night:", *range(len(b)))
        print("  b_k  :", *(f"{v:.3f}" for v in b))
        print(
            f"b_k hand-built positions: cold {t_cold:.2f} s, warm {t_warm:.2f} s"
        )

        positions = epochs.epoch_positions(
            epochs.Epochs({f"n{k}": d for k, d in enumerate(data)}),
            tn.GRID | {"flux": onp.geomspace(*flux_range, epochs._N_FLUX)},
            n_peaks=3,
            s_max=S_MAX,
        )

        def with_positions():
            epochs._BROAD_CACHE.clear()
            return epochs.broad_loglike(data, dof, positions=positions, **kw)

        b2, t_pos = timed(with_positions)
        print(
            f"b_k with positions from epoch_positions: {t_pos:.2f} s "
            f"(level 0 reused if the grids match); max |change| "
            f"{onp.abs(b2 - b).max():.4f} nats"
        )

        settings = ta._settings(
            outlier="mixture", s_max=S_MAX, log_broad=tuple(map(float, b))
        )
        problem = aliases._Problem(data, settings)
        bounds = aliases._Bounds.of(
            aliases.AliasBand(20, 12.0, 12.2), settings
        )
        x0 = ta._x_truth()
        aliases._optimize(problem, bounds, x0, 50)  # compile
        _, t_fit = timed(lambda: aliases._optimize(problem, bounds, x0, 50))
        print(
            f"warm fit (one band, from the truth): {t_fit:.2f} s; b_k with "
            f"positions is {100 * t_pos / t_fit:.1f}% of it"
        )


if __name__ == "__main__":
    main()
