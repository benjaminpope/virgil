"""Machinery shared by the grid searches and the contrast limits (private).

Both [`virgil.grid_fit`][virgil.grid_fit] and
[`virgil.limits`][virgil.limits] import from here, so neither
depends on the other's internals.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ._utils import concrete, is_flux_param, resolve_flux_param


# Grid points are evaluated in batches, which bounds memory on large grids:
# each point holds a full model evaluation and, in the optimizers, a BFGS
# state, so memory per point grows with the number of model visibilities
# (measured on CPU: ~25 bytes per visibility for a float32 likelihood grid,
# up to ~450 for a float64 flux optimization of a three-component System).
# By default a batch holds about batch_visibilities() model visibilities,
# and never fewer than MIN_BATCH_SIZE points, the earlier fixed default, so
# large data use as little memory as before. Each batch is a loop iteration
# with fixed overheads (on a GPU, a few kernel launches), so small data run
# much faster in large batches. Warm grid times stop improving at ~1e6
# visibilities per batch on an M4 CPU and ~1e7 on an A100 GPU, which is
# 10-50x faster there than at 256 points for 24 baselines (2026-10-03,
# ozstar_scripts drpangloss_prof PROF=batch). The public functions take
# ``batch_size=`` to override it.
CPU_BATCH_VISIBILITIES = 2**20
ACCELERATOR_BATCH_VISIBILITIES = 2**23
MIN_BATCH_SIZE = 256


def batch_visibilities():
    """Model visibilities per batch by default, for JAX's default backend."""
    if jax.default_backend() == "cpu":
        return CPU_BATCH_VISIBILITIES
    return ACCELERATOR_BATCH_VISIBILITIES


def batch_size_or_default(batch_size, data_obj):
    """Validate a ``batch_size`` argument, or choose one for ``data_obj``.

    The default holds about ``batch_visibilities()`` model visibilities per
    batch, and at least ``MIN_BATCH_SIZE`` grid points.
    """
    if batch_size is None:
        n_vis = max(np.size(data_obj.u), np.size(data_obj.wavel), 1)
        return max(MIN_BATCH_SIZE, batch_visibilities() // n_vis)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    return batch_size


def map_points(fn, *xs, batch_size):
    """Apply ``fn`` to every row of ``xs``, ``batch_size`` rows at a time."""
    return jax.lax.map(lambda args: fn(*args), xs, batch_size=batch_size)


def check_flux_axes(samples_dict, flux_param=None):
    """Reject grid axes that would give a flux parameter negative values.

    Axes whose name ends in ``flux`` are checked, as is the explicitly
    selected ``flux_param`` whatever its name.
    """
    for key, values in samples_dict.items():
        if not (is_flux_param(key) or key == flux_param):
            continue
        values = concrete(values)
        if values is not None and np.any(values < 0.0):
            raise ValueError(
                f"The grid axis {key!r} contains negative values, but fluxes "
                "must be non-negative."
            )


def resolve_grid_keys(samples_dict, flux_param=None):
    """Return ``(params, coord_keys, flux_key)`` for a grid-fitting call."""
    params = tuple(samples_dict.keys())
    check_flux_axes(samples_dict, flux_param)
    flux_key = resolve_flux_param(params, flux_param)
    coord_keys = tuple(key for key in params if key != flux_key)
    if not coord_keys:
        raise ValueError(
            "samples_dict needs at least one coordinate parameter besides "
            f"the flux {flux_key!r}."
        )
    return params, coord_keys, flux_key


def meshgrid_vectors(samples_dict, params):
    """Build flattened meshgrid vectors with axis order matching ``params``."""
    samples = [jnp.asarray(samples_dict[param]) for param in params]
    grid_shape = tuple(sample.shape[0] for sample in samples)
    grids = jnp.meshgrid(*samples, indexing="ij")
    vals_vec = jnp.stack([grid.reshape(-1) for grid in grids], axis=1)
    return vals_vec, grid_shape


def ordered_values(flux, coord_vals, params, coord_keys, flux_key):
    """Build parameter values in ``params`` order without traced dict objects."""
    flux_value = jnp.asarray(flux).reshape(-1)[0]
    coord_vals = jnp.asarray(coord_vals)
    return [
        flux_value
        if param == flux_key
        else coord_vals[coord_keys.index(param)]
        for param in params
    ]


def first_crossing(
    reached, log_start, log_low, log_high, max_steps, bisections
):
    """Bracket a crossing in log flux by decades, then bisect it.

    ``reached(log_flux)`` says whether the target (e.g. a significance) is
    reached at ``10**log_flux``. From ``log_start`` (inside
    ``[log_low, log_high]``, which may be infinite), the flux steps up by
    decades while the target is not reached, or down while it is, clamped
    to the range and for at most ``max_steps`` steps, until ``reached``
    changes. The last step is then bisected ``bisections`` times. Starting
    below the target, this finds its first crossing above ``log_start``,
    even where the target is not reached again at higher flux.

    Returns ``(log_limit, crossed)``. Without a crossing, ``log_limit`` is
    where the search stopped (an end of the range, or ``max_steps``
    decades away) and ``crossed`` is False; ``reached(log_start)`` then
    tells which end.
    """
    above = reached(log_start)
    direction = jnp.where(above, -1.0, 1.0)
    edge = jnp.where(above, log_low, log_high)

    def keep_going(state):
        _, log_flux, same_side, n = state
        return same_side & (log_flux != edge) & (n < max_steps)

    def step(state):
        _, log_flux, _, n = state
        new = jnp.clip(log_flux + direction, log_low, log_high)
        return log_flux, new, reached(new) == above, n + 1

    near, stop, same_side, _ = jax.lax.while_loop(
        keep_going, step, (log_start, log_start, jnp.asarray(True), 0)
    )

    # `near` is on the starting side of the target and `far` on the other.
    def bisect(_, edges):
        near, far = edges
        mid = 0.5 * (near + far)
        same = reached(mid) == above
        return jnp.where(same, mid, near), jnp.where(same, far, mid)

    near, far = jax.lax.fori_loop(0, bisections, bisect, (near, stop))
    crossed = ~same_side
    return jnp.where(crossed, 0.5 * (near + far), stop), crossed


def coordinate_points(samples_dict, coord_keys):
    """Flattened ``(n_points, n_coords)`` coordinate grid and its shape.

    The grid uses ``indexing="ij"``: axis ``k`` follows ``coord_keys[k]``,
    so for ``(dra, ddec)`` axis 0 is ``dra``.
    """
    coord_grids = jnp.meshgrid(
        *[jnp.asarray(samples_dict[key]) for key in coord_keys],
        indexing="ij",
    )
    points = jnp.stack([grid.reshape(-1) for grid in coord_grids], axis=1)
    return points, coord_grids[0].shape


def warn_unconverged(success, caller):
    """Warn if an optimizer failed to converge at some grid positions."""
    failed = int(np.sum(~np.asarray(success, dtype=bool)))
    if failed:
        warnings.warn(
            f"{caller}(): the optimizer did not converge at {failed} of "
            f"{np.size(success)} grid positions; values there may be "
            "inaccurate.",
            RuntimeWarning,
            # user -> public function -> here
            stacklevel=3,
        )
