"""Constants and helpers several test modules used to copy."""

import json
from pathlib import Path

import jax
import numpy as onp

from numpyro.diagnostics import effective_sample_size

from virgil.oidata import OIData, cp_indices


def nufft_problem(shape, n=200, seed=0, max_freq=2e6):
    """A random unit-flux image of ``shape`` and ``n`` random uv points."""
    rng = onp.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, shape)
    image /= image.sum()
    uu, vv = rng.uniform(-max_freq, max_freq, (2, n))
    return image, uu, vv


PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])

STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])

CALIBRATED_VISIBILITY = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)

_H, _C, _K = 6.62607015e-34, 299792458.0, 1.380649e-23


def planck(wavel, temperature):
    """Unnormalized Planck B_λ, written from scratch."""
    return wavel**-5.0 / onp.expm1(_H * _C / (wavel * _K * temperature))


def float_leaves(grads):
    """The floating-point leaves of a gradient tree."""
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(grads)
        if hasattr(leaf, "dtype") and onp.issubdtype(leaf.dtype, onp.floating)
    ]


def vlti_night(
    mjd, frames=1, rotation=0.0, d_vis=0.01, d_phi=0.5, stations=False
):
    """VLTI-like data: ``frames`` four-telescope snapshots 1 h apart.

    ``stations=True`` records the station pair of every sample, which writing
    the data as OIFITS needs."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )
    shift = lambda i: onp.concatenate([i + k * n for k in range(frames)])  # noqa: E731
    extra = {"stations": onp.tile(PAIRS, (frames, 1))} if stations else {}
    return OIData(
        {
            **extra,
            "u": u,
            "v": v,
            "wavel": 2.2e-6,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, d_vis),
            "phi": onp.zeros(len(i1) * frames),
            "d_phi": onp.full(len(i1) * frames, d_phi),
            "i_cps1": shift(i1),
            "i_cps2": shift(i2),
            "i_cps3": shift(i3),
            "mjd": onp.repeat(mjd + onp.arange(frames) / 24.0, n),
        }
    )


def checks_by_name(res):
    return {c.name: c for c in res.checks}


def stage_outputs(pipeline, stage):
    return {s.name: s.outputs for s in pipeline._stages()}[stage]


def started_times(pipeline):
    run = json.loads((pipeline.output / "run.json").read_text())
    return {name: s.get("started") for name, s in run["stages"].items()}


def executed_quicklook(folder, images):
    """Execute the quicklook notebook of the run in ``folder``; check it.

    The run fixtures are built with ``quicklook_execute=False`` (a kernel
    start is the slowest part of a run and only this check needs it); the
    folder is left untouched. The
    cell structure is common to every pipeline: prose before each code
    cell, one output per code cell, none an error, ``images`` figures.
    Returns ``(cells, code)``: the cells and the indices of the code cells.
    """
    from virgil.pipeline import _quicklook

    # In memory: the run folder (module-scoped, hashed into run.json) must
    # stay as the run left it, or a later `validate` would report a change.
    nb = _quicklook.execute_notebook(_quicklook.build_notebook(folder), folder)
    cells = nb["cells"]
    assert (
        cells[0]["cell_type"] == "markdown"
        and cells[-1]["cell_type"] == "markdown"
    )
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    for i in code:
        assert cells[i - 1]["cell_type"] == "markdown"  # prose before code
        assert len(cells[i]["outputs"]) == 1, cells[i]["source"]  # one output
        assert cells[i]["outputs"][0]["output_type"] != "error"
    n_images = sum(
        "image/png" in o.get("data", {})
        for i in code
        for o in cells[i]["outputs"]
    )
    assert n_images == images
    return cells, code


def thinned(draws):
    """``draws`` thinned to about their effective sample size.

    NUTS draws are autocorrelated, and KS assumes independent ones: keep
    every ``n / ESS``-th draw (numpyro's ESS of the single chain).
    """
    draws = onp.asarray(draws, float)
    ess = float(effective_sample_size(draws[None]))
    step = max(1, int(onp.ceil(len(draws) / max(ess, 1.0))))
    return draws[::step]
