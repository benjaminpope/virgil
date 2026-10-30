"""The deterministic quicklook notebook of a run.

The notebook is generated from ``run.json`` and ``summary.json`` alone: a
markdown cell before every code cell, one output per code cell, and only
templated prose (no judgement). Its first code cell reloads the run with
[`load`][virgil.pipeline.load], so it is also the starting point for
refining the analysis by hand.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from . import _io
from ._checks import format_sigma

# Annuli of the contrast curve for runs that stored no ``limit_bins``.
_LIMIT_BINS = 20
_STATUS_WORD = {"pass": "pass", "warn": "**warn**", "fail": "**FAIL**"}
_OPENING = {
    "pass": "Every check passed.",
    "warn": "Some checks warn: read their messages before using the numbers.",
    "fail": "At least one check failed: do not use these numbers as they stand.",
}


def _nuts_sentence(post):
    return (
        f"NUTS drew {post['num_samples']} samples in each of "
        f"{post['num_chains']} chains after {post['num_warmup']} warm-up steps"
    )


_KERNELSPEC = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}


def _completed(run):
    """Names of the stages that completed."""
    return {
        n for n, e in run["stages"].items() if e.get("status") == "complete"
    }


def _data_cell(has_data):
    """The overview of the fitted data (shared by both pipelines)."""
    return (
        "## Data\n\nThe overview shows the uv coverage, the visibilities "
        "and the phases of the data that were fitted, after any "
        "wavelength selection and error floor.",
        "data = res.data()\nfig, _ = plotting.plot_oidata_overview(data)"
        if has_data
        else 'display(Image("plots/overview_data.png"))',
    )


def _corner_code(sites):
    """Code of the corner plot of the sampled ``sites`` (a summary key)."""
    return (
        "import pandas as pd\n\n"
        f"params = list({sites})\n"
        "samples = res.samples(group_by_chain=False)\n"
        "frame = pd.DataFrame({p: samples[p].astype(float) for p in params})\n"
        '_, corner, walks = plotting.plot_chainconsumer_diagnostics({"posterior": frame}, columns=params)\n'
        "plt.close(walks)"
    )


def _binary_cells(run, summary, path):  # deliberate: dispatch-table callable
    done = _completed(run)
    has_data = (path / "data" / "processed.oifits").exists()
    cells = []
    if "overview" in done:
        cells.append(_data_cell(has_data))
    if {"search", "limits"} <= done:
        search, limits = summary["search"], summary["limits"]
        # The curve of the limits stage: same annuli as the saved plot.
        bins = run["config"].get("limit_bins", _LIMIT_BINS)
        cap = search.get("alias_cap_mas")
        capped = (
            f" The search box is capped at ±{cap:.0f} mas, the alias limit of "
            "the uv lattice; set `max_sep_mas` to search further out."
            if cap
            else ""
        )
        cells.append(
            (
                "## Search and limits\n\n"
                "The left panel is the Δχ² of a point companion over the grid, "
                f"peaking at Δχ² = {search['delta_chi2']:.3g} "
                f"({format_sigma(search['local_nsigma'])} local, "
                f"{format_sigma(search['global_nsigma'])} after the "
                f"comparison with {search['n_null']} simulated companion-free "
                f"searches).{capped} "
                f"The right panel is the {limits['sigma']:g}σ contrast curve "
                "(Absil method), the median limit in annuli about the star.",
                "grid = res.grid()\n"
                "fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))\n"
                'plotting.plot_grid_map(grid["delta_chi2"], grid["axes"], '
                'kind="loglike", label="Δχ²", title="Δχ²", ax=left)\n'
                'plotting.plot_contrast_curve(grid["limit_flux"], grid["axes"], '
                f"sigma={limits['sigma']!r}, r_max={search['max_sep_mas']!r}, "
                f"bins={bins!r}, ax=right)\n"
                "fig.tight_layout()",
            )
        )
    if "fit" in done:
        posterior = "posterior" in done
        chi2 = summary["chi2"]
        source = "posterior" if posterior else "fit"
        samples = (
            "samples = res.samples(group_by_chain=False)\n"
            "step = max(1, samples[params[0]].size // 100)\n"
            "draws = {p: samples[p][::step] for p in params}\n"
            if posterior
            else "draws = {p: [fit.values[p]] for p in params}\n"
        )
        if has_data:
            code = (
                "data = res.data()\nfit = res.fit_result()\n"
                'params = list(res.summary["fit"]["params"])\n'
                f"{samples}"
                "pred = posterior_predictive_summary(draws, fit.model, data, params)\n"
                f'fig = plotting.plot_data_model_correlation(data, {{"{source}": pred}})'
            )
        else:
            code = 'display(Image("plots/fit_correlation.png"))'
        cells.append(
            (
                "## Fit quality\n\n"
                f"On the quoted errors, χ²/N = {chi2['companion_reduced']:.3g} for "
                f"the best-fit binary and {chi2['null_reduced']:.3g} for the star "
                f"alone, over N = {chi2['n_independent']} independent observables. "
                "The correlation plot compares the model "
                f"({source} predictive mean) with the data.",
                code,
            )
        )
    if "posterior" in done:
        post = summary["posterior"]
        cells += [
            (
                "## Posterior\n\n"
                f"{_nuts_sentence(post)}. The corner plot shows the fitted parameters.",
                _corner_code('res.summary["fit"]["params"]'),
            ),
            (
                "The sampler diagnostics per parameter: median with its 16th and "
                "84th percentiles, split R-hat and bulk effective sample size, "
                f"with {100 * post['divergence_fraction']:.2g} per cent divergent "
                "transitions overall.",
                'for name, row in res.summary["posterior"]["params"].items():\n'
                "    print(\n"
                "        f\"{name:>6}: {row['median']:.5g} "
                "[{row['q16']:.5g}, {row['q84']:.5g}]  \"\n"
                "        f\"R-hat {row['r_hat']:.4f}  ESS {row['ess_bulk']:.0f}\"\n"
                "    )",
            ),
        ]
    return cells


def _star_cells(run, summary, path):  # deliberate: dispatch-table callable
    done = _completed(run)
    has_data = (path / "data" / "processed.oifits").exists()
    cells = []
    if "overview" in done:
        cells.append(_data_cell(has_data))
    if "fit" in done:
        posterior = "posterior" in done
        fit, star = summary["fit"], summary["star"]
        chi2 = ", ".join(
            f"{k.replace('_', ' ')} {v:.3g}"
            for k, v in summary["chi2"]["reduced"].items()
        )
        reach = summary.get("resolution")
        reach = (
            f" The longest baseline reaches {reach['first_null_fraction']:.3g} "
            "of the first null."
            if reach
            else ""
        )
        lobes = _io.lobe_text(summary)
        reach += f" Diameter lobes: {lobes}." if lobes else ""
        names = list(fit["models"])
        cells.append(
            (
                "## Result\n\n"
                "The squared visibilities against spatial frequency, with the "
                "fitted model curves. On the quoted errors, χ²/N is "
                f"{chi2}, over N = {summary['chi2']['n_independent']} "
                "independent observables. The preferred model is "
                f"`{fit['best']}`, with a diameter of {star['diam_mas']:.4g} "
                f"mas (from the {star['source']}).{reach}",
                "import virgil.pipeline.star as star\n\n"
                "data = res.data()\n"
                f"fitted = {{name: res.model(name) for name in {names!r}}}\n"
                "fig = star.plot_v2_models(data, fitted)"
                if has_data
                else 'display(Image("plots/fit_v2.png"))',
            )
        )
        source = "posterior" if posterior else "fit"
        if posterior:
            draws = (
                'params = list(res.summary["posterior"]["sampled"])\n'
                "samples = res.samples(group_by_chain=False)\n"
                "step = max(1, samples[params[0]].size // 100)\n"
                "draws = {p: samples[p][::step] for p in params}\n"
            )
        else:
            draws = (
                'params = [p for p in fit.values if not p.startswith("noise.")]\n'
                "draws = {p: [fit.values[p]] for p in params}\n"
            )
        cells.append(
            (
                "## Fit quality\n\n"
                "The correlation plot compares the preferred model "
                f"({source} predictive mean) with the data.",
                "data = res.data()\n"
                "fit = res.fit_result()\n"
                f"{draws}"
                "pred = posterior_predictive_summary(draws, fit.model, data, params)\n"
                f'fig = plotting.plot_data_model_correlation(data, {{"{source}": pred}})'
                if has_data
                else 'display(Image("plots/fit_correlation.png"))',
            )
        )
    if "posterior" in done:
        post = summary["posterior"]
        cells += [
            (
                "## Posterior\n\n"
                f"{_nuts_sentence(post)}, for each fitted model. The corner plot shows the "
                f"preferred model, `{post['best']}`.",
                _corner_code('res.summary["posterior"]["sampled"]'),
            ),
            (
                "The sampler diagnostics per model and parameter: median with "
                "its 16th and 84th percentiles, split R-hat and bulk effective "
                "sample size (derived quantities have none), with "
                f"{100 * post['divergence_fraction']:.2g} per cent divergent "
                "transitions in the worst model.",
                'for model, rows in res.summary["posterior"]["models"].items():\n'
                '    for name, row in rows["params"].items():\n'
                "        diag = (\n"
                "            f\"R-hat {row['r_hat']:.4f}  ESS {row['ess_bulk']:.0f}\"\n"
                "            if 'r_hat' in row\n"
                "            else 'derived'\n"
                "        )\n"
                "        print(\n"
                "            f\"{model:>13} {name:>5}: {row['median']:.5g} \"\n"
                "            f\"[{row['q16']:.5g}, {row['q84']:.5g}]  {diag}\"\n"
                "        )",
            ),
        ]
    return cells


# deliberate: dispatch-table callable (_SECTIONS)
def _disk_cells(run, summary, path):
    done = _completed(run)
    has_data = (path / "data" / "processed.oifits").exists()
    cells = []
    if "overview" in done:
        cells.append(_data_cell(has_data))
    if "geometry" in done:
        geo = summary["geometry"]
        cells.append(
            (
                "## Geometry\n\n"
                "The Δχ² of a single ring over the inclination and position "
                "angle (clipped), with the inclination uniform in cos *i*. "
                "The geometry grid has "
                f"{geo['grid']['n_minima']} local minimum"
                f"{'' if geo['grid']['n_minima'] == 1 else 's'}; the "
                f"single-ring fit from the best cell has *i* = "
                f"{geo['inc_deg']:.4g}° and PA = {geo['pa_deg']:.4g}°.",
                'display(Image("plots/geometry_grid.png"))',
            )
        )
    if "compare" in done:
        disk, model = summary["disk"], summary["model"]
        chi2 = ", ".join(
            f"{v:.3g}" for v in summary["chi2"]["reduced"].values()
        )
        what = (
            f"{disk['n_rings']} ring{'' if disk['n_rings'] == 1 else 's'}"
            if disk["model"] == "rings"
            else f"a Gaussian-process profile of ℓ = {disk['ell_mas']:.3g} mas"
        )
        cells.append(
            (
                "## Result\n\n"
                f"The preferred model is {what} (`{disk['best']}`), with "
                f"χ²/N = {chi2} on the quoted errors over N = "
                f"{summary['chi2']['n_independent']} independent observables. "
                "The panels are the model, the disk alone, its radial "
                "profile, the spectral index, the lopsidedness and the "
                "normalized residuals.",
                "import virgil.pipeline.disk as disk\n\n"
                "data = res.data()\n"
                "fig = disk.plot_summary(\n"
                f"    res.model(), data, r_out_mas={model['r_out_mas']!r},\n"
                f"    r_in_mas={model['r_in_mas']!r},\n"
                ")"
                if has_data
                else 'display(Image("plots/disk_summary.png"))',
            )
        )
    if "posterior" in done and "posterior" in summary:
        cells.append(
            (
                "## Posterior\n\n"
                f"{_nuts_sentence(summary['posterior'])}, with the geometry "
                "released. The scalar sites: median with its 16th and 84th "
                "percentiles, split R-hat and bulk effective sample size.",
                'for name, row in res.summary["posterior"]["params"].items():\n'
                "    print(\n"
                "        f\"{name:>12}: {row['median']:.5g} \"\n"
                "        f\"[{row['q16']:.5g}, {row['q84']:.5g}]  \"\n"
                "        f\"R-hat {row['r_hat']:.4f}  ESS {row['ess_bulk']:.0f}\"\n"
                "    )",
            )
        )
    return cells


_IMAGE_CAVEAT = (
    "> **What this run can and cannot claim.** The image fits the data on "
    "the quoted errors. It does **not** promise that the morphology is "
    "right: data-fidelity checks alone passed contest reconstructions that "
    "were judged terrible (blobby, or a chain of dots), and none of the "
    "checks below measures whether an image looks like the source.\n\n"
)


def _image_cells(run, summary, path):  # deliberate: dispatch-table callable
    done = _completed(run)
    cells = []
    if "overview" in done:
        cells.append(_data_cell((path / "data" / "processed.oifits").exists()))
    if "combine" in done:
        chi2 = ", ".join(
            f"{k} {v:.3g}"
            for k, v in summary["fit"]["chi2_red_by_observable"].items()
        )
        cells.append(
            (
                "## Image\n\nRaw χ²/N on the quoted errors: " + chi2 + ".",
                "img = res.image()\n"
                "h = 0.5 * img.pixels.shape[0] * img.pixel_scale_mas\n"
                "x, y = img.dra, img.ddec\n"
                "plt.imshow(img.pixels, cmap='inferno',\n"
                "           extent=(x + h, x - h, y - h, y + h))\n"
                "plt.colorbar(label='relative flux per pixel')",
            )
        )
    return cells


def _orbit_cells(run, summary, path):  # deliberate: dispatch-table callable
    done = _completed(run)
    cells = []
    if "epochs" in done:
        nights = summary["nights"]
        cells.append(
            (
                "## Nights\n\n"
                f"{len(nights)} nights, each with the companion's best "
                "position (separation, flux ratio), the margin of its best "
                "peak over its best rival (`gap_marginal`, nats), the flags "
                "that stop it seeding an orbit, and the raw χ²/N of the "
                "best orbit.",
                'import pandas as pd\n\npd.DataFrame(res.summary["nights"])',
            )
        )
    if "search" in done:
        search = summary["search"]
        low, high = search["p_range"]
        cells.append(
            (
                "## Alias search\n\n"
                f"{search['n_bands']} alias bands tile the period window "
                f"{low:.4g}–{high:.4g} d, which the data set (no published "
                "period was used). The left panel is each band's evidence "
                "against the best; the right panel its raw χ²/ν.",
                'display(Image("plots/search_bands.png"))',
            )
        )
    if "modes" in done:
        modes = summary["modes"]
        win, null = modes["winner"], modes["no_orbit"]
        gap = modes["delta_log_z"]
        margin = (
            "there is no other band"
            if gap is None
            else f"it leads the runner-up by {gap:.4g} nats"
        )
        cells.append(
            (
                "## Modes\n\n"
                f"The winner is the band of N = {win['n']} orbits "
                f"(probability {win['p']:.3g}); {margin}. Against a single "
                f"star it gains {null['delta_log_z']:.4g} nats "
                f"(verdict: {null['verdict']}). "
                "The sky plot shows the best "
                "orbit, each night's position and error ellipse (hollow if "
                "flagged), and the orbit at the nights.",
                'display(Image("plots/modes_sky.png"))\n'
                "import pandas as pd\n\n"
                'pd.DataFrame(res.summary["modes"]["table"])',
            )
        )
    if "posterior" in done and "modes" in summary.get("posterior", {}):
        post = summary["posterior"]
        sampled = [m["n"] for m in post["modes"]]
        sites = list(post["modes"][0]["params"]) if sampled else []
        if sampled:
            cells.append(
                (
                    "## Posterior\n\n"
                    f"NUTS drew {post['num_samples']} samples in each of "
                    f"{post['num_chains']} chains after "
                    f"{post['num_warmup']} warm-up steps in each of the "
                    f"bands N = {sampled}. The corner plot shows the "
                    "winner's elements; the other bands are in "
                    "`res.samples(name='n<N>')`.",
                    _corner_code(sites),
                )
            )
    return cells


_SECTIONS = {
    "binary": _binary_cells,
    "disk": _disk_cells,
    "image": _image_cells,
    "orbit": _orbit_cells,
    "star": _star_cells,
}


def build_notebook(path):
    """The quicklook notebook of the run in ``path``, unexecuted."""
    nbformat = _io.require_extra("nbformat")
    path = Path(path)
    run = _io.read_json(path / "run.json")
    summary = _io.read_json(path / "summary.json")
    v4 = nbformat.v4
    inputs = run["inputs"]
    if inputs["files"]:
        files = "\n".join(
            f"- `{Path(f['path']).name}` (sha256 `{f['sha256'][:12]}…`)"
            for f in inputs["files"]
        )
    else:
        files = (
            "- data passed in memory (observables sha256 "
            f"`{inputs['data_sha256'][:12]}…`)"
        )
    stages = ", ".join(
        f"`{name}` ({entry.get('seconds', 0):.1f} s)"
        for name, entry in run["stages"].items()
        if entry.get("status") == "complete"
    )
    settings = ", ".join(
        f"`{k}={f'{v:.4g}' if isinstance(v, float) else v}`"
        for k, v in run["config"].items()
    )
    prov = run["provenance"]
    note = _IMAGE_CAVEAT if run["pipeline"] == "image" else ""
    intro = (
        note + f"# Quicklook: {run['class']}\n\n"
        f"This notebook was written by the virgil `{run['pipeline']}` pipeline "
        f"({run['stability']}, schema `{run['schema']}`) and fitted a "
        f"`{run['model']['class']}` template.\n\n"
        f"Inputs:\n\n{files}\n\n"
        f"Stages run: {stages or 'none'}.\n\n"
        f"Settings: {settings}.\n\n"
        f"Versions: virgil {prov['virgil']}, JAX {prov['jax']}, numpyro "
        f"{prov['numpyro']}, on {prov['backend']} (x64 "
        f"{'on' if prov['x64'] else 'off'}).\n\n"
        "The first cell reloads the run from this folder, without recomputing "
        "anything, and prints its key numbers."
    )
    cells = [
        v4.new_markdown_cell(intro),
        v4.new_code_cell(
            "import warnings\n\n"
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            "# Third-party import and load-time warnings would add a stderr\n"
            "# output to this cell; virgil's own warnings still show.\n"
            'warnings.simplefilter("ignore")\n'
            'warnings.filterwarnings("default", module="virgil")\n\n'
            "import virgil.pipeline as vp\n"
            "from virgil import plotting\n"
            "from virgil.likelihood import posterior_predictive_summary\n\n"
            'res = vp.load(".")\n'
            "print(res.describe())"
        ),
    ]
    for markdown, code in _SECTIONS[run["pipeline"]](run, summary, path):
        cells.append(v4.new_markdown_cell(markdown))
        cells.append(v4.new_code_cell(code))
    checks = summary.get("checks", [])
    rows = "\n".join(
        f"| `{c['name']}` | {_STATUS_WORD[c['status']]} | {c['message']} |"
        for c in checks
    )
    closing = (
        "## Checks\n\n"
        f"{_OPENING[summary.get('worst_check', 'pass')]}\n\n"
        "| Check | Status | Message |\n|---|---|---|\n"
        f"{rows}\n\n"
        "Each message is a stock sentence chosen by the check's status; see "
        "the pipeline reference for the thresholds."
    )
    cells.append(v4.new_markdown_cell(closing))
    nb = v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = _KERNELSPEC
    nb.metadata["virgil"] = {"schema": _io.SCHEMA, "pipeline": run["pipeline"]}
    return nb


# The quicklook is light: the kernel runs on the CPU, so it neither fights
# the parent process for the GPU nor preallocates its memory.
_KERNEL_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


def execute_notebook(nb, path, timeout=600):
    """Execute ``nb`` in a fresh kernel of this Python, with cwd ``path``.

    The kernel runs on the CPU, and its own stderr (the ``IPKernelApp``
    banner, XLA logging) is written to ``quicklook.log`` in ``path``
    instead of the caller's terminal.
    """
    nbclient = _io.require_extra("nbclient")
    _io.require_extra("ipykernel")
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager

    name = "virgil-pipeline"
    with tempfile.TemporaryDirectory() as tmp:
        kernel_dir = Path(tmp) / name
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "virgil pipeline",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name=name,
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]),
        )
        client = nbclient.NotebookClient(
            nb,
            km=manager,
            kernel_name=name,
            timeout=timeout,
            resources={"metadata": {"path": str(path)}},
        )
        # MPLBACKEND=Agg (batch jobs) would embed no figures in the notebook
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        env.update(_KERNEL_ENV)
        with open(Path(path) / "quicklook.log", "wb") as log:
            client.execute(env=env, stderr=log)
    nb.metadata["kernelspec"] = _KERNELSPEC
    return nb


def write_quicklook(path, execute=True, timeout=600):
    """Build (and by default execute) ``quicklook.ipynb`` in ``path``."""
    nbformat = _io.require_extra("nbformat")
    path = Path(path)
    nb = build_notebook(path)
    if execute:
        nb = execute_notebook(nb, path, timeout=timeout)
    with _io.atomic_path(path / "quicklook.ipynb") as tmp:
        nbformat.write(nb, str(tmp))
    return path / "quicklook.ipynb"
