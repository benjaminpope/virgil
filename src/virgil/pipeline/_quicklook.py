"""The deterministic quicklook notebook of a run.

The notebook is generated from ``run.json`` and ``summary.json`` alone: a
markdown cell before every code cell, one output per code cell, and only
templated prose (no judgement). Its first code cell reloads the run with
[`load`][virgil.pipeline.load], so it is also the starting point for
refining the analysis by hand.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

from . import _io

_STATUS_WORD = {"pass": "pass", "warn": "**warn**", "fail": "**FAIL**"}
_OPENING = {
    "pass": "Every check passed.",
    "warn": "Some checks warn: read their messages before using the numbers.",
    "fail": "At least one check failed: do not use these numbers as they stand.",
}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _intro(run, summary):
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
    settings = ", ".join(f"`{k}={_fmt(v)}`" for k, v in run["config"].items())
    prov = run["provenance"]
    return (
        f"# Quicklook: {run['class']}\n\n"
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


def _closing(summary):
    checks = summary.get("checks", [])
    rows = "\n".join(
        f"| `{c['name']}` | {_STATUS_WORD[c['status']]} | {c['message']} |"
        for c in checks
    )
    return (
        "## Checks\n\n"
        f"{_OPENING[summary.get('worst_check', 'pass')]}\n\n"
        "| Check | Status | Message |\n|---|---|---|\n"
        f"{rows}\n\n"
        "Each message is a stock sentence chosen by the check's status; see "
        "the pipeline reference for the thresholds."
    )


def _binary_cells(run, summary, path):
    stages = run["stages"]
    done = {n for n, e in stages.items() if e.get("status") == "complete"}
    has_data = (path / "data" / "processed.oifits").exists()
    cells = []
    if "overview" in done:
        cells.append(
            (
                "## Data\n\nThe overview shows the uv coverage, the visibilities "
                "and the phases of the data that were fitted, after any "
                "wavelength selection and error floor.",
                "data = res.data()\nfig, _ = plotting.plot_oidata_overview(data)"
                if has_data
                else 'display(Image("plots/overview_data.png"))',
            )
        )
    if {"search", "limits"} <= done:
        search = summary["search"]
        limits = summary["limits"]
        cells.append(
            (
                "## Search and limits\n\n"
                "The left panel is the Δχ² of a point companion over the grid, "
                f"peaking at Δχ² = {search['delta_chi2']:.3g} "
                f"({search['local_nsigma']:.2f}σ local, "
                f"{search['global_nsigma']:.2f}σ after the look-elsewhere "
                f"correction for {search['n_trials']:.0f} resolution elements). "
                f"The right panel is the {limits['sigma']:g}σ contrast curve "
                "(Absil method), the median limit in annuli about the star.",
                "grid = res.grid()\n"
                "fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))\n"
                'plotting.plot_grid_map(grid["delta_chi2"], grid["axes"], '
                'kind="loglike", label="Δχ²", title="Δχ²", ax=left)\n'
                'plotting.plot_contrast_curve(grid["limit_flux"], grid["axes"], '
                f"sigma={limits['sigma']!r}, ax=right)\n"
                "fig.tight_layout()",
            )
        )
    if "fit" in done:
        chi2 = summary["chi2"]
        source = "posterior" if "posterior" in done else "fit"
        samples = (
            "samples = res.samples(group_by_chain=False)\n"
            "step = max(1, samples[params[0]].size // 100)\n"
            "draws = {p: samples[p][::step] for p in params}\n"
            if source == "posterior"
            else "draws = {p: [fit.values[p]] for p in params}\n"
        )
        data_line = "data = res.data()\n" if has_data else ""
        if has_data:
            code = (
                f"{data_line}fit = res.fit_result()\n"
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
        cells.append(
            (
                "## Posterior\n\n"
                f"NUTS drew {post['num_samples']} samples in each of "
                f"{post['num_chains']} chains after {post['num_warmup']} warm-up "
                "steps. The corner plot shows the fitted parameters.",
                "import pandas as pd\n\n"
                'params = list(res.summary["fit"]["params"])\n'
                "samples = res.samples(group_by_chain=False)\n"
                "frame = pd.DataFrame({p: samples[p].astype(float) for p in params})\n"
                '_, corner, walks = plotting.plot_chainconsumer_diagnostics({"posterior": frame}, columns=params)\n'
                "plt.close(walks)",
            )
        )
        cells.append(
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
            )
        )
    return cells


def _star_cells(run, summary, path):
    done = {
        n for n, e in run["stages"].items() if e.get("status") == "complete"
    }
    has_data = (path / "data" / "processed.oifits").exists()
    cells = []
    if "overview" in done:
        cells.append(
            (
                "## Data\n\nThe overview shows the uv coverage, the visibilities "
                "and the phases of the data that were fitted, after any "
                "wavelength selection and error floor.",
                "data = res.data()\nfig, _ = plotting.plot_oidata_overview(data)"
                if has_data
                else 'display(Image("plots/overview_data.png"))',
            )
        )
    if "fit" in done:
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
        source = "posterior" if "posterior" in done else "fit"
        if source == "posterior":
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
        cells.append(
            (
                "## Posterior\n\n"
                f"NUTS drew {post['num_samples']} samples in each of "
                f"{post['num_chains']} chains after {post['num_warmup']} warm-up "
                "steps, for each fitted model. The corner plot shows the "
                f"preferred model, `{post['best']}`.",
                "import pandas as pd\n\n"
                'params = list(res.summary["posterior"]["sampled"])\n'
                "samples = res.samples(group_by_chain=False)\n"
                "frame = pd.DataFrame({p: samples[p].astype(float) for p in params})\n"
                '_, corner, walks = plotting.plot_chainconsumer_diagnostics({"posterior": frame}, columns=params)\n'
                "plt.close(walks)",
            )
        )
        cells.append(
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
            )
        )
    return cells


_SECTIONS = {"binary": _binary_cells, "star": _star_cells}


def build_notebook(path):
    """The quicklook notebook of the run in ``path``, unexecuted."""
    nbformat = _io.require_extra("nbformat")
    path = Path(path)
    run = _io.read_json(path / "run.json")
    summary = _io.read_json(path / "summary.json")
    v4 = nbformat.v4
    cells = [
        v4.new_markdown_cell(_intro(run, summary)),
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
    cells.append(v4.new_markdown_cell(_closing(summary)))
    nb = v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["virgil"] = {"schema": _io.SCHEMA, "pipeline": run["pipeline"]}
    return nb


@contextlib.contextmanager
def _without_mplbackend():
    """Unset ``MPLBACKEND`` so the kernel uses the inline backend.

    Batch jobs often set ``MPLBACKEND=Agg``; a kernel that inherits it
    draws figures without embedding them in the notebook.
    """
    saved = os.environ.pop("MPLBACKEND", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["MPLBACKEND"] = saved


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
    with tempfile.TemporaryDirectory() as tmp, _without_mplbackend():
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
        env = {**os.environ, **_KERNEL_ENV}
        with open(Path(path) / "quicklook.log", "wb") as log:
            client.execute(env=env, stderr=log)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
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
