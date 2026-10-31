<!-- AUTO-GENERATED FROM README.md by scripts/sync_tutorial_docs.py. Edit README.md, not this file. -->
# virgil
[![PyPI version](https://badge.fury.io/py/virgil-astro.svg)](https://badge.fury.io/py/virgil-astro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/virgil/actions/workflows/zensical-pages.yml/badge.svg)](https://benjaminpope.github.io/virgil/)
<!-- dev-carbon-badge -->
[![dev carbon | 337 kg CO2e](https://img.shields.io/endpoint?url=https%3A%2F%2Fbenjaminpope.github.io%2Fvirgil%2Fgenerated%2Fdev_carbon_badge.json)](dev_carbon.md)
<!-- /dev-carbon-badge -->

The badge is an estimate with a range, not a measurement; it covers AI assistants, continuous integration and cluster jobs for development, and not data analysis ([details](dev_carbon.md)).

**V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary.

Contributors: [Dori Blakely](https://github.com/blakelyd), [Benjamin Pope](https://github.com/benjaminpope), [Louis Desdoigts](https://github.com/LouisDesdoigts), [Shashank Dholakia](https://github.com/shashankdholakia), [Toon De Prins](https://github.com/DePrinsT), [Jonah Goldfine](https://github.com/JonahDG), [Max Charles](https://github.com/maxecharles).

*facilis descensus averno*

## What is virgil?

virgil models optical interferometry data in JAX. It fits parametric models and reconstructs images in one framework, and runs on GPUs and clusters.

## Stability

Everything public is **stable** (a change needs a deprecation cycle) except a list of **provisional** features that are new in 0.4 and may change in any 0.4.x release without one: orbit fitting across epochs and aliases, the `MultipoleDisk` family, Gaussian-process ensembles, variational inference, the pipelines, sparse-imaging CLEAN, radio visibilities, radio CLEAN and the Toeplitz χ² (`radio`, `radio_clean`, `radio_toeplitz`), cross-validation (`crossval`) and the image-recovery metrics. See the [list of provisional features](stability.md).

## Installation

Install virgil from PyPI, where it is called `virgil-astro`:

```
pip install virgil-astro
```

`pip install virgil` installs an unrelated package. Use Python 3.12 or newer: JAX 0.11 needs it, and Python 3.11 gets the last JAX release that supports it.

Optional extras: `pip install "virgil-astro[plots]"` adds pandas and
ChainConsumer for the corner plots in `virgil.plotting`, and `[legacy]` adds
astroquery for the SIMBAD lookups in `virgil.legacy`, `[radio]` reads CASA
measurement sets for `virgil.radio`, and `[pipeline]` adds the HDF5 outputs and executed
quicklook notebooks of the scriptable pipelines. `virgil.orbits` solves
Kepler's equation with
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet), from
`pip install "virgil-astro[orbits]"`. The module imports without it; only
evaluating an orbit's positions or calling `to_jaxoplanet` needs it, and then
the error names the extra.

The tutorials are in the [GitHub repository](https://github.com/benjaminpope/virgil/tree/main/notebooks), not in the wheel. `pip install "virgil-astro[notebooks,pipeline]"` installs everything they use.

To build from source, clone the repository:

```
git clone --filter=blob:none https://github.com/benjaminpope/virgil
cd virgil
pip install .
```

`--filter=blob:none` makes a partial clone: you get the full history, but old
file versions download only when you ask for them. That skips large data files
no longer in use, so the download is about 15 MB rather than about 280 MB.

Work in a virtual environment. With `uv` (recommended):

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest tests/test_models_core.py -q
```

## Where to start

Documentation is published at [benjaminpope.github.io/virgil](https://benjaminpope.github.io/virgil/). New here? The [quickstart](quickstart.md) loads an OIFITS file and fits a binary in about twenty lines.

| I want to | Go to |
| --- | --- |
| Read my OIFITS file | [Reading data](data_io.md) |
| Find and fit a companion | [Binary search](binary_search.md) |
| Set a contrast limit or a detection limit | [Contrast limits](contrast_limits.md) |
| Fit a binary's orbit | [Which orbit tool?](which_orbit_tool.md), then [Orbits from interferometric data](orbit_fitting.md) |
| Make an image | [Simulating AMI data](imaging_ami.md), then [Regularized maximum likelihood](imaging_rml.md) |
| Run a whole analysis in one command | [Pipelines](pipeline.md) |
| Check which features may change | [Stability](stability.md) |

## Independent validation

virgil's own tests mostly check virgil against itself. The companion repository [virgil-validation](https://github.com/benjaminpope/virgil-validation) checks it against code that shares nothing with it: textbook visibilities for geometric primitives, uv tracks and OIFITS files built from first principles, and aperture-masking images simulated with [dLux](https://github.com/LouisDesdoigts/dLux) for virgil to read and fit. Some provisional features have no independent check yet: the orbit alias search end to end, tempering, the pipelines, CLEAN, bandwidth smearing and harmonix. To ask for another check, [open an Issue there](https://github.com/benjaminpope/virgil-validation/issues) naming the model or function, the independent result it should match, and the precision you expect.

## Collaboration & Development

Contributions are welcome. [CONTRIBUTING.md](https://github.com/benjaminpope/virgil/blob/main/CONTRIBUTING.md) covers setup, testing and the pull-request workflow. Release notes are in the [changelog](https://github.com/benjaminpope/virgil/blob/main/CHANGELOG.md).

## Name

VIRGIL is the **V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary. In Dante's *Divine Comedy*, the poet Virgil is Dante's guide through the Inferno and Purgatory. In Virgil's own *Aeneid*, when Aeneas enters the underworld he draws his sword on the monsters crowding its threshold. His guide, the Cumaean Sibyl, warns him that they are only thin, bodiless lives flitting in a hollow semblance of form (*Aeneid* VI.292–294). Images reconstructed from sparse interferometric data are full of such false visions: artefacts that look like structure but have no substance in the data. VIRGIL helps you tell the difference. The acronym is Jonah Goldfine's.

### Formerly drpangloss

Until version 0.1.1 this package was called **drpangloss**, after Voltaire's Dr Pangloss and as a nod to Antoine Mérand's [CANDID](https://github.com/amerand/CANDID). From version 0.2.0 it is **virgil**: `import virgil`, installed with `pip install virgil-astro`. A final `drpangloss` release (0.2.0) depends on `virgil-astro` and points here, so old installs find the new package.

*e quindi uscimmo a riveder le stelle*
