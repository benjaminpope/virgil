<!-- AUTO-GENERATED FROM README.md by scripts/sync_tutorial_docs.py. Edit README.md, not this file. -->
# virgil
[![PyPI version](https://badge.fury.io/py/virgil-astro.svg)](https://badge.fury.io/py/virgil-astro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![integration](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml/badge.svg)](https://github.com/benjaminpope/virgil/actions/workflows/tests.yml)
[![Documentation](https://github.com/benjaminpope/virgil/actions/workflows/zensical-pages.yml/badge.svg)](https://benjaminpope.github.io/virgil/)

Versatile Interferometric Reconstruction and Gradient-based Inference Library.

Contributors: [Dori Blakely](https://github.com/blakelyd), [Benjamin Pope](https://github.com/benjaminpope), [Louis Desdoigts](https://github.com/LouisDesdoigts), [Shashank Dholakia](https://github.com/shashankdholakia), [Toon De Prins](https://github.com/DePrinsT), [Jonah Goldfine](https://github.com/JonahDG), [Max Charles](https://github.com/maxecharles), [Anand Sivaramakrishnan](https://github.com/anand0xff), [Ian Czekala](https://github.com/iancze) and [Jens Kammerer](https://github.com/kammerje). The [Contributors](contributors.md) page says who did what.

## What is virgil?

virgil is a package for modelling optical interferometry data in JAX.

## Installation

virgil is hosted on PyPI; the easiest way to install it is:

```
pip install virgil-astro
```

Optional extras add the corner-plot helpers in `virgil.plotting`
(`pip install "virgil-astro[plots]"`, for pandas and ChainConsumer) and the
SIMBAD lookups in `virgil.legacy` (`[legacy]`, for astroquery).

You can also build from source. To do so, clone the git repo and enter the directory:

```
git clone --filter=blob:none https://github.com/benjaminpope/virgil
cd virgil
pip install .
```

`--filter=blob:none` makes a partial clone: you get the full history, but old
versions of files are fetched only if you ask for them. It skips large data
files that are no longer used, so the download is about 15 MB rather than
about 280 MB.

We recommend using a virtual environment to avoid dependency conflicts.

Using `uv` (recommended):

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[test]"
uv run --python .venv/bin/python pytest -q
```


## Use & Documentation

Documentation is published at [benjaminpope.github.io/virgil](https://benjaminpope.github.io/virgil/).

### Using these docs

The sections in the sidebar hold worked examples on simulated and bundled data:
- **Background:** [who contributed what](contributors.md), [Gaussian-process priors and information field theory](gp_and_ift.md), and [coordinate, sign and flux conventions](conventions.md).
- **Data Handling:** [reading OIFITS files into `OIData`](data_io.md), and [AMIGO's DISCO data from JWST aperture masking](amigo_disco.md).
- **Binaries:** [searching for companions](binary_search.md), [detection limits](contrast_limits.md), and [fitting several datasets together](hierarchical_inference.md).
- **Sources:** [visibility models](model_syntax.md), [extended sources](source_models.md), [composing scenes](composition.md), [spotted stars](harmonix.md), [limb-darkened stars](limb_darkening.md) and [gravity-darkened stars](gravity_darkened_star.md).
- **Imaging:** image reconstruction in six parts: [simulating data](imaging_ami.md), [regularised maximum likelihood](imaging_rml.md), [Gaussian-process priors](imaging_gp.md), [a ring around a binary](imaging_composite.md) [sampling the posterior](imaging_sampling.md) and [sparse images and CLEAN](imaging_clean.md).
- **[API Reference](api/index.md)** documents every public class and function.

Documentation tooling is currently migrating from MkDocs to Zensical. During this transition, both builders are supported from the same configuration file.

Local docs checks:

```bash
uv run --python .venv/bin/python mkdocs build --strict
uv run --python .venv/bin/python zensical build --clean
```

## Independent validation

virgil's own tests mostly check virgil against itself. The companion repository [virgil-validation](https://github.com/benjaminpope/virgil-validation) checks it against code that shares nothing with it: geometric primitives with textbook visibilities, uv tracks and OIFITS files built from first principles, and aperture-masking images simulated with [dLux](https://github.com/LouisDesdoigts/dLux), which virgil then reads and fits. If you would like something else validated, please [open an Issue there](https://github.com/benjaminpope/virgil-validation/issues) describing the model or function, the independent result it should match, and the precision you expect.

## Collaboration & Development

We welcome collaboration and development contributions. See [CONTRIBUTING.md](https://github.com/benjaminpope/virgil/blob/main/CONTRIBUTING.md) for development setup, testing, and pull request workflow. Release notes are in the [changelog](https://github.com/benjaminpope/virgil/blob/main/CHANGELOG.md).

## Name

Why is it called virgil?

VIRGIL is the **V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary. In Dante's *Divine Comedy*, the poet Virgil is Dante's guide through the Inferno and Purgatory. In Virgil's own *Aeneid*, when Aeneas enters the underworld he draws his sword on the monsters crowding its threshold. His guide, the Cumaean Sibyl, warns him that they are only thin, bodiless lives flitting in a hollow semblance of form (*Aeneid* VI.292–294). Image reconstruction from sparse interferometric data is full of false visions like these: artefacts that look like structure but have no substance in the data. VIRGIL aims to help you tell the difference. The acronym is Jonah Goldfine's.

### Formerly drpangloss

Until version 0.1.1 this package was called **drpangloss**, after Voltaire's Dr Pangloss and as a nod to Antoine Mérand's [CANDID](https://github.com/amerand/CANDID). From version 0.2.0 it is **virgil**: `import virgil`, installed with `pip install virgil-astro`. A final release of `drpangloss` under its own name will depend on `virgil-astro` and point here, so old installs find the new package.

