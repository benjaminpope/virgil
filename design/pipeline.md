# Pipelines: a stable contract over virgil's analyses

Status: **BinaryPipeline built** (2026-10-08). `StarPipeline` (stable),
`OrbitPipeline` and `ImagingPipeline` (both provisional) follow as stacked
PRs.

## Goal

A tutorial shows how to do an analysis; a pipeline does it the same way
every time and leaves a folder that a person, a script or an agent can
read without running the code again. The pipelines in `virgil.pipeline`
wrap virgil's public functions in fixed stages. Each stage writes its own
outputs, the run resumes after an interruption, and every run ends with
deterministic quality checks and an executed quicklook notebook.

A pipeline does not add science. Every number it reports comes from a
public virgil function that a tutorial also calls, so a pipeline result
can always be reproduced by hand. When a pipeline needs something that
virgil does not have, the function is added to virgil first.

## Contract

A pipeline class

- takes the data, a model template (or, for `StarPipeline`, a short model
  name such as `"uniform"` or `"limb_darkened"`), a run folder and keyword
  settings. Settings resolve as defaults, then a JSON config file, then
  keyword arguments;
- runs named stages in a fixed order; `run(through=stage)` stops early;
- resumes by default: a completed stage whose outputs exist is skipped.
  Once a stage reruns, every later stage reruns. A resumed run refuses
  (with `ConfigMismatchError`) if the settings, the data's hash or the
  model template differ from those of the stored run;
- writes only inside its run folder, each file atomically (a temporary
  file in the same folder, then `os.replace`), so a crash never leaves a
  half-written file under a final name;
- stores everything as JSON, HDF5, NumPy `.npz` (no pickles), PNG, OIFITS
  and one notebook.

### The run folder

| Path | Contents |
| --- | --- |
| `run.json` | schema, class, stability, status (`running`, `partial`, `complete`, `failed`), resolved settings, inputs and their sha256, provenance (versions, platform), per-stage status and timings, the error of a failed run, and finally a sha256 of every file |
| `summary.json` | the key numbers by section, the checks, and the worst check status |
| `stages/<name>/report.json` | each stage's own numbers |
| `data/processed.oifits` | the data as fitted, after selections and error floors (skipped, with the reason in the load report, when the layout cannot be written as OIFITS) |
| `models/<name>/` | `manifest.json` (a field spec of public virgil classes) and `values.npz`, enough to rebuild the model without a pickle |
| `grids.h5`, `samples.h5` | grid maps and posterior samples, with axis names and units as attributes |
| `plots/*.png` | the figures |
| `quicklook.log` | stderr of the quicklook kernel |
| `quicklook.ipynb` | an executed notebook that reloads the folder with `virgil.pipeline.load` and shows the figures and checks |

Warnings raised inside a stage are captured, not printed. The runner
wraps each stage in `warnings.catch_warnings(record=True)` with
`simplefilter("always")` and stores each distinct message once, as
`{stage, category, message}`, in the stage's `report.json` and in the
`warnings` list of `summary.json` (an additive schema change).
`Result.describe()` gives one line counting them. Pure functions in
`_checks.py` map the known messages to checks, by keywords rather than
exact text: optimizer non-convergence to `convergence` (value: worst
fraction of grid positions), an unresolved flux peak to
`flux_axis_resolution`, clipped absil limits to `limits_clipped`; any other
non-deprecation warning goes to one generic `stage_warnings` check. All
are `warn` and appear, after the fixed checks, only when the warning
occurred. The default `n_flux` is unchanged: the peak of a high-SNR dataset
can be under one step wide on a 40-point log axis, and resolving it by
brute force costs several times the search, so the check reports it
instead. The quicklook kernel runs with `JAX_PLATFORMS=cpu` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and its stderr goes to
`quicklook.log` in the run folder.

`virgil.pipeline.load(path)` returns a `Result` that reads all of these;
`virgil-pipeline validate` checks a folder against `run.json`'s hashes.

### Checks

Checks are pure functions of plain numbers and return a `Check` (name,
status `pass`/`warn`/`fail`, value, threshold, a templated message). The
same run always gives the same checks. Each pipeline lists its checks in
a fixed order. The first is always χ²/N of the model **on the quoted
errors**: a fitted error scale makes χ² ≈ 1 by construction, so it is
never the headline number. With `error_scale="fit"`, a further check
flags scales well above 1 as a failed fit, not a calibration.

### Priors

Pipelines use group-invariant priors: uniform in locations and angles
(angles as an `AngleVector`, with no wrap at 0°/360°), log-uniform in
scales such as fluxes, diameters and error scales. Their ranges are
settings.

## Stability policy

A pipeline is either **stable** or **provisional**; the tier is the class
attribute `STABILITY` and is written to `run.json` and `summary.json`.

**Stable** pipelines keep, within a schema version
(`virgil-pipeline-run-v1`):

- the constructor and `run` signatures, stage names and order, setting
  names and their defaults;
- every key of `summary.json` and `run.json`, and its meaning;
- check names, order and thresholds;
- file names in the run folder.

Changes are additive: new settings with defaults that keep the old
behaviour, new summary keys, new checks appended at the end, new files.
Anything else (a renamed key, a changed default, a removed check) needs a
new schema version, and `load` refuses schemas it does not know. Snapshot
tests (`tests/test_pipeline_*.py`) pin the signatures, defaults, summary
keys and check order, so an accidental change fails the tests.

**Provisional** pipelines write the same folder layout and use the same
`Result`, but their stages, settings and summary keys may change while
the virgil APIs underneath settle. They emit one `FutureWarning` per
session. A provisional pipeline becomes stable when its tutorial and its
underlying functions have stopped changing for a release; the promotion
is recorded here.

| Pipeline | CLI | Tier | Stages |
| --- | --- | --- | --- |
| `BinaryPipeline` | `binary` | stable | load, overview, search, limits, fit, posterior, quicklook |
| `StarPipeline` | `star` | stable (planned) | load, overview, fit, posterior, quicklook; `model=` takes a model or a short name |
| `OrbitPipeline` | `orbit` | provisional (planned) | multi-epoch positions, orbit search, fit, posterior |
| `ImagingPipeline` | `imaging` | provisional (planned) | regularised images, weight selection, uncertainty |

## Dependencies

`import virgil` never imports the pipeline. `virgil.pipeline` loads
lazily on first access, and a pipeline's constructor checks for the
`pipeline` extra (h5py, nbformat, nbclient, ipykernel and the plotting
extra) with an `ImportError` naming `pip install 'virgil-astro[pipeline]'`.

The CLI's `info` and `validate` read only JSON and hashes. They still
import `virgil` and therefore JAX, but do no JAX work.

## Worked example

The binary-search tutorial's NIRISS AMI analysis maps onto the stages:
building the `OIData` is `load`, the Δχ² and SNR maps are `search`, the
Absil contrast map is `limits`, the LM fit is `fit`, and the NUTS chains
are `posterior`. `BinaryPipeline(data, BinaryModelCartesian(...),
max_sep_mas=250, ...).run()` reproduces the tutorial's numbers, and the
tutorial asserts that it does.

## Open questions

- Whether `info` and `validate` should avoid importing JAX, which needs
  `virgil/__init__.py` to stop importing it eagerly.
- Multi-file inputs with different instruments: one `OIData` per file, or
  joint fits through `joint_*` likelihoods (a later, additive setting).
- Before the first stable release, settle `global_nsigma`. It is a Šidák
  estimate with a nominal number of resolution elements, not a simulated
  false-alarm probability: grid maxima are correlated and the resolution
  element is anisotropic, so the true trial count may be far off. It is
  labelled as an approximation in `summary.json`
  (`global_nsigma_method`), the `detection` message and the docs, which
  point to `injection_recovery` and `gaussian_null`. Either calibrate it by
  simulation or drop it from the frozen summary keys and check.
