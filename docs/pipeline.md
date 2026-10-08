# Pipelines

A tutorial shows how to do an analysis; a pipeline does it the same way every time and leaves a folder that a person, a script or an agent can read without running the code again. `virgil.pipeline` wraps virgil's public functions in fixed, named stages. Each stage writes its own outputs, an interrupted run resumes, and every run ends with deterministic quality checks and an executed quicklook notebook.

A pipeline adds no science. Every number it reports comes from a public virgil function that a tutorial also calls, so any result can be reproduced by hand. The binary-search, contrast-limits and limb-darkening tutorials end with an "As a pipeline" section that does exactly that and asserts that the numbers agree.

Install the extra (it pulls in the plotting extra, `h5py`, `nbformat`, `nbclient` and `ipykernel`):

```bash
pip install 'virgil-astro[pipeline]'
```

`import virgil` never imports the pipelines; `virgil.pipeline` loads on first use.

## Running a pipeline

```python
import virgil as vg
from virgil.pipeline import BinaryPipeline, load

data = vg.OIData("hd1234.oifits")
res = BinaryPipeline(data, output="runs/hd1234", sigma=3.0).run()

res.summary["companion"]["sep_mas"]
res.checks            # a list of Check(name, status, value, threshold, message)
print(res.describe())

res = load("runs/hd1234")   # later, in a fresh session; nothing is recomputed
model = res.model()         # the best-fit model, a real virgil object
```

A pipeline takes the data, a model template, a run folder and keyword settings. Settings resolve as the class defaults (`BinaryPipeline.defaults()`), then a JSON config file (`BinaryPipeline.from_config(path)`), then keyword arguments. `run(through="limits")` stops after a stage, which is how a non-detection skips the fit and the sampler.

A run resumes by default: a completed stage whose outputs exist is skipped, and once a stage reruns every later stage reruns. A resumed run refuses, with `ConfigMismatchError`, when the settings, the data hash or the model template differ from those stored in the folder. `run(resume=False)` starts again.

## The stages of `BinaryPipeline`

| Stage | What it does | Reuses |
| --- | --- | --- |
| `load` | Applies `wavel_range` and `error_floor`, and writes the data that are fitted | `OIData.select`, `OIData.with_error_floor`, `write_oifits` |
| `overview` | Plots the data and the uv coverage | `plot_oidata_overview`, `plot_uv_coverage` |
| `search` | Delta chi-squared, best flux and SNR maps on a point-companion grid, with the significance of the peak before and after an approximate look-elsewhere correction (a Šidák estimate, not a simulated false-alarm probability) | `detection_statistics`, `optimized_likelihood_grid`, `local_nsigma` |
| `limits` | Contrast limits at `sigma` and their radial profile | `absil_limits`, `radial_profile` |
| `fit` | MAP fit of the model template from the best grid point | `fit` |
| `posterior` | NUTS from the fit | `numpyro_model`, `chain_init_params` |
| `quicklook` | Writes and executes `quicklook.ipynb` | `load` |

The fit is on the quoted errors unless `error_scale="fit"`, and chi-squared per independent datum is always reported on the quoted errors. Priors are group-invariant: uniform in position (`dra` and `ddec` on a box of half-width `max_sep_mas`) and log-uniform in flux and in error scales. Both templates sample the same `dra`, `ddec` and `flux`, so they give the same prior on the sky; a `BinaryModelAngular` template reports `sep` and `pa` derived from them, with no wrap at 0 or 360 degrees in the fit. The flux range is a setting (`flux_range`).

The search and the limits always use a point companion in Cartesian offsets; the model template (`BinaryModelAngular` or `BinaryModelCartesian`) sets the parameters of the fit and the posterior.

## `StarPipeline`: stellar diameters

```python
from virgil.pipeline import StarPipeline

res = StarPipeline(data, output="runs/star").run()   # uniform and limb-darkened, compared
res = StarPipeline(data, "uniform", output="runs/ud").run()
res.summary["star"]["diam_mas"], res.model(), res.model("uniform")
```

`model=` is a short name, `"uniform"` ([`UniformDisk`][virgil.models.UniformDisk]) or `"limb_darkened"` ([`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk] with Kipping's `q1`, `q2`), or an instance of one of those classes (its `q1` and `q2` are the starting values). The default fits both and compares them. Each fitted model is saved under `models/<name>/`, with `models/best` the preferred one: the limb-darkened model only if its delta chi-squared exceeds the BIC penalty of its two extra parameters. `res.model()`, `res.model_values()`, `res.fit_result()`, `res.samples()` and `res.sample_stats()` take an optional `name=` to pick one.

| Stage | What it does | Reuses |
| --- | --- | --- |
| `load`, `overview` | As for `BinaryPipeline` | |
| `fit` | A scan of the uniform-disk diameter (a uniform disk's closure phases flip at every null, so its chi-squared is not smooth and a scan is the reliable fit), then `fit` of each other model from the scanned diameter, with chi-squared evaluated in float64 | `likelihood_grid`, `fit` |
| `posterior` | NUTS for every fitted model | `numpyro_model`, `chain_init_params` |
| `quicklook` | The main cell is the squared visibilities against spatial frequency with the model curves | |

Priors are group-invariant: log-uniform in the diameter over `diam_range_mas` (by default from a tenth of the resolution limit to four times the field of view) and uniform on [0, 1] in `q1` and `q2`, which cover exactly the physical quadratic laws. The fit is on the quoted errors, and chi-squared per independent datum is reported on them for each model.

## The run folder

| Path | Contents |
| --- | --- |
| `run.json` | Schema, class, stability tier, status (`running`, `partial`, `complete` or `failed`), resolved settings, inputs and their sha256, provenance, per-stage status and timings, the error of a failed run, and the sha256 of every file |
| `summary.json` | The key numbers by section, the checks, and the worst check status |
| `stages/<name>/report.json` | Each stage's own numbers |
| `data/processed.oifits` | The data as fitted. It is skipped, with the reason recorded in the load report, when the layout cannot be written as OIFITS |
| `models/<name>/` | `manifest.json` and `values.npz`: enough to rebuild the model without a pickle |
| `grids.h5`, `samples.h5` | Grid maps and posterior samples, with axis names and units as attributes |
| `plots/*.png` | The figures |
| `quicklook.ipynb` | An executed notebook that reloads the folder and shows the figures and checks |

Everything is JSON, HDF5, NumPy `.npz`, PNG, OIFITS or one notebook, with no pickles. Each file is written atomically, so a crash never leaves a half-written file under its final name, and nothing is written outside the run folder.

## Reading a run: `load` and `Result`

`load(path)` returns a `Result`. Nothing is recomputed: each accessor reads the files the run wrote.

| Accessor | Returns |
| --- | --- |
| `run`, `summary` | The contents of `run.json` and `summary.json` |
| `status` | `"running"`, `"partial"`, `"complete"` or `"failed"` |
| `checks` | The quality checks, as `Check` objects |
| `describe()` | The key numbers and the checks as short text |
| `plots` | Paths of the plots, sorted by name |
| `data()` | The fitted data, as an `OIData` |
| `model()`, `model_values()` | The best-fit model (`name=` picks one of several) with its values set, or the values as arrays |
| `fit_result()` | The MAP fit as a `FitResult`, to continue with `fit` |
| `samples()`, `sample_stats()` | Posterior samples and NUTS diagnostics, shaped `(chain, draw)` |
| `grid()` | (`StarPipeline`: the uniform-disk delta chi-squared scan over `diam`.) The grid axes (usable as `grid=`) and each map, such as `delta_chi2`, `snr` and `limit_flux` |

## Checks

Checks are pure functions of plain numbers. The same run always gives the same checks, in a fixed order, each with a status of `pass`, `warn` or `fail`. The first is always the chi-squared per independent datum of the model on the quoted errors: a fitted error scale makes chi-squared close to one by construction, so it is never the headline number.

| Check | Warns | Fails |
| --- | --- | --- |
| `chi2` (on the quoted errors) | above 2 (errors likely underestimated) or below 0.5 (errors likely overestimated) | above 5 (the model does not describe the data, or the errors are much too small) |
| `error_scale` (with `error_scale="fit"`) | a scale above 2 | a scale above 5, a failed fit rather than a calibration |
| `detection` (approximate look-elsewhere-corrected significance against `detection_sigma`; use `injection_recovery` or `gaussian_null` for a calibrated threshold) | below the threshold | |
| `grid_edge` (peak on the edge of the `dra` or `ddec` axis; the flux axis is not a search boundary) | at the edge | |
| `prior_bound` (posterior mass within 1% of a prior bound) | above 5% | |
| `r_hat` | above 1.01 | above 1.05 |
| `ess` (smallest bulk effective sample size) | below 400 | below 100 |
| `divergences` | any | above 1% of transitions |
| `residual_normality` (skewness and excess kurtosis of the independent whitened residuals, without the periodic closure-phase penalty terms) | beyond three standard errors | |
| `field_of_view` (companion inside the resolution limit or beyond the field) | outside | |
| `chi2_uniform`, `chi2_limb_darkened` (`StarPipeline`; as `chi2`) | as `chi2` | as `chi2` |
| `limb_darkening_gain` (delta chi-squared of the limb-darkened over the uniform fit, against a BIC penalty of 2 ln N for two parameters) | the limb-darkened model fits worse than the uniform disk it contains | |
| `resolution` (the longest B/λ against the first null of the fitted diameter, θB/λ = 1.22) | below the first null (limb darkening is degenerate with the diameter) | below 0.15 of it (unresolved: an upper limit) |
| `limb_darkening_constrained` (posterior against prior standard deviation of `q1`, `q2`) | a ratio of 0.8 or more | |

## Stability

A pipeline is either **stable** or **provisional**; the tier is the class attribute `STABILITY`, and it is written to `run.json` and `summary.json`.

A stable pipeline keeps the following within a schema version (`virgil-pipeline-run-v1`):

- the constructor and `run` signatures, the stage names and their order, and the setting names and defaults;
- every key of `summary.json` and `run.json`, and its meaning;
- the check names, their order and their thresholds;
- the file names in the run folder.

Changes are additive: new settings whose defaults keep the old behaviour, new summary keys, new checks appended at the end, and new files. Anything else needs a new schema version, and `load` refuses schemas it does not know. Snapshot tests pin the signatures, defaults, summary keys and check order.

A provisional pipeline writes the same folder layout and uses the same `Result`, but its stages, settings and summary keys may change while the virgil functions underneath settle. It emits one `FutureWarning` per session.

| Pipeline | CLI command | Tier |
| --- | --- | --- |
| `BinaryPipeline` | `binary` | stable |
| `StarPipeline` | `star` | stable |

Further pipelines (orbit and imaging, both provisional) will be added to this table as they land.

## Command line

`virgil-pipeline` runs and inspects pipelines from a shell.

```bash
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --set sigma=3 --set num_chains=2
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --through limits --fresh
virgil-pipeline star star.oifits -o runs/star --model limb_darkened
virgil-pipeline info runs/hd1234        # status and checks
virgil-pipeline validate runs/hd1234    # complete, and every file matches its hash
virgil-pipeline quicklook runs/hd1234   # rebuild the quicklook notebook
```

`--set KEY=VALUE` overrides a setting (the value is parsed as JSON where it can be). `info` and `validate` read only JSON and hashes and do no JAX work, although `import virgil` still imports JAX.

## API reference

::: virgil.pipeline.binary.BinaryPipeline

::: virgil.pipeline.star.StarPipeline

::: virgil.pipeline.load

::: virgil.pipeline.Result

::: virgil.pipeline.Check

::: virgil.pipeline.ConfigMismatchError
