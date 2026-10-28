# Pipelines

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

A tutorial shows how to do an analysis; a pipeline does it the same way every time and leaves a folder that a person, a script or an agent can read without rerunning the code. `virgil.pipeline` wraps virgil's public functions in fixed, named stages. Each stage writes its own outputs, an interrupted run resumes, and every run ends with deterministic quality checks and an executed quicklook notebook.

A pipeline adds no science: every number it reports comes from a public virgil function that a tutorial also calls, so you can reproduce any result by hand. The binary-search, contrast-limits and limb-darkening tutorials end with an "As a pipeline" section that does exactly that and asserts that the numbers agree; the [orbit tutorial](orbit_fitting.md) leads with its pipeline and then shows the functions underneath.

Install the extra (it brings in the plotting extra, `h5py`, `nbformat`, `nbclient` and `ipykernel`):

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

A run resumes by default: a completed stage whose outputs exist is skipped, and once a stage reruns every later stage reruns. A resumed run raises `ConfigMismatchError` if the settings, the data hash or the model template differ from those stored in the folder. `run(resume=False)` starts again.

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

`max_sep_mas` is by default the coarsest λ/B. Where the uv samples of one wavelength lie on a lattice (aperture masks, DISCO products), the visibilities repeat when the companion moves by a vector of the reciprocal lattice, so the default box is limited to half an alias cell, the largest square with no two points that differ by such a vector. The summary records the limit as `search.alias_cap_mas` (`null` when it was not applied), and the quicklook notes it. An explicit `max_sep_mas` is never capped; `OrbitPipeline` limits its position grid in the same way.

### `BinaryPipeline` settings

`BinaryPipeline.defaults()` lists the values:

| Setting (default) | Meaning |
| --- | --- |
| `sigma` (3.0) | Significance of the contrast limits. |
| `detection_sigma` (3.0) | Threshold of the detection check, on the look-elsewhere-corrected significance. |
| `max_sep_mas` (`None`) | Half-width of the search grid; by default `fov_fraction` times the coarsest λ/B of the samples (each sample's own wavelength over its own baseline, not λ_max/B_min), capped as above on lattice coverage. |
| `fov_fraction` (1.0) | See `max_sep_mas`. |
| `grid_step_mas` (`None`) | Grid spacing; by default a quarter of the finest λ/B of the samples, coarsened so that no axis exceeds `max_grid` points (the search report then records the effective step and warns). |
| `max_grid` (101) | Most points per position axis; the grid has the largest odd number of points not above it (at least 3). |
| `flux_range` (`[1e-5, 0.5]`) | Flux axis of the grid and bounds of the log-uniform flux prior (companion/primary). |
| `n_flux` (40) | Points on the flux axis. |
| `wavel_range` (`None`) | `[min, max]` wavelengths to keep, in metres. |
| `error_floor` (`None`) | Absolute error floors by observable, for example `{"vis": 0.01, "phi": 0.005}` (phases in radians), as [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor]. |
| `error_scale` (`"quoted"`) | `"fit"` also fits log-uniform error scales `vis_scale` and `phi_scale` on [0.1, 10]. |
| `num_warmup`, `num_samples`, `num_chains` (1000, 1000, 4) | NUTS. |
| `chain_method` (`"vectorized"`), `seed` (0) | NUTS chain method and random seed. |
| `batch_size` (`None`) | Grid batch size. |
| `limit_bins` (20) | Annuli of the contrast curve. |
| `quicklook_execute` (`True`) | Execute the quicklook notebook. |

The search and the limits always use a point companion in Cartesian offsets; the model template (`BinaryModelAngular` or `BinaryModelCartesian`) sets only the parameters of the fit and the posterior.

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

### `StarPipeline` settings

| Setting (default) | Meaning |
| --- | --- |
| `diam_range_mas` (`None`) | `[min, max]` of the log-uniform diameter prior; by default from 0.1 times the finest λ/B (well inside the unresolved regime) to 4 times the coarsest (beyond the first null on every baseline), taken over the observed samples. |
| `n_scan` (2000) | Points of the log-spaced diameter scan; each lobe's minimum is then refined on 201 points within one scan step either side. |
| `max_lobes` (5) | The most diameter lobes (separate minima of the scan) fitted for each model. The best lobe gives the fit and bounds the posterior's diameter prior; the table is `summary["fit"]["models"][name]["lobes"]`. |
| `wavel_range` (`None`), `error_floor` (`None`) | As for `BinaryPipeline`. |
| `error_scale` (`"quoted"`) | `"fit"` also fits log-uniform error scales `vis_scale` and `phi_scale` on [0.1, 10], profiled along the scan. With fitted scales χ² is N by construction, so lobes and models are compared on −2 log L (on the quoted errors, χ² up to a constant). |
| `num_warmup`, `num_samples`, `num_chains` (1000, 1000, 4), `chain_method` (`"vectorized"`), `seed` (0) | NUTS. |
| `batch_size` (`None`) | Scan batch size. |
| `quicklook_execute` (`True`) | Execute the quicklook notebook. |

## `ImagingPipeline`: a model-independent image (provisional, first version)

```python
from virgil.pipeline import ImagingPipeline

res = ImagingPipeline(data, output="runs/image").run()
img = res.image()          # a SkyImage: pixels, pixel_scale_mas, dra, ddec, beam
```

`ImagingPipeline` reconstructs a pixel image with a Gaussian-process prior whose amplitude σ and length ℓ are chosen by evidence (`star="auto"` also decides, by ΔlogZ against a threshold of 5 nats, whether a point source sits at the centre). The stages are `load`, `overview`, `star`, `hyperparameters`, `members` (default 8 fits at the shared hyperparameters, resumable one by one with `run_member(k)`), `combine` and `quicklook`. The result is `images/mean.fits` (a sky-oriented FITS with a WCS and the star's share of the flux) and `images/sigma.fits` (the spread of the members, which excludes hyperparameter uncertainty), `models/best` (`res.model()`) and a quicklook that plots the image.

!!! warning "A good fit is not a promise about the picture"
    The checks (a raw χ²/N split into V² and closure phases, the quoted errors, the error scale, the effective number of parameters, the short baselines, the star decision, the sky orientation and the grid and field edges) test the data fidelity. Fits that passed them alone were judged badly wrong, so the quicklook's first cell says that the morphology is not promised. The χ²/N is always on the quoted errors; the rescaled value is reported beside it and never checked.

This first version fits an ensemble of up to 8 members (`members`, default 8), which by default share the hyperparameters and differ only in start, pixel scale and field; `ensemble.combine` keeps those that fit V² and closure phases, and `sigma.fits` is their spread, which with shared hyperparameters leaves out hyperparameter uncertainty (`member_grids=True` gives each member its own). CLEAN starts, classical regularizers chosen by L-curve, the `anneal` option and per-channel imaging are to come.

```bash
virgil-pipeline image star.oifits -o runs/image --set star=auto
```

### `ImagingPipeline` settings

| Setting (default) | Meaning |
| --- | --- |
| `oversample` (3.0), `field_fwhms` (6.0), `field_mas` (`None`), `max_npix` (128) | The grid: pixels per resolution element, the field in beam FWHMs (or `field_mas` in mas), and the most pixels per side. |
| `star` (`"auto"`), `star_threshold` (5.0) | `True`, `False` or `"auto"`, which compares fits with and without a central point source by log evidence and decides at \|ΔlogZ\| ≥ `star_threshold`. |
| `sigmas` ([1, 2, 4]), `lengths_beam` ([0.5, 1, 2]), `grid_growths` (2) | The evidence grid for the Gaussian-process amplitude σ and length ℓ (in beams). |
| `members` (8), `member_grids` (`False`) | Members fitted at the shared hyperparameters, varying only the start, the pixel scale and the field (1 or 1.5 times). `member_grids=True` gives each its own evidence grid: slow, for cluster runs. |
| `method` (`None`), `max_steps` (`None`) | The optimizer of [`fit`][virgil.fitting.fit]. |
| `seed` (0), `wavel_range` (`None`), `error_floor` (`None`), `quicklook_execute` (`True`) | As for `BinaryPipeline`. |

## `DiskPipeline`: rings and haloes (provisional)

```python
from virgil.pipeline import DiskPipeline

res = DiskPipeline(data, output="runs/disk").run()                 # a Gaussian-process disk
res = DiskPipeline(data, "rings", output="runs/rings", companion=True).run()
res.summary["disk"], res.model(), res.model("geometry")
```

`DiskPipeline` fits a [`MultipoleDisk`][virgil.models.MultipoleDisk] around one star. It is provisional: the settings, stages and summary keys may change in any 0.4.x release without a deprecation. `model=` picks one of two parallel options, never a chain:

- `"gp"` (the default, the option that assumes least about the disk): one [`RadialGP`][virgil.models.RadialGP] profile with low-degree harmonics (`harmonic_degree`) and, with more than `spectral_index_min_channels` channels, a spectral index that is itself a Gaussian process. At each length scale of `ell_halffringes` the profile is fitted with the geometry held at the single-ring fit's values (fast), then again with the inclination and position angle released, and the length of largest Laplace evidence is kept. A grid must not choose its own edge: if the best length is the smallest or the largest, the grid is extended once in that direction (never below `ell_floor_halffringes`, half a half-fringe), and the `length_scale` check warns if the best is still on an edge (`on_grid_edge` in the summary). With `error_scale="rescale"` (the default here) one MacKay error scale is estimated from a baseline fit at the middle length, with the companion if there is one and the geometry released, and every fit and evidence after it is on the rescaled data, so they share one footing; the scale is reported (`chi2.error_scale_factor`), and χ²/N is still quoted on the quoted errors.
- `"rings"`: *N* [`GaussianRing`][virgil.models.GaussianRing] profiles that share the inclination and position angle, added one at a time (each starting at the radius of a one-ring scan) until the BIC stops improving or `max_rings` is reached; after each fit every ring is re-placed in turn (dropped, rescanned on a finer radius grid, everything refitted) and the result kept if the BIC falls (not if it leaves two rings closer than twice the narrower one's width or a half-fringe). A count with such a pair is one ring fitted as two, or a ring still to be found: it is left out of the comparison and the search goes on; a ring whose removal lowers the BIC (one parked at the inner radius with no light, say) is dropped after the fit, which ends the search, the pruned fit standing as the candidate of its own count; `ring_count` warns when such a count had the lower BIC. The Laplace evidence and the BIC of every count are reported and the count of least BIC is kept.

The resolution scale is the half-fringe λ/2B of the longest baseline. The inner radius, the ring-width floor and the length-scale grid are stated in half-fringes, and the outer radius defaults to half the [`image_field`][virgil.imaging.image_field]: set `r_out_mas` if the disk is larger. A companion search is opt-in (`companion=True`): the residuals are searched for a point source on a grid, and a detection above `companion_min_delta_chi2` is fitted jointly with the disk. A search of the best model's residuals always runs, to check that the disk has not absorbed a compact source.

| Stage | What it does | Reuses |
| --- | --- | --- |
| `load`, `overview` | As for `BinaryPipeline`, and the resolution scale and radial range | |
| `geometry` | χ² of a single ring over a grid of inclination (uniform in cos *i*) and position angle on [0°, 180°), then a fit of that ring from the best cell: the geometry is multimodal and is never left to a local optimizer from one start. Its dynamic range sets the Gaussian process's σ | `likelihood_grid`, `fit`, `laplace_cov` |
| `profile` | The ring counts, or the length scales, each fitted by Levenberg–Marquardt; the companion search. For `"gp"` it also estimates the MacKay error scale, scores the Laplace evidence of each length (so a run stopped after `profile` has them) and extends the grid once if the best is on an edge | `fit`, `likelihood_grid`, `gp_priors`, `MultipoleDisk.freeze_geometry`, `log_evidence`, `error_scale` |
| `compare` | The Laplace evidence (and BIC) of each ring count (`"gp"`: it chooses among the evidences `profile` scored); the preferred one is `models/best` | `log_evidence`, `laplace_cov` |
| `posterior` (`posterior=True`) | NUTS with the geometry released; the position angle is folded into [0°, 180°) with the harmonics rotated to match | `numpyro_model`, `chain_init_params` |
| `quicklook` | The geometry grid and a six-panel summary of the disk | |

Priors are group-invariant: uniform in cos *i*, an `AngleVector` for the position angle (no wrap), log-uniform in ring radii, widths and fluxes, and an isotropic Gaussian (scale 0.4) on the real and imaginary parts of each ring's harmonics, so that the phase of the lopsidedness is uniform. The rings are fitted on the quoted errors (`error_scale="quoted"`, the default there), and `error_scale="fit"` is available for them only; `"rescale"` is for the Gaussian process (see above). The star is a grey point source unless `star_diameter_mas` (a uniform disk; the inner radius is then at least its radius), `star_temperature_k` or `star_index` says otherwise, and the disk's spectral index is relative to that star: it is conditional on the star's assumed spectrum, so read it with the `model.star` section of the summary beside it (on real discs a different star index moved the disk's index by 0.3 to 1.4). The rings' colour is a power-law index per ring by default; `ring_colour="blackbody"` (rings only) fits a temperature per ring and for the planet instead, which follows blackbody-like spectra across a broad band that a power law misses. A model saved under `models/` is a real virgil object: `res.model()` is the preferred one, `res.model("geometry")` the single-ring fit and `res.model("rings_2")` or `res.model("gp_ell1")` a candidate.

### `DiskPipeline` settings

| Setting (default) | Meaning |
| --- | --- |
| `r_in_mas` (`None`), `r_out_mas` (`None`) | The radial range; by default one half-fringe and half of [`image_field`][virgil.imaging.image_field]. |
| `n_r` (`None`) | Quadrature nodes of a profile; by default [`suggest_n_r`][virgil.models.MultipoleDisk.suggest_n_r] (for the Gaussian process, at least enough to resolve the shortest ℓ). |
| `n_inc` (10), `n_pa` (12), `inc_max_deg` (85) | The geometry grid. |
| `max_rings` (4) | The most rings fitted (`"rings"`). |
| `ell_halffringes` ([1, 2, 4]), `ell_floor_halffringes` (0.5) | The grid of ℓ/√3 in half-fringes (`"gp"`), and the smallest ℓ/√3 that the grid or its extension may reach. |
| `harmonic_degree` (1) | Degree of the harmonics (0 for none). |
| `spectral_index` (`None`), `spectral_index_min_channels` (3) | `True` or `False` forces the colour on or off; by default it is on with more than `spectral_index_min_channels` channels. |
| `ring_colour` (`"power_law"`) | Colour of each ring and of the planet when the colour is on (`"rings"` only): `"power_law"` fits a spectral index, `"blackbody"` a temperature (300 to 3000 K, log-uniform) with the flux at the reference wavelength. |
| `wavel0_m` (`None`), `star_temperature_k` (`None`), `star_diameter_mas` (`None`), `star_index` (`None`) | The reference wavelength (default the mean) and the star: a grey point source unless a uniform disk (`star_diameter_mas`), a blackbody (`star_temperature_k`) or a power law (`star_index`) is given; only one of the last two may be set. |
| `companion` (`False`), `companion_min_delta_chi2` (25), `companion_max_sep_mas` (`None`), `companion_step_mas` (`None`) | Search the residuals for a point source and fit a detection jointly; the grid defaults to `r_out_mas` and half a half-fringe. |
| `length_scale_nats` (3), `ring_count_nats` (3), `inner_edge_fraction` (0.1), `inner_edge_correlation` (0.9), `outer_edge_fraction` (0.01), `inclination_min_deg` (15), `render_tolerance` (0.1), `unconstrained_correlation` (0.8) | Thresholds of the [checks](#checks) of the same names. |
| `error_scale` (`"auto"`) | `"auto"` is `"rescale"` for `"gp"` and `"quoted"` for `"rings"`; `"fit"` is for rings only. |
| `wavel_range` (`None`), `error_floor` (`None`), `batch_size` (`None`) | As for `BinaryPipeline`. |
| `posterior` (`False`), `num_warmup`, `num_samples`, `num_chains` (1000, 1000, 4), `chain_method` (`"vectorized"`), `seed` (0) | NUTS. |
| `quicklook_execute` (`True`) | Execute the quicklook notebook. |

## `OrbitPipeline`: a binary seen on several nights (provisional)

`OrbitPipeline` fits the orbit of a binary observed on several nights, and it does not need to be told the period. The default run is a blind search over the period aliases: the window of periods comes from the data, and no published period, warm start or paper prior is used. Give it the OIFITS files (each is split into nights at gaps of `gap_days`), or an `Epochs`, or `{night: data}`:

```python
from virgil.pipeline import OrbitPipeline

res = OrbitPipeline.from_oifits(files, output="runs/hd1234").run()

res.summary["modes"]["winner"]      # the winning alias band and its orbit
res.summary["modes"]["delta_log_z"] # its margin over the runner-up, in nats
res.aliases().table()               # every band: log Z, probability, flags
```

The [orbit tutorial](orbit_fitting.md) runs this on synthetic two-season data, reads the result and then calls the functions underneath (`epoch_positions`, `fit_orbit_aliases`) for the control the pipeline's settings do not give.

`OrbitPipeline` takes data that the instrument package has already prepared (the channel, band edges, outlier exposures and noise model): it neither selects a channel, trims the band nor screens exposures, and it has no instrument-specific settings, so it works the same for every instrument. Per-file rulings go in `metadata` (`epoch`, and `mjd` for a file whose time stamp is wrong or absent) and `exclude` (`{file, night or dataset: reason}`, recorded in `run.json`). The command line is `virgil-pipeline orbit FILES... --output DIR [--metadata FILE.json]`.

| Stage | What it does | Reuses |
| --- | --- | --- |
| `load` | Groups the data into nights as given and records the nights | `OIData.split_by_epoch`, `Epochs` |
| `epochs` | The companion's position on every night, on a grid widened until no best peak is on its edge; flags the nights that cannot seed an orbit (aliased peaks, a low gap, a high χ², a separation below the resolution, a flux at its bound) | `epoch_positions` |
| `search` | The alias search with the engine's defaults, band by band, checkpointed in `stages/search/bands`: the period window from the data, the primary's diameter estimated, the V² amplitude fitted per night, the labels free, the likelihood tempered | `fit_orbit_aliases` |
| `modes` | Ranks the bands; the winner's margin over the runner-up; the band probabilities; the best orbit against a single star (`no-orbit`) | `AliasResult` |
| `posterior` | NUTS in the winning band and in every band with a probability of at least `posterior_p_min` (at most `max_modes` besides the winner), on the engine's own posterior, so the tempering and the V² amplitude are those of the search | `AliasResult`, `chain_init_params` |
| `quicklook` | Writes and executes `quicklook.ipynb` | `load` |

A run that interrupts resumes: the bands already fitted are read back from their checkpoints, so only the missing bands are refitted. `run(resume=False)` starts over, bands included. The winner is the band of highest evidence; the summary's `delta_log_z` is its log Z over the runner-up's, and `alias_ambiguity` warns below 10 nats and fails below 1. `no-orbit` compares the best orbit's evidence (with its prior volume) with the best log likelihood of the star alone (a maximum, not an evidence, so the test leans towards no orbit), and a run that finds no orbit skips the posterior. The posterior is sampled with NUTS (dense mass matrix) on the engine's own tempered, scale-marginalized density, not with `gauss_newton_mass`: that builds its curvature from the Jacobian of a virgil model's whitened residuals, which this density (tempered, with the V² amplitude profiled) is not. On the two-season synthetic case a diagonal mass matrix gave a similar minimum effective sample size (225 against 156, 300 warm-up and 300 draws on 2 chains) at 60% of the cost, so the dense mass is a conservative choice for the narrow ridges that real, correlated orbits can have.

### `OrbitPipeline` settings

| Setting (default) | Meaning |
| --- | --- |
| `gap_days` (0.5) | Frames further apart than this are in different nights (`from_oifits`). |
| `exclude` (`{}`) | `{file, night or dataset: reason}` left out; a ruling recorded in `run.json`. |
| `flag_nights` (`True`), `flag_sep_blocks` (`False`) | Flag the nights that cannot seed an orbit; whether a separation below λ/2B also keeps a night from seeding (reported either way). |
| `outlier` (`"mixture"`) | The night-outlier component of the orbit likelihood, of which the fit marginalizes one rate $\epsilon$ shared by all nights (Beta(½, ½), exactly: `group_outlier_mixture`). `"mixture"`: the outlier component is a companion anywhere in the field, the night's mean likelihood over the prior's disc; `"inflation"`: it is the night's own model with its errors inflated by a scale in $[1,$ `s_max`$]$ (needs `s_max`). The default is the nearest continuation of the previous behaviour, not a verdict between the two: the choice rests on the no-orbit test on constructed data. The no-orbit null carries the same component. The posterior outlier probability of each night is in the search report, with the `orbit-prefers-secondary` flags (reported only: no night is ever excluded). |
| `batch_size` (`None`), `memory_budget` (`None`, bytes), `coarse_nats` (30) | Grid points and candidate orbits scored together (by default what fits in the budget, 1 GiB on a CPU); for data with gains or offsets, the nats below the best within which the position grid integrates the error scale in full (`None`: every point). See `epoch_positions`. None of the three is a prior range. |
| `n_peaks` (5), `max_sep_mas` (`None`), `grid_step_mas` (`None`), `max_grid` (201) | The position search; the step is by default a quarter of the finest λ/B (coarsened to `max_grid` points) and the half-width `a_range[1] (1 + ecc_max)`, the prior's reach, limited on lattice coverage (aperture masks, DISCO) to the alias cell. |
| `period_range` (`None`) | `[min, max]` days; by default from the data (see `fit_orbit_aliases`). |
| `a_range` ([1, 30]), `flux_range` ([0.01, 1]), `ecc_max` (0.9), `s_max` (`None`) | The engine's prior ranges (mas; flux ratio at most 1; eccentricity; error-scale bound). |
| `primary_diam` (`"auto"`), `vis_amp`, `sign_free`, `flip_flux`, `temper` (`True`) | As for the engine. |
| `n_bands_max`, `n_candidates`, `n_refine`, `max_steps`, `n_is` | Search effort; the defaults are those of `fit_orbit_aliases`, the one place to raise them (the `mode_converged` check says when to). |
| `alias_nats` (10), `no_orbit_nats` (10) | Thresholds of the `alias_ambiguity` and `no_orbit` checks. |
| `posterior_p_min` (0.05), `max_modes` (3) | Bands sampled by NUTS besides the winner. |
| `num_warmup`, `num_samples`, `num_chains` (1000, 1000, 4), `chain_method` (`"vectorized"`), `seed` (0) | NUTS. |
| `posterior_max_tree_depth` (8), `posterior_max_hours` (12) | NUTS uses each mode's Laplace covariance as the dense mass matrix, fixed in warm-up (adapted only if the Hessian was clipped or the mode is on a prior bound; raising `num_warmup` then changes only the step size). A band that exceeds the time left reports `converged: false`, `budget_exhausted: true`, its step size, tree depth and post-warm-up gradient count (`n_gradients_sampling`). |
| `quicklook_execute` (`True`) | Execute the quicklook notebook. |

## The run folder

| Path | Contents |
| --- | --- |
| `run.json` | Schema, class, stability tier, status (`running`, `partial`, `complete` or `failed`), resolved settings, inputs and their sha256, provenance, per-stage status and timings, the error of a failed run, and the sha256 of every file |
| `summary.json` | The key numbers by section, the checks, the worst check status, and the `warnings` raised inside the stages |
| `stages/<name>/report.json` | Each stage's own numbers, and its `warnings` |
| `data/processed.oifits` | The data as fitted. It is skipped, with the reason recorded in the load report, when the layout cannot be written as OIFITS |
| `models/<name>/` | `manifest.json` and `values.npz`: enough to rebuild the model without a pickle |
| `grids.h5`, `samples.h5` | Grid maps and posterior samples, with axis names and units as attributes |
| `plots/*.png` | The figures |
| `quicklook.ipynb` | An executed notebook that reloads the folder and shows the figures and checks |
| `quicklook.log` | The stderr of the quicklook's kernel, kept out of the terminal |

Everything is JSON, HDF5, NumPy `.npz`, PNG, OIFITS or one notebook, with no pickles. Files are written atomically, so a crash never leaves a half-written file under its final name, and nothing is written outside the run folder.

## Reading a run: `load` and `Result`

`load(path)` returns a `Result`. Nothing is recomputed: each accessor reads the files the run wrote.

| Accessor | Returns |
| --- | --- |
| `run`, `summary` | The contents of `run.json` and `summary.json` |
| `status` | `"running"`, `"partial"`, `"complete"` or `"failed"` |
| `checks` | The quality checks, as `Check` objects |
| `describe()` | The key numbers, the checks and a one-line count of recorded warnings, as short text |
| `plots` | Paths of the plots, sorted by name |
| `data()` | The fitted data, as an `OIData` |
| `model()`, `model_values()` | The best-fit model (`name=` picks one of several) with its values set, or the values as arrays |
| `fit_result()` | The MAP fit as a `FitResult`, to continue with `fit` |
| `samples()`, `sample_stats()` | Posterior samples and NUTS diagnostics, shaped `(chain, draw)` |
| `grid()` | (`StarPipeline`: the uniform-disk delta chi-squared scan over `diam`; `DiskPipeline`: the single-ring `delta_chi2` over `cos_inc` and `pa`.) The grid axes (usable as `grid=`) and each map, such as `delta_chi2`, `snr` and `limit_flux` |

## Checks

Checks are pure functions of plain numbers, so the same run always gives the same checks in a fixed order, each with a status of `pass`, `warn` or `fail`. The first is always the chi-squared per independent datum of the model on the quoted errors: a fitted error scale makes chi-squared close to one by construction, so it is never the headline number.

| Check | Warns | Fails |
| --- | --- | --- |
| `chi2` (on the quoted errors) | above 2 (errors likely underestimated) or below 0.5 (errors likely overestimated) | above 5 (the model does not describe the data, or the errors are much too small) |
| `error_scale` (with `error_scale="fit"`, or the MacKay estimate of `"rescale"`) | a scale above 2 | a scale above 5, a failed fit rather than a calibration |
| `detection` (approximate look-elsewhere-corrected significance against `detection_sigma`; use `injection_recovery` or `gaussian_null` for a calibrated threshold) | below the threshold | |
| `grid_edge` (peak on the edge of the `dra` or `ddec` axis; the flux axis is not a search boundary) | at the edge | |
| `start_at_bound` (only when triggered: the best fit, where NUTS starts, within 0.1% of a prior bound; typically a non-detection, with the flux or diameter at the edge of its range) | any: NUTS starts just inside the bound and the posterior is prior-dominated | |
| `prior_bound` (posterior mass within 1% of a prior bound) | above 5% | |
| `r_hat` | above 1.01 | above 1.05 |
| `ess` (smallest bulk effective sample size) | below 400 | below 100 |
| `divergences` | any | above 1% of transitions (a warning instead, naming the walls, when `StarPipeline`'s wall probe finds hard closure-phase walls around the posterior) |
| `residual_normality` (skewness and excess kurtosis of the independent whitened residuals, without the periodic closure-phase penalty terms) | beyond three standard errors | |
| `field_of_view` (companion inside the resolution limit or beyond the field) | outside | |
| `chi2_uniform`, `chi2_limb_darkened` (`StarPipeline`; as `chi2`) | as `chi2` | as `chi2` |
| `limb_darkening_gain` (delta chi-squared of the limb-darkened over the uniform fit, against a BIC penalty of 2 ln N for two parameters) | the limb-darkened model fits worse than the uniform disk it contains | |
| `resolution` (the longest B/λ against the first null of the fitted model's visibility, θB/λ = 1.22 for a uniform disk) | below the first null (limb darkening is degenerate with the diameter) | below 0.15 of it (unresolved: an upper limit) |
| `multimodal` (the per-lobe table of the preferred model, `fit.models.<name>.lobes`: the next-best diameter lobe against a Δχ² of 25) | the next-best lobe is within Δχ² of 25; the message gives its diameter | |
| `limb_darkening_constrained` (posterior against prior standard deviation of `q1`, `q2`) | a ratio of 0.8 or more | |
| `length_scale` (`DiskPipeline`, `"gp"`: the spread of the Laplace evidences across the length-scale grid) | under `length_scale_nats` (3): the data do not measure ℓ, which is then a prior choice | |
| `ring_count` (`DiskPipeline`, `"rings"`: the margin of the chosen count over its neighbours, the smaller of ΔBIC/2 and the log-evidence difference, and every ring's width against the half-fringe) | a margin under `ring_count_nats` (3), a ring narrower than the half-fringe, or a count with a lower BIC left out because two of its rings are too close to tell apart | |
| `structure_unconstrained` (`DiskPipeline`, `"gp"`: the evidence spread across ℓ with the smallest correlation between the radial profiles of the fits at each ℓ) | a spread under `length_scale_nats` with a correlation under `unconstrained_correlation` (0.8): the data do not constrain the structure and the image is mostly the prior | |
| `inner_edge` (`DiskPipeline`, `"gp"`: the disk's light within a half-fringe of the inner radius, and its Laplace correlation with the disk flux) | above `inner_edge_fraction` (0.1) or a correlation above `inner_edge_correlation` (0.9): an unresolved core trades against the star's flux | |
| `outer_edge` (`DiskPipeline`: the light within a half-fringe of `r_out_mas`) | above `outer_edge_fraction` (0.01): refit with a larger `r_out_mas` | |
| `inclination` (`DiskPipeline`) | an inclination under `inclination_min_deg` (15°, the position angle is then unconstrained), a position-angle standard deviation above 45°, or several comparable minima of the geometry grid | |
| `absorbed_companion` (`DiskPipeline`: Δχ² of a point source added to the best model's residuals) | at least `companion_min_delta_chi2` (25); the message says whether a harmonic of one degree higher removes it | |
| `render_consistency` (`DiskPipeline`, `"gp"`: χ²/N of the model and of its disk rendered to pixels, on the channel nearest the reference wavelength) | the rendered image's χ²/N differs by more than `render_tolerance` (0.1) plus 0.02: the image is not the model the fit found; refit with a larger `n_r` (`suggest_n_r` sizes the nodes for the visibilities, not for pixels) | |
| `at_bound` (`DiskPipeline`) | any fitted parameter within 0.1% of a prior bound: the data do not constrain it there, or the prior is too tight | |
| `night_flags` (`OrbitPipeline`: nights whose position cannot seed an orbit; their data still count in the likelihood) | any night flagged | |
| `labels_free` (`OrbitPipeline`: a point-reflected orbit with an evidence within 3 nats of the best) | the data do not say which star is which | |
| `bands_failed` (`OrbitPipeline`: alias bands whose fit raised) | any band failed: the winner's margin does not cover it | |
| `alias_ambiguity` (`OrbitPipeline`: log Z of the best alias band over the runner-up's, `alias_nats`) | under 10 nats: the runner-up is not excluded | under 1 nat: the period is not determined |
| `winner_flags` (`OrbitPipeline`: the winning band's evidence flags, such as `not-converged`, `laplace-is-differ`, `low-ess`) | any flag: its log Z is less reliable | |
| `mode_converged` (`OrbitPipeline`: the best sampled point against the search's best mode) | over 1 nat: the search stopped short of the optimum; raise `n_candidates`, `n_refine` or `max_steps` | |
| `tempering` (`OrbitPipeline`: the correlated-channel factor of the likelihood) | a factor above 2, a raw factor below 0.5, or a residual correlation that is not AR(1) | |
| `primary_resolved` (`OrbitPipeline`) | the primary is resolved and its diameter is the largest tried | |
| `no_orbit` (`OrbitPipeline`: the orbit's evidence over a single star's, `no_orbit_nats`; both carry the night-outlier component, so under `"mixture"` the single star may have an unrelated companion on any night and the check asks whether the positions follow an orbit) | under 10 nats | under 0: no orbit is detected |
| `period_bound` (`OrbitPipeline`) | the best period is at the top of the window (a short arc) or at its bottom (the sampling limit) | |

### Warnings raised inside stages

A stage runs under `warnings.catch_warnings(record=True)` with the filter set to `"always"`, so a warning raised inside a stage never reaches the terminal or the notebook. Each distinct message is recorded once per stage, with its stage, category and message, in `stages/<name>/report.json` and in the `warnings` list of `summary.json`; `describe()` counts them. The known messages are also turned into checks, matched by keywords so that rewording a message does not break the mapping. These checks appear only when the warning occurred, after the checks above, and always warn:

| Check | Raised by | Meaning |
| --- | --- | --- |
| `convergence` | the optimizer did not converge at some grid positions | value: the worst fraction of positions; values there may be inaccurate |
| `flux_axis_resolution` | the flux axis does not resolve the likelihood peak (fewer than 2 steps across its FWHM) | `log_bayes_factor` is inaccurate; use a larger `n_flux` |
| `limits_clipped` | contrast limits fell outside the flux bounds | value: the number of clipped limits; widen `flux_range` |
| `stage_warnings` | any other warning, except deprecation notices | value: the number of warnings; the message quotes the first three |

Deprecation notices are recorded in `warnings` but raise no check. The quicklook notebook's own kernel runs on the CPU (`JAX_PLATFORMS=cpu`, no memory preallocation), so it does not compete with the run for a GPU, and its stderr goes to `quicklook.log`.

## Stability

The pipelines are provisional in 0.4: their constructor keywords, settings and defaults, stage names, `summary.json` keys, check names, thresholds and order, and file names may change in any 0.4.x release, without a deprecation. They are promoted to stable after a release of use. The tier is the class attribute `STABILITY` (`"provisional"`), written to `run.json` and `summary.json`; nothing warns at runtime. See [Stability](stability.md) for the tiers.

What does not change under the provisional tier:

- the folder layout is versioned by its schema, `virgil-pipeline-run-v1`, and `load` refuses schemas it does not know;
- every number a pipeline reports comes from a public virgil function, so a result can always be reproduced by hand.

When a pipeline is promoted to stable, it keeps the following within a schema version:

- the constructor and `run` signatures, the stage names and their order, and the setting names and defaults;
- every key of `summary.json` and `run.json`, and its meaning;
- the check names, their order and their thresholds;
- the file names in the run folder.

After that, changes are additive: new settings whose defaults keep the old behaviour, new summary keys, new checks appended at the end, and new files. Anything else needs a new schema version. Snapshot tests already pin the signatures, defaults, summary keys and check order, so a change is visible in review; while a pipeline is provisional, updating the snapshot is allowed.

Open before promotion: `global_nsigma` is a Šidák estimate with a nominal number of resolution elements, not a simulated false-alarm probability (the grid maxima are correlated, and the resolution element is anisotropic). It must be calibrated by simulation, or dropped from the summary keys and the `detection` check, before the pipelines become stable. Until then it is labelled an approximation in `summary.json` (`global_nsigma_method`).

| Pipeline | CLI command | Tier |
| --- | --- | --- |
| `BinaryPipeline` | `binary` | provisional |
| `StarPipeline` | `star` | provisional |
| `DiskPipeline` | `disk` | provisional |
| `OrbitPipeline` | `orbit` | provisional |
| `ImagingPipeline` | `image` | provisional (an ensemble of up to 8 members; channels follow) |

Further pipelines will be added to this table as they land, all provisional.

## Command line

`virgil-pipeline` runs and inspects pipelines from a shell.

```bash
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --set sigma=3 --set num_chains=2
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --through limits --fresh
virgil-pipeline star star.oifits -o runs/star --model limb_darkened
virgil-pipeline disk disk.oifits -o runs/disk --model rings --set companion=true --set r_out_mas=25
virgil-pipeline orbit night*.oifits -o runs/orbit --metadata rulings.json
virgil-pipeline info runs/hd1234        # status and checks
virgil-pipeline validate runs/hd1234    # complete, and every file matches its hash
virgil-pipeline quicklook runs/hd1234   # rebuild the quicklook notebook
```

`--set KEY=VALUE` overrides a setting (the value is parsed as JSON where possible). `info` and `validate` read only JSON and hashes and do no JAX work, although `import virgil` still imports JAX.

## API reference

The classes and functions are documented on the [pipeline API page](api/pipeline.md).
