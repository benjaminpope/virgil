# Pipelines: a stable contract over virgil's analyses

Status (2026-10-22): **all five pipelines are built** (`BinaryPipeline`,
`StarPipeline`, `DiskPipeline`, `OrbitPipeline`, `ImagingPipeline`) and
provisional (`docs/stability.md`). `OrbitPipeline` is designed in
[`orbit_pipeline.md`](orbit_pipeline.md); `ImagingPipeline` has its first
version (one member), designed in [`imaging_pipeline.md`](imaging_pipeline.md).

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
`Result`, but their stages, settings and summary keys may change in any
0.4.x release while the virgil APIs underneath settle. Nothing warns at
runtime (the tier is in the docs, `virgil._stability` and `run.json`). A
provisional pipeline becomes stable when its tutorial and its underlying
functions have stopped changing for a release; the promotion is recorded
here.

As of 0.4 every pipeline is provisional, `BinaryPipeline` and
`StarPipeline` included: the stable contract above is what they promise
after promotion, and the snapshot tests already pin it.

| Pipeline | CLI | Tier | Stages |
| --- | --- | --- | --- |
| `BinaryPipeline` | `binary` | provisional (stable after one release of use) | load, overview, search, limits, fit, posterior, quicklook |
| `StarPipeline` | `star` | provisional (stable after one release of use) | load, overview, fit, posterior, quicklook; `model=` takes a model or a short name |
| `OrbitPipeline` | `orbit` | provisional (planned) | multi-epoch positions, orbit search, fit, posterior |
| `DiskPipeline` | `disk` | provisional | load, overview, geometry, profile, compare, posterior, quicklook; `model=` takes `"rings"` or `"gp"` |
| `ImagingPipeline` | `image` | provisional | load, overview, start, star, hyperparameters, members, combine, quicklook (one member; the ensemble and channels follow); see [`imaging_pipeline.md`](imaging_pipeline.md) |

## StarPipeline and its model registry

`StarPipeline(data, model=None)` fits stellar diameters. `model` is a short
name or a model instance, and the pipeline looks it up in a registry of
`StarModel` entries (`virgil.pipeline.star.models()`): each entry says how
to build the model from a dict of values, its parameter paths, starting
values, group-invariant priors, whether it is fitted by a diameter scan,
which parameters shape the star (their posterior widths are compared with
their prior widths), derived quantities, and the simpler model it contains
(for the Δχ² comparison). PR 5 adds `"gravity_darkened"`, `"harmonix"` and
the other limb-darkened laws as entries, with no new class and no new
stage; model-specific checks are appended to the end of the check list.

Decisions:

- The default fits `uniform` and `limb_darkened` and compares them by Δχ²
  against a BIC penalty (2 ln N for two extra parameters). The preferred
  model is saved as `models/best`; `Result.model(name)` and the other
  accessors take `name=` (an additive change to `Result`).
- A uniform disk is fitted by a scan of the diameter (coarse, then ±1%),
  not by `fit`: closure phases flip by 180° at each null, so χ² is
  discontinuous. The scan also seeds the other models' diameters, and its
  Δχ² curve is written to `grids.h5` (`Result.grid()["delta_chi2"]`).
- χ² and the scan are evaluated in float64: in float32, a closure phase at
  a visibility near zero can flip, and χ² then jumps by orders of
  magnitude. NUTS runs in the ambient precision, as in the tutorials and
  `BinaryPipeline`.
- Observing a star past its first null makes the closure-phase likelihood
  a union of thin slivers (a 180 degree flip costs (180/sigma)^2 in chi^2),
  and NUTS diverges at the walls whatever the precision or mass matrix.
  The fit and posterior therefore run in float64, the diameter prior is
  bounded to the best lobe. The scan's separate minima (at most
  `max_lobes`) are fitted lobe by lobe for each model, with lobe bounds at
  the midpoints between minima; the table is `fit.models.<name>.lobes`
  (`diam_bounds_mas`, `diam_mas`, `chi2`, `chi2_reduced`, `delta_chi2`),
  and the `multimodal` check cites it. NUTS runs on the best lobe only;
  lobe evidence is left to nested sampling (virgil#578). The limb-darkened
  fit starts from a grid of diameters as well as the scan.
  Residual divergences are still reported, not hidden: within the best
  lobe the posterior is a cell with hard walls in the diameter and in
  q1, q2 (a limb-darkened null moves with q), and NUTS rejects the
  trajectories that cross them. A wall probe (steps of two posterior
  standard deviations that raise chi^2 by more than 1000) confirms them,
  and `divergences` still fails but names the walls as the probable cause.
  The draws match a brute-force grid posterior on the tutorial star, with a
  reduced ESS. A bispectrum likelihood (virgil#578) would remove the walls.
- `processed`, `_noise` and `_report` moved from `BinaryPipeline` into the
  base class, since every pipeline needs them.

## DiskPipeline (provisional)

Status: **built** (2026-10-15), in `src/virgil/pipeline/disk.py`. The
`MultipoleDisk` code it needs (radial profiles, Hankel transform, `Harmonics`,
`RadialGP` priors, `spectral_index`, `freeze_geometry`; see
`design/radial_profiles.md`) is on `main`. The tutorial `multipole_disk_rings`
ends with an "As a pipeline" section that runs both models on its data and
asserts against the truth. The pipeline is provisional from the start and
stays so until the `MultipoleDisk` API has been unchanged for a release.

As built, where the text below left a choice open or the build differed:

- `model=None` is `"gp"`, the option that assumes least about the disk
  (Ben's decision). `self.model` is the name, and `_model_digest` and
  `__repr__` are overridden for it.
- `posterior` is a setting that defaults to `False` (the Gaussian process's
  latents make NUTS heavy); the stage then records `skipped` and writes no
  `samples.h5`.
- `error_scale="fit"` is for `"rings"` only: `log_evidence` needs fixed
  errors. Constructing `"gp"` with it raises `ValueError`. The default,
  `"auto"`, is `"rescale"` for `"gp"` (one MacKay scale from a baseline fit,
  then fixed: see "Lessons from imaging real discs") and `"quoted"` for
  `"rings"`.
- Each GP fit starts with the geometry of the single-ring `geometry` fit
  held (fast, cached Bessel matrices), then is refitted with `inc` and `pa`
  released: with harmonics and a companion the single ring's position angle
  is a few degrees off the truth, and a frozen error is paid for by the
  profile and the companion.
- The harmonic fold is checked against `MultipoleDisk` in
  `tests/test_pipeline_disk.py`: `pa + 180°` with the order-*m* harmonic
  times `(-1)^m` is the same disk.
- The summary adds `disk.companion` (`dra_mas`, `ddec_mas`, `flux`,
  `spectral_index`, `detection_delta_chi2`; `None` without one).

`DiskPipeline(data, model=None, *, output="run", **settings)`, CLI `disk`,
`STABILITY = "provisional"`. The disk is always a `MultipoleDisk` and the star
is a single point source. A companion search is an opt-in expert setting,
never a default stage. The run folder, resume rules, `Result` and
stability policy are those above; only the stages, settings and checks are
new.

### Models

`model` is a short name from a registry, as in `StarPipeline`. Each entry
says how to build the disk and the star from a dict of values, the
parameter paths, group-invariant priors and starting values, its own
`profile` and `compare` behaviour, and which checks apply to it. The two
entries are **parallel, separate options**, not a chain: the analytic
model and the non-parametric model each run start to finish on their own,
with their own settings and checks, and a run fits one of them.

| Name | Disk | Role |
| --- | --- | --- |
| `"rings"` | `MultipoleDisk` with *N* `GaussianRing` profiles sharing `inc` and `pa`; *N* chosen by evidence (Laplace) and BIC | the analytic option: measures structure of an assumed family |
| `"gp"` | `MultipoleDisk` with a `RadialGP` profile, optional `Harmonics` (low degree) and `spectral_index` | the non-parametric option: finds structure with no assumed family |

`model=None` means `"gp"`, the option that assumes least about the disk
(Ben's decision, 2026-10-15; it is not an error). The tutorial's workflow (a GP run to
find structure, then a ring run to confirm and measure it) is two runs of
the pipeline, and a ring count and radii proposed from a finished GP
run's profile could be a setting of the second
(`rings_from="path/to/gp-run"`). Neither run reads the other's folder by
default.

The Matérn smoothness ν of the GP profile is fixed at 3/2 in v1
(`RadialGP`'s default). Other values change the feature-width equivalent
quoted below, so a `nu` setting would change the ℓ grid and the check's
message with it.

### Stages

Every run has the same stage names; `profile` and `compare` do different
work for each model.

1. `load`: the `OIData`, selections and error floors, as elsewhere.
2. `overview`: uv coverage, the half-fringe λ/2B of the longest baseline,
   V² against baseline. It sets the **resolution scale** that later
   defaults use (inner radius, ring-width floor, ℓ grid), so every such
   default is stated in half-fringes, not in mas.
3. `geometry`: a grid over `inc` and `pa` with a single-ring profile at
   each point, then a single-ring fit from the best cell, then (for the
   GP) `MultipoleDisk.freeze_geometry(data)`. Inclination and position angle
   are multimodal (and `pa` is undefined as `inc` tends to 0), so they
   are never left to a local optimizer from one start. The grid is
   uniform in cos `inc` (as `IsotropicInclination`) and `pa` is on
   [0°, 180°): an axisymmetric disk without `height` is identical under
   `pa` → `pa` + 180°, so a grid over the full circle would have two
   equal minima on every disk. The grid map goes to `grids.h5`. The
   single-ring fit's profile also gives the dynamic range that sets the
   GP's σ. It is a geometry device only: fitted to a multi-ring disk, one
   ring lands near the flux-weighted radius with an inflated width, so
   nothing else (the radial bounds, the ring count) is read from its
   radius.
4. `profile`:
   - `"rings"`: fit *N* = 1, 2, … rings by Levenberg–Marquardt with the
     geometry free from its grid value, to the MAP, with the Laplace
     covariance. Each added ring starts at the radius of a one-ring scan
     over a grid in half-fringes (the other rings held at their fit), at
     a width of one half-fringe, with *N* capped by a setting. This
     needs no GP run; `rings_from` replaces the scan. After each fit
     every ring is dropped in turn, rescanned (half-fringe/2 steps) with
     the rest held and everything refitted; a lower BIC is kept (two
     passes at most), since a greedy addition can leave a ring in a
     local minimum.
   - `"gp"`: fit the `RadialGP` profile (and harmonics, and spectral
     index when on) at frozen geometry, so each refit is one
     matrix-vector product per call, at each ℓ of the grid (below).
5. `compare`: within the model.
   - `"rings"`: Laplace evidences and BIC across *N*; the preferred *N*
     is kept.
   - `"gp"`: Laplace evidences (`log_evidence`) across the ℓ grid; the
     best is kept. The evidence is conditional on the held parameters,
     which is the cleaner use, as in the tutorial.
6. `posterior` (optional): NUTS over the free parameters in flat
   coordinates, geometry released.
7. `quicklook`: the executed notebook, laid out like the tutorial's
   figure: image, residual map, profile, colour, lopsidedness, residuals.

The preferred fit is saved as `models/best`, alongside `models/geometry`.
There is no rings-against-GP evidence in a run; it would compare two
runs, and the tutorial's caveats about it (the GP value holds the other
parameters at their MAP with no Occam factor, and the ring family may not
contain the truth) stay in the tutorial.

### Checks

In this order. The first is χ²/N of the model on the quoted errors, never
rescaled; the same rule as above (a fitted error scale makes χ² ≈ 1 by
construction, so `error_scale="fit"` adds the check for a scale well above
1). Most of the rest are the caveats of the `multipole_disk_rings` tutorial,
turned into pure functions of plain numbers. The model-specific ones apply
to the entries named.

1. `chi2_reduced`: χ²/N on the quoted errors.
2. `length_scale` (`"gp"`): the Laplace evidences across the ℓ grid. If
   they differ by less than a few nats (threshold a setting), `warn`:
   the data do not measure ℓ, which is a prior choice and not a result,
   and the profile's fine structure is the prior wherever the data are
   silent. A Matérn length is not a feature width; the message quotes the
   squared-exponential equivalent (matching the curvature at zero lag,
   ℓ·√((ν−1)/ν): ℓ/√3 at ν = 3/2, the pipeline's value) and the
   half-fringe.
3. `ring_count` (`"rings"`): the evidence and BIC differences between *N*
   and *N* ± 1; `warn` when they do not clearly prefer the chosen *N*, or
   when a ring's width is below the half-fringe (its signature in the
   visibilities is then almost nil and the width is pressed against its
   prior edge).
4. `inner_edge` (`"gp"`): light the profile puts in an unresolved core
   that trades against the star's flux. Defined on the single fit: the
   fraction of the disk's light within one half-fringe of `r_in`, and the
   Laplace correlation between the disk's flux and the light in the
   innermost nodes. `warn` when the fraction is above a setting or the
   correlation magnitude is above a setting (both reported). The tutorial
   saw about a quarter of the light go to such a core with the edge at
   0.5 mas, which this catches without a second fit.
5. `outer_edge`: the fraction of the model's light within one half-fringe
   of `r_out`, a fixed radial width that does not depend on the node
   count or layout (`n_r`, which may change with the data through
   `suggest_n_r`), as a proxy for light beyond the outer edge that the
   model cannot place. `warn` above 1 percent. The remedy it names is to
   refit with a larger `r_out`.
6. `inclination`: `inc` near 0 leaves `pa` unconstrained (the Laplace
   standard deviation of `pa`, or the width of the geometry grid's
   ridge); `warn` below a setting, and `pa` is then reported without an
   uncertainty. Also `warn` when the `geometry` grid has more than one
   comparable minimum.
7. `absorbed_companion`: the disk absorbing a compact companion. The
   harmonic degree is compared with the point-source residual (a grid
   search on the residuals, as in `BinaryPipeline`, run as a check and not
   as a fitted stage); `warn` when a significant point source remains in
   the residuals, or when raising the degree removes it. A low degree is
   the default for this reason.

Added 2026-10-18 (see "Lessons from imaging real discs"), after the checks
above in `CHECKS` order: `structure_unconstrained` (`"gp"`, after
`length_scale`), `render_consistency` (`"gp"`) and `at_bound` (both models).
`length_scale` also warns when the best ℓ is on an edge of the grid after its
one extension.

As for `StarPipeline`, warnings captured inside a stage (non-convergence
and so on) become checks appended after these, only when they occurred.

### Priors and defaults

Group-invariant priors throughout: `IsotropicInclination` for `inc`,
an isotropic Gaussian for harmonic components (a uniform box would favour
the diagonals of the complex plane), uniform for companion offsets, log-uniform for
scales (ring radii and widths, fluxes, temperatures, GP `sigma`). Position
angles are `AngleVector`s: a bounded uniform prior puts a hard wall at the
wrap, and NUTS chains pile up against it. With `Harmonics` on, `pa` and
`pa` + 180° with the first-harmonic phase turned by 180° are the same
disk, so a sampler can move between them; the pipeline reports `pa` folded
into [0°, 180°) with the harmonic phase rotated to match, per sample, so
the summary is unimodal. (To be checked against `MultipoleDisk`'s harmonic
convention when the build starts.)

Radial bounds are settings, defaulting to an inner edge at one
half-fringe and an outer edge from the field the data set
(`virgil.imaging.image_field`), not from the geometry stage's single
ring. The `outer_edge` check reports whether under 1 percent of the light
lies beyond the model's reach.

GP hyperparameters are set from the data, not from a known truth: the ℓ
grid has ℓ/√3 ≈ 1, 2 and 4 half-fringes, chosen by evidence, and σ comes
from the dynamic range of the `geometry` stage's single-ring profile (the
spread of ln I₀ over the radial range). NUTS sampling of the
hyperparameters, over the whitened latents, is a later provisional
upgrade, not in v1.

`spectral_index` turns on automatically when there are more than a few
channels (the number is a setting); below that it is off. A companion
search is off by default and enabled by an expert setting that adds the
point-source grid search on the disk-subtracted residuals and fits any
detection jointly, as in the tutorial.

### Relation to the tutorial

`multipole_disk_rings` gets a closing "As a pipeline" section running
`DiskPipeline(data, "rings", ...)` and `DiskPipeline(data, "gp", ...)` on
the same simulated data. The tutorial's scene has a planet, so the section
runs with the opt-in companion setting on; without it the pipeline has no
planet to compare. The tutorial fits the rings first, from a start 5
percent from the truth, then starts the GP from that fit's geometry, disk
flux and planet; the pipeline runs the two models independently, from a
geometry grid, so the GP's χ²/N and evidence differ from the tutorial's by
more than rounding, and a tolerance loose enough to absorb that would test
little. The section therefore asserts against the truth (the ring radii and
the planet's position within a few Laplace standard deviations, χ²/N on the
quoted errors near 1 for the rings and near the tutorial's for the GP),
not against the tutorial's evidences. The tutorial fixes its GP
hyperparameters by hand from the scene's scale, which a pipeline cannot
do: the GP run uses its own ℓ grid and σ, and the difference is stated.

### Lessons from imaging real discs (2026-10-18)

From virgil-validation#133: nine VLTI/PIONIER post-AGB discs (De Prins et
al. 2026), five strategies (MaxEnt, isotropic and anisotropic Cartesian
`GaussianField`, a polar `MultipoleField`, and a `MultipoleDisk` with a
residual field), compared with the published ORGANIC images. Each lesson is
marked **built** (in `DiskPipeline`), **already** (it was the default) or
**planned** (a model or API change, not built).

1. **Error scale before evidence. Built** for `"gp"`. The MacKay scales on
   the nine targets were 0.81 to 1.34 (below 1 on five, so PIONIER's errors
   are conservative on about half the targets, not all), and every fit
   stalled at χ²/N of 0.5 to 0.7 *on the rescaled errors*, which is the
   flexible profile fitting noise, not a sign of large errors. The scale is
   estimated once, from a baseline fit at the middle ℓ with the companion in
   and the geometry released (a missing component would be absorbed into
   the scale), and then fixed, so `log_evidence` is on one footing across ℓ.
   It is the "rescale then fix" mode that the earlier prohibition of
   `error_scale="fit"` with `"gp"` allowed for. χ²/N is still quoted on the
   quoted errors, and the scale is reported. Planned: the same for `"rings"`
   (`laplace_cov` without Gaussian-process latents needs γ from the Hessian).
2. **A grid must not choose its own edge. Built.** The ℓ/√3 grid already
   started at one half-fringe (ℓ ≥ 1.7 half-fringes, well above the pixel);
   the failures on PIONIER were evidence picking the smallest ℓ of a
   narrow grid (beads, χ²/N about 0.5). Now the best ℓ at either end extends
   the grid once that way, never below `ell_floor_halffringes` (0.5);
   `on_grid_edge` is recorded; `length_scale` warns if the best is still on
   an edge. A σ grid does not exist in the pipeline (σ comes from the
   single ring's dynamic range), so there is nothing to widen there.
3. **The outer edge comes from the field, never from the ring. Already.**
   `r_out` is half of `image_field`; capping a polar GP at 1.5 to 2.5 times
   the rim radius cost about 560 nats (HD 95767) and 1080 (IW Car) and
   degraded the images. The `outer_edge` check's remedy (a larger
   `r_out_mas`) stays the only route to changing it; no stage sets `r_out`
   from a fitted radius. Faint resolved outer flux is real and a resolved
   background cannot stand in for it.
4. **Inner edge is the larger of the half-fringe and the star's radius.
   Built.** The hole is `max(λ/2B_max, star_diameter_mas / 2)`; the
   `inner_edge` check stays. On poorly constrained data (PS Gem) shrinking
   the hole changed nothing (flat evidence, an image that failed against
   ORGANIC), which `structure_unconstrained` now flags: flat evidences
   across ℓ and radial profiles that correlate below 0.8 between the fits.
   It is a within-run proxy for "unstable across strategies"; comparing
   strategies is a comparison of runs.
5. **The star model. Built.** A uniform disk (`star_diameter_mas`) and a
   power-law spectrum (`star_index`; or `star_temperature_k`) are settings.
   The disk's spectral index is conditional on the star's spectrum (on the
   real discs it moved by 0.3 to 1.4 with the star's index, and ln Z by 32
   and 98 for HD 95767 and IW Car), so the summary's `model.star` section
   records it beside the index. Planned: a fitted star diameter.
6. **An over-resolved background. Planned.** PIONIER's short baselines
   need a `models.Resolved` term and its index runs to a prior bound on
   nearly every target. It is a model change (a fourth part in the
   `"gp"` System, with `at_bound` already catching the bound), so it is a
   future `resolved=True` setting. When built, report the total
   over-resolved flux (the background plus the disk's light beyond `r_out`)
   and not the separate indices; today `outer_edge` reports the light near
   `r_out`.
7. **Prior bounds. Built.** An `at_bound` check lists every fitted parameter
   within 0.1% of a prior bound (from `fit`'s `at_bound`, as
   `BinaryPipeline`'s `start_at_bound` does). The companion flux prior is
   already `LogUniform(1e-4, 1)`, wider than the proposed `(1e-3, 1)`.
8. **Render consistency. Built as a check.** `render_consistency` compares
   χ²/N of the model with that of its disk rendered to pixels, on the channel
   nearest the reference wavelength and with the disk's colour removed (a
   pixel image has one brightness), the star and companion kept. On the
   nine targets the disc-plus-field strategy failed a 10 per cent guard on
   seven (not six), the rendered χ²/N being 32 to 50 per cent above the
   model's. `suggest_n_r` sizes the nodes for the visibilities, where χ²
   converges early, and under-resolves a *pixel* image (IW Car: rendered
   2.44 at 48 nodes, 1.92 at 192, the model's 1.86); the pipeline does not
   change `n_r`, and the check's remedy is a larger `n_r`.
9. **Tied geometry. Not in `DiskPipeline` (it uses no `MultipoleField`).**
   Reading `brightness` of a `MultipoleField(geometry_from=...)` outside
   `System.resolved()` uses the fallback geometry by design (it is
   documented as the value for an Image evaluated outside a System), so a
   user who renders the field before resolving sees a wrong image without an
   error. A separate small fix: warn, or raise, when `Image.brightness`
   reads an unresolved tie. Tracked outside this pipeline.
10. **Which model wins. Guidance; the default is unchanged.** The polar GP
    (geometry from our own parametric fit) matched ORGANIC best on most
    targets (NCC 0.93 to 0.98), but the Laplace evidence usually preferred
    the Cartesian `GaussianField` (by about 20 nats for IW Car and EN TrA;
    polar was preferred only on IRAS 15469, and the anisotropic Cartesian
    field on HR 4049 and EN TrA). It strongly disfavoured polar on
    IRAS 08544, a binary core plus a ring (−262 nats), and by 186, 57 and
    44 nats on AI Sco, PS Gem and HD 95767. So `"gp"` (a `RadialGP` with
    harmonics, polar by construction) is the model that assumes a
    smooth, star-centred, low-order azimuthal disk; read a large evidence
    deficit against a Cartesian field (not computed by this pipeline) as a
    reason to doubt that assumption (a binary core, a clump), not as noise.
    `"gp"` stays the default (Ben's decision, 2026-10-15); changing it, or
    adding a Cartesian entry to the registry, is Ben's call.

### Decisions

Ben's, 2026-10-15.

- **Two options, not one.** The analytic (`"rings"`) and non-parametric
  (`"gp"`) models are separate, parallel options, each with its own
  stages, settings and checks. This note's reading of Ben's "split the GP
  and the analytic possibilities as two options". The GP's ℓ grid by
  evidence with σ from the data is its default (NUTS over the
  hyperparameters is a later provisional upgrade).
- **Companions.** One star by default; a companion search is opt-in, an
  expert workflow.
- **Colour.** `spectral_index` turns on automatically with more than a
  few channels. The rings model's colour is a power-law index
  per ring; `ring_colour="blackbody"` fits a temperature instead, since over a
  1.5× band a power law misses a blackbody ring by several percent (job
  18666218: rings χ²/N 1.32 against 1.01 for the parametric blackbody fit).
- **Scope.** The colliding-wind cone geometry of the Apep analysis is out
  of v1; it may become a registry entry later, with no new stage.
- **Timing.** Build only after the `MultipoleDisk` PR stack merges (its code
  is on `main`; virgil#356 is the last piece); stay provisional until the
  `MultipoleDisk` API is unchanged for a release.

### Decisions

1. **Default model.** `model=None` is `"gp"` (Ben, 2026-10-15), over an error
   that asks for a choice or `"rings"`. The snapshot test of the defaults
   pins it.

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
- Before the pipelines are promoted to stable, settle `global_nsigma`. It is a Šidák
  estimate with a nominal number of resolution elements, not a simulated
  false-alarm probability: grid maxima are correlated and the resolution
  element is anisotropic, so the true trial count may be far off. It is
  labelled as an approximation in `summary.json`
  (`global_nsigma_method`), the `detection` message and the docs, which
  point to `injection_recovery` and `gaussian_null`. Either calibrate it by
  simulation or drop it from the frozen summary keys and check.
