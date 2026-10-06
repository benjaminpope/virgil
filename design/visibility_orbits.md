# Orbits fitted directly to visibilities across epochs

Status: plan for 0.4. Stage 1 is implemented alongside this note.

This note extends `design/orbit_scene_joint_fitting.md` (requirements
R0–R9, conventions §2), which built the pieces: `KeplerOrbit` and its
`frame`, `SourceModel.at`, `Attached`, the per-sample time path of
`OIData.model`, a list of models (one per dataset) in `fit` and
`numpyro_model`, and per-dataset `noise`. What it leaves open is the
workflow of fitting one orbit to many epochs of interferometric data
efficiently and reproducibly, from setting up the epochs through starting,
sampling and checking the fit to forecasting the next epoch. That is
the subject of this note.

Examples are generic (a binary, a binary with a disc or a dust cone). Where
a real analysis motivated a requirement, it is cited as a worked example:
Apep (a WC+WC binary with a dust cone) with VLTI/GRAVITY and NACO SAM, a
joint fit that exists as a prototype outside virgil.

## 1. Motivation

* **Position-only fits of extended scenes are biased.** Fitting a
  companion's position epoch by epoch and then fitting an orbit to the
  positions works for two point sources. With extended emission (a disc, a
  dust cone, a halo) the per-epoch binary fit absorbs the extended
  structure into the companion's position and flux. In the Apep example, a
  position-only fit gave a separation of 47 mas against 26.40 ± 0.27 mas
  from the joint visibility fit. The orbit has to be fitted to the
  visibilities and closure phases themselves, with the extended scene in
  the model.
* **Cost.** The per-sample time path evaluates the scene at every sample's
  own time. With a 64-ring `TruncatedCone`, NUTS over five datasets took
  8 h on a CPU node and 24 GB of memory (it ran out at 8 GB). For a
  binary whose period is years, the scene moves by microarcseconds within
  a night: one snapshot per dataset gives the same likelihood at a cost of
  one orbit solution per dataset.
* **Bookkeeping.** Per-dataset nuisances keyed by list position caused a
  real bug when the list was reordered. Epochs and datasets must be keyed
  by name.

## 2. Requirements

General requirements, numbered for reference in the stages below.

### Epochs and nuisances

* **V1 Epochs hold datasets.** An epoch is one or more datasets: several
  files of one night, or two filters a day apart (treated as one epoch or
  two). A snapshot at each dataset's mean MJD, or at the epoch's, is enough
  when the scene moves much less than λ/B within it; the spread of times
  about the snapshot is reported so this can be checked.
* **V2 Names.** Epochs and datasets are keyed by name, never by position.
  Names are unique across epochs and datasets.
* **V3 Per-dataset parameters beside the shared orbit.** Error terms
  (`vis_scale`, `phi_scale`, or `hierarchical_scales` across datasets),
  the North angle and wavelength (plate) scale per instrument, the flux of
  an extended component per instrument, and a halo per filter. These are
  set by name, per epoch or per dataset, and may be tied between datasets
  (one value per instrument). A tied per-instrument flux scale for an
  extended component must accept a spectral flux (a `BlackBody` or any
  `Spectrum`, e.g. a scale times a spectrum shared across instruments), not
  only a scalar, so that one SED is shared and only its normalisation is
  per instrument.
* **V4 Shared shape with optional per-epoch offsets.** The extended
  component's shape is shared across epochs, with an option for per-epoch
  offsets δ_e ~ N(0, τ²) of chosen parameters (a hierarchical shape), with
  τ log-uniform.
* **V5 Anchors and the scene origin.** `Attached` places a component on
  either star or at a fraction of the way between them (`anchor=`). The
  scene's origin is the primary: `KeplerOrbit` gives the secondary's
  offset from the primary, and the phase reference of the visibilities is
  the primary's position. The photocentre is not the origin; a scene whose
  reference should be the photocentre or barycentre places both stars by
  `anchor` fractions. Documented in the conventions page.

### Conventions

* **V6 Convention tests, one each.** dz is positive away from the
  observer; `inc` < 90° is anticlockwise motion on the sky; Ω is the node
  where the secondary recedes; ω is the secondary's argument of periastron.
  Without radial velocities, (Ω, ω) and (Ω + 180°, ω + 180°) give the same
  sky motion: summaries report Ω mod 180° (axial) and say so.
* **V7 Conversions.** A docs section converting other conventions to
  virgil's (ω of the primary vs the secondary, Ω of the ascending vs the
  receding node, z towards vs away from the observer, +x East vs West, PA
  measured from North through East), with a table of the main external
  tools (orbitize!, orvara, the Gaia DR3 NSS Thiele–Innes constants,
  the Sixth Catalog of Orbits of Visual Binary Stars) and one
  regression test per row against a published orbit. Worked example: the
  White et al. (2025) Apep frame (+x West, z towards the observer) gives
  virgil Ω = 105.9° (mod 180°) and ω_WC = 169.4°, which would put
  projected apastron at PA 96.2°, close to the GRAVITY measurement of
  96.1°. These numbers are **to be checked** against the output of the
  Apep prototype's orbit build before they become a regression test; they
  are not yet verified.
* **V8 t_ref.** `KeplerOrbit` counts `dt_peri` from `t_ref`, by default 0.
  With MJD times and the default, the phase is silently wrong. Warn when
  times look like MJDs (|t − t_ref| > 15000 d) and `t_ref` is 0. Requiring
  `t_ref` is a breaking change, left for Ben (§7).
* **V9 PA unwrapping.** Summaries and plots of PA over time unwrap it,
  anchored near a measured epoch, so a track through 0°/360° is continuous.

### Bound components

* **V10 A cone bound to the orbit.** A cone's axis PA follows the line of
  centres (`line_pa` or `towards_primary`) plus a skew, and its tilt
  follows `line_tilt`. Two modes: bound (no offset) and bound plus a free
  offset (`offsets=`), each a fitted parameter. Binding must work with
  traced orbital elements under `jit` and `grad` (a `__check_init__`
  check on traced values broke this before #206).

  **Tilt sign.** An optically thin cone looks the same tilted by +β and
  −β out of the sky: the two are mirror images through the sky plane, and
  their projected brightness is identical. Binding the tilt to the signed
  `line_tilt` therefore gives a likelihood that sees only |`line_tilt`|:
  the data cannot tell which side of the sky plane the secondary is on
  from the cone alone, and the sign of `line_tilt` (hence of dz, i.e.
  which node is receding) comes only from the orbit's motion. Posteriors
  of the tilt should be reported folded (|tilt|), and samplers should
  expect the mirror mode. The stage-1 binding test asserts that ±tilt give
  identical visibilities.

### Calibration

* **V11 Plate scale and North for visibilities (#212).** For `OIData`,
  the plate scale is the wavelength scale: `wavel_scale` = 1/m stretches
  uv by m, equivalent to scaling the sky by m. The North angle is
  `OIData.with_north_angle` and the `north_angle` noise term, with the
  North angle prior from the instrument's astrometric calibration (e.g.
  N(0, 0.3°) for an adaptive-optics camera).

  There is **no fixed default** for the plate-scale prior. For broad-band
  aperture masking the dominant term is the effective wavelength, which
  depends on the filter transmission times the source's SED (times the
  detector response and atmosphere): λ_eff = ∫ λ T(λ) F(λ) dλ / ∫ T(λ)
  F(λ) dλ. Derive the prior from it: compute λ_eff for the range of SEDs
  the source could have (e.g. temperatures or spectral slopes allowed by
  photometry) and for the calibrator, and take the spread of
  λ_eff,source / λ_eff,calibrator as the width of a log-normal prior on
  the scale. For broad-band SAM on red sources this is a few tenths of a
  per cent to about 1 %. The width matters: in the Apep example, at a
  separation of 28 mas, 3 % is 0.8 mas, about half the change in
  separation between 2019 and 2025, so a 3 % prior would decide whether
  that change is detected at all. Stage 6 documents the calculation (and
  may provide a helper taking a filter curve and an SED).

  For GRAVITY the honest analogue is a wavelength scale (its calibration
  is spectral), with a prior from the instrument's wavelength calibration.
* **V12 Bandwidth smearing.** In a broad filter, smearing is chromatic,
  not a pure plate scale. Provide (or document) a bandpass-integrated
  model: evaluate over a few sub-channels across the filter and average,
  as for a Spectrum flux.

### Starting and sampling

* **V13 Ranking starting orbits.** `rank_orbits(scene_fn, epochs,
  orbits, ...)`: the log likelihood of each of many starting orbits (from
  `starting_orbits` or a prior draw), vmapped or `lax.map`ped over orbits,
  sorted. This generalises the tutorial's inline ranking.
* **V14 Start from a positions fit.** The likelihood of visibilities is
  multimodal at the scale of λ/B (fringe aliases), and NUTS started from
  the defaults stuck at a 20-yr alias. Start from a fit of positions (one
  binary fit per epoch, then an orbit fit to the positions), refine with
  `fit` on the visibilities, and start NUTS there with `init_to_value`.
* **V15 Per-chain starts.** Chains start from different good modes (the
  top-ranked distinct orbits) to diagnose multimodality, rather than all
  at one best fit.
* **V16 Short arcs.** For arcs much shorter than the period, sample the
  PA at `t_ref` (`KeplerOrbit.from_position_angle`) and 2Ω/ϖ
  (`orientation_priors`), as 0.3.0 does, not phase, Ω and ω.
* **V17 Prior configurations.** Literature priors may be axial or von
  Mises with period aliases (P vs 2P). Run one model under several prior
  configurations and compare evidence (e.g. by nested sampling or by the
  Laplace evidence at each mode).

### Curvature and forecasts

* **V18 A public Laplace covariance at a fit.** The covariance of the
  fitted parameters from the Hessian of the negative log posterior,
  including priors, noise terms and lists of datasets, as
  `FitResult.covariance()` (or `laplace_cov(result, ...)`). Note: since
  0.3.0 a `LogUniform` parameter's MAP is in log x, so the covariance must
  be reported in the space the fit used and transformed explicitly.

  **Two-stage workflow (later stage, not stage 1).** Full NUTS over scene
  and orbit together is expensive for extended scenes. A documented,
  validated alternative:

  1. *Scene stage.* At each epoch, fit the scene (with its nuisance shape,
     flux and calibration terms) and take its Laplace approximation;
     marginalise it to the per-epoch companion positions, keeping the full
     cross-covariance between epochs. The positions correlate with the
     nuisance shape and flux parameters (shared across epochs), so those
     terms must be kept, not dropped to a block-diagonal form.
  2. *Orbit stage.* NUTS on the orbit with that Gaussian on the positions
     as its likelihood (cheap: no visibilities), with a folded |tilt|
     where the scene has a bound tilt (V10).
  3. *Correction.* Pareto-smoothed importance sampling (PSIS) of the orbit
     samples with the exact visibility likelihood, reporting the Pareto
     k̂ (k̂ < 0.7 reliable; above that, fall back to full NUTS).
  4. *Validation.* Compare against full NUTS on simulated data and on one
     real dataset (an OzSTAR job) before recommending it.

  The evidence so far (Apep example): the scene posteriors are close to
  Gaussian, but the orbit posteriors are not: 2–3 % of samples lie beyond
  the χ² 99 % level, and the variance of the potential energy is 12–15 %
  above k/2 (k parameters). So the Gaussian is used only for the
  positions, never for the orbit, and the PSIS step is required.
* **V19 `gauss_newton_mass` with noise.** It takes `noise=` (and lists of
  datasets) like `fit`, so the NUTS mass matrix includes the error terms;
  plus the no-data check (the Gauss–Newton matrix with no data is the
  prior's) with noise terms present.
* **V20 Forecasts.** Given a template OIData (or several, one per array
  configuration), a new date and an error scaling: simulate the next epoch
  from posterior draws (the median and 16/84 % draws, or a set of
  posterior samples), refit with the new epoch added, and report the ratio
  of new to current posterior width for each parameter. `shift_days` moves
  the times only: real uv tracks depend on hour angle and LST, which are
  not changed. Documented as an assumption; a template from the right
  season (or `coverage.vlti_oidata` with a chosen hour-angle range) is the
  better choice.

### Priors

Invariant (Jeffreys) priors by default (see the priors rule in AGENTS.md
and `orientation_priors`): log-uniform for scales (period, semi-major axis,
fluxes, error scales, τ, plate scale), uniform for locations and angles
(times of periastron within one period, ω, Ω, PA), isotropic inclination
(`IsotropicInclination`), uniform eccentricity. A normal or log-normal
prior is used only for calibration terms with an instrumental prior (North,
plate scale) or as a proposal to be reweighted.

## 3. API

### Stage 1 (implemented)

```python
from virgil import Epochs, KeplerOrbit, OrbitalBinary, fit

epochs = Epochs(
    {
        "2023-05": {"ut": gravity_ut, "at": gravity_at},  # one night, two files
        "2024-03": [naco_ks, naco_h],                     # named "2024-03[0]", "[1]"
        "2025-04": gravity_2025,
    },
    at="dataset",  # snapshot at each dataset's mean MJD; or "epoch"
)
epochs.times        # float64 MJD per dataset
epochs.spread_days  # max |t - snapshot| per dataset: check << λ/B motion

def scene(period, dt_peri, ecc, inc, omega, Omega, a_mas, flux):
    orbit = KeplerOrbit(period, dt_peri, ecc, inc, omega, Omega, a_mas,
                        t_ref=60500.0)
    return OrbitalBinary(orbit, flux)  # or a System with Attached parts

noise = epochs.noise({
    "2023-05": {"vis_scale": dist.LogUniform(0.5, 5.0)},  # each file its own
    "2024-03[1]": {"phi_scale": 2.0},                      # one dataset
})
result = fit(epochs.model_fn(scene), priors, epochs.data, noise=noise,
             init=start)
epochs.loglike(scene(**values), noise={...values by name...})
epochs.index("2024-03[1]")  # -> 3, for the "noise[3].phi_scale" site
```

* `Epochs(epochs, at="dataset", times=None)`: `epochs` maps names to an
  `OIData`, a list or a dict of them. `times=` gives snapshot times by
  epoch or dataset name (for data without times, such as AMIGO DISCOs).
* `Epochs.snapshots(scene)`: `[scene.at(t) for t in times]`, with `t` a
  concrete float64 MJD (no float32 loss of phase). A list of scenes, one
  per dataset, is accepted (per-dataset fluxes).
* `Epochs.model_fn(scene_fn)`: a model function for `fit` and
  `numpyro_model` returning the list of snapshots.
* `Epochs.noise(terms)`: one dict per dataset from name-keyed terms;
  dataset entries override epoch entries; unknown names raise.
* `Epochs.loglike(scene, noise=None)`: Σ `model_loglike(snapshot_k,
  data_k, **noise_k)`, traceable.
* `OrbitalBinary(orbit, flux)`: a time-dependent binary whose snapshot is a
  `BinaryModelCartesian` (the fast binary path); the same scene as
  `System(primary=PointSource(), comp=Attached(PointSource(flux), orbit))`.
* The t_ref warning (V8) in `KeplerOrbit.relative`, `frame`,
  `Attached.at` and `OrbitalBinary.at`.

The per-sample path is unchanged: passing the datasets and a
time-dependent scene to `fit` directly evaluates each sample at its own
time, for scenes that move within an observation.

### Later stages (proposed)

```python
ranked = rank_orbits(scene_fn, epochs, orbits, noise=None)   # stage 2
start = fit_positions_then_visibilities(...)                  # stage 2, name TBD
init = chain_starts(ranked, n_chains, distinct_deg=...)       # stage 2
cov = result.covariance()           # stage 3: Laplace, priors + noise
mass = gauss_newton_mass(model, priors, data, values, noise=noise)  # stage 3
report = forecast(scene_fn, posterior, templates, mjd, error_scale=1.0)  # stage 4
report.width_ratio  # new / current posterior width per parameter
```

Per-epoch shape offsets (V4) are built in the scene function from
per-epoch parameters, with `Epochs.names` giving the keys; stage 6 adds a
helper (`epoch_offsets(name, epochs, tau)`) mirroring
`hierarchical_scales`.

## 4. Stages

Each stage is one reviewable PR.

1. **Snapshots and named epochs** (this PR): `Epochs`, `OrbitalBinary`,
   the t_ref warning, the Attached + TruncatedCone binding test, the
   multi-epoch recovery test. Requirements V1, V2, V3 (noise by name),
   V5 (anchors: existing; scene origin documented), V8, V10.
2. **Starting and sampling**: `rank_orbits`, start from a positions fit
   (`init_to_value`), per-chain starts; the orbit tutorial switches to
   `Epochs` snapshots (results change slightly; see §7). V13–V16.
3. **Curvature**: `gauss_newton_mass(noise=...)`, the no-data check with
   noise terms, a public Laplace covariance of a `FitResult` (priors,
   noise, lists of datasets, the LogUniform log-space note), and the
   two-stage Laplace → NUTS → PSIS workflow with its validation against
   full NUTS (marked for OzSTAR). V18, V19.
4. **Forecasts**: `forecast` from posterior draws over one or more
   templates, the width-ratio report, and the documented `shift_days`
   assumption. V20.
5. **Conventions**: the conversions docs section and its regression tests
   (including the White et al. 2025 example, once checked), the
   convention tests (V6),
   axial Ω reporting, PA unwrapping in summaries and plots. V6, V7, V9.
6. **Calibration and shape**: plate scale and North for visibilities,
   with the plate-scale prior derived from λ_eff over filter × SED
   (#212 follow-up), spectral tied flux scales (V3), a bandpass-integrated
   option for chromatic smearing, per-epoch shape offsets δ_e with τ, and
   evidence over prior configurations. V4, V11, V12, V17.

Stages 2–4 depend on 1; 5 and 6 are independent of 2–4.

## 5. Tests per stage

All tests are small (a few epochs of a four-telescope array, ≤ 10 data
per epoch), run in seconds, use `jax.enable_x64(True)` locally where
needed, and never set x64 globally. Anything over about a minute is
written as a script for OzSTAR and marked so.

1. Stage 1 (`tests/test_epochs.py`, 12 tests, ~15 s):
   * names and ordering; unique names; snapshot times per dataset, per
     epoch and given; name-keyed noise with dataset overrides;
   * an `OrbitalBinary` snapshot equals `BinaryModelCartesian` at the
     orbit's position and the `System` + `Attached` snapshot;
   * snapshot and per-sample log likelihoods agree (exactly for one time
     per dataset, to < 0.05 for a 2 h night);
   * the t_ref warning fires with MJDs and t_ref = 0, and not otherwise;
   * a cone bound to `towards_primary` + skew and `line_tilt`, under
     `jit` with traced Ω and skew, has the right angles and a finite,
     non-zero gradient, and gives identical visibilities at ±tilt;
   * **recovery**: six epochs simulated from a known orbit with noise; an
     LM fit from a start within a fraction of λ/B recovers all eight
     parameters within 4σ of the Laplace (Hessian) errors, with reduced
     χ² < 2;
   * float32 snapshots agree with float64.
   * Real-data check (outside the test suite): the Apep prototype's job on
     OzSTAR (18077844, results due 2026-10-07) is to compare snapshot and
     per-sample likelihoods and posteriors on real GRAVITY data.
2. Stage 2: `rank_orbits` puts the truth first among `starting_orbits`
   for a simulated system; a positions-fit start converges to the truth
   where a default start does not (a small alias case); per-chain starts
   are distinct modes. A short NUTS recovery (≤ 200 samples) is marked
   for OzSTAR; locally only a 20-step smoke test.
3. Stage 3: the covariance of `fit` on a linear-Gaussian model equals the
   analytic one (with priors and a noise term); `gauss_newton_mass` with
   noise matches the Hessian at the truth; with no data, it is the prior
   precision, noise terms included. Two-stage workflow: on a simulated
   multi-epoch binary, the marginal position covariance matches the full
   Laplace; PSIS k̂ is reported; the reweighted posterior agrees with full
   NUTS (the comparison is an OzSTAR job, not a unit test).
4. Stage 4: a forecast with a template identical to an existing epoch
   shrinks the widths by about √(n+1)/√n in the linear regime; several
   templates give one report each; posterior draws are reproducible from
   a key.
5. Stage 5: one test per convention (V6) and one per conversion row (V7),
   including the White et al. (2025) worked example; PA unwrapping is
   continuous across 0° and anchored at the chosen epoch.
6. Stage 6: λ_eff from a filter curve and an SED matches a hand
   calculation; a plate-scale/North recovery on simulated data with a known
   scale; a tied `BlackBody` flux scale across two instruments; the bandpass model converges to the monochromatic one as the
   bandwidth goes to 0; per-epoch offsets with τ → 0 reproduce the shared
   shape; the evidence comparison picks the true period alias in a toy.

## 6. Risks

* **Multimodality.** Fringe aliases make the likelihood multimodal on the
  scale of λ/B; the recovery test starts close for this reason. Stage 2's
  positions-fit start and ranking are the defence; NUTS alone is not.
* **Snapshot validity.** A snapshot is wrong when the scene moves by a
  fraction of λ/B within a dataset (short periods, long sequences).
  `spread_days` reports the spread; a check comparing the companion's
  motion over the spread with λ/B_max could warn automatically.
* **(Ω, ω) degeneracy.** Without RVs it is exact; posteriors are bimodal
  by construction. Reporting Ω mod 180° and ω accordingly must be
  consistent across summaries, plots and conversions.
* **Memory and cost of extended scenes.** A `TruncatedCone` with many rings
  over many epochs is expensive under NUTS (8 h, 24 GB in the example).
  Snapshots cut the cost by the number of frames per dataset; the
  remaining cost is linear in rings and datasets. Heavy runs go to OzSTAR.
* **Parametrisation.** 0.3.0's short-arc parametrisation and the LogUniform
  MAP in log x change what "the fit" means; forecasts and covariances must
  state their parametrisation.
* **Bandwidth smearing vs plate scale.** Fitting a plate scale to absorb
  chromatic smearing biases the separation; the bandpass model is the
  honest fix.

## 7. Decisions for Ben

* Should the orbit tutorial switch to snapshots by default (stage 2)? It
  is far cheaper, but its numbers change slightly.
* Require `t_ref` in `KeplerOrbit` (breaking) or keep the warning?
* The scene origin is the primary (the current convention). Is a
  photocentre-origin option wanted, or is `anchor=` fractions enough?
* Plate-scale priors: no fixed default; document the λ_eff over
  filter × SED derivation (a few tenths of a per cent to ~1 % for
  broad-band SAM on red sources). Should virgil provide a helper that
  computes it from a filter curve and an SED, or only document it?
* Should the two-stage Laplace → NUTS → PSIS workflow become a
  recommended path once validated against full NUTS, or stay a
  documented option?
* Which external orbit catalogues and codes the conversions table covers.
