# virgil 0.4.0 pre-release review: models, imaging, fields and friends

Checkout reviewed: origin/main 420d1b7. Read-only; nothing edited, committed or pushed.
Scope: `src/virgil/{models,imaging,fields,ensemble,scenes,metrics,coverage,simulate,_elr,_geometry,_utils,plotting,harmonics}.py`.
Baseline not repeated: `design/codebase_review_2026-10b.md` (pre-0.3.0).

Depth note: models.py, imaging.py (to l_curve/diagnose in outline), fields.py, metrics.py, scenes.py, simulate.py and the geometry helpers were read in full or near full. ensemble.py, coverage.py, plotting.py and `_elr.py` were only outlined (function lists and signatures); no correctness claim is made about them. Marks: VERIFIED = reproduced with a tiny float32 snippet; READ = by code reading only; UNVERIFIED = needs GPU or larger input.

Priority order of the findings: correctness bugs, API/debt before names freeze, legibility, performance.

---

## 1. Correctness bugs (failing inputs)

### B1. `PolarDisk.is_physical()` crashes with a `VonMises` harmonics [should-fix-before-0.4.0, effectively a blocker for `reject_unphysical`] VERIFIED
- File: `models.py:4974` (`if self.harmonics.prior is not None:` inside `PolarDisk.is_physical`, starts 4966).
- Evidence: `PolarDisk(GaussianRing(15.,2.), harmonics=VonMises(4.0,30.0,degree=20), inc=40., pa=10., n_r=48).is_physical()` raises `AttributeError: 'VonMises' object has no attribute 'prior'`. `Harmonics` has `prior`, `VonMises` does not. `gp_sites` already uses `getattr(self.harmonics, "prior", None)`, so the author knew.
- Impact: `model_loglike(..., reject_unphysical=True)` (`likelihood.py:674`) and `System.is_physical` fail on any scene with a VonMises PolarDisk, i.e. the headline new azimuthal option cannot be used with the physicality guard.
- Fix: `getattr(self.harmonics, "prior", None)`, or give `_Azimuthal` a class-level `prior = None`. Add a test parametrised over both azimuthal classes.
- Effort: 15 minutes.

### B2. `smeared_model` raises `TypeError` for most scene types, with a misleading message [should-fix-before-0.4.0] VERIFIED
- File: `models.py:5746-5803` (`smeared_model`, `_smeared`; error text near 5801). Caller: `oidata.py:~1270-1330` when `read_oifits(bandwidth=True)` / `with_bandwidth` is used.
- Evidence: fails for `BinaryModelCartesian`, `BinaryModelAngular`, `Resolved` (even the documented `System(star, bg=Resolved())`), `Rotated` and `OrbitalBinary` snapshots. The message says it supports "Systems of Component shapes (not Image...)", which does not tell the user that a binary or `Resolved` is the problem.
- Impact: bandwidth smearing is opt-in, but the flagship binary and overresolved-background models cannot be fitted once bandwidth is on; the failure is at evaluation, not at read time.
- Fix: convert the Binary models via `to_system()` first; treat `Resolved` as a zero-offset, fully-resolved piece (zero visibility at nonzero baseline, or skip smearing); for `Rotated` rotate (u, v) and recurse. Where unsupported, name the offending class in the message.
- Effort: 1-2 hours plus tests.

---

## 2. Precision hazards (AGENTS.md: `Precision.HIGHEST` in matmuls, TF32 on A100/H100)

All three are UNVERIFIED on GPU (no GPU here); flagged from the repo rule. On CPU they are exact, so tests will not catch them.

### P1. `LimbDarkenedDisk._profile` [should-fix-before-0.4.0] READ
- `models.py:1323`: `coeffs = ... - self.u @ expand`. `expand` contains binomials up to C(22,11)=705432 with alternating signs for the documented order <= 22; under TF32 (10-bit mantissa) the cancellation is catastrophic. Use `np.matmul(..., precision=jax.lax.Precision.HIGHEST)` or `np.dot`.
- Effort: 5 minutes.

### P2. `RadialGP` and `PolarKernel` contractions [should-fix-before-0.4.0] READ
- `models.py:3597` (`factor @ self.z`), `3620` (`z @ factor.T`), `3820` (`np.einsum("ij,kmj->kmi", factor, z)` in `PolarKernel.values`). `fields.py` `polar_values` correctly uses HIGHEST, so these are inconsistent with the sibling code.
- Fix: add `precision=jax.lax.Precision.HIGHEST` to all three. Effort: 10 minutes.

### P3. Quadrature rules built in the default precision [note] VERIFIED (measured)
- `_legendre_rule` returns `np.asarray` at the ambient precision. A float64 fit gets only about 5e-9 relative error from float32-built rules (measured), which is small but not float64-clean. Storing the rules as numpy float64 leaves lets `cast_tree` (which handles `onp.ndarray` leaves) cast them losslessly under `run_in`. Note-level.

---

## 3. API and technical debt before names freeze at 0.4.0

PolarDisk and its engine are new since 0.3.0, so every name there is still free to change. Names in the older Component classes are frozen already; for those only additive aliases are possible.

### A1. `RadialGP` is two classes in one [should-fix-before-0.4.0] READ
- `models.py:3362` `class RadialGP(_RadialProfile, _RadialFunction)`. The two roles differ in nodes (Gauss-Legendre vs linspace), latent count (`n` vs `n_r`), and whether an exponential is applied; it carries the unused `rule_x`/`rule_w` fields and a `weight` field that means nothing in the function role.
- Fix: `RadialGP` (profile, exp of GP) and `RadialGPFunction`/`GPFunction` (log-free function), sharing a private base for the factor/jitter code. Do it now: after 0.4.0 the multiple-inheritance shape is public.
- Effort: 3-4 hours including docs and tests.

### A2. Every `_RadialProfile` carries a `weight` field [should-fix-before-0.4.0] READ
- Seen in all profiles in `models.py:2690-3200`; it is meaningless alone and only used by `RadialSum`. Move it to a `Weighted(profile, weight)` wrapper or into `RadialSum`'s term list. Effort: 2 hours. If deferred, document as "only used inside RadialSum" and keep it keyword-only.

### A3. Inconsistent parameter names across shapes [should-fix-before-0.4.0 for the PolarDisk family; 0.4.x for the older shapes (frozen)] READ
- Ring radius: `r0` is a centre in `GaussianRing`/`Gap`, an inner radius in `ExpTail`, a reference radius in `RadialLinear`/`RadialPower`; `ModulatedGaussianRim` uses `diam` (a diameter), PolarDisk profiles use a radius.
- Width: `sigma` (GaussianDisk, GaussianRing), `width` (Gap, a standard deviation), `fwhm` (rim, EllipticalGaussian, Flared), `scale` (ExpTail).
- Angles: `rotation_deg` (Image, Rotated) vs `pa`. Ellipticity: `ratio` (EllipticalGaussian, EllipticalLD, TruncatedCone) vs `inc` (rim, Flared, PolarDisk, GravityDarkenedStar).
- Defaults: `inc`/`pa` are required in the rim and Flared, but default to 0 in PolarDisk.
- Recommendation: before 0.4.0 settle PolarDisk-profile names (`r0` always the characteristic radius, `width` always a standard deviation, document FWHM conversions) and write the table into the docs; do not rename the older classes.
- Effort: 1 hour for the table; renames in the new profiles 2 hours.

### A4. Duplicated model/image code in the PolarDisk special cases [0.4.x; document now] READ
- `GaussianArc` (`models.py:525`), `TruncatedCone` (`757`), `ModulatedGaussianRim` (`1947`) call the PolarDisk engine for `model` but keep hand-written `_centred_image`, with the legacy `az_amps`/`az_pas` parametrization. This is a risk to the AGENTS.md invariant `model_on_grid == model` (checked for cone and rim in a repro: agree to 2e-7 when the field contains the source; the 0.15 mismatch seen first was my test's field being too small, not a bug). Longer-term, derive `_centred_image` from the PolarDisk engine and keep these classes as thin constructors. FlaredDisk* stays (design/flared_disk_vs_polar_disk.md).
- Effort: 1-2 days; not for 0.4.0, but add a parametrised model-vs-image test for all three now (1 hour).

### A5. Four azimuthal APIs and two field classes [note] READ
- `Harmonics`, `VonMises`, `PolarKernel` (GP prior) and the legacy `az_amps` form; `GaussianField` and `PolarField` on the field side. Acceptable if the docs give one decision table ("which azimuthal description do I want").

### A6. `Harmonics.__init__(c, s, degree, form, re, im, w, prior)` is a multi-modal constructor [should-fix-before-0.4.0] READ
- `models.py:3952`. Which arguments are valid depends on `form`. Make everything after `s` keyword-only now (free, since the class is new), or split into `Harmonics` (linear) and `PositiveHarmonics` (Fejer-Riesz). Effort: 1-2 hours.

### A7. `GaussianArc.__init__` emits a `FutureWarning` on every construction [should-fix-before-0.4.0] READ
- `models.py:~600-623` (text at 549): "removed in the next release", fragile `stacklevel=4`. The shape change is unreleased relative to v0.3.0, so users of 0.3.0 get nothing from the warning on 0.4.0 construction paths that tests and docs exercise. Either drop the warning for 0.4.0, or reword to a dated removal version and use `warnings.warn(..., stacklevel=2)` from a helper. Check the docs build is not emitting it (warnings-as-errors in CI).
- Effort: 30 minutes.

### A8. Long positional/keyword lists [should-fix-before-0.4.0 for new signatures] READ
- `imaging.clean` (`imaging.py:1471`, about 20 parameters and about 120 lines of validation) and `starting_image` (`864`): make everything after `data` and `star` keyword-only, and group the CLEAN options (`CleanOptions` dataclass or similar) if signatures keep growing.
- `PolarDisk` has many options (`spectral_index`, `wavel0`, `freeze_geometry`, static `n_r`); fine, but all keyword-only after the profile.

### A9. Mixed sentinels and silent gaps [0.4.x] READ
- `second_moment_fwhm` (`imaging.py:584`) returns `None`, `0.0` or a float. Prefer raising or `nan`.
- `_polar_disks` (`models.py:5350`) recurses through `System` and `source` but not `Attached`, so `gp_priors`/evidence skip a PolarDisk inside an `Attached` component (READ; not reproduced).
- `BinaryModelAngular`/`BinaryModelCartesian` (`6127`, `6180`) do no flux validation in `__init__`, unlike `Component`.

---

## 4. Legibility and file structure

### S1. Should models.py (6610 lines) be split? Yes, by role, with a re-exporting package and no import churn [should-fix-before-0.4.0 if done at all; the move is mechanical]
Convert `virgil/models.py` into a package `virgil/models/` whose `__init__.py` re-exports every public name currently importable, so `from virgil.models import X` and docs cross-references (`virgil.models.X`) keep working:

| Module | Contents | Approx. lines |
|---|---|---|
| `_base.py` | `SourceModel`, `Component`, shared helpers (`_check_nu`, `_gp_factor`, `_log_azimuthal_spectrum` which `fields.py` imports) | 160-520 |
| `components.py` | point/Gaussian/uniform disks, `LimbDarkenedDisk`, `GravityDarkenedStar`, `Image`, `FlaredDisk*`, `GaussianArc`, `TruncatedCone`, `ModulatedGaussianRim` | 520-2690 |
| `polar/` (`profiles.py`, `radial.py`, `gp.py`, `azimuthal.py`, `engine.py`, `disk.py`) | radial profiles, `RadialGP`, `PolarKernel`, `Harmonics`, `VonMises`, heights, `polar_disk_cvis`, `_FrozenGeometry`, `PolarDisk`, `_polar_disks`, `gp_priors` | 2690-5350 |
| `scene.py` | `Resolved`, `System`, `Rotated`, `Attached`, `OrbitalBinary`, `BinaryModel*`, `HarmonixModel`, `smeared_model` | 5428-6350 |
| `cvis.py` | the `cvis_*` functions | 6356-end |

- Costs: AGENTS.md line 107 ("New model code goes in `src/virgil/models.py`") must change to point at the package. Private names imported across modules (`fields.py`, `imaging.py`) need the re-export or an explicit import path. Do the move in a single mechanical PR with no edits, so `git log --follow` and review stay tractable; do the A1/A2/A6 API changes in separate PRs before or after.
- Pickles/serialisation: check that nothing pickles by `virgil.models.X` module path for the `_`-prefixed classes (equinox modules in saved fits); re-export keeps `virgil.models.X` valid but `__module__` would change to `virgil.models.components`. Run the I/O round-trip tests.
- Effort: 1 day, mostly review. Not a blocker for 0.4.0; the PolarDisk group is the clean first cut if only one move is made.

### S2. imaging.py (3316 lines) is a grab-bag; split into a package [0.4.x, same mechanics as S1] READ
`imaging/regularizers.py` (TSV, TV, MaxEntropy, Laplacian, StarletL1, LogSum, Centroid, `image_priors`; lines 82-505), `field.py` (nyquist/field_of_view/`image_field`/`second_moment_fwhm`/`dirty_image`/`beam`/`starting_image`; 505-1115), `clean.py` (`clean`, `_CleanScene`, `_refit`; 1118-1860), `evidence.py` (`log_evidence`, `laplace_samples`, `error_scale`, `l_curve`/`LCurve`; 1859-2890), `diagnostics.py` (`diagnose`, `chi2_by_observable`, `residual_image`; 2888-3316), with a re-exporting `__init__`.
- `imaging._evidence_latents` imports PolarDisk knowledge (`_polar_disks`, `gp_sites`); after the models split, that dependency is on `models.polar` only.

### S3. Long functions [0.4.x] READ
`imaging.clean` (about 380 lines, about 120 of validation): move validation to `_check_clean_options`. `PolarDisk` option handling and `polar_disk_cvis` are long but well commented.

---

## 5. Performance hazards

### F1. `PolarDisk.freeze_geometry` loses its speed-up under `vmap` [note] READ
- `models.py` `_FrozenGeometry` is guarded by `jax.lax.cond`; under `vmap` `cond` becomes a `select` and both branches run, so batched fits recompute the Bessel matrices anyway. Document it, or guard with a static Python flag.

### F2. `RadialGP` recomputes a dense Cholesky per call [note] READ
- Fine for n up to about 100; document the limit.

### F3. `dirty_image` builds dense `(k, npix)` complex128 matrices [0.4.x] UNVERIFIED
- `imaging.py:758`. For large DISCO data (k about 1e5) and npix about 256 this is hundreds of GB. Chunk over samples or use `UVGrid`/FFT when the data are on a lattice.

### F4. `metrics.align` is O(n^2) NCC calls [note] READ
- `metrics.py:258`: each NCC is a separate `map_coordinates`; `max_shift_mas` of 20 pixels means 1681 calls, un-jitted. Acceptable for contest use; jit or `vmap` the shift table if it becomes a hot path.

### F5. `clean._atom_norms` does O(npix^2) jvps [note] READ (by design).

---

## 6. Checked and found fine (no action)
- `fields.py`: `GaussianField.evaluate`, `PolarField.polar_values` (HIGHEST), radial-factor jitter, bilinear `polar_interpolate` with edge fill, `polar_field_priors`, `geometry_from` tie via `System._tied`.
- `_geometry.py`: `rotate`, `position_angle` (the mod/360 rounding case is handled), `find_uv_grid` (lattice test, duplicate-index guard), `fringe_scales`, `smearing_scale`, `check_az_prof_nonnegative` (traceable, true-value minimum).
- `metrics.py`: `resample` (flux-conserving via cumulative sums), `ncc`, `l1_score` (weighted median), `lawson_sigma_over_peak`, `align` sign conventions.
- `scenes.py` (`ring`, `spiral`, `gaussian_blob`) and `simulate.py` (`simulate`, `bias_test`, `_shift_times`): no bugs found. `bias_test` runs `n` sequential `fit` calls; each is jit-cached only if shapes repeat (they do).
- Cone model-vs-render comparison: no bug (see A4).

## 7. Not reviewed in depth
`ensemble.py` (1281+ lines; function outline only), `coverage.py`, `plotting.py`, `_elr.py`, the end of `imaging.py` (`diagnose` internals). A follow-up pass should include an `ensemble.combine` correctness read.

## Summary table

| ID | Severity | Where | Verified | Effort |
|---|---|---|---|---|
| B1 | should-fix (near-blocker) | models.py:4974 | yes | 15 min |
| B2 | should-fix | models.py:5746-5803 | yes | 1-2 h |
| P1 | should-fix | models.py:1323 | no (GPU) | 5 min |
| P2 | should-fix | models.py:3597, 3620, 3820 | no (GPU) | 10 min |
| A1 | should-fix | models.py:3362 | read | 3-4 h |
| A2 | should-fix | models.py 2690-3200 | read | 2 h |
| A3 | should-fix (new) / 0.4.x (old) | many | read | 1-3 h |
| A6 | should-fix | models.py:3952 | read | 1-2 h |
| A7 | should-fix | models.py:~600-623 | read | 30 min |
| A8 | should-fix | imaging.py:1471, 864 | read | 1-2 h |
| S1 | optional split | models.py | n/a | 1 day |
| A4, A5, A9, S2, S3, F1-F5, P3 | 0.4.x / note | see above | mixed | varies |
