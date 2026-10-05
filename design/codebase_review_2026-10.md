# Codebase review, October 2026 (main at `5239a70`)

An adversarial review of virgil before its first release as `virgil-astro`
0.2.0. It covers correctness, references, performance, legibility,
documentation, maintainability and progress against `design/imaging_plan.md`.

Each finding is labelled **CONFIRMED**, with the command and output, or
**PLAUSIBLE**. The reproduction scripts are throwaway and were not committed.
Their names are given as `math/<name>.py`, `perf/<name>.py` and
`pkg/<name>`.

## Executive summary

- **Not ready for 0.2.0 as it stands.**
  - Two packaging defects and one runtime defect would reach users on day
    one. All three are confirmed.
  - A real-data reader failure affects GRAVITY users.
  - All four are small fixes: roughly a day's work, plus a CI job to stop
    them returning.
- **The core mathematics is sound.**
  - I re-derived these and found them correct: the visibility models, the
    sign conventions, the DFT and MFT, the beam, the `GaussianField`
    spectrum, the Laplace evidence, MacKay's error scale, the classic-MaxEnt
    condition, the Ruffio quantile, `nsigma`, the Planck ratio, and ELR11
    eqs 30 and 32.
  - The closure-phase whitening does exactly what its docstring claims.
- **Correctness bugs found:**
  - The rim's non-negativity check passes negative brightness.
  - Phase-error fits use the wrong normaliser at large σ.
  - V² → amplitude error propagation has no floor.
  - LM in float32 never converges.
  - All four are confirmed and none is severe for typical use.
- **Top risk 1: JAX-version fragility.**
  - The declared floor `jax>=0.4.30` cannot run `fit`.
  - On the newest JAX for Python 3.11 (0.10.2), arrays cached under one x64
    mode break the other, so 8 tests fail on macOS.
  - CI tests neither case.
- **Top risk 2: the release artefacts are untested.**
  - No CI job builds or installs the wheel or sdist.
  - `__version__` is missing.
  - Five hard dependencies are unused or only optional.
  - The sdist ships tests that cannot run.
- **Top risk 3: the GRAVITY reader.** It fails on 50 of 136 real
  per-instrument reads of archival files, because it matches triangle rows to
  baseline rows by MJD within 8.6 s.
- **Performance is good where it matters.** Grids, limits, the DFT and MFT,
  and the whitening are already efficient. There are large cheap wins in the
  solvers and evidence helpers: 10× for parametric LM, 17× for
  `gauss_newton_mass`, 11× on the first `log_evidence` call, and no
  recompiles for float hyperparameters.
- **The design notes have drifted from the code.**
  - The stage logs quote numbers from before Stage 6.0.
  - Six MWEs have not been re-run since then.
  - The hour totals don't add up.
  - The Apep work is described as unmerged, but it was merged in PR #124.
  - The rename script left false text behind.
- **References are mostly right.** All 56 arXiv IDs resolve. Two errors
  affect the methods: MacKay's eq. 4.14 should be 4.10, and classic MaxEnt
  should cite Skilling 1989.
- **Keep:** the single residual vector, the module-level jit discipline, the
  traced grid templates, the orientation tests, the render-versus-model FT
  test, and the explicit flux convention.

## Blockers for 0.2.0

1. **B1. The declared dependency floors are false.** `pyproject.toml:7-14`
   declares `jax>=0.4.30`, but `_precision.py:36` calls `jax.enable_x64`,
   which jax 0.4.30 does not have.
   - In a clean venv with jax 0.4.30, optax 0.2.4, numpyro 0.15 and
     optimistix 0.0.10, `tests/test_fitting.py::test_fit_recovers_a_binary[lm]`
     fails with
     `AttributeError: module 'jax' has no attribute 'enable_x64'`.
     **CONFIRMED.**
   - The floors cannot all be installed together:
     - optimistix 0.0.10 needs lineax ≥ 0.0.6 and equinox ≥ 0.11.7;
     - the unbounded deps (`matplotlib`, `numpy` …) resolve to unusable
       versions under `--resolution lowest-direct` (matplotlib 0.86 fails to
       build).
   - Fix:
     - set the floors to what CI tests, e.g. `jax>=0.9`, `optax>=0.2.4`,
       `numpyro>=0.18`;
     - give `numpy`, `matplotlib` and `astropy` sane lower bounds;
     - add one `lowest-direct` CI job.
2. **B2. Mixing x64 modes breaks on the newest JAX for Python 3.11.**
   - With the wheel installed under JAX 0.10.2, 8 tests fail on macOS
     arm64. They include the GDS gradient and golden tests,
     `test_classic_maxent_matches_an_independent_calculation`, and two
     `test_long_baseline` tests.
   - Errors:
     - `RuntimeProgramInputMismatch: expected s64[32] but got s32[32]`, from
       `_elr.py:374` reusing the `lru_cache`d mesh of `_elr.py:318`;
     - `lax.add requires ... int64, int32`, from `_closure.py:156`
       (`out.at[self.groups]`) via `oidata.py:657`.
   - The failures depend on test order. Each test passes alone, and the pair
     fails when run together.
   - Cause: integer JAX arrays built under one x64 mode are reused under the
     other. `fit` switches the mode, so a fit followed by `with_model` can hit
     this.
   - **CONFIRMED** by the packaging audit (`pkg/` logs). CI on ubuntu with
     0.10.2 passes; there is no macOS or 3.11 PR cell.
   - Fix:
     - keep cached index arrays (`_elr` mesh `n`, `ClosureNoise.groups`,
       `keep`, `mask`) as NumPy and convert them at use;
     - add a macOS py3.11 CI cell.
3. **B3. The GRAVITY reader fails on archival data.**
   - `oifits.py:33` sets `_MJD_TOLERANCE = 1e-4` d (8.6 s), and
     `oifits.py:441` raises when a triangle leg has no VIS2 row within it.
   - Reading every file in `~/data/apep_gravity`, per INSNAME
     (`math/readall.py`), gave 86 successful reads and 50 errors ("no
     matching baseline"), all in `reduced/`.
   - The largest T3 → VIS2 MJD gap was 85.7 s (`math/mjd.py`).
     **CONFIRMED.**
   - Fix: match on the row's `INT_TIME` window, the core of Stage 6a.0
     (about 2 h).
4. **B4. Metadata and artefacts.**
   - No `__version__` (**CONFIRMED**: `getattr(virgil, "__version__")` is
     missing).
   - Unused hard dependencies:
     - `termcolor` is imported nowhere;
     - `pyoifits` is used only by tests;
     - `astroquery` only by `legacy`;
     - `chainconsumer` and `pandas` only by one plotting helper.
     - The clean install is 71 packages and 605 MB.
   - The TOML-table `license` form is deprecated and setuptools warns.
   - The sdist ships `tests/test_*.py` without `tests/_test_data.py`,
     `_compiles.py`, `__init__.py` or the data.
   - No CI job builds and installs the wheel.
   - All **CONFIRMED** by the packaging audit (`uv build`, `unzip -l`,
     `tar tzf`, clean venv).
   - Fix:
     - `__version__ = importlib.metadata.version("virgil-astro")`;
     - move those dependencies to extras (`plots`, `legacy`, `test`) and
       delete `termcolor`;
     - `license = "MIT"`, `license-files`, `setuptools>=77`;
     - add a MANIFEST.in;
     - add a `build` job (build, install into a fresh venv, import, test)
       before the publishing PR #128 lands.
5. **B5. Decide on `Tabulated` before it is a public name.**
   - `spectra.py:173` calls itself provisional until Stage 6a's `Nodes`, but
     it is in `virgil.__all__`.
   - `design/chromatic_sources.md:48-56,81` documents a different API:
     `Tabulated(wavels, values)`, `.flux.values`, and a reference at
     `wavel0`. The code has `Tabulated(ratio, wavel)`, whose reference is the
     node mean.
   - Fix: either commit to the shipped API and fix the note, or drop it from
     `__all__` until `Nodes` exists.

## Findings by dimension

### 1. Correctness and mathematics

**Verified correct**, re-derived and, where cheap, checked numerically:

- **Visibilities:**
  - uniform disk 2J₁(x)/x with x = πθρ;
  - Gaussian exp(−2π²σ²ρ²), and its FWHM form;
  - the modulated thin ring (−i)^m J_m(2πr₀ρ) cos(m(ψ−φ_m)), consistent with
    `offset_phase` = exp(−2πi(u·dra + v·ddec)).
- **Coordinates:** `image_coordinates` and the separable DFT. The MFT has the
  same sum. `dirty_image` uses the inverse sign. The beam is
  Σ = M⁻¹/4π² from the quadratic expansion of the dirty beam.
- **`GaussianField`:** the Neumann-Laplacian eigenvalues
  (2/h)² sin²(πj/2n) of DCT-II, and the normalisation Σ S / N = σ² under the
  orthonormal IDCT.
- **`log_evidence`:** −½χ² − ½|z|² − ½ log det(I + JᵀJ) is the Laplace
  evidence for standard-normal latents.
- **`error_scale`:** s² = χ²/(N − γ) with γ = Σ λ/(1+λ).
- **`classic_maxent`:** −2αS = Σ λ/(λ+α), in the entropy metric diag(√b).
- **Ruffio quantile:** the truncated-normal quantile, with Newton steps on
  log Q.
- **`nsigma`:** the two-sided conversion.
- **`_planck_ratio`:** the log form.
- **ELR11:** eq. 30 multiplied by ω², and eq. 32.
- **Closure-phase whitening.** For r in col(C), the code's χ² equals rᵀC⁺r
  (1.80603 = 1.80603) and Σ log err = ½ log pdet C (−10.50730 = −10.50730).
  (`math/closure.py`)

Findings, most severe first:

1. **HIGH. The rim's non-negativity check passes negative brightness.**
   `_geometry.py:395`
   - **What:** `check_az_prof_nonnegative` normalises the companion matrix by
     the top-order coefficient, and falls back to 1 when that coefficient is
     below 1e-6. With a zero top-order amplitude the roots are wrong.
     `az_amps=[1.2, 0.0]`, `az_pas=[45, 0]` returns True although
     1 + f(θ) has minimum −0.2.
   - **Why it matters:** `ModulatedGaussianRim` accepts a negative image, and
     `is_physical` with `reject_unphysical` misses it. Fits often start with
     zeroed higher orders.
   - **Fix:** evaluate 1 + f on a dense θ grid (e.g. 64·k points). That is
     simpler and robust. Alternatively, trim the polynomial to its true
     degree.
   - **CONFIRMED** (`math/az.py`):
     `[1.2, 0.0] [45.0, 0.0] check: True true min: -0.2`.
2. **MEDIUM. Phase likelihood normalisation.** `likelihood.py:72-78`
   - **What:** the chord likelihood keeps the Gaussian normaliser −log σ.
     The exact von Mises normaliser is −log(2π I₀(1/σ²) e^{−1/σ²}).
   - **Why it matters:** with fitted `phi_scale` or `phi_error`, noise
     estimates are biased once σ ≳ 0.5 rad:
     - true σ = 0.58 rad gives an MLE of 0.62;
     - true σ = 1.83 rad gives an MLE of 1.31.
     - At σ = 1 the "density" integrates to 1.17 over the circle.
   - It also biases evidence comparisons across noise levels. The docstring
     calls it the small-σ limit, so it is documented, but nothing warns at
     large σ.
   - **Fix:** use `log(2π) + log(i0e(κ))` for the unprojected,
     uncorrelated phase rows. It is one line, and `jax.scipy.special.i0e`
     exists.
   - **CONFIRMED** (`math/vonmises.py`).
3. **MEDIUM. LM in float32 never converges.** `fitting.py:563` (from the
   packaging audit)
   - **What:** `_lm_tolerance` uses `gtol = 1e-4` with no float32 floor.
     - The 3-parameter binary of `tests/test_fitting.py` converges in 7
       steps in float64.
     - In float32 it runs all 1000 steps unconverged.
   - **Why it matters:** about 140× wasted work and a spurious warning.
     `test_float32_and_float64_fits_agree` (`test_fitting.py:74`) compares
     only values, so it hides this.
   - **Fix:** floor the tolerance at about √eps(dtype) times the gradient
     scale, and assert `info["converged"]` in that test.
   - **CONFIRMED** by the packaging audit.
4. **MEDIUM. Closure-phase covariance model: documentation and
   simulation.** `_closure.py:9-25,152-159`
   - **What:**
     - C = D^½ R D^½ with R = TTᵀ/3 is Kammerer's equal-baseline-noise
       approximation.
     - Real closure phases are exact closures T b, so residuals lie in
       col(T), not in col(C) = D^½ col(T), whenever the σ within a group
       differ.
     - `sample()` therefore simulates noise that does not close.
     - The plan (`imaging_plan.md:382`) and `gravity_calibration_review.md:19`
       describe T diag(s) Tᵀ instead.
   - **Why it matters:** with one noisy baseline, the mean χ² was 2.87
     against an expected 3. The effect is small, but the design text
     describes a model the code doesn't implement.
   - **Fix:**
     - say "approximation" in the module docstring, and correct the plan
       and review text;
     - optionally fit s ≥ 0 so that diag(T diag(s) Tᵀ) = σ², which is the
       same code path with a different `chol`.
   - **CONFIRMED** (`math/closure.py`: code 2.871, exact 2.997, k = 3).
5. **MEDIUM. Unfloored V² → amplitude errors.** `oidata.py:409-423`
   - **What:** converting V² to amplitude uses ½σ/√V² with V² floored at
     1e-30. The opposite branch floors |V| at its error.
   - **Example:** V² = 1e-4 ± 0.01 becomes |V| = 0.01 ± 0.5, and V² ≤ 0
     becomes |V| = 0 ± 5e12.
   - **Why it matters:** silent, badly non-Gaussian weights in `vis_mode="amp"`
     and `"logamp"` near nulls.
   - **Fix:** use σ_amp = ½σ/√max(V², σ), and the same for logamp, mirroring
     the existing amp → V² floor. Also say in the docstring that the
     propagation is evaluated at the noisy data.
   - **CONFIRMED** (`math/amp.py`).
6. **LOW. Evidence helpers ignore fitted noise and per-epoch models.**
   - **What:** `log_evidence`, `error_scale` and `classic_maxent` call
     `model.get(path)` and `whitened_residuals` with no `noise` terms
     (`imaging.py:770-794`).
   - **Why it matters:** fitted `noise=` terms are silently ignored, and
     per-dataset model lists fail.
   - **Fix:** reject both explicitly.
   - **PLAUSIBLE.**
7. **LOW. `laplace_cov` and `fisher` run in ambient float32,** unlike `fit`
   (`inference.py:191-241`). For a binary at Δmag 7.5 the covariance differs
   from float64 by 5e-4 relative. That is harmless here, but inconsistent
   with the precision policy in AGENTS.md. **CONFIRMED**
   (`math/laplace_prec.py`).
8. **LOW. `error_scale` evaluates γ at β = 1.** MacKay's update uses the
   eigenvalues of βJᵀJ. The docstring's "one iteration usually suffices" is
   right in practice. Iterating twice inside the function would remove the
   caveat. **PLAUSIBLE.**
9. **LOW. Docstring claims that don't match the code.**
   - `imaging.field_of_view` says "at the shortest wavelength", but
     min(B/λ) gives λ_max/B_min.
   - `OIData._diagonalised` says "eigenvectors of A D Aᵀ", but it uses the
     correlated A D^½ R D^½ Aᵀ.
   - `TV` claims "no preferred direction on the sky", but averaging four
     flips does not make forward differences isotropic.
   - **CONFIRMED** by reading.
10. **LOW. Regularisers act only on the first model of a per-dataset list**
    (`fitting.py:233`). This is documented in `fit`, but a scene whose image
    differs between epochs gets its other images unregularised. **PLAUSIBLE.**

### 2. References

The full table is in [References checked](#references-checked). Errors, most
severe first:

1. **MEDIUM.** `imaging.py:882` and `regulariser_weight_selection.md:201`
   cite MacKay 1992 eq. 4.14 for the β re-estimate. In the published paper
   it is **eq. 4.10** (γ is eq. 4.9, α is eq. 4.8). **CONFIRMED**
   (paper text).
2. **MEDIUM.** `imaging.py:720`, `docs/contributors.md:23` and
   `regulariser_weight_selection.md:176-181` attribute classic MaxEnt to
   "Gull 1989; Skilling & Bryan 1984".
   - Skilling & Bryan 1984 is "historic" MaxEnt (χ² = N).
   - The evidence rule is Gull 1989 together with **Skilling 1989**,
     "Classic Maximum Entropy" (Kluwer, 45–52).
   - **CONFIRMED.**
3. **LOW.** `regulariser_weight_selection.md:211` attributes
   arXiv:0807.3020 to MiRA. It is Renard et al. 2008. MiRA is Thiébaut 2008,
   SPIE 7013, 70131I. **CONFIRMED.**
4. **LOW.** `regulariser_weight_selection.md:233` cites arXiv:2405.04749 as
   an example of sampled regulariser weights. That paper (Chang et al. 2024)
   fits a geometric model. Use Tiede et al. (HIBI, arXiv:2511.17706).
   **CONFIRMED.**
5. **LOW.** `docs/contributors.md:25` credits Kammerer et al. 2020 with
   reducing to the independent closure combinations. Kammerer gives the ±1/3
   correlations; the reduction is virgil's own. **CONFIRMED.**
6. **LOW.** Preprint years now have journal versions:
   - Blakely et al.: AJ 169, 137 (2025);
   - Desdoigts et al.: PASA 43, e075 (2026);
   - Charles et al.: PASA 43, e048 (2026);
   - Dholakia & Pope: PASP 138, 054504 (2026).

   Also:
   - `gravity_calibration_review.md:254,258` marks references "[unverified]"
     that are now verified;
   - Ramani et al. 2012 is missing the author Nielsen.

### 3. Performance

Measured on an M4 CPU, one benchmark at a time, on 64² images and VLTI data
with 495 observables (`perf/`).

1. **HIGH. LM always runs 50 conjugate-gradient steps.**
   `fitting.py:541-548, 275`
   - **What:** `lx.CG(rtol=0, atol=0, max_steps=50)` runs every LM step,
     even for 7 parameters, where CG is exact after 7 steps.
   - **Why it matters:** parametric fits with expensive forward models are
     dominated by it.
   - **Fix:** below about 200 unconstrained coordinates, use
     `lx.QR()` (the optimistix default); otherwise use
     `cg_steps = min(cg_steps, n)`.
   - **CONFIRMED** (`perf/j_lm.py`): a chromatic `GravityDarkenedStar` plus
     companion takes 45.2 s with CG(50), 7.0 s with CG(7) and 4.6 s with QR.
     All three converge in the same 36 steps to the same χ²_red.
2. **HIGH. `gauss_newton_mass` is unjitted, uses forward mode, and inverts
   twice.** `fitting.py:462-486`
   - **What:**
     - The Jacobian is built eagerly.
     - It includes the identity prior rows, so it chooses `jacfwd` with
       4096 tangents.
     - It then runs both `cholesky` and `inv` on a 4096² matrix, and forms
       `jac.T @ jac` twice.
   - **Fix:**
     - jit it at module level;
     - take `jacrev` of the data residuals only, and add the diagonal prior
       curvature;
     - use Woodbury, (I + JᵀJ)⁻¹ = I − Jᵀ(I + JJᵀ)⁻¹J, which needs a 495²
       solve;
     - otherwise `cho_solve`.
   - **CONFIRMED** (`perf/c_gn.py`): 5.1 s → 0.31 s warm and 15.5 s → 0.9 s
     cold, with max |ΔC| = 3e-14.
3. **HIGH. `_residual_jacobian` is not jitted.** `imaging.py:770-794`
   - It backs `log_evidence`, `error_scale` and `classic_maxent`.
   - **CONFIRMED** (`perf/b_evidence.py`):
     - the first `log_evidence` call takes 18.2 s (319 op-by-op compiles,
       14.8 s of compiling), against 1.6 s jitted;
     - warm, 0.45–0.88 s against 0.25–0.34 s;
     - `classic_maxent` on three weights takes 5.5 s, 211 compiles.
   - **Fix:** a module-level `eqx.filter_jit`, as `inference.py` already
     does.
4. **MEDIUM. Python floats are static, so new values recompile.**
   `_precision.py:50-52` leaves Python scalars uncast.
   - Affected: priors built from floats, `Centroid.sigma_mas`
     (`imaging.py:207`), `TV.epsilon` (`imaging.py:129`), and `gtol`,
     `max_step_size`, `max_steps` and `learning_rate`.
   - `starting_image` builds its priors from data-derived floats, so every
     new dataset recompiles.
   - **CONFIRMED** (`perf/e_param.py`, `perf/f_static.py`):
     - `Uniform(lo, 50.)` with three values of lo: 3 compiles at about 1.2 s
       each, against 0.02 s warm;
     - `Centroid(5/6/7)`: 3 compiles;
     - on a 64² field, each fit compile costs about 4 s.
   - **Fix:** cast float leaves of `priors` to arrays in `_Objective`, and
     store `sigma_mas` and `epsilon` as arrays, as `weight` already is.
5. **MEDIUM. `ClosureNoise.from_indices` scales quadratically.**
   `_closure.py:53-54, 88-94`
   - **What:** one `roots == r` scan per group, then a Python loop of SVDs
     over groups that nearly all share one pattern.
   - **CONFIRMED** (`perf/g_closure.py`): 5.2 s at 40k closure phases and
     24.8 s at 100k. `_groups` takes 13.4 s, against 2.6 s with an argsort
     split.
   - **Fix:** split with argsort, and compute Q and the Cholesky factor once
     per distinct incidence pattern.
6. **LOW.**
   - `diagnose` evaluates the residuals eagerly, twice per dataset, once
     only for `.size` (1.9 s first call, 44 compiles). Use `n_independent`.
   - Chromatic `GravityDarkenedStar` solves the surface three times per
     evaluation (`models.py:784-826`), so about 20–30% could be saved.
   - **CONFIRMED** (first-call timing) / **PLAUSIBLE** (saving).

**Measured and efficient. Keep these:**

- the separable complex DFT;
- the MFT on lattices;
- `ClosureNoise.whiten`: 1.9 ms at 40k closure phases;
- the module-level jitted solvers: an L-curve step or a new GaussianField σ
  gives 0 compiles;
- the grid tools: `likelihood_grid` 41×41×30 in 0.09 s warm, with new axis
  values giving 0 compiles;
- the warm `laplace_cov` (under 1 ms);
- `_smaller_gram` (Gram plus Cholesky in 10 ms).

### 4. Legibility and simplicity

1. **MEDIUM. Four names for curvature.** `inference.py` has
   `observed_information`, `fisher_matrix` ("for backward compatibility",
   in a first release), `fisher` and `gaussian_fisher`, plus
   `laplace_covariance` against `laplace_cov`.
   - Fix: keep `laplace_cov`, `fisher` and `gaussian_fisher`, and delete
     `fisher_matrix` and `observed_information` before 0.2.0, while that is
     free.
   - **CONFIRMED** by reading.
2. **MEDIUM. Legacy conveniences in the public API.**
   - `GaussianDiskModel` (a function that used to be a class).
   - `cvis_gaussian_disk` and `cvis_binary_angular`.
   - The binaries' ratio-valued `flux`, which AGENTS.md itself calls a
     "legacy exception".
   - `loglike_nosignal`.
   - A first release is the cheapest moment to drop or quarantine them.
   - **CONFIRMED** by reading.
3. **MEDIUM. Argument conventions are inconsistent across the public API.**
   - Likelihoods take `(values, params, data_obj, model)`
     (`loglike`, `laplace_cov`).
   - `fit` and `numpyro_model` take `(model, priors, data)`.
   - Grids take `(data_obj, model, samples_dict)`.
   - The data argument is called `data_obj` in some places and `data` in
     others.
   - The top-level `load_oi_data` is AMIGO-specific and uses
     `allow_pickle=True`.
   - Fix: at least unify the name `data`, and keep `load_oi_data` under
     `virgil.amigo`.
   - **CONFIRMED** by reading.
4. **LOW. Long modules.**
   - `models.py` is 2309 lines; `plotting.py` is 1284 lines with 28
     functions.
   - Splitting `models.py` into `components` and scenes would help readers.
   - `OIData.__init__` (`oidata.py:77-253`) is 175 lines. Its dict and
     OIFITS paths could share one `_from_record` helper.
5. **LOW. Cross-module private import.** `imaging.py:39` imports the private
   `_reference` from `fitting`; move it to `_utils`.

### 5. Documentation

1. **MEDIUM. Tutorials depend on private helpers.**
   `docs/imaging_ami.md:24` and `notebooks/imaging_ami.ipynb` import
   `virgil._geometry.pixel_offsets`, and several MWEs use `_precision`.
   That is a public-API gap: export `pixel_offsets`. **CONFIRMED.**
2. **MEDIUM. Docstrings incomplete.**
   - 53 public callables with arguments have no Parameters section,
     including the top-level `flux_to_contrast`, `contrast_to_flux`,
     `flux_to_delta_mag`, `delta_mag_to_flux` and `build_model`, and also
     `image_priors`, `nyquist_pixel_scale`, `OIData.model` and
     `with_model`.
   - Six regulariser methods have no docstring.
   - **CONFIRMED** (`pkg/docaudit.txt`). `mkdocs build --strict` is clean.
3. **MEDIUM. No CHANGELOG.** Stage 6.0 changed every 4-telescope χ², and the
   rename changed every import. Users need one page of release notes.
4. **LOW. Stale contributor docs.**
   - `MIGRATION_VIRGIL.md:41` tells agents to "keep using `drpangloss`
     everywhere".
   - Lines 12 and 100 of the same file name a nonexistent
     `virgil-astro[nufft]`.
   - `AGENTS.md:86-110` omits `_closure.py`, `log_evidence`, `error_scale`,
     `dirty_image`, `convolve_beam` and `gauss_newton_mass`, and the import
     DAG omits `fields`, `coverage` and `spectra`.
   - README links `CONTRIBUTING.md` relatively, which breaks on PyPI.
   - **CONFIRMED.**
5. **LOW.** `HarmonixModel` has an API page but is not in `__all__`, and
   `flux_at` and `reference_flux` are missing from the spectra page.

### 6. Maintainability and technical debt

1. **HIGH. CI gaps.**
   - PRs test only py3.12 on ubuntu.
   - No macOS or 3.11 PR cell; that is how B2 got through.
   - No 3.13.
   - No wheel or sdist build.
   - No minimum-versions job.
   - No notebook execution: the sync test only compares stored outputs.
   - harmonix and jaxoplanet are in no extra, so 5 integration tests skip in
     CI.
   - The x64 job is ubuntu py3.12 only.
   - **CONFIRMED** by reading the workflows and comparing local and CI skip
     counts (2 vs 7).
2. **MEDIUM. Lint autofix commits are never tested.** `lint.yml:45-54`
   pushes with `GITHUB_TOKEN`, so the autofixed head commit triggers no
   workflows. **PLAUSIBLE** (standard GitHub behaviour).
3. **MEDIUM. Brittle tests.**
   - `tests/_test_data.py:12` uses `./data/`, so tests fail when run outside
     the repo root, and the file leaks a handle.
   - The pyoifits verify block is always skipped with a warning.
   - Weak tests:
     - `test_models_core.py:76` builds its own Gaussian and never calls
       virgil's likelihood;
     - several finiteness-only checks;
     - `pytest.raises(Exception)` at `test_fitting.py:151`;
     - a constant comparison at `test_fields.py:178`.
   - Public names never tested: `inflated_errors`, `contrast_to_flux`,
     `noise_sites`, `noise_for`, `FitResult` and `Spectrum`.
   - **CONFIRMED.**
4. **MEDIUM. Dead or one-off code.**
   - `legacy/savefits.py`, a "TODO: remove" alias for downstream users who
     cannot exist yet.
   - `plotting.plot_trace_panels` and `plot_recovery_residuals`, which
     nothing references.
   - `legacy.load_oifits`.
   - `scripts/rename_to_virgil.py`.
   - The `sampling = ["blackjax"]` extra, which nothing imports.
   - `data/chi2_ppf*.npy`: 160 MB tracked and unreferenced.
   - `build/` is missing from `.gitignore`.
   - **CONFIRMED** by grep.
5. **LOW. Tooling.**
   - `mkdocs` is unpinned; Material warns that 2.0 breaks plugins.
   - Actions are pinned by tag, not SHA, including the write-token jobs.
   - The JAX cache key includes `run_id`, so it fills the cache quota.
   - The Pages workflow builds the site twice.
6. **LOW. Reader limitation.** Reversed closure-phase legs raise
   (`oifits.py:437`, `oidata.py:736`). The error is clear, but files with
   unsorted station indices cannot be read.

### 7. Progress against the plan

See [Plan versus progress](#plan-versus-progress).

## References checked

All 56 arXiv IDs in src, docs and design resolve; they were checked against
the arXiv API for title, authors and year. About 35 journal references were
checked against Crossref, and 7 claims against the paper text.

| Citation (where) | Checked against | Result |
| --- | --- | --- |
| MacKay 1992, Neural Comput. 4, 415, "eq. 4.14" (`imaging.py:882`) | paper text | **Wrong equation:** 4.10 |
| Gull 1989; Skilling & Bryan 1984, for classic MaxEnt (`imaging.py:720`) | Crossref, text | **Wrong attribution:** add Skilling 1989; S&B 1984 is historic MaxEnt |
| arXiv:0807.3020 as MiRA (`regulariser_weight_selection.md:211`) | arXiv | **Wrong:** Renard et al. 2008 |
| arXiv:2405.04749, sampled weights (`regulariser_weight_selection.md:233`) | arXiv | **Doesn't support the claim** |
| Kammerer et al. 2020, A&A 644, A110 (`_closure.py:9`, `contributors.md:25`) | text §2.1–2.2 | ±1/3 correct; the "independent combinations" step is virgil's |
| Blakely et al. 2024, arXiv:2404.13032 (`models.py:1253`) | text, eqs 6–8 | Correct; now AJ 169, 137 (2025) |
| Soummer et al. 2007, arXiv:0711.0368, MFT (`_geometry.py:91`) | arXiv, Crossref | Correct |
| Espinosa Lara & Rieutord 2011, A&A 533, A43 (`_elr.py`) | Crossref; eqs 30, 32 re-derived | Correct |
| Ruffio et al. 2018, AJ 156, 196, eq. 8 (`limits.py:240`) | text | Correct |
| Absil et al. 2011, A&A 535, A68 (`limits.py`) | text §3.2 | Correct |
| Kluska et al. 2014, SPARCO, A&A 564, A80 (`spectra.py`) | arXiv | Correct |
| Hansen & O'Leary 1993, SIAM J. Sci. Comput. 14, 1487 (`imaging.py:677`) | Crossref | Correct |
| Hillen et al. 2016, A&A 588, L1 (`spectra.py:115`) | Crossref | Correct |
| Bishop 2006, eqs 3.91–3.95 | standard edition | Consistent; not independently checked |
| Desdoigts 2025, Charles 2025, Dholakia & Pope 2025 | Crossref | Correct; journal versions now exist (2026) |
| Ramani et al. 2012, IEEE TIP 21, 3659 | Crossref | Author Nielsen missing |
| EHT Sgr A* Paper III, arXiv:2311.09479 | arXiv | Give ApJL 930, L14 (2022) |
| The other 50 arXiv IDs (IFT, GRAVITY review, Beauty Contests, PMOIRED, …) | arXiv API | Correct |

**Not checked:** Lapeyrère 2014, Mérand 2022 (SPIE), De Prins et al. 2026,
Tiede 2022 (JOSS), Ramani, Blu & Unser 2008, and Blackburn 2020 (cited
second-hand).

## Plan versus progress

| Stage | Status | Notes |
| --- | --- | --- |
| 0 likelihood unification | done | Plan line 24 ("exactly von Mises, smooth at ±π") is false for 4+ telescopes since 6.0 |
| 1 `Image` + DFT | done | No `backend` argument any more |
| 2 NUFFT | removed, as recorded | Plan line 109 names a `virgil-astro[nufft]` extra, a rename artefact |
| 2b MFT | done | `amigo.simulated_disco_record` (plan 125, 128) no longer exists; it is now `coverage.ami_grid_record` |
| 3 `fit`, regularisers, `l_curve`, `diagnose` | done | The "backend oracle check" in `diagnose` is gone |
| 3c PDS 70 | silently dropped | Blocked by the DISCO miscalibration; should move to Deferred, with the GP prior as the recommendation |
| 4, 4b, 4c | done | Logged numbers predate 6.0 |
| 5a–5d | done | TSV evidence promised (plan 270), not built, no decision recorded; `gauss_newton_diagonal` still listed (plan 273) |
| 6.0 closure phases | done | Plan 382 and review §0.9 describe T diag(s) Tᵀ; the code uses D^½RD^½ (finding 1.4) |
| Apep library commits | merged (#124) | Plan 375 and 511 and the spectro note still say "local branch, merge after 6a" |
| 6a.0 times and frames | not started | Now urgent: blocker B3 |
| 6a.1 orbits | not started | Its table sums to 25–30 h, not the stated 23–28 h |
| 6a spectro-interferometry | not started | 10–13 h is unrealistic once the GRAVITY review's items are included |
| 6d, 6 (6b/6c), 8 | not started | — |
| 7 hardening and release | partial | Version already 0.2.0; no CHANGELOG; MFT caching (plan 260) is no longer in Stage 7's list |

**Claims in the logs that are no longer true:**

- **Numbers measured before 6.0.** These logs quote numbers measured before
  6.0 changed the closure-phase count (`nrm_oidata` now keeps 15 of 35):
  - plan 194–203 (NCC 0.93/0.89, SPARCO 0.509, the stress table);
  - 297–300 (5a);
  - 312–315 (5b);
  - 325 (composite).
- **Composite numbers have drifted** (plan 325 against the re-run docs):
  - companion 0.080 against 0.083;
  - Δlog Z 28 against 30.4.
- **Six MWEs** (`mwe_sam_v2_cp`, `mwe_vlti`, `mwe_stress_test`,
  `mwe_gaussian_field`, `mwe_evidence`, `mwe_elr_chara`) have not been
  re-executed since 6.0. Plan line 608 requires that they be.

**Estimates:**

- Plan 568 says "35–50 h overall", but the plan's own stage estimates sum to
  about 92–125 h.
- Stage 7 is 8–11 h in its heading and 3–4 h in the orchestration table.
- What remains, by the plan's own numbers, is about 68–91 h, before
  re-estimating 6a.
- The orchestration section (619–682) is historical.

**Contradictions between design notes:**

- `chromatic_sources.md` against the shipped `Tabulated` (blocker B5).
- The spectro note §2.4 (per-frame offsets, rank one) against its own §2.6
  and plan 6d (baseline-based, low rank).
- The reference-flux rule (S §4: value at `wavel0`) against the code (node
  mean).
- `regulariser_weight_selection.md`:
  - writes −χ² where it means −½χ² (line 171);
  - promises Lanczos where the code uses a dense Cholesky (183, 279);
  - still lists cross-validation, which was dropped (277).

**What should change in the plan:**

1. Add a **"0.2.0 release" stage now**: blockers B1–B5, the CHANGELOG,
   re-executing the six MWEs (or marking their numbers "pre-6.0"), and the
   text fixes above.
2. Replace "Order of work" with a status table, and list the open debts:
   - `Tabulated` → `Nodes`;
   - `inflated_errors(where=, combine=)`;
   - MFT caching;
   - the discrepancy target N − γ (γ now exists, `imaging.py:930`);
   - the TSV-evidence decision.
3. Split 6a into:
   - 6a-i, spectra: `Nodes`, `Sum`, floors;
   - 6a-ii, observables and the reader;
   - 6a-iii, instrument effects: smearing, primary beam.

   Budget it at about 20–25 h. Do 6a-i first, since it closes the
   `Tabulated` debt; 6a.1 (orbits) can then run in parallel with 6a-ii.
4. Give 6d an explicit design item for how its offsets compose with
   `ClosureNoise` grouping and the wrapped-chord path.
5. Move 3c to Deferred, with its blocker.
6. Give Stage 7's MEM preconditioning its own estimate, separate from
   release chores.
7. Archive the orchestration section and fix the rename artefacts: the path
   `/Users/benpope/code/virgil` (plan 52, 677) does not exist.

## Recommended next steps, in order

1. **Fix blockers B1–B4** in one PR, about half a day:
   - floors and extras;
   - `__version__`;
   - license metadata;
   - MANIFEST.in;
   - NumPy-held index arrays (B2).
2. **Add CI jobs:**
   - wheel and sdist build, install and test;
   - macOS py3.11;
   - `lowest-direct`;
   - a nightly notebook execution;
   - harmonix and jaxoplanet in an extra so their tests run.

   Do this before merging the publishing PR #128.
3. **Stage 6a.0's `INT_TIME` matching** (B3), with a test on a GRAVITY file
   that has a 30–90 s T3/VIS2 offset.
4. **Correctness fixes:**
   - the rim positivity check (1.1);
   - the LM float32 tolerance (1.3), and assert convergence in the test;
   - the V² → amplitude floor (1.5);
   - the von Mises normaliser (1.2).

   Each is small and comes with a regression test.
5. **Decide `Tabulated` (B5) and prune the API:**
   - `fisher_matrix` and `observed_information`;
   - `GaussianDiskModel`, `cvis_gaussian_disk` and `loglike_nosignal`;
   - `legacy/savefits.py`;
   - the dead plotting helpers;
   - `termcolor`.

   Export `pixel_offsets`.
6. **Write a CHANGELOG and fix the stale text:**
   - `MIGRATION_VIRGIL.md`;
   - the AGENTS.md table;
   - the plan's 6.0 description;
   - `chromatic_sources.md`;
   - the citation fixes in §2.

   Then tag 0.2.0.
7. **After the release, the performance work:**
   - QR inside LM for small problems;
   - a jitted `_residual_jacobian`;
   - Woodbury in `gauss_newton_mass`;
   - arrays for float hyperparameters;
   - the argsort split in `ClosureNoise`.

   That is 10–17× on the affected paths.
8. **Re-execute the six MWEs** and refresh the plan's numbers. Then
   restructure the plan as above.

## What is good and should be kept

- **One residual vector** (`whitened_residuals`) behind every likelihood,
  grid, limit, fit and evidence. It made Stage 6.0 a local change, and it
  keeps χ² consistent across tools.
- **Orientation discipline:**
  - `image_coordinates` as the single reference;
  - direct orientation regression tests;
  - the render-versus-model Fourier test covering binaries, UD, Image, rim,
    EllipticalGaussian, GaussianArc, FlaredDisk and GDS;
  - `_enforce_sky_orientation`.
- **Module-level jitted solvers and grid tools with traced templates.**
  Measured: an L-curve, a new σ, or new grid axes cause 0 recompiles.
- **Local float64 via `run_in`/`cast_tree`,** never global, and
  `Precision.HIGHEST` on every Fourier matmul. Its integer-array edge
  (B2) needs fixing, but the policy is right.
- **Correct, well-explained statistics docstrings.** `error_scale`,
  `classic_maxent`, `ruffio_upperlimit` and `nsigma` explain the maths and
  its assumptions in prose a student can follow.
- **The whitened `GaussianField`** (DCT, no dense covariance), and the
  block-diagonal `ClosureNoise.whiten`.
- **The explicit flux-versus-contrast convention,** and the validation of
  concrete inputs with traced-safe `is_physical`.
- **Honest design notes** that record decisions and their reasons. They
  should be kept current, not shortened.
