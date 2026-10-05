# virgil imaging: staged execution plan (revision 4)

## Context
We are adding regularised maximum-likelihood image reconstruction, and optional Bayesian sampling, to virgil.
- **Data:** AMIGO DISCO data from JWST AMI, closure phases from long-baseline interferometers (VLTI, CHARA), and aperture-masking data.
- **Composition:** analytic components (stars, companions) stay analytic, and the image is one more component of a `System`.
- **Build strategy:** incremental, with a **minimal working example (MWE) that does something scientifically useful at every stage**, and a feedback checkpoint on each before the next begins.
  - It starts by reproducing dorito's AMI DISCO results.
  - Then grey long-baseline imaging with closure phases.
  - Then Gaussian-process priors and sampling.
  - Then polychromatic imaging.
- **Priorities:** robustness, minimal technical debt, and readability for new PhD students and sceptical astronomers.

The design rationale was established in the earlier research: the gauge survey, IFT versus Gaussian processes, DFT against NUFFT accuracy, optimisers, and zodiax 0.6. Stage 0 writes it to `design/image_reconstruction.md`, and this plan does not repeat it.

## Standing design decisions
1. **Numerical precision.**
   - Every Fourier transform and matmul uses `precision=lax.Precision.HIGHEST`, which defeats TF32 on A100/H100 GPUs.
   - Fitting entry points (`fit`) take `dtype="float64"` by default and run inside a local `jax.enable_x64(True)` context, casting inputs. `dtype="float32"` is opt-in.
   - Forward-model code is dtype-agnostic and still passes the float32 test suite; x64 is never set globally.
   - `AGENTS.md` is updated to say that library code works in both precisions and that imaging entry points default to local x64.
2. **One likelihood.** `whitened_residuals(model, data)` returns:
   - (model − data)/σ, and Δ/σ for projected (kernel/DISCO) phases;
   - **2 sin(Δ/2)/σ** for unprojected closure phases, which is a von Mises likelihood and is smooth at ±π. (From Stage 6.0, for four or more telescopes this chord is applied to independent, whitened closure-phase combinations whose covariance is not diagonal, so the likelihood is no longer exactly von Mises there; three-telescope data are unchanged.)

   `model_loglike` is redefined on top of it, so grids, limits, `numpyro_model` and fitting all share it.
3. **One specification for fitting and sampling.** `fit(model, priors, data, regularisers)` and `numpyro_model(model, priors, data, regularisers)` take the same arguments:
   - Free parameters are exactly the keys of `priors`, a dict of NumPyro *distributions*; numpyro is used only as a distributions library.
   - Support-respecting transforms come from `biject_to`.
   - `fit` finds the MAP; `numpyro_model` is the posterior for samplers, and accepts only genuine prior regularisers (it rejects MEM, TV and TSV).
4. **Optimisers:** `fit(..., method=...)` with:
   - `"lm"`: optimistix Levenberg–Marquardt with a matrix-free, fixed-length `lx.Normal(lx.CG)` or `lx.LSMR` inner solve. It never materialises a Jacobian and is the default when residuals exist.
   - `"lbfgs"`: optax L-BFGS with a zoom line search (stopped on the gradient), for MEM, TV and free error inflation.
   - `"adam"`: optax, now a declared runtime dependency.
5. **Image parameterisation.**
   - `Image(log_brightness, pixel_scale_mas, support=None, flux, dra, ddec, rotation_deg=0)` is a `Component`, and `brightness = softmax(η, where=support)`.
   - `log_brightness` is either a plain array (free log-pixels, MAP-only) or an object with `evaluate(pixel_scale_mas=...)`, e.g. `GaussianField` in Stage 5. That is the zodiax 0.6 `Expression` protocol; migrate when it is released.
6. **Fourier backends.**
   - The separable DFT is the default and the oracle.
   - On uv lattices (AMI/AMIGO DISCOs) an `Image` whose `rotation_deg` matches the lattice uses the exact two-sided MFT (Stage 2b).
   - A NUFFT backend (jax-finufft) was built in Stage 2 and **removed** after Stage 3 (issue #75): on an A100 it cost a flat ~7 ms per call because jax-finufft re-plans every time (jax-finufft #157). Its code is in the git history of PR #71; revisit only when jax-finufft can reuse plans.
   - Padded FFT plus interpolation is never used.
7. **Rules for users:**
   - The default field of view is the smaller of 500 mas and the interferometric field of view, λ/B_min (`imaging.field_of_view`); pixels are finer than Nyquist (`imaging.nyquist_pixel_scale`).
   - Reconstructed images are shown with the beam (`imaging.beam`, the FWHM of the dirty beam's core) shaded in the lower left, and next to a residual map on a symmetric diverging scale: signed residuals for MAP images, z-scores whenever there are uncertainties.
   - Unresolved things are analytic; resolved emission goes in pixels.
   - Something must fix the origin: an analytic star, a `Centroid` prior, or a centred mean.
   - Initialise from a parametric fit.
8. **Dependencies:** optax (and lineax) become required. `blackjax` (`[sampling]`) is an optional extra. zodiax stays at `>=0.4`.

## Branching and workflow
- **Branch.** Create `imaging` in `~/code/drpangloss` (the local folder keeps its old name), branched from `chromatic-scenes`. It needs the chromatic work (`Spectrum`, `Resolved`, multi-channel `OIData`), and `chromatic-scenes` is 3 commits ahead of `main`. Once `chromatic-scenes` merges, rebase `imaging` onto `main`.
- **Each stage is a PR** from `imaging-sN-<name>` (git cannot hold both `imaging` and `imaging/…` branches), stacked on the previous stage's branch and retargeted to `imaging` as earlier stages merge. Each PR has:
  - the code;
  - its tests;
  - an MWE notebook (or a section of one) in `notebooks/`, synced to the docs by `scripts/sync_tutorial_docs.py`;
  - a short entry in `design/image_reconstruction.md`.
- **Merge `imaging` into `main`** at two milestones: after Stage 4 (the AMI and long-baseline MAP imaging are usable) and after Stage 6.
- **Feedback checkpoint** at the end of every stage: I show you the MWE figures and numbers, and the next stage starts only after your comments. That is where stage scope can change.
- **Every stage must pass:** `uv run pytest` (float32 and x64 paths), `ruff check` (0.11.0), the tutorial sync test, and the docs build.

**Time estimates** are Claude-agent working time (writing code, tests and MWEs, running them, iterating). They exclude waiting for your feedback, data transfers, and OzSTAR GPU queue time. They assume no major surprises; the flagged risks say where the surprises could come from.

---

## Stage 0: groundwork and likelihood unification (about 2–3 h)
**Build:**
- The `imaging` branch.
- `design/image_reconstruction.md`: the decisions above, the research summary, and a deferred list with triggers.
- `AGENTS.md`: the precision policy; new modules (`fitting.py`, `imaging.py`, `fields.py`) and the import DAG.
- `likelihood.whitened_residuals`, and `model_loglike` redefined through it (Gaussian-limit normaliser).

**Tests:**
- The new phase term equals the old for small Δ and is smooth across ±π.
- **Regression:** existing grid, Absil and Ruffio outputs and the tutorial fits on the in-repo data are unchanged within tolerance; the PR lists any shifts.

**MWE ("nothing broke, one thing got better"):** re-run the existing binary-search and contrast-limit tutorials with identical results. Add a two-panel figure: the old and new closure-phase χ² along a companion-position slice through a phase-wrap region, showing the kink gone.

**Checkpoint:** do you accept the likelihood change?

## Stage 1: `Image` with the exact DFT; forward simulation of AMI DISCOs (about 3–4 h)
**Build:**
- `_geometry.image_visibilities(brightness, uu, vv, pixel_scale_mas)` (it had a `backend="dft"` argument until the NUFFT was removed): separable, HIGHEST precision, frequencies in cycles/mas.
- `models.Image`: array `log_brightness`, circular `support`, `Image.from_model(model, npix, pixel_scale_mas)` built through `render()`, the `brightness` property, and `_centred_image` (exact native pixels, bilinear resampling for display).
- Exports.

**Tests:**
- DFT against a float64 reference, and against the `render()`-FT pattern.
- A delta pixel equals `PointSource`; a pixelised Gaussian equals `GaussianDisk`; `V(0) = 1`; masked pixels are exactly 0.
- The orientation test required by AGENTS.md.
- Finite gradients for every observable kind (V², amplitude, log-amplitude, CP, kernel `phi_mat`, `mixed_log_complex`).
- `System` mixing; float32 and x64 at extreme baselines.

**MWE ("simulate dorito-style data"):**
- Take the in-repo AMIGO record (`data/calibrated_visibility.npy`: real AMI uv coverage and DISCO operators).
- Put a WR 137-like dusty spiral image (a parametric spiral rendered to pixels) next to an analytic star.
- Compute the DISCO observables, add noise with `OIData.with_model`, and plot data against model.
- Show that the image evaluated at a single pixel reproduces a `PointSource` binary's DISCOs exactly.

**Checkpoint:** do the image API and the sky conventions read naturally?

## Stage 2: NUFFT backend and benchmark (about 3–4 h, plus your GPU run)
**Status:** done (PR #71), including the OzSTAR A100 run. Outcome: the NUFFT is accurate but too slow on GPUs (issue #75), and was removed after Stage 3; the DFT and MFT are the supported paths.

**Build:**
- `backend="nufft"` in `image_visibilities`, via jax-finufft `nufft2`:
  - `iflag=+1`; the image rows pair with v; even N gets the half-pixel phase factor;
  - `eps` defaults to 1e-7 under x64, and must be ≥ 1e-5 under float32 (below that it raises).
- A `nufft` extra with a lazy import and a clear error if it is missing (removed with the backend; there is no such extra now).
- `scripts/bench_ft.py` (DFT against jax-finufft, value+grad, npix 64–512, M 10³–10⁵, both dtypes).

**Tests** (skipped without the extra): agreement with the DFT per point, |ΔV| ≤ 3·eps·V(0), including phases on low-|V| baselines; odd and even N; orientation; `check_grads`; vmap; inside `System` with CP and DISCO data. A CI job with the extra installed.

**MWE ("which transform, when"):**
- The Stage 1 simulation evaluated with both backends, with an accuracy table of max |ΔV| and phase error on low-|V| baselines, in float32 and float64.
- Laptop timing curves.
- Slurm commands for you to run the same benchmark on an OzSTAR GPU, including a DFT accuracy assertion that confirms HIGHEST beats TF32.
- A documented rule of thumb for the crossover.

**Risk:** jax-finufft's use of private jax internals and the fact that it re-plans every call (#157). The DFT fallback makes this non-blocking.

**Checkpoint:** do you accept the default-backend rule?

## Stage 2b: exact MFT on uv lattices (added after Stage 2)
**Status:** done (PR #72). AMIGO DISCO data lie exactly on a detector-frame uv lattice rotated by the parallactic angle. `OIData.uv_grid` records it, `SourceModel.model_on_grid` defaults to `model`, and `Image(rotation_deg=...)` matching the lattice uses the two-sided MFT (Soummer et al. 2007): exact, and 3–8× faster on the ν Hor coverage. `coverage.ami_grid_record` (then called `amigo.simulated_disco_record`) gives a small AMI-like record with diagonal errors, which replaces large fixtures in tests and tutorials; `disco_covariance` is optional.

## Stage 3: `fit`, regularisers; dorito-style AMI imaging on simulated truth (about 6–9 h)
**Status:** implemented on `imaging-s3-fitting` (see the design note's Stage 3 log). Differences from the plan below: L-BFGS uses optax (gradient stopping) rather than optimistix; `l_curve` has `corner` and `discrepancy` criteria; `diagnose` has no field-of-view check (on a uv lattice the shortest spacing bounds the useful field from above); MWE-B uses `coverage.ami_grid_record` (then `amigo.simulated_disco_record`) with ν Hor-like errors rather than the real ν Hor operators (large files); the dorito recipe (Adam then BFGS, MEM weight 1e5) is not reproduced literally, since MEM with L-BFGS at an L-curve weight does the same job.
**Build:**
- `fitting.py`: `fit` with `lm`, `lbfgs` and `adam` (§3–4), with loss scaling, unscaled χ² reporting, and `info` (converged flag, steps, χ² per block).
- A small private helper, `_precision.py`: a `run_in(dtype)` context and a `cast_tree(tree, dtype)` function used by the entry points (moved here from Stage 0, where nothing would use it yet).
- `imaging.py`:
  - regularisers `MaxEntropy(prior=None)`, `TSV`, `TV` (ε-smoothed) and `Centroid(sigma_mas)`. Each has `value`, an optional `residuals`, and a `probabilistic` flag.
  - `image_priors(scene)`;
  - `nyquist_pixel_scale(data)`;
  - `diagnose(model, data)`: χ²_red per block, Nyquist, edge flux, centroid and anchor, flip Δχ², the **projected-phase regime check** (|arg V| > 0.8π or |V| < 0.05, where unwrapped DISCO/kernel phases become unreliable). (The "backend oracle check" was dropped with the NUFFT backend, and `image_visibilities` has no `backend` argument now.)

**Tests:**
- `fit(lm)` recovers a synthetic binary, consistent with the grid fit.
- LM and L-BFGS agree on a TSV problem.
- Gauge: rolling the image leaves the CP loss invariant; a `Centroid` fit stays centred; a star-anchored offset is recovered.
- Positivity transforms work.
- `numpyro_model` rejects MEM, TV and TSV.
- Every `diagnose` warning fires on a constructed bad case.
- float32 and x64 paths agree to tolerance.

**MWE-A ("recover a known image from AMI DISCOs"):** fit the Stage 1 synthetic WR 137-like data. Compare MEM (L-BFGS), TSV (LM) and TV (L-BFGS); include a weight-sweep L-curve loop, the `diagnose` report, and recovery metrics against the truth after centroid alignment.

**MWE-B ("dorito-style AMI imaging against ground truth"):** simulate, rather than use real WR 137 or NGC 1068 data, because real data have no ground truth to test against.
- **Coverage and noise:** reuse the real ν Hor AMI baselines and DISCO operators from the in-repo AMIGO record (`data/calibrated_visibility.npy`, `NuHor_F480M.oifits`). Set per-observable noise to the ν Hor measured σ, scaled to a few SNR levels.
- **Truth scenes:**
  - (i) a WR 137-like dusty spiral;
  - (ii) an inclined ring or disk;
  - (iii) star + faint companion + extended emission.
- **Fit:** dorito's recipe (MaxEntropy plus a centroid prior, grid and weights as in `wr137_disco.py`), alongside TSV/LM and TV.
- **Report:** recovery metrics against the truth (normalised cross-correlation after centroid alignment, flux fraction, and companion astrometry where present) as a function of SNR, plus the regime check.

**Risk:** the regularisation weight depends on the scene. The SNR sweep shows where each regulariser breaks down.

**Checkpoint:** is the fitting API right, and are the recovery metrics good enough? **Merge milestone candidate.**

## Stage 4: grey long-baseline imaging with closure phases (about 4–6 h)
**Build:**
- Only glue and documentation should be needed; this stage tests generality.
- `Image` inside `System` with an analytic star (SPARCO-style, grey, then a `PowerLaw` spectral index for star and envelope).
- Multi-file joint fits (a list of `OIData`), the support-hole option under the star, and `from_model` initialisation from a parametric fit.
- The ν Hor MATISSE coverage fixture and the synthetic-coverage generator `tests/_coverage.py` (see "Coverage fixtures" below; moved here from Stage 0).

**Tests:**
- Joint multi-file fits.
- Recovery of a star + inclined disk + companion on synthetic VLTI-like coverage (4 telescopes, many channels).
- The star-anchored gauge (no centroid prior needed).
- The support hole prevents flux piling up under the star.

**MWE-A ("grey VLTI imaging"):** synthetic star + ring + companion on **real MATISSE ν Hor uv coverage**, with noise copied from the per-point σ of the nuHor OIFITS. The workflow is parametric fit, then image with TSV/MEM, then `diagnose`.

**MWE-B ("SNR and coverage stress test"):** the same simulated scenes with ν Hor-derived noise scaled up and down. Drop telescopes or epochs to thin the uv coverage, and report where reconstructions fail and what `diagnose` flags. There is no real-data imaging in this plan.

**Risk:** bandwidth smearing in MATISSE LOW-resolution data is not modelled (an existing limitation), so the field of view has to be kept ≲ Rλ/B. The MWE documents this.

**Checkpoint:** is closure-phase imaging robust enough? **Merge `imaging` into `main`** (milestone 1: MAP imaging for AMI and long-baseline data).

**Log (2026-10-01):** the numbers below were measured before Stage 6.0 changed the closure-phase count (`nrm_oidata` keeps 15 of its 35 closure phases now), and have not been re-measured ("pre-6.0").
- No new fitting code was needed: `fit` and `l_curve` already take a list of datasets, and the worst-fitted one sets the discrepancy weight.
- Coverage is synthetic. `coverage.vlti_oidata` gives four-UT Earth-rotation tracks over many channels, and `coverage.nrm_oidata` gives the 21 V² and 35 closure phases of the NIRISS mask. This is the fallback in "Coverage fixtures", so no ν Hor fixture is committed. Stage 4c uses real PIONIER data instead.
- `models.circular_support(..., inner_radius_mas)` makes the support hole, and `starting_image(..., hole_mas=...)` applies it. Both MWEs use it, at half the beam's minor axis.
- Fitting the envelope flux (a prior on `"env.flux"`) turned out to be what removes the spurious spot next to the star. With the flux fixed at `starting_image`'s estimate, the excess piles up at the centre; with the flux fitted, NCC rises from 0.86 to 0.93 in the SAM MWE. A hole of half a beam under the star (`starting_image(..., hole_mas=...)`) then clears the remaining core flux, which is degenerate with the star's. In the VLTI MWE it removes the bias in the SPARCO ratio, which goes from 0.545 to 0.509 against a truth of 0.5.
- MWEs:
  - `mwe_sam_v2_cp`: two NRM rolls, compared with AMIGO-style modes (NCC 0.93 against 0.98).
  - `mwe_vlti`: two nights, eleven channels, with a SPARCO `PowerLaw` ratio and index fitted together with the pixels. Pre-6.0: NCC 0.89, with ratio 0.509 against 0.5 and index 1.87 against 2.
  - **Re-run post-6.0** (2026-10-04, OzSTAR job `virgil_notebooks`, virgil `ad131bd`): `mwe_vlti` ratio 0.506 and index 1.89, χ² per point 0.99 and 0.98; `mwe_sam_v2_cp` extended flux 0.0307 against 0.03, χ² per point 0.45 and 0.72 (roll 1 over-fitted at the discrepancy weight, which `diagnose` flags). Their NCCs are in the figure titles.
- The field is limited to λ/B_min, so VLTI scenes are only 2–3 beams across.
- **MWE-B** (`mwe_stress_test`): the VLTI scene at three coverages × five noise levels (0.5–8× the default errors), with one noise draw per coverage scaled across the levels. Each case is reconstructed with a hole and MEM at the discrepancy weight.
  - Pre-6.0: full coverage: NCC 0.86 at 0.5× and 1×, 0.77 at 4×, and 0.64 at 8×.
  - Two hour angles, or three UTs: NCC about 0.6 even at low noise.
  - χ² per point reached one in every case, so it says nothing about fidelity.
  - `diagnose`'s edge-flux warning flags noise-driven spreading to the edge of the field, but not coverage-driven failures (three UTs at 0.5×: NCC 0.62, no warning).
  - **Re-run post-6.0** (2026-10-04): full coverage NCC 0.84, 0.81, 0.80, 0.73 and 0.56 from 0.5× to 8×; two hour angles 0.68 to 0.35; three UTs 0.62 to 0.21; χ² per point 0.95–1.05 throughout. The pattern is unchanged, and the three-UT case at 0.5× still gets no warning.

## Stage 4b: a rotating scene over two epochs
A spiral that turns by 60° between two epochs, fitted jointly with the rotation known and then unknown. For an unknown rotation:
- the angle must be traceable;
- the likelihood can be multimodal in angle, so a coarse scan comes before a joint refinement;
- an Archimedean spiral has a near-degeneracy between rotation and expansion.

**Log (2026-10-01):**
- `fit` (and so `l_curve`) accepts a model function that returns one model per dataset, sharing parameters; regularisers act on the first.
- `models.Rotated(source, rotation_deg)` wraps any model with a traceable angle. It rotates the uv coordinates, so the MFT is not used for that epoch.
- In `mwe_rotating_epochs`, two AMI epochs with the known 60° give NCC 0.92, against 0.90 for one epoch at the same weight.
- With the angle unknown, a 15° scan of fixed-angle fits (warm-started, with a 300-step limit) has a single sharp minimum at 60°. Reconstructing the image again with the angle free (an L-curve started from the scan's best angle) gives 59.89° and the same NCC, 0.92, as with the angle known.
- No secondary minimum appeared for this spiral: its fading ends break the rotation–expansion degeneracy.

## Stage 4c: real PIONIER data (SPARCO), and the FlaredDisk merge
Real-data notebooks live in `nuHor/notebooks/`, next to the data, and are not committed here:
- `pionier_iras08544`: one star;
- `pionier_binary`: Hillen et al.'s model;
- `pionier_iwcar`, for Toon.

They compare results with the papers' text, not their images.

**Library additions:**
- `spectra.BlackBody`, a Planck F_λ spectrum normalised at `wavel0` with a fittable `temperature`.
- SPARCO verification tests: the published mixing formula with blackbody components, per-channel weighting of multichannel data, temperature gradients, and the Rayleigh–Jeans limit.
- PR #80's `FlaredDisk` components (Blakely et al. 2024), merged into this stack. Edge-on inclinations are now rejected, and g < 0 is documented.

**IRAS 08544-4431** (Hillen et al. 2016; 27 files, 828 V² and 504 closure phases):
- **Parametric model:** a 7250 K primary; a secondary, a ring with m = 1, 2 modulations and a resolved background, all with blackbody spectra; and the ring anchored to the binary's centre of mass (κ = q/(1+q) = 0.75). It converges to one solution with χ² per point 2.50.
  - Fractions: 57.0 / 6.1 / 20.7 / 16.2%, against the paper's 59.7 / 3.9 / 20.9 / 15.5%.
  - Temperatures: secondary 3430 K, ring 1098 K, background 2620 K, against 4000, 1120 and 2400 K.
  - Binary separation 0.72 mas (paper 0.81 mas).
  - Ring 14.33 mas across, FWHM 3.03 mas, i = 20.1° (paper 14.15 mas, 3.2 mas, 19°).
- **Images:** a λ⁻⁴ primary and a power-law environment, as in the paper. With the primary alone, the primary has 61.4% (paper 61%). With the binary subtracted, the knot next to the primary is gone.
- **Open:** the companion's PA is 219°, against the paper's 56°, though the ring's bright side agrees. To raise with Toon.

**IW Car** (De Prins et al. 2026; the 23 files within the paper's 81-day window):
- No single-ring model fits, as the paper found.
- The parametric secondary lands at (1.06, −1.69) mas with 2.3% of the flux (paper: (1.12, −1.90) mas, about 2%), with the primary at 62% (paper 63.6%).
- The SPARCO image recovers the inner arcs at about 5 mas.

**SPARCO convention (for Toon):** a spectral index of the environment is only defined relative to the star's assumed spectrum, since the ratio goes as λ^(d_env − d_star). Both apparent mismatches with the papers came from this, not from the code:
- IRAS 08544: a blackbody primary in place of λ⁻⁴ shifts d_env by about 0.7.
- IW Car: the paper fixes d_prim = −3.17, so their d_rim of 0.89 is 4.06 relative to the star, against our 4.0 with a λ⁻⁴ star.

A background split off as a `Resolved` component also changes what the image's index means, and a smooth image is degenerate with a resolved background.

## Status after Stage 4, and next steps
**Done:**
- AMI and long-baseline MAP imaging: per-dataset models (`Rotated` epochs), SPARCO with power-law and blackbody spectra, the support hole, and synthetic VLTI and NRM coverage.
- Synthetic MWEs: SAM V² + CP, two-night VLTI, and rotating epochs.

**Next:**
1. Merge this stack into `imaging`, then `imaging` into `main` (milestone 1).
2. Stage 5: Gaussian-process pixels, sampling (posteriors for the companion's flux and separation, which are correlated), and evidence-based weights.
3. Stage 3c: PDS 70, now deferred (see Deferred).
4. Cache MFT matrices (Stage 7).

## Stage 5: Gaussian-process pixels and sampling (about 6–8 h)
**Build:**
- `fields.GaussianField(latent, sigma, length_mas, order=2, mean=None, mean_floor=1e-3)`:
  - η = log(μ/max μ + ε) + IDCT₂[√S ⊙ latent] (orthonormal);
  - S ∝ (κ² + λ_jk)^(−order) on reflecting-boundary Laplacian eigenvalues, with S₀₀ = 0, scaled to a mean variance of σ².
- `image_priors` gives N(0,1) latents.
- `fit` warns if you MAP the hyperparameters.
- **Regularisation weights from the evidence**, building the matrix-free curvature machinery once:
  - for quadratic and GP priors (TSV, `GaussianField`), the Laplace-approximated evidence as a function of the weight (or σ, length), with the log-determinant of the Gauss–Newton Hessian (planned by stochastic Lanczos quadrature or Hutchinson probes; built as a dense Cholesky factorisation, see the 5b log); maximise it, or marginalise when sampling;
  - for maximum entropy, Gull & Skilling's "classic MaxEnt" choice of the weight (−2αS equal to the number of well-measured directions, from the eigenvalues of the same curvature);
  - both as helpers alongside `LCurve.corner` / `discrepancy`, returning a weight; `fit` never chooses it silently.
- `imaging.gauss_newton_diagonal` (a Gauss–Newton–Bartlett estimate, for mass-matrix initialisation; not built: Stage 5d's `fitting.gauss_newton_mass` took its place).
- The `[sampling]` extra (blackjax).
- There is no sampler wrapper: tutorials sample `numpyro_model` with numpyro's NUTS, and show BlackJAX on the potential from `numpyro.infer.util.initialize_model` as the alternative.

**Tests:**
- Exact 8×8 covariance against the inverse of (κ²I + L)^order; `order=1` equals TSV + L2 on η.
- Prior σ and length calibration.
- `latent = 0` reproduces the template, and template-parameter gradients are finite.
- 16² NUTS coverage of the injected hyperparameters, under x64.

**MWE-A:** a SPARCO star + GP envelope with a centred `GaussianDisk` mean, on the Stage 4 synthetic data. Compare the MAP (LM) with TSV/MEM from Stage 4.

**MWE-B:** posterior sampling at 32² (BlackJAX NUTS and MCLMC/MAMS), with marginalised σ and length. Show the posterior mean and standard-deviation maps, hyperparameter posteriors, and chains against wall-clock time on laptop CPU. Optionally repeat on the simulated AMI scene.

**Risk:** sampling cost and multimodality at larger grids. Scope stays at 32–64², with the DFT/MFT (on an A100 the DFT is ~1 ms at 256² × 10⁴ points, so GPU sampling does not need the NUFFT).

**Checkpoint:** is the GP prior worth it compared with classical regularisers, and does sampling go into the docs as supported?

**Log, 5a (2026-10-01):** this stage is split into three PRs: 5a the GP field, 5b evidence-based weights, and 5c sampling.
- **Library:** `fields.GaussianField(latent, sigma, length_mas, order=2, mean=None, mean_floor=1e-3)` and `fields.field_spectrum`.
  - `Image` accepts the field in place of its log-brightness, through the `evaluate(pixel_scale_mas)` protocol, and gains an `eta` property.
  - `image_priors` gives the latents N(0, 1) priors, so the fit is least squares.
  - `fit` warns when σ or ℓ is fitted by MAP.
- **Tests:** the exact covariance against (κ²I + L)^order with the constant mode removed (8×6, orders 1 and 2); `order=1` equals TSV + L2 on η; σ and ℓ calibration; `latent = 0` reproduces the template; finite hyperparameter gradients; an LM fit and `diagnose`; the MAP warning.
- **MWE-A** (`mwe_gaussian_field`), on the two-night VLTI SPARCO scene (numbers pre-6.0):
  - Levenberg–Marquardt converges in 23–53 steps, a few seconds per fit.
  - At the discrepancy pair (σ = 4, ℓ = 1 mas): NCC 0.93, ratio 0.511 and index 1.91, against MEM's 0.89, 0.509 and 1.87 (truth 0.5 and 2).
  - The result is insensitive to σ between 1 and 4.
  - **Re-run post-6.0** (2026-10-04): the discrepancy pair is now σ = 4, ℓ = 2 mas. GP: NCC 0.939, ratio 0.503 and index 1.92; MEM: 0.852, 0.506 and 1.89. The GP's lead over MEM grew (0.09 against 0.04).

**Log, 5b (2026-10-01; numbers pre-6.0):**
- **Library:**
  - `imaging.log_evidence(model, data, path)`: the Laplace evidence of a `GaussianField` MAP, −½χ² − ½|z|² − ½ log det(I + JᵀJ). The determinant is computed exactly, from a Cholesky factor in the smaller of the data and latent dimensions; other parameters are held at their MAP values.
  - `LCurve.classic_maxent(data, path)`: Gull and Skilling's weight, where −2wS = Σ λ/(λ + w), with λ the Gauss–Newton curvature in the entropy metric. It is interpolated in log w across the sweep, and gives `None` if the sweep doesn't bracket it.
  - Both share `_residual_jacobian`, a dense Jacobian in float64, using forward or reverse mode, whichever is cheaper.
  - No stochastic log-determinant yet: dense is fine up to about 10⁴ data or pixels.
- **Tests:**
  - On data drawn from a GP with σ = 1.5, the evidence prefers σ = 1.5 over 0.3 and 6.
  - The evidence needs a field.
  - Classic MaxEnt lies inside a bracketing sweep, and gives `None` otherwise.
- **MWE** (`mwe_evidence`), on the VLTI scene:
  - Evidence over σ and ℓ: it picks σ = 2, ℓ = 0.5 mas, which has the best NCC of the 4×5 grid (0.94). Each evaluation takes about 0.2 s.
  - Classic MaxEnt picks w = 15.7, NCC 0.91. The discrepancy principle picks w = 32.5, NCC 0.88, and the corner w = 100.
  - All fits converge: the sweep runs w ≥ 3, and refits are allowed 2 × 10⁵ L-BFGS steps.

**Log, 5c (2026-10-01; synthetic-data numbers pre-6.0, and the composite ones have since drifted: companion 0.083 and Δlog Z 30.4 on re-running the docs):** this PR covers the GP prior in practice, sampling basics and the imaging tutorials.
- **Library:**
  - `imaging.error_scale(model, data)` is MacKay's noise re-estimate, s = √(χ²/(N − γ)) with γ = Σ λ/(1 + λ), and `OIData.with_error_scale(s)` rescales a dataset. The docstring gives the derivation and references (MacKay 1992; Bishop 2006 §3.5; Gull 1989). It was motivated by PIONIER's conservative errors, χ²/N ≈ 0.5–0.8.
  - `dirty_image(flux_ratio=...)` now removes the star by subtracting the best-fitting point source. Subtracting a fixed one amplified the DISCO normalisation error by 1/f and left a residual star at the centre.
  - The `sampling` extra (blackjax), and a 16² NUTS test that the injected σ lies in its 90% interval (x64 only). An eight-seed check gave 68% coverage in 6/8 runs and 90% in 8/8.
- **Docs tutorials** on simulated data (Imaging parts 2–4):
  - **`imaging_rml`:** the dirty image, moments against dirty starts, the L-curve, the discrepancy principle and the corner, and `diagnose`. Both starts reach NCC 0.98.
  - **`imaging_gp`:** prior draws, an LM MAP fit, the evidence over σ and ℓ gridded in units of the beam, the `error_scale` check (s = 1.04 with honest errors, 0.52 with errors overstated twofold, and 0.99 after rescaling), and a comparison with MEM.
  - **`imaging_composite`:** a ring around a binary. With one star, the image absorbs the companion as a knot (NCC 0.22). The knot locates it, and the binary fit recovers it (0.080 at (1.47, −0.99) mas, against a truth of 0.080 at (1.5, −1.0)) and the ring (NCC 0.96). The evidence prefers the binary by Δlog Z ≈ 28, although χ² differs by only 3.
- **Real data** (nuHor, run on OzSTAR A100s in about 6 min each, against more than 30 min on the laptop):
  - IRAS 08544: GP d_env 0.45 and MEM 0.45, against the paper's 0.42 (0.37 for the GP before the errors were rescaled).
  - IW Car: GP d_env 0.82 and MEM 0.88, against the paper's d_rim 0.89.
  - The notebooks now rescale the errors first, grid ℓ in units of the beam, and run converged sweeps.
- **Not done:** a sampling MWE at 32–64², which is Stage 5d.

**Checkpoint (2026-10-02):**
- **The GP prior is worth it, and it becomes the recommended prior.** The reasons:
  - The evidence chooses σ and ℓ with no L-curve.
  - `error_scale` calibrates PIONIER's conservative errors.
  - LM fits converge in tens of steps, where MEM sweeps need up to 2 × 10⁵ L-BFGS steps.
  - On real data the GP matches or beats MEM: on IRAS 08544 it is the only image that shows the compact emission near the binary.
  - MEM and TSV stay supported, as the classical comparison.
- **Sampling large images needs a mass matrix.** This reverses an earlier version of this checkpoint, which was based on a low signal-to-noise VLTI scene. On 150 VLTI points, plain NUTS took 127 leapfrog steps per draw at 32²–64². On part 3's 588 AMI points at 62², it saturated the tree depth (1023), as at 41² before. Stage 5d adds the helper, in its dense form (see the 5d log).

**Log, 5d (2026-10-03):** sampling, with a dense Gauss–Newton mass matrix.
- **Study** (OzSTAR job 17937020, A100, part 3's 62² AMI ring, 300 warmup steps and 300 draws):

  | Mass matrix | σ, ℓ | Flux | Steps per draw | Sampling time | Pixel ESS (5th–50th percentile) |
  |---|---|---|---|---|---|
  | Identity | sampled | sampled | 1023 | 243 s | 146–269 (σ 75, ℓ 96) |
  | Identity | fixed | fixed | 1023 | 194 s | 238–427 |
  | Diagonal | fixed | fixed | 1023 | 172 s | 346–666 |
  | **Dense Gauss–Newton** | **fixed** | **fixed** | **63** | **33 s** | **195–263** |
  | Dense, flux outside the block | fixed | sampled | 1023 | 359 s | 149–212 |
  | Dense, adapted in warmup | fixed | fixed | 1023 | 307 s | 3–6 |

  The dense matrix whitens the posterior, but only if every sampled parameter is in its block (the flux is measured to about 1%) and warmup adaptation is off. With σ and ℓ sampled it doesn't fit, because the curvature depends on them.
- **Library:** `fitting.gauss_newton_mass(model, priors, data, values)` returns NUTS's arguments: the inverse of JᵀJ (data and prior residuals) in numpyro's unconstrained coordinates, as one dense block over all sites, with adaptation off. It raises on a singular curvature. `gauss_newton_diagonal` is not added, since the diagonal didn't help.
- **Tests:** the block's layout, symmetry and prior bound; rejection of an unconstrained parameter; and on a 16² scene, the matrix whitening the exact Hessian of numpyro's potential at the MAP: eigenvalues 0.7–2.0, against a raw condition number of about 10⁴ (x64). A test that counted NUTS steps per draw was brittle: 255 → 31 on macOS, but 127 → 63 on Linux CI.
- **MWE** (`imaging_sampling`, Imaging part 5):
  - NUTS at the evidence's σ and ℓ, with the flux sampled.
  - 63 steps per draw, with no divergences; 7 min on the laptop.
  - The flux is 0.0503, with a 90% interval of 0.0500–0.0507 (truth 0.05).
  - Shows the posterior mean, the standard deviation and z-score residuals. The z-scores are within ±1, apart from a few at about −2.5.
  - Sampling σ and ℓ is described, with the GPU timing, but not run.

## Order of work after Stage 5 (updated 2026-10-03)
These stages draw on three notes:
- [`pmoired_parity.md`](pmoired_parity.md), which compares virgil with PMOIRED;
- [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md) (S);
- [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md) (O).

The last two came out of fitting GRAVITY data on Apep, but are written as general capabilities.

0. **6.0: urgent reader and likelihood fixes** (PR #120; from the GRAVITY review). These come first because GRAVITY and PIONIER are in active use.
1. **6a.0: times and frames in `OIData`** (S §2.5). A small PR, which unblocks the orbits, VISPHI and Stage 6d.
2. **6a.1: orbits and binary-frame scenes.** Decided 2026-10-03: they are needed now, for the Apep analysis.
3. **6a: spectro-interferometry.**
4. **The Apep agent's library commits.** These were merged in PR #124 (they sat on the local branch `apep-gravity`): `EllipticalGaussian`, `Tabulated`, `noise=`, the anisotropic `GaussianField`, `GaussianArc` and the rim-gradient fix. `Tabulated` is provisional and private, and 6a's `Nodes` and error floors replace it (decided).
5. **6d: correlated channel nuisances,** then the wavelength-scale nuisance. (The dual-field recipe is done, in virgil-vlti.)
6. **6b and 6c** (joint multi-filter AMI; `ImageCube`), then **milestone 2**.
7. **Stage 7** (hardening and release), then **Stage 8** (the rest of the PMOIRED parity).

**Binary-detection parity (from validation).** Two features that CANDID and fouriever have for finding companions are being implemented, each in its own PR:
- injection-method detection limits (CANDID): in progress;
- a fast linearised flux map (fouriever's `lincmap`): in progress.

**Log (2026-10-05):** the additions to Stages 6a, 6e, 7 and 8 and the note above come from feature suggestions made in the virgil-validation session, which compared virgil with eht-imaging, CANDID, fouriever, MPoL and PMOIRED. Ben approved them the same day.

## Stage 6.0: urgent reader and likelihood fixes (done; PR #120)
From the GRAVITY calibration review ([`gravity_calibration_review.md`](https://github.com/benjaminpope/virgil-vlti/blob/main/design/gravity_calibration_review.md), now in virgil-vlti), footguns 1, 2, 3 and 9. These affected every existing analysis of GRAVITY and of PIONIER (or any four-telescope) data.
- **Only independent closure phases, with their covariance diagonalised first.** Triangles that share baselines (one frame and channel) are grouped. Their covariance is modelled as C = D^½ R D^½ with R = T Tᵀ/3 (Kammerer et al. 2020's equal-noise approximation: the reported σ on the diagonal, and correlations of ±1/3 between triangles that share a baseline), not T diag(s) Tᵀ from per-baseline errors, as an earlier draft of this plan had it. The likelihood whitens the independent combinations (`OIData.cp_noise`, `n_independent`). Three-telescope data are unchanged.
- **`read_oifits(insname=...)`.** GRAVITY FT and SC tables are never merged silently, and FT data are skipped.
- **`PHITYP`.** A differential VISPHI is never read as an absolute phase.
- **No silent diagonal truncation.** A linear operator whose outputs correlate is rotated onto independent outputs.
- **Follow-up:** re-run the PIONIER (nuHor, Toon) and Apep GRAVITY analyses, and re-execute the docs tutorials before the release.

## virgil-vlti: the instrument project (separate)
Decided 2026-10-03: serious GRAVITY-specific development is its own project. On 2026-10-04 it became the repository [virgil-vlti](https://github.com/benjaminpope/virgil-vlti), widened to PIONIER and MATISSE, and the GRAVITY review, the calibrator-PCA plan and the questions for the GRAVITY team moved there. Its [`PLAN.md`](https://github.com/benjaminpope/virgil-vlti/blob/main/PLAN.md) sets the boundary with virgil. It covers:
- empirical covariances from the per-DIT products;
- calibration from pipeline products, including the dual-field (in-field) calibrator recipe, generalised from Apep (done there);
- the archival-calibrator PCA of systematics ([`gravity_calibrator_pca.md`](https://github.com/benjaminpope/virgil-vlti/blob/main/design/gravity_calibrator_pca.md));
- the GRAVITY-specific parts of fibre injection;
- telluric and wavelength calibration;
- PIONIER error models, and a MATISSE review.

The core keeps generic interfaces: a primary beam (6a), low-rank nuisance modes (6d), and an `insname` selector. It never imports virgil-vlti. Questions for the instrument team are in [`questions_for_gravity_team.md`](https://github.com/benjaminpope/virgil-vlti/blob/main/design/questions_for_gravity_team.md).

## Stage 6a.0: times and frames in `OIData` (done; branch `stage6a0-times`)
From S §2.5. It comes first because the orbits (6a.1), VISPHI (6a) and the per-frame nuisances (6d) all need it.
- **Per-sample times and frames.** `OIData` gains `mjd` and `frame` per sample.
- **Triangle matching.** Closure triangles are matched to baselines by frame (`INT_TIME`), not by nearest MJD.
- **`epochs()`** splits a dataset by night.

**Tests:**
- Multi-file round trips keep `mjd` and `frame`.
- Triangles match correctly on files whose MJDs differ slightly between OI_VIS2 and OI_T3.

**Log:**
- Triangle matching within an exposure (twice the longest `INT_TIME`) landed first, in #130, for the GRAVITY reader blocker.
- Frames are the baselines that closure phases tie together (a union-find in the reader's baseline lookup), plus rows of one `INSNAME` at the same MJD. On all 136 per-instrument reads of the Apep GRAVITY archive, every file gets exactly one frame per exposure.
- `OIData` stores `t_ref` (static float64) and `dt` (days, a leaf) rather than raw MJD, per S §2.5 item 4; `frame` is int32, the same in both x64 modes.
- `split_by_epoch` slices the stored observables and rebuilds `ClosureNoise`, so the parts' log-likelihoods add up to the whole (tested). It refuses projected data.
- Not done: closure-phase legs from the OI_T3 coordinates (the reader still uses the matched VIS2 rows' (u, v); TODO in `oifits.py`).

## Stage 6a.1: orbits and binary-frame scenes (after 6a.0; about 25–30 h by its own table)
Design: [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md) (O).

**Decided (2026-10-03):**
- Orbits are built in virgil.
- They run on jaxoplanet, as an optional `[orbits]` extra. Neither orbitize! nor orvara is used.
- The user-facing conventions are those of O §2.1.

**Defaults in force** (O §7, until Ben says otherwise):
- time dependence through a snapshot, `at(mjd)`;
- a fitted offset for orbital skew, rather than the physical one;
- simulated systems first, then real data.

**Build**, in this order (agent hours from O §6):

| Item | Effort |
|---|---|
| `orbits.py`: `KeplerOrbit` and `ThieleInnesOrbit`, converters to and from jaxoplanet, convention unit tests and a reference ephemeris | 4–5 |
| Starting orbits: per-epoch positions from the existing binary tools, then a Thiele–Innes least-squares solve on a grid of (P, e, T₀); `PositionData` | 2–3 |
| `StateVectorOrbit` and its regular forms, for short arcs | 3 |
| `RVData` and axial priors; `distance_pc` and the derived mass | 3 |
| `SourceModel.at(mjd)` and time-dependent `OIData.model` | 3–4 |
| `Attached(component, orbit, anchor, bind, offsets)`: any component's angles tied to the binary's frame (line of centres, nodes, inclination, "facing the primary") | 3 |
| `simulate(scene, template)` and `bias_test`, for bias tests across instruments and for planning | 2–3 |
| `TruncatedCone`, a thin conical shell (analytic, with an elliptical cross-section option and a render ↔ model test; the prototype is in the Apep data folder) | 3–4 |
| OIFITS position-angle round trips (GRAVITY layout; AMICAL and virgil writers) | 2, then 2 per real anchor |

**Later:** physical orbital skew from aberration (2 h), once a system near periastron needs it.

**Log:**
- `orbits.py` (item 1): `KeplerOrbit` and `ThieleInnesOrbit`. jaxoplanet only solves Kepler's equation (`jaxoplanet.core.kepler`, with its exact derivatives); positions come from our Thiele–Innes constants (§2.4), so every convention lives in `orbits.py`, and `OrbitalBody` appears only in the converters. Velocities are exact JVPs. `[orbits]` is a new extra (and `integrations` includes it). Tests: §5.1.1 (an independent NumPy ephemeris, 1e-10 of a in float64, 1e-5 in float32) and §5.2.1–6; §5.2.7–8 come with `Attached` and the starting orbits. The α Cen and interferometric anchors (§5.1.2–3) wait for Ben's choice.
- Starting orbits (item 2): `PositionData(mjd, dra, ddec, cov)` and `.from_sep_pa`, with `loglike` and `whitened_residuals`; `starting_orbits(positions, periods, eccs, n_phase, n_best)` vmaps a whitened least-squares solve for (A, B, F, G) over the (P, e, t_peri) grid and returns the best `KeplerOrbit`s by χ² (Ω in [0°, 180°)). §5.2.8 passes (the true grid point, χ² ≈ 0); on 12 noisy epochs off the grid the best start is within 5% in period and 2 mas in position. Per-epoch positions come from the existing binary fits; a helper for that comes with the first real example.
- `SourceModel.at` and `Attached` (items 5–6): `at(mjd, t_ref=0.0)` returns the model at time t_ref + mjd, so float32 offsets keep their precision; static models return themselves (`time_dependent` is False) and keep the fast path. `System.at` and `Rotated.at` map over their parts. `OIData.model` vmaps a time-dependent model over samples (each at its own time); JAX computes the time-independent parts once. `Attached` anchors at the secondary, the primary or a fraction along r, binds attributes to `KeplerOrbit.frame` angles (with `az_pas` deprojected for the rim), and adds fitted offsets. Tests: an attached companion equals static binaries at each epoch; float32 with a t_ref 30 years away; the §5.2.7 orientation (bright side within 2° of the primary at three phases); gradients. Shared orbits for several components go through a model function.

**Log (TruncatedCone):** ported from the Apep prototype (`~/data/apep_gravity/scripts/truncated_cone.py`) into `models.py`. A stack of rings: each projects to an ellipse with visibility J0(2πρ√(q⊥² + (ratio · q∥ sin β)²)) at its centre's offset `(s cos α − tip) cos β` along the projected axis, weighted by area × exp(−(s − s0)/length) over five e-folds (midpoint rule, `n_rings`), times the shell's Gaussian envelope. New: the elliptical cross-section `ratio`, parameter checks, `is_physical`, and an image accumulated ring by ring, with points no further apart than a third of the shell's width added in blocks of 64 (`lax.scan`; a 512² render of a large thin cone peaks at 0.8 GB), so thin rings render as continuous bands. Tests: render ↔ model (2e-3, and to 0.35 cycles/mas for a narrow resolved cone), tilt-sign symmetry, a face-on cone real and rotation-symmetric, `ratio` irrelevant at tilt 0, the sky orientation (North at PA 0, East at PA 90), second-order convergence in `n_rings`, bad shapes rejected, and `is_physical` after `.set`. Attaching it to an orbit (`bind={"pa": "line_pa", "tilt": "line_tilt"}`) is what the Apep sketch does.

**Log (StateVectorOrbit; item R4):** parameters (dra, ddec, vra, vdec, dz, vz) at `t_ref` (mas, mas/yr) and μ = 4π²a³/P² (mas³/yr², distance-free). `to_kepler` goes through the Thiele–Innes constants (P̂ = e/|e| and Q̂ = Ŵ × P̂ give A, B, F, G, C, H), so the angle conventions are those of `ThieleInnesOrbit.to_kepler`, and the line-of-sight constants C, H choose the node. A circular state uses the position at `t_ref` for P̂. Unbound states are rejected. Round trips KeplerOrbit → state → KeplerOrbit reproduce the 3-D orbit to 1e-7 mas for generic, e = 0.9, e = 1e-6, i = 2° and retrograde orbits; gradients are finite under jit. The (h, k), (p, q) forms of the design were not needed: the Thiele–Innes route has no removable singularity at i = 0, and at e = 0 only the periastron is undefined.

**Log (RVs, masses, axial priors; items R5–R6):** `RVData(mjd, rv, d_rv, star)` predicts γ + share · v_rel, with v_rel the orbit's exact dz/dt in mas/day times D (pc) × 1.496e5/86400 km/s, and share −q/(1+q) for the primary, 1/(1+q) for the secondary. jaxoplanet's RV code is not used, so the physical scale stays explicit. `total_mass` and `distance_pc` (Kepler's third law, a in au = a_mas · D / 1000, P in years). `AxialVonMises(mean, kappa)`: a von Mises in 2θ on [0°, 360°), for a node known modulo 180°. `fit(..., likelihoods=[...])` takes callables of the fitted values returning whitened residuals (`PositionData.term(orbit_fn)`, `RVData.term(params_fn)`); data may then be `()`. Tests: the RV peak-to-peak equals 2K from Kepler's laws (to 1e-4); total mass 1 M☉ for 1 au at 1 pc in a year; the axial prior is normalised with period 180°; a joint position + RV fit from both nodes, where only the true node fits (χ² < 1e-3 against > 1e3). `numpyro_model(..., likelihoods=[...])` is done: term `i` is the site `likelihood_<i>`, adding the data's full normalised Gaussian log density for `PositionData.term` and `RVData.term` (as the OIData terms do) and `-χ²/2` alone for a plain callable. `RVData.term(params, jitter="rv_jitter")` fits RV jitter s (errors sqrt(d_rv² + s²)): a term may define `has_log_norm` and `log_norm(values)` (Σ log σ_eff), which `fit` adds to the loss (L-BFGS by default, as for `noise=`; LM raises `TypeError`); `numpyro_model` uses the term's `loglike`. Tests: loglike equals scipy's Gaussian with σ_eff, zero jitter equals the plain term, the numpyro log density, and a 30-point fit recovers s = 1 km/s. `RVData(..., instrument=labels)` and `RVData.term(..., marginalise_offsets=(mean, sd))` marginalise one velocity zero point per instrument analytically (Luger, Foreman-Mackey & Hogg 2017): `d ~ N(m + Aμ, C + AΛAᵀ)` by Woodbury and the determinant lemma in O(N k²), the log-determinant (including the jitter) in `log_norm`, residuals of length N from an exact square root `z - W(I + Λ^-1/2 L⁻ᵀ)⁻¹Wᵀz`, and `term.posterior(values)` for the conditional mean and covariance of the zero points. One zero point per instrument and no separate γ is the non-degenerate parameterisation (`params` returns γ = 0); only finite prior sd is supported, `True` is N(0, 1000²) with a warning. Tests: dense scipy multivariate normal (2 instruments, jitter on), dense conditioning, a fit matching free offsets, and one instrument with a tiny prior equal to a fixed γ.

**Log (simulate):** `virgil.simulate`: `simulate(scene, template, key, noise_scale, shift_days)` is `with_model` on the template, with every sample at its own time for moving scenes and `shift_days` to move the epochs (R8's `mjd=`); `bias_test(scene, template, model, priors, n, key, **fit_kwargs)` returns the fitted values and χ²_red of each draw. Not yet: R8's drawn nuisances (`noise=` dicts, correlated blocks, a wavelength scale; they come with 6d) and bandpass smearing (6a).

**Log (PA round trips):** §5.3.1, synthetic, in `tests/test_pa_round_trip.py`: a binary with flux ratio 0.2 is written in a GRAVITY layout (STA_INDEX 28/23/18/1, baselines from the highest index, five channels, three hour angles) and a 7-hole-mask layout, read back, and found by a grid search over the whole field: within 1° of its PA and fainter than the primary. A negative control (closure phases negated in the file) comes back 180° away, so the test can see a flip. AMICAL's writer is not tested yet (AMICAL is not in the test environment); real anchors (§5.3.2) are with the validation plan (virgil-validation#9).

**Tests:**
- Conventions and ephemerides: O §5.1–5.2.
- Synthetic position-angle round trips: O §5.3.1 and §5.3.3. These need no real data.
- End to end on simulated systems: O §5.4, where Apep is optional as the real-data case.

**MWE:** a companion on its orbit with a disc attached to it, every data point evaluated at its own time. This is the sketch in `design/sketches/orbit_attached.py`.

**Real anchor binaries** for the ephemeris and position-angle tests (O §5.1.2–3, §5.3.2) will be found in the ESO archive (Ben). They don't block the rest of the stage.

## Stage 6a: spectro-interferometry, matching PMOIRED (about 10–13 h)
This stage matches PMOIRED's spectral modelling. [`pmoired_parity.md`](pmoired_parity.md) compares the two packages feature by feature. It comes before 6b and 6c, because 6b's per-filter fluxes are node spectra.

**Workflow and additions.** [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md) sets out an end-to-end spectro-interferometric workflow (worked example: GRAVITY data on Apep) and adds to 6a:
- `Nodes(..., outside=0.0)` for line excesses on a continuum, with positivity checked on the total; `Tabulated` (merged in PR #124, provisional) becomes a node spectrum;
- `System.total_spectrum` for OI_FLUX;
- VISPHI with the pipeline's continuum normalisation, and a test in the resolved regime;
- error floors sharing one function with the fitted `noise=` terms;
- a documented rule for when smearing matters, with a real-data check;
- a prior on the reference component's spectrum, and docs on its degeneracy with the others (S §2.2b);
- `with_error_floor`, sharing `likelihood.inflated_errors` (S §2.6).

**Log (spectra, PR A):** `GaussianLine` / `LorentzianLine` (`amplitude` = peak flux, `line_wavel`, `fwhm`, `wavel0`), `Nodes(values, wavel, kind="linear"|"cubic", outside="constant"|float, wavel0)` with a natural cubic spline, and `Sum(parts, wavel0=, **named)` with `System`-style paths (`"star.flux.brg.amplitude"`). Every spectrum's reference flux is its value at `wavel0` (`Spectrum()` evaluates there); positivity is checked on the total at the characteristic wavelengths (`wavel0`, each node, each line centre), not on parts; smoothness stays a prior. `Tabulated` is deprecated with a `DeprecationWarning` but unchanged (the Apep scripts use it). `SourceModel.total_spectrum(wavel)` (a `System`: the sum over parts) and `OIData.select(wavel_min, wavel_max)` (closure triangles kept whole). Tests in `tests/test_spectra_lines.py`: line integral and centroid (x64), node interpolation, `outside=0.0` excesses, a line in a component moves the visibility only near the line, `select` splits the likelihood. Left for PR B, after 6d merges: the OI_FLUX / T3AMP / VISPHI readers, continuum normalisation of VISPHI, error floors through `inflated_errors` (6d), bandpass smearing, the spectral-resolution kernel and the primary beam.

**Defaults in force** (S §4, until Ben says otherwise):
- `noise=` grows into the general per-dataset nuisance argument;
- every spectrum's reference flux is its value at `wavel0`;
- closure phases are used everywhere, plus the closure-free part of continuum-normalised VISPHI in the line windows (S §2.3), so nothing is counted twice.

**Build:**
- **Spectra.** New `Spectrum` types:
  - Gaussian and Lorentzian lines (`wl0`, FWHM, amplitude; emission or absorption);
  - node spectra (linear or cubic spline through free fluxes at fixed wavelengths);
  - sums, so that continuum plus lines is one spectrum.
- **Observables.**
  - Read and model T3AMP, OI_FLUX (FLUX and NFLUX), differential phase, normalised |V| and V², and correlated flux.
  - Allow V² with |V|, and closure phase with VISPHI, in one dataset.
  - Normalisation uses continuum ranges, or the analytic continuum of the lines.
  - **Optional debiasing of OI_VIS amplitudes** (eht-imaging's `debias`), in the reader PR that reads VISAMP. |V| measured at low SNR is biased high, because noise adds in quadrature to the amplitude, so low-SNR GRAVITY VISAMP should either offer a debiasing option or at least document the bias and when it matters. Off by default.
- **Instrumental effects.** A spectral-resolution kernel, and bandwidth smearing by oversampling in wavelength.
- **Primary beam (fibre coupling), optional** (GRAVITY review §1; decided 2026-10-03). A Gaussian coupling of FWHM ≈ λ/D, optionally broadened by tip-tilt jitter, per telescope. It weights each component's flux, integrated over its brightness for extended components, by its position. One coupling model feeds both the coupled photometry (OI_FLUX) and the visibilities' normalisation. `System.total_spectrum` stays the *intrinsic* total.
- **Errors.** `OIData` error floors (absolute and relative) and flags, alongside `with_error_scale`.

**Tests:**
- A line's flux integral and centroid.
- Node spectra interpolate their nodes.
- Differential phase of an offset line-emitting component, against the analytic photocentre shift.
- Smearing against brute-force integration over the band.
- The primary beam:
  - an offset point source loses flux and visibility amplitude consistently;
  - the attenuation is chromatic, scaling with λ/D;
  - photometry and visibilities agree for an extended component.
- OIFITS round-trips for each new table.

**MWE:** a GRAVITY-like Brγ disk: continuum star plus a line-emitting Gaussian offset with velocity, fitted to simulated V², differential phase and NFLUX.

## Stage 6d: calibration nuisances correlated across channels (after 6a and the Apep agent's library commits; about 9–13 h)
Decided 2026-10-03. Spectro-interferometric systematics (transfer-function jitter, piston and injection drifts) are mostly common to all channels of a frame, which diagonal error inflation does not describe; GRAVITY data on Apep are the first dataset here that needs this. The design is in [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md) §2.4.

**Build:**
- **Gains on log |V|** (decided), with low-rank blocks per frame (GRAVITY review §4c):
  - telescope-based gains;
  - per-baseline gains;
  - a chromatic mode from coherence loss, exp(−a/λ²).

  They are marginalised analytically by Woodbury, as a small-log-gain approximation, so the likelihood keeps one whitened residual vector plus a log-determinant.
- **No closure-phase offsets by default.** If calibrators show non-closing errors, use the baseline-based form T·e, a small-phase approximation under the chord likelihood (GRAVITY review §13).
- **Widths** are fitted or known (`vis_gain`, and optionally `phi_offset`, in the per-dataset `noise=` specification). This needs `OIData.frame` (§2.5 of the same note).

**Tests:**
- The whitened residuals' squared norm and the log-determinant against a dense covariance.
- A simulated GRAVITY-like dataset with injected per-frame gains: the widths are recovered, and parameter errors are calibrated where the diagonal model's are not.

**MWE:** a simulated multi-channel binary with an extended component and injected per-frame gains, fitted with 6d and with diagonal error terms, comparing the parameter errors with the truth. A real-data check (e.g. Apep's GRAVITY data) is optional.

**Log (gains, branch `stage6d-nuisances`):**
- `virgil.gains`: `GainModes` and `gain_modes`, set with `OIData.with_gains(telescope=, baseline=, chromatic=, modes=)`. The groups are telescope, baseline, chromatic ((λ_ref/λ)²) and supplied 1σ modes, e.g. a calibrator PCA's from virgil-vlti. Widths are fitted with the noise terms `vis_gain_<group>`, one per group, rather than a single `vis_gain`.
- Marginalisation: blocks are the connected groups of modes, whitened by successive rank-one steps (smooth gradients, also for degenerate modes and zero widths). The log-determinant is spread over effective errors, so `_gaussian_loglike` and `fit`'s normalisation are unchanged. The Jacobian dObs/dlog|V| is taken from the model (2V², |V| or 1).
- `fit` includes the normalisation and refuses LM when any dataset has gains. `numpyro_model` gets them through `model_loglike`.
- `OIData.stations` (from `STA_INDEX`, or a dictionary's `stations`), and `gains` survive `_subset`/`split_by_epoch` (dropping rows is an exact marginal). `with_model(key)` draws the gains at their default widths and applies them exactly.
- Tests: whitened norm and log-det against the dense covariance (V², |V|, log|V|); block structure; width terms; zero widths reduce to the diagonal likelihood, with finite gradients on degenerate data; splitting by epoch; covariance of the drawn gains; refusal for projected data; fit defaults to L-BFGS.
- Not yet: the width-recovery and calibration study, and the MWE (OzSTAR); `phi_offset` (after #174); `wavel_scale`.

**Also in 6d:**
- **`wavel_scale`** (and `wavel_offset`). A per-dataset wavelength nuisance in `noise=` (S §2.6; 1–2 h). The GRAVITY default is λ′ = λ(1 + s) + δ, with s ~ N(0, 2×10⁻⁴) and δ = 0 unless lines constrain it.
- **The dual-field (in-field) calibrator recipe.** Done in [virgil-vlti](https://github.com/benjaminpope/virgil-vlti) (`virgil-vlti-calibrate --dual-field`), not in the core (S §2.7). Its transfer-function error should become 6d's known-width gains.

**Merge order (historical).** The Apep agent's library commits (`EllipticalGaussian`, `Tabulated`, fitted error terms `noise=`, the anisotropic `GaussianField`, `GaussianArc`, the rim-gradient fix) were merged in PR #124, before 6a. 6a's node spectra will replace `Tabulated`, and 6d extends `noise=`.

## Stage 6e: closure amplitudes (after 6d; not yet estimated)
From the virgil-validation comparison with eht-imaging, which fits closure amplitudes (`camp`) and log closure amplitudes (`logcamp`, `logcamp_diag`).
- **Why.** Closure amplitudes are immune to telescope-based amplitude miscalibration. That is valuable for masking data with a varying pupil.
- **Build.** Closure amplitudes (and their logarithms) beside closure phases in `OIData`, correlated as the closure phases are (quadrangles that share baselines), so the likelihood still has one whitened residual vector.
- **Design note first.** For VLTI this overlaps with 6d: 6d models telescope-based gains, while closure amplitudes discard them. The note should say when each is preferable, and it is written after 6d lands.

## Stage 6: polychromatic imaging (about 8–12 h; design is finalised at the Stage 5 checkpoint)
**Build**, in increasing order of complexity; stop where the science needs stop: Item 2 is Stage 6b and item 3 is Stage 6c. Both use 6a's node spectra for per-filter or per-channel fluxes.
1. **Grey image with a spectral index.** This already works through `flux=PowerLaw(...)`. Add documentation and an MWE only.
2. **Joint multi-filter AMI.** Several filters (F380M/F430M/F480M) share one image, with per-filter flux ratios (the analogue of dorito PR #32's `JointResolvedDiscoModel`). The per-observation models this needs exist since Stage 4b: `fit` accepts a model function returning one model per dataset.
3. **`ImageCube`.** Per-channel log-brightness with a GP along wavelength: a separable DCT field over (λ, y, x). The translation gauge is fixed per channel (one centroid per channel), unless an analytic star anchors it.

**Tests:**
- Cube visibilities per channel equal grey `Image` visibilities when all channels are equal.
- Joint AMI recovers the per-filter fluxes.
- Chromatic recovery of a synthetic disk whose colour changes with radius.

**MWE-A:** joint three-filter AMI imaging (simulated, on ν Hor-like coverage and noise in each filter).

**MWE-B:** a MATISSE chromatic disk across the L band, with the star analytic.

**Risk:** this is the least-specified stage, and per-channel transforms use the DFT (or one MFT per channel on lattices); the NUFFT is shelved (issue #75).

**Checkpoint:** **merge milestone 2**.

## Stage 6.5: the 0.2.0 release (added after the October 2026 codebase review)
The version is already 0.2.0 but the release is not out. The review (`codebase_review_2026-10.md`, summarised in the PRs that fix it) found blockers B1 to B5 and a set of documentation and reference errors. All of these are being fixed on parallel branches, in review at the time of writing, and will be combined into one correctness PR and one performance PR:
- the blockers (B1 to B5), including the `Tabulated` API (B5; see `chromatic_sources.md`);
- `CHANGELOG.md`, with the rename and the Stage 6.0 change to every four-telescope χ²;
- the text fixes in this plan, the design notes, `AGENTS.md` and `MIGRATION_VIRGIL.md`, and the reference corrections;
- re-executing the six MWEs (`mwe_sam_v2_cp`, `mwe_vlti`, `mwe_stress_test`, `mwe_gaussian_field`, `mwe_evidence`, `mwe_elr_chara`): **done 2026-10-04** on OzSTAR (job `virgil_notebooks`, virgil `ad131bd`); the new numbers are logged beside the old ones;
- publishing `virgil-astro` 0.2.0 and a final `drpangloss` 0.2.0 that forwards to it with a `FutureWarning` (the human's job).

**Open debts** (not blockers):
- `Tabulated` is private in 0.2.0 (Ben's decision) and becomes `Nodes` in Stage 6a;
- `inflated_errors(where=, combine=)`;
- MFT caching (formerly in Stage 7's list);
- the discrepancy target N − γ for `l_curve` (γ now exists, `imaging.error_scale`);
- the TSV-evidence decision: Stage 5 promised an evidence for TSV, which was not built, and no decision is recorded.

The review also suggests restructuring the plan: replace "Order of work" with a status table, split 6a into spectra, observables with the reader, and instrument effects (budgeted at about 20–25 h, spectra first), give 6d a design item for composing its offsets with `ClosureNoise` grouping, give MEM preconditioning its own estimate, and archive the orchestration section. None of that is done here; see the review's "What should change in the plan".

## Stage 7: hardening and release (about 8–11 h, of which 3–4 h is release chores and the rest MEM preconditioning)
- **Maximum-entropy preconditioning and better solvers** (Ben, 2026-10-03; important in the long run).
  - **The symptom.** At weak weights, MEM's L-BFGS fits are badly conditioned: faint pixels are almost unconstrained, so steps along those flat directions are tiny. One weight in `mwe_gaussian_field`'s sweep did not converge in 200,000 steps. The tutorials and MWEs keep their sweeps to w ≳ 3 with up to 2–5 × 10⁴ steps, which works around the problem rather than fixing it. The GP prior doesn't have it, because LM converges in 20–50 steps in whitened coordinates.
  - **Options:**
    - precondition L-BFGS in the entropy metric (diag 1/b, as Skilling & Bryan 1984 do), or in a basis whitened by the default image;
    - a Skilling–Bryan-style subspace solver;
    - a least-squares (LM) form of the entropy term;
    - a looser, better-founded stopping rule.
  - **Success test:** an L-curve from w = 0.1 to 10⁴ that converges at every weight within the default step limit, in float32 and float64.
  - **Multi-start or annealed-weight image fitting** (priority: image fits are non-convex, and a single start can settle in a poor local minimum). Two independent pieces of evidence:
    - virgil-validation found that, from the same start, eht-imaging's L-BFGS-B and virgil end in different local minima, although from each other's solution both agree to 5e-8 in loss. The minima differ; the optimisers do not.
    - An independent real-data analysis of a GRAVITY dataset found Levenberg–Marquardt fits of `GaussianField` images from random latent starts getting trapped: not converged after 10⁴ steps, with χ²/N about 3.

    Possible helpers: a multi-start that keeps the lowest-loss converged fit, or imaging in rounds with blurring and restarting, as eht-imaging users do.
- API review for consistency and naming, with docstrings (units and examples) for every public object.
- A mkdocs API page and a "choosing a regulariser and prior" guide.
  - **K-fold cross-validation for the regularisation weight** (MPoL's `CrossValidate`): hold out folds of the data, fit the rest at each weight, and choose the weight with the lowest held-out χ². Unlike `l_curve`, `discrepancy` and the evidence, it does not assume the error bars are right, which matters for real VLTI data. The guide should compare it with those three.
- Update `design/chromatic_sources.md` to mark `Image` as done.
- Remove any leftovers, and do a final full-suite run under float32 and x64.
- A version bump and a changelog entry.

---

## Stage 8: parametric parity with PMOIRED (after milestone 2; about 8–11 h)
See [`pmoired_parity.md`](pmoired_parity.md). Orbits, first listed here, are now Stage 6a.1.
- **`Projected(source, inc, pa)`.** A wrapper that stretches uv, like `Rotated`. It makes any component elliptical, and replaces per-class `inc`/`pa`.
- **`RadialProfile`.** A callable I(r) with inner and outer radii, transformed by a fixed-quadrature Hankel transform: order 0, plus order n for azimuthal harmonics. It covers thick rings, power-law and limb-darkened disks, and harmonics on any profile. Crescents are differences of offset disks.
- **`bootstrap_fit`.** Resamples by date and baseline, keeping spectral vectors whole. Resample whole snapshots or whole baselines, never single points, since the points within one are correlated (as PMOIRED and CANDID's `fitBoot` do). The virgil-validation PMOIRED comparison is its test.
- **Spectral correlations between channels.** Now Stage 6d, because GRAVITY data need them.

**Tests:**
- Hankel profiles against analytic UD, Gaussian and ring visibilities.
- `Projected` against inclined analytic models.
- Bootstrap spreads against the Laplace errors on a simple fit.

## Totals
- **Stages 0–4** (MAP imaging for AMI and long-baseline data, the transform benchmark, reproduction of the dorito result): about 18–26 h of agent time.
- **Stages 5–7:** about 22–31 h more (Stage 7 now includes MEM preconditioning), plus:
  - 6a.0 and 6a: about 12–16 h;
  - 6a.1 (orbits and binary-frame scenes): about 25–30 h;
  - 6d: about 9–13 h.
- **Stage 8** (the rest of the PMOIRED parity): about 8–11 h.
- **Overall:** the stage headings sum to about 94–127 h of agent time (6a.1 at 25–30 h, Stage 7 at 8–11 h), of which about 70–93 h remains from Stage 6a.0 on, before re-estimating 6a. The old "35–50 h" was the serial estimate for Stages 0–7 before 6a.1, 6d and Stage 8 were added. There are 8 feedback checkpoints.

External waits: only the OzSTAR GPU benchmark run, which you launch. All test data are simulated using ν Hor baselines and noise.

**Coverage fixtures.** The MATISSE ν Hor OIFITS files live in `/Users/benpope/code/nuHor/data/`, where they are git-ignored. In Stage 4, a script (`scripts/make_coverage_fixture.py`) extracts **only the geometry and noise**: u, v, wavelengths, closure-triangle indices, flags and per-point σ. These go into a small `data/coverage_nuhor_matisse.npz`, with no measured visibilities or phases, so tests and notebooks are self-contained. The AMI record already in the repo covers AMI. If you'd rather not commit even the derived coverage, the fallback is synthetic 4-telescope coverage with σ statistics matched to ν Hor.

## Deferred (triggers recorded in the design note)
| Item | Revisit when |
|---|---|
| Forward-model centroid recentring | Sampling drifts in position despite the centroid prior |
| Twin-folding helpers | We fit V²-only data routinely |
| Basis and decoder parameterisations | zodiax 0.6 `Map`/`Mask`/decoders are released |
| MGVI/geoVI | Not planned |
| Cross-validation for the weight (e.g. held-out DISCO modes) | K-fold cross-validation is planned in Stage 7; held-out DISCO modes only if that proves too coarse |
| Deep Probabilistic Imaging and learned priors | Not planned; covered by other work in the group |
| Top-Set surveys over synthetic truths | Not planned here; sophisticated PDS 70 truths are being built separately |
| Bandwidth-smearing forward model | A real field-of-view need |
| Stage 3c, real AMI data on PDS 70 (the original scope is kept at the end of this file) | The AMIGO DISCO miscalibration on PDS 70 is fixed (bare-star rms c/σ of 27–800; `amigo-pipeline` issue 2). Until then the recommendation is the GP prior with a fitted noise scale rather than MEM on those data |
| Reinstating a NUFFT backend (issue #75; code in PR #71) | jax-finufft reuses plans (jax-finufft #157) and we have datasets of ≳10⁵ irregular points with ≳256² images |

## Critical files
- **New:**
  - `src/virgil/fitting.py`, `src/virgil/imaging.py`, `src/virgil/fields.py`, `src/virgil/_precision.py`
  - `design/image_reconstruction.md`
  - `tests/test_image_model.py`, `tests/test_fitting.py`, `tests/test_imaging.py`, `tests/test_fields.py`
  - Notebooks: `imaging_ami.ipynb` (Stages 1–3), `imaging_long_baseline.ipynb` (Stage 4), `imaging_gp_sampling.ipynb` (Stage 5), `imaging_chromatic.ipynb` (Stage 6)
- **Modified:**
  - `models.py`: `Image`, later `ImageCube`.
  - `_geometry.py`: `image_visibilities`.
  - `likelihood.py`: `whitened_residuals`, `model_loglike`.
  - `__init__.py`, `pyproject.toml` (optax, lineax; `[sampling]` extra), `AGENTS.md`, `mkdocs.yml`, `scripts/sync_tutorial_docs.py`, `design/chromatic_sources.md`.
- **Reused unchanged:** `OIData` (including `with_model`), `amigo.load_oi_data`, `build_model`, `System`, `spectra.PowerLaw`, `plot_model`, `inference.gaussian_fisher` (for parametric uncertainties), `_grid.warn_unconverged`.

## Verification
- Each stage has its tests plus a green full suite (float32 and x64), `ruff`, the tutorial sync test, and the docs build.
- **Stage-specific quantitative gates:**
  - Stage 0: regression tolerances.
  - Stage 3: synthetic recovery metrics and recovery metrics against simulated truth (AMI).
  - Stage 4: star-anchored recovery and the SNR/coverage sweep on ν Hor coverage.
  - Stage 5: exact GP covariance and sampling coverage.
  - Stage 6: per-channel consistency.
- The MWE notebooks are re-executed locally at every later stage, so earlier examples keep working. The docs-sync test keeps them matched to the docs. Heavy steps (real-data fits, sampling) are cached or reduced so re-running stays quick.

**Scope changes since revision 4:**
- The OzSTAR GPU benchmark was run on 2026-09-30 (A100; results in `design/image_reconstruction.md`). It confirmed the TF32 fix and led to shelving the NUFFT (issue #75).
- Coverage is synthetic, the fallback named under "Coverage fixtures": `coverage.vlti_oidata` (four-telescope Earth-rotation tracks over many channels) and `coverage.nrm_oidata`. No ν Hor fixture was committed, and real data (PIONIER) are analysed in nuHor instead.

### OzSTAR GPU test (done 2026-09-30)
Run by the user (Claude never connects to OzSTAR) on an A100. It confirmed that `Precision.HIGHEST` gives float32-accurate DFTs on the GPU (7.9e-7, against 2.7e-5 with TF32), and that the jax-finufft NUFFT was too slow there, which led to its removal. Results and figures are in `design/image_reconstruction.md`; the scripts are in the git history of PR #71.

---

## Implementation orchestration

### Dependency graph (chunks)
```
C0a design note ─────────────────────────────────────────────┐
C0b whitened_residuals + model_loglike + regression ──┐       │
C4a coverage fixture + synthetic coverage generator ─────────┼──▶ C4 long-baseline MWEs
C1a image_visibilities (DFT) ──┬─▶ C1b Image ──┬─▶ C1c tests/MWE (AMI simulation)
                               │               ├─▶ C1d scene library (spiral/ring/star+comp)
                               │               ├─▶ C5a GaussianField (math; can start here)
                               └─▶ C2 NUFFT backend + CPU bench
C0b + C1b ─▶ C3a fit + _precision ─┬─▶ C3c diagnose ─▶ C3d AMI MWEs (needs C1d)
C1b ─▶ C3b regularisers ────────────┘
C3a + C5a ─▶ C5b image_priors/GN diagonal/sampling MWE
C3 + C5 ─▶ C6 polychromatic (design first) ─▶ C7 hardening
```
- **Critical path:** C1a → C1b → C3a → C3c/C3d → C4.
- **Parallel lanes:**
  - C0a runs alongside C0b.
  - C2 runs alongside C1b–C1d.
  - C3b runs alongside C3a.
  - C5a starts as soon as C1b exists, alongside Stage 3.
- **Shared files** (`models.py`, `__init__.py`, `AGENTS.md`, `pyproject.toml`) have a single owner per stage, which is the orchestrator. Sub-agents return diffs to those files and do not edit them directly, which avoids merge conflicts between parallel worktrees.

### Model allocation
| Chunk | Model | Why |
|---|---|---|
| C0b likelihood unification | **Opus** | Changes existing semantics; the regression judgement is subtle |
| C1a DFT conventions and precision | **Opus** | Sign and orientation errors are the historical bug source |
| C2 NUFFT convention mapping | **Opus** | Half-pixel factors, axis pairing, eps floors |
| C3a `fit` (transforms, LM pitfalls, dtype policy) | **Opus** | The core API; numerical subtleties |
| C5a `GaussianField` (DCT spectrum, normalisation, exact-covariance test) | **Opus** | Mathematical correctness |
| C6 polychromatic design | **Opus** (orchestrator) | Open design questions |
| C1b `Image`, C3b regularisers, C3c `diagnose`, C5b helpers | **Sonnet** | Well specified by this plan; Opus reviews the diff |
| All tests except the exactness tests, C1d scenes, MWE notebooks, tutorials | **Sonnet** | Clear specs; iterate against running code |
| C0a design note, C7 docstrings, mkdocs pages | **Sonnet** | Writing from existing material |
| C4a fixture extraction, running the suite/ruff/docs build, tutorial sync, benchmark runs and reporting | **Haiku** | Mechanical |
| Every PR's final review before a checkpoint | **Opus** (orchestrator) | Consistency across chunks |

Fable 5.1 is also available in this environment, but I have no evidence about how it compares to Opus on this kind of numerical work. Try it on one Opus-tier chunk, e.g. C5a with its exact tests as the arbiter, before relying on it.

### Wall-clock estimates with this orchestration
These are agent time only, excluding your checkpoint reviews. The serial estimate was 35–50 h.

| Stage | Serial | Orchestrated | Notes |
|---|---|---|---|
| 0 | 2–3 h | **~1.5 h** | Four parallel chunks; C0b is the long pole |
| 1 + 2 | 6–8 h | **~3.5–4.5 h** | C2 runs in parallel with C1b–d |
| 3 | 6–9 h | **~4–5 h** | C3a and C3b in parallel, then C3c/C3d |
| 4 | 4–6 h | **~2.5–3 h** | Mostly Sonnet MWE work plus the fixture |
| 5 | 6–8 h | **~3–4 h** | C5a is largely done during Stage 3 |
| 6 | 8–12 h | **~5–7 h** | Design by Opus first, then parallel implementation |
| 7 | 3–4 h (release chores only; 8–11 h with MEM preconditioning) | **~1.5–2 h** | Sonnet and Haiku |
| **Total** | 35–50 h | **~21–27 h** | Plus the 8 checkpoint waits |

Orchestration overhead (reviews, integration, fixing cross-chunk mismatches) is included, at about 20% per stage. The main risk to these numbers is iteration on numerical issues in C1a, C3a and C5a, which could add 1–3 h each.

### Recommended interface
- **Run the orchestrator in the Claude Code CLI or the desktop app, one session per stage,** in `~/code/drpangloss` (the local folder keeps its old name):
  - They handle long autonomous runs, background sub-agents and `isolation: "worktree"` (each parallel chunk gets its own git worktree and branch) better than an editor-bound session.
  - A session per stage keeps context clean. The plan file and `design/image_reconstruction.md` are the handoff between sessions.
- **Use VS Code (this extension) for checkpoints:** reviewing diffs, running the MWE notebooks and looking at the figures. It is good for that, less so for multi-hour orchestration tied to an open editor window.
- **For the parallel stages** (0, 1+2, 3), you can say "use a workflow" to have the orchestrator run a scripted multi-agent workflow (a fan-out of chunks, each followed by an Opus review) instead of hand-launching agents. The size guideline is under 10 agents per workflow, which fits these stages.
- **Checkpoints as GitHub PRs** (`imaging-sN-*`, stacked, into `imaging`), so you can comment asynchronously; the next session starts by reading those comments.

### Stage 3c: real AMI data, PDS 70 (deferred; original scope)
**Data:** the AMIGO DISCO products for PDS 70 in `/Users/benpope/code/nuHor/data/PDS70/` (local only; never committed). Read only the fields needed (operators, coefficients, σ, uv, wavelength, rotation), and avoid listing or printing large files.
**Scope:** virgil supplies a fast JAX library with the features interferometrists expect; synthetic truths for calibrating PDS 70 reconstructions are being built separately, so this stage does not do that.
**Build:** an agent that deconvolves PDS 70 in each filter, separately and jointly (Stage 6's joint multi-filter machinery when available), with a wide range of options: regularisers (maximum entropy expected best, then TSV, then TV), weights from L-curves (discrepancy and corner), fields of view and pixel scales, starts (flat, parametric fit), analytic star or not, supports, and centroid priors. "Beat it to death": the aim is a general picture of what is robust across choices.
**Compute:** demo locally first on a reduced set. If the full grid would take hours or exceed the laptop's RAM, hand the user an OzSTAR GPU script (`ozstar` skill) rather than running it here.
**Report:** a notebook (not in the docs) comparing the reconstructions across options and filters, with beams, residual maps and `diagnose` output.

