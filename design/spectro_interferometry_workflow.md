# Spectro-interferometry workflow (GRAVITY, MATISSE and similar)

Status: **design**, 2026-10-03. Nothing here is implemented, except where it says it is already in the library (the Apep library commits were merged in PR #124).

This note describes an end-to-end workflow for fitting multi-channel long-baseline data (V², closure phase, differential phase, spectra), the gaps in virgil, and proposed APIs. It extends Stage 6a ([`imaging_plan.md`](imaging_plan.md), [`pmoired_parity.md`](pmoired_parity.md)) and does not replace it: another agent is building 6a. Orbits are in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md).

The colliding-wind binary Apep (VLTI/GRAVITY, three epochs) is the worked example that exposed the gaps. It is a hard case for this workflow:
- a binary with strong emission lines;
- resolved dust brighter than both stars;
- an in-field calibrator;
- calibrated errors underestimated by 2–7×.

The analysis and its scripts are in `~/data/apep_gravity` (`notes/lessons_for_drpangloss.md`, `scripts/calibrate.py`, `scripts/apep_models.py`).

## 1. A reference workflow

| Step | What to do | What virgil has | Example (Apep) |
|---|---|---|---|
| 1. Calibrate | Divide V² by a transfer function interpolated in time per baseline and channel. Subtract the calibrator's closure phases, averaged as unit vectors | Nothing (pipeline products are read as given) | The in-field calibrator recipe, §2.7 |
| 2. Times and frames | Give each frame one time, shared by V² and T3. Keep it per datum | Triangles matched to baselines within 1e-4 d; no times kept | GRAVITY stamps baselines up to 12 s apart, so triangles failed to match (§2.5) |
| 3. Channels | Choose the channels; bin only for compute cost | Multi-channel `OIData` | Binned ×2–×4 by hand to tame correlated errors (§2.4) |
| 4. Continuum fit | Grey geometry with chromatic fluxes (SPARCO), fitted to V² and closure phase over all epochs, with fitted error terms. Multi-start, then NUTS | `PowerLaw`, `BlackBody`, `fit`, `numpyro_model`; `noise=` on `apep-gravity` | 21 parameters, three epochs, no divergences |
| 5. Lines | Continuum plus line excess spectra, in **one** model across the band | Free fluxes per channel (`Tabulated`, merged in PR #124 as a provisional, private class) | Separate continuum and line fits disagreed by 15–20% (§2.1) |
| 6. Spectra and differential phase | Fit OI_FLUX/NFLUX and VISPHI with the visibilities | Not read | Unused, so the reference star's spectral index was assumed (§2.2, §2.3) |
| 7. Compare models | χ²/N with the **uninflated** errors per dataset, scatter plots of data against model, and evidence where Laplace applies | `fit`'s χ², `imaging` evidence | Five models ranked clearly |
| 8. Image | `GaussianField` pixels, with point-like components analytic | Stage 5 | — |

**What already generalises:** grey geometry with spectra (SPARCO), fitted error terms, multi-start fits then NUTS, and model ranking on the uninflated χ². Steps 1, 2, 3, 5 and 6 are where the gaps are.

## 2. Gaps and proposals

### 2.1 One model across the whole band (6a: sum and node spectra)
**Problem.** Fitting the continuum and the line windows separately gives two descriptions of the scene that need not agree where the windows meet. (Apep: the star–halo split differed by 15–20% at the boundary.)

**Proposal.**
1. Fit the whole band at once. Each component's flux is a 6a `Sum` of a parametric continuum and a node **excess** that lives only in line windows:
   ```python
   star = PointSource(
       flux=Sum(PowerLaw(f, index, W0),
                Nodes(wavel=line_nodes, values=excess, outside=0.0)),
       dra=..., ddec=...)
   ```
   - `Nodes(..., outside=0.0)` is zero outside its node range. The continuum is then fixed by the out-of-window channels, and the excess is identifiable.
   - With constant extrapolation (as `Tabulated` does now), continuum and excess are degenerate everywhere.
2. **Positivity applies to the total, not to the excess.** An absorption line has a negative excess. So 6a's non-negativity check moves from each spectrum's amplitude to the evaluated `Sum` at the data's wavelengths. That is a likelihood-time check (or a soft barrier), since the values are traced.
3. **Smoothness of node values** is a prior, not a model feature: a GP over wavelength, or a Gaussian-line `Spectrum` when R resolves the line. This follows the rule in [`chromatic_sources.md`](chromatic_sources.md) that what stays hierarchical lives in the priors.
4. **Reconciling `Tabulated`** (merged in PR #124). `Tabulated(ratio, wavel)` is exactly a linear node spectrum with constant extrapolation, and a reference flux equal to the mean of the nodes.
   - 6a's `Nodes` subsumes it: `kind="linear" | "cubic"` and `outside="constant" | 0.0`.
   - 6a's `Nodes` replaces `Tabulated`, which was merged earlier (PR #124) as provisional and private.
   - **One semantic to settle in 6a:** the reference flux (`wavel=None`, used when rendering). `Tabulated` uses the node mean, but `PowerLaw` uses its value at `wavel0`. Default: every spectrum takes a `wavel0` and is evaluated there.

### 2.2 The total spectrum: OI_FLUX/NFLUX, and the reference component's spectrum (6a, plus a small new item)
**Problem.** V = Σ fᵢ(λ)Vᵢ / Σ fᵢ(λ) is unchanged when every fᵢ(λ) is multiplied by a common g(λ). So with V² and closure phase alone:
- the reference component's spectral shape is **unidentifiable**;
- every fitted index or temperature is measured relative to it.

Only a measurement of the total spectrum breaks this. (Apep: with the reference star fixed at λ⁻⁴, the dust temperature depended on the geometric model, 2200 K against 2990 K, and was degenerate with the halo's index.)

**Proposal.**
1. 6a reads OI_FLUX (FLUX and NFLUX). **One representation:** each component's `flux` is that component's own spectrum fᵢ(λ), up to one grey scale common to all. This is what `System` already assumes, since a common factor cancels in V. The model of the total spectrum is then F(λ) = k Σ fᵢ(λ), with k the grey scale.
   - Add `System.total_spectrum(wavel) -> Σ fᵢ(λ)`, so the same scene object predicts OI_FLUX.
   - The SPARCO habit of a grey reference star (`flux=1`) with the other components' spectra *relative* to it is the special case where the star's spectral shape is left out. Then `total_spectrum` predicts F(λ)/S_ref(λ). Compare it with NFLUX divided by a model spectrum of the reference, or give the reference its own spectrum (item 2), not both.
   - **Intrinsic against coupled flux.** `total_spectrum` is the *intrinsic* total, Σ fᵢ(λ). What a single-mode instrument measures is the flux *coupled into each telescope's fibre*: F_k(λ) = k Σ fᵢ(λ) ηₖ(xᵢ, λ), where ηₖ is that telescope's coupling at the component's position (integrated over the brightness of an extended component). The same weights enter every visibility's normalisation √(F_k F_l). With the optional primary beam of the GRAVITY review (§1), photometry and visibilities must use one coupling model. Without it, η = 1, which holds only for a scene much smaller than the fibre beam.
   - **FLUX:** one grey scale nuisance per dataset (absolute calibration and fibre injection), optionally times a low-order polynomial in λ.
   - **NFLUX:** normalised by the same continuum ranges as the data, with one helper shared with differential phase.
2. **New:** a prior on the reference spectrum instead of a fixed shape. The reference component gets its own `PowerLaw` or `BlackBody`, with its index or temperature freed under a prior from a model atmosphere. The other components' spectra are then absolute, as in item 1. No library change is needed beyond documentation. The docs must say that, without OI_FLUX, this prior *is* the systematic on every other component's colour.
3. **Calibrator spectra.** Any calibrator observed through the same optics gives F_target/F_cal(λ). Times a model spectrum of the calibrator, that is an NFLUX-quality spectrum of the target, with the injection ratio as a grey nuisance (§2.7).

### 2.3 Differential phase (VISPHI) (6a)
**Requirements on 6a's differential phase:**
1. **Model arg V(λ) directly.** Apply the *same* continuum normalisation operator as the pipeline: subtract the mean, or the mean and slope, over the continuum channels, per baseline and frame. This is a linear operator on phases, so it fits `OIData`'s existing linear-operator machinery (`phi_mat`).
2. **Do not use the photocentre approximation for resolved systems.** φ_diff ≈ −2π **u**·Δ**p**(λ) holds only when the line-emitting structure is unresolved (|Δp| ≪ λ/B). 6a's planned test "against the analytic photocentre shift" should be joined by a test in the resolved regime: a binary with a line in one star, against the exact arg V. (Apep: a 28 mas binary against λ/B ≈ 3.3 mas at 130 m, where the WC star's share of the flux triples across C IV.)
3. **Join with closure phase.** When VISPHI and T3PHI come from the same frame, VISPHI's closure (the closure of the differential phases) is the closure phase minus its own continuum mean, so it duplicates the chromatic part of T3PHI. 6a's "closure phase with VISPHI in one dataset" must not double-count it. **A closure-null projection alone is not statistical independence.** After continuum normalisation N, the VISPHI covariance is N D Nᵀ, not D. The projection must use that propagated covariance, including the cross-covariance with T3PHI, and whiten the combined spectral-and-baseline operator as one block (GRAVITY review §3c). Restricting VISPHI to the line windows does not avoid this, because T3PHI is used in those channels too. **Default:** closure phase everywhere, plus, in the line windows, the part of the continuum-normalised VISPHI that has no closure. Per frame and channel, the baseline phases are projected onto the (N − 1)-dimensional subspace of telescope-differenced phases (φᵢⱼ = aᵢ − aⱼ), which is orthogonal to the closures. That projection is one more linear operator (`phi_mat`), with its projected covariance. Modelling the joint covariance of VISPHI and T3PHI is the alternative, if the pipeline provides it.

*As built (6a PR B, 2026-10-05), departing from the requirement above in one respect:* `virgil.observables.DifferentialPhase`. N = I − L removes an offset and a slope in wavenumber (a delay) fitted over the continuum channels, per baseline and frame. With closure phases in the data, each frame is projected onto the telescope-differenced subspace, and the propagated covariance (Q ⊗ W) D (Q ⊗ W)ᵀ is whitened per frame. **The departure:** the cross-covariance with T3PHI is neglected, so VISPHI and T3PHI are whitened as separate blocks rather than as one. It is exactly zero when the baseline errors within a frame and channel are equal, and nonzero otherwise (of the order of their spread). The reason is that virgil's closure-phase covariance is itself the equal-noise model (Kammerer et al. 2020, `_closure`): it is not built from baseline phase errors, so it defines no cross-covariance to propagate, and under its assumption the cross term vanishes. Whitening T3PHI and VISPHI jointly needs the exact baseline-noise model C = T diag(s) Tᵀ (the `_closure` docstring) or a pipeline joint covariance (OI_CORR). It is **open for Ben**: accept this as the 6a default, or make the joint block a PR C item.

### 2.4 Calibration nuisances correlated across channels (Stage 6d, after 6a)
**Problem.** For most spectro-interferometers, the dominant systematics are **common to all channels of one frame and baseline**: transfer-function jitter between calibrator frames, piston and injection drifts. Diagonal error inflation (a scale, or an additive term) does not describe them. Binning channels by hand only hides them. (Apep: errors underestimated by 2–7×. Additive per-epoch terms beat multiplicative ones by ΔlogL ≈ 700, but both are diagonal. The calibrator scatter is one number per frame and baseline, yet it was added to every channel.)

This is the trigger that Stage 8's "spectral correlations" was waiting for. **Decided (2026-10-03): it becomes Stage 6d, after 6a.**

**Model.** Per block b, with one block per (frame, baseline) for V² and one per (frame, triangle) for closure phase:
- **V²:** the gain is on the **amplitude** |V|, in log space (decided; GRAVITY review §12): |V|_obs = e^{a_b} |V|_model, so V²_obs = e^{2a_b} V²_model, with a_b ~ N(0, τ²) shared by every channel of the block. Linearised for small gains, δV² ≈ 2 a_b V²_model, so the mode vector is m = 2 V²_model and a 10% width on |V| is about 20% on V². The Gaussian marginal below is therefore a **small-log-gain approximation** (τ ≲ 0.2), not exact. The GRAVITY review §4(c) generalises this rank-one block to low rank per frame: telescope-based, per baseline, and chromatic.
- **Closure phase:** φ_obs = φ_model(λ) + δ_b + noise, with offsets δ_b ~ N(0, τ_φ²) shared by every channel of the block.
- **Optionally,** a slope across the band per block, for chromatic drift.

**Marginalise analytically.** Each block's covariance is a diagonal matrix plus a rank-one term, C_b = D + τ² m mᵀ:
- for closure phase, m = 1;
- for V², m = 2 V²_model (the linearised log-gain on |V|), so C_b depends on the model.

With x = D^{−1/2} r, w = D^{−1/2} m, s = wᵀw and β = (1 + τ²s)^{−1/2}:

  r_w = x − (τ² / (q(q + 1))) w (wᵀx),  with q = √(1 + τ²s) = 1/β,
  ‖r_w‖² = rᵀC_b⁻¹r,  log det C_b = Σ log σᵢ² + log(1 + τ²s).

The coefficient τ²/(q(q + 1)) equals (1 − β)/s, but it is finite at s = 0 (a block whose model V² is zero) and has no cancellation when τ²s is small, which matters in float32.

So:
- The correlated model still produces **one whitened residual vector**, preserving the rule in `AGENTS.md`, plus a log-determinant. That log-determinant is the same kind of Σ log σ term that `noise=` already adds.
- **Which optimiser.** For closure phase (m = 1) with τ fixed, C_b doesn't depend on the parameters, so the whitened residuals are the whole likelihood and LM works. For V², m = V²_model, so log det C_b depends on the scene even with τ fixed: least squares on r_w would optimise a different objective. Use L-BFGS or NUTS on the full likelihood for `vis_gain`, and whenever τ is fitted.
- No per-block parameters are sampled; every block is integrated out exactly.
- With unprojected closure phases, the chord residual 2 sin(Δ/2)/σ makes δ enter non-linearly. Applying the rank-one whitening to the chord residuals is then an approximation, good while the offsets are small (δ ≲ 0.3 rad). Document the limit. (As built, on #174's sines rather than chords: the same limit, stated in `OIData.with_closure_offsets` and `ClosureOffsets`.)

**Interface.** The rank-one terms extend `noise=`'s per-dataset vocabulary (§2.6): `vis_gain` (τ_V) and `phi_offset` (τ_φ). They need block labels in `OIData` (§2.5). A *known* τ (from calibrator scatter) is the same code with τ fixed. OIFITS v2's `OI_CORR` table can carry such correlations from a pipeline. Reading `OI_CORR` into a block structure is a later extension.

*As built, closure offsets (Stage 6d, 2026-10-05):* `OIData.with_closure_offsets(baseline=, triangle=, modes=)` and the noise terms `phi_offset_baseline`, `phi_offset_triangle` and `phi_offset_modes`. After #174 the correlated closure phases are whitened as sines (plus a periodic penalty per closure phase); each offset mode is whitened the same way, group by group, and a frame's modes are then one block of the same rank-one whitening as the gains. The penalty rows are unchanged. Baseline offsets enter as T·e, signed by each sample's stored station pair; triangle offsets (per frame and triangle, what the PIONIER pair test measures) and supplied per-frame modes (virgil-vlti's templates) are also allowed. Not yet: three telescopes (their phases use the von Mises chord, not sines), and splitting data that carry offsets (add them after splitting).

*As built (Stage 6d, 2026-10-05):* the modes live in `OIData.gains` (`virgil.gains.GainModes`, set by `OIData.with_gains(telescope=, baseline=, chromatic=, modes=)`), and each group's width is a noise term, `vis_gain_telescope`, `vis_gain_baseline`, `vis_gain_chromatic` or `vis_gain_modes`, rather than one `vis_gain`. Blocks are the connected groups of modes, not fixed frames, so supplied modes may span frames. Rank k is whitened by k successive rank-one steps (R = I − w wᵀ/(q(q+1)) applied to the residual and to the remaining modes), which needs only scalar square roots and so keeps gradients finite where modes are degenerate; an SVD or eigendecomposition does not. Telescope and baseline gains need the station pair of each sample, now read from `STA_INDEX` into `OIData.stations`.

**Binning.** Once the correlated model exists, binning is no longer an error-model decision; only compute cost remains a reason to bin. The tutorial should say so.

### 2.5 Times, frames and epochs in `OIData` (new; also a prerequisite for orbits)
**Problem.** `read_oifits` matches each triangle to the baselines with the nearest MJD, within a fixed 1e-4 d. `OIData` keeps no time after reading. Some pipelines stamp each baseline of a frame with its own MJD, beyond that tolerance. (Apep: GRAVITY, up to 12 s.)

**Proposal.**
1. **Keep the times.** `OIData` stores `mjd` (float64, shape of `u`) per baseline sample. Closure phases inherit their time through `i_cps*`. It also stores `frame` (int, shape of `u`): which exposure a sample belongs to. Orbits need `mjd`; Stage 6d needs `frame`.
2. **Match by frame, not by exact MJD.** A triangle matches the baselines of the same `INSNAME` whose MJD lies within the row's `INT_TIME` (falling back to the current tolerance when `INT_TIME` is absent). After matching, `frame_mjd="mean"` (the default) gives each frame one time.
3. **Epoch grouping.** `OIData.epochs(gap_days=0.5)` returns integer labels for per-epoch nuisances. `split_by_epoch()` returns a list of `OIData`, for per-dataset models.
4. **Keep MJD in float64 on the host.** float32 resolves MJD ≈ 60000 to about 0.004 d. Model code receives `mjd − t_ref`, with `t_ref` a static float64.

### 2.6 A wavelength-scale nuisance per instrument (new, small)
**Problem.** Instruments and epochs have wavelength scales good to about 0.1%. That scales every angular size by the same factor, which can exceed the statistical error of a precise fit. (Apep: 0.028 mas on a 28.05 mas separation, twice its 0.013 mas statistical error.) Position angles are unaffected.

**Proposal.** `u` and `v` are stored in metres and models divide by `wavel` (`Component.model`). So a scale nuisance s only has to evaluate the model at `wavel·s`. That rescales the spatial frequencies and the spectra together, which is physically right. It is one line in `OIData.model`, or `OIData.with_wavelength_scale(s)`, plus a per-dataset term `wavel_scale` (a generic example prior is `Normal(1, 1e-3)`). With one dataset it is degenerate with every angular size, so the prior *is* the systematic. Report it.

*As built (Stage 6d, 2026-10-05):* `OIData.with_wavelength_scale(scale, offset)` evaluates models at λ′ = scale·λ + offset (metres), and the noise terms `wavel_scale` (a factor, starting at 1; GRAVITY prior `Normal(1, 2e-4)`) and `wavel_offset` (starting at 0) call it. Their priors may have either sign, unlike the error terms. A `uv_grid` is kept: it is in metres and divided by the wavelength where it is used.

**The GRAVITY default (interim, GRAVITY review §6 and §12):** λ′ = λ(1 + s) + δ, with s ~ N(0, 2×10⁻⁴), and the offset δ fixed at zero unless spectral lines constrain it. That is five times narrower than the generic example, and is under review with the GRAVITY team.

**One per-dataset nuisance vocabulary.** `noise=` (merged in PR #124) takes one dict of priors per dataset. It grows, under the same name, into the per-dataset nuisance specification:

| Term | Effect | Status |
|---|---|---|
| `vis_scale`, `phi_scale` | multiply σ | merged (PR #124) |
| `vis_error_rel`, `phi_error` | add in quadrature (relative to the **model** V², and absolute) | merged (PR #124) |
| `vis_gain_<group>` | low-rank correlated blocks on log \|V\|, marginalised (§2.4; GRAVITY review §4c) | Stage 6d |
| `phi_offset` | baseline-based closure offsets, **off by default**, only if calibrators need them (GRAVITY review §13) | Stage 6d, optional |
| `wavel_scale` (+ `wavel_offset`) | evaluate at λ(1 + s) + δ; GRAVITY default s ~ N(0, 2×10⁻⁴), δ = 0 | Stage 6d |
| `flux_scale` (+ `flux_poly`) | OI_FLUX calibration (§2.2) | 6a PR B: marginalised analytically, not a `noise=` term (`OIData.with_flux_scale`) |

**Reconciling with 6a's error floors.** Floors (PMOIRED's `min error`, `min relative error`) are *fixed* changes to the data and belong on `OIData` (`with_error_floor`, like `with_error_scale`). `noise=` is the *fitted* counterpart, in the likelihood. Two differences must be explicit:
- `vis_error_rel` multiplies the **model** V², while PMOIRED's relative floor multiplies the **data**. The model version is the right one for a fitted term, because it does not reward the model for low data points.
- A floor is a `max(σ, floor)`, whereas `noise=` adds in quadrature.

Both go through one function (`likelihood.inflated_errors`, merged in PR #124 without the `where=` and `combine=` arguments yet), with `where="data" | "model"` and `combine="max" | "quadrature"`, so that the two cannot drift apart.

### 2.7 The in-field (dual-field) calibrator recipe (done in virgil-vlti, not the core)
In dual-field instruments (GRAVITY's dual-field mode), archival data often include frames on the fringe-tracking star, interleaved with the target. When that star is unresolved, it is a calibrator observed minutes apart, in the same mode, with no separate calibration needed. GRAVITY-specific tools were "not planned" in [`pmoired_parity.md`](pmoired_parity.md) when this was written, so it was planned as a recipe, not a reader. Those tools now belong to [virgil-vlti](https://github.com/benjaminpope/virgil-vlti), which implements the recipe:
1. Classify frames by `SOBJ` offset: ≈ 0 means the fringe-tracking star (the calibrator); otherwise the target.
2. Per baseline and channel, interpolate the calibrator's V² transfer function in time to each target frame. Average its closure phases as unit vectors and subtract them.
3. Record the calibrator scatter as the **known τ** of §2.4, not as a per-channel error.
4. Unify MJD per frame (§2.5) before writing.
5. Optionally, produce F_target/F_cal(λ) for NFLUX (§2.2).

**Done (2026-10-04)** in [virgil-vlti](https://github.com/benjaminpope/virgil-vlti), as `virgil-vlti-calibrate --dual-field`, generalised from Apep's `scripts/calibrate.py`. Its known τ is a leave-one-out prediction error, as the GRAVITY review (§7, now in virgil-vlti) recommends. **Caveat for the docs:** the calibrator must be unresolved, and close enough on the sky that the transfer function is shared.

### 2.8 Smearing (6a)
Once 6a's smearing exists, its docs should give the rule for when it matters: the number of fringes across the scene's extent, B·θ/λ, compared with the resolving power R. They should also include a check on real data near the limit. (Apep: R ≈ 130–240 after binning, with up to 8 fringes across the scene on 130 m baselines. Refitting with smearing on should move the separation by ≲ 0.01 mas.)

## 3. Mapping and effort

| # | Item | Goes in | Effort (agent h) | Depends on |
|---|---|---|---|---|
| 2.1 | `Sum` and `Nodes(outside=0.0)`; positivity on the total; `Tabulated` → `Nodes`; one reference-flux rule | 6a (as planned) | +0.5 | — |
| 2.2a | OI_FLUX/NFLUX, `System.total_spectrum`, `flux_scale` | 6a (as planned) | +0.5 for `total_spectrum` | 2.1 |
| 2.2b | A prior on the reference spectrum; docs on the degeneracy | 6a | 1 | 2.2a |
| 2.3 | VISPHI with pipeline-matched normalisation; resolved-regime test; no double counting with T3 | 6a (as planned, plus tests) | +1 | 2.5 |
| 2.4 | Rank-one correlated nuisances (`vis_gain`, `phi_offset`), analytic marginalisation | **Stage 6d** (decided), after 6a | 5–7 | 6a; 2.5 (`frame`) |
| 2.5 | `mjd` and `frame` in `OIData`; INT_TIME matching; `epochs()` | 6a.0 | 2–3 | — |
| 2.6 | `wavel_scale`; the `noise=` vocabulary; `with_error_floor` sharing `inflated_errors` | 6a (`with_error_floor`) and 6d (`wavel_scale`) | 1–2 | `Nodes` (6a) |
| 2.7 | Dual-field calibration | done in virgil-vlti | — | — |
| 2.8 | Smearing rule and a real-data check | 6a's docs | 0.5 | 6a smearing |

**Order.**
1. 2.5 (small; it unblocks 2.3, 2.4 and the orbits).
2. 6a's spectra and error floors.
3. **Then** reconcile the merged `apep-gravity` commits (PR #124) with 6a's `Nodes` and floors.
4. Stage 6d (2.4) and 2.6.
5. 2.7 is done, in virgil-vlti.

The 6a agent needs only the "+" items folded into its plan.

## 4. Decisions and open questions

### Decided (Ben, 2026-10-03)
1. **Correlated channel nuisances (2.4) are their own stage, 6d, after 6a.**
2. **The `apep-gravity` library commits** were to merge into `imaging` after 6a's spectra; they were instead merged in PR #124, with `Tabulated` provisional. Those are `0688f34` (`EllipticalGaussian`, `Tabulated`, `noise=`), `c4ef79d` (anisotropic `GaussianField`), `3a6c682` (`GaussianArc`) and `37bddad` (finite rim gradients at u = v = 0). That PR replaces `Tabulated` with 6a's `Nodes`, and routes `noise=` through the same function as 6a's error floors (2.1, 2.6).
3. **The dual-field recipe** lives in [virgil-vlti](https://github.com/benjaminpope/virgil-vlti), not in a `virgil.gravity` module (decided 2026-10-04).

### Still open (defaults in force until Ben says otherwise)
1. **`noise=` grows into the general per-dataset nuisance argument** (gains, offsets, wavelength scale, flux scale), keeping its name (default), rather than a separate `nuisance=` argument.
2. **Reference flux for every spectrum:** the value at `wavel0` (default), not the node mean.
3. **Closure phase everywhere,** plus the closure-free projection of continuum-normalised VISPHI in the line windows (default; §2.3).
4. **Real anchor binaries** for the position-angle round trips are not chosen yet; see the reminder in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md) §7.
