# Bispectrum likelihood (0.4.3)

Status: design, for virgil#578 (milestone 0.4.3). Nothing here is implemented yet.

## Why

The bispectrum B = V₁₂ V₂₃ V₃₁ is the basic observable of phase-closing interferometry. virgil currently fits only its phase, T3PHI, and treats T3AMP, when it is used, as an unrelated block. When T3AMP is calibrated, this throws away information. The same fix also handles the closure-phase wrapping problem without any approximation.

## What main does now, read as a bispectrum

Write the model–data residual in the frame aligned with the data's phase:

    r = (B_m − B_d) · e^{−iφ_d}

- For unit amplitudes (|B_m| = |B_d| = 1), r = e^{iΔ} − 1 = (cos Δ − 1) + i sin Δ, where Δ = φ_m − φ_d.
- **Tangential part, Im r = sin Δ.** Main whitens it with the Kammerer et al. (2020) closure covariance (`_closure.py`).
- **Radial part, Re r = −2 sin²(Δ/2).** This is main's diagonal "periodic penalty". It corresponds to a radial σ equal to σ_φ.

So the current closure-phase likelihood is the **phasor (unit-amplitude) bispectrum**: the convex approximation of Meimon, Mugnier & Le Besnerais (2005). This is why the penalty must not be whitened with the closure covariance, which was Argon's A1 on #570, dropped:
- with phase-only data, the radial component is second order in the noise (about δφ²/2), non-Gaussian, and has no covariance C;
- near the solution it adds only O(Δ⁴) to χ², and its only job is to remove the false minimum at Δ = π, where sin Δ = 0.

## Proposed: complex-bispectrum likelihood

**Data.** B_d = T3AMP · e^{i·T3PHI}, from the OIFITS T3 table, with T3AMPERR and T3PHIERR.

**Residual.** As above, r = (B_m − B_d) · e^{−iφ_d}.
- The rotation e^{−iφ_d} is fixed by the data.
- So r is **linear in the model's complex bispectrum**: there are no wraps, no branch cuts, and no false minimum at π.
- Any fixed linear whitening operator built at load can be applied to r and gives a smooth likelihood. This is the "compute the operator once and apply it identically to data and model" approach: it is valid on complex bispectra, though not on wrapped T3PHI.

**Noise.** At high SNR, an elliptical Gaussian in the data-aligned frame (Buscher 1994; BSMEM):

| Component | σ | Correlation between triangles that share baselines |
|---|---|---|
| radial, Re r | σ_T3AMP | log-amplitude closure: incidence \|T\| (+1,+1,+1), R_amp = \|T\|\|T\|ᵀ/3 |
| tangential, Im r | \|B_d\|·σ_φ | phase closure: incidence T (+1,+1,−1), R_φ = T Tᵀ/3 (Kammerer 2020) |

- At first order, circular per-baseline noise makes the radial and tangential parts uncorrelated, so the two blocks whiten separately.
- Both correlation models are exact only for equal per-baseline errors within a group. OIFITS carries no per-baseline errors, so this is the same approximation the closure phases already make. Document it the same way.
- Build both Cholesky factors once at load, generalizing `ClosureNoise` (for example into a `BispectrumNoise` that handles both blocks).

**Exact reduction.** Phasor mode (|B| set to 1, radial σ := σ_φ, radial block diagonal) must reproduce the current closure-phase likelihood to round-off. Test this. It keeps the default and the new mode in one code path.

## When to use which (advice for users and pipelines)

**Closure phases only, the default** (phasor mode as now), when:
- T3AMP is absent or not calibrated. This is the usual case for ground-based long-baseline data, where seeing and calibrator mismatch dominate the amplitude transfer.
- You only want the robust, calibration-independent part of the data, for example a detection or astrometry of a companion where amplitudes add little.

**Full bispectrum (opt-in)** when T3AMP is calibrated and trustworthy:
- **Imaging.** The bispectrum amplitude constrains the flux distribution, as in BSMEM. Prefer it when V² is sparse or noisy.
- **Closure phases near ±π or with σ_φ ≳ 0.5 rad**, such as faint triangles, resolved or point-symmetric structure, or fits started far from the solution. Here the radial term is a real measurement rather than a penalty.
- **Aperture masking and kernel-phase data** where the reduction forms the complex bispectrum directly and calibrates it against a PSF star.
- Space-based data, where amplitude calibration is stable.

**Avoid the bispectrum, or model the correlation, when:**
- **You also fit V² from the same frames.** |B| = |V₁||V₂||V₃|, so T3AMP and V² share amplitude noise. Fitting both as independent counts the amplitude information twice. Either fit V² plus closure phases (the default), or fit bispectra plus V² with the cross-covariance modelled. The default is to refuse the combination unless the user supplies a noise model for it.
- **SNR is low.** Amplitude and phase errors of a low-SNR bispectrum are non-Gaussian, and |B| is noise-biased. The averaged *complex* bispectrum is unbiased and close to complex Gaussian even when the per-frame SNR is low, by the central limit over frames. But OIFITS stores only amplitude/phase errors, so converting back assumes the elliptical model. The docs should state the regime: the method is valid at high SNR, and at low SNR it needs Re/Im errors from the reduction. Supporting Re/Im errors if a reducer provides them is a later item.

## Interactions to resolve in the implementation PR

- **Closure-phase offsets and antenna gains.** Offsets enter the tangential block as before. Amplitude gains enter the radial block through |T|. Each needs a marginalization consistent with `gains.whiten`.
- **Residual counts.** `OIData.n_residuals` and the residual vector's layout gain one radial and one tangential entry per triangle. LM/GN least-squares form is preserved, because r is linear in B_m.
- **Pipelines.** The data check (`pipeline/_checks.py:343` already mentions a missing bispectrum likelihood) should suggest the bispectrum when T3AMP is present and σ_φ is large. It never switches automatically.
- **Mixed data.** Datasets that mix triangles with and without T3AMP: use the bispectrum per triangle where T3AMP is present and phasor mode elsewhere.

## Acceptance (pre-registered)

1. Provenance entries, ADS-verified:
   - Buscher 1994 (IAU Symp. 158);
   - Meimon, Mugnier & Le Besnerais 2005 (JOSA A 22, 2348);
   - Thiébaut & Young 2017 (JOSA A 34, 904), a review of noise models;
   - Kammerer et al. 2020 (A&A 644, A110);
   - BSMEM (Baron & Young 2008, SPIE 7013).
2. An independent dense test: χ² against an explicit complex covariance built by hand.
3. Continuity: the likelihood and its gradient are continuous across Δ = ±π.
4. Reduction: phasor mode reproduces main's closure-phase likelihood to round-off.
5. virgil-validation: coverage of 68% and 95% intervals in a brute-force bispectrum simulation at high and moderate SNR, plus a comparison with BSMEM on a shared synthetic dataset. Thresholds are fixed before the run.
6. Docs: a short "closure phases or bispectra?" section in the user docs, based on the section above, including the V² double-counting and low-SNR caveats.

The default stays closure phases (phasor mode); the bispectrum is opt-in until the vv checks pass.
