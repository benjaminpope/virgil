# Radial profiles, Hankel transforms and disk-aware imaging

Status: scoping note (2026-10-14). It extends Stage 8 of [`imaging_plan.md`](imaging_plan.md) and the `RadialProfile` / `Projected` items in [`pmoired_parity.md`](pmoired_parity.md). Science targets appear only as worked examples.

## 1. Prior art

| | frank (Jennings et al. 2020, arXiv:2005.07709) | vis-r (Kennedy 2025, arXiv:2502.17619) | FRAP (Yoshida et al. 2026, arXiv:2607.12618) |
|---|---|---|---|
| Profile | Non-parametric I(r) on N Fourier–Bessel nodes | Parametric I(r): Gaussian rings, power laws, erf edges, Gaussian gaps | GP latents for physical quantities Σ_d(r), T(r), a_max(r), mapped to I_ν(r) by two-stream radiative transfer |
| Transform | DHT: V = H(q) I, with nodes at the zeros of J0 and R_out fixed | Same DHT, forward only | Same DHT, times an ALMA primary-beam factor |
| Prior | Gaussian, diagonal in DHT space with power spectrum p_k. p is found by an empirical-Bayes fixed point (inverse-gamma plus a smoothness hyperprior). | None beyond parameter bounds | RBF kernel in r. The length scale is fixed by 10-fold cross-validation, and the latents are whitened (g = μ + L z). |
| Posterior | Closed form given p: D = (HᵀN⁻¹H + S⁻¹)⁻¹. Optional NNLS or log-normal MAP. | emcee or Stan HMC | numpyro NUTS, using JAX in x64 |
| Geometry | inc, PA and centre fixed beforehand; later versions fit them non-parametrically | inc and PA fitted; Gaussian scale height applied as a damping in Fourier space | inc and PA fixed beforehand (from protomidpy) |
| Data | ALMA complex visibilities, real part only | ALMA, binned in uv | ALMA in several bands, binned and Taylor-compressed |
| Code | `discsim/frank`: LGPL-3, numpy and scipy | `drgmk/vis-r`: MIT, numpy, emcee and Stan | `tomyoshida/frap`: MIT, JAX and numpyro; the Bessel zeros and J0 matrix are built with scipy |

The papers' shared core is a discrete Hankel transform (DHT) on nodes at the zeros of J0. A DHT is a weighted stack of thin rings. That the papers use ALMA data does not matter, since the maths is the same for any interferometer. What they lack is V² and closure phases, and any non-axisymmetric structure. This plan adds both.

**No dependencies.** The ideas are simple, so we reimplement them on what virgil already has, and cite the papers in docstrings:
- `jaxbessel.bessel_jn(M, x)` returns every order 0…M in one call, matches scipy to about 1e-14, and has finite gradients everywhere.
- `ModulatedGaussianRim` and `cvis_radial_dirac_delta_modulated` fix our conventions for the azimuthal-harmonic sign and handedness.
- `System`, `Component`, `fit`, `numpyro_model`, `imaging.log_evidence` and `detection.py` are reused as they are.

## 2. What changes for optical/IR and masking data
1. **Nonlinear observables.** V² and closure phases have no closed-form posterior, so the methods are MAP (`fit`), Laplace and NUTS. frank's linear algebra is not needed.
2. **Odd versus even harmonics.** A centred disk with only even m has real visibilities, so its closure phases are zero. Closure phases see only odd m, offsets and flaring. Even m (for example m = 2) is constrained by V² alone, and is partly degenerate with inc and PA.
3. **Few baselines.** Inclination only lowers the face-on frequency, √((u′cos i)² + v′²) ≤ q, so the bandlimit is set by q_max over the radial range the component covers. No binning is needed. Non-parametric profiles will be strongly prior-dominated, so every example shows the prior beside the posterior.
4. **Geometry is fitted jointly.** inc, PA and flaring are parameters of the component.
5. **Flux and position.** The disk has unit flux, and its weight is `flux` within a `System`. Its centre is the usual `dra`/`ddec` of `Component`, measured from the phase centre (the array centre), default 0, and fitted or fixed like any other component's. An m = 1 term is nearly degenerate with a free offset, so fitting both calls for informative priors on at least one.


## 3. Architecture: one component, `MultipoleDisk`

```
MultipoleDisk(profile, harmonics=None, height=None, spectral_index=None, blur_fwhm=None,
          inc, pa, flux, dra, ddec, n_r)
  profile         I0(r), positive, with its own support and quadrature rule:
                  ThinRing(radius) | GaussianRing | RadialPowerLaw (soft edges) | ExpTail | Gap(...) | RadialSum(...) | RadialGP
  harmonics       Harmonics(degree=M, form="positive" | "linear", prior=None | MultipoleKernel(...))
                  | VonMises(kappa, theta0, degree=M)
  height          h(r) = h_ref (r/r_ref)^ψ, Cone(alpha, tip), or None
  spectral_index  α(r): I(r, λ) = I0(r) (λ/λ0)^α(r); a constant, a parametric function or a RadialGPFunction
  blur_fwhm       Gaussian smoothing applied as a Fourier-space envelope (exact and cheap)
```

This mirrors `Image(GaussianField(...))`: one component built from small interchangeable parts. `MultipoleDisk` is a `Component`, so it gets `flux` (including SPARCO spectra from `spectra.py`), `dra`/`ddec`, `System` mixing and `render` for free.

### 3.1 Intensity: positive (Fejér–Riesz) or linear harmonics
`degree=M` is always the degree of the **image**: its harmonics run exactly from 0 to M. Two forms are offered.

- **`form="positive"` (the default for new models).**
  I(r, θ) = I0(r) · |h(e^{iθ})|² / Σ_j |h_j|², with h(z) = 1 + h_1 z + … + h_M z^M and complex h_j(r).
  - **Every positive image of degree M is reachable.** The Fejér–Riesz theorem says every non-negative trigonometric polynomial of degree M is |h|² for such an h.
  - **Exact degree.** |h|² has harmonics exactly 0…M (an O(M²) autocorrelation per node), with no FFT and no truncation.
  - **I0 is the azimuthal mean.** The θ-mean of |h|² is Σ|h_j|² by Parseval, so the normalisation divides it out.
- **`form="linear"` (the legacy form).** I = I0 · [1 + Σ c_m cos mθ + s_m sin mθ]. It is exactly linear in (c, s) and reproduces `ModulatedGaussianRim` and `TruncatedCone`, so the subsumed classes use it. Positivity needs 1 + f ≥ 0, and `min_intensity()` reports violations.

**Multimodality, and how the parametrization removes it** (raised by Ben, 2026-10-14; checked numerically). The scratch script `fr_check.py` lives in the review session; a test in P3 reproduces it.
- **A real square [1 + f]² is unique, but it is not enough.**
  - [1+f]²/N_f = [1+g]²/N_g gives (1+g)² = λ²(1+f)² for trigonometric polynomials. These are analytic, so 1 + g = ±λ(1 + f), and the constant term forces g = f.
  - However, the image degree can only be even.
  - It cannot represent a pure lopsided ring: the best real-square fit to 1 + 0.9 cos θ leaves an 8 % rms error, because the forced m = 2 term is c²/2. Odd m is exactly what closure phases see, so this form is rejected.
- **A complex |h|² is fully expressive but multimodal.**
  - Reflecting any root z_k of h to 1/z̄_k leaves |h|² unchanged up to scale. That gives up to 2^M coefficient vectors for the same image: the spectral-factorisation ambiguity of phase retrieval.
  - The likelihood is identical on each copy, and they join along folds where the image touches zero.
  - **Consequences.** MAP images would be unaffected, but NUTS mixing and Laplace accuracy would degrade. Laplace evidences would also undercount by up to 2^M, which biases comparisons across M.
- **The fix: reflection (Schur) coefficients.**
  - Parametrise h by complex reflection coefficients k_1…k_M with |k_j| < 1, built by the Levinson step-up recursion h^{(j)}_i = h^{(j−1)}_i + k_j conj(h^{(j−1)}_{j−i}).
  - This is a bijection onto the minimum-phase factors, whose roots all lie outside the unit circle. That is the unique canonical representative of each positive image (Schur–Cohn; as for stable AR models, Barndorff-Nielsen & Schou 1973).
  - Each k_j comes from an unconstrained Cartesian latent w_j ∈ ℂ through k = w tanh|w| / |w|, which is smooth at 0. There is no funnel, no label switching, no polar singularity and no constraint.
  - **Numerical check.** Random w at M = 1–6 always gave min-phase h. Every root-flipped twin of a k = (0.6, 0.6, 0.6) model gives the same image, but each twin's step-down coefficients have some |k_j| > 1, outside the disk, so the parametrization excludes it (for the twin that flips one real-axis root, |k| = (1.64, 1.51, 0.79); the other single and double flips give e.g. (2.08, 3.44, 0.96) and (1.64, 1.51, 1.26); flipping all three gives (0.60, 0.60, 1.67)). An earlier draft quoted (2.25, 1.21, 0.71), which no twin reproduces. The degree-1 form fits 1 + 0.9 cos θ exactly (|k| = 0.63).
- **What the reflection coefficients are.**
  - **Step-up recursion.** It builds h one degree at a time, h^{(j)}(z) = h^{(j−1)}(z) + k_j z^j h̄^{(j−1)}(1/z), using the reversed conjugate of the previous polynomial.
  - **Schur–Cohn.** The final h has every root outside the unit circle if and only if every |k_j| < 1. The unit-disk box of k's is therefore a one-to-one chart of the minimum-phase polynomials. The step-down recursion inverts it.
  - **Origin.** This is the Levinson–Durbin algorithm of linear prediction, where the k's are the partial autocorrelations. The name comes from lattice filters and layered-medium (acoustic-tube, seismic) models, where k_j is literally the reflection coefficient at the j-th interface.
  - **Why it applies here.** Our azimuthal profile |h(e^{iθ})|² is mathematically the power spectrum of a moving-average process in θ, and positivity of spectra is the classical problem these tools solve.
  - **Unconstrained version.** Using these coefficients to parametrise stationary autoregressive models without constraints, with a tanh map from the real line, is standard practice (Barndorff-Nielsen & Schou 1973; Jones 1980). We use the complex version.
  - **References.** Dumitrescu 2007, *Positive Trigonometric Polynomials and Signal Processing Applications* (Springer), covers Fejér–Riesz, spectral factorisation and the Schur–Levinson recursion in one place. Barndorff-Nielsen & Schou 1973 (J. Multivariate Anal. 3, 408) gives the unconstrained reflection-coefficient parametrization. Check both on ADS or DOI before adding them.
- **Numerics.** k = w·tanh|w|/|w| is computed in a form that is finite at w = 0, with value and gradient taken from the series tanh(x)/x ≈ 1 − x²/3 below a threshold (a safe `where`).
- **Weak asymmetry is near-linear.** I ≈ I0(1 + 2 Re Σ h_m e^{imθ}), and for small k the first-order coefficients are h_m ≈ k_m.
- **What is reported.** The image harmonics I_m(r) (amplitude and phase per m) are the derived quantities. The reflection coefficients are an internal parametrization, not something to interpret.
- **Priors.** A `MultipoleKernel` or `RadialGP` prior acts on the latents w_j(r). Its variance falling with j plays the role of falling with m, because k_j first enters h at order j.

**Why not log** (decided 2026-10-14; an earlier draft used exp(f)). The reasons below compare linear with log. The positive form keeps every one of them except exact linearity in the coefficients, and adds positivity:
- **Degree M means what it says.** The image, and so the visibilities, contain harmonics up to M exactly. In log form, exp(·) spreads power to every order, which forces a second cutoff m_out > M, a polar-grid FFT and a truncation error.
- **The model is linear in (c, s)** for fixed I0 and geometry. Gauss–Newton is well conditioned, the Laplace approximation is close to exact in those directions, and Gaussian priors on (c, s) stay Gaussian. The log form makes the coefficient posterior non-Gaussian.
- **It is cheaper.** A visibility uses only the orders 0…M, with no FFT, and is analytic for `ThinRing`.
- **I0 is the azimuthal mean.** In log form, the modulation leaks into the m = 0 term, so the profile is no longer the mean intensity.
- **It subsumes the existing rims exactly.** `ModulatedGaussianRim` and `TruncatedCone` use this form.
- **The linear form's cost is positivity.** The positive form removes that cost, so the `Positivity` barrier regulariser of an earlier draft is dropped. A log form stays deferred.

**Coefficients are Cartesian, (c_m, s_m), never amplitude and phase.** Writing a cos m(θ − φ) causes two problems:
- **For a sampler.** As a → 0, φ becomes unconstrained, giving a funnel whose narrow end spans every φ.
- **For an optimiser.** φ wraps at 2π, so the posterior is multimodal.

In (c, s) the posterior is a smooth blob around the origin. A Gaussian prior on (c, s) gives a uniform prior on angle, consistent with the invariant-prior rule. Amplitude and phase are derived outputs. Each c_m or s_m is a constant, a parametric function of r, or a `RadialGP`.

**Converting between conventions: `virgil.harmonics` helpers.** Every form maps to and from the image harmonics I_m(r) (complex, m = 0…M) by expanding the expression and equating coefficients.
- **Linear and legacy forms.** `linear ↔ image` is exact both ways: c_m = 2 Re I_m / I_0 and s_m = −2 Im I_m / I_0. The same holds for the legacy amplitude/phase pairs (`az_amps`/`az_pas`, `az_amps`/`az_phis`), with each class's azimuth origin and sense converted to `MultipoleDisk`'s.
- **`positive → image`** is exact: the autocorrelation of h.
- **`image → positive`** is spectral factorisation.
  - **Method.** Take the 2M roots of z^M Σ_m I_m z^m, keep the M outside the unit circle to form h, then step down to k and invert the tanh to get w. It runs in numpy and is not differentiated; it is used for conversion and warm starts.
  - **Failure modes.** It raises an error if the image is negative somewhere. Roots on the circle (zero intensity) are clipped to |k| = 1 − ε, with a warning.
- **`VonMises → image`** truncates at the requested degree and reports the dropped power.
- **API.** `to_image_harmonics(harmonics)` and `from_image_harmonics(I_m, form=...)`, plus a `Harmonics.convert(form=...)` convenience.
- **Uses.** Warm-starting a positive-form fit from a linear or parametric one; reporting every model in one convention; checking subsumed classes; validation.
- **Tests.** Round trips form → image → form to rounding; root-flip twins map to the same canonical w.

**`VonMises(kappa, theta0, degree=M)`.** The arc shape exp(κ cos(θ − θ0)) has analytic linear harmonics, c_m + i s_m ∝ I_m(κ)/I_0(κ) · e^{−imθ0}, where I_m is the modified Bessel function. They are computed by a small FFT of exp(κ cos θ_j), so no new special function is needed. The truncation at M is a user choice. `truncation_power()` reports the dropped fraction, and narrow arcs (large κ) need a larger M.

### 3.2 Transform
1. **Radial nodes: the profile owns the quadrature rule.**
   - **Default.** Gauss–Legendre on [r_in, r_out], with the weight r dr folded in.
   - **`ExpTail`.** A Gauss–Laguerre rule for exponential tails integrates the whole tail exactly. `TruncatedCone`'s midpoint rule drops the last 0.7 % past 5 lengths.
   - **Why it matters.** Both rules converge spectrally, not at second order like the midpoint rule, so a narrow ring needs only a few nodes. n_r is a user-set static integer. `MultipoleDisk.suggest_n_r(data)` gives a starting value and is never applied silently.
   - **Why not J0-zero nodes.** The order-m Hankel transforms with J_m are exactly what we compute. Only the quadrature rule is at issue. J0-zero nodes give an exact discrete transform only for order 0, and the exact rule for order m uses the zeros of J_m, a different grid for each m. All m share one radial grid, so no Bessel-zero set is special. One test checks the accuracy for every m.
2. **Visibilities.**
   V(q, φ) = 2π Σ_{|m|≤M} (−i)^m e^{imφ} Σ_k w_k I_m(r_k) J_m(2π q r_k),
   with I_{±m} = I0 (c_m ∓ i s_m)/2 for the linear form, and I_m = I0 Σ_j h_{j+m} h̄_j / Σ_j |h_j|² for the positive form (I_{−m} = Ī_m). Only m ≥ 0 is computed.
   - J_m comes from one call to `bessel_jn(M, ·)`.
   - `ThinRing` skips quadrature: the result is J_m(2π q radius).
   - `blur_fwhm` multiplies by exp(−(π fwhm q)²/(4 ln 2)).
   - V(0) = 1 is enforced by the m = 0 sum.
3. **Height (flaring and cones).** The ring at r_k is offset by h(r_k) along the disk normal. On the sky that is a shift of h sin i along the projected minor axis, applied as a phase factor per node in the sky frame after the stretch. This is why the disk owns inc and PA.
   - **Flared surface.** h = h_ref (r/r_ref)^ψ.
   - **Cone.** h = r cot α − tip, with the normal along the cone's axis, so inc = 90° − tilt. That is exactly `TruncatedCone`'s geometry: rings perpendicular to an axis, with centres along it.
   - **Stated approximation.** The emitting surface is thin and nothing is occulted. It is measured against `FlaredDisk*`, not assumed.
4. **Chromatic profile.** When `spectral_index` is set, I0(r, λ) = I0(r) (λ/λ0)^{α(r)}.
   - **Evaluation.** The profile is evaluated per wavelength at the nodes: n_r × n_λ values, which is negligible.
   - **Normalisation.** The visibility is normalised per wavelength.
   - **Flux.** `_weight(λ)` returns ∫ I0(r, λ), so the component's spectrum is computed from the profile rather than set by `flux`, which becomes the flux at λ0.
   - **Harmonics are grey.** They share α(r).
   - **Flux is a scalar.** With `spectral_index` set, `flux` must be a scalar at λ0; passing a `spectra.py` spectrum as well raises an error. Point sources and other components keep their own spectra as usual.
   - **What this enables.** A single non-parametric disk can have rings of different colours, which the tutorial needs (§7). It is also a temperature gradient along a dust plume.

### 3.3 Performance
- **Cost.** N_vis × n_r × (M + 1) Bessel evaluations plus a contraction. A pixel `Image` of the same disk costs N_vis × n_pix, roughly 100× more at Nyquist.
- **Spectral quadrature.** Gauss–Legendre and Gauss–Laguerre per profile need far fewer nodes than midpoint ring stacks, and narrow rings cost about 10 nodes each.
- **Memory.** Above a threshold, chunk over visibilities with `jax.lax.map(batch_size=...)`.
- **Real arithmetic.** Even m feeds only the real part and odd m only the imaginary part, so the sums stay real until the end.
- **Hoisted Bessel matrix for fixed geometry.** When inc, PA and height are fixed, for example after a parametric fit, the Bessel matrix depends only on the data and the nodes. A non-parametric refit of the profile is then a matrix–vector product per call. This is an explicit `MultipoleDisk.freeze_geometry(data)` returning a model with the matrix cached, never automatic. Its first consumer is the dust-plume imaging in §8.
- **Static and traced fields.**
  - **Static:** n_r and M (they set shapes) and the structure of the parts.
  - **Traced, not differentiated:** the node and weight arrays.
  - **Recompilation:** changing n_r recompiles; changing values never does (AGENTS.md: no constants baked into the HLO).

### 3.4 Priors
- **`RadialGP`.** Latents at the r_k, whitened as g = μ + L z with z ~ N(0, I), and a dense Matérn kernel in r (as in FRAP). The profile is exp(g), so it is positive. The same class supplies c_m(r), s_m(r) and α(r), taking g directly without the exp. σ and ℓ take bounded log-uniform priors, so the Laplace evidence is proper. With n_r ≤ 100 the Cholesky is negligible.
- **`MultipoleKernel(ℓ_r, ℓ_θ)`.** A prior over the harmonic coefficients.
  - **Structure.** Each m's (c_m, s_m) profiles are `RadialGP`s with length ℓ_r. Their variance falls with m as the spectral density of a periodic Matérn kernel in θ, roughly S(m) ∝ (1 + (m ℓ_θ / r̄)²)^(−ν−½).
  - **Effect.** A long ℓ_θ favours rings and arcs without imposing them. It is diagonal in m, so it costs nothing extra.
- **`MultipoleField` for `Image`.** The same prior on pixels, as a residual field aligned with the disk. It is evaluated on a polar grid and interpolated onto the pixels in the disk's frame. Its kernels are `RadialGP`'s (radius) and `MultipoleKernel`'s spectral form (azimuth), one shared definition; its geometry can be tied to a sibling disk (`geometry_from=`, resolved by `System` at evaluation).

## 4. Composition and interactions
- **Composition contract.** `System._mix` sums unit-flux visibilities weighted by `flux`. `MultipoleDisk` must therefore:
  - return V(0) = 1;
  - implement `_centred_image` by polar evaluation, so that `render`, `Image.from_model` and the DFT cross-check tests work. Node-valued profiles (`RadialGP`) are linearly interpolated in r for rendering only;
  - match virgil's existing Fourier sign and position-angle conventions (`offset_phase`, `ModulatedGaussianRim` and `TruncatedCone` tests). A sign test is part of P1;
  - rely only on `model`. The default `model_on_grid` fallback is fine.
- **Over-resolved flux and `Resolved`.** Disk emission on scales larger than about 1/q_min shows up in the data only as a uniform loss of visibility, which is exactly what `Resolved` models. So the data constrain the sum of the two, and the split between them is set by their priors (for example, a smooth `RadialGP` that decays to R_out versus a free `Resolved` fraction). Both may be included. The rules:
  - report and plot the total over-resolved flux, never either part alone;
  - warn when the profile has substantial flux near R_out.
- **Primary beam.** A disk wider than a single-mode fibre's field of view needs the Stage 6a primary beam. The beam multiplies the image, so it is not a composable sum. It can be applied to `MultipoleDisk` pointwise on the polar grid before the transform, so that is the place to hook it in. Not in scope until a dataset needs it.
- **Bandwidth smearing** for wide disks remains the deferred image-reconstruction item.
- **`Projected`** stays a separate Stage 8 item for generic shapes. It has no flux option, because a uv stretch preserves V(0) = 1.


## 5. Subsuming the existing disk classes
- **One shared engine.** `MultipoleDisk` and every subsumed class call one engine, `multipole_disk_cvis(...)`.
- **Names and paths are kept.** An old class keeps its public name and parameter names, so prior dicts keyed by path still work. Only its `_centred_cvis` changes, to call the engine.
- **Old code goes after equality tests.** The old private visibility code is deleted once the tests pass.
- **Deprecation rule.** A class is deprecated only if it is reproducible as a special case of the general one, or replaced by a decided change of shape flagged in the changelog (`GaussianArc`, see its row). `FlaredDisk*` is neither (see its row), so it is exempt. At the next minor release, each other subsumed class gets a deprecation note and a changelog migration line, and it is removed one release later.

| Class | Engine form | Test |
|---|---|---|
| `ModulatedGaussianRim` | `ThinRing` + `Harmonics` + `blur_fwhm`. The legacy amplitude/phase parameters are converted to (c, s) inside the wrapper. | Exact to rounding |
| `TruncatedCone` | `ExpTail` profile in slant distance + `Cone` height + `Harmonics`, with the legacy `az_amps`/`az_pas` converted | Agreement with the old class at n_rings → ∞ (the midpoint error is about 1/n²). The new rule should reach 1e-5 with about 16–24 nodes, against 256 midpoint rings. |
| `GaussianArc` | `GaussianRing` + `VonMises` (decided) | A notebook check of old against new shape before switching. The changelog flags the change of shape, and `GaussianArc` raises a `FutureWarning` for one release first. |
| `FlaredDisk*` | **Not subsumed, not deprecated.** Kept as is. | Verified in P6 (PR #340): Blakely+2024 Eq. 3 (sin i, z at the pixel mid-plane radius, z² added, no brightness Jacobian) is an approximation. Only a thin-surface version (tan i, z at the surface point's radius, no z², brightness Jacobian) equals `MultipoleDisk` + `PowerLawHeight(h0 = −z)`, to 1.5e-9. The flat case (aspect = 0) matches to about 1e-9. |
| Per-class inc/pa (`EllipticalGaussian`) | `Projected` (Stage 8) | Separate PR |

Sign convention for the `FlaredDisk*` row: positive h shifts towards pa + 90°. A top surface lifted towards the observer shifts the ring to the far side, so h0 < 0.

Stellar disks (`UniformDisk`, limb-darkened disks, `GaussianDisk`) are out of scope. They are closed forms on the core star path and stay as they are; they serve as the analytic m = 0 tests.

## 6. Stages (ordered so the colliding-wind dust-plume imaging benefits first)
| Stage | Content | Size |
|---|---|---|
| P1 | `multipole_disk_cvis` engine and `MultipoleDisk` with M = 0. Profiles `ThinRing`, `GaussianRing`, `PowerLaw`, `ExpTail`, `Gap` and `Sum`, each with its own quadrature rule (Gauss–Legendre, or Gauss–Laguerre for exponential tails). `blur_fwhm` and the inc/PA stretch. Tests against UD, Gaussian and thin ring; gradients; V(0) = 1. | 3–4 h |
| P2 | `height` with `Cone` (and the flared power law), plus a minimal linear-form `Harmonics` (Cartesian), which `TruncatedCone`'s modulation needs. **Subsume `TruncatedCone`**, converting the legacy `az_amps`/`az_pas` internally. Accuracy and speed against the old midpoint stack. *First payoff for the dust-plume worked example (A1).* | 3–4 h |
| P3 | `Harmonics` positive form (reflection coefficients) added to P2's linear form; the `virgil.harmonics` conversion helpers with round-trip tests; a test that reproduces the multimodality check and `VonMises`. Tests against a DFT of the rendered image. Subsume `ModulatedGaussianRim`; `GaussianArc` becomes von Mises. | 4–5 h |
| P4 | `RadialGP`, `MultipoleKernel`, `spectral_index` and `freeze_geometry`; NUTS with flat coordinates; `log_evidence` support. *Enables A2.* | 5–7 h |
| P5 | Tutorial notebook (§7) | 3–4 h |
| P6 | `MultipoleField` for `Image`. For `FlaredDisk*` (on `main`), a comparison test and report only; `FlaredDisk*` is not switched (decided, see §5 and §10). | 4–6 h |
| V | virgil-validation benchmarks against reference implementations (§12), committed at the end | 6–8 h |

Each stage is one PR with tests. P5 is the only virgil notebook. V comes last: commits to virgil-validation, following its independence rule. Docs describe the component, never this note. Every PR keeps the credit (§11) up to date.

**Deferred until a dataset needs them:**
- `Harmonics(form="log")`;
- FRAP-style physical fields (T, τ);
- the disk-harmonic (Lommel) basis. If it is ever revived: ∫₀^R J_m(j r/R) J_m(2πq r) r dr = R² j J_{m+1}(j) J_m(x)/(j² − x²), with x = 2πqR.

## 7. Tutorial (P5): a planet in a gap between inclined rings
A five-cell notebook in the house style. It depends on P1–P3.

**Truth scene:**
- **Star:** a `BlackBody` spectrum.
- **Rings:** three coplanar `MultipoleDisk` rings (`GaussianRing` profiles), each with its own spectrum. The outer ring has an m = 1 harmonic, so it is lopsided.
- **Gap and planet:** a gap between rings 2 and 3 holds a planet, a `PointSource` with its own `BlackBody`.
- **Shared geometry:** inc and PA are tied by a function returning the model. If that proves clumsy, add a small coplanar helper.

**Cells:**
1. **Simulate.** Chromatic VLTI-like coverage from `coverage.py`; V² and closure phases with realistic noise.
2. **Parametric fit.** The truth model family, with invariant priors: log-uniform fluxes and widths, uniform angles and positions. MAP with `fit`, then Laplace.
3. **Non-parametric deconvolution.** `System(star, disk=MultipoleDisk(RadialGP, Harmonics(degree=1, prior=MultipoleKernel(...)), spectral_index=RadialGP), planet=PointSource(...))`. A single GP disk, with no ring count or radii given, must recover:
   - the three rings and the gap within about one resolution element;
   - each ring's colour through α(r);
   - the lopsidedness of the outer ring;
   - the planet's position and flux, consistent with cell 2.
   The low M stops the disk absorbing the planet, and the radial GP is free to put flux in the gap if the data want it.
4. **Compare.** Laplace evidence for parametric against GP. Prior against posterior for the GP profile and α(r).
5. **Figure.** Truth, parametric fit and GP image side by side; I0(r) and α(r) with bands over the truth; V² and closure-phase residuals; recovered spectra.

**Budget:** it must run on a laptop in a few minutes (small n_r and N_vis; MAP and Laplace only). The NUTS version goes to OzSTAR.

## 8. Worked example: dust-plume imaging with GRAVITY
The colliding-wind binary fits use a binary plus `TruncatedCone`. Its docstring records Δlog L ≈ 29 over three epochs from 24 to 64 midpoint rings. The pixel `GaussianField` images have an evidence cost that grows steeply with npix: about 1 min at 48² and 10 min at 64².

This is done as a real worked example in its own analysis repository, never in virgil. Heavy runs go to OzSTAR, launched by Ben.
- **A1, after P2: faster, more accurate cones.**
  - **Method.** Refit the shared-cone scene with `TruncatedCone` now running on the engine (Gauss–Laguerre in slant distance, exact tail).
  - **Report.** Nodes needed for |Δlog L| < 0.5 against the old 64–256 rings; wall time per likelihood call and per fit; parameter shifts. Use the post-#120 closure-phase likelihood throughout.
- **A2, after P4: non-parametric plume.**
  - **Model.** Binary + `MultipoleDisk(Cone height, profile=RadialGP in slant distance, spectral_index=RadialGP, Harmonics(degree=1–2, form="positive", prior=MultipoleKernel))` + a coarse residual `GaussianField`.
  - **Workflow.** Geometry first from A1. Then `freeze_geometry` for the GP refit (one matrix–vector product per call), then a free-geometry check.
  - **Questions answered:**
    - Where does dust emission start, and how does it fade? (s0 and the exponential decline become data-driven.)
    - Does colour change along the plume? (α(s) bears on the fitted versus radiative-equilibrium dust temperature.)
    - Is the leading edge brighter?
    - How much residual structure is left once the cone takes the bulk flux?
- **A3, optional, after P3:** Cartesian harmonics on the parametric cone alone, as a cheap test of leading-edge brightening before A2.

## 9. Open decisions
- **Profile parametrization.** Proposed: I0 > 0 through exp of a GP or of positive parametric forms, with the vis-r forms as named profiles.

## 10. Review log
- **2026-10-14, adversarial review** (the author plus an independent agent).
  - **Adopted:**
    - one component;
    - shared nodes, with each profile owning its rule;
    - Cartesian coefficients;
    - inc and PA owned by the disk;
    - user-set M;
    - traced node arrays;
    - a scope cut.
  - **Fixed:** the sign of the Lommel integral.
  - **Rejected:** "inclination raises q_max"; "pixels for non-parametric disks", on cost grounds.
- **Ben's decisions, 2026-10-14:**
  - subsume the old disk classes;
  - one `MultipoleDisk`;
  - node quadrature everywhere;
  - user-set M;
  - the centre is the usual `dra`/`ddec` from the array centre;
  - keep the multipole-kernel prior;
  - one tutorial, which must also be deconvolved with the non-parametric GP disk;
  - the `Resolved` split is set by the priors, so report the total;
  - `GaussianArc` becomes von Mises;
  - add a coplanar helper only if needed.
- **Log against linear (2026-10-14).** The log form is deferred: it has no exact degree and needs an FFT. Linear is kept for the subsumed rims.
- **Square, then positive (2026-10-14).**
  - Ben suggested squaring.
  - A real square is unique but inexpressive: it has even degree only and cannot represent a pure m = 1 term.
  - A complex |h|² is expressive but has 2^M root-flip modes.
  - Adopted: the minimum-phase |h|² parametrised by reflection coefficients. It is unique, expressive and positive, with exact image degree M.
  - Ben decided that `degree` counts the image. The `Positivity` regulariser is dropped.
- **More decisions (2026-10-14):**
  - `freeze_geometry` is an explicit method returning a cached model;
  - the stages are reordered so the dust-plume worked example benefits first (A1 after P2, A2 after P4);
  - credit to the original papers and codes goes in docstrings and the contributors page (§11).
- **FlaredDisk\* stays (2026-10-14).** Ben's decision: `FlaredDisk*` stays undeprecated, and no PR carries a deprecation notice. Options for later:
  - add a pixel-evaluated height mode to `MultipoleDisk`, which would reproduce the Blakely approximation; or
  - keep `FlaredDisk*` for flared fits and use `MultipoleDisk` + `PowerLawHeight(h_ref < 0)` for new thin-surface work.

## 11. Credit
We reimplement ideas and copy no code (frank is LGPL). Credit is still mandatory, in two places:
- **Docstrings.** Every class or function that follows a paper cites it in a `References` section:
  - the Fourier–Bessel DHT and GP radial profiles: Jennings et al. 2020, MNRAS 495, 3209 (frank, github.com/discsim/frank);
  - forward parametric radial profiles, the uv deprojection and the Gaussian scale-height envelope: Kennedy 2025, RASTI (vis-r, github.com/drgmk/vis-r);
  - whitened GP latents on Fourier–Bessel nodes and per-quantity radial GPs: Yoshida et al. 2026, arXiv:2607.12618 (FRAP, github.com/tomyoshida/frap);
  - the positive harmonic parametrization: cite only Dumitrescu 2007 (Fejér–Riesz and the spectral factorisation) and Barndorff-Nielsen & Schou 1973 (the reflection-coefficient parametrization). Add them to docstrings and `references/` in P3, but not to the contributors page, since this is standard mathematics rather than borrowed code;
  - the existing `TruncatedCone`, `ModulatedGaussianRim` and flared-disk credits carry over to the engine-backed classes unchanged.
- **`docs/contributors.md`.** Under *Methods*, one entry per work: what it gave virgil and a link to the paper and code. The BibTeX goes in `references/`. Both are added in the PR that first uses the idea (P1 for frank and vis-r; P4 for FRAP), and the PR description says so.

## 12. Validation against reference implementations (stage V, virgil-validation)
The last stage commits benchmarks to virgil-validation. They follow its independence rule: reference values come from the other codes, or from code written in the validation repository, never from virgil itself.

The benchmarks, in the order they are committed:
- **V1. Forward transforms (accuracy and speed).**
  - **Reference codes.** For identical profiles and geometry, compare `MultipoleDisk` (M = 0) with:
    - frank's `DiscreteHankelTransform`;
    - vis-r's `r_prof_*` profiles with `uv_trans` and `setup_dht`, including its Gaussian scale-height factor;
    - FRAP's Hankel matrix (`set_visibility` without the primary beam).
  - **Reports.** Max |ΔV| over baselines, and wall time per call at N_vis = 10³, 10⁴ and 10⁵ (CPU, plus GPU on OzSTAR).
- **V2. Non-axisymmetric and geometric models.**
  - **PMOIRED.** Its rings with azimuthal harmonics and its radial profiles, matching the earlier PMOIRED comparison.
  - **oimodeler.** Its radial-profile and asymmetric-ring components, if available.
  - **Brute force.** Validation-repo numpy DFTs of finely rendered images: harmonics, `VonMises` arcs, cones and flared surfaces.
  - **Old against new.** `TruncatedCone` against its pre-subsumption midpoint form at large n_rings, from a pinned old virgil commit.
- **V3. Non-parametric recovery on public data.**
  - **frank.** Its documented DSHARP AS 209 example, against `RadialGP` MAP and posterior (real visibilities): profile difference within the bands, and runtime.
  - **vis-r.** Its HR 4796 example, with parameters and errors against the paper's.
  - **FRAP.** Its HD 169142 example pickles (single-band I(r) only, without its radiative-transfer layer), comparing profile shapes.
- **V4. Optical/IR recovery, which no reference code covers.**
  - **Data.** Injection and recovery on NRM and VLTI coverage (V² and closure phases) for ring + gap + planet scenes, with the positive harmonics at M = 1–3.
  - **Test.** Coverage of the posterior bands, as calibration checks.
  - **Compute.** It runs on OzSTAR, launched by Ben.

Each benchmark records software versions and a pinned virgil commit hash, and gives its pass criteria up front.

## 13. Execution (autonomous run from 2026-10-14)
**Staffing.**
- **Coordinator.** One thin Sonnet coordinator in its own worktree.
- **Stage implementers.** One Sonnet implementer per stage: a pr-worker session where the app allows the coordinator to start sessions, otherwise an Agent subagent on that stage's branch.
- **PR watchers.** Haiku, only for trivial lint/format CI fixes and status relay.
- **Reviewers.** A Sonnet adversarial reviewer for P3, where the positive form, the conversion helpers and the conventions carry the most risk of a subtle error.
- **No Opus.** The stages are well specified, so no Opus agent is needed; ask Ben first if one ever seems necessary.

**Order.**
1. P1 → P2 → P3 → P4 → P5, as stacked PRs, each branched from the previous stage's branch.
2. P6 after P3, in parallel with P4, since it touches different files.
3. V after P4, in virgil-validation, pinned to the stage branch's commit hash.
4. A1, a light check, after P2.

**Rules for the run.**
- **Merging.** Never merge. Ben merges stacked PRs in order, and they are retargeted to `main` as their bases merge.
- **Git.** Merge, never rebase. Merge `origin/main` before pushing, and never use `--no-verify`.
- **Python environment.** Use a worktree venv (Python 3.12); never `uv sync`/`uv run` in `~/code/virgil`. In virgil-validation, install frank, vis-r, astro-frap and PMOIRED into that repository's own environment.
- **Compute.** *(Overridden 2026-10-14: heavy runs moved to OzSTAR, see §14.)* Nothing on OzSTAR: no ssh, no job scripts, no prompts to Ben. Keep all JAX light on the laptop. That means:
  - timings at N_vis ≤ 10⁴ on CPU, with GPU and 10⁵ timings deferred;
  - MAP and Laplace only for the tutorial and V4, with NUTS at most a few hundred draws on tiny problems;
  - A1 is fixed-parameter likelihood evaluations only. Compare the old and new cone at the stored best-fit parameters: Δlog L against nodes, and time per call. Write a short report in the Apep analysis directory. Commit nothing to virgil. A1 and A2 fits wait for Ben.
- **V3 data.** Use virgil's existing data paths. If complex visibilities can't be ingested without a new loader, compare forward models and profile shapes only, and note it. Do not add loaders.
- **Delegated PRs.** Check `gh pr files` before calling any delegated PR ready; an agent once reported tests it never wrote.
- **Decisions.** Record any decision needed from Ben in a `QUESTIONS.md` at the worktree root, choose the conservative option, and continue.
- **Status.** Keep a `STATUS.md` at the worktree root: stage, branch, PR URL, CI state, open questions. It is the hand-off for Ben in the morning.

## 14. Execution log (2026-10-14)
The run produced stacked PRs P1–P6 (#335 and #336 merged; #338 P3; #343 P4; #347 P5; #340 P6) plus a validation PR in virgil-validation. Heavy runs (experiments, tutorial execution, V4) were moved to OzSTAR by Ben's decision ("Option A"), overriding the no-HPC rule in §13. No deprecations were made.
