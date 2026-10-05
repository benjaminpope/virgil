# Orbit fitting in virgil: prior art, credit and what to adopt

Status: **decided** (Ben, 2026-10-05; §6). This note compares `virgil.orbits` with the two orbit-fitting codes nearest to it, [orbitize!](https://github.com/sblunt/orbitize) and [Octofitter](https://github.com/sefffal/Octofitter.jl), and with orvara where it matters. It records which of our features already exist elsewhere, whom to credit for them, what is worth adopting, and what virgil should not try to rebuild.

Sources: orbitize! 3.4.0 (source and `docs/*.rst`, read 2026-10-05); Octofitter's documentation, paper and `src/likelihoods/` on `main` (read 2026-10-05); the papers below, checked on ADS. Items marked *(unverified)* come from documentation summaries and need a look at the source before anything depends on them.

## 1. Summary

- **The overlap is real but narrow.** The orbit kernel (Campbell, Thiele–Innes and Cartesian elements; Gaussian position and RV likelihoods with per-instrument offsets and jitter) is standard, and all three codes have it. virgil's version exists because it has to run inside JAX and inside the scene: an orbit drives the components of an interferometric model at each datum's time (`at(mjd)`, `Attached`). Neither orbitize! nor Octofitter can do that from Python/JAX, and neither models extended structure.
- **We already avoid the worst duplication.** Kepler's equation is solved by jaxoplanet, not by us. The conventions (§3) agree with those documented by orbitize!, Octofitter and orvara.
- **Credit is missing.** The docs name jaxoplanet but cite nothing, and the orbit docs name none of the prior codes. §5 lists what to cite; this PR adds it to `docs/api/orbits.md`.
- **Octofitter is the closest prior art for our use case.** It fits orbits directly to closure phases, V² and kernel phases, and its paper demonstrates multi-epoch detection with simulated JWST/NIRISS AMI data (Thompson et al. 2023). Our detection and AMI work should cite it and compare with it.
- **No dependency** on orbitize!, Octofitter or orvara in virgil (§6). orbitize! is used only in virgil-validation, as an independent root of trust.
- **Adopt** (§4): angles sampled as 2-D vectors, with von Mises priors as chord residuals that match our phase likelihoods (§4.1); the position angle at a reference epoch as an alternative to the time of periastron (§4.2); per-dataset North-angle and plate-scale nuisances (§4.3); simulation-based calibration in virgil-validation.
- **Do not rebuild** absolute astrometry (Hipparcos IAD, HGCA, Gaia epoch astrometry), N-body dynamics, or a general orbit-fitting front end. If a science case needs them, the user should run orbitize!, Octofitter or orvara, and import the result as a prior.

## 2. Feature comparison

| Feature | virgil (`virgil.orbits`) | orbitize! | Octofitter |
|---|---|---|---|
| Language, autodiff | Python/JAX; exact gradients | Python, Cython/C, optional CUDA; no gradients | Julia; ForwardDiff (and others) |
| Kepler solver | jaxoplanet (`jaxoplanet.core.kepler`) | Newton–Raphson for e < 0.95, Mikkola (1987) otherwise | Markley (1995) by default; Goat Herd (Philcox et al. 2021), Newton |
| Hyperbolic orbits | no | no | yes |
| Campbell elements | `KeplerOrbit` (a_mas, P, e, i, ω, Ω, dt_peri) | Standard basis (sma in au, plx, mtot, τ) | yes (in PlanetOrbits.jl) |
| Epoch parameter | `dt_peri`: time of periastron minus a static float64 `t_ref` | τ: periastron as a fraction of P after `tau_ref_epoch` (MJD 58849) | θ: position angle at a reference epoch, as well as t_p |
| Thiele–Innes | `ThieleInnesOrbit`, and analytic marginalisation (design: `thiele_innes_marginalisation.md`) | no | yes |
| Cartesian state | `StateVectorOrbit` (distance-free μ) | XYZ basis (no sep/PA data) | yes |
| Relative astrometry | `PositionData`, full 2 × 2 covariance per epoch | yes, with `quant12_corr` | yes, with correlation; optional plate-scale and North-angle nuisances |
| Fit to interferometric data | **yes**: any scene, any `OIData`, every datum at its own time | no | **yes** (OctofitterInterferometry): point sources, closure phases, V², kernel phases, GRAVITY fibre coupling |
| Extended or attached components | **yes** (`Attached`, `at(mjd)`) | no | no |
| Images | via the scene and imaging code | no | yes (OctofitterImages: likelihood on image grids) |
| RVs | `RVData`: primary or secondary, jitter, offsets **marginalised analytically** | primary and secondary, offsets and jitter fitted | offsets and jitter fitted *(unverified)*; GP noise; trends; secular acceleration |
| Absolute astrometry | no | Hipparcos IAD, Gaia DR2/eDR3 position, HGCA | Hipparcos IAD, HGCA, Gaia DR2/DR3 (G23H), Gaia DR4 epoch astrometry |
| Multiple companions, N-body | no | multiple planets with perturbations of the star; REBOUND | any hierarchy; symplectic N-body (AHL21) |
| Priors | numpyro distributions; `AxialVonMises` for axes known modulo 180° | Uniform, LogUniform, Sin, Gaussian, KDE, O'Neil (2019) observable prior | any Distributions.jl prior; `UniformCircular`; KDE; O'Neil prior; dynamical stability priors |
| Samplers | `fit`, NUTS (numpyro, BlackJAX); `starting_orbits` grid | OFTI, ptemcee, dynesty | NUTS (AdvancedHMC), Pigeons parallel tempering, rejection |
| Starting points | Thiele–Innes least squares on a (P, e, t_peri) grid | OFTI draws | prior draws → MAP → Pathfinder |
| Validation tools | in virgil-validation | — | prior and posterior predictive checks, simulation-based calibration, cross-validation |
| Licence | MIT | BSD-3-Clause | MIT |

## 3. Conventions

All three codes use the **secondary's** ω, with the primary's at ω + 180°, and +z **away** from the observer, so that the ascending node is the receding one. Octofitter's paper says it matches orbitize! and orvara. Our §2.1 conventions (`orbit_scene_joint_fitting.md`) are the same: secondary's ω, Ω the receding node, dz away from the observer, position angle North through East. Differences in the epoch parameter are only reparameterisations:

| virgil | orbitize! | Octofitter |
|---|---|---|
| t_p = t_ref + dt_peri | t_p = tau_ref_epoch + τ P | t_p, or θ at a reference epoch |
| a_mas | sma × plx | a × plx |
| P (days) | from sma and mtot (years) | P or from a and M |

**This agreement is from documentation and has not been tested numerically.** orbitize! computes positions from Green's (1985) cos²(i/2), sin²(i/2) form, which is independent of our Thiele–Innes route. That makes it a good oracle (§4, item 1). Its documented RV input convention is also worth noting: RVs of a non-primary body are barycentric.

## 4. What to adopt

| # | Item | From | Where | Effort | Priority |
|---|---|---|---|---|---|
| 1 | Ephemeris and convention cross-check against orbitize!: sep, PA and RV over a grid of elements, both stars, after mapping §3 | orbitize! | virgil-validation (independent side), never virgil | 2–3 h | **high**: closes §5.1.3 of the orbit design without a real anchor |
| 2 | Posterior comparison on a published dataset that orbitize! ships (e.g. β Pic b relative astrometry, Nielsen et al. 2020), with the same priors | orbitize! | virgil-validation; OzSTAR | 3–4 h | medium |
| 3 | Angle parameterisation for NUTS: Ω, ω (and PA-like angles in general) sampled as a 2-D vector whose direction is the angle, so that there is no wrap boundary | Octofitter's `UniformCircular`; exoplanet's `Angle` | `virgil` priors, reusable beyond orbits | 3–4 h | high; design in §4.1 |
| 4 | θ, the position angle at `t_ref`, as an alternative to `dt_peri` for sampling `KeplerOrbit` | Thompson et al. (2023) | `orbits.py`, a reparameterisation with a test | 2 h | medium (short arcs, with `StateVectorOrbit`); §4.2 |
| 5 | Per-dataset North-angle and plate-scale nuisances (a rotation and a scale of (dra, ddec) or of (u, v)) | Octofitter | the `noise=` / per-dataset terms of `fit` | 2–3 h | medium: the multi-instrument fits of R7 need it for imaging and masking data; §4.3 |
| 6 | Observable-based priors for short arcs | O'Neil et al. (2019) | a prior helper, after `StateVectorOrbit` is in use | 3 h | low |
| 7 | Simulation-based calibration of orbit posteriors | Octofitter (`sbc.jl`) | virgil-validation campaigns on OzSTAR | 3–4 h | medium |
| 8 | Initialisation by Pathfinder after `starting_orbits` | Octofitter (Zhang et al. 2022) | only if NUTS warm-up is slow on real problems; BlackJAX has Pathfinder | 1–2 h | low |

### 4.1 Angles as 2-D vectors, and how they meet the von Mises likelihoods

**The problem is the parameterisation, not the likelihood.**
- Every angle in a virgil model (Ω, ω, a binary's or disc's position angle, a rim's azimuth) enters through sines and cosines. Unprojected phase residuals are chords, 2 sin(Δ/2)/σ (`likelihood.py`), so the likelihood is a von Mises density, smooth and 2π-periodic, in data and in parameters alike.
- The wall comes from the priors. `fit` and numpyro map each prior's support onto the real line. `Uniform(0, 360)`, numpyro's `VonMises` (support `circular`, [−π, π)) and `AxialVonMises` (support [0°, 360°)) all go through a sigmoid onto an interval (checked: `biject_to(VonMises(0, 1).support)` is a sigmoid–affine composition).
- So an angle whose posterior straddles the wrap sits against a hard boundary in the unconstrained coordinates, and a broad angle cannot pass from 359° to 1°.

**The construction.** Sample v = (x, y) = r(cos θ, sin θ) in ℝ², and report θ = atan2(y, x). Any rotationally symmetric prior on v makes θ exactly uniform and independent of r. The likelihood depends on v only through the direction v̂ = v/r, so r is a pure nuisance.

**A ring, not a Gaussian.**
- Octofitter (`UniformCircular`) and exoplanet (`Angle`) put v ~ N(0, I). Its density peaks at the origin, where θ is undefined. That is harmless for NUTS, since r is then Rayleigh-distributed with its mode at 1. It is not harmless for `fit`: a MAP fit drives r → 0, where the gradients of v̂ grow as 1/r.
- virgil should use the residual (r − 1)/s instead: a density ∝ exp(−(r − 1)²/2s²) in v. It is rotationally symmetric, so θ is still exactly uniform. Its mode is the unit circle. It is a least-squares residual, so Levenberg–Marquardt, the Laplace covariance and `gauss_newton_mass` all take it unchanged. Start with s = 0.25 and tune it in the tests.
- **Sampling geometry.** A tightly measured angle (width σ_θ) gives a blob near the unit circle, s wide radially and σ_θ wide tangentially. It is nearly straight, so the dense Gauss–Newton mass matrix whitens it. A broad angle gives a ring of width s, which NUTS goes round. Unlike with the Gaussian, the sampler cannot cut through the origin. That matters only for well-separated modes, which the next part removes where they are exact.

**Von Mises priors become chord residuals, the same form as the likelihood.**
- A von Mises prior VM(μ, κ) on θ has −log p = κ(1 − cos(θ − μ)) + const = (κ/2)|v̂ − m̂|², with m̂ = (cos μ, sin μ). It is therefore the 2-vector residual √κ (v̂ − m̂).
- This is the chord of virgil's phase likelihood: 2 sin(Δ/2)/σ = |e^{iφ} − e^{iφ_data}|/σ, with κ = 1/σ². One form then serves data and priors.
- `fit` today rejects `VonMises` priors, which have no least-squares form. On an angle vector it can take them.
- Keep the normaliser on the circle, −log(2π I₀(κ)), through `i0e` as `likelihood.py` does, so that evidences stay normalised.
- **`AxialVonMises`** is the same chord on the doubled direction, (cos 2θ, sin 2θ) = (x² − y², 2xy)/r²: the residual is √κ ((cos 2θ, sin 2θ) − (cos 2μ, sin 2μ)).
- **A uniform prior on θ** adds no residual beyond the ring.

**Sample the identified combinations, not the raw angles.** Vectors remove the wall, not multimodality. For orbits, the exact symmetries of the orbit design (§2.3 there) say which angles to make vectors:
- **Positions only.** (Ω, ω) and (Ω + 180°, ω + 180°) predict the same positions. Sample the vector of 2Ω (Ω modulo 180°) and the vector of ϖ = Ω + ω. The flip changes ϖ by 360°, so ϖ is invariant, and each pair of modes collapses to one point. Report Ω in [0°, 180°), as `starting_orbits` does.
- **With RVs, or any other datum that breaks the flip:** the vectors of Ω and ϖ.
- **Near face-on,** astrometry measures ϖ but not Ω and ω separately (orbit design §2.3, symmetry 4). ϖ is then the tight vector and Ω the broad one, instead of two broad, correlated angles.
- **The primary/secondary swap** (ω + 180°, r → −r) is not a parameterisation matter. The flux-ratio convention fixes it.

**Priors stay invariant.** virgil's default priors are the invariant (Jeffreys) priors of each parameter's symmetry group. For the orbit's orientation that is the Haar measure on rotations: uniform in cos i, Ω and ω. The ring makes each vector's angle exactly uniform. The map (Ω, ω) → (2Ω, ϖ) is linear on the torus, with a constant Jacobian (2), so uniform (Ω, ω) is uniform (2Ω, ϖ) and no correction is needed. A von Mises prior is strong information and must say where it comes from, such as an external orbit.

**Evidence.** The radial dimension integrates out exactly, since its prior is normalised. A Laplace evidence in v-space approximates that integral by a Gaussian in r. *Corrected when built:* the error is not of order s². At the mode the Hessian in v is diagonal in (r, rθ), with no cross term, and the mean of r under the ring is 1, so the v-space Laplace evidence equals the angle-space Laplace evidence up to the ring's truncation at r = 0, a relative error of order exp(−1/2s²) (about 1e-6 in log Z at s = 0.25). The only error left is the Laplace error in θ itself. `tests/test_angles.py` checks both.

**Interface (built).**
- `virgil.angles.AngleVector(mean=None, kappa=None, *, axial=False, ring_width=0.25)`, a numpyro distribution on ℝ², placed in the usual `priors` dict under the angle's path. `numpyro_model` samples `<path>_vec` and records the deterministic `<path>` in degrees; `fit` optimises the vector (from the unit vector of the starting angle), its residuals are the ring and the chord, and `FitResult.values` holds both `<path>` and `<path>_vec`. `gauss_newton_mass` keys the block by `<path>_vec` and differentiates the vector's residuals with the data's rows, since they mix its two coordinates.
- `orbits.orientation_priors(positions_only=True, prefix="")` returns uniform angle vectors for `two_Omega` (or `Omega`) and `varpi`; `KeplerOrbit.from_varpi(period, dt_peri, ecc, inc, varpi, a_mas, two_Omega=... | Omega=...)` builds the orbit, through `orientation_from_varpi`.

**Tests.**
1. θ is uniform under the ring prior (Kolmogorov–Smirnov test).
2. The von Mises and axial chords give the right marginals.
3. NUTS on a posterior straddling 0°/360° matches the wrapped truth.
4. A MAP fit converges through the wrap.
5. (2Ω, ϖ) ↔ (Ω, ω) round-trips, and the flip modes collapse to one point.
6. The Laplace evidence agrees with the angle-space evidence.

### 4.2 The position angle at a reference epoch (item 4)
Thompson et al. (2023) replace t_p with θ, the position angle at a reference epoch, because astrometry measures it directly.

**Conversion.** In our elements, with u = ω + f the argument of latitude at `t_ref`, the Thiele–Innes relations (orbit design §2.4) give (cos(θ − Ω), sin(θ − Ω)) ∝ (cos u, sin u cos i). Then:
1. u = atan2(sin(θ − Ω)/cos i, cos(θ − Ω));
2. f = u − ω;
3. f gives E, then M, then `dt_peri`.

**Properties.**
- θ is itself an angle, so it is sampled as a vector (§4.1).
- **The prior needs a Jacobian.** The invariant prior is uniform in the time of periastron, i.e. in the mean anomaly M at `t_ref` (a translation), not in θ. Sampling θ therefore carries the log-Jacobian
  log|∂M/∂θ| = 3/2 log(1 − e²) − 2 log(1 + e cos f) + log|cos i| − log(cos²φ cos² i + sin²φ), with φ = θ − Ω,
  as a prior term at fixed (P, e, i, ω, Ω). Checked numerically: it matches finite differences to 1e-7, and θ weighted by it gives a uniform M (to 1.5% in 12 bins over 4 × 10⁵ draws). A uniform prior placed on θ itself would be a different prior on t_p, one that depends on the orientation and e; this term undoes that. (We have not checked how Octofitter handles this.)
- The map is singular at i = 90°, where the position angle takes only two values. Near edge-on, keep `dt_peri` or use `StateVectorOrbit`.
- θ belongs with the short-arc tools, beside `StateVectorOrbit`.

### 4.3 North angle and plate scale (item 5)
**Per dataset,** a rotation δ and a fractional scale s act on the scene's sky positions: (dra, ddec) → (1 + s) R(δ)(dra, ddec).
- Rotating and scaling the sky is the same as rotating and scaling (u, v), so for `OIData` the plate scale is the existing `wavel_scale` (spectro note §2.6). Only the North angle is new.
- **Long-baseline interferometers** know their baselines geometrically, so δ = 0 there.
- **Imaging, masking and kernel-phase datasets,** and published positions (`PositionData`), take both, with Gaussian priors from the instrument's astrometric calibration.
- A sign test (a known rotation recovered with the right sign) goes with the position-angle round trips of orbit design §5.3.

**Not adopted, and why:**
- **OFTI** (Blunt et al. 2017). Its niche, cheap posteriors from short arcs of relative astrometry, is to be covered more exactly by the Thiele–Innes marginalisation. That is designed (`thiele_innes_marginalisation.md`) but not yet built (Stage 6a.1, "Later"); until then, `starting_orbits` plus NUTS is the route. The marginalisation integrates the four linear elements analytically instead of scaling and rotating prior draws. That note already cites OFTI and compares it (§2.1 there). OFTI is also a good comparison in item 2.
- **Absolute astrometry** (Hipparcos IAD, HGCA, Gaia DR4). It is a large, specialised effort that three codes have already done carefully (Brandt 2021; Nielsen et al. 2020; Leclerc et al. 2023). If a system needs it, fit it in orvara, orbitize! or Octofitter and bring the result into virgil as a prior on the shared elements. Gaia DR4 epoch astrometry (expected December 2026) may change this for binaries resolved by interferometry; revisit then.
- **N-body and hierarchical systems.** Not needed by any current case. A triple would bring it back.
- **GP noise for RVs, photometry, evolutionary-model priors.** Out of scope for an interferometry package.
- **A general orbit front end** (configuration files, orbit plots, results files). virgil's front end is `fit` and numpyro. A plot of posterior orbits on the sky is worth a small helper if it is repeatedly needed.

## 5. Credit

**Cite in the orbit docs and in papers that use `virgil.orbits`:**
- jaxoplanet, which solves Kepler's equation for us: Hattori et al., [Zenodo 10.5281/zenodo.10736936](https://doi.org/10.5281/zenodo.10736936); and exoplanet, from which it descends: Foreman-Mackey et al. 2021, JOSS 6, 3285.

**Cite where we use an idea first published elsewhere:**

| Idea in virgil | Reference |
|---|---|
| Thiele–Innes constants, linear least squares for (A, B, F, G) at fixed (P, e, T) | the classical method (Thiele 1883, AN 104, 245; e.g. Binnendijk 1960); Hartkopf, McAlister & Franz 1989, AJ 98, 1014; Lucy 2014, A&A 563, A126; Wright & Howard 2009, ApJS 182, 205 |
| Analytic marginalisation of linear parameters | Luger, Foreman-Mackey & Hogg 2017, arXiv:1710.11136 |
| Analytic marginalisation of RV zero points | also done by orvara (Brandt et al. 2021, AJ 162, 186), and in The Joker (Price-Whelan et al. 2017, ApJ 837, 20) |
| Rejection sampling for short arcs (comparison) | Blunt et al. 2017, AJ 153, 229 (OFTI) |
| Orbit fits directly to closure phases and kernel phases; multi-epoch detection in the orbital domain | Thompson et al. 2023, AJ 166, 164 (Octofitter) |
| Observable-based priors (if adopted) | O'Neil et al. 2019, AJ 158, 4 |
| Angle vectors (`virgil.angles.AngleVector`, adopted) | Octofitter's `UniformCircular` (Thompson et al. 2023); exoplanet's `Angle` (Foreman-Mackey et al. 2021) |

**Cite as related software in the docs:** orbitize! (Blunt et al. 2020, AJ 159, 89; Blunt et al. 2024, JOSS 9, 6756), Octofitter (Thompson et al. 2023, AJ 166, 164), orvara (Brandt et al. 2021, AJ 162, 186).

**Not copied.** No code from orbitize!, Octofitter or orvara is in virgil. Both orbitize! (BSD-3) and Octofitter (MIT) are permissively licensed. If code is ever ported, keep the licence notice and say so in the docstring.

## 6. Decisions (Ben, 2026-10-05)

1. **No dependency** on orbitize!, Octofitter or orvara in virgil, its tests or its docs build, wherever it can be avoided. No converters (as decided on 2026-10-03).
2. **orbitize! is a root of trust in virgil-validation** (items 1 and 2): ephemerides, RVs, symmetries and derived masses checked against it on the validation repo's independent side.
3. **Adopt:**
   - angles as 2-D vectors (§4.1), designed together with our von Mises likelihoods;
   - the position angle at a reference epoch (§4.2);
   - North-angle and plate-scale nuisances (§4.3);
   - simulation-based calibration of orbit posteriors in virgil-validation (item 7).
4. **Not now:** observable-based priors (item 6) and Pathfinder (item 8).
5. **Absolute astrometry stays out** (default), revisited with Gaia DR4.
6. **No contact with the Octofitter authors** for now; Ben will arrange a comparison in due course.
