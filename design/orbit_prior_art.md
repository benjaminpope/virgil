# Orbit fitting in virgil: prior art, credit and what to adopt

Status: **review**, 2026-10-05. This note compares `virgil.orbits` with the two orbit-fitting codes nearest to it, [orbitize!](https://github.com/sblunt/orbitize) and [Octofitter](https://github.com/sefffal/Octofitter.jl), and with orvara where it matters. It records which of our features already exist elsewhere, whom to credit for them, what is worth adopting, and what virgil should not try to rebuild.

Sources: orbitize! 3.4.0 (source and `docs/*.rst`, read 2026-10-05); Octofitter's documentation, paper and `src/likelihoods/` on `main` (read 2026-10-05); the papers below, checked on ADS. Items marked *(unverified)* come from documentation summaries and need a look at the source before anything depends on them.

## 1. Summary

- **The overlap is real but narrow.** The orbit kernel (Campbell, Thiele–Innes and Cartesian elements; Gaussian position and RV likelihoods with per-instrument offsets and jitter) is standard, and all three codes have it. virgil's version exists because it has to run inside JAX and inside the scene: an orbit drives the components of an interferometric model at each datum's time (`at(mjd)`, `Attached`). Neither orbitize! nor Octofitter can do that from Python/JAX, and neither models extended structure.
- **We already avoid the worst duplication.** Kepler's equation is solved by jaxoplanet, not by us. The conventions (§3) agree with orbitize!, Octofitter and orvara.
- **Credit is missing.** The docs name jaxoplanet but cite nothing, and the orbit docs name none of the prior codes. §5 lists what to cite; this PR adds it to `docs/api/orbits.md`.
- **Octofitter is the closest prior art for our use case.** It fits orbits directly to closure phases, V² and kernel phases, and its paper demonstrates multi-epoch detection with simulated JWST/NIRISS AMI data (Thompson et al. 2023). Our detection and AMI work should cite it and compare with it.
- **Adopt** (§4): angle parameterisations for sampling; a reference-epoch position angle instead of the time of periastron; per-dataset North-angle and plate-scale nuisances; and, in virgil-validation, orbitize! as an independent ephemeris and posterior oracle.
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

**This agreement is from documentation and has not been tested numerically.** orbitize! computes positions from Green's (1985) cos²(i/2), sin²(i/2) form, which is independent of our Thiele–Innes route. That makes it a good oracle (§4.1). Its documented RV input convention is also worth noting: RVs of a non-primary body are barycentric.

## 4. What to adopt

| # | Item | From | Where | Effort | Priority |
|---|---|---|---|---|---|
| 1 | Ephemeris and convention cross-check against orbitize!: sep, PA and RV over a grid of elements, both stars, after mapping §3 | orbitize! | virgil-validation (independent side), never virgil | 2–3 h | **high**: closes §5.1.3 of the orbit design without a real anchor |
| 2 | Posterior comparison on a published dataset that orbitize! ships (e.g. β Pic b relative astrometry, Nielsen et al. 2020), with the same priors | orbitize! | virgil-validation; OzSTAR | 3–4 h | medium |
| 3 | Angle parameterisation for NUTS: Ω, ω (and PA-like angles in general) sampled as a 2-D vector whose direction is the angle, so that there is no wrap boundary | Octofitter's `UniformCircular`; exoplanet's `Angle` | `virgil` priors, reusable beyond orbits | 2 h | high |
| 4 | θ, the position angle at `t_ref`, as an alternative to `dt_peri` for sampling `KeplerOrbit` | Thompson et al. (2023) | `orbits.py`, a reparameterisation with a test | 2 h | medium (short arcs, with `StateVectorOrbit`) |
| 5 | Per-dataset North-angle and plate-scale nuisances (a rotation and a scale of (dra, ddec) or of (u, v)) | Octofitter | the `noise=` / per-dataset terms of `fit` | 2–3 h | medium: the multi-instrument fits of R7 need it for imaging and masking data |
| 6 | Observable-based priors for short arcs | O'Neil et al. (2019) | a prior helper, after `StateVectorOrbit` is in use | 3 h | low |
| 7 | Simulation-based calibration of orbit posteriors | Octofitter (`sbc.jl`) | virgil-validation campaigns on OzSTAR | 3–4 h | medium |
| 8 | Initialisation by Pathfinder after `starting_orbits` | Octofitter (Zhang et al. 2022) | only if NUTS warm-up is slow on real problems; BlackJAX has Pathfinder | 1–2 h | low |

**Not adopted, and why:**
- **OFTI** (Blunt et al. 2017). Its niche, cheap posteriors from short arcs of relative astrometry, is covered more exactly by the Thiele–Innes marginalisation (`thiele_innes_marginalisation.md`), which integrates the four linear elements analytically instead of scaling and rotating prior draws. That note already cites OFTI and compares it (§2.1 there). OFTI is also a good comparison in item 2.
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
| Angle parameterisation (if adopted) | Octofitter's `UniformCircular`; exoplanet's `Angle` (Foreman-Mackey et al. 2021) |

**Cite as related software in the docs:** orbitize! (Blunt et al. 2020, AJ 159, 89; Blunt et al. 2024, JOSS 9, 6756), Octofitter (Thompson et al. 2023, AJ 166, 164), orvara (Brandt et al. 2021, AJ 162, 186).

**Not copied.** No code from orbitize!, Octofitter or orvara is in virgil. Both orbitize! (BSD-3) and Octofitter (MIT) are permissively licensed. If code is ever ported, keep the licence notice and say so in the docstring.

## 6. Decisions for Ben

1. **orbitize! in virgil-validation (items 1, 2, 7).** The 2026-10-03 decision ruled out orbitize! as a dependency or converter in virgil. Using it as an independent oracle in the validation repo leaves that intact, and fits the validation repo's rule that the reference side is never virgil. Approve?
2. **Which of items 3–5 to build in Stage 6a.1** (default: 3 and 5 now; 4 with the short-arc work).
3. **Absolute astrometry stays out** (default), revisited with Gaia DR4.
4. **Talk to the authors?** William Thompson (Octofitter) has an interferometry likelihood and an AMI demonstration close to ours. A comparison on a shared simulated AMI dataset would test both codes; ask first.
