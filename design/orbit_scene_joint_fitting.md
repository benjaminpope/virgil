# Keplerian orbits in virgil: alone, and jointly with the scene

Status: **design**, 2026-10-03. Orbits are built in virgil, which owns the orbital geometry and conventions (Thiele–Innes, §2.4); [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) (0.1.0, an optional dependency) only solves Kepler's equation, and appears otherwise only in the converters. Being built in Stage 6a.1: `virgil.orbits` (`KeplerOrbit`, `ThieleInnesOrbit`) exists; see the implementation log in [`imaging_plan.md`](imaging_plan.md).

This note is the design of Stage 6a.1, orbits and binary-frame scenes, which grew out of the "Keplerian orbits" item first listed in Stage 8 ([`imaging_plan.md`](imaging_plan.md), [`pmoired_parity.md`](pmoired_parity.md)) into a general capability: fitting orbits to interferometric data from any instrument virgil reads, with radial velocities and external priors, and with scene components that move with the binary. The spectral and calibration side is in [`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md). The runnable sketch is [`sketches/orbit_attached.py`](sketches/orbit_attached.py). How this compares with orbitize!, Octofitter and orvara, and what to credit and adopt from them, is in [`orbit_prior_art.md`](orbit_prior_art.md).

The colliding-wind binary Apep (VLTI/GRAVITY, 2023–25) is a worked example. It exposed most of the requirements below. Its science and Apep-specific scripts live with the analysis, in `~/data/apep_gravity` (`notes/lessons_for_drpangloss.md`, `notes/omega_convention_question.md`, `scripts/attached_cone_sketch.py`).

## 1. Use cases

| Case | Data | Scene | What it needs |
|---|---|---|---|
| Visual binary, resolved by long-baseline interferometry (VLTI, CHARA), several epochs | V², closure phase | two stars (point sources or disks) | the orbit drives the companion's position at each datum's MJD (R0, R3) |
| Faint companion from aperture masking, kernel phase or AMI DISCOs, several epochs | closure/kernel phases, DISCOs | star + companion | short arcs and low S/N, so a sampling-friendly parameterisation (R4), priors (R5) and good starting orbits (R0) |
| Short-period binary | motion within a night | two stars | per-datum time evaluation (R3); possibly smearing over an exposure |
| Double-lined spectroscopic binary that is also resolved | visibilities + radial velocities | two stars | an RV likelihood (R5); Ω becomes absolute; masses and a dynamical parallax (R6) |
| Binary with structure fixed in its frame: colliding-wind dust (Apep, WR 140), circumstellar discs in or out of the orbital plane, streams | visibilities | stars + extended components | the 3-D relative vector (R1); components attached to the binary frame (R2); co-rotating against changing scenes (R9) |
| The same system seen with several instruments | per-dataset | one scene | per-dataset bandpass, errors and wavelength scale; fit visibilities, not derived positions (R7); `simulate` to test biases (R8) |

**Apep, briefly.** It needed nearly every row at once:
- GRAVITY positions good to 0.013 mas on a 28 mas binary;
- a dust cone whose vertex and axis follow the line of centres;
- external orbital elements from JWST plume modelling, whose orientation disagrees with the measured position angle (§2.5);
- an aperture-masking measurement from another instrument that gave 47 mas instead of 28, with the position angle flipped by 180°.

## 2. Conventions

### 2.1 Definitions (virgil; decided 2026-10-03)
All quantities below are verified numerically against jaxoplanet, as §5.2's tests 1–6 will check.

| Quantity | Definition |
|---|---|
| Sky axes | `dra` positive East, `ddec` positive North, both in mas, as in `AGENTS.md` |
| Position angle | North through East, from the primary to the secondary |
| Third axis | `dz` positive **away from the observer**. (dra, ddec, dz) is then right-handed, and d(dz)/dt has the sign of the secondary's radial velocity relative to the primary (positive = receding) |
| Primary (reference) | The star at the origin of the relative vector, which is the scene's reference component (`flux=1`). It need not be the more massive star |
| Secondary | The orbiting star; r = secondary − primary |
| Flux ratio | The scene's `flux` of the secondary, relative to the primary, per band or spectrum (`AGENTS.md` flux convention). It belongs to the scene, not the orbit |
| `inc` | 0 ≤ i < 180°. **i < 90° means the position angle increases with time** (counterclockwise on the sky, North through East) |
| `Omega` (Ω) | PA of the **ascending node**, defined as the node where the secondary *recedes* (dz increasing). Visual data alone fix Ω only modulo 180° (§2.3) |
| `omega` (ω) | The **secondary's** argument of periastron, measured from the ascending node in the direction of motion. This is the visual-binary convention; the spectroscopic ω of the primary is ω − 180° |
| `dt_peri` | Time of periastron minus `t_ref`, in days. It is stored relative to `t_ref` because float32 resolves MJD ≈ 60000 to only about 0.004 d (§2.2); the MJD of periastron, t_ref + dt_peri, is reported in float64 |
| `period` | days (reported in years) |
| `a_mas` | Angular semimajor axis of the **relative** orbit, in mas |
| Tilt β | arcsin(dz/\|r\|), the line of centres' elevation out of the sky plane: positive when the secondary is the farther star |

### 2.2 Mapping to jaxoplanet
jaxoplanet's `OrbitalBody.relative_position(t)` returns (X, Y, Z) with X North, Y East and Z **toward** the observer.

| virgil | jaxoplanet |
|---|---|
| (dra, ddec, dz) | (Y, X, −Z) × a_mas / `semimajor` |
| `omega` (secondary) | `omega_peri` + 180° (jaxoplanet's ω is the primary's, the radial-velocity convention: with ω = 0 the relative periastron lies at the descending node) |
| `Omega` | `asc_node` (the receding node: identical) |
| `inc` | `inclination` (identical; i < 90° turns the position angle forward) |
| `dt_peri` | `time_peri` (days) |
| `radial_velocity(t)` | `radial_velocity` (positive = redshift, for the primary) |

**Pitfalls in jaxoplanet 0.1.0:**
1. **`parallax=` scales wrongly.** A 1 au orbit at a parallax of 0.1″ returns a separation of 4624, not 0.1. It multiplies by the au-to-R☉ factor instead of dividing (the code is marked `TODO`). **Never pass `parallax`.** Divide by `semimajor` and multiply by `a_mas` instead. Not yet reported upstream.
2. **Units are R☉, M☉ and days, with G built in.** Fixing `period` and a dummy central mass, then dividing by `semimajor`, gives positions in units of a, independent of mass.
3. **Time precision.** In float32, MJD ≈ 60000 resolves to only about 0.004 d. Always pass `mjd − t_ref`, with `t_ref` a static float64.

### 2.3 Symmetries, checked numerically
1. (Ω, ω) → (Ω + 180°, ω + 180°): identical sky positions, and dz changes sign. Visual orbits cannot tell these apart; radial velocities, or a scene component that is not front–back symmetric, can.
2. ω → ω + 180° alone: r → −r at all times. This is the same as swapping which star is the reference, so a "which star is primary" error and an "ω of which star" error are the same 180° flip.
3. i → 180° − i: reverses the sense of rotation.
4. For a face-on orbit, positions depend on Ω and ω only through their sum, the longitude of periastron ϖ = Ω + ω (at apastron, PA = ϖ + 180°). Astrometry of a nearly face-on orbit measures ϖ well, and Ω and ω separately poorly.

### 2.4 Thiele–Innes constants (in the conventions above, verified against jaxoplanet)
With X = cos E − e and Y = √(1 − e²) sin E:
- `ddec = A X + F Y`, `dra = B X + G Y`, `dz = C X + H Y`, where
- A = a(cos ω cos Ω − sin ω sin Ω cos i), B = a(cos ω sin Ω + sin ω cos Ω cos i),
- F = a(−sin ω cos Ω − cos ω sin Ω cos i), G = a(−sin ω sin Ω + cos ω cos Ω cos i),
- C = a sin ω sin i, H = a cos ω sin i.

With a Gaussian prior, A, B, F, G can also be integrated out analytically, which leaves a three-parameter posterior in (P, e, t_peri). That prior is not the usual one on the Campbell elements, so the result needs reweighting: see [`thiele_innes_marginalisation.md`](thiele_innes_marginalisation.md).

### 2.5 Why conventions matter: an example
Two real cases from Apep show what goes wrong.
1. **External elements disagree.** The JWST plume orbit (Ω = 164°, read as North through East) predicts a line of centres at PA 164°/344°. GRAVITY measures 96°.
   - If that Ω was measured counterclockwise from the East axis instead, PA = 90° − Ω ≡ 106°.
   - That map is a reflection, so it also reverses the predicted sense of rotation. Multi-epoch data can therefore test it.
   - The question is with the authors (`~/data/apep_gravity/notes/omega_convention_question.md`).
2. **Two instruments disagree by 180°.** Aperture masking (NACO) reported the companion at PA 274–278°, which is GRAVITY's axis reversed.

**The general lessons:**
- External elements enter as priors only after their convention is pinned. Until then, use an axial (mod 180°) prior (R5).
- Every instrument path needs a position-angle round-trip test (§5.3).

## 3. Engine: jaxoplanet, under a virgil orbit module
- **Use** `jaxoplanet.orbits.keplerian.Body` / `OrbitalBody` for positions, velocities and radial velocities. That way the Kepler solver (with its custom derivatives) and the radial-velocity sign live in one tested place.
- **Do not expose** jaxoplanet objects as fitted parameters. A virgil `KeplerOrbit` (an equinox/zodiax module) holds the §2.1 parameters, and builds a jaxoplanet `OrbitalBody` inside each call. This keeps the parameter paths ours (`"orbit.omega"`), confines the ω offset to one function, and insulates us from 0.x API changes.
- **Where it lives.** A new `src/virgil/orbits.py`, imported by `models` (where `Attached` goes, per `AGENTS.md`). The import order becomes `_utils` → `orbits` → `models`.
- **Dependency (decided).** jaxoplanet is **optional**: the extra `virgil-astro[orbits]`, imported lazily inside `orbits.py`, with an error naming the extra if it is missing. It needs only `jax` and `equinox`. Tests that use it skip when it is absent, and CI installs the extra.

## 4. Requirements and proposed interfaces

### R0. The common case: an orbiting companion, and good starting orbits
- **The companion.** A companion whose position follows the orbit is `Attached(PointSource(flux), orbit, anchor="secondary")` (R2). A `System` with the primary at the origin and that companion is the time-dependent analogue of `BinaryModelCartesian`. The existing binary classes stay as they are, for single epochs (`AGENTS.md`).
- **Starting orbits are analytic.**
  1. Fit each epoch with the existing binary tools (grids, `fit`, Laplace errors) to get positions and their covariances.
  2. On a grid of (P, e, t_peri), the positions are **linear** in the Thiele–Innes constants A, B, F, G (§2.4), so these are a weighted linear least-squares solve per grid point.
  3. The best grid points start the joint visibility fit (R7) or NUTS. This is the classical approach. It is cheap, it needs no random restarts, and it handles the multimodality of short arcs.
  4. The marginal version of the same solve (a Gaussian prior on A, B, F, G, and the log-determinant kept) ranks grid points by evidence and gives exact posteriors from positions: [`thiele_innes_marginalisation.md`](thiele_innes_marginalisation.md).
- **The fast route, when it is valid.** For two point sources well inside the field, per-epoch Laplace positions are close to sufficient statistics. Fitting the orbit to them (`PositionData`, R7) is then fast and nearly exact. With extended emission it is not, and the joint fit is required (R7).

### R1. An orbit that exposes the 3-D relative vector
```python
orbit = KeplerOrbit(period=..., dt_peri=..., ecc=..., inc=..., omega=...,
                    Omega=..., a_mas=..., t_ref=60500.0)  # t_ref static
dra, ddec, dz = orbit.relative(mjd)            # mas
vra, vdec, vz = orbit.relative_velocity(mjd)   # mas / day
orbit.frame(mjd)  # line_pa, towards_primary, line_tilt, node_pa, inc (deg)
```
- Alternative parameterisations (R4) are classes with the same methods, and each converts to `KeplerOrbit`.
- `to_jaxoplanet()` and `from_jaxoplanet(body, a_mas)` are the only converters. There are no converters to other orbit packages.

### R2. Components attached to the binary frame
```python
disc = Attached(ModulatedGaussianRim(...), orbit, anchor="secondary",
                bind={"pa": "node_pa", "inc": "apparent_inc",
                      "az_pas": "towards_primary"},
                offsets={"az_pas": 0.0})
scene_at_t = disc.at(mjd)   # the component, with dra, ddec and angles set
```
- **Anchor.** `"primary"` (origin), `"secondary"` (at r), `"barycentre"` (at r·q/(1+q), which needs a mass ratio), or a fraction along r.
- **Bind.** This maps the component's angle attributes to angles of the binary frame:

  | Frame angle | Meaning |
  |---|---|
  | `line_pa` | PA of the line of centres, primary → secondary |
  | `towards_primary` | `line_pa` + 180° |
  | `line_tilt` | β |
  | `node_pa` | Ω, the orbital plane's line of nodes |
  | `inc` | the orbit's physical inclination, 0–180° |
  | `apparent_inc` | arccos(|cos i|), 0–90°: the projected tilt, for components such as `ModulatedGaussianRim` whose `inc` is an apparent inclination (they use cos(inc) clipped at zero, so binding a retrograde i = 130° there would render nearly edge-on instead of at 50°) |

  `offsets` adds fitted offsets, such as a skew.
  - **Sky angles bound to a component's own angles must be converted into its frame.** `ModulatedGaussianRim` measures its azimuthal angle `az_pas − pa` in the deprojected disc, then compresses the minor axis. So a sky position angle φ (such as `towards_primary`) becomes az = pa + atan2(sin(φ − pa)/cos i, cos(φ − pa)), after `pa` and `inc` are bound. The sketch does this, and checks it on a rendered image: the disc's flux-weighted centroid points at the primary.
  - **Examples.** A disc in the orbital plane binds `pa → node_pa` and `inc → apparent_inc`, as in the sketch. For a retrograde orbit, which side of the disc is near is then set by the PA binding, not by the inclination. Apep's dust cone binds its axis `pa → line_pa` (plus a 0.4° skew) and its tilt `tilt → line_tilt`. That gives the tilt's sign, which an optically thin cone cannot measure, from the orbit.
- **Orbital skew.** The default is a fitted offset. Optionally, a physical aberration: the axis follows v_wind n̂ − Δv⊥, with Δv the stars' relative velocity. That needs physical velocities and so a distance (R6). It lies in the orbital plane, so for a nearly face-on orbit it shows up as a PA offset. It is small for wide orbits but dominant near periastron in eccentric colliding-wind binaries (WR 140). Its sign needs its own test before it is offered.
- **Time dependence goes through a snapshot method.** Any `SourceModel` may implement `at(mjd) -> SourceModel`. The default returns `self`; `System.at` maps over its children. Model signatures do **not** gain an `mjd` argument, so the binary classes and every static scene keep the current fast path.

### R3. MJD per datum
- `OIData.mjd` and `OIData.frame`, per baseline sample, are specified in the spectro note §2.5. Closure phases inherit their time through `i_cps*`. Their three legs share one frame time, which is needed so that a moving model's closure phase is self-consistent.
- **Evaluation.** `OIData.model` checks whether the model is time-dependent. If it is, it evaluates `scene.at(t_f)` once per unique frame time and maps over frames with `jax.vmap` on padded per-frame index arrays. Otherwise nothing changes. The sketch does the simplest version: one snapshot per sample.
- **Other data types.** AMI and AMIGO records carry one time per file or integration; kernel-phase data likewise. `OIData.epochs()` groups frames for per-epoch nuisances (spectro note §2.5).
- **Motion within one exposure** (time smearing) is ignored. Add an option only when an orbit moves a significant fraction of a fringe during one exposure.

### R4. Parameterisations for short arcs
Sampling Campbell elements (a, e, i, ω, Ω, T, P) over an arc that is a small fraction of the period is pathological: most elements are unconstrained, and correlated along curved ridges. (Apep: 1.8 yr of a 193 yr orbit.) Three classes, all converting to `KeplerOrbit`:

| Class | Parameters | When |
|---|---|---|
| `KeplerOrbit` | P, dt_peri, e, i, ω, Ω, a_mas; samplers see √e cos ω, √e sin ω, cos i, and Ω ± ω for nearly face-on orbits | Well-covered orbits; external element priors |
| `ThieleInnesOrbit` | A, B, F, G (+ C, H, if dz is wanted), P, dt_peri, e | Positions are **linear** in A, B, F, G at fixed (P, e, t_peri), which gives the starting orbits of R0 and good sampling geometry; the Ω/ω degeneracy at small i disappears |
| `StateVectorOrbit` | At `t_ref`: (dra, ddec) [mas], (vra, vdec) [mas/yr], and dz, vz; plus μ = 4π² a_mas³/P² [mas³ yr⁻²] | **Short arcs.** The measured quantities (position and its rate) are parameters with near-Gaussian posteriors; dz, vz and μ carry the physical priors. Converted to elements analytically, by the standard state-vector → elements relations written from first principles |

- Notes on `StateVectorOrbit`:
  - μ is distance-free, and couples to mass only through D (R6).
  - The conversion has removable singularities at e = 0 and i = 0. Use the (h, k) = e(sin ϖ, cos ϖ) and (p, q) = tan(i/2)(sin Ω, cos Ω) forms internally, with tests near both.
  - Unbound states are outside the prior support.
- **What a short arc constrains.** The rate of the position angle does not depend on the angular scale, so it is free of a, M and D. It does depend on the projection (i, ω, Ω) as well as on e, P and the orbital phase: an edge-on orbit has no continuous PA rate at all. So a short arc constrains a combination of phase and projection, and the sense of rotation unless the orbit is close to edge-on. The short-arc parameterisation and its priors must not assume more.

### R5. External priors and radial velocities
- **Element priors** are numpyro distributions on `KeplerOrbit` paths, as for any parameter. For other parameterisations, a prior on a *derived* element goes in through a model function plus a log-density term. Today that is a regulariser in `fit`, and it must also be accepted by `numpyro_model` as a genuine prior. Document that such a prior does not include the Jacobian of the reparameterisation.
- **Ambiguity modulo 180°.** When an external source only knows Ω modulo 180° (or its convention is in doubt; §2.5), the prior is axial: a von Mises on 2Ω (an `axial_von_mises(mean, kappa)` helper).
- **Radial velocities.** `RVData(mjd, rv, d_rv, star="primary" | "secondary")` adds a likelihood term from jaxoplanet's `radial_velocity`. It needs the mass ratio q, the systemic γ, and a physical scale (M_tot or D; R6). Its σ terms reuse `noise=`-style inflation (`rv_jitter`). RVs are the only data that fix Ω absolutely (§2.3, symmetry 1).

### R6. Distance and mass as derived quantities (dynamical parallax)
- **Native parameters:** a_mas and P, which are distance-independent. Derived: M_tot = (a_mas · D)³ / P², with a in arcsec, D in pc, P in yr and M in M☉.
- **An optional `distance_pc` parameter with a prior.** Only distance-dependent terms use it: RV amplitudes, physical skew velocities, expansion speeds of ejecta, luminosities. **Keep distance-free constraints separate from distance-dependent ones.** For example, a period from the angular spacing of dust shells is distance-free, but a dust speed in km/s is a proper motion times D.
- **Report** a posterior on D, or a curve of D against M_tot when M_tot is only bounded. Never report a single number.
- **Example.** Apep's 28 mas separation near apastron, with 20–40 M☉, gives D ≈ 5.5–7 kpc, which changes the physical interpretation of its dust speeds.

### R7. Fits across instruments
- **One scene and one orbit for all datasets.** Each dataset has its own:
  - **bandpass and resolution:** 6a's smearing;
  - **error model:** `noise=` terms, or Stage 6d's correlated nuisances;
  - **wavelength scale:** `wavel_scale` (spectro note §2.6);
  - **closure-phase sign:** a per-instrument test (§5.3), and never a free sign parameter.
- **Interface:** this exists already. `fit(scene_fn, priors, [data_1, ..., data_n], noise=[...])`, where `scene_fn` returns one time-dependent scene (or one per dataset) and each dataset's MJD drives `at`.
- **Fit visibilities, not derived positions,** whenever the scene is more than two point sources. At ~λ/D resolution, a two-point fit to a star plus bright extended emission returns a biased "companion" position, and can swap which source is brighter. Apep's simulated GRAVITY scene seen through a NACO-like mask gave 32–87 mas for a true 28 mas, depending on band and model.
- `PositionData(mjd, sep, pa, cov)` is for the R0 fast route, and for published positions with no raw data. It **warns** that it ignores the scene.

### R8. `simulate(scene, template)`: bias tests and planning across instruments
```python
fake = simulate(scene, template, errors="template", key=key,
                mjd=None, noise=None)
```
- **Inputs.**
  - `template` is any `OIData`: real data from another instrument, or `coverage.vlti_oidata` / `nrm_oidata` / `ami_grid_record`.
  - The scene is evaluated at the template's own MJDs, through its bandpass and smearing.
  - **Errors:**
    - `"template"`: the template's σ;
    - a `noise` dict: inflated σ, with nuisances *drawn*, including correlated blocks and a wavelength scale;
    - `None`: noiseless.
  - `mjd=` shifts or replaces the epochs, for planning.
- **Built on** `OIData.with_model`, which already preserves sampling, closure indices and operators. The new parts are time dependence (R3), drawing nuisances, and 6a's smearing.
- A companion helper, `bias_test(scene, template, fit_model, priors, n)`, reports the distribution of fitted parameters over noise draws. **Uses:**
  - What would instrument B measure for the scene that instrument A sees?
  - What orbital phase coverage constrains P?

### R9. Scenes that are static in the co-rotating frame, and scenes that change
| Hypothesis | Model |
|---|---|
| Static on the sky | Components not attached; the companion alone moves |
| Static in the co-rotating frame | Components `Attached` to the orbit, with shared shape parameters |
| Changing | `Attached`, with per-epoch shape parameters θ_e = θ + δ_e, δ_e ~ N(0, τ²) and τ fitted (per-dataset model function plus a hierarchical prior) |

τ → 0 recovers co-rotation, so the posterior on τ (or the evidence) tests "static in the co-rotating frame" against "changing".

## 5. Test plan

### 5.1 Reference ephemerides
1. **An independent evaluator.** A NumPy-only implementation (SciPy's root finder for Kepler's equation, Thiele–Innes, §2.4) against `KeplerOrbit.relative` over a grid of (e, i, ω, Ω, phase), in float64 to 1e-10 relative. In float32 with `t_ref`, to 1e-5. This is how §2.4 was checked.
2. **A published visual binary with radial velocities, so that Ω is absolute.** α Cen AB is proposed (Pourbaix & Boffin 2016, A&A 586, A90; Akeson et al. 2021, AJ 162, 14). Test the predicted (separation, PA) at the published epochs, with each paper's stated conventions mapped to §2.1. **The values must be copied from the papers when the test is written, not from memory.**
3. **An interferometric visual binary** with a published VLTI or CHARA orbit, to check an interferometrist's convention against ours. **To be chosen; see the reminder in §7.**

### 5.2 Convention unit tests (fast)
1. i < 90° ⇒ PA increases. i → 180° − i reverses it.
2. (Ω + 180°, ω + 180°): the same (dra, ddec), and dz flips.
3. ω + 180°: r → −r. This equals swapping the primary and secondary.
4. dz increases (the secondary recedes) at the node with PA = Ω, and `relative_velocity`'s z component matches the finite difference of dz.
5. Round trips: `KeplerOrbit` ↔ `ThieleInnesOrbit` ↔ `StateVectorOrbit` ↔ `to_jaxoplanet` / `from_jaxoplanet`, including near e = 0 and i = 0.
6. jaxoplanet's ω offset and RV sign: our secondary ω = jaxoplanet ω + 180°. The primary's radial velocity is positive while the secondary approaches.
7. **`Attached` orientation** (in the spirit of `AGENTS.md`'s orientation tests): an attached elongated component at a known orbital phase renders with its axis at the bound frame angle. Its `render()` Fourier-transforms to its `model()`.
8. **The R0 starting orbits:** the Thiele–Innes linear solve on noiseless positions recovers the true orbit on the grid point nearest the truth.

### 5.3 OIFITS closure-phase signs: round trips to catch 180° flips
Closure-phase sign conventions (OI_T3 baseline order, the instrument's conjugation, the sign of u and v) and the choice of "brighter" star each flip a binary by 180°.
1. **Synthetic.** A binary with unequal fluxes, so that a flip is visible, is simulated on long-baseline, masking and AMI coverage. It is then written through each available writer:
   - virgil `write_oifits`;
   - AMICAL's OIFITS writer, when AMICAL is installed (skip otherwise);
   - a GRAVITY-layout file. For this, the column layout and `STA_INDEX` ordering of a real GRAVITY product are copied, and its data replaced.

   Each file is read with `read_oifits`, refitted, and checked: PA within 1° of the truth (not 180° off), and the flux ratio below 1.
2. **Real anchors, one per instrument path:** a binary with a well-known orbit observed with GRAVITY, and one with NACO/SPHERE masking (reduced by AMICAL). Each is fitted with virgil and its PA compared with the orbit's prediction. Only a real anchor tests the *pipeline's* convention, as opposed to our writer's. **The targets are not chosen yet; see the reminder in §7.**
3. **The OIFITS v2 sign convention.** Check `read_oifits`'s phase sign against the OIFITS v2 standard (Duvert et al. 2017) explicitly, once, in a docstring and a test.

### 5.4 End to end
1. **Simulated systems first,** through `simulate`:
   - a visual binary on long-baseline coverage;
   - a faint companion on AMI or masking coverage over a short arc;
   - a binary with an attached extended component on two instruments' coverage.

   Each joint fit recovers the identifiable combinations. The two-point fit on the lower-resolution instrument shows the bias of R7.
2. **Then real data.** Apep is a candidate: GRAVITY 2023–25, NACO 2016 and 2019 once re-reduced, and the JWST plume elements as axial priors. Its scripts stay in `~/data/apep_gravity`.

## 6. Mapping and effort

| Item | Goes in | Effort (agent h) | Depends on |
|---|---|---|---|
| `orbits.py`: `KeplerOrbit`, `ThieleInnesOrbit`, converters to and from jaxoplanet, the `[orbits]` extra; tests 5.1.1 and 5.2 | Stage 6a.1 | 4–5 | — |
| Starting orbits: per-epoch positions → Thiele–Innes grid solve; `PositionData` | Stage 6a.1 | 2–3 | `orbits.py` |
| `StateVectorOrbit` and its regular forms | Stage 6a.1 | 3 | `orbits.py` |
| `SourceModel.at` and time-dependent `OIData.model` | Stage 6a.1 | 3–4 | `OIData.mjd`/`frame` (spectro §2.5) |
| `Attached` (anchor, bind, offsets); orientation test | Stage 6a.1 | 3 | `at`, `orbits.py` |
| Physical skew (aberration) | Stage 6a.1, later | 2 | `Attached`, R6 |
| `RVData`, axial priors | Stage 6a.1 | 2 | `orbits.py` |
| `distance_pc` and derived mass; reporting | Stage 6a.1 | 1 | — |
| `simulate`, `bias_test` | Stage 6a.1 | 2–3 | `at`; 6a smearing optional |
| A `TruncatedCone` component (a thin conical shell of J₀ rings, from the Apep analysis), with an elliptical cross-section option and a render ↔ model test | Stage 6a.1 | 3–4 | — |
| PA round-trip tests (5.3.1, 5.3.3); real anchors (5.3.2) | Stage 6a.1 | 2, then 2 per anchor | AMICAL optional |

## 7. Decisions and open questions

### Decided (Ben, 2026-10-03)
1. **Orbits are built in virgil,** not in a separate package or by another contributor.
2. **jaxoplanet is an optional dependency** (`virgil-astro[orbits]`; §3). No orbitize! or orvara: neither as converters nor as dependencies.
3. **The user-facing conventions are those of §2.1:**
   - ω is the secondary's;
   - Ω is the PA of the receding node;
   - dz is positive away from the observer;
   - i < 90° means the PA increases.

   jaxoplanet's conventions stay internal to `KeplerOrbit`.

### Real anchor binaries (Ben will find them in the ESO archive; not blocking)
Before the tests in §5.1.2–3 and §5.3.2 are written, choose:
- a visual binary with radial velocities, so that Ω is absolute (α Cen AB is proposed);
- an interferometric visual binary with a published VLTI or CHARA orbit;
- a binary with a well-known orbit **observed with GRAVITY**;
- one **observed with NACO or SPHERE masking** (reduced with AMICAL).

The synthetic round trips (§5.3.1, §5.3.3) and the unit tests (§5.2) do not wait for these.

### Still open (defaults in force until Ben says otherwise)
1. **Time dependence:** the snapshot method `at(mjd)` (default), not an `mjd` argument on every `model()`.
2. **Physical orbital skew:** a fitted offset only (default), until a system near periastron needs the physical skew.
3. **End to end:** simulated systems first, then real data (default).
4. **Filing the jaxoplanet `parallax=` bug upstream:** ask first.
