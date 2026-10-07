# Orbit fitting: published practice and a critical review of virgil's plan

Status: **review**, 2026-10-07, with Part 1 revised after a literature
check against full-text methods sections (see the tags at the start of
"Evidence base and its limits"). Read-only analysis; nothing here is
implemented. Part 1 summarizes how published work fits visual and
interferometric binary orbits. Part 2 assesses virgil's current tools
(`epoch_positions`, `start_from_positions`, `rank_orbits`,
`gap_marginal`, #281) and the plan in `design/automatic_orbits.md`
(#279) against that practice and against our own evidence. Part 3
lists prioritized improvements. Spelling is Oxford (-ize).

## Evidence base and its limits

Revised 2026-10-07 after a literature check. Every item below is tagged:

- **[R]** the methods section was read in full text (arXiv or the
  publisher PDF) and the statement is taken from it. The bibcode was
  confirmed in ADS.
- **[A]** the bibcode and title were confirmed in ADS, but the paper
  was not read for the point at issue. The statement rests on another
  paper's description of it, which is named.
- **[U]** unverified, from memory. Do not cite publicly.

Section numbers are those of the arXiv/journal version. Xuan et al.
2024 is a Nature paper with unnumbered sections, so it is cited by
heading. The earlier draft of Part 1 came from a subagent that read
no methods section. Several of its claims were wrong and are
corrected below (see the notes in 1.3 and 1.4).

Part 2 rests on our own evidence:

- the Gl 229 Ba–Bb search failure (#268, comments 6029737065 and
  6029854711);
- the retrograde start that ran for 3.5 h;
- the tutorial result that two-step fits with the scale marginalized
  match the joint fit;
- the PR A cross-check simulations (#281), which did not reproduce the
  per-night reordering.

## Part 1. Published practice

### 1.0 Xuan et al. 2024: the closest precedent

Xuan et al. 2024, Nature 634, 1070 (2024Natur.634.1070X,
arXiv:2410.11953) [R]; the orbit analysis is given in detail in
Thompson et al. 2025, AJ 169, 193 (2025AJ....169..193T,
arXiv:2502.05359) [R]. Both are on Gl 229 Ba–Bb with VLTI/GRAVITY (UTs,
wide dual-field mode, K band) and VLT/CRIRES+ RVs.

**What was fitted.**

- *The orbit was fitted to GRAVITY closure phases directly, jointly
  with the CRIRES+ RVs, not to per-epoch positions* (Xuan, Methods,
  "Orbit fits"). The stated reasons are that this carries the multiple
  possible positions at each epoch into the fit and avoids
  intermediate products, so noise properties are preserved.
- Per-epoch positions were derived for the first epoch only, in the
  main text, with a grid search (their ref. 19, CANDID-style). That
  gave a single-source reduced χ² of 55 (288 degrees of freedom) and a
  binary reduced χ² of 1.27, a companion about 5 mas south and a flux
  ratio near 0.5. That fit included linear motion of the companion
  over the 2.5 h window, 4.6 mas/day (main text, Fig. 1c).
- Wavelengths: PMOIRED used 2.05–2.18 µm binned to six points;
  Octofitter used 2.025–2.15 µm unbinned (Methods, "Orbit fits").
- Reduced χ² of the joint fit is 2.2 (513 degrees of freedom) (main
  text). Errors were not rescaled to 1.

**How the search was started and ambiguities handled.**

- *PMOIRED (the adopted baseline values).* Gradient descent, first on
  the RV data alone, then on the joint RV + closure-phase model. The
  RV-only fit gave P = 12.12 ± 0.04 d, e = 0.22, q = 0.91 (Methods).
  So the visibility-level fit was **seeded from an RV-only orbit**.
  Errors come from 5000 bootstrap resamples, each re-fitted from a
  first guess drawn around the best values with four times the
  uncertainty. Closure phases of the same date and triangle are drawn
  together, to respect their correlation.
- *Octofitter.* Non-reversible parallel tempering "to search the entire
  multi-modal parameter space globally" (Methods). Closure phases are
  first converted to non-redundant kernel phases per wavelength. A
  kernel-phase "jitter" term is added per epoch (five in total). The
  joint fit has a single mode.
- *Validation.* On a binary star with high-quality GRAVITY data and
  UVES RVs (system not named, no numbers shown), a closure-phase
  orbit fit and an orbit fit to per-epoch separations and position
  angles "yielded the same result" (Methods). This is the only
  direct-versus-two-step check we found (see 1.3).
- *Posterior-predicted positions are not for reuse.* Their Extended
  Data Table 4 positions are outputs of the joint fit, and the authors
  say orbit fits should use the closure phases, not those positions.
  Any attempt of ours to compare against Xuan's per-epoch positions
  must keep that caveat.

**What the follow-up shows about ambiguity (Thompson et al. 2025
[R]).** This is the closest published evidence for our Gl 229 failure.

- §3.5: single GRAVITY exposures with the UTs are consistent with many
  companion locations and contrasts. A single-epoch grid over 2D
  positions is feasible, but for multi-epoch data the dimensionality is
  too high, so MCMC is used. NUTS/HMC is judged unsuitable for widely
  separated modes. Non-reversible parallel tempering (Pigeons,
  gradient-free slice sampler locally, with a variational reference)
  is used; it brought sampling "from months or more to only 1-2
  weeks".
- §4.1.1 (Figs. 2 and 3): with the **kernel-phase data alone the
  posterior has four modes**. It is bimodal in period (11.032 or
  12.137 d, nearly equal weight) because on two of the epochs with the
  least uv coverage (2024-02-28 and 2024-03-29) the data are
  consistent with two position angles at essentially the same
  separation. It is also bimodal in ω and Ω from the usual sign
  ambiguity. **Only the RVs resolve this**: the RV-only posterior
  matches exactly one of the four modes. The RV-only fit in turn has an
  inclination bimodality that the kernel phases resolve.
- §4.1.2: allowing the contrast to vary per epoch gave a markedly
  higher contrast at epoch 1, which they attribute to systematics.
- §3.3, Table 3: the orbit parameters are P ~ U(5, 20) d, e ~
  U(0, 0.999), sin i prior, ω and Ω ~ U(0, 2π), τ ~ U(0, 1) (with a
  reference epoch), flux ratio f ~ U(0, 1), a total-mass prior
  N(71.4, 0.6) M_Jup from G. M. Brandt et al. 2021 (2021AJ....162..301B
  [A]), five per-epoch kernel-phase jitters, and a spectral
  correlation C_z. They do not use (√e cos ω, √e sin ω).
- §3.1: the covariance of closure phases sharing baselines is rank
  deficient. They project to three kernel phases per wavelength
  (their Eq. 2).

**Consequences for us (see the notes at the end of Part 3).**

1. Our free search, with no RVs, is in Thompson's kernel-phase-only
   regime. A four-mode posterior, with a period alias, is the
   *published expectation* for these data, not evidence that our
   likelihood is wrong.
2. The fit that reproduces Xuan is effectively RV-seeded. Seeding is
   what the PMOIRED baseline did.
3. A flux-ratio bound f ≤ 1 is the published choice (Table 3).

### 1.1 Global search and initialization

- **Thiele–Innes (TI) linear split.** Grid the nonlinear parameters
  (P, e, T₀ or τ) and solve A, B, F, G by linear least squares.
  - Lucy 2014 (2014A&A...563A.126L) [R, grep level] formulates this on a
    3-D grid in (P, e, τ) for visual binaries with incomplete orbits
    (§2.1, §2.4).
  - Mendez et al. 2017 (2017AJ....154..187M) [R, grep level] state that
    this dimensionality reduction was used before by Hartkopf et al.
    1989 in their "grid search" (footnote 6, §3.3). Hartkopf et al. 1989
    (1989AJ.....98.1014H) was confirmed in ADS only [A].
  - Gaia DR3 does this at scale: Halbwachs et al. 2023 (§5.1) [R]
    use Levenberg–Marquardt from a grid in (P, e, T₀) because "a
    calculation starting from e = 0" works only if the period is
    close. Holl et al. 2023 (§4.1) [R] explore P, T₀, e and a jitter
    with DE-MCMC and solve A, B, F, G by QR at each step.
  - Wright & Howard 2009 (2009ApJS..182..205W) [A] is the usual
    reference for the TI form in the exoplanet literature; Holl 2023
    and Octofitter cite it.
- **Rejection samplers over the full prior.**
  - OFTI (Blunt et al. 2017, 2017AJ....153..229B) [R]: §2.1.1 draws
    all seven elements from priors (sin i, uniform ω and T₀, a
    descending or uniform e prior); §2.1.2 scales a and rotates Ω so the
    orbit passes through one astrometric point; §2.1.3 rejection
    samples. §2.3 says it is most efficient for arcs covering a
    small fraction of the orbit (typically below about 15%), i.e. it
    is slow for well-constrained long arcs.
  - The Joker (Price-Whelan et al. 2017, 2017ApJ...837...20P) [R]:
    §2. Four nonlinear parameters (ln P, e, ω, φ₀) are sampled densely
    from the prior and the two linear ones (K, v₀) are marginalized
    analytically, with rejection on the marginal likelihood. It keeps
    all modes. If too few samples survive and they span a period
    range larger than Δ = 4P²/(2πT), it draws more prior samples;
    only if they are within Δ does it hand over to emcee. Priors are
    uniform in ln P, Beta for e (Kipping 2013, 2013MNRAS.434L..51K
    [A]), uniform in ω and φ₀. It is RV-only.
  - orbitize! (Blunt et al. 2020, 2020AJ....159...89B) [R]: OFTI plus
    ptemcee MCMC (§2.3.1, §2.3.2). Positions only. §4.2 lists RVs,
    other orbital parameterizations and HMC as future work.
  - orvara (Brandt et al. 2021, 2021AJ....162..186B) [R]: relative
    astrometry, RVs and Hipparcos–Gaia astrometry, sampled with
    ptemcee (§4.3). Starting values per walker are drawn from
    user-specified Gaussians (§6).
  - Octofitter (Thompson et al. 2023, 2023AJ....166..164T) [R]:
    initializes NUTS by drawing 250,000 samples from the prior and
    starting at the best one. They say this is more robust than an
    optimizer for multi-modal posteriors (§3.4). It also supports
    parallel tempering (Thompson 2025, above).
- **Common pattern.** Every method treats (P, e, T₀) by grid or
  sampling and the rest as linear or analytic, or uses tempering, and
  **keeps many modes**. Two-step interferometric papers (1.3) instead
  pick one position per night.
- **Period-grid density.** Three published rules:
  - The Joker (§2): resolution Δ = 4P²/(2πT), from the expected
    frequency resolution 1/T, widened by 4.
  - ARMADA II (Gardner et al. 2022, 2022AJ....164..184G, §3) [R]:
    P = 2fT/k with oversampling f = 3, minimum period 2 d, citing
    Muterspaugh et al. (their "2010a"; which paper is unverified
    [U]).
  - Gaia DR3 (Holl 2023, §4.1) [R]: uniform grid up to twice the
    time span, then local χ² minima seed the chains. Halbwachs
    (§5.1) [R]: period range 10 d to the observation span divided by
    0.6; the grid step is not given.

### 1.2 Fitting, parameterization and priors

- **Optimizers.** Levenberg–Marquardt from good starts: Le Bouquin
  et al. 2017 (2017A&A...601A..34L, §2.4) [R] say "a more complex
  search is not necessary" because the sampling pins down the period.
  Halbwachs 2023 (§5.1) [R]. Lester et al. 2019
  (2019AJ....157..140L, §4.2) [R] use the Newton–Raphson of Schaefer
  et al. 2016 (2016AJ....152..213S [A]). ARMADA uses lmfit on TI
  elements (§3 of both papers) [R].
- **Samplers.**
  - emcee (Foreman-Mackey et al. 2013, 2013PASP..125..306F [A]) and
    ptemcee (Vousden et al. 2016, 2016MNRAS.455.1919V [A]) are the
    defaults in the imaging-orbit codes. orbitize! (§2.3.2) [R] says
    the plain affine-invariant sampler "generally fails to fully
    sample" multi-modal posteriors without good starting points, and
    that ω and Ω each show two equal peaks 180° apart when there are no
    RVs, which is a convergence check.
  - ARMADA I/II [R]: lmfit, then emcee with uniform priors, walkers
    started around the best fit.
  - Octofitter: NUTS by default (§3.4); non-reversible parallel
    tempering via Pigeons for the GRAVITY multi-modal case
    (Thompson 2025, §3.5). Non-reversible tempering is from Syed et al.
    (2019arXiv190502939S [A]).
  - Nested samplers (dynesty and others): no verified source for their
    use in orbit fitting beyond Xuan's use of nested sampling for the
    atmosphere fit (Methods). Claims about cost with dimension are
    **[U]**.
- **Parameterization.** Conventions differ, so none is "the standard".
  - orvara (§6, Table 4) [R]: √e sin ω, √e cos ω, mean longitude at a
    reference epoch λ_ref, Ω, a log-flat, sin i prior.
  - Octofitter paper (§3.2, §4.1, App. B) [R]: Campbell elements with
    uniform e, or TI elements with a log-uniform scale. Instead of τ it
    uses the position angle θ at the mean epoch, which it finds
    improves convergence "since θ is directly constrained". It notes
    that orvara uses √e sin ω while orbitize! uses uniform e and ω
    (§4.1).
  - orbitize! (§2.1, Eq. 1) [R]: τ, the orbit fraction at a reference
    epoch, with prior bounds 0–1 whatever the period.
  - Thompson 2025 (Table 3) [R]: uniform e (0–0.999), uniform ω and Ω
    over 0–2π, τ.
  - The Joker (§2) [R]: ω and φ₀ uniform, Beta e.
  - Without RVs, (ω, Ω) → (ω + 180°, Ω + 180°) is an exact degeneracy.
    Conventions: report Ω < 180° (ARMADA I §3, II §3; Le Bouquin 2017
    §2.4; all [R]); orbitize! §2.3.2 uses the symmetric double peak as
    a check.
  - Near-circular orbits: Halbwachs §5.2 [R] "pseudo-circularize"
    e < 0.0005 because T₀ and ω become ill-defined. ARMADA I §3 [R]
    reports a time of maximum RV for the same reason.
- **Priors.** Log-uniform in period or scale, isotropic orientation
  (sin i), uniform or Beta e. The Joker, OFTI, orvara and Octofitter
  all do this. This agrees with our Jeffreys-prior rule.

### 1.3 Direct (visibility-level) fits versus two-step fits

- **Both are in use. The literature is more mixed than the earlier
  draft said.**
  - *Direct, visibility or closure-phase level, orbit fitted to the
    interferometric observables:* Xuan et al. 2024 and Thompson et al.
    2025 [R] (GRAVITY closure phases or kernel phases, jointly with
    RVs); Octofitter (Thompson 2023 §2.3, §3.2 [R]) supports squared
    visibilities and closure phases directly, with Gaussian diagonal
    covariances, and demonstrates it on *simulated* JWST/AMI
    closure phases plus RVs (§4.4); PMOIRED (Mérand 2022,
    2022SPIE12183E..1NM [A]) is the multi-epoch tool Xuan used.
  - *Two-step, per-epoch positions then orbit:* ARMADA I and II [R];
    Lester et al. 2019 [R]; Le Bouquin et al. 2017 [R]; Gallenne et
    al. 2023 [R] (GRAVITY, with CANDID); Klement et al. 2025
    (2025A&A...694A.208K, §3.1, §4.1) [R] (GRAVITY, PMOIRED per
    snapshot, then orbfit-lib).
- **How the two-step papers get positions.**
  - ARMADA II (§2.3) and I (§2.2) [R]: closure phase and differential
    phase (MIRC-X); visibility amplitude also for GRAVITY. A grid in
    (ΔRA, ΔDec) at 0.1 mas steps for the minimum χ², then lmfit. The
    visibilities are poorly calibrated, so squared visibilities are
    not used for MIRC-X. A third component is searched by a second grid.
  - Lester 2019 (§3.2) [R]: V² and closure phase; a grid with MPFIT
    at each point of a wide grid of separations; error ellipse from
    χ² ≤ χ²_min + 1.
  - Gallenne et al. 2023 (§2.2) [R]: CANDID. An N×N grid of fits, with
    the minimum grid resolution estimated afterwards. Bootstrap (10,000
    samples) for the error ellipse. Gallenne 2015 (2015A&A...579A..68G,
    §2.1) [R] states that a coarse grid can miss the detection and that
    the grid should be fine enough that several starting points reach
    the same local minimum.
  - Le Bouquin 2017 (§2.3) [R]: PIONIER, with orbital motion within a
    night *neglected* because P ≈ 0.5–15 yr; ellipses from the fit.
- **Per-epoch ambiguity.**
  - Lester 2019 (§3.2) [R]: on two nights with only two brackets the
    global χ² map had several solutions within χ²_min + 1. They fitted
    a preliminary orbit to nights with three or more brackets and
    **chose the solution nearest its prediction**. This is a committed
    mode, resolved by a held-out-style prediction, close to our
    held-out-epoch check.
  - Thompson 2025 (§4.1.1, Fig. 3) [R]: two position angles equally
    likely at two epochs; solved by carrying both into the orbit fit
    (direct fit, tempering) and by RVs.
  - ARMADA I (§4) [R] mentions tentative detections that "appeared
    to be related to residual structure in the χ² maps".
- **Errors.**
  - ARMADA II (§2.3, §3) [R] map the 2D surface where the **reduced** χ²
    rises by 1, call that conservative, and then *scale the errors*
    "so that each independent dataset contributes a final orbital fit
    χ²_red = 1". Lester 2019 (§4.2) [R] rescales by a factor 5.6 for the
    same reason. This is the tautological rescaling that our
    report-raw-χ² rule rejects, and it is common practice here.
  - Xuan and Thompson [R] do not rescale: a bootstrap that respects
    correlated closure phases (PMOIRED; Lachaume et al. 2019,
    2019MNRAS.484.2656L [A]) or per-epoch kernel-phase jitter terms and
    a spectral-correlation parameter C_z (Octofitter). Kammerer et
    al. 2020 (2020A&A...644A.110K [A]) is the source of the
    correlation model they adopt (as described by Thompson §3.1).
  - Gaia: Halbwachs §2.2.1 [R] computes F2 from raw χ² and also
    rescales the uncertainties by c, but uses a large F2 as a rejection
    flag (1.4).
- **Is there a direct comparison of visibility-level and two-step
  fits?** See the answer to question 3 at the end of this section.

### 1.4 Automated pipelines at scale

Gaia DR3 non-single stars.

- *Halbwachs et al. 2023* (2023A&A...674A...9H) [R].
  - §2.2.1: goodness of fit F2 = (9ν/2)^{1/2}[(χ²/ν)^{1/3} + 2/(9ν) − 1].
    It is N(0, 1) for an adequate model, and they checked by simulation
    that this holds for the orbital model when a₀ is much larger than
    its error. Purely astrometric solutions with uncorrected F2 > 25 are
    "questionable" and not retained in the post-processing.
  - §2.2.2: significance s as the length of the extra vector over its
    error.
  - §5.1: a grid over (P, e, T₀) with LM from each point; period range
    10 d to span/0.6.
  - §5.3 and Table 1: the orbital solutions pass if a₀/σ_a₀ >
    max(5, 158/√P_days), F2 < 25, a parallax-significance condition of
    the form ϖ/σ_ϖ > 20 000/P_days (the extracted PDF text of this
    cell is garbled; check against the paper), and for eccentric
    orbits σ_e < 0.079 ln P_days − 0.244. Orbits are also compared with
    Gaia spectroscopic orbits, with a mass-function sanity check, and
    with the validation of Babusiaux et al. (§5.3).
- *Holl et al. 2023* (2023A&A...674A..10H) [R].
  - §4.1: DE-MCMC with TI linearization and a period search to seed it;
    Gelman–Rubin for convergence.
  - §4.2.1: a genetic algorithm initialized with a Fourier-based
    frequency analysis (Delisle & Ségransan 2022, 2022A&A...667A.172D
    [A]); half the population from analytic e and M₀, half uniform.
  - §5.1.2: force-fitted stochastic solutions were "primarily" of dubious
    quality "due to known aliasing effects with scanning law
    periodicities". The filter: fractional parallax difference below
    5%, a₁/σ_a₁ > 20, an excess-noise ratio above 20, more than 36
    transits, parallax above 0.1 mas.
  - §5.2.2: for the targeted search, BIC(Kep) − BIC(5-par) and − BIC(7-par)
    below −30, a₁/σ_a₁ > 2 or > 5 in two alternative filter sets, P less
    than the time span, mass function below 0.02 M_sun, a residual-scatter
    condition, and then **visual inspection** of orbits. Their own
    summary: the difficulty of defining robust classes illustrates the
    challenge of the NSS processing.
  - §6.2.4 treats spurious orbits.
- *Nagarajan et al. 2024* (2024PASP..136i4203N) [A, abstract only].
  Uses hierarchical triples to calibrate Gaia DR3 orbital-solution
  parallax errors, finding them underestimated by about 1.3 at
  G > 14 and 1.7 at G = 8–14. **The earlier draft described it as
  showing spurious orbit solutions for triples; the abstract says
  otherwise and that sentence is withdrawn.**
- Gaia Collaboration (Arenou et al.) 2023, 2023A&A...674A..34G [A],
  is the DR3 non-single-star overview; not read.

Visual-binary catalogues.

- Hartkopf, Mason & Worley 2001, "The Fifth Catalog of Orbits of
  Visual Binary Stars" (2001AJ....122.3472H) [R], §2. Grades run from
  1 (definitive) to 5 (indeterminate). The scheme was made "more
  objective" by calibrating to the Fourth Catalog grades with weighted
  rms residual in separation and in relative separation, θ and phase
  coverage (rms and maximum gap), number of revolutions and number of
  observations; about 98% of grades agree within one level. Grades
  were deflated for old orbits because interferometry made higher
  accuracy possible, and combined astrometric–spectroscopic orbits
  are graded only on the astrometric measures.
- The **Sixth Orbit Catalog** (USNO WDS-ORB6, Hartkopf, Mason and
  collaborators) has no refereed paper of its own that we found. Its
  web page cites Hartkopf et al. 2001 for the grading
  scheme and gives the same five-grade definitions [R, from the web
  page, not a bibcode]. Mason 2025 (2025CoSka..55c.481M) [A] is a
  general description of the USNO double-star catalogues, not read.
- The earlier claim that Tokovinin "insists on a visual check of every
  fit" is **[U]** and removed.

### 1.5 Reliability

- **Verified.** Convergence checks used in the orbit papers read:
  autocorrelation time and visual inspection, with the symmetric (ω, Ω)
  peaks as a check (orbitize!, §2.3.2 [R]); Gelman–Rubin (Holl 2023
  §4.1 [R]); simulation-based calibration (Octofitter §3.7 [R]); and
  the bootstrap (PMOIRED).
- **[A]** Split-R̂ and bulk/tail ESS: Vehtari et al. 2021
  (2021BayAn..16..667V) exists in ADS. The 1.01 threshold is from
  memory **[U]**.
- **[U]** The "≳50τ" emcee rule.
- No source found on wall-clock limits or watchdogs for orbit fits.
  The only reported run-time figure is Thompson 2025 §3.5: "1-2 weeks"
  for the tempered Gl 229 fit. Long runs are normal for that method.

### 1.6 Method table

| Method | Paper (bibcode), section | What it does | Starts and modes | Parameterization | Limitation |
|---|---|---|---|---|---|
| TI grid + linear solve | Lucy 2014 (2014A&A...563A.126L) §2.1, §2.4 [R]; Mendez 2017 (2017AJ....154..187M) §3.3 fn 6 [R]; Hartkopf 1989 (1989AJ.....98.1014H) [A] | grid (P, e, τ or T₀), solve A, B, F, G | whole grid | TI elements | cost; aliasing at sparse epochs |
| Gaia NSS (Halbwachs) | 2023A&A...674A...9H §5.1, Table 1 [R] | LM from a (P, e, T₀) grid | period 10 d to span/0.6 | TI | grid step not given; F2/significance cuts |
| Gaia NSS (Holl) | 2023A&A...674A..10H §4.1, §4.2.1, §5.1.2, §5.2.2 [R] | DE-MCMC and GA with TI | periodogram-seeded; GA half analytic | TI | aliasing with scanning law; heterogeneous filters |
| OFTI | Blunt 2017 (2017AJ....153..229B) §2.1–2.3 [R] | prior draw, scale a, rotate Ω, reject | independent draws | Campbell | slow for arcs above about 15% of an orbit |
| orbitize! | Blunt 2020 (2020AJ....159...89B) §2.1, §2.3 [R] | OFTI or ptemcee | tempering needs no tuned starts | Campbell, τ | positions only |
| The Joker | Price-Whelan 2017 (2017ApJ...837...20P) §2 [R] | prior samples in (ln P, e, ω, φ₀), linear parameters marginalized | all modes kept; Δ = 4P²/(2πT) | ln P, e, ω, φ₀ | RV only |
| orvara | Brandt 2021 (2021AJ....162..186B) §2, §4, §6 [R] | RV + relative + Hip–Gaia, ptemcee | walker starts from user Gaussians | √e sin ω, √e cos ω, λ_ref, Ω | tempering cost |
| Octofitter | Thompson 2023 (2023AJ....166..164T) §3.2, §3.4, §4.4 [R] | NUTS; visibility and closure-phase likelihoods | best of 250,000 prior draws | Campbell or TI; θ at mean epoch | NUTS not for multi-modal |
| Octofitter + Pigeons | Thompson 2025 (2025AJ....169..193T) §3.5, §4.1.1 [R] | non-reversible parallel tempering, kernel phases | global, all modes sampled | Campbell, τ | 1–2 weeks of sampling |
| Xuan 2024 (PMOIRED branch) | 2024Natur.634.1070X Methods, "Orbit fits" [R] | gradient descent, RV first then joint with closure phases | RV-seeded; bootstrap | Campbell (Table 1) | needs RVs to break ambiguity |
| CANDID | Gallenne 2015 (2015A&A...579A..68G) §2.1 [R] | grid of fits per epoch, bootstrap | grid of starts | position and f | single epoch |
| ARMADA I and II | Gardner 2021 (2021AJ....161...40G) §2.2, §3 [R]; Gardner 2022 (2022AJ....164..184G) §2.3, §3 [R] | grid per epoch, then TI lmfit, then emcee | ORB6 guesses; period grid P = 2fT/k, f = 3 | TI | errors rescaled to χ²_red = 1 |
| CHARA/MIRC orbits | Lester 2019 (2019AJ....157..140L) §3.2, §4.2 [R] | grid and MPFIT per epoch, Newton–Raphson orbit | ambiguity resolved by a preliminary orbit | Campbell | rescaling by 5.6 |
| VLTI/PIONIER orbits | Le Bouquin 2017 (2017A&A...601A..34L) §2.3–2.5 [R] | per-epoch position, then LM with noise-perturbed refits | LM only | Campbell | assumes unambiguous period; ignores intra-night motion |
| VLTI/GRAVITY binary orbits | Gallenne 2023 (2023A&A...672A.119G) §2.2, §3 [R]; Klement 2025 (2025A&A...694A.208K) §3.1, §4.1 [R] | CANDID or PMOIRED per epoch, then emcee or orbfit-lib | least squares from literature values | Campbell | positions only |
| Visual-orbit grading | Hartkopf 2001 (2001AJ....122.3472H) §2 [R] | grades 1–5 from residuals and coverage | n/a | n/a | grades judge the orbit, not the fit |

### 1.7 Answers to the specific questions

1. **Xuan et al. 2024.** See 1.0. Closure phases directly, with RVs,
   not per-epoch positions (except one epoch in the discovery
   figure). Start: RV-only gradient descent (PMOIRED) and global
   parallel tempering (Octofitter). Ambiguity: carried in the fit, and
   removed by RVs. Codes: PMOIRED and Octofitter.
2. **Search methods.** See 1.1 and 1.2.
3. **Direct versus two-step comparison.** No paper we found compares
   the two on the same data. The single statement is Xuan's Methods
   sentence, on an unnamed binary with UVES RVs, with no numbers shown.
   Thompson 2025 does not compare them. ARMADA, Lester, Le Bouquin,
   Gallenne and Klement all use two-step fits and do not report a
   direct fit. Octofitter's AMI demonstration is simulated and
   direct only. ADS searches for such a comparison returned nothing
   relevant. **So our tutorial's result is not "the comparison
   the literature lacks": it is a controlled simulated comparison,
   and Xuan's sentence is a one-line second data point on real data.**
4. **GRAVITY dynamical-mass binary papers.** Gallenne 2023 and Klement
   2025 use positions (two-step), with CANDID or PMOIRED per epoch.
   Xuan 2024 and Thompson 2025 use the closure phases directly. The
   direct fit is used where epochs are individually ambiguous.
5. **Large-scale pipelines.** See 1.4.

## Part 2. Critical assessment of virgil

### 2.1 What we do that is at or ahead of published practice

- **Joint visibility-level likelihood with per-block noise terms.**
  *Revised after the literature check:* this is not ahead of published
  practice. Xuan et al. 2024 and Thompson et al. 2025 fit GRAVITY
  closure or kernel phases directly with per-epoch jitter terms
  (1.0), and Octofitter supports visibility-level likelihoods
  (1.3). It is ahead of the two-step papers (ARMADA, Lester,
  Le Bouquin, Gallenne, Klement), which fit positions. What may be new
  is marginalizing the scale per block and fitting visibilities
  (including V²) rather than kernel phases.
- **TI linear split in `starting_orbits`.** This is the literature's
  standard search, done correctly.
- **Noise-marginalized single-epoch surface (`gap_marginal`, #281).**
  This replaces the tautological rescaling the literature criticizes.
  Raw χ²/N is recorded and a warning is raised above 4.
- **Two-step ≈ joint.** The tutorial shows that two-step fits with the
  scale marginalized match the joint fit. We found no published
  direct-versus-two-step comparison apart from one unnumbered
  sentence in Xuan et al. 2024 (1.3), so this is useful evidence,
  though from one simulated case only.
- **Priors.** Jeffreys/isotropic priors match the standard choice.

### 2.2 Where published practice is better, bluntly

1. **We commit to one mode too early.** Three places do this:
   - `epoch_positions` keeps one peak per night.
   - `start_from_positions` **drops** nights whose gap is below
     `min_gap=5`.
   - The ranking then refines only `n_refine=4` orbits.

   The Joker, OFTI and orbitize! keep the full multimodal set, and
   CANDID-style practice carries the χ² map per epoch. The direct-fit
   precedent (Thompson 2025, §3.5) samples all modes with parallel
   tempering. Gl 229 shows the cost:
   - *Corrected 2026-10-07.* The fit seeded from Xuan's orbit **beat**
     our free search: by Δloss −265.9 on 5 nights and −1249.1 on 7.
     The earlier draft had this the wrong way round;
   - the free search's best mode is i ≈ 110°, against i ≈ 30° in the
     seeded fit that reproduces Xuan;
   - so the failure is in the search, not the likelihood;
   - this is what the literature leads one to expect: without RVs, the
     GRAVITY kernel-phase posterior for this system has four modes
     (Thompson 2025, §4.1.1), and the published PMOIRED baseline was
     itself seeded from an RV-only orbit (1.0).

   The design in #279 (peak catalogue, mixture over aliases, TI-EM)
   moves to published practice, but **none of it has landed yet**.
2. **No time limits, watchdog or progress logging.**
   - The retrograde start (i = 148.6°) ran for about 3.5 h on 7 nights,
     against 18 s on 5.
   - A neighbouring start ended at e = 0.9, the prior bound.
   - Nothing in `fit` limits the time or the steps of one start, or
     reports progress, and one bad start blocks the batch.
   - This is not a literature question: published pipelines
     (Gaia NSS) solve it with fixed budgets and rejection flags. We
     have neither.
3. **We sample at hard bounds in the worst parameterization.**
   - e ∈ [0, 0.9] and cos i are box-bounded.
   - *Revised:* there is no single literature standard. orvara uses
     (√e cos ω, √e sin ω); orbitize!, Octofitter and Thompson 2025
     use uniform e and ω with τ; Octofitter prefers the position angle
     at the mean epoch to τ (1.2). Ω modulo 180° without RVs is
     universal. Whether this removes the edges where L-BFGS-B and NUTS
     stall, the likely cause of the hang, is our hypothesis and has no
     published support.
   - The hang also involved 7 free `phi_scale` terms with closure
     phases only, a degeneracy no published fit carries.
4. **Period-grid density is the user's job.**
   - `periods` is passed in. Phase coherence needs δP ≲ P²/(kT), which
     for Gl 229 (P ≈ 12 d, T ≈ 410 d) is several hundred trial periods.
   - With n_phase=36, a coarse grid can miss the true mode entirely.
   - Published rules set the frequency resolution from the baseline:
     The Joker uses Δ = 4P²/(2πT), ARMADA II uses P = 2fT/k with f = 3
     (spacing about P²/(2fT)), and Holl 2023 uses a uniform grid up to
     twice the span (1.1). For P = 12 d, T = 410 d and P in 5–20 d,
     the ARMADA rule gives roughly 370 trial periods. We should set the
     grid from the baseline by default.
5. **Static snapshot per night.**
   - Gl 229 Bb moves about 3.8 mas a day, roughly 0.3 λ/B over a night
     for GRAVITY.
   - PR A tested 0.2 λ/B and found no effect, so Gl 229 lies just
     outside the tested range.
   - Published two-step work (ARMADA, Le Bouquin 2017, which neglects
     motion within a night for P ≈ 0.5–15 yr) targets slower orbits,
     where this never matters. Xuan 2024 did include linear motion over
     the 2.5 h window in the first-epoch fit (4.6 mas/day). For fast
     pairs, the snapshot assumption needs a check or a linear-motion
     term per night.
6. **The Gl 229 per-night misranking is unexplained.**
   - PR A fixed the *size* of the gap (old 6.5/7.5 → marginal
     0.95/0.76 on the ambiguous simulated nights), not the *order* of
     the peaks.
   - For a single block the scale cannot reorder peaks.
   - The PR A simulations reproduced no reordering at the true
     position, with grid-edge and 0.2 λ/B motion tested.
   - The remaining candidates are:
     - the old flux prior, f up to 2: the f↔1/f, r↔−r twin is exact, and
       f ≤ 1 is now enforced;
     - within-night motion above 0.2 λ/B;
     - the grid range.
   - Validation job 18172340 decides between them. Until it does, we
     cannot claim PR A fixes Gl 229.
7. **No restarts or reproducibility check.** Published practice reruns
   to confirm that mode weights are stable, and reports R̂/ESS for each
   mode. Per-mode NUTS (#279) plans this but does not specify
   acceptance thresholds.
8. **No validation cascade or grade.** There is nothing like Gaia's
   F2/a₀/σ or the USNO grades. Raw χ²/N > 4 only warns; nothing stops
   or flags a failed fit downstream.
9. **Performance.** The Python loop over datasets grows the HLO linearly
   with the number of nights. Literature codes vectorize the epochs.
   This is minor, but it matters for 7+ night fits on OzSTAR.

### 2.3 Where the literature does not help us

- *Revised.* The two-step literature fits positions, so it offers
  nothing on fringe-alias mixtures. The direct-fit papers (Xuan 2024,
  Thompson 2025) handle aliases by sampling every mode with tempering
  and by adding RVs, not by a peak-catalogue mixture. Our
  peak-catalogue mixture likelihood (#279) is therefore a different
  route to the same end, and needs its own validation:
  injection–recovery on simulated nights, and Gl 229 against Xuan. A
  useful test is to reproduce Thompson's four modes from GRAVITY
  alone.
- Error ellipses are common, but marginalizing the scale per block
  rather than adding jitter is not standard. It is defensible
  (Jeffreys 1/s), and should be written up as a choice.

## Part 3. Prioritized improvements

| # | Improvement | Benefit | Cost | PR |
|---|---|---|---|---|
| 1 | Wall-clock and step budget per start in `fit`, NaN/bound guards, progress logging, failed-start flag; batch continues | stops 3.5 h hangs blocking runs; failures visible | small (chunked optimizer loop) | **B** (or B0 if B is large) |
| 2 | Default period grid from baseline: δP = P²/(kT), log-spaced, P_min/P_max from data and priors; warn when user grid is coarser | prevents silently missing short-P modes (Gl 229) | small; grid cost grows ~P_min⁻¹ | **B** |
| 3 | Resolve the Gl 229 reordering with job 18172340: f ≤ 1 vs f ≤ 2, grid range, injected within-night motion 0.2–0.5 λ/B | tells us whether PR A is enough or C is needed first | OzSTAR only, no code | before **C** |
| 4 | Peak catalogue: keep top-K peaks per night with marginal weights; never drop ambiguous nights | removes the main early-commit failure | medium | **C** (as designed) |
| 5 | TI-EM / mixture search over aliases, keep top-N distinct orbits, report mode weights | literature-standard multimodality handling | medium–large | **D** |
| 6 | Reparameterize: (√e cos ω, √e sin ω), τ, Ω mod 180° with documented convention; soft edges | removes bound stalls (likely hang cause), better NUTS mixing | medium; touches orbit models and tests | **E** |
| 7 | Within-night motion: check gap and peak position against a linear-motion term when predicted motion > 0.2 λ/B | correct snapshots for fast pairs | small check, medium if modelled | **C** (check) / **F** (model) |
| 8 | Per-mode NUTS acceptance: split-R̂ < 1.01, bulk/tail ESS, divergences, two seeds agree on mode weights | reproducible mode weights | small on top of #279 | **F** |
| 9 | Validation cascade and grade: raw χ²/N, orbit vs linear-motion Δχ², a/σ_a, held-out-epoch prediction | catches failed fits automatically | small–medium | **G** |
| 10 | Vectorize the dataset loop (stack equal-shape nights, vmap) | constant HLO; faster compile on 7+ nights | medium | **H** |
| 11 | Check Xuan et al. 2024 Methods and the unverified [M] citations before any public claim | credible prior-art section | reading only | any |

> **Literature check, 2026-10-07: notes on Part 3.** No recommendation
> is changed. The literature supports rows 1–5 and 7–9 and qualifies
> two:
>
> - **Row 6 (reparameterize).** *Qualified, not contradicted.* There
>   is no single published standard (1.2). orvara uses
>   (√e cos ω, √e sin ω), but Thompson 2025 and Octofitter use uniform
>   e and ω, and Octofitter reports that the position angle at the mean
>   epoch converges better than τ (Thompson 2023, §3.2, App. B).
>   Consider θ at the mean epoch in place of τ, and treat the
>   claim that bounds cause the hang as a hypothesis.
> - **Row 2 (period grid).** Consistent. Published spacing rules are
>   Δ = 4P²/(2πT) (The Joker) and P = 2fT/k, f = 3 (ARMADA II); the
>   unspecified k in δP = P²/(kT) can be fixed by one of these.
>
> **Evidence for the order of work.** Without RVs, the published
> GRAVITY-only posterior for Gl 229 Ba–Bb has four modes (Thompson 2025,
> §4.1.1). Row 3 and rows 4–5 are therefore about aliases that the
> data genuinely do not resolve, not about a bug.
> A free search that finds a worse mode than the RV-seeded fit is the
> expected outcome, which strengthens the case for row 5 (mixtures and
> top-N modes). Row 11 is now done for the citations listed in Part 1;
> items tagged [A] and [U] there are still unchecked in full text.

**Order:** 1 and 2 first, because they are cheap and remove the
observed failures. Then 3, which decides the urgency of 4. Then 4–6 as
in #279. 7–10 follow.

Held-out evidence that the correct mode matters:

- In the right mode, the 2025-02-11 position is predicted to 0.52 mas,
  and the held-out nights' raw χ²/N are 3.72 and 10.09.
- The wrong mode gives 606 and 355.

Prediction of held-out epochs (row 9) is the most discriminating
automatic check we have, and should become the default acceptance test.
