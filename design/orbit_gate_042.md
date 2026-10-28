# Orbit population injection–recovery gate (0.4.2)

Status: **pre-registered, awaiting Ben's approval.** Drafted 2026-10-28, before
any run of the harness it describes. Companion design:
[`orbit_dehack.md`](orbit_dehack.md).

This file fixes the population, the metrics and the pass thresholds of the
0.4.2 orbit gate **before any run**. Only Ben changes it. Every change is
appended to the change log at the end, with the date, the reason and the
line "changed after seeing a result: yes/no". The harness records this
file's SHA-256 (of the YAML block in §6 and of the whole file) in every
summary it writes, and refuses to aggregate against a different hash
unless the change log has an entry for it.

## 1. Why a population, and what it replaces

The 0.4.0 and 0.4.1 orbit gates were two named systems, Gl 229 Ba–Bb and
ι Peg. The orbit library was tuned on them: two heuristics in the default
path were added after a failed Gl 229 gate (32acb33 tempers the short-arc
sign gate "because the 0.4.0 Gl 229 gate failed"; 52c7013 moves the
outlier mixture to a polish after "the Gl 229 gate regression from #451").
A gate that the code was tuned on is not evidence. This gate replaces it
with held-out evidence: simulated systems that no line of the library has
seen, drawn with fixed seeds, scored by criteria written here first.

Gl 229 Ba–Bb and ι Peg become **showcases** (§7): run with defaults,
reported raw, never pass/fail.

## 2. Independence

- **Simulator.** Data are written by virgil-validation's own simulator,
  `crosscheck.simulate.observe` (independent of `virgil.simulate`), as
  OIFITS files, and read back by virgil's normal I/O. The binary's complex
  visibility, the uniform-disc primary and the Kepler solver used to place
  the companion are the harness's own code (numpy, written from the
  textbook formulae: Kepler's equation by Newton iteration, Thiele–Innes
  projection), not imports from virgil.
- **Scoring.** Truths are written to JSON at draw time, before any fit.
  The aggregator reads truths and virgil outputs; it never calls virgil.
- **No tuning loop.** Workers implementing the de-hacking stages read
  aggregate metrics only (the summary JSON), never per-system results of
  the gate population, while a stage is open. A separate **development
  population** (§3.5, different master seed) is available to them for
  debugging.

## 3. Populations

Master seed `20261028`. System *i* of stratum *s* uses
`numpy.random.default_rng(numpy.random.SeedSequence([20261028, s, i]))`,
with stratum codes A = 1, B = 2, N = 3, D = 4. Draws are made in the order
listed below, one call per quantity, so that adding a stratum never
changes another stratum's systems.

### 3.1 Instrument templates (both strata)

| | T1 "K-band, 4 large telescopes" | T2 "H-band, 4 small telescopes" |
|---|---|---|
| stations | VLTI UT1–UT4 (ENU from the ESO station table) | VLTI A0-G1-J2-K0 |
| channels | 12, 2.00–2.40 µm | 6, 1.53–1.77 µm |
| snapshots per night | 4–8 (uniform integer), hour angles uniform in [−3, +3] h, sorted | same |
| closure-phase noise | `phase_noise="baseline"` (triangles sharing a baseline correlate) | same |
| V² channel correlation | a per-baseline, per-snapshot multiplicative term common to all channels, of variance fraction *c* | same |

Each system draws its template with probability ½. Declination is uniform
in sin δ over [−70°, +20°].

### 3.2 Stratum A: calibration (200 binaries, drawn from the engine's prior)

Purpose: test that the band probabilities and the posteriors are
**calibrated**. Simulation-based calibration (Cook, Gelman & Rubin 2006;
Talts et al. 2018) guarantees nominal coverage *averaged over the prior*
when the parameters are drawn from the prior the fit uses. So every
parameter is drawn from exactly the prior that `fit_orbit_aliases` states,
with `p_range`, `a_range`, `flux_range`, `ecc_max` and `primary_diam=None`
passed explicitly (no data-derived window: a window computed from the data
would make the prior depend on the data and break the guarantee).

- Coverage: *K* nights, uniform on {4, 5, 6, 8, 10}; night times uniform
  in a 120-day season, then sorted; each night 1–4 h long.
- `p_range = (P_lo, P_hi)` with `P_lo` = the engine's documented
  sampling-limit formula evaluated with `field = a_range[1]` (the prior's
  largest separation, so it depends on the coverage only) and
  `P_hi = 2 ×` span.
- log *P* uniform on `p_range`; τ (as a phase) uniform; *e* uniform on
  [0, 0.9]; cos *i* uniform on [−1, 1]; ω uniform on [0°, 360°); Ω uniform
  on [0°, 180°); log *a* uniform on `a_range = (0.3, 10) × λ/B_max`;
  log flux uniform on `flux_range = (0.01, 1)`.
- Noise: white within each channel, *c* = 0 (the tempering factor should
  come out near 1: this stratum also checks that tempering does not break
  a calibrated likelihood); σ(V²) log-uniform on [0.005, 0.05] and
  σ(CP) log-uniform on [0.2°, 3°].
- No contaminants (a bad night has no generative model inside the
  engine's prior, so it cannot enter an SBC stratum).
- Fitted with tempering off (`temper=False`): the noise is white, so
  *f* = 1 is the truth, and the estimated max(f̂, 1) is biased above 1
  (scatter √(2/C)), which would over-cover and fail a two-sided test for
  a correct posterior. A tests the likelihood; the f̂ distribution on A
  is reported separately.
- NUTS draws are thinned to near-independence (every ⌈n/ESS_min⌉-th
  draw, ESS_min the smallest bulk ESS over the ranked parameters) before
  the SBC ranks are formed (Talts et al. 2018): correlated draws give
  non-uniform rank histograms.

### 3.3 Stratum B: blind default runs (240 binaries, realistic)

Purpose: test the **default** `OrbitPipeline` (no period, window or warm
start supplied) on systems that look like real campaigns, with the failure
modes the heuristics were built for.

- Coverage schedule, uniform over three kinds:
  - S1, one season: *K* ∈ {4, 6, 8, 12} nights uniform in 90 days;
  - S2, two seasons a year apart: *K* ∈ {4, 6, 8, 12}, split as evenly as
    possible between two 60-day windows 365 days apart (the season-gap
    alias problem);
  - S3, a dense run then sparse returns: half the nights within 10 days,
    the rest uniform over 2 years.
- *P* log-uniform on [1 d, 3 × span]. A draw whose orbit violates the
  engine's documented domain (the companion moves more than half a
  resolution element within its longest night, the sampling limit of
  `period_window`) is redrawn with the next state of the same generator;
  the number of redraws is reported. This keeps the population inside the
  method's stated domain without selecting on difficulty.
- *e* uniform on [0, 0.8] (roughly flat for P below about 1000 d:
  Raghavan et al. 2010; Moe & Di Stefano 2017); cos *i*, ω, Ω, τ as in A.
- Semi-major axis: *a* / (λ/B_max) log-uniform on [0.5, 10], with
  λ/B_max the finest scale (λ_min). A draw outside the engine's default
  `a_range` (1–30 mas today; on T1, 10 res ≈ 32 mas) is redrawn like
  *P* and counted, so no truth lies outside the prior the default run
  can reach. This clause lapses if decision 12 (`a_range` in resolution
  units) lands, recorded in the change log.
- Contrast: Δmag uniform on [0, 4.5]; then, independently, with
  probability 0.1 the system is replaced by a near-twin with Δmag uniform
  on [0, 0.1] (the label-swap case).
- Primary: point with probability 0.7, else a uniform disc of diameter
  uniform on [0.2, 0.8] λ/B_max.
- Noise: as in A, and *c* uniform on {0, 0.3, 0.6} (correlated channels:
  the tempering case).
- Contaminants, drawn independently:
  - **bad night** (probability 0.2): one night, uniform among all but
    the first, gets V² multiplied by a factor uniform on [0.6, 0.9] and
    closure-phase noise drawn 4× larger than its quoted error;
  - **sign error** (probability 0.1): one night, uniform among all but
    the first, has every closure phase negated (a convention error);
  - **misspecified model** (probability 0.15): an unmodelled third
    point source, fixed on the sky, at separation uniform on
    [2, 6] λ/B_max, position angle uniform, flux ratio uniform on
    [0.02, 0.08] of the primary. The engine fits a binary, so χ²/N on the
    quoted errors rises and stays raised on every night, as in the blind
    gallery runs where the wrong alias won exactly when χ²/N was 5–27
    (ι Peg, ο Leo, Gl 229 B blind). This tests whether the evidence
    stays honest when no model in the family fits;
  - **fringe-alias night** arises naturally from the noise and coverage;
    it is not injected.

### 3.4 Stratum N: single-star nulls (100 systems)

Coverage, templates, noise and contaminants drawn exactly as in B, except
the misspecification contaminant (a fixed third source with no primary
companion is a static binary, not a null); no companion. 30 of them (the first 30 indices) have a uniform-disc primary
(diameter as in B); the bad-night probability is raised to 0.4 so that the
null sample contains enough of them to test the outlier model.

### 3.5 Stratum D: development (not scored)

60 systems drawn as B with stratum code 4. Workers may inspect them
freely. They never enter a gate number.

## 4. Runs

- Stratum A: `fit_orbit_aliases(data, p_range, positions=epoch_positions(...),
  a_range=..., flux_range=(0.01, 1), ecc_max=0.9, primary_diam=None)` and
  the posterior of every band with *p* ≥ 0.01, by the pipeline's sampler.
- Strata B, N: `OrbitPipeline.from_oifits(paths).run()` with **every
  setting at its default** (the gate tests the defaults as shipped at the
  commit under test).
- Oracle runs (B only): the same call with `init=` the truth, which forces
  the truth's band to be fitted from the truth as well as from the blind
  seeds. Used only for metric M4.
- The commit under test, the JAX version and the device are recorded per
  system.

## 5. Metrics and thresholds

All tests in one Holm family at α = 0.01 (the p-value tests), plus the
must-hold count limits. A stage passes when every test passes.

**M1. Band calibration (A).** For system *i* let *C_i* be the smallest set
of bands, by decreasing *p*, with total probability *m_i* ≥ 0.68 (and
again for 0.95). If the band probabilities are calibrated, the number of
systems whose true band is in *C_i* has mean Σ *m_i* and variance
Σ *m_i*(1 − *m_i*) (a Poisson-binomial sum). Test: two-sided normal
approximation to that sum; one p-value per level.
*Why:* an engine whose *p* is overconfident (the 0.4.0 untempered
evidence, *p* = 1.000 at χ²/N 12) fails; one that hedges correctly passes
whatever the population's difficulty.

**M2. Parameter coverage (A).** For *P*, *e*, *a*, cos *i* and flux, the
rank of the truth among the posterior draws (SBC; Talts et al. 2018),
pooled over systems: KS test of the normalized ranks against uniform, and
the 68% and 95% central-interval coverages tested against binomial with
the nominal rate (two-sided). 15 p-values.
*Why:* draws from the prior make nominal coverage exact for a correct
posterior; any departure is a bug, a mis-tempering or a missed mode.

**M3. Band recovery and confidence (B).**
- (a) Recovery is calibrated: the number of systems whose highest-*p*
  band is the true one, against Σ *p*_MAP,i with variance
  Σ *p*_MAP,i(1 − *p*_MAP,i); one-sided test for *fewer* recoveries than
  claimed (overconfidence).
- (b) Confident errors: among systems with *p*_MAP ≥ 0.99, the number
  whose MAP band is wrong must not exceed the 99th percentile of
  Binomial(*n*_conf, 0.01). A calibrated engine is wrong at most 1% of
  the time when it claims 99%.
- The recovery rate itself, by stratum cell (schedule × *K* × contrast
  bin), is **reported**, with a Wilson 95% interval, and compared between
  stages. It has no absolute floor: the population contains systems whose
  band no method can determine, and a floor would reward overconfidence.
  Sharpness is guaranteed instead by M4 (the search finds what the
  likelihood holds) together with M3a (it does not claim more).

**M4. Search completeness (B).** A miss is a system whose blind run's
log Z for the true band is more than 1 nat + 3√(SE_blind² + SE_oracle²)
below the oracle run's, using each run's reported IS standard error
(two independent estimates differ by noise alone; ln Ẑ is also biased
low at small ESS). Pass
if misses ≤ 2% of B (at most 4 of 240).
*Why 1 nat:* a 1-nat shortfall changes that band's odds by a factor e,
the smallest change that moves a reported *p* noticeably. *Why 2%:* the
binomial standard error of a recovery rate near ½ on 240 systems is
3.2%; misses below 2% are smaller than the gate can resolve in M3, so
any larger rate would be visible as a recovery loss.

**M5. Posterior coverage under contamination (B).** As M2, on the 95%
intervals of *P*, *e*, *a* for systems whose true band has *p* ≥ 0.5,
one-sided (under-coverage only). 3 p-values.
*Why one-sided and conditional:* B is not drawn from the prior, so exact
nominal coverage is not guaranteed; over-coverage is conservative and
allowed. Under-coverage is the overconfidence that bad nights, sign
errors and correlated channels cause, and it fails.

**M6. False orbits (N).** A false orbit is an `orbit` verdict
(`modes.no_orbit` evidence at or above the pipeline's decisive threshold)
on a null. Pass if the count is ≤ 2 of 100.
*Why:* 2 of 100 is a **pre-registered limit chosen with a bound as a
guide**, not a derived false-alarm rate. For an untempered likelihood
and nulls drawn from the engine's own null prior, the Bayes factor *B*
obeys P(*B* ≥ *k*) ≤ 1/*k* (Markov's inequality on E_null[*B*] = 1;
Royall 2000), which at ln *B* = 5 (Kass & Raftery 1995, "very strong")
gives at most 0.67 expected in 100 and P(≥ 3) = 3%. None of those
conditions holds exactly here: the likelihood is tempered (∫L^{1/f}π is
not a likelihood ratio), the nulls' diameters and bad-night rate are
not drawn from the engine's null prior, and contaminated nights are
only approximated by the mixture. The false-orbit rate on stratum D's
nulls is reported before the first scored run, as a check that the
limit is not routinely failed by a correct engine. If Ben keeps the current
`no_orbit_nats = 10`, the bound is 4.5 × 10⁻⁵ and the limit becomes
≤ 1 of 100 (P(≥ 2) < 10⁻⁴).

**M8. False no-orbit verdicts (B).** A system of B whose oracle run
has ln *B*(orbit : null) ≥ 10 but whose blind run returns a `no-orbit`
verdict is a lost detection. Must hold: ≤ 2% of the qualifying systems
(the same resolution argument as M4); the count is also reported by *K*.
*Why:* M6 alone rewards a null that is too easy to win; a null model
with too much freedom (e.g. the outlier mixture applied to the null,
orbit_dehack §2A) would lose real detections at small *K*, and only
this metric sees it.

**M7. Must hold (all strata).** Every run ends with a result or a
recorded failure (no hang, no crash without a record); the summary
records the commit, this file's hash and the counts.

**Reported, never pass/fail.** Recovery by cell (M3); the outlier
probability the fit gives each injected bad night and sign-error night
against the others (ROC area); `labels_free` on near-twins; the
tempering factor against the injected *c*; the primary diameter against
the truth; wall time and peak memory per system; the number of redraws
in B; M3 and M5 split by the misspecification flag; and a data-sufficiency
table: recovery against the number of nights and the phase coverage of the
true orbit (NN Del's four nights over 62 days is the case it describes).

## 6. Criteria (machine-readable; the harness hashes this block)

```yaml
version: 1
registered: 2026-10-28
master_seed: 20261028
strata: {A: {code: 1, n: 200}, B: {code: 2, n: 240}, N: {code: 3, n: 100}, D: {code: 4, n: 60, scored: false}}
family_alpha: 0.01
tests:
  M1_band_calibration: {levels: [0.68, 0.95], stratum: A, sided: two}
  M2_parameter_coverage: {params: [period, ecc, a_mas, cos_inc, flux], intervals: [0.68, 0.95], rank_ks: true, stratum: A, sided: two}
  M3a_recovery_calibration: {stratum: B, sided: fewer_than_claimed}
  M5_coverage_contaminated: {params: [period, ecc, a_mas], interval: 0.95, condition: p_true_band_ge_0.5, stratum: B, sided: under}
must_hold:
  M3b_confident_errors: {p_map_min: 0.99, max: binomial_q99(n_conf, 0.01)}
  M4_search_misses: {nats: 1.0, plus_se_multiple: 3, max_fraction: 0.02, stratum: B}
  M6_false_orbits: {stratum: N, max_count_at_lnB5: 2, max_count_at_lnB10: 1}
  M8_false_no_orbit: {stratum: B, oracle_lnB_min: 10, max_fraction: 0.02}
  M7_complete: true
```

## 7. Showcases (reported raw, never a gate)

Gl 229 Ba–Bb (GRAVITY, two seasons) and ι Peg are run with `OrbitPipeline`
defaults at each stage's commit. The report gives, raw: the winning band
*N* and its *p*, the period, χ²/N on the quoted errors (lead with it), the
tempering factor, the posterior summary, and the published values beside
them. A worse showcase is reported as worse, never fixed by a heuristic,
a target-specific setting or a re-tuned constant. The 0.4.0/0.4.1 Gl 229
passes are recorded as **unverified** until the de-hacked code reproduces
them cold.

## 8. How it runs

- **Harness:** virgil-validation, `scripts/orbit_population.py`, parts
  `draw` (simulate and write OIFITS plus truth JSON), `fit` (one system),
  `oracle` (one B system), `aggregate`; criteria read from this file at
  a pinned virgil commit. Tests in vv check the hash, the seeding and the
  aggregator on a toy summary. Built in Track 2 Stage 1.
- **Cluster:** an OzSTAR array job (one task per system, GPU nodes; CPU
  JAX jobs abort with LLVM allocation errors) submitted, watched and
  pulled back **by Ben** with `ozstar_scripts`. Agents never touch the
  cluster; they read the pulled summary locally.
- **Baseline first:** Stage 1 runs the gate on current `main` (884aa7e or
  its successor at the time) so that every later stage reports deltas.
  The baseline's result, pass or fail, is reported raw.
- **Every stage PR** quotes the gate summary at its head beside the
  baseline, plus the showcases.

## 9. Change log

| date | change | by | changed after seeing a result |
|---|---|---|---|
| 2026-10-28 | registered (draft for approval) | Claude (Opus), for Ben | no |
| 2026-10-28 | B: misspecification contaminant (p 0.15) and a reported data-sufficiency table, from the blind gallery runs (high χ²/N ↔ wrong alias); still a draft, no harness run yet | Claude (Opus), for Ben | no |
| 2026-10-28 | Review fixes before any run: A fitted with `temper=False` and thinned NUTS ranks; M4 miss threshold adds 3× the combined IS SE; M6 presented as a guided limit, not a derived rate; new M8 (false no-orbit on B); B's *a* draws kept inside the default `a_range` | Claude (Opus), for Ben | no |

## References

- Cook, Gelman & Rubin 2006, J. Comput. Graph. Stat. 15, 675 (validation
  of Bayesian software by simulation).
- Talts et al. 2018, arXiv:1804.06788 (simulation-based calibration).
- Kass & Raftery 1995, JASA 90, 773 (Bayes factor scale).
- Royall 2000, JASA 95, 760 (the universal bound on misleading evidence).
- Raghavan et al. 2010, ApJS 190, 1; Moe & Di Stefano 2017, ApJS 230, 15
  (eccentricity distributions of short-period binaries).
