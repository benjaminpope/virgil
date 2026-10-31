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

### 3.1 Instrument templates (all strata)

| | T1 "K-band, 4 large telescopes" | T2 "H-band, 4 small telescopes" | T3 "H-band, long baselines" |
|---|---|---|---|
| stations | VLTI UT1–UT4 (ENU from the ESO station table) | VLTI A0-G1-J2-K0 | CHARA S1-S2-E1-E2-W1-W2 (positions from the published CHARA station table, ten Brummelaar et al. 2005); B_max ≈ 330 m |
| channels | 12, 2.00–2.40 µm | 6, 1.53–1.77 µm | 8, 1.50–1.75 µm |
| snapshots per night | 4–8 (uniform integer), hour angles uniform in [−3, +3] h, sorted | same | same |
| closure-phase noise | `phase_noise="baseline"` (triangles sharing a baseline correlate) | same | same |
| V² channel correlation | a per-baseline, per-snapshot multiplicative term common to all channels, of variance fraction *c* | same | same |

Each system draws its template uniformly among the three, so the
long-baseline case is covered in every stratum, not weighted in.
Declination is uniform in sin δ over [−70°, +20°] for T1/T2 and over
[−20°, +80°] for T3 (CHARA at latitude +34°). Resolution units below
are always the drawn template's λ_min/B_max.

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
  log flux uniform on `flux_range = (0.01, 1)`, a fixed explicit floor
  (orbit_dehack §2I: a floor from the data would make the prior depend
  on the data and break SBC).
- Noise: white within each channel, *c* = 0 (the tempering factor should
  come out near 1: this stratum also checks that tempering does not break
  a calibrated likelihood); quoted σ(V²) log-uniform on [0.005, 0.05] and
  quoted σ(CP) log-uniform on [0.2°, 3°]. Approved by Ben 2026-10-29
  ([orbit_mixture_v2.md](orbit_mixture_v2.md) §8): the noise drawn is
  s_g times the quoted σ, with ln s_g uniform on [−ln s_max, ln s_max],
  one draw per calibration group, because the fit infers s_g under that
  prior and SBC needs every inferred parameter drawn from it. s_g's SBC
  ranks are reported, not tested.
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
- *P* log-uniform on [1 d, 3 × span], as before: the draw and the generator
  stream are unchanged. A truth with *P* < P_cut = max(2·Δt_min, P_night),
  computed from the draw's realized epochs by `virgil.aliases.period_cut`
  (Δt_min the shortest gap between consecutive nights, P_night the
  within-night motion bound of `period_window`; orbit_alias_cycles §1.3),
  lies outside the engine's declared prior and is **excluded from every
  scored metric and counted**; the count is reported. The excluded draws form
  a reported stratum B′, not scored in any must-hold: the same draws, scored
  with `p_range` set to [1 d, 3 × span], to show what the declared prior
  excludes. Excluding rather than redrawing keeps the stream identical, so
  runs before and after this commit are paired draw for draw. A draw whose
  orbit violates the engine's documented domain (the companion moves more
  than half a resolution element within its longest night, the sampling
  limit of `period_window`) is still redrawn and counted, as before.
  The factor 2 in P_cut is a prior choice (Ben, 2026-11-01): above it the
  closest pair of nights advances by under half a cycle, so the
  (prograde, retrograde, ν) ≡ (retrograde, prograde, 1/Δt − ν) alias lies
  outside the domain, at every inclination. **Proposed, awaiting Ben's
  sign-off.** The baseline (19560581 and its oracle 19560586) stays reported
  under its own criteria commit as the "before"; runs under this commit are
  judged on their own. The before/after comparison on the *P* ≥ P_cut subset
  is a reported diagnostic, never a gate input, and criteria commits are
  never mixed.
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
  uniform on [0.2, 0.8] λ/B_max; when a disc, it is on the brighter or
  the fainter star with probability ½ each (orbit_dehack §2B label).
- Flux floor: B and N, and the per-data-set FAP null simulations, run with
  the default floor from the data (orbit_dehack §2I). The number of B
  truths whose flux ratio lies below their own system's floor is
  reported; those systems stay in every metric and count as misses in
  M9 (no exclusion).
- Noise: as in A, and *c* uniform on {0, 0.3, 0.6} (correlated channels:
  the tempering case).
- Contaminants, drawn independently:
  - **bad night** (probability 0.2): one night, uniform among all but
    the first, gets V² multiplied by a factor uniform on [0.6, 0.9] and
    closure-phase noise drawn 4× larger than its quoted error;
  - **sign error** (probability 0.1): one night, uniform among all but
    the first, has every closure phase negated (a convention error);
  - **misspecified model** (probability 0.3): an unmodelled third
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

Misspecified nulls: independently, with probability 0.3, the single star
is single but not simple: it carries something the engine's models
(point or uniform-disc primary plus point companion) cannot describe.
Ben (2026-10-28): these are where real false orbits come from, likely
for Wolf–Rayet stars and on the longest-baseline data sets. The kind is
chosen with equal probability among:
- **extended envelope** (WR-like: a compact star in a wind or dust
  envelope): the primary as drawn, plus, with probability ½ each, a
  circular Gaussian (size = FWHM) or a ring of fractional width 0.2
  (size = diameter), centred on the star. Envelope flux fraction
  log-uniform on [0.01, 0.5] and size log-uniform on [0.3, 10] res.
  **Chromatic** (Ben: WR envelopes "might have spectral line
  features"): the envelope flux fraction depends on channel. A line
  occupies a random contiguous block of channels, its width uniform
  over integers in [1, ⌈N_ch/2⌉] and its start uniform; inside it the
  envelope flux is multiplied by an enhancement factor log-uniform on
  [1.5, 10], and, with probability ½, the envelope size is multiplied
  by a factor log-uniform on [0.5, 3]. The ranges are generic and
  broad; no parameter is taken from a named system (Apep, WR 140 or any
  other). The same draws are used for the gate and for the detection
  FAP's null simulations (orbit_dehack §2G);
- **elongated primary**: an elliptical uniform disc, major-axis diameter
  uniform on [0.3, 0.8] res, axis ratio uniform on [0.6, 0.9], position
  angle uniform;
- **faint background source**: an unrelated point source, fixed on the
  sky, flux ratio log-uniform on [0.005, 0.05], separation uniform on
  [2, 6] res, position angle uniform. Because a fixed source could be
  fitted as a very slow orbit, a false orbit here is an `orbit` verdict
  whose winning band is not the long-period band (orbit_dehack §1.2,
  `period_window`); a long-period-band verdict is counted and reported
  separately as "static companion".

These lower V² or shift closure phases on some baselines in ways an
orbiting point companion can partly mimic. The misspecified nulls are a
must-hold subset of their own under M6. False orbits are also reported
(never pass/fail) by template, by kind, and against envelope flux
fraction (binned in thirds of its log range).

### 3.5 Stratum D: development (not scored)

120 systems with stratum code 4: indices 0–59 drawn as B, 60–119 drawn
as N. Workers may inspect them freely. They never enter a scored gate
number. D's only pre-registered use is the null-model choice below.

### 3.6 Null-model choice rule (run on D before any scored run)

The orbit model carries the night-outlier mixture (orbit_dehack §2A). A
null with no mixture is flawed: on a single star with one bad night the
orbit model can call that night an outlier while the null pays its full
χ², so bad nights make false orbits. Two candidates are pre-registered:

- **(ii) shared mixture:** the null carries the same ε-marginalised
  mixture, its outlier component being b_k (a companion at a free
  position that night);
- **(iii′) global-scale inflation** (approved by Ben 2026-10-29, replacing (iii);
  see [orbit_mixture_v2.md](orbit_mixture_v2.md)
  §5): the inlier of each model (orbit and null) has one fitted noise
  scale s_g per calibration group, shared by all nights. The outlier
  component of night *k* is that model's own night likelihood with each
  group's scale inflated beyond the global one,
  Σ_g ln ∫ L_kg(σ s_g t) π(t) dt, with *t* log-uniform on [1, s_max] and
  s_max the engine's existing scale bound (no new constant). The null
  then stays a single star, and an outlier night costs both models the
  same. The registered (iii) applied the same inflation on top of
  per-night scales whose prior already contained it, so it could not
  separate any night (#586, strict xfail).

**Rule.** At the virgil commit that implements both (Stage 2), run
both on all of D with every other setting at its default. For each
candidate compute M6 on D's nulls (limit scaled to 60 systems: the
same per-system rate) and M8 on D's binaries. Then:
1. if exactly one candidate passes both, it is the default;
2. if both pass, the one with the smaller sum of the two observed error
   rates (false orbits on nulls + false no-orbit on binaries) is the
   default; an exact tie goes to (iii′), whose null is a single star;
3. if neither passes, no default is chosen: the result goes to Ben, and
   no scored run happens until he decides.

Verdicts use the detection rule of decision 4 (orbit_dehack §2G):
an orbit is reported only if its per-data-set empirical FAP < α = 0.01. The choice, both candidates'
D numbers and the commit are entered in the change log before the first
scored run; the losing candidate's code is deleted in the PR that sets
the default.

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
  the time when it claims 99%. Must hold twice: on all of B, and on the
  misspecified subset of B alone (about 72 systems at rate 0.3), where
  the blind gallery runs were confidently wrong.
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

**M6. False orbits (N).** A false orbit is an `orbit` verdict on a
null, i.e. an empirical FAP < α = 0.01 (orbit_dehack §2G; decision 4,
Ben 2026-10-28). If the FAP is calibrated, the false-orbit count on *n*
nulls is at most Binomial(*n*, α). Must hold: the count does not exceed
the 99th percentile of Binomial(*n*, 0.01), on all of N and, separately,
on the misspecified-null subset at its own *n*:

| *n* | 2–15 | 16–44 | 45–83 | 84–129 |
|---|---|---|---|---|
| limit | 1 | 2 | 3 | 4 |

So N as a whole (*n* = 100) allows at most 4, and the misspecified
subset (about 30) at most 2. A correct engine fails each by chance with
probability below 1% (0.34% at *n* = 100, 0.33% at *n* = 30).
*Why this is now derived:* the FAP is computed per data set by
simulating nulls like that target (its coverage, errors, bad nights and
misspecified kinds), so P(FAP < α | null) ≤ α holds by construction for
null draws from that simulator. The gate checks that it still holds for
N's nulls, which the simulator only approximates.

**M8. False no-orbit verdicts (B).** A system of B whose oracle run's
detection statistic exceeds every one of its null draws (FAP at the
Monte Carlo floor, 1/(1 + n_null): clearly detectable) but whose blind
run returns a `no-orbit` verdict is a lost detection. Must hold: ≤ 2% of the qualifying systems
(the same resolution argument as M4); the count is also reported by *K*.
*Why:* M6 alone rewards a null that is too easy to win; a null model
with too much freedom (e.g. the outlier mixture applied to the null,
orbit_dehack §2A) would lose real detections at small *K*, and only
this metric sees it.

**M9. Band credible-set coverage (B) — the headline orbit criterion.**
Criterion set by Ben (2026-10-28, before any run): a multimodal
posterior is reported honestly, as long as it is computed accurately;
in practice only RV data or astrometry at other epochs can break such a
tie. So the gate passes when a high-probability band is the true one
*and* the band probabilities are calibrated.
- For system *i*, C_i(m) is the smallest set of bands, by decreasing
  *p*, whose total probability *m_i* ≥ m. At m = 0.95, the number of
  systems whose true band is in C_i(0.95) is compared with the
  Poisson-binomial on the claimed masses (mean Σ *m_i*, variance
  Σ *m_i*(1 − *m_i*)), as in M1, **one-sided for under-coverage**.
- Must hold twice: on all of B, and on the misspecified subset of B
  alone.
- Reported with it, so that hedging is visible: the sharpness of the
  sets (median and 90th percentile of |C_i(0.95)|) and the realised mass
  Σ *m_i* / *n*. Blanket hedging is guarded against by A's two-sided M1;
  M3a (MAP recovery calibration) stays.

**Accuracy of the band probabilities.** Every band in C_i carries its
importance-sampling standard error SE(ln Ẑ_n), and its probability's
error by the delta method,
SE(ln p_n)² ≈ (1 − p_n)² SE_n² + Σ_{m≠n} p_m² SE_m².
A system whose C_i(0.95) contains a band flagged `evidence-noisy` or
`evidence-failed` counts as a **failure for that system under M7**:
its credible set is not computed accurately. So does a system with an
`evidence-failed` band anywhere in its window, since that band's mass is
unknown and could belong in C_i.

**M7. Must hold (all strata).** Every run ends with a result or a
recorded failure (no hang, no crash without a record), and no system
fails the accuracy rule above; the summary records the commit, this
file's hash and the counts.

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
templates: [T1_VLTI_UT_K, T2_VLTI_AT_H, T3_CHARA_H]
strata: {A: {code: 1, n: 200}, B: {code: 2, n: 240}, N: {code: 3, n: 100}, D: {code: 4, n: 120, scored: false}}
family_alpha: 0.01
tests:
  M1_band_calibration: {levels: [0.68, 0.95], stratum: A, sided: two}
  M2_parameter_coverage: {params: [period, ecc, a_mas, cos_inc, flux], intervals: [0.68, 0.95], rank_ks: true, stratum: A, sided: two}
  M3a_recovery_calibration: {stratum: B, sided: fewer_than_claimed}
  M5_coverage_contaminated: {params: [period, ecc, a_mas], interval: 0.95, condition: p_true_band_ge_0.5, stratum: B, sided: under}
null_choice: {stratum: D, candidates: [shared_mixture, global_scale_inflation], must_pass: [M6, M8], tie: global_scale_inflation}
must_hold:
  M3b_confident_errors: {p_map_min: 0.99, max: binomial_q99(n_conf, 0.01), subsets: [B_all, B_misspecified]}
  M4_search_misses: {nats: 1.0, plus_se_multiple: 3, max_fraction: 0.02, stratum: B}
  M6_false_orbits: {stratum: N, alpha: 0.01, max_count: binomial_q99(n, 0.01), limits_by_n: {'2-15': 1, '16-44': 2, '45-83': 3, '84-129': 4}, N_all: {n: 100, max_count: 4}, subsets: {N_misspecified: {max_count: binomial_q99(n_subset, 0.01)}}, report: [by_template, by_kind, by_envelope_fraction]}
  M8_false_no_orbit: {stratum: B, qualifying: oracle_fap_at_mc_floor, max_fraction: 0.02}
  M7_complete: {no_crash: true, credible_set_accurate: {m: 0.95, fail_flags: [evidence-noisy, evidence-failed], failed_band_in_window_fails: true}}
  M9_band_credible_set: {m: 0.95, sided: under, subsets: [B_all, B_misspecified], report: [set_size_median, set_size_q90, realised_mass]}
population:
  B_period: {draw: [1d, 3*span], p_cut: 'max(2*dt_min, P_night)', engine: virgil.aliases.period_cut, below_p_cut: exclude_and_count, report: B_prime, scored_in_must_hold: false}
```

## 7. Showcases (reported raw, never a gate)

Gl 229 Ba–Bb (GRAVITY, two seasons) and ι Peg are run with `OrbitPipeline`
defaults at each stage's commit. The report gives, raw: the winning band
*N* and its *p*, the period, χ²/N on the quoted errors (lead with it), the
fitted global noise scales ŝ_g, the tempering factor, the posterior summary, and the published values beside
them. A worse showcase is reported as worse, never fixed by a heuristic,
a target-specific setting or a re-tuned constant. The 0.4.0/0.4.1 Gl 229
passes are recorded as **unverified** until the de-hacked code reproduces
them cold.

## 8. How it runs

- **Harness:** virgil-validation, `scripts/orbit_population.py`, parts
  `draw` (simulate and write OIFITS plus truth JSON), `fit` (one system),
  `oracle` (one B system), `aggregate`; criteria read from this file at
  a pinned virgil commit. Tests in vv check the hash, the seeding and the
  aggregator on a toy summary. Built in Track 2 Stage 1 on the existing
  infrastructure (orbit_dehack §6 Stage 1): `detection_campaign.py`'s
  hashed criteria and Holm aggregate, `sbc_v2.py`'s SBC ranks for A/M2,
  and `virgil.detection` (`DetectionMC`) for the no-orbit FAP.
- **Prerequisite:** detection v3 (OzSTAR job 18467928) is pulled and
  reported before any scored run, since the no-orbit FAP trusts
  `virgil.detection`, tails included.
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
| 2026-10-28 | Ben's decisions: misspecified nulls in N (p 0.3: extended emission or elongated primary); B misspecification rate 0.15 → 0.3 and M3b must-hold on that subset; D enlarged to 60 B-like + 60 N-like; §3.6 null-model choice rule run on D. Decisive threshold left pending (decision 4) | Claude (Opus), for Ben | no |
| 2026-10-28 | Ben's criterion: new must-hold M9 (one-sided 95% band credible-set coverage on B and on its misspecified subset), the headline orbit criterion; per-band IS SE carried, and a credible set containing an evidence-noisy/failed band fails M7 | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben's change: misspecified nulls widened (WR-like Gaussian/ring envelopes, log-uniform fraction and size, no named-target parameters; elongated primary; faint background source); third template T3 (CHARA, ~330 m) drawn uniformly in all strata; M6 must hold on the misspecified-null subset too; false orbits reported by template, kind and envelope fraction | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben's decision 5: flux in (lo, 1] with no upper number, disc-on-brighter/fainter label enumerated (B draws the disc on either star); lo from the data's 2σ Ruffio contrast curve by default (A keeps its fixed floor; B/N/calibration use the default; B truths below floor reported and count as M9 misses) | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben's decision 4: no nats thresholds; alias/label/resolved verdicts are probabilities with 95% credible sets; detection by per-data-set empirical FAP < α = 0.01 (`virgil.detection` simulators, stratum-N-like nulls); M6 limits now binomial q99 at α = 0.01 (N ≤ 4 of 100, misspecified subset at its own *n*); M8 qualifying systems by oracle FAP at the MC floor; no separate calibration stratum | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben's amendment: WR-like envelopes are chromatic (a line in a random contiguous channel block, generic enhancement, width and optional size change), in gate draws and FAP nulls | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben accepts the contrast-curve flux floor (register entry records his empirical-Bayes framing); stratum A's fixed floor and the below-floor M9 misses stay | Ben (encoded by Claude, Opus) | no |
| 2026-10-28 | Ben: the harness reuses the existing injection–recovery stack (detection_campaign, sbc_v2, `virgil.detection`); detection v3 (job 18467928) pulled and reported first | Ben (encoded by Claude, Opus) | no |
| 2026-10-29 | §3.6 candidate (iii) replaced by (iii′) global-scale inflation (orbit_mixture_v2 §5); YAML name `shared_noise_inflation` → `global_scale_inflation` | Ben (approved in chat 2026-10-29; encoded by Claude, Opus) | yes: the #586 strict xfail showed (iii) cannot separate any night |
| 2026-10-29 | YAML repair: `null_choice` moved to top level; it sat between two `must_hold` entries and the block did not parse (`yaml.safe_load` fails at `M4_search_misses`); no criterion changed, hash changes | Claude (Opus), for Ben | no |
| 2026-10-29 | stratum A draws each global noise scale s_g from the fit's log-uniform prior (SBC exactness under orbit_mixture_v2 §2); s_g ranks reported, not tested; showcases report ŝ_g after raw χ²/N | Ben (approved in chat 2026-10-29; encoded by Claude, Opus) | no |
| 2026-11-01 | §3.3 stratum B: truths with *P* < P_cut = max(2Δt_min, P_night) (`virgil.aliases.period_cut`; orbit_alias_cycles §1.3) are excluded from scored metrics and counted, not redrawn (the draw and its stream are unchanged); reported stratum B′; YAML `population` key added (criteria hash changes); the baseline 19560581/86 stays reported under its own commit, no mixing. Proposed, awaiting Ben's sign-off | Claude (Sonnet), for Ben, on the 0.4.2 coordinator's decisions | no (no G1 aggregate had been seen when written) |

## References

- Cook, Gelman & Rubin 2006, J. Comput. Graph. Stat. 15, 675 (validation
  of Bayesian software by simulation).
- Talts et al. 2018, arXiv:1804.06788 (simulation-based calibration).
- Raghavan et al. 2010, ApJS 190, 1; Moe & Di Stefano 2017, ApJS 230, 15
  (eccentricity distributions of short-period binaries).
- ten Brummelaar et al. 2005, ApJ 628, 453 (CHARA Array station
  positions).
