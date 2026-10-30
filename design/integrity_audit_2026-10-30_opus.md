# Integrity audit, 2026-10-30 (Opus, independent)

**Scope:** virgil at origin/main `05e50c2` and virgil-validation (vv) at origin/main `76873d3`.
**Method:** read-only. Five parallel sweeps covered the orbit stack, tests, the other core modules, vv gates and public docs. Each sweep used grep, `git log -S/-G/-p` and blame. I then re-checked every blocker and most should-fix items against the code myself. No code was run beyond reading.
**Independence:** I did not read the Argon audit, `release_review_0.4.2_integrity.md`, `audit_0.4.2_claude.md`, or issues #567 and #585. Overlap with them is independent agreement, not copying.

**Asked for by Ben (2026-10-30):** gate cheating, target tuning, loosened tests, non-independent validation, silent fallbacks, overclaiming docs, and debt that hides behaviour.

## Summary

The stamp purges worked. No target name survives in `src/` except one literature citation (`models.py:2344`) and one performance comment (`_geometry.py:120`). I found no `if target == …` branch. The tests contain no vacuous NaN asserts, no assertion-swallowing `try/except`, and no virgil-made golden keyed to a named target. Most tolerance changes carry a measured or derived justification. vv generates its data independently (`crosscheck.simulate`) in every gate checked. Its pre-registration helper fails closed where it is used.

What remains falls into four groups:

1. **The orbit gate, which decides the 0.4.2 orbit claims, is not actually locked.**
   - It sits outside vv's criteria registry.
   - Its "hash lock" only compares each run with the file as it is now.
   - The vv and virgil copies of the gate have already diverged.
2. **The 0.4.0/0.4.1 gate-driven heuristics are still the shipped default** in the orbit engine:
   - short arcs on, with a sign gate divided by the tempering factor *f*;
   - nats thresholds decide every verdict;
   - a failed band silently loses its posterior mass.

   The de-hack plan (`design/orbit_dehack.md`) covers most of this, but the code has not caught up with decisions 4 and 7.
3. **Silent fallbacks in DiskPipeline** report "sd = 0" for a non-PD Hessian, and silently use unscaled errors when a report is missing.
4. **Pipelines whose checks mostly cannot fail.** About 35 of about 48 checks return only pass/warn. Default thresholds were set on the data they were developed on (contest bench, nine post-AGB discs) and are not labelled as such.

None of this is concealment: commit messages are candid, and xfails are mostly strict with measured reasons. The risk is a referee reading the defaults and verdicts as validated when they are tuned.

### Blockers for 0.4.2 (ten-line version)

1. **VV-1** The orbit gate isn't in `criteria_registry.json`. `aggregate` compares the run hash only with the current file, and `--allow-hash` accepts any string (`vv scripts/orbit_population.py:806-809, 844`).
2. **VV-2** The vv and virgil copies of `orbit_gate_042.md` differ: the null candidate is (iii) vs (iii′), and the stratum-A noise draw differs. There is no single authoritative pre-registration. **Ben decides which copy is authoritative.**
3. **VV-3** The vv harness rewrites the registered criteria block in memory when it fails to parse as YAML, and never fails on that.
4. **OR-1** `short_arcs=True` is still the default (`aliases.py:2929`), against decision 7. Its sign gate divides a nats gap by *f* (`aliases.py:2266-2280`), a change added after the Gl 229 gate failed (32acb33).
5. **OR-9** A band that raises is given log Z = −∞, and the posterior is renormalized over the rest (`aliases.py:3375-3407`). The pipeline only lists the failed band and warns.
6. **CO-1** DiskPipeline clips negative Laplace variances to sd = 0 (`pipeline/disk.py:1035`): "perfectly constrained" where the truth is "unconstrained".
7. **CO-2** DiskPipeline falls back to error-scale factor 1.0 when the profile report is missing or unreadable, with no warning (`pipeline/disk.py:343-352`).
8. **TS-3** The default StarPipeline model set fails its own sampler checks on a clean star: the limb-darkened chains have r̂ 2.4–4 (virgil#580, strict xfail at `tests/test_pipeline_robust_star.py:444`). **Ben decides:** fix it or drop limb darkening from the default set.
9. **VV-14** The vv evidence files name three different virgil commits. The one-commit-per-release check runs only with `VALIDATION_RELEASE=1`, so it must be run at the 0.4.2 commit before tagging.
10. **OR-4** Nats thresholds still decide alias, no-orbit, label and resolved verdicts (decision 4 says none). This is tracked by de-hack stages 4–5, which decision 15 puts in 0.4.2. Until it lands, the docs must call the verdicts uncalibrated.

## Severity and ID key

- **B**: blocker for 0.4.2.
- **S**: should-fix, before 0.4.2 if cheap.
- **N**: note.
- **Ben** marks a finding that needs Ben's decision.
- **Dehack-n** marks an item already scheduled in `design/orbit_dehack.md` stage *n*. Those go into that stage's brief; no new package is needed.
- **ID prefixes:**
  - **OR**: orbit modules.
  - **CO**: other core modules and pipelines.
  - **TS**: tests.
  - **VV**: virgil-validation.
  - **DC**: public docs and claims.
- **Sizes:** S < 1 h, M ≈ half a day, L > 1 day (Sonnet worker time).

---

## 1. Orbit modules (virgil)

No target name or published period appears in `aliases.py`, `arcs.py`, `epochs.py`, `orbits.py`, `orbit_search.py`, `tempering.py` or `pipeline/orbit.py`. Stage 0 and Stage 2 landed: there is a single outlier mixture, and the plain-search/polish split is gone. The rest of the 0.4.0/0.4.1 cascade is still on the default path.

### OR-1 [B, Dehack-4] Short arcs on by default; sign gate divided by f

- **Where:**
  - `aliases.py:2929`: `short_arcs=True,  # deliberate: Nyquist-style bands alone leave hundreds of cycle counts`
  - `aliases.py:2266-2280` (`_signs_trusted`)
  - `arcs.py:66, 74` (`ARC_DELTA_CHI2 = 25`, `ARC_MIN_GAP = 5.0`)
- **Evidence:**
  ```python
  gaps = onp.asarray(positions.gap_marginal, float)[keep] / temper_f
  return gaps.size > 0 and bool(onp.all(gaps >= min_gap))
  ```
- **History:**
  - 6a2a7b4 (2026-10-19) replaced a median-flux < 0.8 constant with the 5-nat gap.
  - 32acb33 (same day) is "Temper the short-arc sign gate … by the channel factor f". Its body says the 0.4.0 Gl 229 gate failed because three nights' gaps (44, 8.6 and 140 nats at f = 41) passed.
- **Why it matters:**
  - This is the hack the 2026-10-28 review named, and it is the shipped default.
  - Decision 7 (`orbit_dehack.md:1062`) says "Short arcs default off; F is opt-in and experimental".
  - Whether short arcs act at all depends on *f*, which is itself a symptom of misfit.
- **Fix:**
  - Default `short_arcs=False`.
  - Delete `_signs_trusted`, the division by *f* and `ARC_MIN_GAP` from the default path.
  - Keep F behind an explicit opt-in marked experimental, in the same PR (decision 15: the PR that flips a default deletes the old path).
  - **Size:** M.

### OR-2 [S, Dehack-4] Short-arc stage re-admits excluded bands from three seeds

- **Where:** `aliases.py:2331` (`_N_CHECK = 3`), `:2355-2400`, `:2374` (`readmitted = [b for b in excluded if score[b.n] > top]`). Introduced in 34208a2.
- **Why it matters:**
  - This is a heuristic that undoes a heuristic.
  - The band set depends on the unrefined likelihood of three seed orbits.
  - §0(F) of the plan says "no re-admission".
- **Fix:** delete it with OR-1. **Size:** S.

### OR-3 [S, Dehack-3] Flux-flip swap-back queue and hard clamps

- **Where:**
  - `aliases.py:1350-1394` (`_flip_modes`)
  - `:1362` (`bounds.upper[7] - 0.02`)
  - `:1385` (`if len(queue) < 2 * len(modes_xy):  # one swap back`)
  - `:1187-1202` (`0.9 * top`)
- **History:** the swap-back came in 52c7013 (2026-10-21), whose body cites the Gl 229 gate regression.
- **Why it matters:** label-swap modes are found by a capped cascade rather than enumerated. The 0.02 and 0.9 constants have no derivation.
- **Fix:** §2I of the plan (flux in (lo, 1], with explicit brighter/fainter labels). Use two deterministic starts per mode, with no queue. **Size:** M.

### OR-4 [B, Dehack-4/5] Nats thresholds decide every verdict

- **Where:**
  - `pipeline/_checks.py:1432`: `alias_ambiguity(warn=10.0, fail=1.0)`
  - `pipeline/_checks.py:1650`: `no_orbit(threshold=10.0)`
  - `pipeline/orbit.py:160-161, 978-979`: `"verdict": "orbit" if delta >= s["no_orbit_nats"]`
  - `aliases.py:123`: `LABELS_FREE_NATS = 3.0`, with the comment "Why 3: docs/orbit_fitting.md"
  - `epochs.py:2401`: `RESOLVE_NATS = 5.0`
  - `epochs.py:2100-2101`: flag nats 2 and 5
  - `epochs.py:2004`: `sidelobe_flags(delta_nats=10.0)`
  - `aliases.py:2186`: `_POOR_FIT_CHI2_RED = 10.0`
- **Why it matters:**
  - Decision 4 says no nats thresholds anywhere, and that is not implemented.
  - The labels users read ("orbit", "no-orbit", "decisive") are fixed cuts with no calibrated false-alarm rate.
- **Fix:**
  - Long term: §2G (probabilities, credible sets, an empirical FAP). **Size:** L.
  - Interim, before tagging: every verdict docstring and check message says "uncalibrated threshold". This goes in WP-11. **Size:** S.

### OR-5 [S, Ben, Dehack-2 rule] The "no orbit" test is not a Bayes factor

- **Where:** `pipeline/orbit.py:945-981` (`_null_verdict`).
- **Evidence:**
  ```python
  null = float(max(onp.max(ll), onp.max(best(near))))
  delta = float(result.best.log_z - null)
  ```
  The orbit side is a Laplace evidence with an Occam penalty. The null side is a maximum over a 100-point diameter grid, with no prior volume.
- **Why it matters:**
  - The comparison is biased toward "no orbit".
  - The docstring ("the orbit's prior volume is in its evidence") does not say that the null's prior volume is not.
- **Fix:**
  - Marginalize the null's diameter under its prior, giving a true evidence ratio.
  - Alternatively, the shared-mixture null chosen by the gate rule.
  - **Size:** M.
  - **Ben:** confirm the interim wording until the gate's null choice is run.

### OR-6 [S] Pipeline retypes engine defaults and silently inherits others

- **Retyped literals:**
  - `pipeline/orbit.py:148-153` repeats `a_range`, `flux_range`, `ecc_max` and `vis_amp` as literals instead of reading `_engine_defaults`.
  - `pipeline/orbit.py:596` uses `geomspace(lo, hi, 13)`, which duplicates `epochs._N_FLUX`.
  - `"max_grid": 201` duplicates `epochs._MAX_AXIS`.
  - `"coarse_nats": 30.0` duplicates the `epoch_positions` default.
  - `_STEP_FRACTION = 0.25` (`pipeline/orbit.py:44`) must equal `aliases._BROAD_STEP` (`aliases.py:2738`), and only a comment ties them.
- **Values that differ:** `posterior_p_min = 0.05` (`pipeline/orbit.py:162`) against the engine's `p_min_samples = 0.01` (`aliases.py:2915`).
- **Never passed:** `short_arcs`, `skip_flagged`, `flag_nats`, `k`, `eccs`, `diam_range` and `ess_min`. The engine defaults apply silently, including OR-1.
- **Fix:**
  - Use `_engine_defaults` for every shared name and import the constants.
  - Either pass every engine setting explicitly, or list the ones the pipeline does not expose in its docstring.
  - **Size:** S.

### OR-7 [S, Ben: decision 12] Absolute-scale prior defaults

- **Where:**
  - `aliases.py:2899-2900` (`a_range = (1, 30)` mas, `flux_range = (0.01, 1)`), unchanged since ca88753.
  - They set the prior volume (`aliases.py:675-690`), the outlier disc (`:2766`) and the pipeline grid cap (`pipeline/orbit.py:584`).
  - `orbits.py:875` grids eccentricity on `arange(0, 0.91, 0.05)` regardless of `ecc_max`.
- **Why it matters:**
  - The orbit-vs-null evidence and the outlier component scale with these numbers.
  - The Gl 229 and ι Peg orbits lie inside them.
- **Fix:** decision 12 (`a_range` in resolution units), and tie the eccentricity grid to `ecc_max`. **Size:** M.

### OR-8 [S, Dehack-3] Window and seeding constants with no derivation

| Constant | Where |
|---|---|
| `span_factor = 4` (decision 13 deletes it) | `aliases.py:214-268` |
| `_P_FLOOR = 0.2` d | `:138` |
| `_N_BANDS_MAX = 200`, which silently raises the lower window edge (warning only) | `:141`, `:3025-3035` |
| random-start `0.6 * ecc_max` and cos i ∈ ±0.95 | `:1150-1151`, from b98564f |
| flux seeds 0.7 and 0.95 | `:836-840` |
| `_N_RANDOM_FEW = 200`, `_N_RANDOM_NONE = 1000` | `:2189-2192` |
| sign-pair lags `[1, 2, 3, 5, 8, 13]` | `orbits.py:708, 746` |

- **Fix:**
  - Stage 3 replaces these.
  - Until then, document that they affect only the starts. The exception is `_N_BANDS_MAX`, which changes the window: make its cap a check, not only a warning.
  - **Size:** L planned, S interim.

### OR-9 [B] A failed band silently loses its posterior mass

- **Where:** `aliases.py:3375-3407`.
  ```python
  except Exception as caught:  # noqa: BLE001 - keep the other bands
  ...
  for b, w in zip(results, weights / max(weights.sum(), 1e-300)):
      b.p = float(w)
  ```
- **Behaviour:**
  - A band that raises gets log Z = −∞, and the probabilities are renormalized over the survivors.
  - The engine raises only if every band fails (`:3393`).
  - The pipeline lists `failed_bands` in the summary (`pipeline/orbit.py:860`) and warns.
  - Similar swallows are at `aliases.py:1256` and `:1373` (starts), `:2815` (`sidelobe-check-failed`) and `epochs.py:2910`.
- **Why it matters:** a systematic failure at long period, for example, removes that alias from the posterior and promotes the rest. The reported p of the winner is then wrong, not merely uncertain.
- **Fix:**
  - Narrow the catch to the numerical errors expected (`FloatingPointError`, `LinAlgError`, the solver's own non-convergence).
  - Make the existing `bands_failed` check (`pipeline/_checks.py:1477`, wired at `pipeline/orbit.py:1475`) return **fail** instead of `warn`, and leave `p = NaN` (not renormalized) when any band failed. The failure is already reported by a named check; the defect is that the check stops at `warn`.
  - **Size:** S (engine) + S (check).

### OR-10 [S, Dehack-4] Band evidence ranks by Laplace even when importance sampling disagrees

- **Where:** `aliases.py:1269-1335`.
  - `laplace-is-differ` is flagged when the two differ by more than 1 nat, but the band keeps `z_lap`.
  - Modes with a singular Hessian are dropped (`:1049-1058`).
- **Fix:**
  - §2C (truncated importance sampling at bounds, with a reported standard error).
  - Interim: report `z_is` and its standard error in `summary.modes`, and rank by the lower of the two when they disagree.
  - **Size:** M.

### OR-11 [S, Ben] Tempering is unbounded and only warns

- **Where:**
  - `aliases.py:2635, 2653` (`_INADEQUATE_F = 3`, warning only, and only for calibrated data)
  - `pipeline/_checks.py:1585` (`tempering` is never `fail`)
  - `tempering.py:35-40` (`FLAG_RANGE`, `MIN_CHAINS`, `FLAT_EXCESS = 0.15` with no derivation)
  - The *f* estimate passes through `_auto_primary` up to three times (`aliases.py:3109-3128`).
- **Why it matters:**
  - At f = 41 (Gl 229) the evidence is flattened 41-fold, and the pipeline still names a winner with a warning.
  - The warning's own text says tempering "hides a misfit".
- **Fix:**
  - Fail, or refuse to rank, above a stated *f*.
  - Run one auto-primary pass per diameter hypothesis.
  - **Size:** M.
  - **Ben:** the fail threshold, and whether tempering stays on by default (see DC-7).

### OR-12 [S] Claims stronger than the evidence in orbit code

- `arcs.py:63-66`: "generous so that the true cycle count is never cut". There is no test, and OR-2 exists because it can be cut.
- `aliases.py:2360`: "never silently drops a band the cheap check prefers".
- `pipeline/orbit.py:1, 87, 766`: "blind period-alias search".
  - It is blind to published periods only.
  - The design note's own gallery shows blind runs picking the wrong alias (ι Peg χ²/N 27, Gl 229 B 12).
- `_checks.py`: "the winning period is decisive" at ≥ 10 nats.
- **Fix:** reword. **Size:** S. Files that de-hack stages own wait for those stages; `pipeline/orbit.py` and `_checks.py` text goes in WP-5.

### OR-13 [N] Warning text parsed as control flow

- **Where:**
  - `pipeline/orbit.py:804-812` rewrites the engine's message, replacing "Pass p_range" with "Set period_range".
  - `pipeline/_checks.py:1185-1210` maps warnings to checks by keyword.
  - CO-9 is the general case.
- **Fix:** typed warning classes. **Size:** M (shared with CO-9).

### OR-14 [N, Dehack-5] Dead and compatibility code in a "provisional" module

- `orbit_search.py` (1110 lines) is imported by no `src` module (decision 9: remove in 0.5).
- `AliasResult.load` has a format-0 reader (`aliases.py:2043-2140`) for files `_band_target` then refuses to use.
- `outlier_eps` is accepted and ignored (`:2998`), and `pipeline/orbit.py:129-132` keeps `_IGNORED`. The module promises no deprecation period, so remove these outright.
- `estimate_primary_diameter` is scheduled for removal but still on the main path.
- `fit_orbit_aliases` is 606 lines with 40 parameters (`aliases.py:2892-3497`).
- **Size:** S each, L for the split (Stage 5).

### OR-15 [S] Night flags use untempered nats

- **Where:**
  - `epochs.py:2093`: `epoch_positions(dof=1.0)` by default.
  - `epochs.py:2100-2103`: flag thresholds.
  - `aliases.py:2931`: `skip_flagged=True`.
  - `epochs.py:1904-1925`: `flux-at-bound` is blocking.
- **Why it matters:**
  - The same gap means different things in the two layers, and the *f* division in OR-1 papers over that.
  - Near-equal pairs lose their seeding nights.
- **Fix:** pass the tempering dof into `epoch_positions` from the pipeline, or make the flags advisory. **Size:** M.

---

## 2. Core modules and pipelines (virgil)

### CO-1 [B] Negative Laplace variance reported as sd = 0; any failure swallowed

- **Where:** `pipeline/disk.py:1032-1044`.
  ```python
  sd = onp.sqrt(onp.clip(onp.diag(cov), 0.0, None))
  except Exception as err:  # noqa: BLE001 - a Laplace failure is a result
  ```
- **Why it matters:**
  - A non-PD Hessian gives negative diagonal entries, which are published as zero error bars.
  - `except Exception` also turns a programming error into "Laplace failed".
- **Fix:**
  - Return NaN for any variance ≤ 0.
  - Catch only `LinAlgError` and `FloatingPointError`.
  - Add a `laplace_pd` check that warns or fails.
  - **Size:** S.

### CO-2 [B] Missing report silently gives unscaled errors

- **Where:** `pipeline/disk.py:343-352`.
  ```python
  except (OSError, ValueError, KeyError): value = None
  self._error_factor = 1.0 if value is None else float(value)
  ```
- **Why it matters:**
  - With `error_scale="auto"` → `"rescale"` (DiskPipeline's default), a missing or corrupt profile report gives χ², evidences and widths on the wrong errors, with no message.
  - The same path runs when resuming from disk.
- **Fix:** raise, or record a failed check. Never fall through to 1.0. **Size:** S.

### CO-3 [S] Evidence and render-check failures become warning + None

- **Where:** `pipeline/disk.py:1632` (the ℓ scan silently loses that ℓ) and `:1965`.
- **Fix:**
  - Catch only the numerical exceptions.
  - `length_scale` and `render_consistency` report "k of n candidates failed" as their own status.
  - **Size:** S.

### CO-4 [S] Absolute 1e-10 ridge in Laplace covariances

- **Where:** `inference.py:67, 91, 310`.
- **Why it matters:**
  - The ridge is unit-dependent, and it caps the variance of a poorly constrained parameter at about 1e10 without saying so.
  - `fisher_projection` (`inference.py:151`) uses a relative floor and warns, so the two are inconsistent.
- **Fix:** scale the ridge by `max(diag H)`, warn when it matters, and document it. **Size:** S.

### CO-5 [S, Ben] Observed visibilities floored at their own error in `vis_mode` conversion

- **Where:** `oidata.py:936-968`, introduced in 25a5805 (review fixes).
  ```python
  floored = np.maximum(vis, np.maximum(d_vis, 1e-30))
  return 2.0 * np.hypot(vis, d_vis) * d_vis   # |V| -> V² error
  ```
- **Why it matters:**
  - Data below 1σ are replaced by σ in log-amplitude, which biases the channel high and shrinks its error.
  - The V² error is not first-order propagation.
  - So χ² depends on `vis_mode` by convention, with no warning and no record of which rows were floored.
- **Fix:**
  - Warn with the count of floored points, and cite or derive the convention.
  - Alternatively, refuse logamp when many points are below 1σ.
  - **Size:** S (warning), M (derived treatment).
  - **Ben:** which treatment counts as state of the art here.

### CO-6 [S] Two different "at the bound" fractions

- **Where:**
  - `fitting.py:961` (`_AT_BOUND = 1e-3`, used for the MAP)
  - `pipeline/_common.py:23` (`_BOUND_FRACTION = 0.01`, used for samples). Its docstring says it is "the one `fit` reports".
- **Fix:** one shared constant. **Size:** S.

### CO-7 [S, Ben] Detection threshold and trial count

- **Where:** `pipeline/binary.py:147` (`detection_sigma = 3.0`) and `:454` (`n_trials = max(1, (2*max_sep/(λ/B))²)`, Šidák).
- **Why it matters:**
  - A "pass: companion detected" at an approximate 3σ global, with an ad hoc trial count, is the headline claim of BinaryPipeline.
  - Other paths use Δχ² 25 (`_checks.py:504`; DiskPipeline `companion_min_delta_chi2`).
- **Fix:** default to the simulated null (`gaussian_null` exists), or a 5σ pass with 3σ as "candidate". **Size:** M. **Ben.**

### CO-8 [S, Ben] Most checks cannot fail; χ² thresholds ignore N

- **Where:**
  - About 35 of about 45 public checks can return only pass or warn. Seven do so through `_two_way` (`pipeline/_checks.py:1739`: `quoted_errors`, `effective_parameters`, `star_decision`, `orientation`, `edges`, `ensemble_kept`, `ensemble_spread`). The rest build their `Check` by hand, including `detection`, `prior_bound`, `multimodal`, `tempering` and `convergence`. Changing `_two_way` alone does not fix this.
  - `chi2_reduced` and `error_scale` warn at 2 and fail at 5 for any N (`:90`, `:129`), unchanged since ffb97a8 and 1ee26df.
- **Why it matters:**
  - A run whose headline conclusion is unsupported can still report `warn`.
  - χ²/N = 2 is about 10σ at N = 1000.
- **Fix:**
  - N-aware thresholds (a χ²_N p-value).
  - Allow `fail` in checks that invalidate the headline result.
  - **Size:** M. **Ben:** the policy.

### CO-9 [S] Warning-to-check mapping by English keywords

- **Where:** `pipeline/_checks.py:1081-1100, 1200-1211`. The keywords are "converge"+"optimizer", "resolve"+"peak" and "clipped", and the regex `(\d+)\s+of\s+(\d+)` parses counts out of messages.
- **Why it matters:** rewording a library warning silently changes which check fires.
- **Fix:** typed warnings with attributes. **Size:** M. Shared with OR-13, and goes in WP-6.

### CO-10 [S, Ben] Ensemble weight ranges tuned on the contest bench

- **Where:** `ensemble.py:124-135`. The comment says "each range holds the L-curve corners of the contest bench with a decade to spare". Set in efcd558.
- **Why it matters:**
  - The defaults that set the ensemble's σ map were chosen on the benchmark used to judge it.
  - No held-out dataset checks them.
- **Fix:**
  - Say "set on the SPIE contest bench" in the docstring and in `docs/imaging_ensemble.md`.
  - Add a held-out check in vv, or choose the range per dataset from an L-curve.
  - **Size:** S (label), M (held-out check).

### CO-11 [S] Engine and pipeline defaults differ for the same quantity

| Quantity | Engine | Pipeline |
|---|---|---|
| ensemble oversample | `(2, 3, 4)` (`ensemble.py:137`) | `3.0` (`pipeline/image.py:101, 229`) |
| GP growths | 3 | 2 (`pipeline/image.py:109, 238`) |
| field factors | `(0.75, 1, 1.5)` | `(1.0,)` |
| starts | `("moments", "flat")` | `("moments",)` |
| `error_scale` | Star/Binary `"quoted"` | Disk `"auto"` |
| companion flux prior | Binary `[1e-5, 0.5]` | Disk planet `(1e-4, 1.0)` |

- **Why it matters:** ImagingPipeline's σ map is measured over a smaller ensemble than the one documented, so it comes out smaller. Binary rules out companions brighter than half the primary, and `limits_clipped` only warns.
- **Fix:** default the pipeline settings to `None`, meaning "engine default". Document every deliberate difference. **Size:** S.

### CO-12 [N] Defaults set on the data they were developed on

- DiskPipeline defaults come from 1ee26df, "defaults from imaging nine PIONIER post-AGB discs": the ℓ grid, `max_rings`, `_HARMONIC_SCALE`, `_GP_SIGMA_BOUNDS` and others.
- `_geometry.py:119-120` justifies the FT backend rule with IM Lup/A100 timings and job IDs. These are performance thresholds, not accuracy ones.
- **Fix:** add a "defaults provenance" table (commit, data, held-out check) to the pipeline design note, and move the job IDs out of the code comment. **Size:** S.

### CO-13 [N] Unreferenced constants

- `fitting.py:1040` (`_P_FLOOR`)
- `fitting.py:1349` (the gtol rule)
- `ensemble.py:763` (a pedestal on the start template)
- `imaging.py:555` (`largest_mas = 500`)
- `tempering.py:364` (*f* floored at 1; documented)
- `FLAT_EXCESS = 0.15`

Most of these set starts or speed only. **Fix:** a "convention, not derived" line where one is missing. **Size:** S.

### CO-14 [N] Long functions that hide behaviour

| Function | Lines |
|---|---|
| `plotting.plot_field_draws` | 305 |
| `imaging.clean` | 273 |
| `OIData.__init__` | 269 |
| `fitting.fit` | 253 |
| `ensemble.combine` | 244 |
| `radio_clean.clean` | 241 |
| `detection.injection_recovery` | 237 |
| `svi.variational` | 212 |

`fit`, `OIData.__init__` and `combine` hide method selection, casting and member filtering. Not a blocker. **Size:** M each.

### CO-15 [S, Ben] Ensemble members dropped by cuts relative to the best member

- **Where:** `ensemble.py:1040-1058`:
  - `chi2_ratio = 2.0`, relative to the best member;
  - a 5×MAD cut;
  - `max_chi2_red = inf` by default.
- **Why it matters:** if every member fits badly, every member passes. `ensemble_kept` only warns.
- **Fix:** report the number dropped for each reason in the summary, and consider a finite absolute default. **Size:** S.

---

## 3. Tests (virgil)

These checks came back clean:

- no `assert not (x > tol)` patterns;
- the NaN-median paths go through `allclose`, which fails on NaN;
- no `try/except` swallows an assertion;
- no virgil-made golden is keyed to a named target;
- the calibration move to main-only (c7d4a22) changed only markers;
- the thresholds 1.45 and 2.0 have not moved since a tightening (4c92716);
- the radio PSF self-golden was loosened in a54a1fc, then tightened back and labelled as a pin in #591;
- the Toeplitz σ rtol change (77b99bb) is derived and paired with a tighter same-parameters check.

### TS-1 [S, Ben] The night-flag test's bad-night bar was lowered; the fit starts at the truth

- **Where:** `tests/test_night_flags.py:148, 153-177`.
- **History:** 2011949 (2026-10-29) changed `prob[BAD] > 0.9` to `> 0.5`. In the same change the clean-night bound tightened from 0.1 to 1e-6. The new comment derives q ≈ 0.7–0.8 from a 3-nat likelihood ratio and Beta(½, ½); the measured values were 0.83 and 0.69.
- **Why it matters:**
  - The derivation is honest, but it shows the mixture's headline property is only half met: a −105 log-likelihood night keeps weight 0.2–0.3.
  - The test starts at the truth (`x0 = ta._x_truth()`), and the strict xfail at `:199` shows a 140-nat stall from 0.2 prior-scales away.
- **Fix:** add an off-truth variant once continuation in the outlier rate lands (mixture v2), and state the q margin as a ratio. **Size:** S/M. **Ben:** is q ≈ 0.7 acceptable for the default?

### TS-2 [S, Ben] Strict xfails record real defects in the default mixture

- **Where:** `tests/test_night_flags.py:186`, the inflation candidate; it is being replaced by (iii′).
- **Where:** `tests/test_night_flags.py:199`. The default `mixture` stalls 140 nats below its optimum from near-truth starts.
- **Assessment:** recording these as strict xfails is the right thing to do.
- **Fix:** before 0.4.2, land the continuation fix (mixture v2 PR A) or list the stall as a known limitation in the 0.4.2 notes. **Size:** M. **Ben.**

### TS-3 [B, Ben] The default StarPipeline model set fails its sampler checks

- **Where:** `tests/test_pipeline_robust_star.py:444`. It is a strict xfail: "virgil#580: limb-darkened star NUTS chains do not mix (r̂ 2.4–4, ESS 2.5)".
- **History:**
  - 8274abc (2026-10-28, review fixes with argon-reviewer) changed `StarPipeline` to report sampler health for the selected model only, and **removed this test's xfail**.
  - 7f6464d restored the xfail, and 2011949 made it strict.
  - On main, `pipeline/star.py:483-502` takes the worse status of selected and non-selected models.
- **Why it matters:**
  - Main is honest now, but the episode shows how close this came to a check being softened to make a test pass.
  - Users running the default StarPipeline on a clean star get a `fail`.
- **Fix:**
  - Fix the limb-darkening parameterization (#580), or drop limb darkening from the default set until it mixes.
  - Do not reinstate "selected only" reporting without Ben's sign-off.
  - **Size:** M–L. **Ben.**

### TS-4 [S] Module-level `importorskip` can skip whole stacks silently

- **Where:** about 20 files (`test_pipeline_*.py`, `test_aliases.py`, `test_epochs.py`, `test_arcs.py`) skip at module level on h5py, nbclient or jaxoplanet.
- **Why it matters:** if a CI job lacks one of these, the orbit stack and every pipeline test pass vacuously.
- **Fix:** add a CI-only test asserting that these imports succeed in the main job, using the existing lint test pattern `tests/lint/test_ci_config.py`. **Size:** S.

### TS-5 [S, Ben] A non-strict xfail on float32 L-curve convergence

- **Where:** `tests/test_fit_precondition.py:558-566` (`strict=False`).
- **Why it matters:** float32 is JAX's default dtype, and this records an unstable convergence result that is allowed either outcome.
- **Fix:** bound it (for example, "converges at ≥ N of M weights"), or make it strict once deterministic. **Size:** S.

### TS-6 [N] Search effort cut in alias tests

- **Where:** 3909210 (2026-10-12) cut `n_candidates` from 40 to 10, `n_random` from 300 to 64 and `n_refine` from 4 to 2 for speed. Assertions are unchanged.
- **Why it matters:** less effort means more dependence on lucky seeds.
- **Fix:** run the same assertions over 5 seeds in the slow tier. **Size:** S.

### TS-7 [N] The alias-night fixture is a built positive control

- **Where:** `tests/test_aliases.py:1291-1370`, changed in c14a325 after the scanned-seed fixture stopped producing an alias.
- **Assessment:** it now tests that the flag fires on a constructed reflection, not that a realistic Gl 229-type case is flagged. The commit is candid about this.
- **Fix:** keep it, and add a sentence to the test docstring. **Size:** S.

---

## 4. virgil-validation

These checks came back sound:

- `crosscheck.simulate` and `crosscheck.sky`/`orbits` (numpy) generate all gate data, with virgil only the system under test.
- `crosscheck.preregistration.check` raises on a mismatch or a missing pin.
- `--allow-unregistered` stamps the run and downgrades PASS.
- No `try/except` swallows a gate failure.
- No `by:` sign-off counts while `allowed_signers` lists no key.

### VV-1 [B] The orbit gate is outside the registry; its lock only checks a run against the current file

- **Where:**
  - `scripts/orbit_population.py:806-809, 844`.
  - `design/orbit_gate_042.md` is absent from `design/criteria_registry.json`, which I confirmed with grep.
  - The only pin is the `REGISTERED` constant in `tests/test_orbit_population.py:18`.
- **Evidence:**
  ```python
  if seen != {now} and not (name == "criteria_hash" and seen <= set(args.allow_hash)):
  ```
  `--allow-hash` takes any string, although the gate doc (`:12`) says the change log is checked. `draw`, `fit` and `oracle` never refuse an edited block.
- **Why it matters:** edit the criteria, rerun, and the gate is self-consistent. This is the gate behind the 0.4.2 orbit claims.
- **Fix:**
  - Pin the YAML block in the registry, using a block-hash variant in `preregistration`.
  - Route `draw`, `fit`, `oracle` and `aggregate` through `prereg.check` and `stamp`.
  - Drop `--allow-hash`, or require it to match a registry `changed_after_seeing_a_result` entry.
  - **Size:** M.

### VV-2 [B, Ben] The vv and virgil copies of the orbit gate have diverged

- **Where:**
  - vv `design/orbit_gate_042.md`, with one commit only (82bce75).
  - virgil `design/orbit_gate_042.md`, last changed in 6dc50c5 (2026-10-29, "Record Ben's approval of the mixture v2 design and gate changes").
- **Evidence:** `diff` shows two differences. Stratum A now draws `s_g`-scaled noise in virgil but not in vv. Candidate (iii) became (iii′) global-scale inflation, and the YAML `null_choice` candidates and tie-break changed with it.
- **Change log:** virgil's records "changed after seeing a result: yes", following the #586 strict xfail. That disclosure is good.
- **Why it matters:**
  - There are two non-identical "pre-registered" gates.
  - vv's harness would reject virgil's file, and it draws stratum A differently from what virgil now says.
  - The gate is edited in the repo being validated.
- **Fix:**
  - Pick one home. My recommendation is vv, hash-pinned per VV-1, with virgil keeping a pointer.
  - Re-vendor with a registry entry marked post-hoc.
  - Bring `crosscheck.orbit_population` draws into line before any scored run.
  - **Size:** S–M. **Ben:** which copy is authoritative, and whether (iii′) is accepted as a post-hoc change.

### VV-3 [B] Run-time "YAML repair" of the criteria block

- **Where:** `scripts/orbit_population.py:83-100`.
- **Behaviour:** when the strict parse fails, the lines after `must_hold:` are re-indented in memory. The result is recorded as `spec_repaired` but never fails.
- **Why it matters:** the harness decides which tests are must-hold by guessing at a malformed registered file.
- **Fix:** fail on a strict-parse error, and fix the file with a change-log entry. virgil's copy was fixed in 6dc50c5. **Size:** S.

### VV-4 [S, Ben] IM Lup gate (d) moved to the tapered image after the base image showed dots

- **Where:**
  - `src/virgil_bridge/alma_start.py:167, 180`.
  - The `alma_imlup_gate_criteria.yml` entry in `design/criteria_registry.json`.
- **History:**
  - 88b1e7d (2026-10-23): gate approved, with (d) judged on the base image.
  - aa0c2e3 (2026-10-27): "tapered gate (d)". The docstring says "Ben, 2026-10-27: the base image may break into dots".
  - 5ab23a7 extracted the constants and recorded "no value differs".
- **Why it matters:**
  - Ben authorised the change, but it is a rule change after a result.
  - The registry's wording ("no value changed") is true of values and silent on the rule.
- **Fix:** add a dated post-hoc note to the registry entry and the YAML header. Say on any public page that the cold-start "pass" is on the tapered image. **Size:** S. **Ben:** wording for public disclosure.

### VV-5 [N] Pull-campaign design hash narrowed and the summary hash hand-edited

- **Where:** e74f17a, `scripts/pull_campaign.py:55-82`, `trust/campaigns/pull_campaign.json:5`.
- **Assessment:** probably benign, since no scene draws an arc, but it sets a precedent for editing a guard instead of rerunning.
- **Fix:** add a test that no scene in `SCENES` calls the arc functions, and rerun at the release commit. **Size:** S + OzSTAR.

### VV-6 [S] Gates outside the registry, or with a bypass

- `scripts/trackb_fit.py:61-65` skips its hash check if the criteria file is missing.
- `design/oidb_binaries_criteria.md` is enforced by no script.
- `sbc_addendum.yml` and the `CRITERIA` dict in `sbc_numpyro.py` are hashed only against their own summaries.
- **Fix:** pin each in the registry and route each through `prereg.check`; Track B fails if its file is missing. **Size:** M.

### VV-7 [S, Ben] The lock is self-referential

- **Where:** `src/crosscheck/preregistration.py:22-24`. A re-pin in the same commit passes CI, and there is no CODEOWNERS file and no branch protection.
- **Why it matters:** the lock detects drift, not deliberate re-registration.
- **Fix:** a CI job that diffs the registry against the base branch. It fails if an existing pin changes without the PR adding a non-"no" `changed_after_seeing_a_result` entry. **Size:** M. **Ben:** appetite for CI that blocks his own edits.

### VV-8 [S] Orbit gate M4/M6/M8 judge virgil with virgil

- **Where:** `scripts/orbit_population.py:555-575, 609, 641-643`. "Qualifying" systems are those where virgil's own truth-initialized run says `orbit`. The independent FAP basis the gate registered is not implemented.
- **Mitigation:** the top-level `pass` is `None` while `unavailable` is non-empty, but the sub-dict `pass` keys print.
- **Fix:**
  - Label the result "incomplete: judged by virgil's own verdict" and hide the per-test `pass` keys.
  - Add an independent numpy null-distribution FAP to `crosscheck`.
  - **Size:** S (label), L (FAP).

### VV-9 [N] Vacuous pass on empty subsets

- **Where:** `scripts/orbit_population.py:467` (`p = 1.0` if empty), `:546` (`limit = 0` with no confident systems), and `alma_start.py:317-319` (se = 0 with one fold).
- **Fix:** return `pass: None` with n reported. **Size:** S.

### VV-10 [N] The independent orbit population copies virgil's band definition

- **Where:** `src/crosscheck/orbit_population.py:46, 116-123` ("of virgil's documentation").
- **Fix:** say so in the gate doc §2, or derive the bands independently. **Size:** S.

### VV-11 [N] IM Lup gates (a) and (b) are evaluated by virgil on both arms

- **Fix:**
  - Cite the trust-graph test that virgil's χ² equals an independent DFT χ².
  - Record the MPoL run id for the hard-coded 2.7062.
  - **Size:** S.

### VV-12 [S once a key is listed] Sign-off weaknesses

- **Where:** `scripts/trust.py:397-399`. Any `citation:` string counts as a sign-off with no check.
- **Where:** `trust/allowed_signers`. It is editable in an ordinary PR.
- **Fix:** resolve citations against the provenance census, and pin the signer fingerprint outside the repo. **Size:** M. **Ben:** key choice.

### VV-13 [N] Per-target analyst choices in Gl 229 showcases

- **Where:** `scripts/reanalyse_orbits.py:51-55` (`GL229_EXCLUDE`, `GL229_BAND`) and `scripts/orbit_gallery.py:203`.
- **Assessment:** showcases, not gates, so they are allowed.
- **Fix:** label the exclusion on the page as post-hoc, and show the all-nights result beside it. **Size:** S.

### VV-14 [B for the tag] Release evidence pinned to mixed commits

- **Where:** `tests/test_evidence_pin.py:61` is skipped without `VALIDATION_RELEASE`.
- **Evidence:** `trust/evidence/*.jsonl` name 8c51dc5 (v0.4.0), 884aa7e, and a third commit. The pull and detection summaries are at 0726734.
- **Fix:** run `VALIDATION_RELEASE=1` with `--expect <0.4.2 sha>` as a release-checklist step (Ben, after the OzSTAR reruns), and show a Trust-page banner when evidence commits differ. **Size:** S + runs.

---

## 5. Public docs and claims (virgil)

Prose in `src/` is well hedged: a grep for strong words found nothing that makes a claim. No public page links `design/*.md`. Oxford -ize spelling is systematic, with a few stray -ise forms (DC-15).

### DC-1 [S] `comparison.md` says virgil has no measurement-set reader

- **Where:** `docs/comparison.md:43`: "there is no reader for UVFITS or measurement sets".
- **Why it's wrong:** `ComplexVisData.read_ms` exists (`radio.py:230`), and the table is a snapshot at 0771bcc.
- **Fix:** refresh the virgil column at the release commit, and mark the CLEAN, orbit and VI rows as provisional. **Size:** S.

### DC-2 [S] The radio performance table is mostly empty, and "34×" compares two setups

- **Where:** `docs/radio_interferometry.md:394-409`. Six of eight cells are "—", and no committed benchmark reproduces the rest.
- **Fix:** drop "34×", and give each cell its setup or remove the table. **Size:** S.

### DC-3 [S] `conventions.md` promises named tests that are not named

- **Where:** `docs/conventions.md:3`: "the tests named below fail if one changes". Only two lines of the page name a test.
- **Fix:** add a `Test:` line to each section, or soften the sentence. **Size:** M.

### DC-4 [S] The 0.4.1 Highlights drop a hedge

- **Where:** `CHANGELOG.md:29-31`, "30–40% fewer steps". The conventions page says "in the two L-curves measured", and no test asserts it.
- **Fix:** add an erratum line in the 0.4.2 notes. **Size:** S.

### DC-5 [S, Ben] Wording of the Gl 229 gate's retirement

- **Where:** `CONTRIBUTING.md:196-200` and the changelog fragment.
- **Why it matters:** "0.4.1 was the last release it gated" reads as a pass, but the cold-gate pass is unverified.
- **Fix:** "ran once at 0.4.1; result not independently verified". **Size:** S. **Ben.**

### DC-6 [S, Ben] The orbit tutorial is one selected seed on Gl 229-like geometry

- **Where:** `docs/orbit_fitting.md:11, 28, 95, 130` and the notebook.
- **Evidence:** P = 12.38 d, a 238 d gap, and "the first seed from 0 for which …" (disclosed).
- **Fix:**
  - One sentence: "one simulated realization, on the geometry the search was developed on".
  - Optionally, an OzSTAR seed sweep with the recovery fraction quoted, whatever it is.
  - **Size:** S / M. **Ben:** whether to run the sweep.

### DC-7 [S, Ben] Novel default-on methods without provenance

- **Methods:** tempering (`tempering.py`), short arcs (`arcs.py`), the alias search, and the night-outlier mixture.
- **Gaps:**
  - No `references/provenance.toml` entry or `Source:` line was found by grep.
  - There is no calibration test that tempered evidences are calibrated; `test_tempering` checks posterior offsets, not evidence.
  - `docs/orbit_fitting.md:577` and `docs/pipeline.md:229` call the mixture "exact". Only its integral is exact; the changelog fragment says the model was "not chosen on merit".
- **Fix:**
  - Provenance entries marked "virgil's own design".
  - One "Status" line on each orbit page.
  - **Ben:** whether tempering stays on by default before 0.5 (rule: experimental features are opt-in).
  - **Size:** S.

### DC-8 [S] `imaging_ensemble.md` says the ensemble "does the same" as PYRA/MYTHRA

- **Where:** `docs/imaging_ensemble.md:8`. The page's code is not executed at build, and the defaults were set on the contest bench (CO-10).
- **Fix:** "follows the selection rule of …; defaults set on the SPIE contest bench". **Size:** S.

### DC-9 [S] Internal phrase and unverified references in `references.md`

- **Where:** `docs/references.md:499`: "the brief says 2025".
- **Where:** 26 entries are "Not yet verified".
- **Fix:** rewrite the note in `provenance.toml`, and run `scripts/fetch_references.py`, or add a banner with the count. **Size:** S.

### DC-10 [S] Broken and internal links in docstrings

- **Where:** `models.py:5177` links `../notes.md`, which does not exist.
- **Where:** `tempering.py:9-10` points readers at AGENTS.md.
- **Fix:** link the bandwidth-smearing section of `spectro_observables.md`, and remove the AGENTS.md sentence. **Size:** S.

### DC-11 [S] The README "Independent validation" paragraph does not list what is uncovered

- **Where:** `README.md:82-84`.
- **Fix:** one sentence naming the provisional features without independent checks: the alias search end to end, tempering, pipelines, CLEAN, bandwidth smearing and harmonix. **Size:** S.

### DC-12 [N] Bandwidth smearing is modelled, but vv says virgil lacks it

- **Where:** vv `trust/not_covered.md`.
- **Fix:** correct vv, and say "not independently validated" in `spectro_observables.md`. **Size:** S.

### DC-13 [N] The 0.4.1 Highlight describes the removed night drop

- **Fix:** keep the removal prominent in the 0.4.2 notes, and say that showcase results changed. **Size:** S.

### DC-14 [N, Ben] The carbon badge shows one number without its range

- **Where:** `README.md:8`.
- **Fix:** Ben's call on the wording.

### DC-15 [N] Stray -ise spellings

- **Where:** "normalised", "minimise" and "discretisation" in `tempering.py`, a few docs and `AGENTS.md:169`.
- **Fix:** convert to -ize. **Size:** S.

---

## 6. Work packages for Sonnet workers

Each package owns the files listed. Packages that share a file are marked "after". Items tagged Dehack-n are not repackaged here: the coordinator should check that the stage-3/4/5 briefs name OR-1, OR-2, OR-3, OR-4, OR-8, OR-10 and OR-14, and that the stage-4 brief flips `short_arcs` (OR-1 is a blocker, so stage 4 or a standalone flip must land before 0.4.2).

| WP | Repo | Findings | Files owned | Size | Ben first? |
|---|---|---|---|---|---|
| **WP-1** Orbit gate lock | vv | VV-1, VV-2, VV-3, VV-4 (registry note), VV-8 (label), VV-9, VV-10 | `scripts/orbit_population.py`, `design/orbit_gate_042.md`, `design/criteria_registry.json`, `src/crosscheck/preregistration.py`, `tests/test_orbit_population.py`, `tests/test_criteria_registry.py` | M | **yes**: D1 (authoritative copy) |
| **WP-2** Remaining gates in the registry | vv | VV-6, VV-7 (CI diff check) | `scripts/trackb_fit.py`, `scripts/sbc_numpyro.py`, `design/oidb_binaries_criteria.md`, `.github/workflows/*`; registry entries **after WP-1** | M | VV-7 policy (D2) |
| **WP-3** vv trust and showcase hygiene | vv | VV-5 (test), VV-11, VV-12, VV-13, DC-12 (vv side) | `scripts/pull_campaign.py`, `tests/test_pull_campaign*.py`, `scripts/trust.py`, `trust/not_covered.md`, `scripts/reanalyse_orbits.py`, `scripts/orbit_gallery.py`, `src/virgil_bridge/alma_start.py` (docstring only) | S–M | D3 for VV-4 wording |
| **WP-4** Release evidence pin | vv + OzSTAR | VV-14, VV-5 rerun | release checklist in `CONTRIBUTING.md` (virgil), vv `docs/method/index.md` | S + runs | Ben runs jobs |
| **WP-5** Orbit pipeline honesty | virgil | OR-5 (interim wording), OR-6, OR-9, OR-11 (once decided), OR-12 (pipeline text), OR-13 (orbit message rewrite) | `src/virgil/pipeline/orbit.py`, the orbit checks in `pipeline/_checks.py` (`alias_ambiguity`, `no_orbit`, `tempering`, `bands_failed`), `tests/test_pipeline_orbit*.py` | M | D5, D6 |
| **WP-5b** Engine band failure | virgil | OR-9 (engine side: narrow catches, `p = NaN`) | `src/virgil/aliases.py` lines 1250-1380 and 3370-3410 only, **or folded into Dehack-4** if that stage is next | S | no |
| **WP-6** Generic pipeline checks | virgil | CO-8, CO-9/OR-13 (typed warnings), CO-6 (`_common` side) | `pipeline/_checks.py` (generic parts) **after WP-5**, `pipeline/_common.py`, `fitting.py` (`_AT_BOUND` only), `tests/test_checks*.py` | M | D8 |
| **WP-7** DiskPipeline fallbacks | virgil | CO-1, CO-2, CO-3, CO-4, CO-12 (provenance table) | `pipeline/disk.py`, `inference.py`, `tests/test_pipeline_disk*.py`, `tests/test_inference*.py` | S–M | no |
| **WP-8** vis_mode flooring | virgil | CO-5 | `oidata.py` (`_visibility_*`), `tests/test_oidata*.py` | S (warn) / M | D9 |
| **WP-9** StarPipeline #580 | virgil | TS-3 | `pipeline/star.py`, `tests/test_pipeline_robust_star.py` | M–L | **yes**: D4 |
| **WP-10** Ensemble and imaging defaults | virgil | CO-10 (label), CO-11, CO-15 | `ensemble.py`, `pipeline/image.py`, `pipeline/binary.py` (flux prior and `error_scale` docs only), `tests/test_ensemble.py`, `tests/test_pipeline_image*.py` | S–M | D10 |
| **WP-11** Docs and claims | virgil | DC-1 to DC-11, DC-13, DC-15, OR-4 interim wording in docs, CO-10 doc side | `docs/comparison.md`, `docs/radio_interferometry.md`, `docs/conventions.md`, `docs/orbit_fitting.md` + `notebooks/orbit_fitting.ipynb` (markdown only), `docs/imaging_ensemble.md`, `docs/pipeline.md`, `README.md`, `references/provenance.toml`, `docs/references.md`, `changelog.d/` (new fragments), `src/virgil/models.py` (one docstring line), `src/virgil/tempering.py` (module docstring) | M | D6, D11, D13 for some lines |
| **WP-12** Test hygiene | virgil | TS-4, TS-5, TS-6 (slow seeds), TS-7, TS-1 (comment) | `tests/lint/test_ci_config.py` or a new `tests/test_ci_imports.py`, `tests/test_fit_precondition.py`, `tests/test_aliases.py` (docstrings and new slow test), `tests/test_night_flags.py` (comment only) | S | D7, D12 |
| **WP-13** Binary detection threshold | virgil | CO-7 | `pipeline/binary.py` (`detection_sigma`, `n_trials`), `tests/test_pipeline_binary*.py`; **after WP-10** (shared file) | M | **yes**: D8 |

**Parallel-safe now (no decision needed):** WP-5b, WP-7, WP-11 (lines needing no decision), WP-12, and WP-3 (except VV-4 wording).

**Order:** WP-1 → WP-2 on the vv registry. WP-5 → WP-6 on `_checks.py`. WP-10 → WP-13 on `binary.py`.

**Reviews:** every PR that changes a default or a check gets an Opus equivalence or adversarial review, per house rule.

## 7. Decisions for Ben

| ID | Decision | Findings |
|---|---|---|
| D1 | Which `orbit_gate_042.md` is authoritative (recommend vv, pinned), and whether (iii′) is accepted as a post-hoc change | VV-2 |
| D2 | CI that fails a PR re-pinning a criteria file without a post-hoc entry | VV-7 |
| D3 | Public disclosure of IM Lup gate (d) moving to the tapered image | VV-4 |
| D4 | StarPipeline default model set: fix #580 before 0.4.2, or drop limb darkening from the default | TS-3 |
| D5 | Interim wording for the no-orbit verdict until the gate's null choice runs | OR-5, OR-4 |
| D6 | Tempering: fail above which *f*; on by default or opt-in until 0.5 | OR-11, DC-7 |
| D7 | Is q ≈ 0.7 on a −105 log-L night acceptable for the default mixture; land continuation before 0.4.2? | TS-1, TS-2 |
| D8 | Pipeline check policy: N-aware χ² thresholds; which checks may fail; 3σ Šidák vs simulated null or 5σ for "detected" | CO-7, CO-8 |
| D9 | Treatment of sub-σ visibilities in `vis_mode` conversion | CO-5 |
| D10 | Keep contest-tuned ensemble weight ranges (labelled) or choose them per dataset; absolute `max_chi2_red` default | CO-10, CO-15 |
| D11 | Orbit tutorial: run a seed sweep and quote the recovery fraction? | DC-6 |
| D12 | float32 L-curve: bound or strict xfail | TS-5 |
| D13 | Gl 229 retirement wording; carbon badge wording | DC-5, DC-14 |
| D14 | `a_range` in resolution units (decision 12, pending) | OR-7 |

## 8. Not covered by this audit

- No test suite was run. Calibration-test status on main/nightly since #586 is unverified (TS notes).
- No AST scan for assert-free tests was done.
- Not every one of the ~392 seed sites in `tests/` was reviewed.
- vv `trust/graph.yml` was not read node by node, and the OiDB/ESO campaign outputs were not checked.
- Provenance coverage (DC-7) is by grep, not by script.
- virgil-vlti, harmonix and jaxbessel are out of scope.
