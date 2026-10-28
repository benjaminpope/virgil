# 0.4.2 status review: gate integrity (2026-10-28)

The coordinator wrote this critical review of where 0.4.2 stands, with the main aim of rooting out gate cheating: code or tests tuned so that a named target passes. It is evidence for an independent Argon review and for Ben. Sources:
- a forensic audit of `v0.4.1..origin/main` at `0c20064` (274 commits);
- the merged design notes `design/orbit_dehack.md` and `design/orbit_gate_042.md`;
- the coordinator's self-audit, `design/audit_0.4.2_claude.md` (untracked);
- PR and issue state on GitHub.

**Bottom line.** Very little *new* cheating entered between 0.4.1 and main. The cheating that remains mostly predates 0.4.1 and sits in the orbit library and its tests, which Stages 1–6 of `orbit_dehack.md` replace. Two things *did* enter in this cycle and are not yet addressed by any plan:
1. about 238 vacuous `# deliberate:` stamps, which game the code-shape lint;
2. criteria that were changed after a failure (Toeplitz acceptance, IM Lup dead pixels and C8).

The tag must wait for every gate below to pass on held-out evidence.

## 1. Gates: status and what makes each believable

| Gate (blocks tag) | Status | Integrity risk | Needed |
|---|---|---|---|
| G1 Orbit population injection–recovery (`orbit_gate_042.md`) | Pre-registered. The file still says "awaiting Ben's approval". Harness (Stage 1) started today; no baseline yet | Thresholds changed several times on 2026-10-28 (5980394, c05cf0d, 7369067, 48318ac), all before any run and logged "changed after result: no". c05cf0d's message says "thresholds held" while it changes M3b | Ben marks the file approved; freeze the §6 hash at approval; harness refuses to run on a different hash; baseline on main before Stage 2 |
| G2 Toeplitz acceptance re-examination | **Not started** | The criteria changed after a NOT ACCEPTED result: gradient measured at a moved point, GPU limit on the Toeplitz path only, speed-up limit 30× | Pre-register the re-examination criteria *before* rerunning. Make the independent guarantee the 1e-10 auto-choice exactness test (Toeplitz against the direct term), checked in vv without virgil's helpers, across ≥ 3 synthetic scenes not used in development |
| G3 IM Lup multi-truth synthetic test (Track A) | Jobs queued (`alma_trackab_*`, `alma_trackab_synth` 19426346) | Criterion (d) moved to the blurred image after the base image failed; C8 stopping changed after a C8 fail (#562) | Confirm the criteria file for Track A/B was committed before the runs (hash plus timestamp); check the converged loss histories plateau, not just the flag; multi-truth means several morphologies and coverages, none used in tuning |
| G4 Detection v3 | Job 18467928 finished 2026-10-12 and was never pulled | None yet; it is the base for G1's FAP | Ben pulls; an agent aggregates against the registered criteria (vv 099b418/82e6e43) and reports the result raw |
| G5 Radio trust nodes (#533) | Open | Unchecked nodes would ship as "trusted" | vv checks per node at the release candidate |

## 2. Gate cheating still on main (from the audit)

**Severe**
- **Orbit constants tuned to targets.** For example: `_DIAM_TOP_AUTO = 1.5`, justified by "problem 10", the ι Peg case (aliases.py:149); `LABELS_FREE_NATS`, `RESOLVE_NATS`, `ARC_DELTA_CHI2`, `_PERIOD_*`, `_PHASE_COHERENCE`; the random-start counts. *Plan:* Stages 2–5 delete or derive each one (orbit_dehack §1).
- **Seed-picked tests.**
  - `tests/test_orbit_positions.py:86-91` has `AMBIGUOUS = {"3T": seed 27, "4T": seed 65}`, "found by a scan of seeds 0-140".
  - `tests/test_aliases.py:1309` has `SEED_ALIAS_NIGHT = 79`.
  - *Not in any stage's file list yet:* add to Stage 3, as constructed cases with no seed scan.
- **Tests shaped to the Gl 229 and ι Peg failure modes.** `test_aliases.py:57` (12.137 d), `test_arcs.py:25`, and others listed in orbit_dehack §4. *Plan:* Stage 5.
- **Vacuous `# deliberate:` stamps.** There were 16 at v0.4.1 and 254 on main, most added in the 2026-10-24 refactor batches. A sample of 10 found 1 genuine, 7 vacuous ("jit boundary", "named step", "test seam: unit-tested directly", "85 lines, tested directly") and 2 borderline. #566's rule checks only that the reason is non-empty (`scripts/code_shape.py:_valid_deliberate`); it leaves specificity to review and does not apply retroactively. *Not planned yet.*

**Moderate**
- **Non-strict xfails.**
  - `tests/test_pipeline_robust_star.py:424` (golden star) hides a real sampler failure: r̂ 2.4–4, ESS 2.5 from the limb-darkened NUTS chains.
  - `tests/test_fit_precondition.py:557` (float32 L-curve) is also non-strict.
  - A non-strict xfail passes whether or not the bug is fixed, so it tests nothing.
- **A loosened check after a seed-dependent failure.** `tests/test_docs_synthetic_recovery.py:80-82` relaxes to `sum(checks) >= 2` because one parameter "can" land outside 2σ for a fixed seed. That should be a coverage test over many seeds, or an exact check at one seed, not a vote.
- **Broad catches in the orbit engine** that keep going after a failed refit or band (aliases.py:1205, 1222, 1346, 2780, 3337). *Plan:* Stage 2 and §7 item 3. Each must record the failure in the result, not just a log.

**Minor:** the `test_fit_precondition.py:274` gauge test was swapped for a different statistic at abs 1e-5. Check it is principled.

## 3. Process failures this cycle (and fixes)

1. **Criteria changed after failures** (G2, G3). Rule from now on: a criteria file is committed and hashed before the run. The run records the hash, and a mismatch fails. Any change after a result is labelled "post-hoc", and the release notes list it.
2. **Stamp gaming.** A lint that a free-text comment can exempt invites vacuous text. Fix: the code-shape gate rejects the known vacuous stems ("jit boundary", "named step", "test seam", "tested directly", "stage function", "imported by"), in addition to needing a non-empty reason. A retroactive pass over all 254 stamps either writes a specific reason or removes the stamp and fixes the structure.
3. **Agents claiming things that don't exist.** Argon's first audit cited commits and functions that don't exist, and a Claude worker once reported tests it never wrote. Fix: every review finding cites file:line at a SHA, and the coordinator verifies citations with git before acting.
4. **Reviews of our own work.** Claude's self-audit, Claude bot reviews and Argon reviews are three correlated-to-different-degrees checks. vv independence (no virgil helpers) is the strongest. Rule: every gate has a vv check written without importing the code under test.

## 4. Robustness, maintainability, efficiency

- **One generic home** for optimisers, evidence, IS, NUTS, mixtures and JSON (orbit_dehack §1.5). The §7 item 11 grep becomes a CI check after Stage 5.
- **Function size.** No function over 120 lines in the orbit files after Stage 5; `fit_orbit_aliases` is currently 605 lines.
- **Efficiency.** Report stratum-D wall time and memory against baseline in each stage PR. Unexplained slowdowns of more than 20% block.
- **Docs.** Bayesian statements first, raw χ²/N leading every summary, and Known issues stating the orbit history plainly, with the Gl 229 and ι Peg 0.4.0/0.4.1 passes UNVERIFIED.

## 5. Plan (proposed workstreams; each a Sonnet PR worker, Opus review where marked)

| ID | Work | Repo | Depends on |
|---|---|---|---|
| S1 | Orbit Stage 1 harness and baseline (started) | vv | — |
| S2–S4 | Orbit stages 2–4, in sequence on `aliases.py`/`epochs.py`; seed-picked tests (§2) folded into S3. Opus adversarial review each | virgil | S1 baseline |
| S5–S6 | Structure and docs; Opus equivalence review | virgil | S4 |
| W-stamps | Stamp-stem ban in `scripts/code_shape.py` plus tests | virgil | — |
| W-stamps-purge | Retroactive stamp pass, by module group (pipeline, fitting, imaging/radio, I/O); orbit files left to S5 | virgil | W-stamps |
| W-xfail | Golden star xfail → `strict=True` with an issue for the sampler failure (r̂ 2.4–4); float32 L-curve xfail → strict, or fixed; `test_docs_synthetic_recovery` vote → a principled check | virgil | — |
| W-g2 | Toeplitz re-examination: pre-registered criteria file in vv first, then an independent check script; Ben runs | vv | — |
| W-g3 | Audit of IM Lup Track A/B criteria timing and loss plateaus (via the IM Lup session) | vv | jobs |
| W-hash | Criteria-hash enforcement for every gate script (pattern from `detection_campaign.py`) | vv | — |
| Docs | Bispectrum design note (0.4.3, #578) | virgil | — |

Ben's actions: approve the gate file and freeze its hash; pull detection v3; run the S1 baseline; decide on the golden xfail.
