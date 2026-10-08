# Period aliases of undersampled orbits

Status: implemented in `virgil.aliases` (`fit_orbit_aliases`).

## Problem

When epochs are separated by many periods, the number of orbital cycles
between them is ambiguous. The likelihood has one narrow ridge per cycle
count, and a fit started in the wrong ridge cannot find the right one. A
position per epoch is a lossy summary (at λ/B resolution, positions are
biased, errors are non-Gaussian, and several fringe peaks per night are
common), so evidences computed from positions are unreliable. Closure
phases of every epoch discriminate the ridges far better.

## Method

**Alias bands.** Let T be the first-to-last epoch span. Band N holds the
periods with `round(T/P) = N`: `T/(N+½) ≤ P ≤ T/(N−½)`, clipped to the prior
range. Bands are one cycle wide in frequency and tile the period prior.
The spectral window of the epoch times, `|⟨exp 2πi f t⟩|²`, shows which
frequency offsets the timing cannot tell apart; a cheap position-level
Thiele–Innes profile (`position_profile`: best χ² per band from
`period_grid` and `starting_orbits`) cross-checks that the bands are viable.
Positions are used for initialization and diagnostics only.

**Per-band fit.** The likelihood is the joint closure-phase (and
visibility) likelihood of all epochs, evaluated exposure by exposure.
It is hierarchical: each epoch's observable blocks have an error scale
`s_b` with a Jeffreys prior `1/s_b`, integrated out analytically
(`marginal_loglike`, `m = −Σ ν_b/2 ln χ²_b`), so `m` is unchanged by a
constant rescaling of an epoch's quoted errors and no epoch dominates
through an optimistic error bar. Candidate orbits come from the position
profile (both senses of motion, a flux scan) and are ranked by `m`;
the best distinct few are refined by L-BFGS in the flat coordinates of the
Jeffreys priors (log P, fractional time of periastron, e, cos i, ω, Ω,
log a, log flux). The (Ω+180, ω+180) mirror is exactly degenerate for
closure phases and is folded by restricting Ω to [0, 180). Fits are
deduplicated by their tracks at the epochs.

**Evidence.** Per mode, Laplace at the optimum (Hessian of `m` in the flat
coordinates; the priors are uniform there, so they add only a constant),
and an importance-sampling estimate from a Student-t proposal with the
Laplace covariance. The band evidence is the sum over its distinct modes.
Report ESS; flag the band when the two estimates differ by more than 1 nat
or the ESS is below 50 (then the IS estimate is not trustworthy either).
Modes with a non-positive-definite Hessian, or within 3σ of a hard prior
bound (e.g. e ≈ 0), are flagged. The band probability is
`p_N ∝ Z_N` under the log-uniform prior on P over the whole range.

**Posterior samples.** Importance resampling from the modes' proposals, in
the winning band and in every band with `p_N > 0.01`.

**Reporting.** Lead with the raw χ²/ν on the quoted errors (before any
error scale), then the fitted per-epoch scales ŝ, `log Z_N`, `p_N`, ESS.
A rescaled χ² near 1 is tautological, and large scales mean a failed fit.

## Limits

* The sum over modes misses modes the multistart does not find.
* The sum over modes uses a Gaussian local approximation: strongly
  non-Gaussian modes (very eccentric orbits) are what the IS check catches.
* The scale marginal assumes Gaussian small-σ closure-phase noise; for weak
  closure phases pass `s_max`.
* Errors correlated within an epoch carry fewer degrees of freedom than ν;
  see `dof` in `marginal_loglike`.

## Worked example

Gl 229 Ba–Bb, 7 GRAVITY epochs over 414 d, P ≈ 12.1 d: seven bands
N = 30–41. A prototype of this method fitting closure phases picked N = 34
(the published orbit) over N = 37 and 41 by more than 2000 in marginal
log likelihood, whereas evidences from single-night positions were
unstable (Laplace vs importance sampling differed by 2 nats, and
2–4 mas jitters were needed). Positions are a start, not the evidence.
