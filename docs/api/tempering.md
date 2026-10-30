# `virgil.tempering`

Tempering an evidence for correlated spectral channels: the effective number
of independent channels from the within-chain covariance of the residuals of
neighbouring channels, and the factor ``f = N / N_eff`` by which the log
likelihood of `fit_orbit_aliases` is divided.

**The problem.**
Spectral channels of one baseline (and of one closure triangle) are not
independent measurements: the same atmosphere, the same calibrator and the
same detector rows sit under all of them. A likelihood that treats `N`
channels as `N` independent points is therefore too sharp by the factor
`f = N / N_eff`. On real data (χ²/N ≈ 12 in virgil-validation) the
alias band with the best evidence got p = 1.000, and every posterior was
too narrow (sky-plane offset/spread 3.7 to 61).

The cure is a *tempered* likelihood, `L^(1/f)`: the log likelihood, and
with it every evidence difference and posterior curvature, is divided by
`f` (so the posterior is wider by √f). It is applied per observable
block, `dof_b = 1 / f_b`, through the `block_dof` argument of
`estimate_primary_diameter` and `marginal_loglike` and the traced
`dof` of the orbit engine; with `f = 1` nothing changes.

## The estimator

`f` comes from the residuals of the best static fit of
each night, not from χ²: the error scale already absorbs a global
underestimate of the errors (χ²/N = 12 alone says the errors are 3.5 times
too small, which the per-block scale integrates out, not that the channels
are correlated), whereas correlation across channels is what the scale
cannot absorb. The channels of one baseline (or triangle) and frame form a
*chain* of `n` consecutive wavelengths. `f` is the empirical
within-chain covariance factor, pooled over the chains of a block,


    f = Σ_chains (Σ_i z_i)² / Σ_chains Σ_i z_i²  =  Var(Σ_chain z) / Σ Var(z),

for whitened residuals `z` (zero mean). It is the ratio of the variance
of the chain's sum to the sum of the variances, which is what `N / N_eff`
means for a mean over the chain: exactly `f` for any stationary
correlation, with no model of its shape. It is exact for AR(1) chains
(where it is `1 + 2 Σ (1 - k/n) ρ^k`) and for common-mode noise (a
calibration error shared by all channels, correlation `c` at every lag:
`f = 1 + (n - 1) c`) alike; an AR(1) estimate from the lag-1 correlation
alone undercounts common-mode correlation (n = 8, c = 0.6: 3.2 for a truth
of 5.2). A single scale shared by all chains cancels; when pooling across
nights of different error scales, `fit_orbit_aliases` divides each night's
block residuals by its empirical scale $\sqrt{\chi^2_{k,b}/\nu_{k,b}}$ first
so a high-$\chi^2$ night does not dominate. It never exceeds `n`. White noise
has `f = 1` on average.

## Biased up on purpose

 `f` used is `max(f̂, 1)`: no threshold is
applied to `f̂`, so white noise gives a `f̂` scattering about 1 by
√(2 / chains) and the floor raises its mean slightly: the safe direction,
as the evidence is then a little cautious rather than overconfident. The
unfloored values are kept (`ChannelFactors.raw`).

## Diagnostics, not inputs

 The lag-1 and lag-2 correlations ρ1 and ρ2 of
each block are kept (`ChannelFactors.rho`, `.rho2`). An AR(1) chain
has ρ2 = ρ1²; a flat correlation (common mode, or a model error that is
smooth in wavelength) has ρ2 ≈ ρ1. `flat_signature` flags ρ2 well above
ρ1²: that is also the signature of misfit (a resolved primary fitted as a
point, a drifting companion), which tempering absorbs by widening the
posterior instead of repairing the model, so the result warns and sets
`AliasResult.tempering`.

## What tempering by 1/f is exact for

 Dividing the log likelihood by
`f` is exact for signals that are flat across the channels of a chain
(a position or a flux, which every channel measures alike), since the sum
of the chain then carries the information of `n / f` points. It is
conservative for chromatic templates (a signal that varies across the
chain is partly measured by the differences between channels, which
correlated noise affects less than it affects their sum).

Blocks with fewer than `MIN_CHAINS` chains of their own, without a
channel axis (extra observables) or whose residuals the layout cannot
reach (closure phases with `with_closure_offsets`) take the pooled `f`
of all chains (`ChannelFactors.fallback`). Data with a single wavelength
have no chain and `f = 1`: the tempering is then off, and a fixed `f`
is the way to temper them.

## Tempering never sharpens

 A significantly negative correlation
(differenced or over-fitted residuals) gives `f̂` below 1, which would
make every evidence gap and posterior *more* confident than the untempered
fit. The `f` used is at least 1; the raw value is kept so that the
result is still flagged (`AliasResult.tempering`) when `f` is above
`FLAG_RANGE[1]` or the raw value below `FLAG_RANGE[0]`.

::: virgil.tempering
    options:
      members:
        - estimate_factors
        - channel_chains
        - adjacent_channels
        - block_dof
        - flat_signature
        - out_of_range
        - ChannelFactors
        - FLAG_RANGE
        - MIN_CHAINS
