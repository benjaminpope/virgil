# Spectro-interferometric observables

Spectro-interferometers such as VLTI/GRAVITY and MATISSE record more than
squared visibilities and closure phases: a spectrum of the target (OI_FLUX),
visibility amplitudes or correlated fluxes (OI_VIS `VISAMP`), triple
amplitudes (OI_T3 `T3AMP`) and differential phases (OI_VIS `VISPHI`). virgil
reads them on request, as *extra observables* that follow the visibilities
and phases in the data vector. Every likelihood, fit and grid then uses them
through the same [`whitened_residuals`][virgil.likelihood.whitened_residuals].
The API is [`virgil.observables`](api/observables.md).

## Reading them

Nothing changes unless you ask: `read_oifits` and `OIData` read V² (or |V|)
and closure phases as before. Name the extras you want:

```python
from virgil.oidata import OIData

data = OIData("gravity.fits", extras=("flux", "visphi"))
# State the grey scale's prior, in the data's units (here Jy): required for
# "flux" and correlated fluxes, never taken from the data.
data = data.with_flux_scale(scale=(2.0, 2.0))
data = OIData(
    "gravity.fits", extras=("nflux", "t3amp", "visamp", "visphi")
)
[block.kind for block in data.extras]
```

| `extras` | Table and column | Model |
|---|---|---|
| `"flux"` | OI_FLUX `FLUXDATA` (or `FLUX`) | k Σ fᵢ(λ), k marginalised |
| `"nflux"` | the same, as a normalised spectrum | Σ fᵢ(λ) / continuum, a scale near 1 marginalised |
| `"visamp"` | OI_VIS `VISAMP`, `AMPTYP='absolute'` | \|V\| |
| `"visamp"` | OI_VIS `VISAMP`, `AMPTYP='correlated flux'` | k \|Σ fᵢ(λ) Vᵢ\|, k marginalised (kind `"corrflux"`) |
| `"t3amp"` | OI_T3 `T3AMP` | \|V_ab V_bc V_ac\| |
| `"visphi"` | OI_VIS `VISPHI`, any `PHITYP` | continuum-normalised arg V (below) |

Choose one of `"flux"` and `"nflux"`: they are two readings of the same
table. V² and |V| of one measurement are not independent, so read both only
when the pipeline measured them separately.

## Spectra and the grey scale

Visibilities only constrain flux *ratios*: multiplying every component's
spectrum by a common g(λ) leaves V unchanged. A measured spectrum breaks
that degeneracy, through
[`total_spectrum`][virgil.models.SourceModel.total_spectrum], the sum of
the components' fluxes. Its absolute level (calibration, fibre injection)
is unknown, so the model is k Σ fᵢ(λ), where k is a grey scale.

The scale is linear, so virgil does not fit it: it integrates it out
analytically, under a Gaussian prior that **you state**, in the data's
units, with `with_flux_scale(scale=(mean, sd))` (Luger, Foreman-Mackey &
Hogg 2017). The prior is never taken from the data, and for `"flux"` and
correlated fluxes the likelihood raises until it is given. Normalised
spectra (`"nflux"`) default to `(1, 0.1)`. This is the same low-rank
marginalisation as the Stage 6d gains (`virgil._linear`). The likelihood
keeps the log-determinant, which depends on the model's spectral shape.

The Gaussian is a *proposal*: k is a positive scale, whose Jeffreys prior is
1/k on stated bounds. For a well-measured spectrum the two differ by about
σ_k/k. For the Jeffreys posterior, reweight samples of k from its conditional
posterior by 1/(k N(k; mean, sd²)) within the bounds, or sample log k
directly. Make no evidence claims that depend on the Gaussian's width.
After a fit, report the scale's conditional posterior:

```python
from virgil.likelihood import flux_scale_posterior

post = flux_scale_posterior(best_model, data)["flux"]
post["scale"]  # data ≈ scale × total_spectrum, per group
post["mean"], post["cov"]  # the weights, in template units
```

Change the prior and the grouping with
[`with_flux_scale`][virgil.oidata.OIData.with_flux_scale]. Use
`per="row"` (one scale per spectrum) or `per="station"` (per telescope)
when the injection differs between telescopes and exposures, as it does
for uncalibrated GRAVITY spectra. `poly_order=1` also marginalises a slope
in λ times the spectrum, for a chromatic calibration error.

`"nflux"` divides the model spectrum by its continuum fit per row, over the
continuum ranges set with
[`with_continuum`][virgil.oidata.OIData.with_continuum]. The same helper,
[`continuum_operator`][virgil.observables.continuum_operator], normalises
the differential phases. A small scale, with a prior width of 10%, absorbs
the difference between the pipeline's continuum and ours.

**The degeneracy without OI_FLUX.** If you fit no spectrum, the reference
component's spectral shape is not measured. Every other component's
spectral index or temperature is then measured *relative to it*. Give the
reference its own `PowerLaw` or `BlackBody` with a prior from a model
atmosphere: that prior then sets the systematic error on every other
component's colour.

## Differential phases

A pipeline's differential phase is arg V with a fit removed over the
continuum channels, per baseline and frame: a mean and a slope in
wavenumber, that is, an offset and a delay. virgil applies the same
operator, N = I − L, to the **exact** model phase arg V(λ), unwrapped
along wavelength. Its covariance is N D Nᵀ. If our basis contains the
pipeline's, then N N_pipe = N, so re-normalising the data does no harm.

```python
data = data.with_continuum(
    continuum=[(2.150e-6, 2.162e-6), (2.170e-6, 2.180e-6)],
    lines=[(2.163e-6, 2.169e-6)],
    order=1,  # offset and delay; 0 removes the mean only
)
```

Either range defaults to the complement of the other. Without either, every
channel is used as both continuum and line, which is right for data
without closure phases.

**Do not use the photocentre formula for resolved systems.** The
approximation φ ≈ −2π **u**·Δ**p**(λ)/λ holds only while the line-emitting
structure is unresolved (|Δp| ≪ λ/B). For a binary much wider than λ/B with
a line in one star, it is wrong by much more than typical errors. virgil
always uses arg V (tested in `tests/test_observables.py`).

**Projection is marginalisation.** Removing a linear basis (offset and
delay) per baseline and frame is the flat-prior limit of marginalising those
nuisances. The likelihood of N d, with covariance N D Nᵀ, does not depend on
which projection with that null space is used (restricted maximum
likelihood). The finite-prior version is an option:

```python
data = data.with_continuum(continuum, lines=line, prior_width=(0.5, 2.0))
```

It marginalises each baseline and frame's offset (and slope, per unit of
wavenumber scaled to span 1 across the channels) under Gaussian priors of
these widths, in radians. Then every channel of both windows is kept. This
works whether or not the pipeline has already normalised the data, because
the pipeline's subtraction only shifts the offset and slope that are
marginalised. It reuses the low-rank whitening of the 6d gains. As the
widths grow, its χ² tends to the projection's when every channel is used
(tested). The projection remains the default, because it matches what the
pipeline did.

### Closure phases and differential phases together

When VISPHI and T3PHI come from one frame, the closure of the differential
phases is the continuum-normalised closure phase. Fitting both would count
it twice. The default (design note S §2.3) is:

* closure phases in every channel;
* in the line windows only (both windows with `prior_width`), the part of
  the differential phase that has no closure: per frame, the baseline phases
  are projected onto the
  telescope-differenced subspace φ_ab = a_a − a_b, which is orthogonal to
  every closure. That leaves N − 1 combinations per channel for N
  telescopes.

Per line channel there are then (N − 1)(N − 2)/2 closure phases plus N − 1
independent closure-free differential-phase combinations, not one per
baseline. The projection's propagated covariance is whitened as one dense
block per frame.

**What is approximated.** The cross-covariance between the projected
differential phases and the closure phases is neglected as an approximation.
It is exactly zero only when the baseline errors of one frame and channel
are equal; for unequal errors it is nonzero, and the independent-block
likelihood is not exact. The joint covariance is preferred when available.
The residuals are treated as linear (Gaussian), which holds while
differential phases stay well below π. Channels flagged on any baseline of
a frame are dropped from that frame.

## Error floors

[`with_error_floor`][virgil.oidata.OIData.with_error_floor] sets a minimum
uncertainty per observable, as PMOIRED's `min error` and `min relative
error` do:

```python
import numpy as np

data = data.with_error_floor(
    absolute={"phi": np.deg2rad(0.5), "visphi": np.deg2rad(0.3)},
    relative={"vis": 0.02, "flux": 0.01},
)
```

Each σ becomes max(σ, absolute, relative × |data|). This is a fixed change to
the data. The fitted counterpart is the `noise=` terms (`vis_scale`,
`vis_error_rel`, `vis_error`, `phi_error`, …), which add in quadrature
relative to the *model*. Both use one rule, `_utils.inflate_errors`, which
[`inflated_errors`][virgil.likelihood.inflated_errors] exposes with
`where=` and `combine=`. Closure phases from four or more telescopes are
floored before they are whitened. Their periodic penalty rows keep the
effective error 1/√(2π), so the floor never enters the normalisation
twice. The `noise=` terms act on V² and closure phases only; the extra
observables take fixed floors.

## Bias of low-S/N amplitudes

A visibility amplitude measured at low S/N is biased high. The noise adds
in quadrature, so E|V̂| ≈ √(|V|² + σ²). The bias is then about
σ / (2 S/N): a sixth of σ at S/N = 3, and 5% of σ at S/N = 10. virgil does
not debias VISAMP. The bias matters when many channels are averaged by the
fit, since it does not average down. For low-S/N data, either fit V² (which is unbiased once the pipeline has
subtracted the noise power), or debias the amplitudes before reading, as
eht-imaging's `debias` does: |V| → √max(|V|² − σ², 0).

## Not yet

These come with Stage 6a PR C: bandwidth smearing, the spectral-resolution
kernel, the primary beam (fibre coupling), differential visibility
amplitudes (`AMPTYP='differential'`), and fitted `noise=` terms for the
extra observables. The 6d wavelength scale (`wavel_scale`) does not yet
rescale the OI_FLUX wavelengths.
