# Spectro-interferometric observables

Spectro-interferometers such as VLTI/GRAVITY and MATISSE record more than
squared visibilities and closure phases: a spectrum of the target (OI_FLUX),
visibility amplitudes or correlated fluxes (OI_VIS `VISAMP`), triple
amplitudes (OI_T3 `T3AMP`) and differential phases (OI_VIS `VISPHI`). virgil
reads them on request as *extra observables*, which follow the visibilities
and phases in the data vector, so every likelihood, fit and grid uses them
through the same [`whitened_residuals`][virgil.likelihood.whitened_residuals].
The API is [`virgil.observables`](api/observables.md). The extra observables, including the bandwidth-smearing treatment, are not independently validated.

## Reading them

By default `read_oifits` and `OIData` read only V² (or |V|) and closure
phases; name the extras you want. A GRAVITY file holds fringe-tracker tables
(`GRAVITY_FT`) beside the science channel (`GRAVITY_SC`), measuring the same
baselines, so choose one with `insname=`. `OIData` accepts the record that
[`read_oifits`][virgil.oifits.read_oifits] returns as well as a path:

```python
from virgil.oidata import OIData
from virgil.oifits import read_oifits

record = read_oifits(
    "gravity.fits",
    insname="GRAVITY_SC",
    extras=("flux", "t3amp", "visphi"),
)
data = OIData(record)
# State the grey scale's prior, in the data's units (here Jy): required for
# "flux" and correlated fluxes, never taken from the data.
data = data.with_flux_scale(scale=(2.0, 2.0))
[block.kind for block in data.extras]
```

The example leaves out `"visamp"`: GRAVITY's `VIS2` and `VISAMP` come from the same frames, so reading both would count the visibility information twice.

| `extras` | Table and column | Model |
|---|---|---|
| `"flux"` | OI_FLUX `FLUXDATA` (or `FLUX`) | k Σ fᵢ(λ), k marginalized |
| `"nflux"` | the same, as a normalized spectrum | Σ fᵢ(λ) / continuum, a scale near 1 marginalized |
| `"visamp"` | OI_VIS `VISAMP`, `AMPTYP='absolute'` | \|V\| |
| `"visamp"` | OI_VIS `VISAMP`, `AMPTYP='correlated flux'` | k \|Σ fᵢ(λ) Vᵢ\|, k marginalized (kind `"corrflux"`) |
| `"t3amp"` | OI_T3 `T3AMP` | \|V_ab V_bc V_ac\| |
| `"visphi"` | OI_VIS `VISPHI`, any `PHITYP` | continuum-normalized arg V (below) |

Choose one of `"flux"` and `"nflux"`, two readings of the same table. Read
V² and |V| together only when the pipeline measured them separately.

## Choosing channels and observables

[`OIData.select`][virgil.oidata.OIData.select] restricts data to wavelength
windows and to some observables, so you never need to flag a `read_oifits`
record by hand. Wavelengths are in metres and ranges are inclusive:

```python
data.select(2.05e-6, 2.40e-6)                       # one window
data.select(ranges=[(2.05e-6, 2.10e-6), (2.20e-6, 2.40e-6)])
data.select(exclude=[(2.162e-6, 2.170e-6)])         # drop a line
data.select(2.05e-6, 2.18e-6, observables="phi")    # closure phases only
data.select(observables=("phi", "visphi"))          # no V², no spectrum
```

`observables` names what to keep: `"vis"`, `"phi"` and the `extras` kinds in
the table above; the rest are dropped at every wavelength. Closure triangles
are kept whole, since their legs share a wavelength, and calibration gains
([`with_gains`][virgil.oidata.OIData.with_gains]) go with the visibilities.
When the closure phases are dropped, the differential phases keep their
closure part (below), since it is then counted only once. Projected
observables (kernel phases, DISCOs) cannot be selected, and closure-phase
offsets must be added after selecting.

To use different windows for different observables, make them separate
datasets of one fit. V² and closure phases are independent, so nothing is
counted twice. Here a line window is left out of the V² while the closure
phases are used everywhere:

```python
import numpyro.distributions as dist

from virgil.fitting import fit
from virgil.models import PointSource, System
from virgil.spectra import GaussianLine, PowerLaw, Sum

# A star and a fainter companion with a Brγ emission line.
model = System(
    star=PointSource(),
    blob=PointSource(
        dra=0.5,
        ddec=0.2,
        flux=Sum(
            cont=PowerLaw(0.1, 0.0, wavel0=2.166e-6),
            brg=GaussianLine(0.5, line_wavel=2.166e-6, fwhm=3e-9),
        ),
    ),
)
priors = {
    "blob.dra": dist.Uniform(-5.0, 5.0),
    "blob.ddec": dist.Uniform(-5.0, 5.0),
    "blob.flux.cont.ratio": dist.LogUniform(1e-3, 1.0),
}

line = [(2.162e-6, 2.170e-6)]
datasets = [
    data.select(exclude=line, observables="vis"),
    data.select(observables="phi"),
]
result = fit(model, priors, datasets)
best_model = result.model
```

## Spectra and the grey scale

Visibilities constrain only flux *ratios*: multiplying every component's
spectrum by a common g(λ) leaves V unchanged. A measured spectrum breaks the
degeneracy through
[`total_spectrum`][virgil.models.SourceModel.total_spectrum], the sum of the
components' fluxes. Its absolute level (calibration, fibre injection) is
unknown, so the model is k Σ fᵢ(λ) with a grey scale k.

The scale is linear, so virgil integrates it out analytically rather than
fitting it, under a Gaussian prior that **you state** in the data's units
with `with_flux_scale(scale=(mean, sd))` (Luger, Foreman-Mackey & Hogg 2017).
The prior never comes from the data: for `"flux"` and correlated fluxes the
likelihood raises until you give one. Normalized spectra (`"nflux"`) default
to `(1, 0.1)`. This is the same low-rank marginalization, and code, as the
calibration gains of `OIData.with_gains`. The likelihood keeps the
log-determinant, which depends on the model's spectral shape.

The Gaussian is a *proposal*: k is a positive scale, whose Jeffreys prior is
1/k on stated bounds, and for a well-measured spectrum the two differ by about
σ_k/k. For the Jeffreys posterior, reweight samples of k from its conditional
posterior by 1/(k N(k; mean, sd²)) within the bounds, or sample log k
directly. Make no evidence claims that depend on the Gaussian's width: the
Jeffreys prior is improper, so an evidence needs finite positive bounds
$[k_{\min}, k_{\max}]$ and the density $1/(k\ln(k_{\max}/k_{\min}))$.
Marginalizing in log k is not linear and is not done here.
After a fit, report the scale's conditional posterior:

```python
from virgil.likelihood import flux_scale_posterior

post = flux_scale_posterior(best_model, data)["flux"]
post["scale"]  # data ≈ scale × total_spectrum, per group
post["mean"], post["cov"]  # the weights, in template units
```

Change the prior and grouping with
[`with_flux_scale`][virgil.oidata.OIData.with_flux_scale]. Use `per="row"`
(one scale per spectrum) or `per="station"` (per telescope) when injection
differs between telescopes and exposures, as for uncalibrated GRAVITY
spectra. `poly_order=1` also marginalizes a slope in λ times the spectrum,
for a chromatic calibration error.

`"nflux"` divides the model spectrum by its per-row continuum fit over the
ranges set with [`with_continuum`][virgil.oidata.OIData.with_continuum].
The same helper,
[`continuum_operator`][virgil.observables.continuum_operator], normalizes
the differential phases. A small scale with a 10% prior width absorbs the
difference between the pipeline's continuum and ours.

**The degeneracy without OI_FLUX.** If you fit no spectrum, the reference
component's spectral shape is unmeasured, and every other component's
spectral index or temperature is measured *relative to it*. Give the
reference its own `PowerLaw` or `BlackBody` with a prior from a model
atmosphere; that prior sets the systematic error on every other component's
colour.

## Differential phases

A pipeline's differential phase is arg V with a continuum fit removed per
baseline and frame: a mean and a slope in wavenumber, that is, an offset and
a delay. virgil applies the same operator, N = I − L, to the **exact** model
phase arg V(λ), unwrapped along wavelength, with covariance N D Nᵀ. If our
basis contains the pipeline's, then N N_pipe = N, so re-normalizing the data
does no harm.

```python
continuum = [(2.150e-6, 2.162e-6), (2.170e-6, 2.180e-6)]
# `line` is the window defined with the fit above.
data = data.with_continuum(
    continuum=continuum,
    lines=line,
    order=1,  # offset and delay; 0 removes the mean only
)
```

Either range defaults to the complement of the other. With neither, every
channel serves as both continuum and line, which is right for data without
closure phases.

**Do not use the photocentre formula for resolved systems.** The
approximation φ ≈ −2π **u**·Δ**p**(λ)/λ holds only while the line-emitting
structure is unresolved (|Δp| ≪ λ/B). For a binary much wider than λ/B with a
line in one star, it is wrong by far more than typical errors. virgil always
uses arg V (tested in `tests/test_observables.py`).

**Projection is marginalization.** Removing an offset and delay per baseline
and frame is the flat-prior limit of marginalizing those nuisances, and the
likelihood of N d, with covariance N D Nᵀ, does not depend on which
projection with that null space is used (restricted maximum likelihood). A
finite-prior version is an option:

```python
data = data.with_continuum(continuum, lines=line, prior_width=(0.5, 2.0))
```

It marginalizes each baseline and frame's offset (and slope, per unit of
wavenumber scaled to span 1 across the channels) under Gaussian priors of
these widths in radians, keeping every channel of both windows. It works
whether or not the pipeline has already normalized the data, since that
subtraction only shifts the marginalized offset and slope, and it reuses the
low-rank whitening of the 6d gains. As the widths grow, its χ² tends to the
projection's when every channel is used (tested). The projection stays the
default because it matches what the pipeline did.

### Closure phases and differential phases together

When VISPHI and T3PHI come from one frame, the closure of the differential
phases is the continuum-normalized closure phase, so fitting both would count
it twice. By default virgil uses:

* closure phases in every channel;
* in the line windows only (both windows with `prior_width`), the part of the
  differential phase with no closure: per frame, the baseline phases are
  projected onto the telescope-differenced subspace φ_ab = a_a − a_b, which is
  orthogonal to every closure, leaving N − 1 combinations per channel for N
  telescopes.

Each line channel then has (N − 1)(N − 2)/2 closure phases plus N − 1
closure-free differential-phase combinations, not one per baseline. The
projection's propagated covariance is whitened as one dense block per frame.

**What is approximated.** The cross-covariance between the projected
differential phases and the closure phases is neglected. It is exactly zero
only when a frame and channel's baseline errors are equal; otherwise the
independent-block likelihood is not exact, and the joint covariance is
preferable where available. The residuals are treated as linear (Gaussian),
which holds while differential phases stay well below π. A channel flagged on
any baseline of a frame is dropped from that frame.

## Error floors

[`with_error_floor`][virgil.oidata.OIData.with_error_floor] sets a minimum
uncertainty per observable, like PMOIRED's `min error` and `min relative
error`:

```python
import numpy as np

data = data.with_error_floor(
    absolute={"phi": np.deg2rad(0.5), "visphi": np.deg2rad(0.3)},
    relative={"vis": 0.02, "flux": 0.01},
)
```

Each σ becomes max(σ, absolute, relative × |data|), a fixed change to the
data. The fitted counterpart is the `noise=` terms (`vis_scale`,
`vis_error_rel`, `vis_error`, `phi_error`, …), which add in quadrature
relative to the *model*. Both use one rule, `_utils.inflate_errors`, exposed by
[`inflated_errors`][virgil.likelihood.inflated_errors] with `where=` and
`combine=`. Closure phases from four or more telescopes are floored before
whitening, and their periodic penalty rows keep the effective error 1/√(2π),
so the floor never enters the normalization twice. The `noise=` terms act on
V² and closure phases only; extra observables take fixed floors.

## Bias of low-S/N amplitudes

Visibility amplitudes measured at low S/N are biased high, because noise adds
in quadrature: E|V̂| ≈ √(|V|² + σ²). The bias is about σ / (2 S/N), a sixth
of σ at S/N = 3 and 5% of σ at S/N = 10. virgil does not debias VISAMP. The
bias matters when the fit averages many channels, since it does not average
down. For low-S/N data, either fit V² (unbiased once the pipeline has
subtracted the noise power), or debias the amplitudes before reading, as
eht-imaging's `debias` does: |V| → √max(|V|² − σ², 0).

## Bandwidth smearing

`OIData(path, bandwidth=True)` keeps the channel widths
(`OI_WAVELENGTH.EFF_BAND`) as `OIData.bandwidth`, and
`data.with_bandwidth(dlambda)` sets them for simulated data; by default they
are ignored. Models are then evaluated with each component's offset phasor
multiplied by the boxcar-channel envelope
`sinc(π Δλ (u dra + v ddec) / λ²)`, applied to a `System`'s components before
they are weighted and summed. An extended component is smeared as if it sat
at its centre offset. See `virgil.models.smeared_model`.

Supported models are `System`s of components (nested to any depth),
`BinaryModelCartesian` and `BinaryModelAngular` (smeared as their
`to_system()`), `Resolved` pieces (zero visibility, so nothing to smear) and
`Rotated` scenes of these. `Image` and other models raise a `TypeError`
naming them; an `OrbitalBinary` is smeared through its `at(mjd)` snapshot,
which `OIData` evaluates for each sample.

## Not yet

virgil does not yet model the spectral-resolution kernel, the primary beam
(fibre coupling), differential visibility amplitudes
(`AMPTYP='differential'`), or fitted `noise=` terms for the extra
observables. The wavelength scale (`wavel_scale`, from
`OIData.with_wavelength_scale`) does not yet rescale the OI_FLUX wavelengths.
