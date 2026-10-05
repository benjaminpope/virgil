# Design note: chromatic sources and SPARCO

Status: **partly implemented.** Multi-channel `OIData` (prerequisite 1),
the `Image` component (prerequisite 2), `Resolved` (prerequisite 3) and the
`PowerLaw` and `BlackBody` spectra (prerequisite 4, in `virgil.spectra`)
exist, and `Component`/`System` accept a spectrum as `flux`. The first
component with a chromatic *shape*, `GravityDarkenedStar`, is described
below. `Tabulated` (a free flux per channel) exists, provisionally: Stage 6a's
`Nodes` will replace it. Still to do: rendering at a given wavelength.

## Goal

SPARCO (Kluska et al. 2014, A&A 564, A80) models a chromatic scene as a
parametric star plus an image of its environment, each with its own spectral
law. Their Eq. 4 is

$$
V_\mathrm{tot}(\mathbf{b}/\lambda, \lambda) =
\frac{f_*^0 (\lambda/\lambda_0)^{-4} V_*(\mathbf{b}/\lambda)
      + (1 - f_*^0) (\lambda/\lambda_0)^{d_\mathrm{env}} V_\mathrm{env}(\mathbf{b}/\lambda)}
     {f_*^0 (\lambda/\lambda_0)^{-4} + (1 - f_*^0) (\lambda/\lambda_0)^{d_\mathrm{env}}},
$$

with the star in the Rayleigh–Jeans regime (index −4), the environment's
spectral index `d_env` fitted, `f_*^0` the stellar fraction of the total flux
at the reference wavelength `λ0`, and the environment image fitted pixel by
pixel. Later work (e.g. Hillen et al. 2016) adds a fully resolved background
with its own index.

This is exactly a `System` whose component weights depend on wavelength:
`V = Σ f_i(λ) V_i / Σ f_i(λ)`.

## What already exists

- `SourceModel._weight(wavel)` receives the wavelength (or `None` for the
  reference flux, e.g. when rendering), and `System.model` passes it
  through. Every current model ignores it. This was added so that chromatic
  fluxes do not require changing every subclass later.
- `System` already mixes components by weight, so no change to the mixing
  rule is needed.

## Proposed syntax

A component's `flux` is either a number (achromatic, as now) or a small
**spectrum** object, itself an equinox module, that `_weight(wavel)`
evaluates:

| Spectrum | Flux at `λ` | Parameters (paths) |
|---|---|---|
| `PowerLaw(ratio, index, wavel0)` | `ratio * (λ/λ0)**index` | `.flux.ratio`, `.flux.index` |
| `Blackbody(ratio, temperature, wavel0)` | `ratio * B_λ(T) / B_λ0(T)` | `.flux.ratio`, `.flux.temperature` |
| `Tabulated(ratio, wavel)` (provisional, private in 0.2.0; Stage 6a's `Nodes` will replace it) | `ratio` interpolated linearly between nodes at `wavel` (metres), constant beyond the end nodes | `.flux.ratio` |

`wavel0` is a reference wavelength (traceable, not normally fitted). With
`wavel=None`, `PowerLaw` and `BlackBody` return their value at `wavel0`, so
`render()` and the zero-baseline normalization keep working. `Tabulated` has
no `wavel0`: its reference flux is the mean of its nodes, a rule that Stage 6a
is to settle (see `spectro_interferometry_workflow.md` §2.1).

SPARCO then reads:

```python
System(
    star=UniformDisk(diam, flux=PowerLaw(1.0, index=-4.0, wavel0=1.65e-6)),
    env=Image(pixels, pixel_scale_mas, flux=PowerLaw(f_env, index=d_env, wavel0=1.65e-6)),
    bg=Resolved(flux=PowerLaw(f_bg, index=d_bg, wavel0=1.65e-6)),
)
```

- We keep the star at ratio 1 (our reference-flux convention). SPARCO's
  `f_*^0` parameterization is `f_env = (1 - f_*^0) / f_*^0`, which is a
  one-line reparameterization using a model-building function (the same
  mechanism the composition tutorial uses for separation and position angle).
- The star's index is fixed at −4 simply by not fitting `star.flux.index`.

## Relation to the hierarchical-inference tutorial

The hierarchical tutorial fits one binary position shared across filters with
one flux per filter, by threading a filter index through
`model_fn(params, index)`. With spectra, the scene is a single model evaluated
at each observation's own wavelength, so the index plumbing disappears:

- `comp=PointSource(dra, ddec, flux=Tabulated(fluxes, filter_wavels))`
  reproduces the current tutorial exactly;
- `flux=PowerLaw(...)` reduces three free fluxes to two parameters.

What stays hierarchical lives in the **priors**, not the model: smoothness or
GP priors on `Tabulated` `ratio` values, hyperpriors on spectral indices, and
per-epoch or per-instrument calibration terms, which remain the job of
`joint_loglike`/`model_fn`. So spectra replace the tutorial's plumbing and
complement its priors. Once spectra exist, rewrite the tutorial in two steps:
`Tabulated` (same result as now), then `PowerLaw`.

## Prerequisites before implementing

1. **Multi-channel `OIData`.** It currently stores `wavel` from `EFF_WAVE`
   but does not broadcast baselines against channels (for AMI it is a single
   wavelength; for CHARA it loads an array without matching it to
   baselines). SPARCO needs data on a baselines × channels grid, with the
   weights broadcasting over channels.
2. **`Image` component.** A pixel-grid component whose visibility is a DFT
   (or NUFFT) of its pixels, normalized to unit flux, with positivity and
   regularization handled as priors. This is the "semi-parametric" part.
3. **`Resolved` component.** Fully resolved flux: visibility 0 except at zero
   baseline, i.e. it contributes only to the normalization.
4. **Spectrum modules** as above, with `_weight(wavel)` implemented on
   `Component` by evaluating `self.flux` when it is a spectrum.

## Surface-resolved chromatic components

Spectra make a component's *weight* chromatic, but its *shape* stays grey:
`Component._centred_cvis(uu, vv)` never sees the wavelength. A
gravity-darkened star (`GravityDarkenedStar`, Espinosa Lara & Rieutord 2011,
ported from Shashank Dholakia's jax-interferometry) is the first component
whose shape depends on wavelength. Its hot pole and cool equator have
different spectra, so the pole-to-equator contrast rises towards short
wavelengths.

- **Grey mode (`t_pole=None`, the default).** This is Dholakia's model. Each
  surface triangle is weighted by its bolometric flux times its projected
  area, the same at every wavelength, and wavelength enters only through
  `u / wavel`.
- **Chromatic mode (`t_pole` in kelvin).**
  - Each triangle has `T = t_pole * Teff_ratio / Teff_ratio(pole)`.
  - Its intensity is the Planck function `B_λ(T)`.
  - The weights are `projected area * B_λ(T)`, with shape
    `(n_wavel_samples or 1, n_triangles)`.
  - `model(u, v, wavel)` is overridden to normalise each row separately, so
    every sample is evaluated at its own wavelength.
  - Memory is `n_samples × n_triangles`, about 200 MB of complex64 for 10⁴
    samples at the default `n_lat = 32`.
- **The star supplies its own spectrum to `System`.**
  - In chromatic mode, `_weight(wavel) = flux * SED(λ) / SED(wavel0)`, where
    `SED(λ) = Σ area * B_λ(T)` over the visible triangles.
  - A companion with `flux=BlackBody(...)` then gets physically consistent
    flux ratios at every wavelength, with no separate stellar spectrum.
  - This follows `BlackBody`'s convention: normalised to `flux` at `wavel0`,
    with `wavel=None` giving `flux`.
  - `flux` must therefore be a number, not a `Spectrum`, which would count
    the spectrum twice.
- **`render()`** shows the chromatic star at `wavel0`.
- **One Planck implementation.** `spectra._planck_ratio(wavel, temperature,
  wavel0, temperature0)` returns `B(λ, T) / B(λ0, T0)` without overflow.
  `BlackBody` calls it with `T0 = T`.
- **Not handled:**
  - limb darkening, grey or temperature-dependent;
  - bandwidth smearing, which is a scene-level wrapper (see below).

## Open questions

- Positivity: spectra must be non-negative at every wavelength. Checking the
  amplitude (`ratio`) is enough for `PowerLaw`/`BlackBody`, and `Tabulated`
  likewise needs a non-negative `ratio` at every node (its `is_physical`);
  `numpyro_model`'s flux-prior check covers `.flux.ratio` paths.
- Should `render()` take an optional `wavel` to show the scene at a given
  wavelength? Probably yes, once any component is chromatic.
- Bandwidth smearing across a wide filter is a scene-level operation (an
  integral over λ within a channel), not a component property; it should be
  a wrapper around a scene rather than part of the spectrum objects.
- Grid tools use a single flux path; for a chromatic companion this would be
  `"comp.flux.ratio"`. Its last part is not `flux`, so it would be passed as
  `flux_param=`, unless the automatic rule learns about spectrum objects.
