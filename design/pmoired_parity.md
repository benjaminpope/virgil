# Matching PMOIRED's features

PMOIRED (Mérand 2022, [arXiv:2207.11047](https://arxiv.org/abs/2207.11047); [GitHub](https://github.com/amerand/PMOIRED)) is the standard VLTI code for parametric spectro-interferometric modelling. This note compares its features with virgil's as of 2026-10-03 and assigns the gaps to stages.

virgil does not copy PMOIRED's interface.
- **Models are Python objects, not string expressions.** PMOIRED builds models from a string language such as `'$inner,fwhm'`. virgil models are equinox modules. Tied parameters come from a model function (as `fit` already accepts), and extra residuals come from regularisers.
- **Priors are numpyro distributions.** PMOIRED expresses priors as inequality penalties.

## Already in virgil, or better

| Feature | PMOIRED | virgil |
|---|---|---|
| Uniform disk, Gaussian, resolved background, offsets, fluxes | `ud`, `fwhm`, no size, `x`/`y`, `f` | `UniformDisk`, `GaussianDisk`, `Resolved`, `dra`/`ddec`, `flux` |
| Inclined ring with azimuthal harmonics | `diam`+`fwhmin`/`fwhmout`, `az ampN`, `incl`, `projang` | `ModulatedGaussianRim` (Gaussian radial profile only) |
| Power-law spectra | `spectrum` string in `$WL` | `PowerLaw` |
| Blackbody spectra | not built in | `BlackBody` |
| LM least squares and covariance | `scipy.optimize.leastsq` | `fit` (LM, L-BFGS, Adam); `inference.laplace_cov`, `fisher` |
| Grid and random-start searches | `gridFit` | `grid_fit` |
| Detection limits | `detectionLimit` (Absil/CANDID χ² ratio) | `limits.absil_limits`, `ruffio_upperlimit` |
| Simulated data | `oifake.makeFakeVLTI` | `coverage.vlti_oidata`, `nrm_oidata`, `OIData.with_model` |
| Error scaling | `mult error` | `OIData.with_error_scale`, `imaging.error_scale` |
| Pixel images | only fixed `sparse` point lists; no reconstruction | `Image`, regularisers, `GaussianField`, evidence |
| Sampling | none | numpyro NUTS through `numpyro_model` |
| Aperture masking, kernel phases, DISCOs | none | native |

## Gaps, by stage

### Stage 6a: spectro-interferometry (before joint AMI and the cube)
These are what GRAVITY and MATISSE line data need.
- **Lines and non-parametric spectra.**
  - New `Spectrum` types:
    - Gaussian and Lorentzian lines (`wl0`, FWHM, amplitude; emission or absorption);
    - node spectra (linear or cubic spline through free per-channel fluxes; this is PMOIRED's `flin_`/`fspl_`, and is also what joint multi-filter AMI needs);
    - sums, so that continuum plus lines is one spectrum.
- **Observables.**
  - Read and model T3AMP, OI_FLUX (FLUX and NFLUX), differential phase (DPHI), normalised |V| and V², and correlated flux.
  - Allow V² with |V|, and closure phase with VISPHI, in one dataset.
  - Normalisation uses continuum ranges, or the analytic continuum of the lines.
- **Instrumental effects.**
  - A spectral-resolution kernel (PMOIRED's `wl kernel`).
  - Bandwidth smearing by oversampling in wavelength. PMOIRED's own smearing is approximate.
- **Error handling.** Floors (`min error`, `min relative error`) and flags (`max error`), as `OIData` methods like `with_error_scale`.

### Stage 6b and 6c: as planned
- **6b:** joint multi-filter AMI, using 6a's node spectra.
- **6c:** `ImageCube`. PMOIRED cannot make a source's geometry depend on wavelength; it uses several components with different spectra instead. So the cube goes beyond it, and is not needed to match it.

### Stage 8: parametric parity (after milestone 2)
- **Projection for any component.** A `Projected(source, inc, pa)` wrapper that stretches uv, like `Rotated`. It gives elliptical Gaussians and uniform disks, and replaces per-class `inc`/`pa`.
- **Radial profiles by Hankel transform.**
  - A `RadialProfile` component with a callable I(r) and inner and outer radii, transformed by a fixed-quadrature Hankel transform: order 0, plus order n for azimuthal harmonics.
  - It covers thick rings, power-law and limb-darkened disks (I(μ)), doughnuts, and harmonics on any profile.
  - Crescents (`crin`, `crout`, `croff`) are the difference of two offset disks, so they need no new class.
- **Bootstrap.** `bootstrap_fit(model, priors, data, n)` resamples by date and baseline, keeping spectral vectors whole.
- **Orbits** are now Stage 6a.1, designed in [`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md).
- **Spectral correlations** between channels are now Stage 6d, because GRAVITY data need them ([`spectro_interferometry_workflow.md`](spectro_interferometry_workflow.md) §2.4).

## Not planned
- **The string expression language.** Python functions do the same job.
- **GRAVITY-specific tools** (telluric correction, fibre losses, polarisation averaging, pipeline parameters): no longer "not planned". They move to the separate instrument project, [virgil-vlti](https://github.com/benjaminpope/virgil-vlti) (decided 2026-10-03; see [`imaging_plan.md`](imaging_plan.md)).
- **Physical templates:** the Keplerian-disk and rotating/pulsating-star models. `HarmonixModel` already wraps rotating stars, and these can be added on demand.
- **Other utilities:** microlensing, SATLAS tables, JSDC calibrator diameters, `slant`, `spatial kernel`, and barycentric velocity.
