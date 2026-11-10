# API Reference

The everyday names are importable from the top level, e.g.
`from virgil import OIData, System, PointSource, likelihood_grid`.

Everything here is stable except the entries marked **Provisional**, which
may change in any 0.4.x release without a deprecation. See
[Stability](../stability.md) for the list.

## Data

- [OIData](oidata.md): observables and their conventions
- [OIFITS](oifits.md): reading and writing OIFITS files
- [AMIGO](amigo.md): AMIGO mixed-DISCO products
- [Observables](observables.md): OI_FLUX spectra, |V|, T3AMP and differential phases after the visibilities and phases
- [Gains](gains.md): calibration gains correlated across channels, marginalized analytically

## Models

- [Models](models/index.md): source models and visibilities
- [Harmonics](harmonics.md): azimuthal harmonics of a `MultipoleDisk` and conversions between their forms
- [Spectra](spectra.md): wavelength-dependent fluxes
- [Fields](fields.md): Gaussian-process log-brightness priors for pixel images

## Likelihood and fitting

- [Likelihood](likelihood.md): likelihoods and numpyro models
- [Likelihood terms](likelihood_terms.md): the attributes `fit` and `crossval` read from a term passed as `likelihoods`
- [Fitting](fitting.md): `fit`, maximum a posteriori fits; `gauss_newton_mass`, a NUTS mass matrix from a fit
- [SVI](svi.md): `variational`, a guide fitted to the posterior of `numpyro_model` with numpyro's SVI
- [Inference](inference.md): Laplace and Fisher curvature
- [Priors](priors.md): isotropic inclination and latitude priors, and hierarchical (population) scales
- [Angles](angles.md): `AngleVector`, angles sampled as 2-D vectors with no wrap boundary, and von Mises priors in least-squares form
- [Linear marginalization](linear.md): the shared algebra for parameters marginalized analytically

## Sampling and evidence

- [SMC](smc.md): adaptive tempered sequential Monte Carlo on BlackJAX, for evidences and posterior particles (provisional)
- [Mode search](mode_search.md): evidence and posterior draws from the modes of a multimodal target (provisional)
- [Cross-validation](crossval.md): K-fold cross-validation of a hyperparameter (provisional)

## Searches and limits

- [Grid Fit](grid_fit.md): grid searches
- [Limits](limits.md): contrast limits and flux/contrast/Δmag conversions
- [Detection](detection.md): detection statistics for false-alarm rates and ROC curves

## Imaging

- [Imaging](imaging.md): regularizers and helpers for image reconstruction
- [Ensemble](ensemble.md): randomized reconstruction ensembles, selected and averaged into a mean image with a per-pixel spread
- [Metrics](metrics.md): image-recovery scores for reconstructions
- [Scenes](scenes.md): synthetic truth images for testing reconstructions

## Radio

- [Radio](radio.md): `ComplexVisData`, `CalibrationNuisance`, complex visibilities from CASA measurement sets with per-observation calibration nuisances (provisional)
- [Radio CLEAN](radio_clean.md): `clean`, CLEAN deconvolution of complex visibilities (provisional)
- [Radio Toeplitz](radio_toeplitz.md): `ToeplitzVisData`, the radio χ² of a one-Image fit through precomputed dirty beams (provisional)

## Orbits

- [Orbits](orbits.md): Keplerian orbits in virgil's conventions, solved with jaxoplanet
- [Epochs](epochs.md): datasets grouped into epochs, and starting multi-epoch orbit fits (provisional)
- [Orbit Search](orbit_search.md): `score_orbits`, candidate orbits scored on all epochs with shared nuisances (provisional)
- [Orbit aliases](aliases.md): `fit_orbit_aliases`, evidences of the period alias bands of an undersampled orbit (provisional)
- [Short arcs](arcs.md): each night's velocity, the periods its positions and velocities admit, and each alias band's seed orbits (provisional)
- [Tempering](tempering.md): the effective-channel factor that tempers an orbit evidence for correlated spectral channels (provisional)
- [Night summaries](night_summaries.md): per-night, per-peak position, velocity and flux summaries at visibility level (provisional)
- [Orbit cells](orbit_cells.md): the position-level fit of the alias-resolved orbit, cell by cell (provisional)

## Pipelines and utilities

- [Pipeline](pipeline.md): `BinaryPipeline`, `StarPipeline`, `DiskPipeline`, `OrbitPipeline` and `ImagingPipeline`, scriptable pipelines that write a run folder
- [Simulate](simulate.md): a scene observed with a template's sampling, and bias tests
- [Coverage](coverage.md): synthetic uv coverage and noise for simulations
- [Plotting](plotting.md): figures
- [Legacy](legacy.md): ImPlaneIA-derived OIFITS tools
