# API Reference

The everyday names are importable from the top level, e.g.
`from virgil import OIData, System, PointSource, likelihood_grid`.

- [OIData](oidata.md): observables and their conventions
- [OIFITS](oifits.md): reading and writing OIFITS files
- [AMIGO](amigo.md): AMIGO mixed-DISCO products
- [Models](models/index.md): source models and visibilities
- [Likelihood](likelihood.md): likelihoods and numpyro models
- [Gains](gains.md): calibration gains correlated across channels, marginalized analytically
- [Linear marginalization](linear.md): the shared algebra for parameters marginalized analytically
- [Inference](inference.md): Laplace and Fisher curvature
- [Grid Fit](grid_fit.md): grid searches
- [Limits](limits.md): contrast limits and flux/contrast/Δmag conversions
- [Priors](priors.md): isotropic inclination and latitude priors, and hierarchical (population) scales
- [Detection](detection.md): detection statistics for false-alarm rates and ROC curves
- [Spectra](spectra.md): wavelength-dependent fluxes
- [Fitting](fitting.md): `fit`, maximum a posteriori fits; `gauss_newton_mass`, a NUTS mass matrix from a fit
- [Angles](angles.md): `AngleVector`, angles sampled as 2-D vectors with no wrap boundary, and von Mises priors in least-squares form
- [Imaging](imaging.md): regularizers and helpers for image reconstruction
- [Scenes](scenes.md): synthetic truth images for testing reconstructions
- [Metrics](metrics.md): image-recovery scores for reconstructions
- [Ensemble](ensemble.md): randomized reconstruction ensembles, selected and averaged into a mean image with a per-pixel spread
- [Coverage](coverage.md): synthetic uv coverage and noise for simulations
- [Plotting](plotting.md): figures
- [Legacy](legacy.md): ImPlaneIA-derived OIFITS tools
