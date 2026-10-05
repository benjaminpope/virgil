# API Reference

The everyday names are importable from the top level, e.g.
`from virgil import OIData, System, PointSource, likelihood_grid`.

- [OIData](oidata.md): observables and their conventions
- [OIFITS](oifits.md): reading and writing OIFITS files
- [AMIGO](amigo.md): AMIGO mixed-DISCO products
- [Models](models/index.md): source models and visibilities
- [Likelihood](likelihood.md): likelihoods and numpyro models
- [Gains](gains.md): calibration gains correlated across channels, marginalised analytically
- [Inference](inference.md): Laplace and Fisher curvature
- [Grid Fit](grid_fit.md): grid searches
- [Limits](limits.md): contrast limits and flux/contrast/Δmag conversions
- [Detection](detection.md): detection statistics for false-alarm rates and ROC curves
- [Spectra](spectra.md): wavelength-dependent fluxes
- [Fitting](fitting.md): `fit`, maximum a posteriori fits; `gauss_newton_mass`, a NUTS mass matrix from a fit
- [Imaging](imaging.md): regularisers and helpers for image reconstruction
- [Scenes](scenes.md): synthetic truth images for testing reconstructions
- [Coverage](coverage.md): synthetic uv coverage and noise for simulations
- [Plotting](plotting.md): figures
- [Legacy](legacy.md): ImPlaneIA-derived OIFITS tools
