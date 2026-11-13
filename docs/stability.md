# Stability

virgil has two tiers.

- **Stable**: a change needs a deprecation cycle. A release keeps accepting the old form and warns before the new one replaces it.
- **Provisional**: new features still settling. The name, signature, defaults and output may change in any 0.4.x release without a deprecation.

Everything public not listed below is stable. Provisional pages and API entries carry this notice:

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Nothing warns at runtime. The tier lives in the documentation and in each module's `__provisional__` tuple; the tuples are authoritative for names, and argument-level rows are listed on this page. New public features start provisional and are promoted at a release.

## Provisional in 0.4

| Area | Provisional names |
| --- | --- |
| Orbit fitting | the modules [`virgil.epochs`](api/epochs.md), [`virgil.aliases`](api/aliases.md), [`virgil.arcs`](api/arcs.md), [`virgil.night_summaries`](api/night_summaries.md), [`virgil.orbit_cells`](api/orbit_cells.md), [`virgil.mode_search`](api/mode_search.md), [`virgil.orbit_search`](api/orbit_search.md) and [`virgil.tempering`](api/tempering.md) (the whole of each) |
| `MultipoleDisk` family ([`virgil.models`](api/models/classes/multipole_disk.md)) | `MultipoleDisk`, `ThinRing`, `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`, `RadialSum`, `RadialLinear`, `RadialPower`, `RadialGP`, `RadialGPFunction`, `MultipoleKernel`, `Harmonics`, `VonMises`, `Cone`, `PowerLawHeight`, `multipole_disk_cvis`, `gp_priors` |
| Pinwheel spiral ([`virgil.models`](api/models/classes/composition.md)) | `ArchimedeanSpiral` |
| Linear pixel images ([`virgil.models`](api/models/classes/image.md)) | `LinearImage` |
| Polar fields ([`virgil.fields`](api/fields.md)) | `MultipoleField`, `multipole_field_priors` |
| Ensembles' Gaussian-process members ([`virgil.ensemble`](api/ensemble.md)) | `evidence_grid`, `gp_members`, `GPFit`, `EvidenceGrid` |
| Tempered SMC | the module [`virgil.smc`](api/smc.md) (`tempered_smc`, `bridge_smc`, `SMCResult`) |
| Variational inference | the module [`virgil.svi`](api/svi.md) (`variational`, `VariationalResult`) |
| Pipelines | the whole of [`virgil.pipeline`](pipeline.md): `BinaryPipeline`, `StarPipeline`, `OrbitPipeline`, `Result`, `load` and the rest. Promoted to stable after a release of use |
| Sparse imaging ([`virgil.imaging`](api/imaging.md)) | `clean`, `CleanResult`, `DofDiscrepancy` (and `LCurve.dof_discrepancy`) |
| Fourier-transform choice | the `ft` argument of `Image`, `FlaredDisk*` and `image_visibilities`, `set_ft_defaults`, its automatic thresholds and its tolerances |
| Rendered disks and images on uv lattices | the `rotation_deg` argument of `FlaredDisk*` and its `"grid"` default; the opt-in `rotation_deg="grid"` of `Image`, `starting_image`, `clean`, `dirty_image` and `residual_image` (their defaults stay North up); the lattice tolerance for DISCO dictionaries |
| Diagnostic plots ([`virgil.plotting`](api/plotting.md)) | `plot_field_draws` |
| Preconditioned fits ([`virgil.fitting`](api/fitting.md)) | `gauss_newton_diag`, `kkt_report` and the `precondition` argument of `fit` |
| Night-outlier mixture ([`virgil.likelihood`](api/likelihood.md)) | `group_outlier_mixture`, `group_outlier_probabilities` |
| Radio interferometry | the module [`virgil.radio`](api/radio.md) (`ComplexVisData`, `CalibrationNuisance`, `primary_beam`, `dartboard_folds`; site names, default priors and the measurement-set convention) |
| Radio Toeplitz χ² | the module [`virgil.radio_toeplitz`](api/radio_toeplitz.md) (`ToeplitzVisData`), `ComplexVisData.toeplitz`, and the scalar-term protocol of `fit` (`chi2_and_log_norm`, `curvature_params`, `curvature_loss`) |
| Radio CLEAN | the module [`virgil.radio_clean`](api/radio_clean.md) (`clean`, `RadioCleanResult`, `fit_beam`, `restore`) and the `uvtaper` argument of `ComplexVisData.dirty_image` and `dirty_beam` |
| Image blurring | `Image(blur_fwhm_mas=...)`, `Image.smoothed()` and the `smoothed` option of the regularizers |
| Image-recovery scores | the module [`virgil.metrics`](api/metrics.md) |
| Cross-validation | the module [`virgil.crossval`](api/crossval.md) (`cv_task`, `cross_validate`, `aggregate`, `aggregate_files`, `fold_masks`, `save_record`, `load_records`) |

The user-guide pages about these features carry the same notice: [orbits](orbit_fitting.md), [orbit methods](orbit_methods.md), [`MultipoleDisk` rings](multipole_disk_rings.md), [CLEAN](imaging_clean.md), [ensembles](imaging_ensemble.md), [variational inference](variational_inference.md) and [pipelines](pipeline.md), [radio interferometry](radio_interferometry.md) (with its Toeplitz χ²), [radio CLEAN](radio_clean.md) and [cross-validation](cross_validation.md).

The module `__provisional__` tuples (`("*",)` means the whole module) are authoritative for names, checked by `tests/test_stability.py`; the argument-level rows above are documented only here.
