# Comparison with other packages

Which features each package supports natively, checked on 12 October 2026 against the code and documentation of virgil (commit `0771bcc`), PMOIRED (commit `b5a762e`, version 26.10.5), CANDID (commit `c255e90`, version 1.1.0), eht-imaging (commit `7acf089`, version 1.3.2), MPoL (commit `8850b48`, version 0.3.1) and fouriever (commit `0aaffe0`, version 0.4.15). It is a snapshot, and the packages have different aims: virgil overlaps a lot with some and little with others. If you spot an error, please open an issue.

| Feature | virgil | [PMOIRED](https://github.com/amerand/PMOIRED) | [CANDID](https://github.com/amerand/CANDID) | [eht-imaging](https://github.com/achael/eht-imaging) | [MPoL](https://github.com/MPoL-dev/MPoL) | [fouriever](https://github.com/kammerje/fouriever) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Data** | | | | | | |
| OIFITS input | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } |
| JWST AMI / kernel phase | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } |
| VLTI spectro-interferometry (GRAVITY, PIONIER, MATISSE) | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } |
| Radio visibilities (ALMA, VLBI) | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } |
| Correlated closure-phase covariance | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } |
| **Companions** | | | | | | |
| Binary grid search | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } |
| Contrast / detection limits | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } |
| Bayesian upper limits | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| Orbit fitting | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| **Sources** | | | | | | |
| Geometric models (uniform disk, Gaussian, ring, disk) | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-help-circle:{ .maybe } | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-help-circle:{ .maybe } |
| Limb darkening | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| Gravity-darkened or spotted stars | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| Chromatic SPARCO-style scenes | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| Composing models | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-help-circle:{ .maybe } | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-help-circle:{ .maybe } |
| **Imaging** | | | | | | |
| RML / MaxEnt | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } |
| Gaussian-process image priors | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } |
| CLEAN | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } |
| Posterior sampling of images | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } |
| **Inference and engine** | | | | | | |
| MCMC / nested sampling | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-check-circle:{ .yes } |
| HMC / NUTS | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } |
| Variational inference | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } |
| Automatic differentiation | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } |
| GPU | :material-check-circle:{ .yes } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-close-circle:{ .no } | :material-check-circle:{ .yes } | :material-close-circle:{ .no } |
| Hierarchical multi-filter fits | :material-check-circle:{ .yes } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } | :material-close-circle:{ .no } | :material-help-circle:{ .maybe } |

:material-check-circle:{ .yes } supported natively  :material-help-circle:{ .maybe } partial, experimental, or through an add-on or another package  :material-close-circle:{ .no } not supported

## Notes

**virgil**

- Radio visibilities: complex visibilities are fitted; measurement sets are read with `ComplexVisData.read_ms` and there is no UVFITS reader.
- Provisional: the CLEAN, orbit-fitting and variational-inference rows describe features marked provisional in virgil, without independent validation of every path (see [Stability](stability.md)). The table is a snapshot at the commit named above, not at the release.

**PMOIRED**

- Closure-phase covariance: a correlation parameter and bootstrapping, not a full covariance matrix.
- Orbit fitting: Keplerian elements can drive component positions; no orbit priors or dedicated fitter.
- Limb darkening: only in the rotating-star model, not the uniform disk.
- Gravity-darkened stars: von Zeipel model is included; spots are not.
- Chromatic scenes: components have independent spectra, so a SPARCO-like scene can be built.
- Hierarchical fits: one model fitted across datasets, without population-level structure.

**CANDID**

- VLTI: PIONIER is the demonstration case, GRAVITY is mentioned, one instrument setup per file.
- Geometric models: uniform or limb-darkened primary, point or uniform companion, one Gaussian.
- Limb darkening: a single-parameter law in the slower code path.
- Composing models: fixed primary plus companion plus resolved flux, no general interface.

**eht-imaging**

- Hierarchical fits: multifrequency imaging fits frequencies jointly with spectral index terms.

**MPoL**

- Geometric models: pixel images are the core; disk models appear only in a tutorial.
- Composing models: PyTorch layers compose, not interferometric source components.
- Gaussian-process priors: a preliminary power-spectral-density loss.
- Image posterior sampling, HMC / NUTS, variational inference: only through Pyro in a tutorial.

**fouriever**

- VLTI: PIONIER and GRAVITY branches exist; MATISSE does not.
- Geometric models: uniform disk plus unresolved companion only.
- Composing models: a fixed composite, plus subtracting or injecting a companion.
- Hierarchical fits: several datasets can be opened; joint fitting was not verified.
