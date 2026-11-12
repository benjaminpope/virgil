# Source models in virgil: departures from published practice

The register that `AGENTS.md` ("Methods beyond published practice") asks for, for source models. The radio
imaging register is `radio_imaging_innovations.md`. Each item gives the claim, the references it rests on, the
tests that pin it, the virgil-validation (vv) check or "none yet", and a **Status**: *established*, *derived* or
*novel / experimental*. Novel or experimental items are opt-in, never defaults.

## M1. Exponential fade along an `ArchimedeanSpiral` arm

- **What.** `ArchimedeanSpiral(fade=f)` weights the arm's brightness per unit arc length by
  $\exp(-f\,\theta/2\pi)$: $f$ e-foldings lost per turn. `fade = 0`, the default, is a uniform arm.
- **Status: novel / experimental.** The geometry of the arm is established: an Archimedean spiral for dust
  carried out at constant speed from a rotating source (Tuthill, Monnier & Danchi 1999; Tuthill et al. 2008;
  Han et al. 2020, appendix C, for Apep). The brightness law is not. Tuthill et al. (2008) measure a profile
  along the arm of WR 104 that has no closed form, and no paper found gives an exponential in the winding
  angle. It is a one-parameter convenience, chosen because it is positive, monotonic and reaches the uniform
  arm continuously.
- **Why opt-in.** The default `fade = 0` keeps the established model; a fit that frees `fade` should say so.
- **Tests.** `tests/test_archimedean_spiral.py`: the visibility of an inclined, faded arm against an
  independent quadrature along the arm, the image against its discrete Fourier transform, and gradients with
  respect to `fade` against finite differences. These check that the code computes the stated law, not that
  the law describes a real pinwheel.
- **vv check.** None yet. An independent check of the spiral's Fourier transform has been requested in
  virgil-validation.
- **References.** In `references/provenance.toml` (checked on arXiv, not on ADS): Tuthill, Monnier & Danchi
  1999, Nature 398, 487 (arXiv:astro-ph/9904092); Tuthill et al. 2008, ApJ 675, 698 (arXiv:0712.2111); Han et
  al. 2020, MNRAS 498, 5604 (arXiv:2008.05834).
