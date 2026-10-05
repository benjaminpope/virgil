# `virgil.angles`

Angles sampled as 2-D vectors, so that no prior has a wrap boundary. Put an
`AngleVector` in a `priors` dict under an angle's path (in degrees, such as a
binary's `"pa"` or an orbit's node), and `fit`, `gauss_newton_mass` and
`numpyro_model` sample a vector v = r(cos θ, sin θ) at the site
`"<path>_vec"` instead of the angle. The model receives θ in degrees, in
[0°, 360°): `numpyro_model` records it as the deterministic site `"<path>"`,
and `fit` reports both.

```python
from virgil.angles import AngleVector

priors = {"pa": AngleVector()}            # uniform, no wall at 0°/360°
priors = {"pa": AngleVector(350.0, 20.0)} # von Mises, mean 350°, κ = 20
priors = {"node": AngleVector(100.0, 4.0, axial=True)}  # known modulo 180°
```

The radius has a ring prior, ∝ exp(−(r − 1)²/2s²), whose mode is the unit
circle, so that maximum a posteriori fits stay away from the origin, where
θ is undefined. A von Mises prior enters as the chord √κ (v̂ − m̂), the same
form as virgil's unprojected phase residuals (2 sin(Δ/2)/σ, with κ = 1/σ²),
so every term has a least-squares form and Levenberg–Marquardt takes it. The
density is normalised in the plane (with the von Mises normaliser on the
circle, through `i0e`), so evidences stay normalised. The default, a uniform
angle, is the invariant prior; a von Mises prior is strong information from
an external measurement.

For orbits, [`orientation_priors`](orbits.md) samples 2Ω and ϖ = Ω + ω for
position-only fits (so the node ambiguity is one point, not two modes), or Ω
and ϖ when RVs fix the node.

**Credit.** The construction follows Octofitter's `UniformCircular`
(Thompson et al. 2023, AJ 166, 164) and exoplanet's `Angle`
(Foreman-Mackey et al. 2021, JOSS 6, 3285), which sample v ~ N(0, I). The
ring and the von Mises chords are virgil's. No code is taken from either.

::: virgil.angles
    options:
      members:
        - AngleVector
        - vector_angle
        - vector_site
        - is_angle_vector
