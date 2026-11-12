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

The density of $v$ is

$$p(v) = \frac{e^{-(r-1)^2 / 2s^2}}{Z_s}\, p(\theta), \qquad Z_s = \int_0^\infty r\, e^{-(r-1)^2/2s^2}\,dr,$$

so $r$ and $\theta$ are independent and $\theta$ has exactly the density $p(\theta)$ (per radian): uniform, $1/2\pi$, by default; a von Mises, $\exp[\kappa(\cos(\theta - \mu) - 1)] / (2\pi\, \mathrm{i0e}(\kappa))$, given `mean` and `kappa`; or, with `axial=True`, the von Mises of $2\theta$, so that $\theta$ and $\theta + 180°$ are equally likely. The radius is a nuisance with a ring prior whose mode is the unit circle, which keeps a maximum a posteriori fit away from the origin, where $\theta$ is undefined. The ring width $s$ (`ring_width`, default 0.25) does not change the prior on $\theta$. The default, a uniform angle, is the invariant prior; a von Mises prior is strong information from an external measurement.

Every term has a least-squares form for Levenberg–Marquardt and [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass], and the density is normalized in $\mathbb{R}^2$, so evidences stay normalized. The ring's residual is $(r - 1)/s$. A von Mises enters as a chord, $\sqrt{\kappa}\,(\hat v - \hat m)$ with $\hat v = v / r$ and $\hat m = (\cos\mu, \sin\mu)$; half its square is $\kappa(1 - \cos(\theta - \mu))$, the von Mises exponent, just as virgil's phase residuals $2\sin(\Delta/2)/\sigma$ are chords with $\kappa = 1/\sigma^2$ (see [`whitened_residuals`][virgil.likelihood.whitened_residuals]). The axial chord is the same on the doubled direction, $(\cos 2\theta, \sin 2\theta) = (x^2 - y^2, 2xy)/r^2$.

For orbits, [`orientation_priors`](orbits.md) samples 2Ω and ϖ = Ω + ω for
position-only fits (so the node ambiguity is one point, not two modes), or Ω
and ϖ when RVs fix the node.

**Credit.** The construction follows Octofitter's `UniformCircular` (Thompson et al. 2023, AJ 166, 164) and exoplanet's `Angle` (Foreman-Mackey et al. 2021, JOSS 6, 3285), which sample $v \sim N(0, I)$ and so peak at the origin; the ring and chord priors are virgil's own. No code is taken from either.

::: virgil.angles
    options:
      members:
        - AngleVector
        - vector_angle
        - vector_site
        - is_angle_vector
