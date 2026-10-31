# Orbit sampling coordinates (TI′, z, ζ)

Register entry (AGENTS.md, "Methods beyond published practice"). Code:
`src/virgil/_orbit_coordinates.py`. Part of the orbit redesign (Stage 2a); the
fuller design note is Stage D. Not used by `OrbitPipeline` yet.

## Claim

Sample a sky orbit as `(P, z, ζ)` at a fixed anchor epoch, with
`z ∈ ℝ⁴` and `ζ` in the unit disc. An isotropic orientation with
log-uniform `a ∈ [a_min, a_max]` is then `p(z) = |z|⁻⁴ / (π² ln(a_max/a_min))`,
and uniform `e` with uniform anchor mean anomaly `M_a` is uniform `ζ`. Both
are smooth where Campbell elements are not (face-on, `e = 0`), and have no
`cos i` bound or `Ω` fold. `z` covers each sky orbit 4:1 and `ζ` 2:1.

## Derivation

- Thiele–Innes `ddec = A X + F Y`, `dra = B X + G Y`. Rotating the `(X, Y)`
  basis by `M_a` gives TI′: `(A′, F′) = R(−M_a)(A, F)`, likewise `(B′, G′)`.
- `u = (A+G) + i(B−F) = a(1+cos i) e^{i(ω+Ω)}` and
  `V = (A−G) − i(B+F) = a(1−cos i) e^{i(ω−Ω)}`; TI′ multiplies both by `e^{iM_a}`.
  So `|u| + |V| = 2a`, and `|u|`, `|V|`, hence `a` and `i`, are unchanged by the
  rotation.
- `d⁴(u, V) = 4 d(ABFG)` (constant, from the linear map), and polar
  coordinates give `d(ABFG) = a³ sin²i da d(cos i) dω dΩ`.
- Isotropic orientation × `da/a` is `1/(a⁴ sin²i)` in `(A, B, F, G)`.
- `u = z_u²`, `V = z_v²`: `d⁴(u, V) = 16 |z_u|²|z_v|² d⁴z` with
  `|z_u|²|z_v|² = a² sin²i`, so `p(z) ∝ a⁻² ∝ |z|⁻⁴`. `a = |z|²/2`.
- `k + ih = e e^{iM_a} = ζ²`: `d²(k+ih) = 4e d²ζ`, so the uniform-`e`,
  uniform-`M_a` density `1/(2π)` per `de dM_a` is constant in `ζ`.
- Quaternion link: unit `(z_u, z_v)` is a unit quaternion, uniform on S³ for
  Haar measure on SO(3) (Shoemake 1992).

## Pinning tests (`tests/test_orbit_coordinates.py`)

Round trips; `|det d(ABFG)/d(a, cos i, ω, Ω)| = a³ sin²i`;
`|det d(ABFG)/dz| = 4|z_u|²|z_v|²`; the radial integral of the prior is 1;
KS recovery of `a`, `cos i`, `ω`, `Ω`, `e`, `M_a` from the priors; the covers give
identical sky orbits; positions match `KeplerOrbit`, including a closed-form
Campbell oracle that does not call the module; the direction of the anchor
rotation (continuity at `ζ → 0`); finite gradients at face-on and near circular.

## virgil-validation check

None yet: to-do before the pipeline uses these coordinates (an independent
prior-recovery check of the pipeline's sampled `(a, cos i, ω, Ω, e)` in vv).

## Status

*Derived*, experimental. Known limit: positions from `ζ` use
`M_a = arg(ζ²)` and the jaxoplanet Kepler solve, so the gradient is not finite
at `ζ = 0` exactly; a Kepler solve in `(k, h)` is part of the Stage 2b likelihood.
