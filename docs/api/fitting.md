# `virgil.fitting`

`fit` finds maximum a posteriori parameters, taking the same arguments as
[`numpyro_model`](likelihood.md), plus optional regularizers. `gauss_newton_mass` turns a fit into a dense
mass matrix for numpyro's NUTS, for sampling large images.
Angles with an [`AngleVector`](angles.md) prior are fitted as 2-D vectors,
with no wrap boundary at 0°/360°.

::: virgil.fitting
    options:
      members:
        - fit
        - FitResult
        - gauss_newton_mass
