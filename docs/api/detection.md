# `virgil.detection`

Detection statistics of a companion search over a grid: the profile
likelihood ratio Δχ², the grid-marginalised log Bayes factor and the best
flux SNR. They are traceable in the data, so they can be computed for many
simulated observations under `jax.lax.map` with one compilation, which is
what empirical false-alarm probabilities and ROC curves need.

The Monte Carlo: `gaussian_null` and `bootstrap_null` simulate the null
hypothesis (Gaussian noise from the errors, or a residual bootstrap of the
data), `rescale_errors` first brings the null to χ²_r = 1,
`injection_grid` lays out companions to inject, and `injection_recovery`
runs the search on null and injected simulations with one compiled kernel.
Its `DetectionMC` result gives empirical false-alarm probabilities,
thresholds, ROC curves, completeness maps and contrast curves, and saves,
loads and concatenates for array jobs.

::: virgil.detection
    options:
      members:
        - detection_statistics
        - local_nsigma
        - gaussian_null
        - bootstrap_null
        - rescale_errors
        - injection_grid
        - injection_recovery
        - DetectionMC
