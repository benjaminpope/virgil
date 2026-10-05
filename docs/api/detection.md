# `virgil.detection`

Detection statistics of a companion search over a grid: the profile
likelihood ratio Δχ², the grid-marginalised log Bayes factor and the best
flux SNR. They are traceable in the data, so they can be computed for many
simulated observations under `jax.lax.map` with one compilation, which is
what empirical false-alarm probabilities and ROC curves need.

::: virgil.detection
    options:
      members:
        - detection_statistics
        - local_nsigma
