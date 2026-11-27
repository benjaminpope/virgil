# `virgil.flat_nuts`

blackjax NUTS on one packed unconstrained vector, for small models given as
likelihood terms (an orbit fit to astrometry). Same posterior as
`MCMC(NUTS(numpyro_model(...)))`, with less per-step overhead; see
[Performance](../performance.md).

::: virgil.flat_nuts
    options:
      members:
        - flat_nuts
        - flat_nuts_sampler
        - packed_posterior
