# `virgil.svi`

`variational` fits a guide (an approximate posterior) to the posterior of
[`numpyro_model`](likelihood.md) with numpyro's SVI, taking the same
arguments as [`fit`](fitting.md) and starting from a fit and its
Gauss–Newton covariance. See [Variational inference](../variational_inference.md).

::: virgil.svi
    options:
      members:
        - variational
        - VariationalResult
