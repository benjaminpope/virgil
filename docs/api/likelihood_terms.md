# Likelihood terms

A likelihood term is any object passed in the `likelihoods` argument of
[`fit`](fitting.md), [`variational`](svi.md) and
[`numpyro_model`](likelihood.md), or returned by `make_term(mask)` in
[`cross_validate`](crossval.md). There is no base class: `fit` and
`crossval` test for attributes, and the term does only what its attributes
say. The built-in terms are
[`PositionData.term`][virgil.orbits.PositionData.term],
[`RVData.term`][virgil.orbits.RVData.term],
[`ComplexVisData.term`][virgil.radio.ComplexVisData.term] and
[`ToeplitzVisData`][virgil.radio_toeplitz.ToeplitzVisData].

`values` below is the dict of constrained parameter values, keyed like
`priors`.

## Required

| Attribute | Contract |
|---|---|
| `term(values)` | Whitened residuals, an array of any shape. The term's χ² is the sum of their squares, and its size is the size of the array. Not needed when the term has `chi2_and_log_norm`. |

## Optional

| Attribute | Read by | Effect |
|---|---|---|
| `has_log_norm`, `log_norm(values)` | `fit`, `crossval` | If `has_log_norm` is true, `log_norm(values)` is added to ½χ² (the Gaussian normalization, for a fitted jitter or a prior Jacobian). A term with one has no least-squares form: use `method="lbfgs"` or `"adam"`. |
| `residuals_and_log_norm(values)` | `fit` | `(residuals, log_norm)` in one evaluation; used instead of `term(values)` plus `log_norm` when the normalization needs the residuals (a profiled scale). |
| `chi2_and_log_norm(values)` | `fit` | A scalar-χ² term: `(χ², log_norm)` with no residual vector. Needs the scalar-χ² attributes below. It has no least-squares form either. |
| `chi2_constant`, `loss_constant` | `fit` | Constants left out of `chi2_and_log_norm` and of the loss; added back to the reported χ² and loss. Scalar-χ² terms only. |
| `curvature_params(values)`, `curvature_loss(params)` | `gauss_newton_mass` | The curvature of a scalar-χ² term, through a function of a few parameters. Scalar-χ² terms only. |
| `n_data` | `fit` | The number of data points, for the per-point loss scale; otherwise the size of `term(start)`. |
| `resolve(start, least_squares)` | `fit` | Returns `(term, form)`: the term `fit` should actually use at `start`, and a note of which form ran (a radio term may swap in its Toeplitz form). |
| `derived(values)` | `fit` | A dict of extra quantities (not parameters) added to `info["derived"]`. |
| `arrays` | `fit` | A dict with `"u"` and `"v"` (in wavelengths), so the default pixel scale and field of view see the term's baselines. |
| `loglike(values)` | `numpyro_model` | The term's own normalized log density, added with `numpyro.factor`. Without it, the factor is $-\tfrac12\sum r^2$ of `term(values)`. |

## Cross-validation

`cross_validate` calls only `term(values)` and, if present, `has_log_norm`
and `log_norm(values)`. Its `make_term(mask)` must return a term that uses
the points where `mask` is true and gives zero residuals elsewhere, without
changing array shapes. See [Cross-validation](../cross_validation.md#other-data).
