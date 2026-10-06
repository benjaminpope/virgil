# One argument order for models and data

Status: **approved for 0.4, not built**, 2026-10-06. Surveyed against
`main` at c064b82 (0.3.0 plus the post-release review merges).

## Decision

Ben has approved the following, and this note does not reopen it:

- every public function that takes a model and data follows one
  argument-order rule;
- 0.4 ships the new order with a deprecation shim: the old positional
  order still works and raises a `FutureWarning`, and the shim is removed
  in 0.5;
- in the same pass `samples_dict` is renamed `grid`, and `samples_dict=`
  stays as a deprecated keyword alias until 0.5.

What this note settles is *which* rule, how the shim works, and the order
of the code PRs.

## Survey

Line numbers are for `main` at c064b82. "Model" means a `SourceModel`, a
model class or a callable returning one (all three are accepted by most
of these functions); "data" means an `OIData` or a sequence of them.

### Already model first

| Function | Signature | File |
| --- | --- | --- |
| `whitened_residuals` | `(model_object, data_obj, **noise)` | `likelihood.py:472` |
| `model_loglike` | `(model_object, data_obj, *, reject_unphysical, **noise)` | `likelihood.py:531` |
| `flux_scale_posterior` | `(model_object, data_obj)` | `likelihood.py:920` |
| `numpyro_model` | `(model, priors, data_obj, regularisers, noise, likelihoods, **options)` | `likelihood.py:708` |
| `posterior_predictive_summary` | `(samples, model, data_obj, params)` | `likelihood.py:880` |
| `fit` | `(model, priors, data, regularisers, *, noise, init, method, ...)` | `fitting.py:480` |
| `gauss_newton_mass` | `(model, priors, data, values, *, likelihoods)` | `fitting.py:741` |
| `log_evidence` | `(model, data, path)` | `imaging.py:1852` |
| `laplace_samples` | `(model, data, n, key, path, npix, fov_mas)` | `imaging.py:1930` |
| `error_scale` | `(model, data, path, *, by_observable)` | `imaging.py:2163` |
| `l_curve` | `(model, priors, data, regulariser, weights, others, *, warm_start, **fit_options)` | `imaging.py:2339` |
| `diagnose` | `(model, data, regularisers)` | `imaging.py:2503` |
| `simulate` | `(scene, template, key, noise_scale, shift_days)` | `simulate.py:26` |
| `bias_test` | `(scene, template, model, priors, n, key, **fit_kwargs)` | `simulate.py:72` |

### Data first

| Function | Signature | File |
| --- | --- | --- |
| `likelihood_grid` | `(data_obj, model, samples_dict, *, batch_size)` | `grid_fit.py:163` |
| `optimized_likelihood_grid` | `(data_obj, model, samples_dict, *, flux_param, batch_size)` | `grid_fit.py:247` |
| `optimized_flux_grid` | `(data_obj, model, samples_dict, *, flux_param, batch_size)` | `grid_fit.py:282` |
| `linear_flux_grid` | `(data_obj, model, samples_dict, *, flux_param, batch_size, n_iter, prior)` | `grid_fit.py:553` |
| `laplace_flux_uncertainty_grid` | `(data_obj, model, samples_dict, flux, *, flux_param, batch_size)` | `grid_fit.py:735` |
| `absil_limits` | `(data_obj, model, samples_dict, sigma, *, flux_param, flux_bounds, batch_size)` | `limits.py:307` |
| `injection_limits` | `(data_obj, model, samples_dict, sigma, *, flux_param, flux_bounds, batch_size)` | `limits.py:392` |
| `detection_statistics` | `(data, model, samples_dict, *, flux_param, batch_size)` | `detection.py:97` |
| `gaussian_null` | `(template, null_scene, *, error_scale)` | `detection.py:475` |
| `rescale_errors` | `(data, null_scene)` | `detection.py:521` |
| `bootstrap_null` | `(data, null_scene, *, method)` | `detection.py:582` |
| `injection_recovery` | `(template, null_scene, model, samples_dict, key, *, n_null, injections, ...)` | `detection.py:737` |
| `loglike` | `(values, params, data_obj, model, **options)` | `likelihood.py:634` |
| `laplace_cov` | `(values, params, data_obj, model, *, dtype)` | `inference.py:242` |
| `laplace_parameter_uncertainty` | `(values, params, data_obj, model, target_param)` | `inference.py:295` |
| `fisher` | `(values, params, data_obj, model, ridge, *, dtype)` | `inference.py:348` |
| `joint_prediction` | `(params, observations, model_fn)` | `likelihood.py:582` |
| `joint_loglike` | `(params, observations, model_fn, **options)` | `likelihood.py:610` |

### `samples_dict` without a data/model pair

| Function | Signature | File |
| --- | --- | --- |
| `best_grid_point` | `(loglike_grid, samples_dict)` | `grid_fit.py:833` |
| `plot_grid_map` | `(values, samples_dict, kind, *, units, sigma, ...)` | `plotting.py:809` |
| `plot_contrast_curve` | `(values, samples_dict, *, units, sigma, ...)` | `plotting.py:1024` |

The private helpers in `_grid.py` (`check_flux_axes`, `resolve_grid_keys`,
`meshgrid_vectors`, `coordinate_points`) and `detection._resolve_keys` also
use the name; they are renamed without a shim.

### Out of scope

- **Methods on the data**: `OIData.model(model_object)` (`oidata.py:1083`)
  and `OIData.with_model(model_object, key, noise_scale)`
  (`oidata.py:1582`). The receiver is the data, which is the normal shape
  of a method and reads as "the data, under this model".
- **Builders that take data and at most an optional model**: `clean(data,
  npix, pixel_scale_mas, base=None, ...)` (`imaging.py:1215`),
  `dirty_image`, `starting_image`, `beam`, `nyquist_pixel_scale`,
  `field_of_view`, and `LCurve.classic_maxent(self, data, path)`
  (`imaging.py:1666`). Their subject is the data; `clean`'s `base` is
  passed by keyword in every call site.
- **Arrays rather than models**: `inflated_errors(data_obj, prediction,
  ...)` (`likelihood.py:231`), `FluxSpectrum.whiten(prediction, data,
  errors)` and the other observable-block methods, which take predicted
  arrays, not models.
- **Regularizers** (`TSV.value(model)` and so on) take only a model.

### Call sites

Calls of the functions above (all calls, keyword or positional; nearly
every call in the notebooks is positional). The tutorial pages under
`docs/` are generated from the notebooks, so they mirror the notebook
counts; `docs/conventions.md` is the only hand-written page affected.

| Where | Model-first functions | Data-first functions |
| --- | ---: | ---: |
| `notebooks/*.ipynb` | 114 | 46 |
| `examples/*.py` | 2 | 8 |
| `tests/` | 331 | 178 |
| `src/virgil/` (internal calls and docstrings) | 45 | 43 |

`fit` alone is called 57 times in the notebooks. `samples_dict` appears in
no notebook, example or doc page (they already call the dictionary `grid`
and pass it positionally), 51 times in `tests/` and 138 times in
`src/virgil/`.

## The rule

**The model comes before the data.** In every public function that takes
both, the model argument (a `SourceModel`, a model class or a callable
building one, or the scene that generates simulated data) is placed
before the `OIData` it is compared with or simulated with. Other arguments
keep their roles around that pair:

- an argument that is the *subject* of the computation, the thing
  differentiated or summarized, may come first, as in `loglike(values,
  params, model, data)` and `posterior_predictive_summary(samples, model,
  data)`;
- what modifies the model (priors) may sit between model and data, as in
  `fit(model, priors, data)`;
- the search `grid`, then other required arguments, come after the data.

The new signatures:

| Function | 0.4 signature |
| --- | --- |
| grid family (5) | `likelihood_grid(model, data, grid, *, ...)`, and likewise `optimized_likelihood_grid`, `optimized_flux_grid`, `linear_flux_grid`, `laplace_flux_uncertainty_grid(model, data, grid, flux, *, ...)` |
| limits | `absil_limits(model, data, grid, sigma, *, ...)`, `injection_limits(model, data, grid, sigma, *, ...)` |
| detection | `detection_statistics(model, data, grid, *, ...)`, `gaussian_null(null_scene, template, *, ...)`, `rescale_errors(null_scene, data)`, `bootstrap_null(null_scene, data, *, ...)`, `injection_recovery(null_scene, template, model, grid, key, *, ...)` |
| values-first | `loglike(values, params, model, data, **options)`, `laplace_cov(values, params, model, data, *, ...)`, `laplace_parameter_uncertainty(values, params, model, data, target_param)`, `fisher(values, params, model, data, ridge, *, ...)` |
| joint | `joint_prediction(params, model_fn, observations)`, `joint_loglike(params, model_fn, observations, **options)` |
| rename only | `best_grid_point(loglike_grid, grid)`, `plot_grid_map(values, grid, kind, ...)`, `plot_contrast_curve(values, grid, ...)` |

`injection_recovery` follows `bias_test(scene, template, model, ...)`: the
null scene and template that generate the draws, then the model fitted to
them.

### Why model first

1. **It is already the majority.** The most-used entry points (`fit`,
   `numpyro_model`, `whitened_residuals`, `model_loglike`, the imaging
   diagnostics, `simulate`) are model first. Model-first calls outnumber
   data-first calls about 2.5 to 1 in the notebooks and 1.9 to 1 in the
   tests, so this rule changes the smaller set everywhere except
   `examples/` (8 calls).
2. **JAX differentiates the first argument.** `jax.grad`, `jax.hessian`
   and `eqx.filter_grad` default to `argnums=0`, and Equinox's idiom is
   `loss(model, x, y)`. virgil differentiates with respect to models or
   parameter values, never data, so with the model (or `values`) first
   `jax.grad(model_loglike)(model, data)` works with no `argnums`.
3. **The fitting side of the scientific Python ecosystem puts the model
   first**: `scipy.optimize.curve_fit(f, xdata, ydata)`, astropy's
   `fitter(model, x, y)`, `lmfit.minimize(fcn, params, args=...)`.
   Libraries that look data first, such as scikit-learn's
   `estimator.fit(X, y)`, `scipy.stats` distributions' `.fit(data)` and
   statsmodels' `OLS(endog, exog)`, attach the model as the receiver or
   the class, so the model still comes first in reading order. virgil's
   functions are free functions, and the closest analogue is `curve_fit`.
4. **It reads as the science**: "the likelihood of this model given these
   data", "search this model over that grid in those data".

### Rejected: data first

The case for `(data, model, ...)` is real. In a grid search or a detection
Monte Carlo the data are fixed while models vary, so the data are the
natural first argument for `functools.partial(f, data)`; and the grid,
limit and detection families, the newest code, already use it. It was
rejected because it would move the larger and more visible set of calls
(`fit` alone has 57 notebook calls), break the `argnums=0` default for
every gradient of a likelihood, and disagree with `curve_fit` and
astropy. Fixing data with `functools.partial(f, data=data)` still works
under the chosen rule, because the arguments stay positional-or-keyword.

Also considered: making `model` and `data` keyword-only everywhere. It is
the most explicit option but turns every call in the tutorials into
`fit(model=..., priors=..., data=...)`, which is verbose for the commonest
calls and is a bigger break than reordering.

## The deprecation shim

A decorator in a new private module, `src/virgil/_deprecate.py`, applied
to each function whose order changes. The function body is written for the
new order only; the decorator translates old calls.

### Telling data from model

Only the **data** are detected, because a model can be a `SourceModel`, a
model class, a `FitResult` or any callable, while data have one type:

```python
def _is_data(x):
    from .oidata import OIData  # lazy: keeps _deprecate import-free

    if isinstance(x, OIData):
        return True
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(isinstance(d, OIData) for d in x)
    )
```

A call is in the **old order** when the positional argument at the old
data position is data and the one at the new data position is not (or is
absent, as in `likelihood_grid(data, model=m, grid=g)`).

**Ambiguous calls pass through unchanged, with no warning.** If both
positions hold data, or neither does (an empty tuple, or a data-like
object that is not an `OIData`), the call is bound in the new order and the
function's own errors apply. New-order code therefore never warns, and an
old-order call with exotic data fails as it would have after 0.5.
Keyword calls (`model=`, `data_obj=`) are never ambiguous and are never
reordered.

### Sketch

```python
"""Deprecation helpers for the 0.4 argument-order change (private)."""

import functools
import inspect
import warnings


def old_order(*old_names, data):
    """Accept calls in the pre-0.4 positional order, with a FutureWarning.

    ``old_names`` are the leading positional parameters in their old
    order, under their new names; ``data`` names the data parameter.
    Remove in 0.5.
    """

    def decorate(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        new_names = [n for n in params if n in old_names]
        moved = [params[n] for n in old_names] + [
            p for n, p in params.items() if n not in old_names
        ]
        old_sig = sig.replace(parameters=moved)
        i_old, i_new = old_names.index(data), new_names.index(data)
        new_call = f"{fn.__name__}({', '.join(new_names)}, ...)"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _grid_alias(fn.__name__, kwargs)
            if (
                len(args) > i_old
                and _is_data(args[i_old])
                and not (len(args) > i_new and _is_data(args[i_new]))
            ):
                warnings.warn(
                    f"{fn.__name__} now takes the model before the data; "
                    f"call {new_call}. The old order stops working in "
                    "virgil 0.5.",
                    FutureWarning,
                    stacklevel=2,
                )
                bound = old_sig.bind(*args, **kwargs)
                return fn(**_as_keywords(bound, params))
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def _grid_alias(name, kwargs):
    if "samples_dict" in kwargs:
        if "grid" in kwargs:
            raise TypeError(
                f"{name} got both grid= and samples_dict=; use grid="
            )
        warnings.warn(
            f"{name}(samples_dict=...) is now {name}(grid=...); "
            "samples_dict= stops working in virgil 0.5.",
            FutureWarning,
            stacklevel=3,
        )
        kwargs["grid"] = kwargs.pop("samples_dict")


def _as_keywords(bound, params):
    """Flatten a binding to keywords, expanding any **kwargs."""
    out = {}
    for name, value in bound.arguments.items():
        if params[name].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[name] = value
    return out
```

Applied as, for example:

```python
@old_order("data", "model", "grid", data="data")
def likelihood_grid(model, data, grid, *, batch_size=None): ...

@old_order("values", "params", "data", "model", data="data")
def loglike(values, params, model, data, **options): ...

@old_order(
    "template", "null_scene", "model", "grid", "key", data="template"
)
def injection_recovery(null_scene, template, model, grid, key, *, ...): ...
```

The three rename-only functions (`best_grid_point`, `plot_grid_map`,
`plot_contrast_curve`) take a smaller decorator that calls only
`_grid_alias`. No reordered function takes `*args`, so the binding needs
no case for it. `functools.wraps` keeps the name, docstring and
`__wrapped__`, so `inspect.signature` and the API pages (griffe reads the
source) show the new signature.

`FutureWarning` rather than `DeprecationWarning`, because Python shows it
by default in notebooks and scripts, where these calls live; a
`DeprecationWarning` is hidden outside `__main__`.

### jit, vmap and grad

The check is plain Python, `isinstance` and `len` on the arguments'
Python types, and never looks at array values, so it is valid at trace
time and never becomes part of traced code:

- **No public function is jitted at definition** today, and the decorator
  goes outermost if one ever is (above `jax.jit` or `eqx.filter_jit`), so
  the jitted function only ever sees the new order and its
  `static_argnames` refer to new names.
- **A user's `jax.jit`, `vmap` or `grad` around a shimmed function** calls
  the wrapper while tracing. An `OIData` whose leaves are tracers is still
  an `OIData`, so detection works; the warning fires once per trace, not
  once per call of the compiled function, which is enough.
- **Positional `in_axes` and `argnums` stay correct.** JAX assigns them by
  position *before* the wrapper runs, so a user who wrote
  `jax.vmap(likelihood_grid, in_axes=(0, None, None))` for the old order
  still maps over the data after the wrapper swaps it into place, and
  `jax.grad(loglike)` still differentiates `values`, which does not move.
- **Inside virgil's own compiled kernels** (`injection_recovery` maps
  `detection_statistics` over draws under `lax.map`), internal calls are
  migrated to the new order in the same PR as the function, so the shim
  is a no-op there; a test that turns these warnings into errors catches
  any that are missed.

## Migration plan

### Code PRs, in order

Each PR migrates every internal call, docstring example and test for the
functions it touches, so `main` stays free of the warnings it introduces.

1. **`_deprecate.py`** with `old_order`, the `samples_dict` alias and
   unit tests on small dummy functions (no JAX models needed).
2. **Grids, limits and detection**: the five grid functions,
   `absil_limits`, `injection_limits`, `detection_statistics`,
   `gaussian_null`, `rescale_errors`, `bootstrap_null`,
   `injection_recovery`, `best_grid_point`, `plot_grid_map`,
   `plot_contrast_curve`, and the private `_grid` helpers; `samples_dict`
   becomes `grid` throughout `grid_fit`, `limits`, `detection`, `_grid` and
   `plotting`, including `DetectionMC`'s saved metadata (load files that
   carry the old key).
3. **Values-first and joint likelihoods**: `loglike`, `laplace_cov`,
   `laplace_parameter_uncertainty`, `fisher`, `joint_prediction`,
   `joint_loglike`, and their callers in `grid_fit` and `inference`.
4. **Notebooks, examples and docs**, then the pytest filter below.

### Tests

For each reordered function, on the smallest data that exercises it (a
few baselines from `coverage.nrm_oidata`, a 2 × 2 grid), so the suite
stays cheap:

- the old positional order raises one `FutureWarning` whose message names
  the new call, and returns the same result as the new order (exactly
  equal: the same code runs);
- the new order, positional and by keyword, raises no warning (run under
  `warnings.simplefilter("error")`);
- `samples_dict=` warns and equals `grid=`; passing both is a `TypeError`;
- an ambiguous call (neither argument is data) does not warn;
- under `jax.jit` (for `loglike` and `detection_statistics`) the old order
  warns while tracing and matches the new order, and a `jax.vmap` over the
  data with old-order `in_axes` matches.

PR 4 adds to `[tool.pytest.ini_options]`

```toml
filterwarnings = [
    "error:.*now takes the model before the data:FutureWarning",
    "error:.*samples_dict.*:FutureWarning",
]
```

so a missed internal or test call fails CI, with the deprecation tests
opting out with `pytest.warns`.

### CHANGELOG

Under `## 0.4.0`, `### Migration from 0.3.0`:

> - **Model before data.** Every function that takes a model and data
>   now takes the model first, as `fit` already did. Changed:
>   `likelihood_grid`, `optimized_likelihood_grid`, `optimized_flux_grid`,
>   `linear_flux_grid`, `laplace_flux_uncertainty_grid`, `absil_limits`,
>   `injection_limits` and `detection_statistics` are now `f(model, data,
>   grid, ...)`; `loglike`, `laplace_cov`, `laplace_parameter_uncertainty`
>   and `fisher` are `f(values, params, model, data, ...)`;
>   `joint_prediction` and `joint_loglike` are `f(params, model_fn,
>   observations)`; `gaussian_null`, `rescale_errors` and `bootstrap_null`
>   take the null scene first; `injection_recovery` is `(null_scene,
>   template, model, grid, key, ...)`. Calls in the 0.3 order still work,
>   with a `FutureWarning` naming the new call; they stop working in 0.5.
>   Keyword calls are unaffected.
> - **`samples_dict` is now `grid`** in every grid, limit, detection and
>   plotting function. `samples_dict=` still works with a `FutureWarning`
>   until 0.5.

### Notebooks and docs

Positional calls to change, found by grep: `notebooks/amigo_disco`,
`binary_search`, `composition`, `contrast_limits`, `detection_roc`,
`hierarchical_inference`, `model_syntax` and `orbit_fitting`; the three
scripts in `examples/`; and `docs/conventions.md`. Only code cells change;
the results are identical, so outputs stay valid without re-running.
Then `scripts/sync_tutorial_docs.py`, `pytest
tests/test_tutorial_docs_sync.py` and `mkdocs build --strict`. Downstream
repositories (virgil-vlti and the analysis repositories) migrate before
they move to 0.5; the warnings tell them where.

### Removal in 0.5

Delete `_deprecate.py`, the decorators and the deprecation tests, remove
the `samples_dict` alias (an unexpected keyword is then a `TypeError`),
drop the pytest filters, and add a `### Migration from 0.4.0` line saying
the old order and `samples_dict=` are gone.

## Open questions for Ben

1. **Parameter names.** The same functions call the data `data`,
   `data_obj` or `observations` and the model `model`, `model_object` or
   `model_fn`. The shim could also accept `data_obj=` and `model_object=`
   as deprecated aliases and standardize on `data` and `model` in 0.4,
   at little extra cost. Only `samples_dict` was approved.
2. **`injection_recovery`'s order.** This note follows `bias_test`
   (`null_scene, template, model, grid, key`). The alternative is the
   fitted model first (`model, null_scene, template, grid, key`), closer
   to `detection_statistics`.
3. **0.5 behaviour.** Removing the shim lets an old-order call fail deep
   inside (an `AttributeError` on the model). Keeping the cheap type check
   for 0.5 and raising a `TypeError` that names the new call would be
   friendlier; deleting it entirely is simpler.
