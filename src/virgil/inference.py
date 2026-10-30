"""Local curvature tools: Hessians, Laplace covariances and Fisher matrices.

The *observed information* is the Hessian of the negative log likelihood at a
point ([`observed_information`][virgil.inference.observed_information] for a
model and data, [`hessian_matrix`][virgil.inference.hessian_matrix] for a
flat objective). The *expected Fisher* matrix is ``Jᵀ Σ⁻¹ J`` for a
prediction with fixed Gaussian errors
([`expected_fisher`][virgil.inference.expected_fisher]); the two agree at the
maximum of a linear-Gaussian problem. The Laplace covariance is the inverse
of the observed information:
[`laplace_cov`][virgil.inference.laplace_cov] for a model and data, and
[`objective_laplace_covariance`][virgil.inference.objective_laplace_covariance]
for a flat objective.

Objectives are negative log likelihoods of a flat 1D parameter vector,
except for ``expected_fisher``, which accepts any parameter pytree. Checks
that need concrete values are skipped inside ``jax.jit``.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.flatten_util import ravel_pytree

from ._deprecate import deprecated_name, external_stacklevel, old_order
from ._precision import cast_tree, run_in
from ._utils import concrete as _concrete
from .likelihood import loglike
from .models import SourceModel

__all__ = [
    "expected_fisher",
    "fisher_projection",
    "hessian_matrix",
    "laplace_cov",
    "laplace_parameter_uncertainty",
    "objective_laplace_covariance",
    "observed_information",
    "regularized_inverse",
]


def _add_ridge(matrix, ridge):
    """``matrix + max(ridge, 0) * I``."""
    ident = np.eye(matrix.shape[-1], dtype=matrix.dtype)
    return matrix + np.maximum(ridge, 0.0) * ident


def hessian_matrix(objective, x):
    """Return the Hessian matrix of ``objective`` evaluated at ``x``.

    ``x`` must be a flat 1D parameter vector; use
    ``jax.flatten_util.ravel_pytree`` to flatten a pytree first.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"hessian_matrix expects a 1D parameter vector; got shape "
            f"{x.shape}."
        )
    return np.asarray(jax.hessian(objective)(x), dtype=float)


def regularized_inverse(matrix, ridge=1e-10):
    """Return ``inv(matrix + ridge * s * I)``, ``s`` the largest diagonal entry.

    The ridge is relative, so it does not depend on the units of the
    parameters (``s`` is 1 if no diagonal entry is positive). A parameter the
    data do not constrain has curvature below ``ridge * s``: its variance is
    then set by the ridge, not the data. A ``RuntimeWarning`` is raised
    (outside ``jax.jit``) when that happens, and when the regularized matrix is
    not positive definite, since its inverse is then not a covariance.
    """
    matrix = np.asarray(matrix, dtype=float)
    top = np.max(np.diagonal(matrix))
    scale = np.where(top > 0.0, top, 1.0)
    regularized = _add_ridge(matrix, ridge * scale)
    value = _concrete(regularized)
    raw, floor = _concrete(matrix), _concrete(ridge * scale)
    if (
        value is not None
        and raw is not None
        and onp.all(onp.isfinite(raw))
        and onp.min(onp.linalg.eigvalsh(raw)) <= floor
        and onp.min(onp.linalg.eigvalsh(value)) > 0.0
    ):
        warnings.warn(
            "A parameter is nearly unconstrained: the Hessian's smallest "
            "eigenvalue is below the relative ridge, so its variance is set "
            "by the ridge, not the data.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    if (
        value is not None
        and onp.all(onp.isfinite(value))
        and onp.min(onp.linalg.eigvalsh(value)) <= 0.0
    ):
        warnings.warn(
            "The matrix to invert is not positive definite (the point is not a "
            "likelihood maximum, or a parameter is unconstrained); the "
            "result has negative or infinite variances.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    return np.linalg.inv(regularized)


def objective_laplace_covariance(objective, x, ridge=1e-10):
    """Return the Laplace covariance, the inverse Hessian of ``objective``.

    For a model and data use [`laplace_cov`][virgil.inference.laplace_cov].
    ``objective`` is a negative log likelihood of the flat vector ``x``,
    which should be at (or near) its minimum. A small relative ``ridge``
    (default 1e-10 of the largest diagonal entry) is added to the diagonal
    before inverting; see
    [`regularized_inverse`][virgil.inference.regularized_inverse].
    """
    hess = hessian_matrix(objective, x)
    return regularized_inverse(hess, ridge=ridge)


def expected_fisher(prediction_fn, params, errors, ridge=0.0):
    """Return expected Fisher information for fixed independent Gaussian errors.

    Not the Hessian of a likelihood: for that, see
    [`observed_information`][virgil.inference.observed_information].

    Parameters
    ----------
    prediction_fn : callable
        Function mapping the parameter pytree to a one-dimensional prediction.
    params : pytree
        Parameter values at which to evaluate the local model sensitivity.
    errors : array-like
        Standard deviations corresponding to the prediction vector.
    ridge : float, optional
        Diagonal regularization term.

    Returns
    -------
    tuple[array-like, callable]
        Expected Fisher matrix ``Jᵀ Σ⁻¹ J`` and a function restoring a flat
        parameter vector to the structure of ``params``.
    """
    flat_params, unravel = ravel_pytree(params)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    concrete_errors = _concrete(errors)
    if concrete_errors is not None and (concrete_errors <= 0.0).any():
        raise ValueError("Gaussian errors must be strictly positive.")

    def flat_prediction(values):
        return np.asarray(prediction_fn(unravel(values)), dtype=float).reshape(
            -1
        )

    # Forward mode: one pass per parameter, and there are usually far fewer
    # parameters than data points.
    jacobian = jax.jacfwd(flat_prediction)(flat_params)
    if jacobian.shape[0] != errors.size:
        raise ValueError(
            "Prediction and error vectors must have the same length; "
            f"got {jacobian.shape[0]} and {errors.size}."
        )

    weighted_jacobian = jacobian / errors[:, None]
    information = weighted_jacobian.T @ weighted_jacobian
    return _add_ridge(information, ridge), unravel


def fisher_projection(fmat, eps=1e-12):
    """Return projection matrix mapping unit-normal latent vectors to parameter steps.

    If ``u ~ N(0, I)``, then ``x = x0 + P @ u`` has local covariance approximately
    ``F^{-1}`` for Fisher matrix ``F``.

    Parameters
    ----------
    fmat : array-like
        Symmetric Fisher (or observed information) matrix.
    eps : float, optional
        Eigenvalues below ``eps`` times the largest eigenvalue (or below
        ``eps`` itself, if every eigenvalue is zero) are raised to that
        floor, with a ``RuntimeWarning`` (outside ``jax.jit``), so that flat
        directions get large but finite steps.

    Returns
    -------
    array-like
        ``P = V diag(λ^-1/2)``, so that ``P Pᵀ = F⁻¹``.
    """
    evals, evecs = np.linalg.eigh(np.asarray(fmat, dtype=float))
    largest = np.max(np.abs(evals))
    # Relative floor; a matrix with no information at all has no scale to be
    # relative to, so its directions get the absolute floor eps instead.
    floor = np.where(largest > 0.0, eps * largest, eps)
    concrete = _concrete(evals)
    if concrete is not None and (concrete < _concrete(floor)).any():
        warnings.warn(
            "fisher_projection(): the Fisher matrix has eigenvalues below "
            f"eps={eps} times its largest; they were raised to that floor.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    safe = np.maximum(evals, floor)
    return evecs * (1.0 / np.sqrt(safe))[None, :]


# === MODEL-LEVEL WRAPPERS ===

# The model-level curvatures are jitted once at module level, with the data
# and model as arguments, so repeated calls reuse one compilation. Without
# jit, jax.hessian runs hundreds of small operations one at a time (~50 ms
# per call for a binary); a jitted closure would recompile on every call.


def _leaf_shapes(params, model):
    """Static shapes of the template leaves at ``params``, or ``None``.

    ``None`` means every parameter is a scalar (or ``model`` is a callable
    with no template), so ``values`` is used as given and the compiled
    scalar-only path is unchanged.
    """
    if not isinstance(model, SourceModel):
        return None
    shapes = tuple(tuple(onp.shape(model.get(p))) for p in params)
    return None if all(s == () for s in shapes) else shapes


def _unpack(values, shapes):
    """Cut a flat ``values`` into one array per path; ``values`` itself if
    ``shapes`` is None."""
    if shapes is None:
        return values
    sizes = [int(onp.prod(shape, dtype=int)) for shape in shapes]
    if sum(sizes) != values.shape[0]:
        raise ValueError(
            f"values has {values.shape[0]} entries but the parameter paths "
            f"take {sum(sizes)} (the total size of their template leaves)."
        )
    out, start = [], 0
    for shape, size in zip(shapes, sizes):
        out.append(values[start : start + size].reshape(shape))
        start += size
    return out


@eqx.filter_jit
# deliberate: jit boundary, compiled once per model structure
def _neg_loglike_hessian(values, params, data, model, shapes=None):
    return hessian_matrix(
        lambda x: -loglike(_unpack(x, shapes), params, model, data),
        values,
    )


@eqx.filter_jit
# deliberate: jit boundary, compiled once per model structure
def _neg_loglike_curvature(values, idx, params, data, model, shapes=None):
    """``d² -log L / d values[idx]²``, the others held fixed."""

    def objective(x):
        return -loglike(
            _unpack(values.at[idx].set(x), shapes), params, model, data
        )

    return jax.grad(jax.grad(objective))(values[idx])


def _ambient_dtype():
    """``"float64"`` if JAX's 64-bit mode is on outside any local context."""
    return "float64" if jax.config.jax_enable_x64 else "float32"


def _hessian_then(finish, values, params, data, model, dtype):
    """``finish(H)`` for the Hessian ``H`` of ``-log L``, computed in ``dtype``.

    As in [`fit`][virgil.fitting.fit], the inputs are cast to ``dtype``
    inside a local ``jax.enable_x64`` context, and the result is cast back
    to JAX's precision outside it (float32, unless x64 is enabled).
    """
    ambient = _ambient_dtype()
    shapes = _leaf_shapes(params, model)
    with run_in(dtype):
        values, data, model = cast_tree(
            (np.asarray(values, dtype=float), data, model), dtype
        )
        hess = _neg_loglike_hessian(values, tuple(params), data, model, shapes)
        result = finish(hess)
    return cast_tree(result, ambient)


@old_order("values", "params", "data", "model", data="data")
def laplace_cov(values, params, model, data, *, dtype="float64"):
    """Compute the full Laplace covariance matrix for all model parameters jointly.

    The inverse of the Hessian of the negative log-likelihood with respect to
    all parameters in ``params`` simultaneously, an ``N x N`` matrix. A path
    may be array-valued (e.g. ``"rim.az_amps"``): then ``values`` is the flat
    concatenation of every path's elements (each flattened C-order), ``N`` is
    the total number of elements, and the template must be a
    [`SourceModel`][virgil.models.SourceModel] whose leaves give the sizes. For
    the uncertainty of one parameter with the others held fixed, use
    [`laplace_parameter_uncertainty`][virgil.inference.laplace_parameter_uncertainty].

    Parameters
    ----------
    values : array-like
        1D flat parameter vector: the elements of each path in ``params``,
        concatenated in order.
    params : list
        List of parameter names.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see
        [`build_model`][virgil.likelihood.build_model]); a callable takes
        scalar parameters only.
    data : OIData
        Object containing the data to be fitted.
    dtype : {"float64", "float32"}, optional
        Precision of the calculation, as for [`fit`][virgil.fitting.fit].

    Returns
    -------
    array-like
        ``N x N`` covariance matrix over the flattened parameter elements.
    """
    return _hessian_then(
        lambda hess: regularized_inverse(hess, ridge=1e-10),
        values,
        params,
        data,
        model,
        dtype,
    )


@old_order("values", "params", "data", "model", data="data")
def laplace_parameter_uncertainty(
    values, params, model, data, target_param, *, dtype="float64"
):
    """Compute scalar Laplace uncertainty for one parameter with all others fixed.

    Parameters
    ----------
    values : array-like
        Flat parameter vector at which to evaluate the curvature (the
        elements of every path in ``params``, concatenated in order, as for
        [`laplace_cov`][virgil.inference.laplace_cov]).
    params : list[str]
        Parameter paths corresponding to ``values``.
    model : SourceModel or callable
        Template model or class, as for [`loglike`][virgil.likelihood.loglike].
    data : OIData
        Data to fit.
    target_param : str
        The scalar parameter whose uncertainty is returned (a path whose
        template leaf has more than one element is rejected).
    dtype : {"float64", "float32"}, optional
        Precision of the calculation, as for
        [`laplace_cov`][virgil.inference.laplace_cov].

    Returns
    -------
    float
        ``(d² -log L / d target²)^(-1/2)``. It is NaN where the curvature is
        not positive, i.e. away from a likelihood maximum along
        ``target_param``.
    """
    params = list(params)
    if target_param not in params:
        raise ValueError(
            f"target_param '{target_param}' is not present in params={params}."
        )
    shapes = _leaf_shapes(params, model)
    if shapes is None:
        idx = params.index(target_param)
    else:
        sizes = [int(onp.prod(sh, dtype=int)) for sh in shapes]
        k = params.index(target_param)
        if sizes[k] != 1:
            raise ValueError(
                f"target_param '{target_param}' has {sizes[k]} elements; "
                "it must be a single scalar parameter."
            )
        idx = sum(sizes[:k])
    ambient = _ambient_dtype()
    with run_in(dtype):
        values, data, model = cast_tree(
            (np.asarray(values, dtype=float), data, model), dtype
        )
        d2_axis = _neg_loglike_curvature(
            values, idx, tuple(params), data, model, shapes
        )
        sigma = np.sqrt(1.0 / np.asarray(d2_axis, dtype=float))
    return cast_tree(sigma, ambient)


@old_order("values", "params", "data", "model", data="data")
def observed_information(
    values, params, model, data, ridge=0.0, *, dtype="float64"
):
    """Observed information (Hessian of ``-log L``) at a parameter point.

    At the maximum-likelihood point this approximates the Fisher matrix
    (for the expected Fisher matrix of a fixed-error prediction, see
    [`expected_fisher`][virgil.inference.expected_fisher]).

    Parameters
    ----------
    values : array-like
        Flat parameter vector at which to evaluate the local curvature (the
        elements of every path in ``params``, concatenated in order, as for
        [`laplace_cov`][virgil.inference.laplace_cov]).
    params : list[str]
        Parameter paths corresponding to ``values``.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][virgil.likelihood.build_model]).
    data : OIData
        Observational data object.
    ridge : float, optional
        Diagonal regularization term.
    dtype : {"float64", "float32"}, optional
        Precision of the calculation, as for [`laplace_cov`][virgil.inference.laplace_cov].

    Returns
    -------
    array-like
        Observed information matrix, ``N x N`` for ``N`` the total number
        of parameter elements.
    """

    return _hessian_then(
        lambda hess: _add_ridge(hess, ridge),
        values,
        params,
        data,
        model,
        dtype,
    )


# Names before 0.4 (released in 0.3.0); they stop working in 0.5.
fisher = deprecated_name("fisher", observed_information)
gaussian_fisher = deprecated_name("gaussian_fisher", expected_fisher)
laplace_covariance = deprecated_name(
    "laplace_covariance", objective_laplace_covariance
)
