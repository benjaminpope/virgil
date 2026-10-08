"""The 0.4 model-before-data argument order and argument names.

Calls in the 0.3 order still work, with one ``FutureWarning`` naming the
new call, and give exactly the new-order result (the same code runs).
New-order calls, positional or by keyword, do not warn. The old keyword
names (``samples_dict=``, ``data_obj=``, ``observations=``,
``model_object=``, ``model_fn=``) warn and still work. The data are the
smallest that exercise each function: the 7-hole mask and a 2 x 2 grid.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as onp  # noqa: E402
import pytest  # noqa: E402

from virgil.coverage import nrm_oidata  # noqa: E402
from virgil.detection import (  # noqa: E402
    bootstrap_null,
    detection_statistics,
    gaussian_null,
    injection_recovery,
    rescale_errors,
)
from virgil.grid_fit import (  # noqa: E402
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    linear_flux_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from virgil.inference import (  # noqa: E402
    fisher,
    laplace_cov,
    laplace_parameter_uncertainty,
)
from virgil.likelihood import (  # noqa: E402
    flux_scale_posterior,
    joint_loglike,
    joint_prediction,
    loglike,
    model_loglike,
    numpyro_model,
    posterior_predictive_summary,
    whitened_residuals,
)
from virgil.limits import absil_limits, injection_limits  # noqa: E402
from virgil.models import BinaryModelCartesian  # noqa: E402
from virgil.plotting import plot_contrast_curve, plot_grid_map  # noqa: E402

ORDER = "now takes the model before the data"
MODEL = BinaryModelCartesian
NULL = BinaryModelCartesian(0.0, 0.0, 0.0)
DATA = nrm_oidata().with_model(
    BinaryModelCartesian(60.0, -40.0, 5e-3), key=jax.random.PRNGKey(0)
)
GRID = {
    "dra": jnp.array([40.0, 80.0]),
    "ddec": jnp.array([-60.0, -20.0]),
    "flux": jnp.array([1e-3, 1e-2]),
}

GRID_TOOLS = [
    likelihood_grid,
    optimized_likelihood_grid,
    optimized_flux_grid,
    linear_flux_grid,
    laplace_flux_uncertainty_grid,
    detection_statistics,
]
LIMITS = [absil_limits, injection_limits]


def _future(record):
    return [w for w in record if issubclass(w.category, FutureWarning)]


def _old_order(fn, *args, **kwargs):
    """Call ``fn`` and check it raised one order warning naming ``fn``."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    future = _future(record)
    assert len(future) == 1, [str(w.message) for w in future]
    message = str(future[0].message)
    assert ORDER in message and f"call {fn.__name__}(" in message
    assert "stops working in virgil 0.5" in message
    assert future[0].filename == __file__
    return out


def _new_order(fn, *args, **kwargs):
    """Call ``fn`` and check it raised no FutureWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # optimizer notes
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        return fn(*args, **kwargs)


def _assert_same(a, b):
    leaves_a, tree_a = jax.tree_util.tree_flatten(a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(b)
    assert tree_a == tree_b
    for x, y in zip(leaves_a, leaves_b):
        onp.testing.assert_array_equal(onp.asarray(x), onp.asarray(y))


@pytest.mark.parametrize("fn", GRID_TOOLS, ids=lambda f: f.__name__)
def test_grid_tools_accept_the_old_order_with_a_warning(fn):
    new = _new_order(fn, MODEL, DATA, GRID)
    _assert_same(_new_order(fn, model=MODEL, data=DATA, grid=GRID), new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old = _old_order(fn, DATA, MODEL, GRID)
    _assert_same(old, new)


@pytest.mark.parametrize("fn", LIMITS, ids=lambda f: f.__name__)
def test_limits_accept_the_old_order_with_a_warning(fn):
    new = _new_order(fn, MODEL, DATA, GRID, 3.0)
    _assert_same(_new_order(fn, MODEL, DATA, grid=GRID, sigma=3.0), new)
    _assert_same(_old_order(fn, DATA, MODEL, GRID, 3.0), new)


@pytest.mark.parametrize("fn", GRID_TOOLS[:3], ids=lambda f: f.__name__)
def test_old_keyword_names_warn_and_still_work(fn):
    new = _new_order(fn, MODEL, DATA, GRID)
    for old_name, new_name in (("samples_dict", "grid"), ("data_obj", "data")):
        kwargs = {"model": MODEL, "data": DATA, "grid": GRID}
        kwargs[old_name] = kwargs.pop(new_name)
        with pytest.warns(FutureWarning, match=f"{old_name}=") as record:
            out = fn(**kwargs)
        assert len(_future(record)) == 1
        _assert_same(out, new)
    with pytest.raises(TypeError, match="both grid= and samples_dict="):
        fn(MODEL, DATA, grid=GRID, samples_dict=GRID)


def test_old_order_with_keyword_grid_still_warns():
    new = _new_order(likelihood_grid, MODEL, DATA, GRID)
    _assert_same(_old_order(likelihood_grid, DATA, MODEL, grid=GRID), new)
    with pytest.warns(FutureWarning) as record:
        out = likelihood_grid(DATA, MODEL, samples_dict=GRID)
    messages = [str(w.message) for w in _future(record)]
    assert any("samples_dict=" in m for m in messages)
    assert any(ORDER in m for m in messages)
    _assert_same(out, new)


def test_simulators_take_the_null_scene_first():
    key = jax.random.PRNGKey(1)
    # gaussian_null calls its data the template.
    for fn, data_name in (
        (gaussian_null, "template"),
        (bootstrap_null, "data"),
    ):
        new = _new_order(fn, NULL, DATA)(key)
        by_name = _new_order(fn, null_scene=NULL, **{data_name: DATA})
        _assert_same(by_name(key), new)
        _assert_same(_old_order(fn, DATA, NULL)(key), new)
    scaled, factors = _new_order(rescale_errors, NULL, DATA)
    old_scaled, old_factors = _old_order(rescale_errors, DATA, NULL)
    assert old_factors == factors
    _assert_same(old_scaled.d_vis, scaled.d_vis)


def test_injection_recovery_takes_the_model_first():
    kwargs = {"n_null": 2, "progress": False}
    new = _new_order(injection_recovery, MODEL, NULL, DATA, GRID, 0, **kwargs)
    old = _old_order(injection_recovery, DATA, NULL, MODEL, GRID, 0, **kwargs)
    assert new.null.keys() == old.null.keys()
    for name in new.null:
        onp.testing.assert_array_equal(old.null[name], new.null[name])
    assert old.meta == new.meta


def test_detection_statistics_old_order_under_jit():
    new = _new_order(detection_statistics, MODEL, DATA, GRID)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old = jax.jit(lambda d: detection_statistics(d, MODEL, GRID))(DATA)
    assert any(ORDER in str(w.message) for w in _future(record))
    for name in new:
        onp.testing.assert_allclose(old[name], new[name], rtol=1e-5)


def test_ambiguous_calls_pass_through_without_warning():
    # Neither the first nor the second argument is data: bound as given, so
    # the function's own error applies.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(Exception):
            likelihood_grid(MODEL, MODEL, GRID)


def test_grid_rename_in_best_point_and_plots():
    values = _new_order(likelihood_grid, MODEL, DATA, GRID)
    best = _new_order(best_grid_point, values, GRID)
    with pytest.warns(FutureWarning, match="samples_dict="):
        assert best_grid_point(values, samples_dict=GRID) == best
    with pytest.warns(FutureWarning, match="samples_dict="):
        fig, _ = plot_grid_map(values, samples_dict=GRID)
    plt.close(fig)
    limits = _new_order(absil_limits, MODEL, DATA, GRID, 3.0)
    with pytest.warns(FutureWarning, match="samples_dict="):
        fig, _ = plot_contrast_curve(limits, samples_dict=GRID)
    plt.close(fig)


# --- Values-first and joint likelihoods (argument-order PR 3) -------------

PARAMS = ["dra", "ddec", "flux"]
VALUES = jnp.array([60.0, -40.0, 5e-3])
DATA2 = nrm_oidata(rotation_deg=30.0).with_model(
    BinaryModelCartesian(60.0, -40.0, 5e-3), key=jax.random.PRNGKey(1)
)
JOINT_PARAMS = {"dra": 60.0, "ddec": -40.0, "flux": 5e-3}


def _joint_model(params, index):
    return BinaryModelCartesian(params["dra"], params["ddec"], params["flux"])


@pytest.mark.parametrize(
    "fn", [loglike, laplace_cov, fisher], ids=lambda f: f.__name__
)
def test_values_first_functions_accept_the_old_order(fn):
    new = _new_order(fn, VALUES, PARAMS, MODEL, DATA)
    by_name = _new_order(
        fn, values=VALUES, params=PARAMS, model=MODEL, data=DATA
    )
    _assert_same(by_name, new)
    _assert_same(_old_order(fn, VALUES, PARAMS, DATA, MODEL), new)
    with pytest.warns(FutureWarning, match="data_obj=") as record:
        out = fn(VALUES, PARAMS, model=MODEL, data_obj=DATA)
    assert len(_future(record)) == 1
    _assert_same(out, new)


def test_laplace_parameter_uncertainty_accepts_the_old_order():
    fn = laplace_parameter_uncertainty
    new = _new_order(fn, VALUES, PARAMS, MODEL, DATA, "flux")
    _assert_same(_old_order(fn, VALUES, PARAMS, DATA, MODEL, "flux"), new)


@pytest.mark.parametrize(
    "fn", [joint_prediction, joint_loglike], ids=lambda f: f.__name__
)
def test_joint_functions_take_the_model_first(fn):
    data = (DATA, DATA2)
    new = _new_order(fn, JOINT_PARAMS, _joint_model, data)
    _assert_same(_new_order(fn, JOINT_PARAMS, _joint_model, list(data)), new)
    _assert_same(_old_order(fn, JOINT_PARAMS, data, _joint_model), new)
    with pytest.warns(FutureWarning) as record:
        out = fn(JOINT_PARAMS, model_fn=_joint_model, observations=data)
    messages = [str(w.message) for w in _future(record)]
    assert len(messages) == 2
    assert any("model_fn=" in m for m in messages)
    assert any("observations=" in m for m in messages)
    _assert_same(out, new)


def test_loglike_old_order_under_jit_and_grad():
    new = _new_order(loglike, VALUES, PARAMS, MODEL, DATA)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old = jax.jit(lambda v: loglike(v, PARAMS, DATA, MODEL))(VALUES)
        # argnums=0 is the values, whose position does not change.
        grad_old = jax.grad(loglike)(VALUES, PARAMS, DATA, MODEL)
    assert sum(ORDER in str(w.message) for w in _future(record)) == 2
    onp.testing.assert_allclose(old, new, rtol=1e-6)
    grad_new = _new_order(jax.grad(loglike), VALUES, PARAMS, MODEL, DATA)
    _assert_same(grad_old, grad_new)


@pytest.mark.parametrize(
    "fn",
    [whitened_residuals, model_loglike, flux_scale_posterior],
    ids=lambda f: f.__name__,
)
def test_model_first_functions_accept_their_old_names(fn):
    scene = BinaryModelCartesian(*VALUES)
    new = _new_order(fn, scene, DATA)
    _assert_same(_new_order(fn, model=scene, data=DATA), new)
    with pytest.warns(FutureWarning) as record:
        out = fn(model_object=scene, data_obj=DATA)
    messages = [str(w.message) for w in _future(record)]
    assert len(messages) == 2
    assert any("model_object=" in m for m in messages)
    assert any("data_obj=" in m for m in messages)
    _assert_same(out, new)
    with pytest.raises(TypeError, match="both data= and data_obj="):
        fn(scene, data=DATA, data_obj=DATA)


def test_numpyro_model_and_posterior_summary_accept_data_obj():
    import numpyro.distributions as dist

    priors = {"flux": dist.LogUniform(1e-4, 0.1)}
    template = BinaryModelCartesian(60.0, -40.0, 5e-3)
    with pytest.warns(FutureWarning, match="data_obj="):
        numpyro_model(template, priors, data_obj=DATA)
    samples = {"flux": jnp.array([4e-3, 5e-3, 6e-3])}
    new = _new_order(posterior_predictive_summary, samples, template, DATA)
    with pytest.warns(FutureWarning, match="data_obj="):
        old = posterior_predictive_summary(samples, template, data_obj=DATA)
    _assert_same(old, new)


def test_fit_and_numpyro_model_accept_the_old_regularisers_keyword():
    import inspect

    from virgil.fitting import fit

    for fn in (fit, numpyro_model):
        wrapped = inspect.signature(fn).parameters
        assert "regularizers" in wrapped and "regularisers" not in wrapped
    with pytest.warns(FutureWarning, match="regularisers="):
        numpyro_model(MODEL, {}, DATA, regularisers=())
