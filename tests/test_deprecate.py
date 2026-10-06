"""The 0.4 argument-order and argument-name shims in ``virgil._deprecate``.

The decorators are exercised on small dummy functions that return their
arguments, so no model is evaluated. ``OIData`` instances are made
without ``__init__``: the shim only checks their type.
"""

import inspect
import warnings

import pytest

from virgil._deprecate import old_order, renamed
from virgil.oidata import OIData

ORDER = "now takes the model before the data"


def _oidata():
    return OIData.__new__(OIData)


@old_order("data", "model", "grid", data="data")
def grid_like(model, data, grid, *, batch_size=None):
    """A grid function in the 0.4 order."""
    return model, data, grid, batch_size


@old_order("values", "params", "data", "model", data="data")
def loglike_like(values, params, model, data, **options):
    return values, params, model, data, options


@old_order("params", "data", "model", data="data")
def joint_like(params, model, data, **options):
    return params, model, data, options


@old_order("template", "null_scene", "model", "grid", "key", data="template")
def recovery_like(model, null_scene, template, grid, key, *, n_null=1):
    return model, null_scene, template, grid, key, n_null


@old_order("data", "model", "grid", data="data", removed=True)
def grid_removed(model, data, grid, *, batch_size=None):
    return model, data, grid, batch_size


@renamed()
def residuals_like(model, data, **noise):
    return model, data, noise


@renamed()
def best_point_like(loglike_grid, grid):
    return loglike_grid, grid


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def _one_order_warning(fn, *args, **kwargs):
    with pytest.warns(FutureWarning, match=ORDER) as record:
        out = fn(*args, **kwargs)
    assert len(record) == 1
    return out, str(record[0].message)


def test_old_order_warns_and_matches_new_order():
    data, model, grid = _oidata(), object(), {"dra": [0.0]}
    new = _no_warnings(grid_like, model, data, grid, batch_size=2)
    old, message = _one_order_warning(
        grid_like, data, model, grid, batch_size=2
    )
    assert old == new
    assert "grid_like(model, data, grid, ...)" in message
    assert "virgil 0.5" in message


def test_old_order_with_keywords_after_the_data():
    data, model, grid = _oidata(), object(), {"dra": [0.0]}
    new = grid_like(model, data, grid)
    old, _ = _one_order_warning(grid_like, data, model=model, grid=grid)
    assert old == new


def test_old_order_with_values_first_and_var_keywords():
    data, model = _oidata(), object()
    new = _no_warnings(loglike_like, 1.0, ["x"], model, data, noise=3)
    old, message = _one_order_warning(
        loglike_like, 1.0, ["x"], data, model, noise=3
    )
    assert old == new
    assert old[-1] == {"noise": 3}
    assert "loglike_like(values, params, model, data, ...)" in message


def test_old_order_recognizes_a_sequence_of_data():
    data, model = [_oidata(), _oidata()], object()
    new = _no_warnings(joint_like, {"a": 1}, model, data)
    old, _ = _one_order_warning(joint_like, {"a": 1}, data, model)
    assert old == new


def test_old_order_with_the_model_first_of_several():
    template, null, model = _oidata(), object(), object()
    grid, key = {"dra": [0.0]}, 7
    new = _no_warnings(recovery_like, model, null, template, grid, key)
    old, message = _one_order_warning(
        recovery_like, template, null, model, grid, key, n_null=1
    )
    assert old == new
    assert "recovery_like(model, null_scene, template, grid, key" in message


def test_new_order_does_not_warn():
    data, model, grid = _oidata(), object(), {"dra": [0.0]}
    _no_warnings(grid_like, model, data, grid)
    _no_warnings(grid_like, model=model, data=data, grid=grid)
    _no_warnings(grid_like, model, data=data, grid=grid)
    _no_warnings(loglike_like, 1.0, ["x"], model=model, data=data)


@pytest.mark.parametrize(
    "first, second",
    [
        (object(), object()),  # neither is data
        ((), object()),  # an empty sequence is not data
        (_oidata(), _oidata()),  # both are data
    ],
)
def test_ambiguous_calls_pass_through_unchanged(first, second):
    out = _no_warnings(grid_like, first, second, {})
    assert out[:2] == (first, second)


@pytest.mark.parametrize(
    "fn, old, new",
    [
        (grid_like, "samples_dict", "grid"),
        (grid_like, "data_obj", "data"),
        (grid_like, "model_object", "model"),
        (joint_like, "observations", "data"),
        (joint_like, "model_fn", "model"),
        (residuals_like, "data_obj", "data"),
        (residuals_like, "model_object", "model"),
        (best_point_like, "samples_dict", "grid"),
    ],
)
def test_old_names_warn_and_match_new_names(fn, old, new):
    args = {
        "model": object(),
        "data": _oidata(),
        "grid": {"dra": [0.0]},
        "params": {"a": 1},
        "loglike_grid": [0.0],
    }
    names = list(inspect.signature(fn).parameters)
    args = {k: v for k, v in args.items() if k in names}
    expected = _no_warnings(fn, **args)
    args[old] = args.pop(new)
    with pytest.warns(FutureWarning, match=rf"{old}=\.\.\.\) is now") as rec:
        assert fn(**args) == expected
    assert len(rec) == 1
    assert f"{fn.__name__}({new}=...)" in str(rec[0].message)


def test_old_name_with_old_order_warns_twice_and_matches():
    data, model, grid = _oidata(), object(), {"dra": [0.0]}
    new = grid_like(model, data, grid)
    with pytest.warns(FutureWarning) as record:
        old = grid_like(data, model, samples_dict=grid)
    assert old == new
    messages = " ".join(str(r.message) for r in record)
    assert ORDER in messages and "samples_dict=" in messages


@pytest.mark.parametrize("fn", [grid_like, best_point_like])
def test_both_samples_dict_and_grid_is_a_type_error(fn):
    with pytest.raises(TypeError, match="both grid= and samples_dict="):
        fn([0.0], {"dra": [0.0]}, grid={}, samples_dict={})


def test_both_old_and_new_data_names_is_a_type_error():
    with pytest.raises(TypeError, match="both data= and data_obj="):
        residuals_like(object(), data=_oidata(), data_obj=_oidata())


def test_unrelated_keywords_are_not_renamed():
    # samples_dict is only an alias where the function has a grid.
    out = _no_warnings(residuals_like, object(), _oidata(), samples_dict=1)
    assert out[-1] == {"samples_dict": 1}


def test_removed_mode_raises_a_type_error_naming_the_new_call():
    data, model, grid = _oidata(), object(), {"dra": [0.0]}
    with pytest.raises(TypeError, match=r"grid_removed\(model, data, grid"):
        grid_removed(data, model, grid)
    assert _no_warnings(grid_removed, model, data, grid)[:3] == (
        model,
        data,
        grid,
    )


def test_wrapper_keeps_the_new_signature_and_docstring():
    assert list(inspect.signature(grid_like).parameters) == [
        "model",
        "data",
        "grid",
        "batch_size",
    ]
    assert grid_like.__doc__ == "A grid function in the 0.4 order."
    assert grid_like.__name__ == "grid_like"


def test_warning_points_at_the_caller():
    with pytest.warns(FutureWarning, match=ORDER) as record:
        grid_like(_oidata(), object(), {})
    assert record[0].filename == __file__
    with pytest.warns(FutureWarning, match="samples_dict") as record:
        best_point_like([0.0], samples_dict={})
    assert record[0].filename == __file__


def test_misnamed_parameters_fail_at_decoration():
    with pytest.raises(ValueError, match="no parameters"):

        @old_order("data", "model", data="data")
        def f(model, data_obj):
            pass
