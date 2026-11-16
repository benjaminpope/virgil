"""The 0.4 argument-order and argument-name shims in ``virgil._deprecate``.

The decorators are exercised on small dummy functions that return their
arguments, so no model is evaluated. ``OIData`` instances are made
without ``__init__``: the shim only checks their type.
"""

import inspect
import warnings

import jax
import jax.numpy as np
import pytest

from virgil._deprecate import (
    deprecated_name,
    keyword_only,
    old_order,
    renamed,
)
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


def test_warning_during_tracing_points_at_the_caller():
    # A warning raised while JAX traces a function is not attributed to a
    # jax/_src line: the frames of JAX and numpyro are looked through.
    # The warner lives in a virgil module (built with exec), and is traced
    # by jit and vmap, which add JAX's frames between it and this file.
    namespace = {"__name__": "virgil._fake"}
    exec(
        "import warnings\n"
        "from virgil._deprecate import external_stacklevel\n"
        "def warner(x):\n"
        "    warnings.warn('old', FutureWarning,"
        " stacklevel=external_stacklevel())\n"
        "    return x\n",
        namespace,
    )
    warner = namespace["warner"]
    for transform in (jax.jit, jax.vmap):
        with pytest.warns(FutureWarning, match="old") as record:
            transform(warner)(np.ones(2))
        assert record[0].filename == __file__


def test_misnamed_parameters_fail_at_decoration():
    with pytest.raises(ValueError, match="no parameters"):

        @old_order("data", "model", data="data")
        def f(model, data_obj):
            pass


def _small_data():
    return OIData(
        {
            "u": [1.0, 2.0],
            "v": [0.0, 1.0],
            "wavel": 2e-6,
            "vis": [0.9, 0.8],
            "d_vis": [0.02, 0.02],
        }
    )


def test_warning_points_at_the_caller_for_equinox_methods():
    from virgil.models import UniformDisk

    data = _small_data()
    with pytest.warns(FutureWarning, match="model_object") as record:
        data.model(model_object=UniformDisk(2.0))
    assert record[0].filename == __file__
    assert "OIData.model(model_object=" in str(record[0].message)


def test_new_name_positional_with_old_keyword_is_the_shims_type_error():
    with pytest.raises(TypeError, match="got both data= and data_obj="):
        residuals_like("m", "d", data_obj="d")
    with pytest.raises(TypeError, match="got both"):
        residuals_like("m", data="d", data_obj="d")


def test_messages_use_the_qualified_name():
    class A:
        @renamed()
        def method(self, model):
            return model

    with pytest.warns(FutureWarning, match=r"A\.method\(model_object=") as r:
        A().method(model_object=1)
    assert r[0].filename == __file__


def test_l_curve_accepts_the_old_regulariser_keyword():
    import jax.numpy as np

    from tests.test_imaging import DATA, NPIX, _image
    from virgil.imaging import TSV, image_priors, l_curve
    from virgil.models import PointSource, System

    start = System(
        star=PointSource(), env=_image(np.ones((NPIX, NPIX)), flux=0.2)
    )
    args = (start, image_priors(start), DATA)
    kw = {"weights": [1e3], "method": "lm"}
    new = l_curve(*args, regularizer=TSV(1.0, path="env", beam_mas=12.0), **kw)
    with pytest.warns(FutureWarning, match="regulariser=") as record:
        old = l_curve(
            *args, regulariser=TSV(1.0, path="env", beam_mas=12.0), **kw
        )
    assert record[0].filename == __file__
    assert np.allclose(old.chi2, new.chi2)


@old_order("data", "model", "grid", data="data", keywords=("data", "grid"))
def keyword_grid(model, *, data, grid, batch_size=None):
    return model, data, grid, batch_size


def test_keyword_only_parameters_reject_any_other_positional_call():
    data = _oidata()
    for args in [(1, data, {}), (1, {}, data), ({}, 1, data), (1, data)]:
        with pytest.raises(TypeError, match="by keyword") as err:
            keyword_grid(*args)
        assert "keyword_grid(model, data=..., grid=..., ...)" in str(err.value)
    assert keyword_grid(1, data=data, grid={}) == (1, data, {}, None)
    assert keyword_grid(model=1, grid={}, data=data)[0] == 1


def test_keyword_only_parameters_keep_the_old_order_shim():
    data = _oidata()
    with pytest.warns(FutureWarning, match=ORDER):
        assert keyword_grid(data, 1, {}) == (1, data, {}, None)
    with pytest.warns(FutureWarning, match=ORDER):
        assert keyword_grid(data, 1, grid={}) == (1, data, {}, None)


def test_keywords_must_be_keyword_only_parameters_of_the_function():
    with pytest.raises(ValueError, match="not keyword-only"):

        @old_order("data", "model", data="data", keywords=("model",))
        def bad(model, data):
            return model, data

    with pytest.raises(ValueError, match="no parameters"):

        @old_order("data", "model", data="data", keywords=("grid",))
        def worse(model, *, data):
            return model, data


@keyword_only("b", "c")
def kw_only_like(a, *, b=1, c=2, d=3):
    return a, b, c, d


def test_keyword_only_warns_on_the_0_3_positional_use():
    assert kw_only_like(0, b=5, d=6) == (0, 5, 2, 6)
    with pytest.warns(FutureWarning, match="stops working in virgil 0.5"):
        assert kw_only_like(0, 5) == (0, 5, 2, 3)
    with pytest.warns(FutureWarning, match="pass b, c by keyword"):
        assert kw_only_like(0, 5, 6, d=7) == (0, 5, 6, 7)


def test_keyword_only_rejects_positionals_it_never_allowed():
    # ``d`` is new since 0.3, so it was never positional.
    with pytest.raises(TypeError, match="pass the rest by keyword"):
        kw_only_like(0, 1, 2, 3)
    with pytest.raises(TypeError, match="multiple values for b"):
        kw_only_like(0, 1, b=2)


def test_keyword_only_needs_keyword_only_parameters():
    with pytest.raises(ValueError, match="not keyword-only"):

        @keyword_only("b")
        def bad(a, b):
            return a, b


def test_deprecated_name_forwards_and_warns():
    def new(x, y=2):
        """New."""
        return x + y

    old = deprecated_name("old", new)
    assert old.__name__ == "old"
    with pytest.warns(FutureWarning, match=r"old is now new; .*virgil 0.5"):
        assert old(1, y=3) == 4


def test_starting_image_options_are_keyword_only():
    from virgil.imaging import starting_image

    params = inspect.signature(starting_image).parameters
    assert [
        n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD
    ] == ["data"]
    # Parameters new in 0.4 were never positional, so they have no shim.
    with pytest.raises(TypeError):
        starting_image(_oidata(), True, 4.0, None, "moments", None, 6.0)
