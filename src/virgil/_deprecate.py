"""Deprecation helpers for the 0.4 argument-order change (private).

virgil 0.4 puts the model before the data in every public function that
takes both, renames ``samples_dict`` to ``grid``, and standardizes the
argument names ``model`` and ``data``. :func:`old_order` lets calls in
the 0.3 positional order keep working with a :class:`FutureWarning`, and
:func:`renamed` accepts the old keyword names. In 0.5 the aliases go and
``old_order(..., removed=True)`` turns the old order into a
:class:`TypeError` naming the new call.

The checks look only at the Python types of the arguments, never at
array values, so they are valid while JAX traces a call.
"""

import functools
import inspect
import warnings

__all__ = ["old_order", "renamed"]

#: Old keyword names accepted until 0.5, mapped to their 0.4 names.
NAME_ALIASES = {
    "data_obj": "data",
    "observations": "data",
    "model_object": "model",
    "model_fn": "model",
    "samples_dict": "grid",
}


def _is_data(x):
    """Whether ``x`` is an ``OIData`` or a non-empty sequence of them."""
    from .oidata import OIData  # lazy: keeps this module import-free

    if isinstance(x, OIData):
        return True
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(isinstance(d, OIData) for d in x)
    )


def _alias_map(params, aliases):
    """The aliases in ``aliases`` whose new name is a parameter of ``fn``."""
    return {
        old: new
        for old, new in aliases.items()
        if new in params and old not in params
    }


def _apply_aliases(name, kwargs, aliases):
    """Rename deprecated keywords in ``kwargs`` in place, with a warning."""
    for old, new in aliases.items():
        if old not in kwargs:
            continue
        if new in kwargs:
            raise TypeError(f"{name} got both {new}= and {old}=; use {new}=")
        warnings.warn(
            f"{name}({old}=...) is now {name}({new}=...); "
            f"{old}= stops working in virgil 0.5.",
            FutureWarning,
            stacklevel=3,
        )
        kwargs[new] = kwargs.pop(old)


def _as_keywords(bound, params):
    """Flatten a binding to keywords, expanding any ``**kwargs``."""
    out = {}
    for name, value in bound.arguments.items():
        if params[name].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[name] = value
    return out


def renamed(aliases=None):
    """Accept the pre-0.4 keyword names of ``fn``'s parameters until 0.5.

    ``aliases`` maps old names to new ones; it defaults to
    :data:`NAME_ALIASES`, restricted to the new names ``fn`` has.
    """

    def decorate(fn):
        params = inspect.signature(fn).parameters
        names = _alias_map(
            params, NAME_ALIASES if aliases is None else aliases
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(fn.__name__, kwargs, names)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def old_order(*old_names, data, aliases=None, removed=False):
    """Accept calls in the pre-0.4 positional order, with a FutureWarning.

    Parameters
    ----------
    *old_names : str
        The leading positional parameters in their old order, under their
        new names.
    data : str
        The name of the data parameter, which is how an old-order call is
        recognized: the argument at its old position is data and the one
        at its new position is not.
    aliases : dict, optional
        Old keyword names mapped to new ones, as for :func:`renamed`.
    removed : bool
        Raise a :class:`TypeError` naming the new call instead of
        rebinding (the 0.5 behaviour).

    Calls with data at both positions or at neither are passed through
    unchanged, so new-order code never warns.
    """

    def decorate(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        unknown = [n for n in (*old_names, data) if n not in params]
        if unknown:
            raise ValueError(f"{fn.__name__} has no parameters {unknown}")
        names = _alias_map(
            params, NAME_ALIASES if aliases is None else aliases
        )
        new_names = [n for n in params if n in old_names]
        moved = [params[n] for n in old_names] + [
            p for n, p in params.items() if n not in old_names
        ]
        old_sig = sig.replace(parameters=moved)
        i_old, i_new = old_names.index(data), new_names.index(data)
        new_call = f"{fn.__name__}({', '.join(new_names)}, ...)"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(fn.__name__, kwargs, names)
            if not (
                len(args) > i_old
                and _is_data(args[i_old])
                and not (len(args) > i_new and _is_data(args[i_new]))
            ):
                return fn(*args, **kwargs)
            if removed:
                raise TypeError(
                    f"{fn.__name__} takes the model before the data since "
                    f"virgil 0.4; call {new_call}."
                )
            warnings.warn(
                f"{fn.__name__} now takes the model before the data; "
                f"call {new_call}. The old order stops working in "
                "virgil 0.5.",
                FutureWarning,
                stacklevel=2,
            )
            bound = old_sig.bind(*args, **kwargs)
            return fn(**_as_keywords(bound, params))

        return wrapper

    return decorate
