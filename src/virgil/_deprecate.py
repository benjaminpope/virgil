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
import sys
import warnings

__all__ = [
    "blur_alias",
    "deprecated_name",
    "keyword_only",
    "old_order",
    "renamed",
]

#: Old keyword names accepted until 0.5, mapped to their 0.4 names.
NAME_ALIASES = {
    "data_obj": "data",
    "observations": "data",
    "model_object": "model",
    "model_fn": "model",
    "samples_dict": "grid",
    "ud": "diam",
    # Oxford -ize spelling (0.4)
    "regularisers": "regularizers",
    "marginalise_offsets": "marginalize_offsets",
    "regulariser": "regularizer",
}

#: Top-level packages whose frames a deprecation warning looks through.
_INTERNAL = (
    "virgil",
    "equinox",
    "functools",
    "jax",
    "jaxlib",
    "jaxtyping",
    "lineax",
    "numpyro",
    "optax",
    "optimistix",
)


def external_stacklevel():
    """The ``stacklevel`` of the first caller outside virgil and its wrappers.

    Meant for ``warnings.warn(..., stacklevel=external_stacklevel())`` in
    the function that warns, so the warning is attributed to user code
    even when equinox's method wrappers, ``functools``, jaxtyping, or the
    tracing of JAX, numpyro, optimistix, lineax or optax add frames
    between the two. Falls back to the outermost frame.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        top = frame.f_globals.get("__name__", "").partition(".")[0]
        if top not in _INTERNAL:
            return level
        frame = frame.f_back
        level += 1
    return level - 1


def _is_data(x):
    """Whether ``x`` is an ``OIData``, an ``Epochs`` or a non-empty sequence
    of ``OIData``."""
    from .epochs import Epochs  # lazy: keeps this module import-free
    from .oidata import OIData

    if isinstance(x, (OIData, Epochs)):
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


def _apply_aliases(name, args, kwargs, aliases, positional=()):
    """Rename deprecated keywords in ``kwargs`` in place, with a warning.

    ``positional`` lists the parameter names in positional order, to catch
    a new name passed positionally beside its old keyword.
    """
    for old, new in aliases.items():
        if old not in kwargs:
            continue
        if new in kwargs or (
            new in positional and positional.index(new) < len(args)
        ):
            raise TypeError(f"{name} got both {new}= and {old}=; use {new}=")
        warnings.warn(
            f"{name}({old}=...) is now {name}({new}=...); "
            f"{old}= stops working in virgil 0.5.",
            FutureWarning,
            stacklevel=external_stacklevel(),
        )
        kwargs[new] = kwargs.pop(old)


def _positional_names(params):
    """Names of the parameters that can be passed positionally, in order."""
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return tuple(n for n, p in params.items() if p.kind in kinds)


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
        positional = _positional_names(params)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(fn.__qualname__, args, kwargs, names, positional)
            return fn(*args, **kwargs)

        return wrapper

    return decorate


def deprecated_name(old, new_fn):
    """A function named ``old`` that calls ``new_fn``, with a FutureWarning.

    For a public function renamed in 0.4: the old name keeps working, and
    warns that it stops working in virgil 0.5. The wrapper passes every
    argument through, so ``new_fn``'s own shims (``old_order``,
    ``renamed``) still apply to calls made through the old name.
    """
    new = new_fn.__name__

    @functools.wraps(new_fn)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{old} is now {new}; {old} stops working in virgil 0.5.",
            FutureWarning,
            stacklevel=external_stacklevel(),
        )
        return new_fn(*args, **kwargs)

    wrapper.__name__ = wrapper.__qualname__ = old
    wrapper.__doc__ = f"Deprecated name of :func:`{new}`; use that."
    return wrapper


def keyword_only(*legacy):
    """Make the parameters after a bare ``*`` keyword-only, softly.

    ``fn`` is declared with ``*`` after its leading positionals, but 0.3
    let the parameters named in ``legacy`` (in that order) be passed
    positionally. Such a call still works until 0.5, with a
    :class:`FutureWarning` naming the keywords; any further positional
    argument is a :class:`TypeError`, as for any function.
    """

    def decorate(fn):
        params = inspect.signature(fn).parameters
        unknown = [
            n
            for n in legacy
            if n not in params
            or params[n].kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        if unknown:
            raise ValueError(f"{fn.__name__}: {unknown} are not keyword-only")
        n_pos = len(_positional_names(params))
        qualname = fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            extra = args[n_pos:]
            if not extra:
                return fn(*args, **kwargs)
            if len(extra) > len(legacy):
                raise TypeError(
                    f"{qualname} takes {n_pos} positional arguments but "
                    f"{len(args)} were given; pass the rest by keyword."
                )
            names = legacy[: len(extra)]
            clash = [n for n in names if n in kwargs]
            if clash:
                raise TypeError(
                    f"{qualname} got multiple values for {clash[0]}="
                )
            warnings.warn(
                f"{qualname}: pass {', '.join(names)} by keyword "
                f"({', '.join(f'{n}=...' for n in names)}); "
                "positional use stops working in virgil 0.5.",
                FutureWarning,
                stacklevel=external_stacklevel(),
            )
            return fn(*args[:n_pos], **dict(zip(names, extra)), **kwargs)

        return wrapper

    return decorate


def old_order(*old_names, data, aliases=None, removed=False, keywords=()):
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
    keywords : tuple of str
        Parameters that ``fn`` declares keyword-only and that no call may
        pass positionally. A call with more positional arguments than
        ``fn`` has positional parameters (the 0.4 order spelled
        ``f(model, data, grid)``, or any other permutation of the three)
        raises a :class:`TypeError` naming the keyword call, rather than
        letting the arguments land in the wrong parameters or fail deep
        inside. Old-order calls are rebound as before: the first
        ``old_names`` are positional in the 0.3 signature.

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
        positional = _positional_names(params)
        qualname = fn.__qualname__
        new_names = [n for n in params if n in old_names]
        misplaced = [n for n in keywords if n not in params]
        if misplaced:
            raise ValueError(f"{fn.__name__} has no parameters {misplaced}")
        not_kw = [
            n
            for n in keywords
            if params[n].kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        if not_kw:
            raise ValueError(f"{fn.__name__}: {not_kw} are not keyword-only")
        # The 0.3 signature: the old names positional, in the old order.
        moved = [
            params[n].replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in old_names
        ] + [p for n, p in params.items() if n not in old_names]
        old_sig = sig.replace(parameters=moved)
        i_old, i_new = old_names.index(data), new_names.index(data)
        new_call = (
            f"{qualname}("
            + ", ".join(f"{n}=..." if n in keywords else n for n in new_names)
            + ", ...)"
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _apply_aliases(qualname, args, kwargs, names, positional)
            if not (
                len(args) > i_old
                and _is_data(args[i_old])
                and not (len(args) > i_new and _is_data(args[i_new]))
            ):
                if keywords and len(args) > len(positional):
                    raise TypeError(
                        f"{qualname} takes {', '.join(positional)} "
                        f"positionally and {', '.join(keywords)} by keyword "
                        f"(got {len(args)} positional arguments); "
                        f"call {new_call}."
                    )
                return fn(*args, **kwargs)
            if removed:
                raise TypeError(
                    f"{qualname} takes the model before the data since "
                    f"virgil 0.4; call {new_call}."
                )
            warnings.warn(
                f"{qualname} now takes the model before the data; "
                f"call {new_call}. The old order stops working in "
                "virgil 0.5.",
                FutureWarning,
                stacklevel=external_stacklevel(),
            )
            bound = old_sig.bind(*args, **kwargs)
            # flatten the binding to keywords, expanding any ``**kwargs``
            out = {}
            for name, value in bound.arguments.items():
                if params[name].kind is inspect.Parameter.VAR_KEYWORD:
                    out.update(value)
                else:
                    out[name] = value
            return fn(**out)

        return wrapper

    return decorate


def blur_alias(owner, old, old_value, blur_fwhm_mas):
    """``blur_fwhm_mas``, or the deprecated ``old`` keyword's value.

    Warns (``DeprecationWarning``) when ``old`` is given and raises when
    both are.
    """
    if old_value is None:
        return blur_fwhm_mas
    if blur_fwhm_mas is not None:
        raise TypeError(
            f"{owner} got both blur_fwhm_mas= and {old}=; use blur_fwhm_mas="
        )
    warnings.warn(
        f"{owner}({old}=...) is now {owner}(blur_fwhm_mas=...); "
        f"{old}= stops working in virgil 0.5.",
        DeprecationWarning,
        stacklevel=external_stacklevel(),
    )
    return old_value
