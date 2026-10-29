"""K-fold cross-validation of a hyperparameter, generic over the data.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The unit of work is [`cv_task`][virgil.crossval.cv_task]: fit with one
hyperparameter setting on the training points of one fold, then score the
held-out points. It returns a small JSON-friendly record, so the (setting,
fold) pairs can run in one process
([`cross_validate`][virgil.crossval.cross_validate]) or as separate cluster
array tasks whose records [`aggregate`][virgil.crossval.aggregate] (or
[`aggregate_files`][virgil.crossval.aggregate_files]) combines.

**The data protocol.** The data enter only through ``make_term(mask)``, which
returns a likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``
using the points where the boolean ``mask`` (length ``N``) is true and
ignoring the rest, *without changing array shapes*, so that fits for
different masks share one compilation. The term is called on the fitted
values and returns whitened residuals: ``m`` consecutive blocks of ``N``
point residuals (``m = 2`` for the real and imaginary parts of a visibility,
``m = 1`` for a real datum), zero at masked points. If it has
``has_log_norm`` true, ``log_norm(values)`` is the normalization that ``fit``
adds to ½χ², of the unmasked points only.
[`ComplexVisData.term`][virgil.radio.ComplexVisData.term] satisfies this; so
can a term for any other data.

**The score** of a fold is the negative log likelihood of its held-out
points under the training fit, up to constants::

    ½ Σ r² + log_norm

summed over the held-out points, at the fitted values including the fitted
error scales. The record also holds the raw χ² (``Σ r²`` with every error
scale set to 1, if ``unit_scale`` is given) and, if ``groups`` are given, the
same sums per group.
"""

import contextlib
import json
import math
import warnings
from numbers import Real

import equinox as eqx
import jax
import numpy as onp

from ._geometry import set_ft_defaults
from ._precision import run_in
from .fitting import fit

__all__ = [
    "aggregate",
    "aggregate_files",
    "cross_validate",
    "cv_task",
    "fold_masks",
    "load_records",
    "save_record",
]

__provisional__ = ("*",)


def fold_masks(fold, k):
    """Training and held-out masks of fold ``k``: ``fold != k`` and ``fold == k``.

    ``fold`` is the integer fold of every point (e.g. from
    [`dartboard_folds`][virgil.radio.dartboard_folds]). The masks are disjoint
    and together cover all points. Raises if fold ``k`` is empty.
    """
    fold = onp.asarray(fold)
    test = fold == k
    if not test.any():
        raise ValueError(f"fold {k} has no points.")
    return ~test, test


@contextlib.contextmanager
def _ft(settings):
    """Run with these ``set_ft_defaults`` settings, restoring the old ones."""
    if not settings:
        yield
        return
    old = set_ft_defaults(**settings)
    try:
        yield
    finally:
        set_ft_defaults(**old)


@eqx.filter_jit
def _evaluate(term, values, ft_key):
    """Residuals and log normalization of a term.

    ``ft_key`` is static, so the cache is keyed by the precision settings,
    which are read at trace time.
    """
    log_norm = (
        term.log_norm(values) if getattr(term, "has_log_norm", False) else 0.0
    )
    return term(values), log_norm


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (onp.generic, onp.ndarray, jax.Array)):
        return _jsonable(onp.asarray(x).tolist())
    return x


def _sums(residual, n, labels, ids):
    """Σ r² per group label from a residual vector of ``m`` blocks of ``n``."""
    r2 = onp.asarray(residual, dtype=onp.float64) ** 2
    per_point = r2.reshape(-1, n).sum(axis=0)
    return onp.bincount(ids, weights=per_point, minlength=len(labels))


def cv_task(
    setting,
    k,
    fold,
    *,
    fit_kwargs,
    make_term,
    init=None,
    groups=None,
    unit_scale=None,
    fit_ft=None,
    score_ft=None,
    dtype="float64",
    fit_options=None,
):
    """Fit on the training points of fold ``k`` and score the held-out points.

    Parameters
    ----------
    setting : float or JSON-friendly
        The hyperparameter value (e.g. a regularization weight λ).
    k : int
        The held-out fold label (a value in ``fold``).
    fold : (N,) int array
        The fold label of every point, e.g. from
        [`dartboard_folds`][virgil.radio.dartboard_folds]; the caller loops over
        folds and settings.
    fit_kwargs : callable
        ``fit_kwargs(setting)`` returns the mask-independent arguments of
        [`fit`][virgil.fitting.fit] (``model``, ``priors``, ``data=()``,
        ``regularizers=...``). Vary the setting through array leaves (a
        regularizer's weight, a prior's scale), not static structure, so
        every setting reuses one compilation.
    make_term : callable
        ``make_term(mask)``: the likelihood term of the points where ``mask``
        is true (fitted on the training mask, scored on the held-out one);
        pass ``data=()`` in ``fit_kwargs``, or the held-out points leak into
        the fit. Same function for every task.
    init : dict, optional
        Starting values for the fit, the same for every task.
    groups : (N,) int array, optional
        A label per point (uv annulus, spectral window): the record breaks the
        held-out sums down by group.
    unit_scale : callable, optional
        ``unit_scale(values)`` returns the fitted values with every error
        scale set to 1, for the raw χ²; ``chi2`` is ``None`` without it.
    fit_ft, score_ft : dict, optional
        [`set_ft_defaults`][virgil._geometry.set_ft_defaults] settings in force
        while fitting and scoring (``chi2="auto"`` terms stay direct unless you
        set ``toeplitz_min_points``). Keep them the same in every task of a
        process: they are read when a function is first traced.
    dtype : {"float64", "float32"}, optional
        Precision of the fit and the scoring.
    fit_options : dict, optional
        Further arguments of ``fit`` (``method``, ``max_steps``, ...).

    Returns
    -------
    dict
        JSON-friendly record: ``setting``, ``fold``, ``score`` (−ln L of the
        held-out points), ``chi2`` (raw, s = 1), ``n_test``, ``n_train``,
        ``loss``, ``converged``, ``steps``, ``stop``, and with ``groups``
        ``group_labels``, ``group_score``, ``group_chi2``, ``group_n``.
    """
    fold = onp.asarray(fold)
    n = fold.size
    train, test = fold_masks(fold, k)
    options = dict(fit_options or {})
    # deliberate: fits stay direct under chi2="auto" (a precompute per fold)
    with _ft({"toeplitz_min_points": float("inf"), **(fit_ft or {})}):
        result = fit(
            **fit_kwargs(setting),
            likelihoods=[make_term(train)],
            init=init,
            dtype=dtype,
            **options,
        )
    values = result.values
    ft_key = tuple(sorted((score_ft or {}).items()))

    def evaluate(term, vals):
        with run_in(dtype):
            residual, log_norm = _evaluate(term, vals, ft_key)
        return onp.asarray(residual), float(log_norm)

    with _ft(score_ft):
        held_out = make_term(test)
        residual, log_norm = evaluate(held_out, values)
        if residual.size % n:
            raise ValueError(
                f"the term returned {residual.size} residuals for {n} "
                "points; they must be whole blocks of N."
            )
        r2 = float(onp.sum(residual.astype(onp.float64) ** 2))
        raw = None
        if unit_scale is not None:
            raw_residual, _ = evaluate(held_out, unit_scale(values))
            raw = float(onp.sum(raw_residual.astype(onp.float64) ** 2))
        record = {
            "setting": _jsonable(setting),
            "fold": int(k),
            "score": 0.5 * r2 + log_norm,
            "chi2": raw,
            "n_test": int(test.sum()),
            "n_train": int(train.sum()),
            "loss": float(result.info["loss"]),
            "converged": result.info["converged"],
            "steps": int(result.info["steps"]),
            "stop": result.info["stop"],
        }
        if groups is not None:
            groups = onp.asarray(groups)
            labels, ids = onp.unique(groups[test], return_inverse=True)
            # A group's normalization is that of the term on its points only.
            group_norm = onp.zeros(len(labels))
            if getattr(held_out, "has_log_norm", False):
                for i, label in enumerate(labels):
                    _, group_norm[i] = evaluate(
                        make_term(test & (groups == label)), values
                    )
            # Points outside the held-out fold have zero residual.
            residual_test = residual.reshape(-1, n)[:, test]
            sums = _sums(residual_test, test.sum(), labels, ids)
            raw_sums = [None] * len(labels)
            if raw is not None:
                raw_sums = _sums(
                    raw_residual.reshape(-1, n)[:, test],
                    test.sum(),
                    labels,
                    ids,
                )
            record.update(
                group_labels=_jsonable(labels),
                group_score=_jsonable(0.5 * sums + group_norm),
                group_chi2=_jsonable(raw_sums),
                group_n=_jsonable(onp.bincount(ids, minlength=len(labels))),
            )
    return record


def cross_validate(settings, n_folds, fold, **task_kwargs):
    """Run every (setting, fold) in this process and aggregate the records.

    ``task_kwargs`` are those of [`cv_task`][virgil.crossval.cv_task] after
    ``setting``, ``k`` and ``fold``. ``init`` may also be a function of the
    setting. ``aggregate`` options (``larger_is_simpler``) go in a
    ``aggregate_options`` dict.

    Returns
    -------
    dict
        [`aggregate`][virgil.crossval.aggregate]'s summary, with the records
        under ``"records"``.
    """
    options = task_kwargs.pop("aggregate_options", {})
    init = task_kwargs.pop("init", None)
    records = [
        cv_task(
            s,
            k,
            fold,
            init=init(s) if callable(init) else init,
            **task_kwargs,
        )
        for s in settings
        for k in range(n_folds)
    ]
    return {**aggregate(records, **options), "records": records}


def save_record(record, path):
    """Write one task's record as JSON."""
    with open(path, "w") as f:
        json.dump(record, f)


def load_records(paths):
    """Read records from JSON files holding a record or a list of them."""
    records = []
    for path in paths:
        with open(path) as f:
            loaded = json.load(f)
        records.extend(loaded if isinstance(loaded, list) else [loaded])
    return records


def aggregate_files(paths, **options):
    """[`aggregate`][virgil.crossval.aggregate] the records in JSON files."""
    return aggregate(load_records(paths), **options)


def aggregate(records, larger_is_simpler=True):
    """Combine the records of every (setting, fold) into the CV curve.

    Parameters
    ----------
    records : sequence of dict
        Records of [`cv_task`][virgil.crossval.cv_task] covering each
        (setting, fold) pair exactly once.
    larger_is_simpler : bool, optional
        Whether a larger setting is the more regularized (the default, for a
        penalty weight λ), for the one-standard-error choice.

    Returns
    -------
    dict
        ``settings`` (sorted when all numbers, else in order of appearance),
        ``folds``; ``fold_scores`` of shape ``(n_settings, n_folds)``;
        ``score`` (the sum over folds), ``mean`` and ``sd`` (over folds, ``ddof
        = 1``), ``se`` (``sd/√n_folds``); ``chi2``, the raw χ² summed over folds
        (``None`` if absent); ``best`` (index of the smallest mean; a setting whose mean is NaN ranks
        last, with a warning),
        ``one_se`` (index of the most regularized setting whose mean is within
        one ``se`` of the best's) and ``edge``, true if either is the first or
        last of a numeric grid; and, with groups, ``groups`` (labels) and
        ``group_score``, ``group_chi2`` and ``group_n`` of shape
        ``(n_settings, n_groups)`` summed over folds, with ``group_best``,
        the index of each group's smallest score (-1 for a group whose
        scores are all NaN). A group's curve can differ
        from the total: it shows which part of the data prefers which setting.
        Without a numeric grid, ``one_se`` and ``edge`` are ``None``.

    Raises
    ------
    ValueError
        If there are no records, records are missing or duplicated, or every
        setting has a NaN score.
    """
    records = list(records)
    if not records:
        raise ValueError("no records.")
    keys = {}
    for r in records:
        keys.setdefault(json.dumps(r["setting"]), r["setting"])
    settings = list(keys.values())
    numeric = all(
        isinstance(s, Real) and not isinstance(s, bool) for s in settings
    )
    if numeric:
        settings.sort()
    folds = sorted({r["fold"] for r in records})
    index = {json.dumps(s): i for i, s in enumerate(settings)}
    find = {f: j for j, f in enumerate(folds)}
    # Sums run in a fixed order, so the result does not depend on file order.
    records.sort(key=lambda r: (index[json.dumps(r["setting"])], r["fold"]))
    shape = (len(settings), len(folds))
    scores = onp.full(shape, onp.nan)
    chi2 = onp.full(shape, onp.nan)
    seen = set()
    for r in records:
        i, j = index[json.dumps(r["setting"])], find[r["fold"]]
        if (i, j) in seen:
            raise ValueError(
                f"duplicate record for setting {r['setting']!r}, fold "
                f"{r['fold']}."
            )
        seen.add((i, j))
        scores[i, j] = r["score"]
        chi2[i, j] = onp.nan if r["chi2"] is None else r["chi2"]
    if len(seen) != scores.size:
        missing = [
            (settings[i], folds[j])
            for i in range(shape[0])
            for j in range(shape[1])
            if (i, j) not in seen
        ]
        raise ValueError(f"missing (setting, fold) records: {missing}.")
    mean = scores.mean(axis=1)
    sd = (
        scores.std(axis=1, ddof=1)
        if len(folds) > 1
        else onp.full(len(settings), onp.nan)
    )
    se = sd / math.sqrt(len(folds))
    bad = onp.isnan(mean)
    if bad.all():
        raise ValueError("every setting has a NaN score.")
    if bad.any():
        warnings.warn(
            f"NaN scores for settings {[settings[i] for i in onp.flatnonzero(bad)]}"
            "; ranked last.",
            stacklevel=2,
        )
    best = int(onp.nanargmin(mean))
    out = {
        "settings": settings,
        "folds": folds,
        "fold_scores": scores,
        "score": scores.sum(axis=1),
        "mean": mean,
        "sd": sd,
        "se": se,
        "chi2": None if onp.isnan(chi2).any() else chi2.sum(axis=1),
        "best": best,
        "one_se": None,
        "edge": None,
    }
    if numeric:
        tol = se[best] if onp.isfinite(se[best]) else 0.0  # one fold: no se
        within = onp.flatnonzero(mean <= mean[best] + tol)
        # settings is sorted: the most regularized is an end of ``within``.
        one_se = int(within[-1] if larger_is_simpler else within[0])
        out["one_se"] = one_se
        last = len(settings) - 1
        out["edge"] = bool({best, one_se} & {0, last})
    if "group_labels" in records[0]:
        out.update(_group_curves(records, index, find, shape[0]))
    return out


# deliberate: sums the per-fold group scores into per-setting, per-group tables and picks each group's best setting
def _group_curves(records, index, find, n_settings):
    labels = sorted({g for r in records for g in r["group_labels"]})
    col = {g: i for i, g in enumerate(labels)}
    score = onp.zeros((n_settings, len(labels)))
    chi2 = onp.zeros((n_settings, len(labels)))
    count = onp.zeros((n_settings, len(labels)), dtype=int)
    has_chi2 = all(
        r["group_chi2"] and None not in r["group_chi2"] for r in records
    )
    for r in records:
        i = index[json.dumps(r["setting"])]
        for g, s, c, m in zip(
            r["group_labels"],
            r["group_score"],
            r["group_chi2"],
            r["group_n"],
        ):
            score[i, col[g]] += s
            chi2[i, col[g]] += c if has_chi2 else 0.0
            count[i, col[g]] += m
    return {
        "groups": labels,
        "group_score": score,
        "group_chi2": chi2 if has_chi2 else None,
        "group_n": count,
        "group_best": onp.where(
            onp.isnan(score).all(axis=0),
            -1,
            onp.argmin(onp.where(onp.isnan(score), onp.inf, score), axis=0),
        ),
    }
