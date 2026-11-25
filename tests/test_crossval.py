import json

import equinox as eqx
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import CalibrationNuisance, ComplexVisData, Image
from virgil._geometry import _FT_DEFAULTS
from virgil.crossval import (
    aggregate,
    aggregate_files,
    cross_validate,
    cv_task,
    fold_masks,
    save_record,
)
from virgil.imaging import MaxEntropy
from virgil.radio import dartboard_folds

from ._compiles import count_compiles

SIGMA = 0.5
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
P, N = 12, 72


class RidgeTerm(eqx.Module):
    """Whitened residuals of y = X beta at the unmasked points."""

    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray

    has_log_norm = False

    def __call__(self, values):
        return self.mask * (self.y - self.x @ values["beta"]) / SIGMA


@pytest.fixture(scope="module")
def ridge():
    rng = onp.random.default_rng(3)
    x = rng.normal(size=(N, P))
    beta = rng.normal(size=P) * 0.5
    y = x @ beta + SIGMA * rng.normal(size=N)
    fold = onp.arange(N) % 4
    return x, y, fold


def _model(**values):  # one function object, so that fits share a compilation
    return None


def _kwargs(setting):
    # Ridge as a Normal prior of scale 1/sqrt(lambda), whitened by SIGMA
    return dict(
        model=_model,
        priors={"beta": dist.Normal(np.zeros(P), 1.0 / np.sqrt(setting))},
        data=(),
    )


def _task_kwargs(x, y, **extra):
    return dict(
        fit_kwargs=_kwargs,
        make_term=lambda m: RidgeTerm(x, y, np.asarray(m, float)),
        init={"beta": onp.zeros(P)},
        **extra,
    )


def _closed_form(x, y, fold, lam, k):
    train = fold != k
    xt, yt = x[train], y[train]
    a = xt.T @ xt / SIGMA**2 + lam * onp.eye(P)
    beta = onp.linalg.solve(a, xt.T @ yt / SIGMA**2)
    r = (y[~train] - x[~train] @ beta) / SIGMA
    return 0.5 * onp.sum(r**2)


def test_masks_are_disjoint_and_cover(ridge):
    _, _, fold = ridge
    seen = onp.zeros(N, int)
    for k in range(4):
        train, test = fold_masks(fold, k)
        assert not (train & test).any()
        assert (train | test).all()
        seen += test
    assert (seen == 1).all()
    with pytest.raises(ValueError, match="no points"):
        fold_masks(fold, 9)


def test_ridge_cv_matches_closed_form_and_one_se(ridge):
    x, y, fold = ridge
    out = cross_validate(
        LAMBDAS, 4, fold, **_task_kwargs(x, y), fit_options={"method": "lm"}
    )
    expected = onp.array(
        [
            [_closed_form(x, y, fold, lam, k) for k in range(4)]
            for lam in LAMBDAS
        ]
    )
    onp.testing.assert_allclose(out["fold_scores"], expected, rtol=1e-5)
    best = int(onp.argmin(expected.mean(axis=1)))
    assert out["best"] == best
    assert 0 < best < len(LAMBDAS) - 1 and not out["edge"]
    # the one-SE rule prefers a larger (more regularized) lambda
    assert out["one_se"] > out["best"]
    assert out["mean"][out["one_se"]] <= out["mean"][best] + out["se"][best]
    onp.testing.assert_allclose(out["score"], expected.sum(axis=1), rtol=1e-5)


def test_groups_and_edge_flag(ridge):
    x, y, fold = ridge
    groups = onp.arange(N) % 3
    record = cv_task(
        1.0,
        0,
        fold,
        **_task_kwargs(x, y, groups=groups),
        fit_options={"method": "lm"},
    )
    assert record["chi2"] is None
    assert sum(record["group_score"]) == pytest.approx(
        record["score"], rel=1e-9
    )
    assert sum(record["group_n"]) == record["n_test"]
    # an optimum at the edge of the grid is flagged
    edge = cross_validate(
        LAMBDAS[:2],
        4,
        fold,
        **_task_kwargs(x, y),
        fit_options={"method": "lm"},
    )
    assert edge["edge"]
    assert json.dumps(record)  # JSON-friendly


def test_array_records_aggregate_to_single_process(ridge, tmp_path):
    x, y, fold = ridge
    kw = _task_kwargs(x, y, groups=onp.arange(N) % 3)
    single = cross_validate(
        LAMBDAS[:3], 4, fold, **kw, fit_options={"method": "lm"}
    )
    paths = []
    for i, s in enumerate(LAMBDAS[:3]):
        for k in range(4):  # one array task each, in any order
            record = cv_task(s, k, fold, **kw, fit_options={"method": "lm"})
            paths.append(tmp_path / f"{i}_{k}.json")
            save_record(record, paths[-1])
    out = aggregate_files(paths[::-1])
    for key in (
        "fold_scores",
        "score",
        "mean",
        "sd",
        "group_score",
        "group_n",
    ):
        onp.testing.assert_array_equal(out[key], single[key])
    assert (out["best"], out["one_se"]) == (single["best"], single["one_se"])
    with pytest.raises(ValueError, match="missing"):
        aggregate_files(paths[1:])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_files(paths + paths[:1])


def test_no_recompiles_across_folds_and_settings(ridge):
    x, y, fold = ridge
    kw = _task_kwargs(x, y, groups=onp.arange(N) % 3)
    opts = {"method": "lm"}
    cv_task(1.0, 0, fold, **kw, fit_options=opts)  # warm up
    with count_compiles() as compiles:
        for s, k in [(10.0, 1), (0.1, 2), (1.0, 3)]:
            cv_task(s, k, fold, **kw, fit_options=opts)
    assert compiles == []


def test_ft_settings_are_restored(ridge):
    x, y, fold = ridge
    before = dict(_FT_DEFAULTS)
    seen = []

    def make_term(m):
        seen.append(_FT_DEFAULTS["eps_f64"])
        return RidgeTerm(x, y, np.asarray(m, float))

    cv_task(
        1.0,
        0,
        fold,
        fit_kwargs=_kwargs,
        make_term=make_term,
        init={"beta": onp.zeros(P)},
        fit_ft={"eps_f64": 1e-5},
        score_ft={"eps_f64": 1e-13},
        fit_options={"method": "lm"},
    )
    assert seen == [1e-5, 1e-13]
    assert dict(_FT_DEFAULTS) == before


@pytest.mark.filterwarnings("ignore:.*did not converge:RuntimeWarning")
def test_radio_smoke():
    rng = onp.random.default_rng(0)
    n, npix, pix = 4000, 32, 2.0
    u, v = rng.uniform(-2e6, 2e6, n), rng.uniform(-2e6, 2e6, n)
    sigma = 0.05
    r = onp.hypot(u, v)
    vis = 1.0 * onp.exp(-0.5 * (r / 1.0e6) ** 2) + sigma * (
        rng.normal(size=n) + 1j * rng.normal(size=n)
    )
    data = ComplexVisData.from_arrays(u, v, vis, onp.full(n, sigma**-2), 230e9)
    nuis = CalibrationNuisance(data, sigmas=True, alpha=False)
    fold, annulus = dartboard_folds(data, k=2, n_q=4, n_phi=6)

    def scene(values):
        return Image(values["log_brightness"], pixel_scale_mas=pix)

    def fit_kwargs(weight):
        priors = nuis.priors()
        priors["log_brightness"] = dist.Normal(np.zeros((npix, npix)), 3.0)
        return dict(
            model=lambda **values: scene(values),
            priors=priors,
            data=(),
            regularizers=[MaxEntropy(weight)],
        )

    init = nuis.defaults()
    init["log_brightness"] = onp.full((npix, npix), -np.log(npix**2))
    unit = lambda values: {
        **values,
        "cal.sigma": np.ones_like(values["cal.sigma"]),
    }  # noqa: E731
    out = cross_validate(
        [1e-3, 1.0],
        2,
        fold,
        fit_kwargs=fit_kwargs,
        make_term=lambda m: data.term(scene, nuis, mask=m),
        init=init,
        groups=annulus,
        unit_scale=unit,
        fit_options={"method": "lbfgs", "max_steps": 30},
    )
    assert out["fold_scores"].shape == (2, 2)
    assert onp.isfinite(out["fold_scores"]).all()
    assert out["chi2"].shape == (2,) and (out["chi2"] > 0).all()
    assert out["group_score"].shape[0] == 2
    onp.testing.assert_allclose(
        out["group_score"].sum(axis=1), out["score"], rtol=1e-6
    )
    assert sum(r["n_test"] for r in out["records"][:2]) == n


def _record(setting, fold, score):
    return {"setting": setting, "fold": fold, "score": score, "chi2": None}


@pytest.mark.parametrize("settings", [[0.1, 1.0, 10.0], ["a", "b", "c"]])
def test_aggregate_ranks_nan_scores_last(settings):
    scores = {settings[0]: onp.nan, settings[1]: 2.0, settings[2]: 3.0}
    records = [_record(s, k, v) for s, v in scores.items() for k in range(3)]
    with pytest.warns(UserWarning, match="NaN"):
        out = aggregate(records)
    assert out["settings"][out["best"]] == settings[1]
    if out["one_se"] is not None:
        assert out["settings"][out["one_se"]] != settings[0]


def test_aggregate_all_nan_scores_raise():
    records = [_record(s, k, onp.nan) for s in (1.0, 2.0) for k in range(2)]
    with pytest.raises(ValueError, match="NaN"):
        aggregate(records)


def test_aggregate_one_fold_has_no_standard_error():
    records = [
        _record(s, 0, v) for s, v in ((0.1, 3.0), (1.0, 1.0), (10.0, 2.0))
    ]
    out = aggregate(records)
    assert out["best"] == 1 and out["one_se"] == 1


def test_aggregate_group_with_only_nan_scores_has_no_best():
    def record(setting, fold):
        return {
            **_record(setting, fold, 1.0 + setting),
            "group_labels": ["g0", "g1"],
            "group_score": [1.0 + setting, onp.nan],
            "group_chi2": [None, None],
            "group_n": [1, 1],
        }

    out = aggregate([record(s, k) for s in (1.0, 2.0) for k in range(2)])
    assert out["group_best"].tolist() == [0, -1]


def test_cv_task_refuses_profiled_sigma():
    rng = onp.random.default_rng(0)
    n, npix = 200, 8
    u, v = rng.uniform(-2e6, 2e6, n), rng.uniform(-2e6, 2e6, n)
    vis = onp.exp(-0.5 * (onp.hypot(u, v) / 1e6) ** 2) + 0j
    data = ComplexVisData.from_arrays(u, v, vis, onp.full(n, 400.0), 230e9)
    nuis = CalibrationNuisance(data, sigmas="profile", alpha=False)
    fold = onp.arange(n) % 2

    def scene(values):
        return Image(values["log_brightness"], pixel_scale_mas=2.0)

    priors = nuis.priors()
    priors["log_brightness"] = dist.Normal(np.zeros((npix, npix)), 3.0)
    init = nuis.defaults()
    init["log_brightness"] = onp.full((npix, npix), -np.log(npix**2))
    task = dict(
        fit_kwargs=lambda w: dict(
            model=lambda **values: scene(values),
            priors=priors,
            data=(),
            regularizers=[MaxEntropy(w)],
        ),
        init=init,
        fit_options={"method": "lbfgs", "max_steps": 3},
    )
    with pytest.raises(ValueError, match="profile"):
        cv_task(
            1.0,
            0,
            fold,
            make_term=lambda m: data.term(scene, nuis, mask=m),
            **task,
        )
    # a fixed σ is a predictive score and still runs
    fixed = lambda m: data.term(scene, nuis, mask=m, sigma=onp.ones(1))  # noqa: E731
    record = cv_task(1.0, 0, fold, make_term=fixed, **task)
    assert onp.isfinite(record["score"])
