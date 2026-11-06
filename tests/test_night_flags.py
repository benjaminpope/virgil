"""Night flags that inform rather than exclude, the night-outlier mixture of
the orbit likelihood and the post-fit sidelobe flag
(design/orbit_noise_models.md, "tempered night flags")."""

import dataclasses
import json
import warnings

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import aliases  # noqa: E402
from virgil._precision import cast_tree  # noqa: E402
from virgil.aliases import fit_orbit_aliases  # noqa: E402
from virgil.epochs import (  # noqa: E402
    Epochs,
    NightFlag,
    broad_loglike,
    epoch_positions,
    sidelobe_flags,
)
from virgil.models import BinaryModelCartesian  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from tests import test_aliases as ta  # noqa: E402
from tests.test_epochs import _binary_night, _one_night_flags  # noqa: E402
from tests.test_epochs import T_REF as EPOCH_T_REF  # noqa: E402


def _catalogue(dra, ddec, flux=0.3):
    from virgil.epochs import EpochPeaks

    return EpochPeaks(
        dra=onp.array([dra, 9.0]),
        ddec=onp.array([ddec, 9.0]),
        flux=onp.array([flux, flux]),
        loglike=onp.zeros(2),
        marginal=onp.array([0.0, -50.0]),
        weight=onp.ones(2) / 2,
        edge=onp.zeros(2, bool),
        laplace=onp.ones(2, bool),
    )


# --- 1. sep-below-resolution is reported, not excluding ---------------------


def test_sep_below_resolution_is_reported_but_does_not_block():
    inside = _catalogue(0.5, 0.2)  # resolution 4 mas, limit 2 mas
    (flag,) = _one_night_flags(inside)
    assert flag.code == "sep-below-resolution" and not flag.blocking
    (hard,) = _one_night_flags(inside, sep_blocks=True)
    assert hard.blocking


def test_a_night_with_only_non_blocking_flags_is_not_flagged():
    from virgil.epochs import EpochPositions

    names = ("a", "b")
    found = EpochPositions(
        names=names,
        mjd=onp.zeros(2),
        dra=onp.zeros(2),
        ddec=onp.zeros(2),
        cov=onp.zeros((2, 2, 2)),
        flux=onp.ones(2),
        gap_marginal=onp.ones(2),
        chi2_raw=({}, {}),
        scale=({}, {}),
        flags=(
            (NightFlag("sep-below-resolution", "r", blocking=False),),
            (NightFlag("sidelobe", "r"),),
        ),
    )
    assert found.flagged.tolist() == [False, True]
    assert found.flag_codes() == {
        "a": ["sep-below-resolution"],
        "b": ["sidelobe"],
    }


# --- 2. the outlier mixture --------------------------------------------------

AXIS = onp.arange(-20.0, 20.01, 0.5)
GRID = {"dra": AXIS, "ddec": AXIS, "flux": [0.1, 0.3, 0.6]}
BAD = 3  # the night with the sidelobe


def _with_bad_night(shift, n_nights=4):
    """The synthetic orbit's first nights, the BAD one replaced by a binary
    ``shift`` mas north of the orbit's position there (a sidelobe)."""
    data, _ = ta._simulate_orbit(12.1)
    data = list(data)[:n_nights]
    orbit = ta.KeplerOrbit(**ta.TRUTH, t_ref=ta.T_REF)
    dra, ddec, _ = orbit.relative(ta.EPOCHS)
    data[BAD] = simulate(
        BinaryModelCartesian(
            float(dra[BAD]), float(ddec[BAD]) + shift, ta.FLUX
        ),
        ta._night(ta.EPOCHS[BAD], rotation=7.0 * BAD),
        jax.random.PRNGKey(99),
    )
    return data


def _positions(data):
    epochs = Epochs({f"n{k}": d for k, d in enumerate(data)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return epoch_positions(epochs, GRID, n_peaks=3)


def _broad(data, settings):
    return tuple(
        broad_loglike(
            data,
            _unit_dof(data),
            a_max=settings.a_range[1] * (1 + settings.ecc_max),
            flux_range=settings.flux_range,
            step=1.0,
        )
    )


def _unit_dof(data):
    return aliases._Problem(
        [cast_tree(d, "float64") for d in data],
        ta._settings(outlier="inflation", s_max=50.0),
    ).dof


def _optimum(data, kind="mixture", start=None, off=False):
    """The best-fit orbit of ``data`` in the 12.0-12.2 d band, with the outlier
    component ``kind`` ("mixture": a wide-prior companion; "inflation": the
    night's own model at an inflated noise scale). ``off``: the mixture with
    an outlier level of -1e6, i.e. no outliers, as the reference."""
    data = [cast_tree(d, "float64") for d in data]
    kw = dict(outlier=kind, s_max=50.0)
    if kind == "mixture":
        kw["log_broad"] = (
            (-1e6,) * len(data) if off else _broad(data, ta._settings(**kw))
        )
    settings = ta._settings(**kw)
    problem = aliases._Problem(data, settings)
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    # Known limitation: the default fit starts at the truth, and a start
    # 0.2 prior-scales away stalls 140 nats short (the strict xfail below).
    # An off-truth variant waits for continuation in the outlier rate
    # (mixture v2); until then q ~ 0.7 on a -105 log-L night is what is met.
    x0 = ta._x_truth() if start is None else start
    x, value = aliases._optimize(problem, bounds, x0, 50)
    return problem, onp.asarray(x), value


def _check_one_bad_night(kind):
    near, far = _with_bad_night(3.0), _with_bad_night(10.0)
    clean = _optimum(near[:BAD], kind)[1]
    fits = {k: _optimum(d, kind) for k, d in (("near", near), ("far", far))}
    off = _optimum(near, "mixture", off=True)[1]  # follows the bad night
    for problem, x, _ in fits.values():
        prob = problem.outlier_probability(x)
        # q > 0.99 is out of reach here, and not for want of a bad night: its
        # log likelihood at the orbit is -105 against about -14 for a good
        # night, but the outlier component, a flux-averaged companion anywhere
        # in the prior's disc, scores -102 to -103 (it includes companions as
        # faint as the null). The night is only ln(L/B) ~ 3 nats below it, so
        # with the Beta(1/2, 1/2) rate q ~ 0.7-0.8 (measured 0.83, 0.69). The
        # mixture's job is to give the night half the weight at least, and the
        # good nights none (measured 1e-31); the orbit checks below are the
        # test that it does not follow the night.
        assert prob[BAD] > 0.5 and prob[:BAD].max() < 1e-6
    # The orbit is the one the good nights give, wherever the bad night is,
    # and nearer to it than the fit that follows the bad night.
    near_x, far_x = fits["near"][1], fits["far"][1]
    # Not 1e-3: the night keeps weight 1 - q ~ 0.2-0.3 (above), so the two
    # orbits differ by a little (measured 8e-3 at most, in a parameter of
    # order 100).
    assert onp.abs(near_x - far_x).max() < 0.02
    assert onp.abs(near_x - clean).max() < 0.5 * onp.abs(off - clean).max()


@pytest.mark.slow
def test_one_sidelobe_night_cannot_move_the_orbit():
    _check_one_bad_night("mixture")


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the inflated component's support [1, s_max] lies inside the inlier's "
        "scale prior [1/s_max, s_max], so B/L <= 2**blocks and no night can "
        "be separated (review of #586); candidate (iii) needs a redesign"
    ),
)
def test_one_sidelobe_night_cannot_move_the_orbit_with_inflation():
    _check_one_bad_night("inflation")


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "the M4 local-optimum risk of design/orbit_dehack.md, measured: from "
        "starts 0.2 prior-scales off the truth the mixture fit stalls 140 "
        "nats below its optimum (-309 against -169), because the redescending "
        "weights switch off the nights the start misses. Proposed fix: "
        "continuation in the outlier rate, not a second path"
    ),
)
def test_the_mixture_keeps_the_optimum_from_starts_near_the_truth():
    # The redescending mixture gives no weight to a night a start misses
    # (the M4 risk of design/orbit_dehack.md): from starts off by a little,
    # the fit must still reach the optimum that it reaches from the truth.
    data = _with_bad_night(10.0, n_nights=6)
    optimum = _optimum(data)[2]
    rng = onp.random.default_rng(0)
    scale = onp.array([0.01, 0.2, 0.3, 0.3, 10, 10, 0.3, 0.3])
    truth = onp.asarray(ta._x_truth())
    starts = [
        truth + 0.2 * rng.standard_normal(len(truth)) * scale for _ in range(4)
    ]
    for start in starts:
        assert _optimum(data, start=start)[2] > optimum - 10.0


@ta.in_float64
def test_tempered_nights_the_orbit_fits_are_not_outliers():
    # Block-wise tempering (V^2 by f = 5, closure phases by f = 2): the
    # outlier component is scored with it, so a night the orbit fits is not
    # an outlier. (One factor for all blocks would put the two components
    # thousands of nats apart.)
    clean = [cast_tree(d, "float64") for d in _with_bad_night(3.0)[:BAD]]
    tempered = (("vis", 1 / 5.0), ("phi", 1 / 2.0))
    base = ta._settings(dof=tempered, s_max=50.0)
    dof = aliases._Problem(
        clean, dataclasses.replace(base, outlier="inflation")
    ).dof
    broad = broad_loglike(
        clean,
        dof,
        a_max=base.a_range[1] * (1 + base.ecc_max),
        flux_range=base.flux_range,
        step=1.0,
    )
    settings = dataclasses.replace(base, log_broad=tuple(map(float, broad)))
    problem = aliases._Problem(clean, settings)
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    x, _ = aliases._optimize(problem, bounds, ta._x_truth(), 50)
    assert problem.outlier_probability(onp.asarray(x)).max() < 0.01


@ta.in_float64
def test_the_broad_component_is_a_prior_average_of_the_night_likelihood():
    data = [cast_tree(d, "float64") for d in _with_bad_night(3.0)[:2]]
    dof = _unit_dof(data)
    kw = dict(a_max=30.0, flux_range=(0.01, 1.0), step=1.0)
    base = onp.asarray(broad_loglike(data, dof, **kw))
    assert base.shape == (2,) and onp.isfinite(base).all()
    # The flux axis is a quadrature: refining it moves the level by about a
    # nat, against the hundreds of nats by which a bad night misses the orbit.
    fine = onp.asarray(broad_loglike(data, dof, n_flux=25, **kw))
    assert onp.abs(fine - base).max() < 2.0
    # A wider prior spreads the same likelihood over a larger area, so the
    # mean cannot rise, and a point-like peak means it falls.
    wide = onp.asarray(broad_loglike(data, dof, **{**kw, "a_max": 60.0}))
    assert (wide <= base + 1e-9).all()


@ta.in_float64
def test_a_mixture_problem_needs_its_broad_component():
    with pytest.raises(ValueError, match="log_broad"):
        aliases._Problem(
            [cast_tree(d, "float64") for d in _with_bad_night(3.0)[:2]],
            ta._settings(outlier="mixture"),
        )


@pytest.mark.slow
def test_fit_orbit_aliases_reports_each_nights_outlier_probability(tmp_path):
    data = _with_bad_night(10.0, n_nights=6)
    found = _positions(data)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=found,
        t_ref=ta.T_REF,
        primary_diam=None,
        temper=False,
        # A warm start keeps the search itself out of this test: it is about
        # what the fit does with the bad night.
        init={**ta.TRUTH, "flux": ta.FLUX},
        **ta.FAST,
    )
    assert result.outlier["kind"] == "mixture"
    prob = dict(zip(result.outlier["names"], result.outlier["probability"]))
    assert prob[f"n{BAD}"] > 0.5  # see _check_one_bad_night: q ~ 0.7-0.8
    assert max(v for k, v in prob.items() if k != f"n{BAD}") < 1e-6
    assert abs(result.best.best["period"] - 12.1) < 0.1
    result.save(tmp_path / "r.json")
    loaded = aliases.AliasResult.load(tmp_path / "r.json")
    assert loaded.outlier == json.loads(json.dumps(result.outlier))


def test_outlier_eps_is_deprecated_and_ignored():
    with pytest.warns(FutureWarning, match="outlier_eps"):
        with pytest.raises(Exception):  # no data: stops after the warning
            fit_orbit_aliases([], (11.0, 13.0), outlier_eps=0.02)


# --- 4. the post-fit sidelobe flag ------------------------------------------


@pytest.fixture(scope="module")
def twin():
    # flux 1: the mirror peak ties with the true one.
    night = _binary_night(
        EPOCH_T_REF,
        (5.0, -3.0),
        1.0,
        jax.random.PRNGKey(1),
        d_vis=0.01,
        d_phi=0.02,
    )
    epochs = Epochs({"twin": night})
    axis = onp.arange(-12.0, 12.01, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        found = epoch_positions(
            epochs, {"dra": axis, "ddec": axis, "flux": [0.5, 1.0]}, n_peaks=3
        )
    return epochs, found


@pytest.mark.slow
def test_an_orbit_on_the_secondary_peak_is_flagged_not_excluded(twin):
    epochs, found = twin
    x, y = float(found.dra[0]), float(found.ddec[0])
    flux = [float(found.flux[0])]
    at_best = sidelobe_flags(epochs, found, ([x], [y], flux))
    mirror = sidelobe_flags(epochs, found, ([-x], [-y], flux))
    elsewhere = sidelobe_flags(epochs, found, ([x + 8.0], [y], flux))
    assert at_best == [None]  # on the best peak
    assert mirror[0].code == "orbit-prefers-secondary"
    assert not mirror[0].blocking
    assert elsewhere == [None]  # far from every peak: not a secondary peak


@pytest.mark.slow
def test_sub_resolution_nights_are_located_and_still_seed():
    # A binary whose nights all sit below lambda/2B (the 5 mas resolution
    # here puts every night inside).
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    spots = [(0.9, 0.05), (-1.4, -0.1), (1.7, 0.1)]
    nights = {
        f"n{k}": _binary_night(
            EPOCH_T_REF + k, spots[k], 0.3, keys[k], d_vis=0.005, d_phi=0.02
        )
        for k in range(3)
    }
    axis = onp.arange(-6.0, 6.01, 0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        found = epoch_positions(
            Epochs(nights),
            {"dra": axis, "ddec": axis, "flux": [0.2, 0.3, 0.5]},
        )
    for k, spot in enumerate(spots):
        assert onp.hypot(found.dra[k] - spot[0], found.ddec[k] - spot[1]) < 0.3
    assert (
        all(
            "sep-below-resolution" in codes
            for codes in found.flag_codes().values()
        )
        and len(found.flag_codes()) == 3
    )
    assert not found.flagged.any()  # reported, nobody excluded
    kept = found.positions(t_ref=EPOCH_T_REF)
    assert kept.dra.shape == (3,)


def test_a_companion_above_flux_one_is_compared_as_the_reflected_star(twin):
    epochs, found = twin
    x, y = float(found.dra[0]), float(found.ddec[0])
    f = float(found.flux[0])
    # The same scene with the stars' labels swapped: (-r, 1/f).
    swapped = sidelobe_flags(epochs, found, ([-x], [-y], [1.0 / f]))
    assert swapped == [None]


@ta.in_float64
def test_the_inflated_surface_has_the_base_prior_mass_on_s_above_one():
    from virgil.epochs import _Surface

    data = cast_tree(ta._night(ta.T_REF), "float64")
    base = _Surface(data, s_max=10.0)
    twin = base.inflated
    # Residuals so large that the error scale is well above 1: both integrals
    # then cover the whole peak, and the base prior (log-uniform on
    # [1/s_max, s_max]) has half the density of the twin's ([1, s_max]).
    chi2 = onp.full(len(base.names), 1e3)
    diff = float(twin.score(chi2, 1.0, data)) - float(
        base.score(chi2, 1.0, data)
    )
    assert diff == pytest.approx(len(base.blocks) * onp.log(2.0), rel=1e-6)
    with pytest.raises(ValueError, match="s_max"):
        _Surface(data).inflated


@ta.in_float64
def test_the_outlier_component_is_built_per_kind_and_must_be_finite():
    data = [cast_tree(d, "float64") for d in _with_bad_night(3.0)[:2]]
    mixture = ta._settings(outlier="mixture", s_max=50.0)
    built = aliases._broad_components(data, mixture, None, None, 4.0)
    assert len(built.log_broad) == 2 and onp.isfinite(built.log_broad).all()
    inflation = ta._settings(outlier="inflation", s_max=50.0)
    assert (
        aliases._broad_components(data, inflation, None, None, 4.0)
        is inflation
    )
    with pytest.raises(ValueError, match="s_max"):
        aliases._broad_components(
            data, ta._settings(outlier="inflation"), None, None, 4.0
        )
    with pytest.raises(ValueError, match="outlier must be"):
        aliases._broad_components(
            data, ta._settings(outlier="none"), None, None, 4.0
        )
