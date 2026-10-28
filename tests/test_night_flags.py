"""Night flags that inform rather than exclude, per-mode χ², the outlier
mixture of the orbit likelihood and the post-fit sidelobe flag
(design/orbit_noise_models.md, "tempered night flags")."""

import dataclasses
import inspect
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
    _flag_chi2,
    _mode_keys,
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
            (NightFlag("chi2-high", "r"),),
        ),
    )
    assert found.flagged.tolist() == [False, True]
    assert found.flag_codes() == {
        "a": ["sep-below-resolution"],
        "b": ["chi2-high"],
    }


# --- 2. chi2-high within an instrument mode ---------------------------------


def test_chi2_high_is_judged_within_each_mode():
    # Two instrument modes with different baseline chi2/N (around 10 vs 40):
    # a global median of 10 flags the second mode, whereas within each mode
    # (>= 3 nights) none is an outlier.
    chi2 = [9.0, 10.0, 11.0, 10.0, 35.0, 40.0, 45.0]
    mode = ["A"] * 4 + ["B"] * 3
    assert sum(bool(f) for f in _flag_chi2(chi2, 3.0)) == 3
    assert not any(_flag_chi2(chi2, 3.0, mode))
    # Even when mode B has only 2 well-fitted nights (>= 3x mode A's), its
    # lower night floors the fallback median so neither is flagged.
    assert not any(_flag_chi2(chi2[:-1], 3.0, mode[:-1]))
    # A real outlier inside a mode is still found.
    chi2[5] = 400.0
    found = _flag_chi2(chi2, 3.0, mode)
    assert [bool(f) for f in found] == [False] * 5 + [True, False]
    assert "of its 3 nights of the same mode" in found[5][0].reason
    # A mode with fewer than three nights falls back to the pooled median
    # across all nights, so a corrupted night in a 1- or 2-night mode is
    # still caught while normal nights pass.
    pooled = _flag_chi2([1.0, 500.0, 1.0, 1.0, 1.0], 3.0, list("aabbb"))
    assert [bool(f) for f in pooled] == [False, True, False, False, False]
    assert "pooled across 5 nights" in pooled[1][0].reason


def test_mode_keys_tolerate_wavelength_jitter_but_not_other_setups():
    one = ta._night(ta.T_REF)
    jitter = one.set("wavel", one.wavel * 1.004)
    other = one.set("wavel", one.wavel * 0 + 1.6e-6)
    keys = _mode_keys(Epochs({"a": one, "b": jitter, "c": other, "d": one}))
    assert keys[0] == keys[1] == keys[3] != keys[2]


def test_mode_keys_separate_spectral_setups():
    one = ta._night(ta.T_REF)
    two = one.set("wavel", one.wavel * 0 + 1.6e-6)
    keys = _mode_keys(Epochs({"a": one, "b": one, "c": two}))
    assert keys[0] == keys[1] != keys[2]


# --- 3. the outlier mixture --------------------------------------------------

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


def _optimum(data, eps=0.0, log_broad=()):
    kw = dict(outlier_eps=eps, log_broad=tuple(log_broad)) if eps else {}
    settings = ta._settings(**kw)
    problem = aliases._Problem(
        [cast_tree(d, "float64") for d in data], settings
    )
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    x, value = aliases._optimize(problem, bounds, ta._x_truth(), 50)
    return problem, onp.asarray(x), value


@pytest.mark.slow
def test_one_sidelobe_night_cannot_move_the_orbit_with_the_mixture():
    near, far = _with_bad_night(3.0), _with_bad_night(10.0)
    clean = _optimum(near[:BAD])[1]
    found = {k: _positions(d) for k, d in (("near", near), ("far", far))}
    plain = {k: _optimum(d)[1] for k, d in (("near", near), ("far", far))}
    mixed = {}
    for k, d in (("near", near), ("far", far)):
        problem, x, value = _optimum(d, 0.02, found[k].log_broad)
        mixed[k] = x
        prob = problem.outlier_probability(x)
        assert prob[BAD] > 0.99 and prob[:BAD].max() < 0.01
        assert value > _optimum(d)[2] + 20.0  # the night costs a cap, not 100s
    # Without the mixture the orbit follows the bad night; with it the
    # orbit is the one the good nights give, wherever the bad night is.
    assert onp.abs(plain["near"] - plain["far"]).max() > 0.02
    assert onp.abs(mixed["near"] - mixed["far"]).max() < 1e-3
    assert (
        onp.abs(mixed["near"] - clean).max()
        < 0.5 * onp.abs(plain["near"] - clean).max()
    )


@pytest.mark.slow  # 36-37 s on a CI runner, over the 30 s budget
@ta.in_float64
def test_the_mixture_polishes_the_modes_the_plain_search_finds():
    # Gl 229 gate regression: the mixture's IRLS weights vanish on the
    # nights a start misses, so a mixture fit from a start that is off by a
    # little settles on the nights it already fits, 100s of nats short of
    # the optimum that the plain fit reaches from the same start. The
    # search refines on the plain likelihood and only polishes with the
    # mixture, so a basin the plain search finds is not hidden by it.
    data, _ = ta._simulate_orbit(12.1)
    data = list(data)
    found = _positions(data)
    settings = ta._settings(
        outlier_eps=0.02,
        log_broad=tuple(found.log_broad),
        n_refine=3,
        max_steps=50,
    )
    problem = aliases._Problem(
        [cast_tree(d, "float64") for d in data], settings
    )
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 11.9, 12.3), settings)
    truth = onp.asarray(ta._x_truth())
    optimum = float(onp.asarray(problem.batch(truth[None]))[0])
    # Starts near the truth (seeded perturbations in the orbit's elements)
    # that the mixture fit alone leaves 160 to 330 nats short.
    rng = onp.random.default_rng(0)
    scale = onp.array([0.01, 0.2, 0.3, 0.3, 10, 10, 0.3, 0.3])
    starts = [
        truth + 0.2 * rng.standard_normal(len(truth)) * scale for _ in range(4)
    ]
    stalls = [
        aliases._optimize(problem, bounds, x, 50)[1] < optimum - 100.0
        for x in starts
    ]
    assert any(stalls), "the starts no longer exercise the failure"
    modes, failed = aliases._refine(
        problem, settings, bounds, onp.asarray(starts), ta.EPOCHS
    )
    assert not failed
    best = max(v for _, v in modes)
    assert best > optimum - 10.0  # the polished mode is the optimum
    # The mixture is applied to the mode: it is scored under the mixture.
    x = max(modes, key=lambda m: m[1])[0]
    assert onp.isclose(
        best, float(onp.asarray(problem.batch(onp.asarray(x)[None]))[0])
    )


def test_the_broad_component_shares_the_orbit_likelihoods_scale():
    from virgil.epochs import broad_loglike

    data = _with_bad_night(3.0)[:2]
    found = _positions(data)
    untempered = [onp.ones(2), onp.ones(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = broad_loglike(data, found.grid, untempered)
    assert onp.allclose(again, found.log_broad)

    # Block-wise tempering (V^2 by f = 5, closure phases by f = 2): the
    # broad component is rescored with it, so a night the orbit fits is
    # not an outlier. (One factor for all blocks would put the two
    # components thousands of nats apart.)
    tempered = (("vis", 1 / 5.0), ("phi", 1 / 2.0))
    clean = _with_bad_night(3.0)[:BAD]
    settings = ta._settings(dof=tempered)
    problem = aliases._Problem(
        [cast_tree(d, "float64") for d in clean], settings
    )
    broad = broad_loglike(problem.data, _positions(clean).grid, problem.dof)
    settings = dataclasses.replace(
        settings, outlier_eps=0.02, log_broad=tuple(broad)
    )
    problem = aliases._Problem(problem.data, settings)
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    x, _ = aliases._optimize(problem, bounds, ta._x_truth(), 50)
    assert problem.outlier_probability(x).max() < 0.01
    # The null model is scored without the mixture.
    xs = onp.asarray(x)[None]
    plain = aliases._Problem(problem.data, ta._settings(dof=tempered))
    assert onp.allclose(
        onp.asarray(problem.batch(xs, mixed=False)),
        onp.asarray(plain.batch(xs)),
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
        outlier_eps=0.02,
        # A warm start keeps the search itself out of this test: it is about
        # what the fit does with the bad night.
        init={**ta.TRUTH, "flux": ta.FLUX},
        **ta.FAST,
    )
    prob = dict(zip(result.outlier["names"], result.outlier["probability"]))
    assert prob[f"n{BAD}"] > 0.99
    assert max(v for k, v in prob.items() if k != f"n{BAD}") < 0.05
    assert abs(result.best.best["period"] - 12.1) < 0.1
    result.save(tmp_path / "r.json")
    loaded = aliases.AliasResult.load(tmp_path / "r.json")
    assert loaded.outlier == json.loads(json.dumps(result.outlier))
    # Off by default in the engine: nothing reported.
    default = inspect.signature(fit_orbit_aliases).parameters["outlier_eps"]
    assert default.default == 0.0


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
    # AK For-like: an edge-on binary whose nights sit below lambda/2B
    # (a = 1.83 mas; the 5 mas resolution here puts every night inside).
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
