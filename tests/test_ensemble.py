import jax
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.ensemble import (
    Draw,
    EnsembleSpec,
    Group,
    combine,
    draw_groups,
    ensemble,
    reference_starts,
    _mixture,
    run_group,
)
from virgil.fitting import FitResult
from virgil.imaging import LCurve, _chi2, field_of_view
from virgil.metrics import score
from virgil.models import Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

from ._compiles import count_compiles

NPIX, SCALE = 12, 24.0
DATA = OIData(ami_grid_record(pitch_m=0.5))


def _scene(sigma=40.0, dra=48.0, flux=0.3):
    blob = gaussian_blob(NPIX, SCALE, sigma, dra=dra)
    image = Image.from_brightness(blob, SCALE, flux=flux)
    return System(star=PointSource(), env=image)


TRUTH = _scene()
# Two noise draws of the same scene stand for two datasets.
DATASETS = [DATA.with_model(TRUTH, key=jax.random.PRNGKey(k)) for k in (1, 2)]


def _group(index, models):
    """A Group of ready-made "fits", weights largest first."""
    weights = onp.array([1e3, 1e2, 1e1][: len(models)])
    results = []
    for model in models:
        chi2 = _chi2(model, DATASETS)
        ndata = [d.n_independent for d in DATASETS]
        results.append(FitResult(model, {}, {"chi2": chi2, "ndata": ndata}))
    chi2 = onp.array([sum(r.info["chi2"]) for r in results])
    curve = LCurve(
        weights,
        chi2,
        onp.array(
            [
                [c / n for c, n in zip(r.info["chi2"], r.info["ndata"])]
                for r in results
            ]
        ),
        onp.array([1.0, 2.0, 4.0][: len(models)]),
        results,
    )
    draw = Draw(index, "tsv", NPIX, SCALE, "flat", tuple(weights))
    return Group(draw, curve)


def test_selection_drops_a_member_that_fits_badly():
    # The window keeps the two weakest weights of each group (the corner
    # of a three-point L-curve is its middle point).
    good = _group(0, [_scene(38.0), _scene(42.0), _scene(40.0)])
    bad = _group(1, [_scene(40.0), _scene(dra=-48.0), _scene(40.0)])
    result = combine(DATASETS, [good, bad])
    reasons = [(m.draw.index, m.weight, m.reason) for m in result.members]
    assert (1, 1e2, "chi2") in reasons
    assert all(m.reason == "window" for m in result.members if m.weight == 1e3)
    assert all(m.kept for m in result.members if m.reason is None)
    assert 1 <= len(result.kept) <= 3


def _sweep(models, weights, penalty, diverged=()):
    """A Group of ready-made fits; the ``diverged`` ones have a NaN chi2."""
    results = []
    for k, model in enumerate(models):
        chi2 = _chi2(model, DATASETS)
        if k in diverged:
            chi2 = [float("nan")] * len(DATASETS)
        ndata = [d.n_independent for d in DATASETS]
        results.append(FitResult(model, {}, {"chi2": chi2, "ndata": ndata}))
    # As in l_curve: the total chi2 of a diverged fit is NaN too.
    total = onp.array([sum(r.info["chi2"]) for r in results])
    red = onp.array(
        [
            [c / n for c, n in zip(r.info["chi2"], r.info["ndata"])]
            for r in results
        ]
    )
    weights = onp.asarray(weights, dtype=float)
    curve = LCurve(weights, total, red, onp.asarray(penalty), results)
    return Group(Draw(0, "tsv", NPIX, SCALE, "flat", tuple(weights)), curve)


def test_a_diverged_member_is_dropped_and_does_not_move_the_corner():
    sigmas = [37.0, 38.0, 40.0, 44.0, 52.0, 64.0]
    models = [_scene(s) for s in sigmas]
    weights = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
    penalty = [1.0, 1.1, 1.4, 2.5, 6.0, 20.0]
    spec = EnsembleSpec(window_dex=2, chi2_ratio=1e9, mean_rtol=1e9)

    # The fit at weight 1e1, inside the window, diverged: the same sweep
    # without it is the reference for where the corner and the window
    # should be.
    diverged = _sweep(models, weights, penalty, diverged=(4,))
    keep = [0, 1, 2, 3, 5]
    reference = _sweep(
        [models[i] for i in keep],
        [weights[i] for i in keep],
        [penalty[i] for i in keep],
    )
    corner = reference.curve.corner()
    assert diverged.curve.corner() == corner

    result = combine(DATASETS, [diverged], spec=spec)
    expected = combine(DATASETS, [reference], spec=spec)
    assert corner == 1e3
    assert result.members[4].reason == "diverged"
    assert not result.members[4].kept
    windowed = {m.weight for m in result.members if m.reason == "window"}
    assert windowed == {
        m.weight for m in expected.members if m.reason == "window"
    }
    assert result.kept
    assert all(onp.isfinite(m.chi2_red).all() for m in result.kept)


def test_window_keeps_the_weights_just_below_the_corner():
    sigmas = [37.0, 38.0, 40.0, 44.0, 52.0, 64.0]
    weights = [1e5, 1e4, 1e3, 1e2, 1e1, 1e0]
    penalty = [1.0, 1.05, 1.2, 2.5, 6.0, 20.0]
    group = _sweep([_scene(s) for s in sigmas], weights, penalty)
    corner = group.curve.corner()
    assert 1e0 < corner < 1e5
    spec = EnsembleSpec(window_dex=1, chi2_ratio=1e9, mean_rtol=1e9)
    result = combine(DATASETS, [group], spec=spec)
    inside = {m.weight for m in result.members if m.reason != "window"}
    # MYTHRA: from a decade below the corner up to it, never above it.
    assert inside == {corner, corner / 10}


@pytest.mark.parametrize("mean_rtol", [None, 0.0])
def test_iterative_mean_stays_within_mean_rtol_of_the_best(mean_rtol):
    groups = [
        _group(0, [_scene(36.0), _scene(44.0), _scene(40.0)]),
        _group(1, [_scene(40.0, flux=0.25), _scene(dra=40.0), TRUTH]),
    ]
    spec = EnsembleSpec(chi2_ratio=1e3, mean_rtol=mean_rtol, min_kept=1)
    result = combine(DATASETS, groups, spec=spec)
    trace = onp.array(result.trace)
    if mean_rtol is None:
        rtol = onp.sqrt(2.0 / onp.array([d.n_independent for d in DATASETS]))
    else:
        rtol = onp.zeros(len(DATASETS))
    # Judged against the best member, up to rounding, so the tolerance
    # cannot compound.
    assert onp.all(trace <= trace[0] * (1 + rtol + 1e-9))
    assert onp.all(onp.array(result.chi2_red) <= trace[0] * (1 + rtol + 1e-6))
    assert len(trace) == len(result.kept)
    assert "raw chi2/N per dataset" in result.summary()


def test_too_few_members_warns():
    groups = [_group(0, [_scene(36.0), _scene(dra=-48.0), _scene(40.0)])]
    spec = EnsembleSpec(chi2_ratio=1.01, mean_rtol=0.0, min_kept=3)
    with pytest.warns(UserWarning, match="min_kept"):
        combine(DATASETS, groups, spec=spec)


def test_identical_members_have_zero_spread():
    groups = [_group(i, [TRUTH, TRUTH, TRUTH]) for i in range(2)]
    result = combine(DATASETS, groups)
    assert len(result.kept) == 4
    assert onp.allclose(result.std, 0.0, atol=1e-7)
    assert onp.allclose(
        result.mean.brightness, TRUTH.env.brightness, atol=1e-6
    )
    assert onp.isclose(float(result.mean.flux), 0.3, rtol=1e-5)


def test_mean_is_judged_on_the_members_own_grids():
    # The common grid has the finest pixels and the largest field, so a
    # member on a coarser, incommensurate grid is resampled to it, which
    # smooths the image and on these data raises its chi2 several-fold.
    # Judged on its own grid, the best member alone keeps its own chi2.
    coarse = SCALE * 4 / 3
    best = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(9, coarse, 40.0, dra=48.0), coarse, flux=0.3
        ),
    )
    group = _group(0, [_scene(36.0), best, _scene(44.0)])
    group = Group(
        Draw(0, "tsv", 9, coarse, "flat", group.draw.weights), group.curve
    )
    other = _group(1, [_scene(30.0), _scene(32.0), _scene(34.0)])
    # Keep the fine-grid members live, so that the common grid is theirs.
    spec = EnsembleSpec(chi2_ratio=1e9, mad_cut=1e9)
    result = combine(DATASETS, [group, other], spec=spec)
    first = min(
        (m for m in result.members if m.reason != "window"),
        key=lambda m: m.total_chi2_red,
    )
    assert first.result.model is best
    assert onp.allclose(result.trace[0], first.chi2_red, rtol=1e-4)
    ndata = [d.n_independent for d in DATASETS]
    chi2 = [c / n for c, n in zip(_chi2(result.model, DATASETS), ndata)]
    assert onp.allclose(result.chi2_red, chi2, rtol=1e-4)


def test_mixture_is_the_mean_of_the_members_scenes():
    # Members with different image fluxes, on different grids.
    members = [
        _scene(36.0, flux=0.1),
        System(
            star=PointSource(),
            env=Image.from_brightness(
                gaussian_blob(9, SCALE * 4 / 3, 44.0, dra=-30.0),
                SCALE * 4 / 3,
                flux=0.6,
            ),
        ),
        _scene(40.0, flux=0.3),
    ]
    images = [m.env for m in members]
    fractions = [
        float(m.env.flux) / (1.0 + float(m.env.flux)) for m in members
    ]
    u, v, wavel = DATA.u, DATA.v, DATA.wavel
    mixture = _mixture(images, fractions, [0, 1], True)
    expected = (
        members[0].model(u, v, wavel) + members[1].model(u, v, wavel)
    ) / 2
    assert onp.allclose(mixture.model(u, v, wavel), expected, atol=1e-6)


def test_draws_are_reproducible_and_grouped_by_geometry():
    spec = EnsembleSpec(n_weights=4)
    a = draw_groups(DATA, 8, jax.random.PRNGKey(3), spec)
    b = draw_groups(DATA, 8, jax.random.PRNGKey(3), spec)
    assert a == b
    assert [d.geometry for d in a] == sorted(d.geometry for d in a)
    assert sorted(d.index for d in a) == list(range(8))
    n_data = DATA.n_independent
    for d in a:
        assert d.npix <= spec.max_npix
        low, high = spec.weight_ranges[d.family]
        assert all(low * n_data <= w <= high * n_data for w in d.weights)
        assert list(d.weights) == sorted(d.weights, reverse=True)
    with pytest.raises(ValueError, match="Unknown regulariser"):
        EnsembleSpec(families=("l2",))


def test_draws_span_each_range_and_keep_the_field():
    spec = EnsembleSpec(field_factors=(4.0,), max_npix=16)
    n_data = DATA.n_independent
    field = field_of_view(DATA)
    for d in draw_groups(DATA, 8, jax.random.PRNGKey(0), spec):
        # One weight in each equal bin of log w.
        low, high = onp.log(spec.weight_ranges[d.family]) + onp.log(n_data)
        edges = onp.linspace(low, high, spec.n_weights + 1)
        bins = onp.digitize(onp.log(d.weights), edges[1:-1])
        assert sorted(bins) == list(range(spec.n_weights))
        # Too big for max_npix: the field is kept, the pixels coarsened.
        assert d.npix == 16
        assert onp.isclose(d.npix * d.pixel_scale_mas, 4.0 * field)


@pytest.mark.parametrize("family", ["tsv", "tv", "maxent", "starlet"])
def test_default_weight_ranges_hold_the_corner_with_room_below(family):
    data = DATASETS[0]
    spec = EnsembleSpec()
    n_data = data.n_independent
    low, high = spec.weight_ranges[family]
    weights = tuple(
        float(w) for w in onp.geomspace(high * n_data, low * n_data, 11)
    )
    starts = reference_starts(data, True, ("flat",))
    method = "lm" if family == "tsv" else "lbfgs"
    group = run_group(
        data,
        Draw(0, family, NPIX, SCALE, "flat", weights),
        starts=starts,
        method=method,
        max_steps=200,
    )
    corner = group.curve.corner() / n_data
    # The whole window below the corner lies inside the range.
    assert low * 10**spec.window_dex <= corner * (1 + 1e-9)
    assert corner < high


SMALL = EnsembleSpec(
    families=("tsv",),
    n_weights=3,
    oversample=(2.0,),
    field_factors=(0.5,),
    starts=("flat",),
    max_npix=8,
)


def test_ensemble_smoke():
    data = DATASETS[0]
    result = ensemble(
        data, 2, jax.random.PRNGKey(0), spec=SMALL, method="lm", max_steps=50
    )
    assert len(result.members) == 6
    assert len(result.groups) == 2
    assert result.kept
    assert onp.all(onp.isfinite(onp.asarray(result.mean.brightness)))
    assert result.std.shape == result.mean.brightness.shape
    assert len(result.chi2_red) == 1
    assert "kept" in result.summary()


def test_groups_sharing_a_geometry_compile_once():
    data = DATASETS[0]
    starts = reference_starts(data, True, ("flat",))
    first, second = (
        Draw(i, "tsv", 8, 38.0, "flat", weights)
        for i, weights in enumerate([(1e4, 1e3, 1e2), (5e4, 2e3, 7e1)])
    )
    with count_compiles() as compiles:
        run_group(data, first, starts=starts, method="lm", max_steps=20)
    assert compiles  # the first group compiles its fit
    with count_compiles() as compiles:
        run_group(data, second, starts=starts, method="lm", max_steps=20)
    assert not compiles


def test_without_a_star_members_are_recentred_on_the_best():
    def image(dra):
        blob = gaussian_blob(NPIX, SCALE, 40.0, dra=dra)
        return Image.from_brightness(blob, SCALE)

    data = [DATA.with_model(image(0.0), key=jax.random.PRNGKey(5))]
    weights = onp.array([1e3, 1e2, 1e1])
    results = [
        FitResult(m, {}, {"chi2": _chi2(m, data), "ndata": [220]})
        for m in (image(-SCALE), image(0.0), image(SCALE))
    ]
    curve = LCurve(
        weights,
        onp.array([sum(r.info["chi2"]) for r in results]),
        onp.array([[r.info["chi2"][0] / 220] for r in results]),
        onp.array([1.0, 2.0, 4.0]),
        results,
    )
    group = Group(Draw(0, "tsv", NPIX, SCALE, "flat", tuple(weights)), curve)
    # Keep every member whatever its fit: only the recentring is tested.
    spec = EnsembleSpec(window_dex=3, chi2_ratio=1e9, mean_rtol=1e9)
    result = combine(data, [group], star=False, spec=spec)
    assert len(result.kept) == 2
    peak = float(result.mean.brightness.max())
    assert onp.abs(result.std).max() < 0.02 * peak
    # The model moves each member by its shift too. These kernel phases
    # cannot see a shift, so compare the complex visibilities: shifted,
    # the second member is the best one's image.
    u, v, wavel = DATA.u, DATA.v, DATA.wavel
    best = result.model.member0.model(u, v, wavel)
    moved = result.model.member1.model(u, v, wavel)
    assert onp.abs(moved - best).max() < 0.01


@pytest.mark.slow
def test_ensemble_recovers_two_blobs():
    npix, scale = 24, 19.0
    blobs = gaussian_blob(npix, scale, 25.0, dra=60.0) + 0.6 * gaussian_blob(
        npix, scale, 35.0, dra=-40.0, ddec=50.0
    )
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(blobs, scale, flux=0.5),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(7))
    # AMI's uv lattice aliases a field larger than field_of_view(data).
    spec = EnsembleSpec(field_factors=(0.5, 0.75, 1.0))
    result = ensemble(data, 8, jax.random.PRNGKey(1), spec=spec)
    best = min(result.kept, key=lambda m: m.total_chi2_red)
    assert all(
        c <= b * (1 + 1e-6) for c, b in zip(result.chi2_red, result.trace[0])
    )
    assert best.chi2_red[0] < 2.0
    scores = score(result.mean, truth.env)
    assert scores["ncc"] > 0.8
