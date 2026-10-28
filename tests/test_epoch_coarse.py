"""Memory-budgeted batches and coarse-then-exact scoring of the epoch
surface for data with gains (``epoch_positions``)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil._precision import cast_tree, run_in
from virgil.coverage import VLTI_UTS, vlti_oidata
from virgil.epochs import (
    Epochs,
    _coarse_then_exact,
    _grid_batch,
    _grid_marginal,
    _grid_peaks,
    _grid_scores,
    _Surface,
    epoch_positions,
)
from virgil.models import BinaryModelCartesian
from virgil.simulate import simulate

TRUTH = (4.0, -2.0, 0.6)
AXIS = onp.arange(-10.0, 10.01, 1.0)
GRID = {"dra": AXIS, "ddec": AXIS, "flux": [0.3, 0.6, 1.0]}


def _night(gains, seed=1, mjd=60000.0):
    template = vlti_oidata(
        stations=VLTI_UTS,
        hour_angles_h=(-1.0, 0.0, 1.0),
        wavelengths_m=onp.asarray((2.1e-6, 2.3e-6)),
        sigma_v2=0.03,
        sigma_cp_deg=3.0,
        nights_mjd=[mjd],
    )
    with jax.enable_x64(True):
        model = simulate(BinaryModelCartesian(*TRUTH), template)
    rng = onp.random.default_rng(seed)
    vis, d_vis = onp.asarray(model.vis), onp.asarray(model.d_vis)
    phi, d_phi = onp.asarray(model.phi), onp.asarray(model.d_phi)
    vis = vis + 2.0 * d_vis * rng.standard_normal(vis.shape)
    phi = phi + 2.0 * d_phi * rng.standard_normal(phi.shape)
    data = model.set(["vis", "phi"], [vis, phi])
    return data.with_gains(telescope=0.05, baseline=0.03) if gains else data


def test_gains_pick_a_smaller_batch_than_plain_data():
    plain, gained = _night(False), _night(True)
    batch_plain = _grid_batch(None, None, plain, _Surface(plain, 10.0))
    batch_gained = _grid_batch(None, None, gained, _Surface(gained))
    assert 1 <= batch_gained < batch_plain
    # the budget scales it; an explicit batch_size overrides it
    surface = _Surface(gained)
    big = _grid_batch(None, 100 * 2**30, gained, surface)
    small = _grid_batch(None, 2**28, gained, surface)
    assert small <= batch_gained <= big
    assert _grid_batch(7, 1, gained, surface) == 7
    with pytest.raises(ValueError, match="memory_budget"):
        _grid_batch(None, -1.0, gained, surface)


def _exact_and_coarse(nats, batch_size):
    data = _night(True)
    surface = _Surface(data)
    xx, yy, ff = onp.meshgrid(AXIS, AXIS, GRID["flux"], indexing="ij")
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    with run_in("float64"):
        d64 = cast_tree(data, "float64")
        dof = np.asarray(1.0)
        _, chi2 = _grid_scores(np.asarray(points), d64, surface, 16, None, dof)
        exact = onp.asarray(_grid_marginal(chi2, dof, d64, surface))
        coarse, share = _coarse_then_exact(
            points,
            xx.shape,
            d64,
            surface,
            None,
            dof,
            batch_size,
            None,
            nats,
            1,
        )
    return exact.reshape(xx.shape), coarse.reshape(xx.shape), share, (xx, yy)


@pytest.mark.slow
def test_coarse_then_exact_matches_the_full_surface_near_the_peak():
    exact, coarse, share, (xx, yy) = _exact_and_coarse(30.0, 8)
    near = exact >= exact.max() - 20.0
    assert onp.abs(coarse - exact)[near].max() <= 1e-3
    assert share < 0.5
    rival = 1.5
    a = _grid_peaks(exact, xx, yy, rival, 5)
    b = _grid_peaks(coarse, xx, yy, rival, 5)
    assert a[0] == b[0]
    assert {c for c in a if exact[c] > exact.max() - 20} <= set(b)
    # far down the surface the approximation is only an offset away
    assert onp.all(coarse[~near] < coarse.max() - 15.0)


@pytest.mark.slow
def test_epoch_positions_agree_with_and_without_coarse_scoring():
    epochs = Epochs({"night": _night(True)})
    full = epoch_positions(epochs, GRID, coarse_nats=None)
    quick = epoch_positions(epochs, GRID, batch_size=8)
    assert full.exact_fraction == (1.0,)
    assert 0.0 < quick.exact_fraction[0] < 1.0
    onp.testing.assert_allclose(quick.dra, full.dra, atol=1e-6)
    onp.testing.assert_allclose(quick.ddec, full.ddec, atol=1e-6)
    onp.testing.assert_allclose(
        quick.peaks[0].marginal[0], full.peaks[0].marginal[0], atol=1e-3
    )
    onp.testing.assert_allclose(
        quick.gap_marginal, full.gap_marginal, atol=1e-3
    )
    assert quick.flags == full.flags
    # peaks within coarse_nats of the best are exact; lower ones approximate
    exact = full.peaks[0].marginal >= full.peaks[0].marginal[0] - 30.0
    k = int(exact.sum())
    onp.testing.assert_allclose(
        quick.peaks[0].marginal[:k], full.peaks[0].marginal[:k], atol=1e-3
    )


def test_candidate_batch_is_64_for_plain_data_and_shares_the_budget():
    from virgil.aliases import _candidate_batch

    plain, gained = _night(False), _night(True)
    sp, sg = _Surface(plain, 10.0), _Surface(gained)
    assert _candidate_batch(None, None, [plain, plain], [sp, sp]) == 64
    one = _candidate_batch(None, 2**22, [gained], [sg])
    five = _candidate_batch(None, 2**22, [gained] * 5, [sg] * 5)
    assert 1 <= five <= one < 64
    assert _candidate_batch(5, None, [gained], [sg]) == 5
    # mixed epochs: the plain one does not take a share
    assert _candidate_batch(None, 2**22, [plain, gained], [sp, sg]) == one


def test_hessian_groups_shrink_with_the_budget():
    from virgil.aliases import _hessian_groups

    nights = [_night(True, seed=k, mjd=60000.0 + k) for k in range(3)]
    surfaces = tuple(_Surface(d) for d in nights)
    assert _hessian_groups(None, nights, surfaces) == [slice(0, 3)]
    tiny = _hessian_groups(1, nights, surfaces)
    assert tiny == [slice(0, 1), slice(1, 2), slice(2, 3)]
    plain = [_night(False)] * 3
    sp = tuple(_Surface(d, 10.0) for d in plain)
    assert _hessian_groups(1, plain, sp) == [slice(0, 3)]
    from virgil.aliases import _HESSIAN_COPIES
    from virgil.epochs import _whitening_floats

    two = 2 * 8 * _HESSIAN_COPIES * _whitening_floats(nights[0])
    assert _hessian_groups(two, nights, surfaces) == [
        slice(0, 2),
        slice(2, 3),
    ]


@pytest.mark.slow
def test_grouped_hessian_sums_to_the_fused_one():
    from virgil.aliases import _Problem

    nights = [_night(True, seed=k, mjd=60000.0 + k) for k in range(2)]
    with run_in("float64"):
        from tests.test_aliases import _settings

        settings = _settings(log_broad=(-50.0,) * 2)
        data = [cast_tree(d, "float64") for d in nights]
        fused = _Problem(data, settings)
        chunked = _Problem(data, settings, memory_budget=1)
        assert fused.log_broad is not None
        assert len(fused.hessian_groups) == 1
        assert len(chunked.hessian_groups) == 2
        x = np.asarray([onp.log(12.1), 0.3, 0.3, 0.2, 1.0, 1.0, 1.5, -1.0])
        onp.testing.assert_allclose(
            chunked.hessian(x), fused.hessian(x), rtol=1e-9, atol=1e-9
        )


@pytest.mark.slow  # compiles the Hessian twice, in one and in several groups
@pytest.mark.parametrize("kind", ["mixture", "inflation"])
def test_the_chain_rule_hessian_is_the_autodiff_hessian(kind):
    """With a live mixture (outlier probabilities well inside (0, 1)) the
    joint-ε Hessian equals ``jax.hessian`` of the full log likelihood, in one
    group and in several."""
    import dataclasses

    import jax

    from virgil.aliases import _loglike, _Problem, _night_terms
    from tests.test_aliases import _settings

    nights = [_night(True, seed=k, mjd=60000.0 + k) for k in range(3)]
    with run_in("float64"):
        data = [cast_tree(d, "float64") for d in nights]
        settings = _settings(outlier=kind, s_max=20.0, log_broad=(0.0,) * 3)
        x = np.asarray([onp.log(12.1), 0.3, 0.3, 0.2, 1.0, 1.0, 1.5, -1.0])
        if kind == "mixture":
            # Put each night's outlier level near its own likelihood.
            probe = _Problem(data, settings)
            ell, _ = _night_terms(
                probe.surfaces, x, tuple(data), probe.cfg, probe.dof,
                np.zeros(3),
            )  # fmt: skip
            settings = dataclasses.replace(
                settings, log_broad=tuple(float(v) for v in ell)
            )
        for budget in (None, 1):
            problem = _Problem(data, settings, memory_budget=budget)
            want = jax.hessian(
                lambda y: _loglike(
                    problem.surfaces, y, tuple(problem.data), problem.cfg,
                    problem.dof, problem.log_broad,
                )
            )(x)  # fmt: skip
            onp.testing.assert_allclose(
                problem.hessian(x), want, rtol=1e-6, atol=1e-6
            )
