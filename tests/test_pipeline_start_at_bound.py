"""NUTS starts inside the prior when the best fit sits on a bound.

With no companion in the data the fitted flux runs to the lower end of its
prior. numpyro cannot start there (the unconstrained value is infinite), so
the pipelines pull the start just inside first; and report it as a check.
"""

import jax
import numpy as onp
import numpyro
import numpyro.distributions as dist
import pytest

from virgil import BinaryModelCartesian, OIData
from virgil.pipeline import BinaryPipeline
from virgil.pipeline import _checks as checks
from virgil._geometry import separation_pa
from virgil.pipeline._common import _START_INSET, _interior_start, _run_chains
from virgil.pipeline.binary import _fit_start
from virgil.pipeline.disk import _inside_priors

pytestmark = pytest.mark.xdist_group("pipeline_start_at_bound")

PRIORS = {
    "flux": dist.LogUniform(1e-5, 0.5),
    "sep": dist.Uniform(0.0, 100.0),
    "free": dist.Normal(0.0, 1.0),
}


def test_interior_start_pulls_bound_values_inside_and_leaves_the_rest():
    start = {
        "flux": onp.asarray(1e-5),
        "sep": onp.asarray(100.0),
        "free": onp.asarray(3.0),
        "unlisted": onp.asarray(7.0),
    }
    moved = _interior_start(PRIORS, start)
    assert 1e-5 < moved["flux"] < 0.5 and 0.0 < moved["sep"] < 100.0
    # The pull is a fixed fraction of the width: in log x for the LogUniform.
    log_fraction = onp.log(moved["flux"] / 1e-5) / onp.log(0.5 / 1e-5)
    assert onp.isclose(log_fraction, _START_INSET, rtol=1e-6)
    assert onp.isclose(moved["sep"], 100.0 * (1 - _START_INSET))
    assert moved["free"] == 3.0 and moved["unlisted"] == 7.0
    # Values already inside are untouched, bit for bit.
    inside = {"flux": onp.asarray(1e-3), "sep": onp.asarray(40.0)}
    assert all(_interior_start(PRIORS, inside)[k] == inside[k] for k in inside)


def test_interior_start_handles_arrays_and_keeps_dtype():
    priors = {"x": dist.Uniform(0.0, 1.0).expand([3]).to_event(1)}
    start = {"x": onp.asarray([0.0, 0.5, 1.0], dtype=onp.float32)}
    moved = _interior_start(priors, start)["x"]
    assert moved.dtype == onp.float32 and moved[1] == 0.5
    assert 0.0 < moved[0] < 0.01 and 0.99 < moved[2] < 1.0


def _toy():
    def model():
        numpyro.sample("flux", PRIORS["flux"])
        numpyro.sample("sep", PRIORS["sep"])

    return model


SETTINGS = {
    "seed": 0,
    "num_warmup": 20,
    "num_samples": 20,
    "num_chains": 2,
    "chain_method": "sequential",
}


def test_run_chains_starts_from_a_value_on_the_bound():
    start = {"flux": onp.asarray(1e-5), "sep": onp.asarray(100.0)}
    mcmc = _run_chains(
        _toy(), SETTINGS, start, dense_mass=False, priors=PRIORS
    )
    flux = onp.asarray(mcmc.get_samples()["flux"])
    assert onp.all(onp.isfinite(flux)) and flux.shape == (40,)


def test_run_chains_without_priors_gives_a_clear_error_not_numpyros():
    start = {"flux": onp.asarray(1e-5), "sep": onp.asarray(100.0)}
    with pytest.raises(
        ValueError, match="NUTS cannot start from the best fit"
    ):
        _run_chains(_toy(), SETTINGS, start, dense_mass=False)


def test_start_at_bound_check():
    check = checks.at_bound(["flux"], start=True)
    assert check.status == "warn" and "flux" in check.message
    assert check.to_dict()["value"] == ["flux"]


TINY = {
    "max_sep_mas": 300.0,
    "max_grid": 13,
    "n_flux": 6,
    "flux_range": [1e-4, 0.2],
    "n_null": 15,
    "detection_sigma": 1.5,
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}


@pytest.mark.slow
def test_binary_pipeline_on_data_without_a_companion_completes(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("nbclient")
    base = OIData("data/NuHor_F480M.oifits")
    # No companion to speak of: some companion parameter runs to a prior bound.
    data = base.with_model(
        BinaryModelCartesian(120.0, 80.0, 1e-9), key=jax.random.PRNGKey(3)
    )
    result = BinaryPipeline(data, output=tmp_path / "run", **TINY).run()
    assert result.status == "complete"
    assert (tmp_path / "run" / "samples.h5").exists()
    # The unconstrained companion runs to a bound of its prior (which one is
    # data-dependent: here the sky position, not the flux).
    at_bound = result.summary["fit"]["at_bound"]
    assert at_bound
    check = {c.name: c for c in result.checks}["start_at_bound"]
    assert check.status == "warn" and check.value == at_bound


def test_inside_priors_pulls_only_values_outside_a_bound():
    """A start beyond a bound has no unconstrained value; one on it is kept."""
    start = {
        "wide": onp.asarray(94.0),  # above the width prior's upper end
        "low": onp.asarray(1e-5),  # exactly on the lower end
        "rounded": onp.asarray(
            94.0886917
        ),  # a rounding error below 94.0886951
        "inside": onp.asarray(40.0),
        "free": onp.asarray(9.0),
    }
    priors = {
        "wide": dist.LogUniform(9.0, 47.0),
        "low": dist.LogUniform(1e-5, 0.5),
        "rounded": dist.LogUniform(94.0886951, 188.0),
        "inside": dist.Uniform(0.0, 100.0),
        "free": dist.Normal(0.0, 1.0),
    }
    moved = _inside_priors(priors, start)
    assert 9.0 < moved["wide"] < 47.0 and moved["rounded"] > 94.0886951
    assert all(moved[k] == start[k] for k in ("low", "inside", "free"))


def _angular_toy():
    """A model of the offsets through ``separation_pa``, as the pipeline's."""

    def model():
        dra = numpyro.sample("dra", dist.Uniform(-100.0, 100.0))
        ddec = numpyro.sample("ddec", dist.Uniform(-100.0, 100.0))
        sep, pa = separation_pa(dra, ddec)
        x = sep * jax.numpy.sin(jax.numpy.radians(pa))
        y = sep * jax.numpy.cos(jax.numpy.radians(pa))
        numpyro.factor("like", -0.5 * ((x - 3.0) ** 2 + y**2) / 25.0)

    return model


def test_run_chains_starts_an_angular_model_on_the_origin():
    """The best fit of a non-detection sits on the origin, where the angle
    is undefined; its NaN gradient made NUTS refuse to start."""
    start = {"dra": onp.asarray(0.0), "ddec": onp.asarray(0.0)}
    priors = {
        "dra": dist.Uniform(-100.0, 100.0),
        "ddec": dist.Uniform(-100.0, 100.0),
    }
    mcmc = _run_chains(
        _angular_toy(), SETTINGS, start, dense_mass=False, priors=priors
    )
    dra = onp.asarray(mcmc.get_samples()["dra"])
    assert onp.all(onp.isfinite(dra)) and dra.shape == (40,)


class _Stub:
    settings = {"flux_range": (1e-4, 1.0)}

    @staticmethod
    def _sampled_model():
        return "template"


@pytest.mark.parametrize(
    "peak", [(0.0, 0.0), (-1.3e-5, -1.3e-5), (30.0, -5.0)]
)
def test_angular_fit_starts_off_the_origin_but_not_off_a_real_peak(peak):
    """An angular template has no position gradient on the origin: a search
    peak on the grid's centre starts the fit a little off it."""
    search = {
        "dra_mas": peak[0],
        "ddec_mas": peak[1],
        "flux": 1e-3,
        "grid_step_mas": 40.0,
    }
    priors = {k: None for k in ("dra", "ddec", "flux")}
    _, start = _fit_start(_Stub, search, priors, angular=True)
    moved = float(onp.hypot(start["dra"], start["ddec"]))
    if peak == (30.0, -5.0):
        assert (float(start["dra"]), float(start["ddec"])) == peak
    else:
        assert moved == pytest.approx(0.4)


class _CartesianStub(_Stub):
    @staticmethod
    def _sampled_model():
        return BinaryModelCartesian(0.0, 0.0, 1e-3)


@pytest.mark.parametrize("angular", [False, True])
def test_the_fit_starts_inside_the_position_prior_when_the_search_peaks_on_its_edge(
    angular,
):
    """The search grid ends where the position priors do, so a peak on the
    grid's edge is on the prior's bound, where the unconstrained start is
    infinite: the fit returned NaN and the summary crashed in
    ``separation_pa`` ("where requires ndarray or scalar arguments, got
    NoneType")."""
    edge = 672.0
    search = {
        "dra_mas": -edge,
        "ddec_mas": edge,
        "flux": 1e-3,
        "grid_step_mas": 40.0,
    }
    priors = {
        "dra": dist.Uniform(-edge, edge),
        "ddec": dist.Uniform(-edge, edge),
        "flux": dist.LogUniform(1e-5, 0.5),
    }
    template, init = _fit_start(_CartesianStub, search, priors, angular)
    start = (
        init
        if angular
        else {k: float(template.get(k)) for k in ("dra", "ddec")}
    )
    inset = _START_INSET * 2 * edge  # the pull, in mas
    assert float(start["dra"]) == pytest.approx(-edge + inset, rel=1e-4)
    assert float(start["ddec"]) == pytest.approx(edge - inset, rel=1e-4)


def test_a_non_finite_fit_is_reported_at_the_fit_not_in_the_summary():
    from virgil.pipeline.binary import _require_finite

    _require_finite({"dra": onp.asarray(1.0)}, {})
    with pytest.raises(ValueError, match=r"\['dra'\].*'failed'"):
        _require_finite(
            {"dra": onp.asarray(onp.nan), "ddec": onp.asarray(2.0)},
            {"stop": "failed", "loss": float("nan")},
        )
