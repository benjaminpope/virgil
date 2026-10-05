import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import log_density

from virgil.grid_fit import (
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from virgil.inference import laplace_cov
from virgil.likelihood import (
    build_model,
    loglike,
    model_loglike,
    numpyro_model,
)
from virgil.limits import absil_limits, nsigma
from virgil.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    GaussianDisk,
    ModulatedGaussianRim,
    PointSource,
    System,
    UniformDisk,
)
from tests._test_data import oidata, samples_dict

U, V, WAVEL = oidata.u, oidata.v, oidata.wavel


def _composed_binary(dra=120.0, ddec=-80.0, flux=4e-3):
    return System(
        star=PointSource(), comp=PointSource(dra=dra, ddec=ddec, flux=flux)
    )


def test_composed_binary_matches_binary_model_cartesian():
    assert np.allclose(
        _composed_binary().model(U, V, WAVEL),
        BinaryModelCartesian(120.0, -80.0, 4e-3).model(U, V, WAVEL),
        atol=1e-12,
    )


def test_components_keep_their_order_through_set():
    system = System(zeta=PointSource(), alpha=PointSource(flux=0.1))
    assert list(system.components) == ["zeta", "alpha"]
    assert list(system.set("alpha.flux", 0.2).components) == ["zeta", "alpha"]
    leaves = jax.tree_util.tree_leaves(system)
    rebuilt = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(system), leaves
    )
    assert list(rebuilt.components) == ["zeta", "alpha"]


@pytest.mark.parametrize(
    "name",
    [
        "flux",
        "dra",
        "components",
        "names",
        "parts",
        "model",
        "render",
        "set",
        "get",
        "_hidden",
        "comp.disk",
        "2nd",
        "",
    ],
)
def test_invalid_component_names_are_rejected(name):
    with pytest.raises(ValueError, match="component name|Component name"):
        System({name: PointSource(), "star": PointSource()})


def test_unknown_component_error_lists_components():
    with pytest.raises(AttributeError, match=r"\['star', 'comp'\]"):
        _composed_binary().planet


def test_fluxes_summing_to_zero_are_rejected():
    with pytest.raises(ValueError, match="sum to zero"):
        System(a=PointSource(flux=0.0), b=GaussianDisk(3.0, flux=0.0))


@pytest.mark.parametrize(
    "build",
    [
        lambda: PointSource(flux=-1e-3),
        lambda: GaussianDisk(3.0, flux=-0.1),
        lambda: System(star=PointSource(), flux=-1.0),
    ],
    ids=["point", "disk", "system"],
)
def test_negative_fluxes_are_rejected(build):
    with pytest.raises(ValueError, match="must be non-negative"):
        build()


def test_zero_flux_is_allowed():
    assert float(PointSource(flux=0.0).flux) == 0.0


def test_traced_fluxes_are_not_checked():
    # Optimizers may step through negative values while fitting; only
    # concrete values are validated.
    fn = jax.jit(lambda f: _composed_binary(flux=f).model(U, V, WAVEL))
    assert np.all(np.isfinite(fn(-1e-3)))


def test_negative_flux_grid_axis_is_rejected():
    grid = {**_path_samples(), "comp.flux": np.array([-1e-3, 1e-3])}
    with pytest.raises(ValueError, match="negative values"):
        likelihood_grid(oidata, _composed_binary(), grid)


def test_negative_explicit_flux_param_axis_is_rejected():
    # The selected flux axis is checked even when its name does not end in
    # "flux".
    grid = {
        "sep": np.array([150.0]),
        "pa": np.array([30.0]),
        "companion_brightness": np.array([-1e-3, 1e-3]),
    }
    with pytest.raises(ValueError, match="negative values"):
        absil_limits(
            oidata,
            lambda sep, pa, companion_brightness: BinaryModelAngular(
                sep, pa, companion_brightness
            ),
            grid,
            3.0,
            flux_param="companion_brightness",
        )


def test_absil_limits_zero_starting_flux_uses_smallest_positive_flux():
    # An unbounded search starts from the flux axis; it must start from the
    # smallest positive flux instead of log10(0).
    coords = {
        "comp.dra": np.array([100.0, 200.0]),
        "comp.ddec": np.array([100.0]),
    }
    with_zero = {**coords, "comp.flux": np.array([0.0, 1e-3])}
    positive = {**coords, "comp.flux": np.array([1e-3])}
    template = _composed_binary()
    kwargs = dict(flux_param="comp.flux", flux_bounds=None)

    # Just above the smallest reachable significance (a chi-squared ratio
    # of 1).
    ndof = oidata.n_independent
    sigma = float(nsigma(1.0, 1.0, ndof)) + 1e-3
    assert np.allclose(
        absil_limits(oidata, template, with_zero, sigma, **kwargs),
        absil_limits(oidata, template, positive, sigma, **kwargs),
    )


def test_absil_limits_rejects_flux_axis_without_positive_values():
    # Only an unbounded search starts from the flux axis.
    grid = {**_path_samples(), "comp.flux": np.array([0.0])}
    with pytest.raises(ValueError, match="positive value"):
        absil_limits(
            oidata,
            _composed_binary(),
            grid,
            3.0,
            flux_param="comp.flux",
            flux_bounds=None,
        )


def test_flux_prior_with_negative_support_is_rejected():
    priors = {"comp.flux": dist.Normal(0.0, 1e-3)}
    with pytest.raises(ValueError, match="allows negative values"):
        numpyro_model(_composed_binary(), priors, oidata)
    with pytest.raises(ValueError, match="allows negative values"):
        numpyro_model(
            _composed_binary(), {"comp.flux": dist.Uniform(-1.0, 1.0)}, oidata
        )
    numpyro_model(
        _composed_binary(), {"comp.flux": dist.LogUniform(1e-5, 1e-1)}, oidata
    )


def test_binary_inside_a_system_renders():
    image = System(
        binary=BinaryModelCartesian(10.0, -5.0, 0.2),
        halo=GaussianDisk(20.0, flux=0.1),
    ).render(npix=32, fov_mas=80.0)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6)


def test_binary_to_system_has_the_same_visibilities():
    binary = BinaryModelCartesian(120.0, -80.0, 4e-3)
    composed = binary.to_system()
    assert list(composed.components) == ["primary", "companion"]
    assert np.allclose(
        composed.model(U, V, WAVEL), binary.model(U, V, WAVEL), atol=1e-6
    )


def test_repr_shows_names_and_values():
    text = repr(_composed_binary())
    assert "star=PointSource(flux=1, dra=0, ddec=0)" in text
    assert "comp=PointSource(flux=0.004, dra=120, ddec=-80)" in text


def test_rim_accepts_scalar_modulation():
    scalar = ModulatedGaussianRim(
        20.0, 2.0, 30.0, 45.0, az_amps=0.3, az_pas=10.0
    )
    listed = ModulatedGaussianRim(20.0, 2.0, 30.0, 45.0, [0.3], [10.0])
    assert scalar.az_amps.shape == (1,)
    assert np.allclose(scalar.model(U, V, WAVEL), listed.model(U, V, WAVEL))


def test_rim_rejects_mismatched_modulation_lengths():
    with pytest.raises(ValueError, match="one position angle per modulation"):
        ModulatedGaussianRim(20.0, 2.0, 30.0, 45.0, [0.3, 0.1], [10.0])


def test_paths_get_and_set_through_named_components():
    system = System(
        star=PointSource(),
        comp=System(core=PointSource(), disk=GaussianDisk(2.0, flux=0.3)),
    )
    assert float(system.get("comp.disk.sigma")) == pytest.approx(2.0)
    updated = system.set(["comp.disk.sigma", "comp.flux"], [5.0, 0.1])
    assert float(updated.comp.disk.sigma) == pytest.approx(5.0)
    assert float(updated.comp.flux) == pytest.approx(0.1)
    assert float(system.comp.disk.sigma) == pytest.approx(2.0)


def test_zero_companion_flux_is_unresolved_star():
    assert np.allclose(
        _composed_binary(flux=0.0).model(U, V, WAVEL), 1.0 + 0j, atol=1e-12
    )


def test_visibility_is_flux_weighted_mean_of_components():
    disk = GaussianDisk(5.0, flux=2.0, dra=3.0)
    ring = UniformDisk(8.0, flux=0.5, ddec=-4.0)
    expected = (
        2.0 * disk.model(U, V, WAVEL) + 0.5 * ring.model(U, V, WAVEL)
    ) / 2.5
    assert np.allclose(System(a=disk, b=ring).model(U, V, WAVEL), expected)


def test_shifted_subsystem_renders_east_on_the_left():
    image = onp.asarray(
        System(
            star=PointSource(),
            comp=System(core=PointSource(), dra=4.0, flux=3.0),
        ).render(npix=5, fov_mas=10.0)
    )
    assert onp.unravel_index(image.argmax(), image.shape) == (2, 0)


def test_build_model_accepts_classes_and_templates():
    params, values = ("dra", "ddec", "flux"), (10.0, -5.0, 0.01)
    from_class = build_model(BinaryModelCartesian, params, values)
    from_template = build_model(
        _composed_binary(), [f"comp.{p}" for p in params], values
    )
    assert np.allclose(
        from_class.model(U, V, WAVEL), from_template.model(U, V, WAVEL)
    )


def _loglike_at(point):
    return loglike(
        list(point.values()), list(point), oidata, _composed_binary()
    )


def _path_samples():
    return {f"comp.{key}": value for key, value in samples_dict.items()}


def _two_flux_template():
    return System(
        star=PointSource(),
        disk=GaussianDisk(5.0, flux=0.1),
        comp=PointSource(flux=1e-3),
    )


def test_ambiguous_flux_needs_flux_param():
    small = {key: value[::6] for key, value in _path_samples().items()}
    grid = {"disk.flux": np.array([0.05, 0.1]), **small}
    with pytest.raises(ValueError, match="Pass flux_param"):
        optimized_flux_grid(oidata, _two_flux_template(), grid)
    explicit = optimized_flux_grid(
        oidata, _two_flux_template(), grid, flux_param="comp.flux"
    )
    assert explicit.shape == (2,) + tuple(v.size for v in small.values())[:2]


def test_flux_inference_matches_explicit_and_model_class():
    small = {key: value[::6] for key, value in _path_samples().items()}
    inferred = optimized_flux_grid(oidata, _composed_binary(), small)
    legacy_samples = {key.split(".")[1]: value for key, value in small.items()}
    legacy = optimized_flux_grid(oidata, BinaryModelCartesian, legacy_samples)
    explicit = optimized_flux_grid(
        oidata, _composed_binary(), small, flux_param="comp.flux"
    )
    assert np.allclose(inferred, explicit)
    # The two models round differently in float32, which moves the optimum
    # by a small fraction of the flux uncertainty.
    sigma = laplace_flux_uncertainty_grid(
        oidata, BinaryModelCartesian, legacy_samples, flux=legacy
    )
    assert onp.all(onp.abs(onp.asarray(legacy - explicit)) < 0.2 * sigma)


def test_new_template_values_do_not_recompile(monkeypatch):
    import virgil.grid_fit as grid_fit

    traces = []
    real_loglike = grid_fit.loglike

    def counting_loglike(*args):
        traces.append(1)
        return real_loglike(*args)

    monkeypatch.setattr(grid_fit, "loglike", counting_loglike)
    # A grid shape used nowhere else, so the first call must compile.
    grid = {
        "comp.dra": np.linspace(-50.0, 50.0, 7),
        "comp.ddec": np.linspace(-50.0, 50.0, 5),
        "comp.flux": np.array([1e-3, 1e-2, 3e-2]),
    }
    for sigma in (3.0, 4.0, 5.5):
        template = System(
            star=PointSource(),
            disk=GaussianDisk(sigma, flux=0.1),
            comp=PointSource(flux=1e-3),
        )
        likelihood_grid(oidata, template, grid)
    assert len(traces) == 1


def test_flux_param_can_be_any_key_regardless_of_order():
    small = {key: value[::6] for key, value in _path_samples().items()}
    reordered = {
        "comp.flux": small["comp.flux"],
        "comp.dra": small["comp.dra"],
        "comp.ddec": small["comp.ddec"],
    }
    assert np.allclose(
        optimized_flux_grid(
            oidata, _composed_binary(), reordered, flux_param="comp.flux"
        ),
        optimized_flux_grid(
            oidata, _composed_binary(), small, flux_param="comp.flux"
        ),
        rtol=1e-4,
    )


def test_unknown_flux_param_is_rejected():
    with pytest.raises(ValueError, match="is not one of the keys"):
        optimized_flux_grid(
            oidata, _composed_binary(), _path_samples(), flux_param="flux"
        )


def test_best_grid_point_returns_named_values():
    grid = _path_samples()
    loglike = likelihood_grid(oidata, _composed_binary(), grid)
    best = best_grid_point(loglike, grid)
    assert list(best) == list(grid)
    assert np.isclose(
        float(np.max(loglike)), float(_loglike_at(best)), rtol=1e-6
    )


def test_likelihood_grid_with_paths_matches_model_class():
    assert np.allclose(
        likelihood_grid(oidata, _composed_binary(), _path_samples()),
        likelihood_grid(oidata, BinaryModelCartesian, samples_dict),
        rtol=1e-4,
    )


def test_optimized_likelihood_grid_with_paths_matches_model_class():
    grid_best = onp.asarray(
        likelihood_grid(oidata, BinaryModelCartesian, samples_dict)
    ).max(axis=2)
    composed = onp.asarray(
        optimized_likelihood_grid(
            oidata, _composed_binary(), _path_samples(), flux_param="comp.flux"
        )
    )
    reference = onp.asarray(
        optimized_likelihood_grid(
            oidata, BinaryModelCartesian, samples_dict, flux_param="flux"
        )
    )
    # In float32, BFGS lands on slightly different optima in ~1% of cells for
    # either input; both must still improve on the grid.
    assert onp.all(composed >= grid_best - 1e-2 * onp.abs(grid_best))
    assert onp.mean(onp.isclose(composed, reference, rtol=1e-4)) > 0.98


def test_absil_limits_with_paths_matches_model_class():
    small = {key: value[::3] for key, value in samples_dict.items()}
    paths = {f"comp.{key}": value for key, value in small.items()}
    assert np.allclose(
        absil_limits(
            oidata, _composed_binary(), paths, 3.0, flux_param="comp.flux"
        ),
        absil_limits(
            oidata, BinaryModelCartesian, small, 3.0, flux_param="flux"
        ),
        # Limits near the flux_bounds ceiling of 1 are barely constrained,
        # so float32 rounding differences move them by up to a few percent.
        rtol=2e-2,
    )


def test_laplace_cov_with_paths_matches_model_class():
    values = np.array([120.0, -80.0, 4e-3])
    params = ["dra", "ddec", "flux"]
    assert np.allclose(
        laplace_cov(
            values, [f"comp.{p}" for p in params], oidata, _composed_binary()
        ),
        laplace_cov(values, params, oidata, BinaryModelCartesian),
        rtol=1e-3,
    )


def test_numpyro_model_log_density_is_prior_plus_loglike():
    priors = {
        "comp.dra": dist.Uniform(0.0, 200.0),
        "comp.ddec": dist.Uniform(-200.0, 0.0),
        "comp.flux": dist.LogUniform(1e-5, 1e-1),
    }
    point = {"comp.dra": 120.0, "comp.ddec": -80.0, "comp.flux": 4e-3}
    logp, _ = log_density(
        numpyro_model(_composed_binary(), priors, oidata), (), {}, point
    )
    expected = sum(
        priors[key].log_prob(value) for key, value in point.items()
    ) + loglike(list(point.values()), list(point), oidata, _composed_binary())
    assert np.isclose(logp, expected, rtol=1e-6)


def test_binary_angular_is_unchanged_by_composition_machinery():
    angular = BinaryModelAngular(sep=144.2, pa=123.7, flux=1.0 / 250.0)
    assert np.allclose(
        angular.model(U, V, WAVEL),
        angular.to_cartesian().model(U, V, WAVEL),
        atol=1e-6,
    )
    assert jax.tree_util.tree_structure(angular).num_leaves == 3


def test_numpyro_model_accepts_a_function_of_new_parameters():
    def polar_binary(sep, pa, flux):
        pa_rad = np.deg2rad(pa)
        return _composed_binary(
            dra=sep * np.sin(pa_rad), ddec=sep * np.cos(pa_rad), flux=flux
        )

    sep, pa = (
        float(np.hypot(120.0, -80.0)),
        float(np.rad2deg(np.arctan2(120.0, -80.0))),
    )
    priors = {
        "sep": dist.Uniform(50.0, 250.0),
        "pa": dist.Uniform(0.0, 360.0),
        "flux": dist.LogUniform(1e-5, 1e-1),
    }
    point = {"sep": sep, "pa": pa, "flux": 4e-3}
    logp, _ = log_density(
        numpyro_model(polar_binary, priors, oidata), (), {}, point
    )
    expected = sum(
        priors[key].log_prob(value) for key, value in point.items()
    ) + loglike(list(point.values()), list(point), oidata, polar_binary)
    assert np.isclose(logp, expected, rtol=1e-5)
    covariance = laplace_cov(
        np.array(list(point.values())), list(point), oidata, polar_binary
    )
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.T, rtol=1e-3)


def _two_epochs(dra1, ddec1, dra2, ddec2, flux):
    """A binary at two epochs: a position each, one flux ratio."""
    return [
        _composed_binary(dra1, ddec1, flux),
        _composed_binary(dra2, ddec2, flux),
    ]


def test_numpyro_model_pairs_a_list_of_models_with_the_datasets():
    truth = {
        "dra1": 120.0,
        "ddec1": -80.0,
        "dra2": 100.0,
        "ddec2": -110.0,
        "flux": 4e-3,
    }
    epochs = [
        oidata.with_model(m, key=jax.random.PRNGKey(k))
        for k, m in enumerate(_two_epochs(**truth))
    ]
    priors = {
        key: dist.Uniform(-200.0, 0.0)
        if key.startswith("ddec")
        else dist.Uniform(0.0, 200.0)
        for key in truth
        if key != "flux"
    } | {"flux": dist.LogUniform(1e-5, 1e-1)}
    sampler = numpyro_model(_two_epochs, priors, epochs)
    point = truth | {"dra2": 105.0, "flux": 5e-3}
    logp, _ = log_density(sampler, (), {}, point)
    paired = sum(
        model_loglike(m, d) for m, d in zip(_two_epochs(**point), epochs)
    )
    log_prior = sum(
        priors[key].log_prob(value) for key, value in point.items()
    )
    assert np.isclose(logp, log_prior + paired, rtol=1e-6)
    # Each model meets its own dataset: swapping them costs likelihood.
    swapped = sum(
        model_loglike(m, d) for m, d in zip(_two_epochs(**point)[::-1], epochs)
    )
    assert swapped < paired - 10.0

    mcmc = MCMC(
        NUTS(sampler, init_strategy=init_to_value(values=truth)),
        num_warmup=100,
        num_samples=100,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    assert all(np.all(np.isfinite(samples[key])) for key in truth)
    assert abs(float(np.median(samples["dra2"])) - 100.0) < 5.0

    with pytest.raises(ValueError, match="2 models for 3 datasets"):
        log_density(
            numpyro_model(_two_epochs, priors, epochs + [oidata]),
            (),
            {},
            point,
        )


def test_function_ties_parameters_between_components():
    def coplanar_rings(inc, pa, inner_diam, outer_diam, outer_flux):
        return System(
            star=PointSource(),
            inner=ModulatedGaussianRim(inner_diam, 2.0, inc, pa, flux=0.5),
            outer=ModulatedGaussianRim(
                outer_diam, 4.0, inc, pa, flux=outer_flux
            ),
        )

    values = (40.0, 25.0, 10.0, 30.0, 0.2)
    names = ["inc", "pa", "inner_diam", "outer_diam", "outer_flux"]
    tied = build_model(coplanar_rings, names, values)
    assert float(tied.inner.inc) == float(tied.outer.inc) == 40.0
    assert float(tied.inner.pa) == float(tied.outer.pa) == 25.0
    assert np.isfinite(loglike(values, names, oidata, coplanar_rings))


def test_nesting_ties_positions_of_a_group():
    host = System(
        star=PointSource(),
        rim=ModulatedGaussianRim(10.0, 2.0, 30.0, 45.0, flux=0.5),
    )
    scene = System(host=host, comp=PointSource(dra=40.0, flux=0.01))
    moved = scene.set(["host.dra", "host.ddec"], [3.0, -2.0])
    by_hand = System(
        host=System(
            star=PointSource(dra=3.0, ddec=-2.0),
            rim=ModulatedGaussianRim(
                10.0, 2.0, 30.0, 45.0, flux=0.5, dra=3.0, ddec=-2.0
            ),
        ),
        comp=PointSource(dra=40.0, flux=0.01),
    )
    assert np.allclose(
        moved.model(U, V, WAVEL), by_hand.model(U, V, WAVEL), atol=1e-6
    )


class _PowerLawPoint(PointSource):
    """Test-only chromatic component: flux * (wavel / wavel0) ** index."""

    index: jax.Array
    wavel0: float = 2e-6

    def __init__(self, flux, index, dra=0.0, ddec=0.0):
        super().__init__(flux=flux, dra=dra, ddec=ddec)
        self.index = np.asarray(index, dtype=float)

    def _weight(self, wavel=None):
        if wavel is None:
            return self.flux
        return self.flux * (wavel / self.wavel0) ** self.index


def test_system_passes_wavelength_to_component_weights():
    wavels = np.array([1.5e-6, 2e-6, 2.5e-6])
    u, v = np.full(3, 3.0), np.full(3, 1.0)
    comp = _PowerLawPoint(flux=0.1, index=-2.0, dra=50.0)
    chromatic = System(star=PointSource(), comp=comp).model(u, v, wavels)
    for i, wavel in enumerate(wavels):
        ratio = 0.1 * (wavel / 2e-6) ** -2.0
        grey = System(
            star=PointSource(), comp=PointSource(flux=ratio, dra=50.0)
        ).model(u[i], v[i], wavel)
        assert np.allclose(chromatic[i], grey, atol=1e-6)
