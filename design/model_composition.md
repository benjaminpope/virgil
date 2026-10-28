# Model composition

> Reference for agents, split out of [AGENTS.md](../AGENTS.md); the rules there still apply. Keep this file in step with the code.

- There are two kinds of `SourceModel`. **Components** (`Component`
  subclasses: `PointSource`, `GaussianDisk`, `EllipticalGaussian`,
  `GaussianArc`, `UniformDisk`, `ModulatedGaussianRim`, and others) are pure shapes normalized to unit flux, with
  `flux`, `dra`, `ddec`; their `flux` is a relative weight. **Scenes**
  (`System`, the binaries, `HarmonixModel`) are whole normalized skies with
  weight 1 inside a `System`, unless they carry their own `flux` weight as
  `System` does.
- `flux` means a relative weight. The binaries' companion/primary `flux` is a legacy exception; no new model may use `flux` as a ratio.
- Components never contain a built-in star; compose one with
  `System(star=PointSource(), ...)`. New shapes subclass `Component` and
  implement `_centred_cvis` and `_centred_image`. Anything that can be drawn
  implements `_image`; `SourceModel.render` handles the grid and
  normalization.
- `System(**named)` mixes components as `sum(f_i V_i) / sum(f_i)`. Only flux
  ratios are identifiable: keep one reference component (usually the star) at
  `flux=1`. Components are stored as ordered tuples (`names` is static), so
  their order survives pytree operations. Names must be identifiers that do
  not clash with `System` attributes.
- There is deliberately no `+`/`*` operator sugar: parameter paths are the
  public fitting interface and must depend only on the names the user chose.
- Parameters are addressed by zodiax dot-paths through component names
  (`"comp.flux"`); tools accept a template model plus paths anywhere they
  accept a model class (`build_model`). The argument is called `model`.
- Grid tools that optimize a flux (`optimized_likelihood_grid`,
  `optimized_flux_grid`, `laplace_flux_uncertainty_grid`, `absil_limits`, `injection_limits`)
  use the one key whose last part is `flux` (`_utils.resolve_flux_param`);
  if there is none or more than one, the caller must pass `flux_param=`.
  Plotting uses the same rule.
- Fluxes are non-negative. `Component`/`System` reject concrete negative
  fluxes in `__check_init__` (traced values cannot be checked),
  `numpyro_model` rejects flux priors with negative support, and grid tools
  reject negative flux axes. The optimizers in `optimized_flux_grid`
  deliberately stay unconstrained, because Ruffio upper limits need the
  unconstrained estimate.
- `SourceModel._weight(wavel)` takes the wavelength (or `None` for the
  reference flux, e.g. when rendering), so chromatic fluxes can be added
  without changing every subclass.
- To fit derived or tied parameters, pass a function returning a model
  wherever a template is accepted (`build_model`, `numpyro_model`,
  `laplace_cov`, grid tools). Positions are tied by nesting.
- Grid tools are `eqx.filter_jit`-compiled with the template's arrays
  **traced**, so new parameter values never recompile and JAX's persistent
  compilation cache can hit across sessions. Do not reintroduce static
  templates or value-dependent code paths (e.g. skipping work when a value
  is a known zero): they bake constants into the HLO, which recompiles for
  every value and defeats the persistent cache.
- `BinaryModelCartesian`/`BinaryModelAngular` stay dedicated classes — the
  core binary-fitting path must not change or slow down. `to_system()` gives
  the equivalent `System` for extension and rendering.
