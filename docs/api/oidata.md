# `virgil.oidata`

Data containers and observable helpers for interferometric observations.

`OIData` maps each supported data product into a likelihood comparison vector.
For ordinary OIFITS-style data, this standardized vector concatenates the
visibility and phase observables. For AMIGO mixed-DISCO products, it is the
single mixed log-complex vector defined by the stored log-amplitude and phase
projection operators. In this context, "standardize" means "put data and model
predictions into the same comparison basis", not z-score normalization.

Every (baseline, wavelength) sample is one entry of `u`, `v` and `wavel`, so
data with several wavelength channels need no special handling in models.
Flagged samples are left out of the observables (see `vis_index` and
`phi_index`). `residuals` wraps phase residuals into `[-π, π)` for display.
Likelihoods and fits use
[`whitened_residuals`][virgil.likelihood.whitened_residuals] instead. For
ordinary unprojected phases, chord residuals `2 sin(Δ/2)` give a squared
contribution that is smooth across phase wraps. Correlated closure phases are
the exception: their residuals are wrapped into `[-π, π)`, combined, and
whitened together as sines sin Δ, with a periodic penalty
2 sin²(Δ/2)/σ (or `hypot(σ, σ_off)` when closure-phase offsets are present)
per closure phase, so the likelihood is continuous
everywhere (the correlated Gaussian for small residuals, with no false
minimum at Δ = π).

Closure phases from four or more telescopes are correlated: the triangles
of one frame and channel share baselines, and only some of them are
independent (three of four, for four telescopes). The likelihood keeps only
the independent combinations and whitens them with their covariance, built
from independent noise on the baseline phases. `n_independent` counts the
observables that remain (the degrees of freedom); `n_residuals` is the
length of `whitened_residuals`, which adds the penalty residuals.

## Calibration nuisances

`with_wavelength_scale` and `with_north_angle` are what the `noise=` terms
`wavel_scale` and `north_angle` of [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model] apply. A North angle δ
means the data see every position angle as the true one plus δ: a source at
`(dra, ddec)` appears at `R(δ) (dra, ddec)` with
`R(δ) = [[cos δ, sin δ], [-sin δ, cos δ]]`. Because the Fourier kernel is
`u dra + v ddec`, models are evaluated at `R(-δ) (u, v)`, as for
[`Rotated`][virgil.models.Rotated]`(model, δ)`.

A wavelength scale rescales the spatial frequencies (`u` and `v` are in
metres), so it magnifies the sky by `1 / scale`: it is also the plate-scale
term, and a sky magnified by m is `wavel_scale = 1/m`. With a single dataset
it is degenerate with every angular size, so its prior *is* the systematic
error (`dist.Normal(1.0, 2e-4)` is about right for GRAVITY).

Neither has a default prior. Their invariant priors (uniform on the circle
for δ) leave them degenerate with the scene's orientation and size for a
single dataset, so a Gaussian prior is strong information and its width
should come from the instrument's astrometric calibration. Long-baseline
interferometers know their baselines' orientation, so leave `north_angle`
out there. The same terms for published positions are in
[`PositionData.term`][virgil.orbits.PositionData.term]; they follow
Octofitter (Thompson et al. 2023, AJ 166, 164).

`with_closure_offsets` marginalizes closure-phase offsets (both per-frame
blocks and frame-spanning `modes`) as a linear perturbation: an offset δ
changes the whitened sine sin Δ by about δ cos Δ, which holds while the
offsets and residuals are small (δ ≲ 0.3 rad, 17°); larger widths are not
marginalized exactly. When `scale_penalty=True` (the default), the periodic
wrap penalty divides by `hypot(σ, σ_off)` at the nominal offset widths so
large instrumental offsets do not incur a spurious quartic penalty at small
photon noise.

## Attributes

- `uv_grid`: a [`UVGrid`][virgil.oidata.UVGrid] for AMIGO DISCO products,
  whose samples lie on a regular (possibly rotated) lattice: the mixed-DISCO
  product, and dictionaries with a `disco_vis_mat` or `disco_phi_mat` whose
  single-wavelength samples lie on one to 1e-5 cells (their float32 rounding).
  It is `None` for all other data, even on a lattice: virgil does not search
  ordinary data for one. Matching models, such as an
  [`Image`][virgil.models.Image] or a [`FlaredDisk`][virgil.models.FlaredDisk]
  with the same `rotation_deg`, then evaluate faster with the same results. For
  other lattice-sampled data set it with
  [`find_uv_grid`][virgil.oidata.find_uv_grid], e.g.
  `eqx.tree_at(lambda d: d.uv_grid, data, find_uv_grid(data.u, data.v),
  is_leaf=lambda x: x is None)`.
- `dt`, `t_ref`, `frame`, `mjd`: data read from OIFITS keep each sample's
  time and exposure. `frame` numbers the exposures, `mjd` gives each time,
  and `epochs` and `split_by_epoch` group the frames into nights. The time
  is stored as `dt`, days since the static float64 `t_ref`, so that models
  of time see small numbers that keep their precision in float32 (about 5 s
  over 1000 days). `stations` holds each sample's station pair
  (`STA_INDEX`).
- `gains` and `phase_offsets`: calibration gains
  ([`GainModes`][virgil.gains.GainModes], from `with_gains`) and
  closure-phase offsets ([`ClosureOffsets`][virgil.gains.ClosureOffsets],
  from `with_closure_offsets`) that the likelihood marginalizes; `None` by
  default.
- `extras`: further observables, read with `extras=` (OI_FLUX, T3AMP,
  VISAMP beside V², differential VISPHI beside closure phases): a tuple of
  blocks from [`virgil.observables`][virgil.observables], which follow the
  phases in the data vector in the order of `observables.KINDS`. Empty by
  default.

## Dictionary input

`OIData(dict)` takes these keys:

- `u`, `v` (metres) and `wavel` (metres): per sample, or `u`/`v` per
  baseline with `vis` of shape `(n_baseline, n_wavel)` for several channels.
- `vis`, `d_vis`: squared visibilities or amplitudes.
- `phi`, `d_phi`: closure or absolute phases, with `phi_unit` (`"rad"`, the
  default, or `"deg"`; `phase_unit` is an alias). For several channels,
  shape `(n_triangle, n_wavel)` or `(n_baseline, n_wavel)`.
- `i_cps1`, `i_cps2`, `i_cps3` (optional): for each closure phase, the
  baselines `(a, b)`, `(b, c)` and `(a, c)` of its triangle.
- `v2_flag` (default True): `vis` holds squared visibilities; otherwise
  amplitudes. `cp_flag` (default: True when closure indices are given):
  `phi` holds closure phases; otherwise absolute phases.
- `vis_flag`, `phi_flag` (optional): boolean masks, True for bad samples.
  Samples with non-finite values or errors are flagged automatically.
- `vis_mode` (`"auto"`, `"v2"`, `"amp"` or `"logamp"`): the visibility
  channel that data and model are compared in; `"auto"` keeps the channel
  of `vis`. The key `observable_vis_mode` is read as an old name for it.
  Neither sets the `vis_encoding` attribute, which says how `vis` and `phi`
  are laid out (`"split"`, or `"mixed_log_complex"` for an AMIGO mixed-DISCO
  product), not what they measure.
- `mjd`, `frame` (optional): the time (days) and an integer exposure label
  of each sample (or of each baseline, for several channels). Without
  `frame`, samples with the same `mjd` form one frame.
- `stations` (optional): the station pair `(a, b)` of each sample, or of
  each baseline, for [`with_gains`][virgil.oidata.OIData.with_gains].
- `vis_mat`, `phi_mat` (optional): linear operators of shape
  `(n_out, n_in)` projecting the channels into, e.g., kernel or DISCO
  observables. The `disco_vis_mat`/`disco_phi_mat` spellings also check
  that the projected covariance is diagonal. Only diagonal uncertainties
  are propagated.
- `flux`/`nflux`, `visamp`, `t3amp`, `visphi` (optional): extra observables
  as dictionaries in the layout [`read_oifits`][virgil.oifits.read_oifits]
  returns. `visamp` and `visphi` give per value the `sample` it belongs to,
  with `value`, `error` and `flag` (`visamp` also `amptyp`); `t3amp` has one
  value per closure phase; `flux` gives `wavel`, `value`, `error`, `flag`
  and optionally `row`, `mjd` and `station`.

A record with `disco_coefficients` is read as an AMIGO mixed-DISCO product
(see [`load_oi_data`][virgil.amigo.load_oi_data]); its `u` and `v` are
negated to match the virgil sign convention.

The bundled `data/calibrated_visibility.npy` fixture is synthetic; see the
AMIGO DISCO tutorial and [`virgil.amigo`](amigo.md) for loading it.

## Classes

::: virgil.oidata.OIData
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - flatten_data
        - has_phases
        - unflagged
        - n_independent
        - n_residuals
        - standardize_model
        - to_vis
        - to_phases
        - model
        - residuals
        - with_model
        - with_error_scale
        - with_error_floor
        - with_continuum
        - with_flux_scale
        - has_model_covariance
        - with_gains
        - with_wavelength_scale
        - with_bandwidth
        - with_north_angle
        - with_closure_offsets
        - mjd
        - epochs
        - split_by_epoch
        - select

## uv grids

AMIGO DISCO data are sampled on a regular uv lattice; `OIData.uv_grid`
records it, so that a matching `Image` or `FlaredDisk` can use an exact
matrix Fourier transform. Other data get `uv_grid=None`, even on a lattice; set it with
`find_uv_grid` (see `OIData`) to use the fast transform for them.

::: virgil.oidata.UVGrid
    options:
      show_root_heading: true
      heading_level: 3

::: virgil.oidata.find_uv_grid
    options:
      show_root_heading: true
      heading_level: 3

## Functions

::: virgil.oidata
    options:
      show_root_heading: false
      members:
        - closure_phases
        - cp_indices