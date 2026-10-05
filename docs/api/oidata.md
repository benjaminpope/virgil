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
2 sin²(Δ/2)/σ per closure phase, so the likelihood is continuous
everywhere (the correlated Gaussian for small residuals, with no false
minimum at Δ = π).

Closure phases from four or more telescopes are correlated: the triangles
of one frame and channel share baselines, and only some of them are
independent (three of four, for four telescopes). The likelihood keeps only
the independent combinations and whitens them with their covariance, built
from independent noise on the baseline phases. `n_independent` counts the
observables that remain (the degrees of freedom); `n_residuals` is the
length of `whitened_residuals`, which adds the penalty residuals.

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
        - n_independent
        - n_residuals
        - standardize_model
        - to_vis
        - to_phases
        - model
        - residuals
        - with_model
        - with_error_scale
        - mjd
        - epochs
        - split_by_epoch

## uv grids

AMIGO DISCO data are sampled on a regular uv lattice; `OIData.uv_grid`
records it, so that a matching `Image` can use an exact matrix Fourier
transform. Other data get `uv_grid=None`, even on a lattice; set it with
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