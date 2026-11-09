# `virgil.orbit_cells`

The period domain, the cells, the mode search and the position-level evidence
of the alias-resolved orbit fit: every cell of the period prior is searched
for the modes of the night-summary likelihood (minimal sets of nights, variable
projection, C-steps), and its evidence is a defensive importance-sampling
estimate from those modes and the images of the target's symmetries; the
posterior draws are an HMC resample-move.

::: virgil.orbit_cells
    options:
      members:
        - period_domain
        - choose_anchors
        - make_cells
        - outlier_prior
        - outlier_log_density
        - search_plan
        - search_cell
        - position_leg
        - smc_cell
        - smc_rounds
        - decode_particles
        - fit_jitter
        - position_cells
        - PeriodDomain
        - Cell
        - OutlierPrior
        - CellFit
        - CellsResult
