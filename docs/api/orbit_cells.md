# `virgil.orbit_cells`

The period domain, the cells and the position-level SMC leg of the
alias-resolved orbit fit: each cell of the period prior is sampled from its
prior to the night-summary likelihood, giving the evidence per cell.

::: virgil.orbit_cells
    options:
      members:
        - period_domain
        - choose_anchors
        - make_cells
        - valley_edges
        - outlier_prior
        - outlier_log_density
        - position_leg
        - decode_particles
        - fit_jitter
        - position_cells
        - PeriodDomain
        - Cell
        - OutlierPrior
        - LegResult
        - CellsResult
