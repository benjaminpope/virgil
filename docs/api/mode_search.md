# `virgil.mode_search`

Mode search results turned into an evidence and posterior draws: a defensive
mixture of Student-t components at the modes of a multimodal target (and the
images of its exact symmetries), importance sampling for `ln Z`, a sentinel
for a missed dominant mode, and an HMC resample-move for posterior draws. The
position-level fit of [`virgil.orbit_cells`](orbit_cells.md) uses it; any
target with a known list of modes can.

::: virgil.mode_search
    options:
      members:
        - Mode
        - Evidence
        - DefensiveMixture
        - distinct_modes
        - laplace_log_evidence
        - importance_evidence
        - resample_move
        - logmeanexp
