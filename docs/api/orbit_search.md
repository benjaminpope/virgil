# `virgil.orbit_search`

Scoring candidate orbits on multi-epoch data with the nuisances an orbit
shares across epochs shared in the score: one companion flux per band
(with an optional chromatic slope) integrated out on a log-uniform grid,
each dataset's error scales marginalized, calibration gains and
closure-phase offsets profiled, and extra terms such as radial velocities
added. Positions are evaluated at every sample's own time.

::: virgil.orbit_search
    options:
      members:
        - score_orbits
        - SharedFlux
        - OrbitScores
        - rank_scores
