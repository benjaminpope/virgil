# `virgil.epochs`

Datasets grouped into named epochs, with one snapshot of a moving scene
per dataset, for orbit fits to visibilities and closure phases across
epochs, and the tools that start and sample those fits: ranking trial
orbits by the data, choosing distinct starts for each chain, and starting
from a fit of positions.

::: virgil.epochs
    options:
      members:
        - Epochs
        - rank_orbits
        - RankedOrbits
        - chain_starts
        - epoch_positions
        - marginal_loglike
        - EpochPositions
        - EpochPeaks
        - start_from_positions
        - OrbitStart
