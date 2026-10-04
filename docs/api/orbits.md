# `virgil.orbits`

Keplerian orbits of a binary's secondary about its primary, in virgil's
conventions: `dra` East, `ddec` North and `dz` away from the observer (mas);
`inc` below 90° turns the position angle forward; `Omega` is the position
angle of the node where the secondary recedes; `omega` is the secondary's
argument of periastron; and times are days since a static float64 `t_ref`.
Kepler's equation is solved by jaxoplanet, installed with
`pip install "virgil-astro[orbits]"`.

::: virgil.orbits
    options:
      members:
        - KeplerOrbit
        - ThieleInnesOrbit
        - StateVectorOrbit
        - PositionData
        - starting_orbits
