# `virgil.orbits`

Keplerian orbits of a binary's secondary about its primary, in virgil's
conventions: `dra` East, `ddec` North and `dz` away from the observer (mas);
`inc` below 90° turns the position angle forward; `Omega` is the position
angle of the node where the secondary recedes; `omega` is the secondary's
argument of periastron; and times are days since a static float64 `t_ref`.
Kepler's equation is solved by jaxoplanet, installed with
`pip install "virgil-astro[orbits]"`.

Radial velocities from several spectrographs can share an orbit fit without
fitting their zero points: `RVData(..., instrument=labels)` with
`RVData.term(params, marginalise_offsets=(mean, sd))` marginalises one velocity
zero point per instrument analytically (Luger, Foreman-Mackey & Hogg 2017,
arXiv:1710.11136), and `term.posterior(values)` reports them after the fit.
With a broad prior this is the profile likelihood plus a log-determinant
correction; only a finite prior width is supported.

::: virgil.orbits
    options:
      members:
        - KeplerOrbit
        - ThieleInnesOrbit
        - StateVectorOrbit
        - PositionData
        - RVData
        - starting_orbits
        - total_mass
        - distance_pc
        - AxialVonMises
