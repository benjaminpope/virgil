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

**Credit and related software.** Kepler's equation is solved by
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) (Hattori et al.,
[doi:10.5281/zenodo.10736936](https://doi.org/10.5281/zenodo.10736936)), the
JAX successor to exoplanet (Foreman-Mackey et al. 2021, JOSS 6, 3285); please
cite it with virgil when you fit orbits. The Thiele–Innes solve in
`starting_orbits` is the classical method (Thiele 1883, AN 104, 245;
Hartkopf, McAlister & Franz 1989, AJ 98, 1014). Analytic marginalisation of RV
zero points is also done by orvara (Brandt et al. 2021, AJ 162, 186). For
orbit fits to relative and absolute astrometry and RVs without an
interferometric scene, mature codes exist:
[orbitize!](https://github.com/sblunt/orbitize) (Blunt et al. 2020, AJ 159, 89),
[Octofitter](https://github.com/sefffal/Octofitter.jl) (Thompson et al. 2023,
AJ 166, 164; it also fits closure phases and kernel phases of point sources)
and [orvara](https://github.com/t-brandt/orvara) (Brandt et al. 2021). They
document the same conventions as virgil (the secondary's ω, +z away from the
observer), not yet checked numerically. virgil's orbits exist to drive
time-dependent scenes, extended components included, in JAX; for Hipparcos and
Gaia absolute astrometry use one of those codes and bring the result in as a
prior. See [`design/orbit_prior_art.md`](https://github.com/benjaminpope/virgil/blob/main/design/orbit_prior_art.md)
for a feature comparison.

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
