# Orbit methods

The methods behind the helpers of [`virgil.orbits`](api/orbits.md). Angle and sign conventions are on the [conventions](conventions.md#orbits) page, and [Orbits from interferometric data](orbit_fitting.md) runs the orbit pipeline and the functions underneath it end to end.

## Orientation priors {#orientation-priors}

An orbit's node $\Omega$ and periastron $\omega$ wrap at 0°/360°, and positions alone cannot tell $(\Omega, \omega)$ from $(\Omega + 180°, \omega + 180°)$. [`orientation_priors`][virgil.orbits.orientation_priors] therefore returns [`AngleVector`][virgil.angles.AngleVector] priors on the well-defined angles, so that exact symmetries become single points rather than separate modes:

- `positions_only=True`: $2\Omega$ (that is, $\Omega$ modulo 180°) under `"two_Omega"`, and the longitude of periastron $\varpi = \Omega + \omega$ under `"varpi"`. Both twins $(\Omega, \omega)$ and $(\Omega + 180°, \omega + 180°)$ have the same $2\Omega$ and $\varpi$, so the two modes collapse to one.
- `positions_only=False`, when radial velocities or other data fix the node: `"Omega"` and `"varpi"`.

Near face-on the positions fix $\varpi$ but not $\Omega$ and $\omega$ separately, so $\varpi$ is well measured and $\Omega$ broad. Build the orbit with [`KeplerOrbit.from_varpi`][virgil.orbits.KeplerOrbit.from_varpi].

The angles are uniform, which together with a prior uniform in $\cos i$ is the invariant (Haar) prior on the orbit's orientation: $(\Omega, \omega) \to (2\Omega, \varpi)$ is linear with a constant Jacobian, so uniform $(\Omega, \omega)$ is uniform $(2\Omega, \varpi)$. With `inclination=True` the call also returns [`IsotropicInclination`][virgil.priors.IsotropicInclination] (density $\propto \sin i$ on 0° to 180°) under `"inc"`, which gives the full isotropic prior in one call. It is off by default so that code placing its own prior on `inc` keeps the keys it had.

## Starting orbits from positions {#starting-orbits}

At fixed period $P$, eccentricity $e$ and time of periastron $T$, the sky positions are linear in the four Thiele–Innes constants, so each point of a $(P, e, T)$ grid is an exact weighted least-squares solve. [`starting_orbits`][virgil.orbits.starting_orbits] solves every grid point and returns the lowest-$\chi^2$ orbits. This needs no random restarts and handles the several minima of a short arc. Refine the best orbits with a fit to the visibilities or the positions. Every returned orbit has $0 \le \Omega < 180°$, and $(\Omega + 180°, \omega + 180°)$ fits equally well.

**Equal-flux binaries (`sign_free=True`).** A binary of nearly equal fluxes has closure phases and $V^2$ that are the same for $(\mathrm{dra}, \mathrm{ddec})$ and $(-\mathrm{dra}, -\mathrm{ddec})$, so the sign of each epoch's position is arbitrary. At fixed $(P, T, e)$ the positions are $s_i M u_i$ with the unit-orbit points $u_i$ known and the $2\times2$ matrix $M$ (the Thiele–Innes constants) unknown. Two epochs fix $M$ up to one relative sign, so for the best-conditioned pairs of epochs and both relative signs, $M$ is solved, every sign is set to the one that agrees with $M u_i$, and the weighted fit is repeated twice on the signed positions. The result is the lowest $\chi^2$; the global sign (the point reflection of the whole orbit) stays free, so the orbit returned and its reflection ($\omega + 180°$) fit equally well.

## Period grids {#period-grid}

A period that is wrong by $\delta P$ drifts by $T\,\delta P / P^2$ cycles over the time baseline $T$, so trial periods must be spaced by no more than $\delta P = P^2 / (kT)$ for the best of them to stay within a fraction of a cycle of the truth at every epoch. That is a uniform grid in frequency $1/P$ with a step of $1/(kT)$, which puts many more periods at short periods than a log-spaced grid and does not miss short-period orbits on a long baseline. [`period_grid`][virgil.orbits.period_grid] builds it; neighbouring trial periods drift apart by $1/k$ of a cycle over the baseline.

The default $k = 9$ puts the true period within half a step of a trial period. Once the trial times of periastron have centred the phase, its error is at most $1/(4k) = 1/36$ of a cycle at the ends of the baseline: one step of the default `n_phase = 36` grid of times of periastron, so the two grids are about equally fine. This is finer than ARMADA's rule, $P = 2fT/n$ with $f = 3$ ($k = 6$), and The Joker's resolution, $\delta P = 4P^2/(2\pi T)$ ($k = \pi/2$). For $P = 5$–20 d over $T = 410$ d, ARMADA's rule gives about 370 periods and $k = 9$ about 550.

## Position angle at a reference epoch {#position-angle}

Astrometry measures the position angle $\theta$ at an epoch directly, so [`KeplerOrbit.from_position_angle`][virgil.orbits.KeplerOrbit.from_position_angle] replaces the time of periastron by $\theta$ at `t_ref` (as in Octofitter), which suits short arcs. With $\varphi = \theta - \Omega$, the argument of latitude $u = \omega + f$ at `t_ref` follows from $(\cos\varphi, \sin\varphi) \propto (\cos u, \sin u \cos i)$, that is $u = \operatorname{atan2}(\sin\varphi / \cos i,\ \cos\varphi)$. Then the true anomaly is $f = u - \omega$, the eccentric and mean anomalies $M$ follow, and `dt_peri` $= -MP/2\pi$ (in $[-P/2, P/2)$). Sample $\theta$ as an [`AngleVector`][virgil.angles.AngleVector] and add [`position_angle_prior`][virgil.orbits.position_angle_prior] to the likelihood terms, so that the prior stays uniform in the time of periastron.

The map is **singular at $i = 90°$**, where the position angle takes only two values ($\Omega$ and $\Omega + 180°$) and does not fix the phase; a concrete `inc` of 90° is rejected. Near edge-on it is badly conditioned: keep `dt_peri`, or use [`StateVectorOrbit`][virgil.orbits.StateVectorOrbit].

## Position calibration terms {#position-term}

[`PositionData.term`][virgil.orbits.PositionData.term] takes `north_angle=` and `plate_scale=`, names of fitted values holding a dataset's North angle $\delta$ (degrees, added to every measured position angle) and plate scale $m$ (a factor, 1 when calibrated), after Octofitter. As for the RV jitter, each is a key of `priors` (which a function model must accept and may ignore), and each dataset needs its own names. Their priors must be stated; there is no default width. The invariant priors (uniform on the circle for $\delta$, log-uniform for $m$) leave each one degenerate with the orbit's orientation and size when a single dataset is fitted, so a Gaussian such as `Normal(0, 0.1)` for $\delta$ and `Normal(1, 1e-3)` for $m$ is strong information and should come from the instrument's astrometric calibration. The terms do not change the likelihood's normalization, so a least-squares fit keeps its form.

## Night flags {#night-flags}

[`epoch_positions`][virgil.epochs.epoch_positions] flags a night whose position should not seed an orbit (`EpochPositions.flags`, each with a code and a reason). The thresholds are keywords; `None` disables a flag.

| Code | Raised when | Keyword (default) |
|---|---|---|
| `alias-peaks` | another peak is within this many nats of the best | `flag_alias_nats` (2) |
| `low-gap` | `gap_marginal` is below this many nats | `flag_min_gap` (5) |
| `sep-below-resolution` | the separation is below this fraction of the night's resolution $\lambda/2B_{\max}$ | `flag_sep_fraction` (0.5) |
| `flux-at-bound` | the flux is within this fraction of a flux bound | `flag_flux_tol` (0.02) |

The two nat thresholds scale with the data volume, so retune them for very different data. `sep-below-resolution` is reported only unless `flag_sep_blocks=True`, because $V^2$ and closure phases locate a companion below $\lambda/2B$ at good signal to noise. For data with gains or closure-phase offsets the scale integral holds many whitenings per grid point: `batch_size` (or `memory_budget`, in bytes) sets the points scored together, and a coarse pass at one fixed scale per block integrates $\ln s$ only within `coarse_nats` (30) of the best point and its `coarse_margin` (1) neighbours (`EpochPositions.exact_fraction` reports the share computed in full; `coarse_nats=None` integrates every point, which makes the result exact). The approximate scores outside that window feed `gap_marginal` and the choice of peaks to refine, so these two can change the peaks; `batch_size` and `memory_budget` never do.

## Alias-search settings {#alias-settings}

The defaults of [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases] need no per-system arguments; the knobs below are for tuning effort and diagnostics.

- **Effort per band:** `n_candidates`, `n_refine`, `n_random`, `min_distance_mas` (distinct modes), `max_steps`; `n_is` importance-sampling draws check each Laplace evidence (`flag_nats` is the `laplace-is-differ` threshold and `ess_min` the `low-ess` one); `n_samples` posterior draws are taken in the winning band and in every band with probability above `p_min_samples`.
- **Primary:** `primary_diam="auto"` fits a shared uniform-disk diameter to the per-night positions and, when the data resolve it (a gain of 5 nats over a point), lets the engine fit it in `diam_range`, by default 0.001 mas to the larger of $1.5\,\lambda/B_{\max}$ and twice the estimate; `"fit"` samples it in `diam_range`, by default (0.05, 5) mas. Either way the prior is log-uniform.
- **Short arcs:** `short_arcs=True` fits only the bands whose cycle counts the short arcs admit ([`virgil.arcs`][virgil.arcs]). `arc_jitter` is a position floor in units of $\lambda/B_{\max}$, `arc_delta_chi2` the $\chi^2$ margin around the best period, and `arc_min_gap` the tempered `gap_marginal / f` (nats) a night needs for its sign to count as decided; the cut's position and velocity errors are tempered by $f$ too. It needs `EpochPositions` and `skip_flagged`; otherwise every band is seeded from the positions.
- **Window and tempering:** `n_bands_max` caps the bands of a default `p_range` (a longer window is cut, with a warning); `temper` is estimated by default, a fixed factor, or `False`.
- **Memory:** `batch_size` and `memory_budget` change speed, never the result.

## Radial-velocity likelihood terms {#rv-term}

[`RVData.term`][virgil.orbits.RVData.term] returns a likelihood term for the `likelihoods=` argument of [`fit`][virgil.fitting.fit].

**Jitter.** `jitter=` names a fitted value holding an RV jitter $s$ (km/s), which inflates the errors to $\sqrt{d_{\rm rv}^2 + s^2}$. The normalization $\sum \log \sigma_{\rm eff}$ then depends on a parameter, so the term reports it as `log_norm`, `fit` adds it to the loss, and the default solver is L-BFGS (there is no least-squares form). Give the jitter a prior with positive support. The jitter is a scale, so the default is `dist.LogUniform(lo, hi)` (the Jeffreys prior); pick `lo` well below the smallest plausible scatter and `hi` above the largest. `dist.HalfNormal` is a deliberate informative choice (a scale about the expected scatter, a few km/s for a spotted star). The likelihood depends on $s^2$ only.

**Marginalized zero points.** `marginalize_offsets=(mean, sd)` analytically marginalizes one velocity zero point per instrument (Luger, Foreman-Mackey & Hogg 2017, arXiv:1710.11136). The model is $m_{\rm kepler} + A w$, with $A$ the indicator matrix of the instruments and $w_j \sim N(\mathrm{mean}_j, \mathrm{sd}_j^2)$. One zero point per instrument, with no separate $\gamma$, avoids a degeneracy: $w_j$ is the systemic velocity as measured by instrument $j$, and offsets are the differences $w_j - w_0$. Return `gamma = 0` from `params` (a nonzero value just shifts the prior mean). `mean` and `sd` are scalars or one per instrument, in the order of `RVData.instruments`, in km/s; `True` is an error, because the prior must be stated.

The density is the dense $N(m + A\mu,\ C + A \Lambda A^\top)$ log density with $C = \mathrm{diag}(\sigma_{\rm eff}^2)$, evaluated by the Woodbury identity and the matrix-determinant lemma in $O(Nk^2)$, so the jitter enters the full log-determinant. For a broad prior this is the profile likelihood plus a log-determinant correction; a flat prior is the $\Lambda \to \infty$ limit up to a $\Lambda$-dependent constant, and only finite `sd` is supported. The term's residuals have length $N$ and its `log_norm` carries the marginal log-determinant. `term.posterior(values)` returns the zero points' conditional mean and covariance, to report after a fit.

## Implementation notes and references {#references}

jaxoplanet (an optional dependency, `pip install "virgil-astro[orbits]"`) solves Kepler's equation with exact derivatives, and the positions follow from the Thiele–Innes constants. jaxoplanet's own conventions differ (its $\omega$ is the primary's and its third axis points toward the observer); they stay inside `KeplerOrbit.to_jaxoplanet` and `KeplerOrbit.from_jaxoplanet`.

The Kepler solver is jaxoplanet (Hattori et al., Zenodo, doi:10.5281/zenodo.10736936), which descends from exoplanet (Foreman-Mackey et al. 2021, JOSS 6, 3285). The linear Thiele–Innes fit at fixed $(P, e, T)$ is the classical method (Thiele 1883, Astron. Nachr. 104, 245), as used for visual binaries by Hartkopf, McAlister & Franz (1989, AJ 98, 1014), by Wright & Howard (2009, ApJS 182, 205) for linear parameters and by Lucy (2014, A&A 563, A126) for incomplete orbits, whose dense grid in $(P, e, T)$ with a linear solve `starting_orbits` follows. The frequency spacing of the period grid is that of a periodogram, as in Lucy (2014) and The Joker (Price-Whelan et al. 2017, ApJ 837, 20). The position angle at a reference epoch and the fits to closure phases follow Octofitter (Thompson et al. 2023, AJ 166, 164). No code is taken from these packages.
