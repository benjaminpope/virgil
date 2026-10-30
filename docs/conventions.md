# Conventions

Which way the axes point, what sign a phase has, what a flux means: none of these is hard, but reasonable people disagree on each, and a wrong guess flips a binary by 180° without an error. This page explains virgil's choices. Each statement has been checked against the code. Only some are pinned by a test, and a test is named where there is one; the others rest on that reading of the code.

1. [Sky coordinates and images](#sky-coordinates-and-images)
2. [Baselines and the Fourier sign](#baselines-and-the-fourier-sign)
3. [Observables](#observables)
4. [Closure phases with four or more telescopes](#closure-phases-with-four-or-more-telescopes)
5. [Fluxes](#fluxes)
6. [Times and frames](#times-and-frames)
7. [Orbits](#orbits)
8. [Which priors?](#which-priors)
9. [Priors and the MAP](#priors-and-the-map)
10. [Precision](#precision)
11. [Preconditioning image fits](#preconditioning-image-fits)

## Sky coordinates and images

Positions on the sky are offsets from a reference point, in milliarcseconds:

- `dra` is the right-ascension-like offset, **positive to the East**;
- `ddec` is the declination-like offset, **positive to the North**.

East is on the left of a sky image, so `dra` increases to the *left*.

**Position angle** (PA) runs from North through East, in degrees: 0° is North, 90° East, 180° South and 270° West, counter-clockwise on a sky image with North up and East left. A companion at separation $s$ and position angle $\theta$ has

$$
\mathrm{dra} = s \sin\theta, \qquad \mathrm{ddec} = s \cos\theta,
$$

which is how [`BinaryModelAngular`][virgil.models.BinaryModelAngular] relates to [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]. The inverse is $\theta = \operatorname{arctan2}(\mathrm{dra}, \mathrm{ddec})$ (note the argument order), modulo 360°. Every other position angle (an ellipse's major axis, an azimuthal modulation) follows the same rule.

### Images

Rendered images and `(dra, ddec)` grids are ordinary 2D arrays: the first index is the row, running from top to bottom, and the second is the column, running left to right. On the sky:

| Array index | Direction on the sky | Coordinate |
| --- | --- | --- |
| row 0 | top, North | largest `ddec` |
| row increases | downwards, towards South | `ddec` decreases |
| column 0 | left, East | largest `dra` |
| column increases | rightwards, towards West | `dra` decreases |

The origin `(0, 0)` is at the *centre of the centre pixel* (for an even number of pixels, the corner shared by the four central pixels), at index $(n-1)/2$ along each axis. A pixel's offset along one axis is $((n-1)/2 - \mathrm{index})\,h$ for pixel size $h$ in mas. Both `image_coordinates` and [`pixel_offsets`][virgil._geometry.pixel_offsets] follow this rule.

[`render`][virgil.models.SourceModel.render] draws a model on `npix` × `npix` pixels spanning `fov_mas`, so the pixel scale is `fov_mas / npix`; an [`Image`][virgil.models.Image] component sets it directly with `pixel_scale_mas`. Both use the orientation above. To check it, put a [`PointSource`][virgil.models.PointSource] at `dra=10` on a 64-pixel, 64 mas grid: the brightest pixel is in row 31, column 21, level with the centre and to its left. With `ddec=10` it is in row 21, column 31, above the centre.

If you plot a sky image yourself, put East on the left and North at the top; [`virgil.plotting`][virgil.plotting] does this for you. A mirrored image looks perfectly sensible, so the regression tests assert where the brightest pixel lands for a known offset (for example `test_gaussian_disk_render_uses_interferometric_image_orientation`).

## Baselines and the Fourier sign

A baseline is the vector between two telescopes, projected onto the plane perpendicular to the line of sight, with components $u$ (East-West) and $v$ (North-South) in metres. Models take `u`, `v` and a wavelength in metres and divide, so the **spatial frequencies** $u/\lambda$ and $v/\lambda$ are in cycles per radian. Positions in milliarcseconds are converted with $1\,\mathrm{mas} = \pi/(180 \times 3600 \times 1000)$ rad.

By the van Cittert–Zernike theorem, the complex visibility is the Fourier transform of the sky brightness $I(x, y)$, normalized to 1 at zero baseline. virgil's sign convention is

$$
V(u, v) = \frac{\iint I(x, y)\, e^{-2\pi i (u x + v y)}\,dx\,dy}{\iint I(x, y)\,dx\,dy},
$$

with $x$ = `dra` and $y$ = `ddec` in radians and $u, v$ in cycles per radian. This is the convention of `offset_phase` (in `virgil._geometry`), and every model and pixel-image Fourier transform in the package follows it. A point source at $(\mathrm{dra}, \mathrm{ddec})$ therefore has

$$
V = \exp\left[-2\pi i\,(u\,\mathrm{dra} + v\,\mathrm{ddec})\right],
$$

with the offsets in radians.

**What a positive phase means.** A point source's phase is *minus* $2\pi$ times the dot product of baseline and offset, in wavelengths. A source displaced East ($\mathrm{dra} > 0$), seen on a baseline with $u > 0$, has a *negative* phase: $\mathrm{dra} = 10$ mas on $u = 5$ m at 1 µm gives $-1.523$ rad. A positive phase means the source is displaced towards negative $u$, West on an East-West baseline.

**Where $u, v$ come from in OIFITS.** An OIFITS file gives each baseline's `UCOORD` and `VCOORD` in metres and its two telescopes in `STA_INDEX`. [`read_oifits`][virgil.oifits.read_oifits] takes `UCOORD` and `VCOORD` as they are, so virgil is only right if the file used the same convention: the vector runs from the first station of `STA_INDEX` to the second, $u$ points East and $v$ North, and the phase is $\arg V$ with the sign above. `tests/test_pa_round_trip.py` builds files this way in several instruments' layouts and checks that a binary comes back at its true PA. Real data test the *reduction pipeline*: if it conjugates phases or reverses baselines, the companion lands on the wrong side, and only a binary with a known orbit will show it. Treat the PA of a first fit to a new instrument as unverified until you have checked it against a known binary.

Two further details. If a closure triangle $(a, b, c)$ needs baseline $(a, b)$ but the file stores only $(b, a)$, the reader adds a sample at the triangle leg's own $(u, v)$, so the model is evaluated there directly. A baseline stored in neither orientation gets a flagged sample in the same way, at the leg's $(u, v)$ from the `OI_T3` row (`U1COORD`, `V1COORD` for $(a, b)$, `U2COORD`, `V2COORD` for $(b, c)$ and their sum for $(a, c)$), with a warning that the V² coverage is incomplete. Baselines are looked up in the visibility table of the same array (`ARRNAME`) and instrument (`INSNAME`), or else of another instrument on that array with identical wavelengths. AMIGO's mixed-DISCO products store $(u, v)$ with the opposite sign, which [`load_oi_data`][virgil.amigo.load_oi_data] negates on reading.

## Observables

`OIData` holds a visibility channel and a phase channel, each with an uncertainty.

**Visibilities** come in three forms, selected by `vis_mode`:

| `vis_mode` | Model value | Notes |
| --- | --- | --- |
| `"v2"` | $\lvert V \rvert^2$ | squared visibility, the commonest OIFITS product |
| `"amp"` | $\lvert V \rvert$ | visibility amplitude |
| `"logamp"` | $\ln \lvert V \rvert$ | log-amplitude |

The default, `"auto"`, keeps the data's own form; ask for another and data and errors are converted, propagating errors linearly, $\sigma_f = \lvert f'(V)\rvert\,\sigma_V$. Near zero, where noisy data can be negative, the derivative blows up, so it is evaluated with the data floored at their uncertainty: $V^2$ converted to amplitude gets $\sigma_{V^2} / (2\sqrt{\max(V^2, \sigma_{V^2})})$. Fit low-signal data in the form they were measured in.

**Phases** are one of:

- **closure phases** (`cp_flag=True`), for a triangle of telescopes $(a, b, c)$:
  $$
  \varphi_{abc} = \varphi_{ab} + \varphi_{bc} - \varphi_{ac},
  $$
  where $\varphi_{ab}$ is the visibility phase on baseline $(a, b)$. The legs are stored as sample indices `i_cps1`, `i_cps2` and `i_cps3`, so the closure phase is `phase[i_cps1] + phase[i_cps2] - phase[i_cps3]`. [`cp_indices`][virgil.oidata.cp_indices] builds the indices from station numbers and [`closure_phases`][virgil.oidata.closure_phases] evaluates the sum, wrapped into $[-\pi, \pi)$. Closure phases cancel telescope phase errors and any shift of the whole source;
- **absolute phases** (`cp_flag=False`), $\arg V$ of each sample, rarer because they need a phase reference.

Phases are **radians inside virgil**: every `phi` and `d_phi` attribute and the output of `closure_phases`. OIFITS stores degrees, and [`read_oifits`][virgil.oifits.read_oifits] converts on reading (from the column's unit, or assuming degrees if there is none). [`write_oifits`][virgil.oifits.write_oifits] does **not** convert: its tables hold phases and errors in degrees, as OIFITS stores them, so convert `phi` and `d_phi` with `numpy.rad2deg` before writing, or both come back about 57 times too small. A dictionary passed straight to `OIData` takes `phi_unit="deg"` for degrees.

**Wrapping and the likelihood.** A phase is an angle, so a model phase of $\pi - \epsilon$ and a data phase of $-\pi + \epsilon$ agree closely. For unprojected phases the residual $\Delta$ (model minus data) enters the likelihood as the chord

$$
r = \frac{2\sin(\Delta/2)}{\sigma},
$$

which is $\Delta/\sigma$ for small $\Delta$. Its square, and so the likelihood, is $2\pi$-periodic and smooth where the phase wraps: a von Mises likelihood with concentration $1/\sigma^2$. `OIData.residuals` instead wraps the difference into $[-\pi, \pi)$ for display; fits and likelihoods never use it, but use [`whitened_residuals`][virgil.likelihood.whitened_residuals].

**Flags.** A sample is *flagged* when its `FLAG` is set in the file or its value or uncertainty is not finite (the `vis_flag` and `phi_flag` masks mark bad samples `True`). Flagged samples are dropped from the observables, but `u`, `v` and `wavel` keep every sample. `vis_index` lists the visibility samples kept, and for absolute phases `phi_index` the phase samples (each `None` when nothing was dropped). Flagged closure phases are removed from `phi`, `d_phi` and the `i_cps*` arrays themselves, so `phi_index` stays `None` for closure phases even when some were dropped.

## Closure phases with four or more telescopes

Three telescopes give one triangle and one closure phase per frame and wavelength. $N$ telescopes give $N(N-1)(N-2)/6$ triangles, of which only $(N-1)(N-2)/2$ are independent: 3 of the 4 for four telescopes, 10 of the 20 for six. Triangles that share a baseline share noise, so treating them as independent counts the same information twice and makes fits overconfident.

virgil handles this by default. For each frame and wavelength it keeps only the independent combinations of triangles and whitens them with the covariance of Kammerer et al. (2020, A&A 644, A110). That assumes equal noise on every baseline phase, giving a correlation of $\pm 1/3$ between triangles that share a baseline (signed by whether it enters both in the same sense), with each triangle's reported error on the diagonal. When errors within a group differ a lot, the $\chi^2$ is slightly off its nominal distribution; the docstring of `virgil._closure` gives the size of the effect.

Use [`n_independent`][virgil.oidata.OIData.n_independent], the number of observables that remain, for degrees of freedom. [`whitened_residuals`][virgil.likelihood.whitened_residuals] returns one residual per independent combination (not the original triangles), then penalty residuals. The chord $2\sin(\Delta/2)$ won't do here: it changes sign when $\Delta$ changes by $2\pi$, which is harmless in one square but makes the $\chi^2$ of a correlated combination jump whenever a residual crosses $\pm\pi$, stalling gradient fits of high-S/N four-telescope data. Instead the smooth, periodic sines $s_i=\sin\Delta_i$ are whitened, and each closure phase adds an uncorrelated penalty residual $q_i/\sigma_i = 2\sin^2(\Delta_i/2)/\sigma_i$. For small residuals $\chi^2 = \Delta^\mathsf{T}C^{-1}\Delta$ up to $O(\Delta^3)$, the correlated Gaussian. At $\Delta_i=\pi$ the sines vanish, a false minimum that the penalty, $(2/\sigma_i)^2$, removes, so the $\chi^2$ is smooth everywhere. The penalty rows carry no normalization. So `whitened_residuals` has [`n_residuals`][virgil.oidata.OIData.n_residuals] entries, `n_independent` plus one per closure phase. For the Gaussian-process image prior that goes with these likelihoods, see [Gaussian processes and information field theory](gp_and_ift.md).

## Likelihood normalization

[`model_loglike`][virgil.likelihood.model_loglike] is $-\tfrac12\sum r^2$ over the residuals of [`whitened_residuals`][virgil.likelihood.whitened_residuals], plus each density's normalization, which matters when the errors are fitted:

- Visibilities and projected (kernel or DISCO) phases are Gaussian: $-\log\sigma - \tfrac12\log 2\pi$ per residual.
- An uncorrelated unprojected phase is von Mises with $\kappa = 1/\sigma^2$ and the exact normalizer $-\log 2\pi - \log i_{0e}(\kappa)$: Gaussian for $\sigma \ll 1$, and still normalized on the circle when $\sigma$ is large (a fitted `phi_error`, say). virgil returns it as the effective error $\sqrt{2\pi}\,i_{0e}(\kappa)$.
- Correlated closure phases keep the Gaussian normalization, their small-$\sigma$ limit, since a correlated circular density has no closed-form normalizer. Their periodic penalty rows carry the effective error $1/\sqrt{2\pi}$ and add nothing; the whitened rows carry effective errors whose log-sum is $\tfrac12\log$ of the covariance's pseudo-determinant.
- With gains or closure-phase offsets, the visibility (or phase) covariance is whitened with the gains marginalized, and the normalization includes $\tfrac12\log\det$ of that model-dependent covariance. The widths are the error terms `vis_gain_<group>` and `phi_offset_<group>`; see [`virgil.gains`][virgil.gains]. With three telescopes, closure-phase offsets change the closure-phase likelihood from von Mises to whitened sines with a periodic penalty, a normalization that differs by about $\log(1+\sigma^2/8)$ per closure phase, so evidences with and without offsets are not comparable.
- Extra observables follow the phases, each block whitened on its own (see [`virgil.observables`][virgil.observables]).

## Fluxes

In virgil, `flux` is always a **relative weight**, never an absolute brightness. The data force this: a visibility is normalized to 1 at zero baseline, so only flux ratios can be measured.

- In a [`System`][virgil.models.System] the visibility is the flux-weighted mean $V = \sum_i f_i V_i / \sum_i f_i$. Each component is a shape normalized to unit flux, and `flux` is its share of the total.
- Keep one **reference component**, usually the star, at `flux=1` and fit the others relative to it. With every flux free, scaling them all together changes nothing, and the fit is degenerate. A companion's `flux` is then its companion/star ratio: 0.01 for a companion 100 times fainter.
- In [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] and [`BinaryModelAngular`][virgil.models.BinaryModelAngular], `flux` *is* the companion/primary ratio, with the primary implicitly 1. That agrees with a `System` whose primary has `flux=1` (`BinaryModelCartesian.to_system` builds it).
- Fluxes are non-negative. [`Component`][virgil.models.Component]s and [`System`][virgil.models.System]s reject negative numeric fluxes when built, but values changed later with `set` or traced in a fit are not checked, and the two legacy binary models never check, so keep negative fluxes out of priors and grid axes. The one deliberate exception is [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid], whose best-fitting flux may be negative, as the Ruffio et al. upper limits need.

Reports and plots use the astronomer's convention. **Contrast** is primary/companion, 100 for a flux ratio of 0.01, and **Δmag** is $2.5\log_{10}(\text{contrast})$, 5 mag in that example. [`flux_to_contrast`][virgil.limits.flux_to_contrast], [`contrast_to_flux`][virgil.limits.contrast_to_flux], [`flux_to_delta_mag`][virgil.limits.flux_to_delta_mag] and [`delta_mag_to_flux`][virgil.limits.delta_mag_to_flux] convert between them, and the plotting functions take `units="flux"`, `"contrast"` or `"delta_mag"`. No model parameter is called `contrast`.

**Chromatic fluxes.** A component's `flux` may be a spectrum from [`virgil.spectra`][virgil.spectra], giving its weight at each wavelength (SPARCO; Kluska et al. 2014). The `ratio` of [`PowerLaw`][virgil.spectra.PowerLaw] or [`BlackBody`][virgil.spectra.BlackBody] is the flux weight *at the reference wavelength* `wavel0`, in metres (default $1.65\,\mu$m, H band); other wavelengths follow the spectrum's shape: $\mathrm{ratio}\,(\lambda/\lambda_0)^{\mathrm{index}}$ for the power law (index $-4$ for a Rayleigh–Jeans star in $F_\lambda$), and a Planck curve scaled to the same value at $\lambda_0$ for the black body. Rendering uses the reference flux, since an image has no wavelength. Put `wavel0` in the middle of your data so `ratio` is easy to read.

**Continuum plus lines, and excesses.** [`Sum`][virgil.spectra.Sum] adds named spectra, so a continuum and its emission or absorption lines are one flux, reached by name like the components of a `System` (`"star.flux.brg.amplitude"`):

```python
from virgil.spectra import GaussianLine, PowerLaw, Sum

flux = Sum(
    continuum=PowerLaw(0.3, index=-4.0, wavel0=2.2e-6),
    brg=GaussianLine(0.1, line_wavel=2.1661e-6, fwhm=2.0e-9),
)
```

A line's `amplitude` is its peak flux (negative for absorption), and `fwhm` is in metres. Every spectrum's reference flux is its value at `wavel0`, and only the *total* must be non-negative, so one part may be negative where the others cover it. For a spectrum measured channel by channel, [`Nodes`][virgil.spectra.Nodes] interpolates (linearly or by a natural cubic spline) between free fluxes at fixed wavelengths; `Nodes(excess, wavel, outside=0.0)` is an excess that vanishes outside its window and can sit on a continuum in a `Sum`. Smoothness is a prior on the node values, not a property of the class. `System.total_spectrum(wavel)` gives the scene's summed flux (the model of an OI_FLUX spectrum, up to a grey scale). `OIData.select` restricts data to wavelength windows (`wavel_min`, `wavel_max`, `ranges=`, `exclude=`) and to some observables (`observables=`), for example the continuum either side of a line, or closure phases alone.

## Times and frames

Times are Modified Julian Dates in days. At MJD 60000 float32 steps are 0.0039 d (about 5.6 minutes), coarse enough to quantize a binary's motion between nights, so virgil never keeps an absolute MJD in a JAX array.

`OIData` stores each sample's time as a static float64 `t_ref` (the earliest time in the data) plus a float32 array `dt` of days since `t_ref`, which resolves a difference of up to 1000 days to about 5 seconds. [`OIData.mjd`][virgil.oidata.OIData.mjd] rebuilds the absolute float64 times as a NumPy array for inspection and plotting. Write time-dependent models in terms of `dt`, never `mjd`.

Each sample also carries a `frame` number. A **frame** is one exposure of one instrument, the unit within which closure phases make sense, since a triangle's three baselines must be measured together. [`read_oifits`][virgil.oifits.read_oifits] groups rows into exposures by their `MJD` (within about 9 s), `INSNAME` and the baselines that closure phases tie together. With `frame_mjd="mean"` (the default) every sample in a frame gets the frame's mean time, which is what you want when a pipeline stamps the closure-phase and visibility tables with slightly different times; `frame_mjd="row"` keeps each row's own time.

To treat nights separately, [`OIData.epochs`][virgil.oidata.OIData.epochs] labels each sample with an epoch number: a run of frames with no gap longer than `gap_days` (default 0.5 d), numbered from 0 in time order. A frame is never split between epochs. [`OIData.split_by_epoch`][virgil.oidata.OIData.split_by_epoch] returns one `OIData` per epoch, for fitting a binary's position night by night; it does not work on projected (kernel or DISCO) observables.

For orbit fits, [`Epochs`][virgil.epochs.Epochs] groups datasets into named epochs and evaluates a moving scene once per dataset, at the mean time of its samples (a **snapshot**). That is exact for one time per dataset, and excellent whenever the scene moves much less than $\lambda/B$ within a dataset, as a binary with a period of months does over a night. [`Epochs.spread_days`][virgil.epochs.Epochs] says how far each dataset's samples lie from its snapshot. Each snapshot is a static model, so a binary on its orbit ([`OrbitalBinary`][virgil.models.OrbitalBinary]) keeps the fast binary path.

An orbit's `t_ref` is the zero of its `dt_peri`. Set it near the data (e.g. `KeplerOrbit(..., t_ref=60500.0)`): with the default `t_ref=0` and MJD times, `dt_peri` counts from MJD 0, and virgil warns.

## Orbits

[`virgil.orbits`](api/orbits.md) implements the conventions below, and [Orbits from interferometric data](orbit_fitting.md) uses them end to end, fitting an orbit to every epoch's visibilities and closure phases without being told the period.

An orbit gives the position of a **secondary** star relative to a **primary** (or reference) star, which sits at the origin as the scene's reference component at `flux=1`; it need not be the more massive star. The relative position is $\mathbf{r} = (\mathrm{dra}, \mathrm{ddec}, dz)$, secondary minus primary.

The primary is also the phase reference: a scene with an orbit has the visibilities of an image centred on the primary, not on the photocentre or barycentre. A component tied to the orbit by [`Attached`][virgil.models.Attached] sits on the secondary by default, on the primary with `anchor="primary"`, or a fraction of the way from primary to secondary (e.g. `anchor=q / (1 + q)` for the barycentre).

- The first two components are the sky offsets above, East and North, in mas.
- The third, `dz`, is positive **away from the observer**, so East, North and away form a right-handed set, and $d(dz)/dt$ has the sign of the secondary's radial velocity relative to the primary: positive means receding.

The angles follow the usual visual-binary conventions, each with one point to watch:

| Symbol | Name in code | Definition |
| --- | --- | --- |
| $i$ | `inc` | 0° to 180°. $i < 90°$ means the position angle *increases* with time (counter-clockwise on a North-up sky image) |
| $\Omega$ | `Omega` | the PA of the **ascending node**, defined as the node where the secondary *recedes* ($dz$ increasing) |
| $\omega$ | `omega` | the **secondary's** argument of periastron, from the ascending node, in the direction of motion |

This $\omega$ is the visual-binary one, for the secondary relative to the primary. The spectroscopic convention describes the *primary's* motion and differs by 180°: $\omega_{\rm spec} = \omega - 180°$, so correct orbits taken from a radial-velocity paper. Other symbols: `period` in days, `a_mas` the angular semimajor axis of the relative orbit in mas, and `dt_peri` the time of periastron minus `t_ref` in days, relative for the float32 reason above.

**What the data cannot tell apart.** These degeneracies belong to the geometry; they are not bugs.

1. $(\Omega, \omega) \to (\Omega + 180°, \omega + 180°)$ gives the same sky positions and flips the sign of $dz$. A visual orbit cannot tell them apart; radial velocities, or a scene component that is not front-back symmetric, can.
2. $\omega \to \omega + 180°$ alone sends $\mathbf{r} \to -\mathbf{r}$ at all times, the same as swapping the reference star. A "which star is the primary" error and an "$\omega$ of which star" error are the same 180° flip.
3. $i \to 180° - i$ reverses the sense of rotation on the sky.
4. For a nearly face-on orbit the positions depend on $\Omega$ and $\omega$ only through their sum, so astrometry measures the sum well and each alone poorly.

**Swapping primary and secondary** looks the same in the data. A companion at $\mathbf{r}$ with flux ratio $f$ and one at $-\mathbf{r}$ with ratio $1/f$ differ only in the image origin: in `BinaryModelCartesian`, `(dra, ddec, f)` and `(-dra, -ddec, 1/f)` give the same $V^2$ and closure phases to float32 rounding, and a visibility ratio of $\exp[+2\pi i(u\,\mathrm{dra} + v\,\mathrm{ddec})]$. So choose the reference once (the brighter star in the band, say) and read every PA, flux ratio and $\omega$ accordingly.

## Which priors?

Unless you have strong information (a measurement, a population model), use the Jeffreys prior under the group acting on each parameter, as virgil's examples do. It does not depend on how the parameter is written: log-uniform in period is log-uniform in frequency. Give every prior finite bounds with a margin around the plausible values, and a scale a nonzero lower bound, since `LogUniform(0, ...)` is undefined.

| Parameter | Group | Prior |
|---|---|---|
| Position offsets `dra`, `ddec`; time of periastron; phase offsets; spectral index; log gains | translation | `Uniform` |
| Fluxes and flux ratios; amplitudes; angular sizes and widths; periods; semi-major axes; noise and gain widths | scaling | `LogUniform`, with a nonzero lower bound |
| Position angle, node, periastron, spin, longitude | rotation of the circle | `AngleVector()` (no edge at 0°/360°), or `Uniform` over exactly one period |
| Inclination of an orbit or spin axis | rotation of the sphere | [`IsotropicInclination`][virgil.priors.IsotropicInclination] (uniform in cos i); over (0, 90) when only \|cos i\| is identifiable |
| Latitude of a point on a sphere | rotation of the sphere | [`IsotropicLatitude`][virgil.priors.IsotropicLatitude] (uniform in sin lat) |
| Eccentricity | none | `Uniform`, as the interim prior |
| Kipping's limb-darkening $(q_1, q_2)$ | none (uniform over the physical triangle) | `Uniform(0, 1)` each |

Two pitfalls. A `Uniform` prior over more than one period of an angle, such as (−360°, 720°), counts the circle several times. And a `Uniform` prior on a flux or size favours large values, so the posterior depends on the upper bound. Use the [binary search](binary_search.md) and [hierarchical inference](hierarchical_inference.md) tutorials as templates: both sample the companion's flux in log space, and the second infers a population of fluxes.

For orbits, [`orientation_priors`][virgil.orbits.orientation_priors] gives the node and periastron as angle vectors.

### Passing priors to functions

Prefer numpyro distributions: `dist.LogUniform(low, high)` and `dist.Normal(mean, sd)` work wherever a prior is a parameter (`fit`, `numpyro_model`, `noise=` and `linear_flux_grid(prior=)`). The exception is the analytically marginalized Gaussians, where a plain `(mean, sd)` pair (scalars, or one per instrument or telescope) stands for $N(\text{mean}, \text{sd}^2)$: `RVData.term(marginalize_offsets=(mean, sd))` and `with_flux_scale(scale=(mean, sd))`. A marginalized prior is never defaulted, so `marginalize_offsets=True` is an error.

### Angle priors without a wrap {#angle-vectors}

A prior on an angle $\theta$ with support on an interval, such as `Uniform(0, 360)`, numpyro's `VonMises` or [`AxialVonMises`][virgil.orbits.AxialVonMises], reaches samplers and [`fit`][virgil.fitting.fit] through a bijection onto the real line, so a posterior that straddles the end of the interval meets a wall. [`AngleVector`][virgil.angles.AngleVector] removes the wall. It samples a vector $v = r(\cos\theta, \sin\theta)$ at the site `"<path>_vec"`, and the model receives $\theta = \operatorname{atan2}(v_2, v_1)$ in degrees in $[0°, 360°)$, which [`numpyro_model`][virgil.likelihood.numpyro_model] records and `fit` reports under `"<path>"`.

The density of $v$ is

$$p(v) = \frac{e^{-(r-1)^2 / 2s^2}}{Z_s}\, p(\theta), \qquad Z_s = \int_0^\infty r\, e^{-(r-1)^2/2s^2}\,dr,$$

so $r$ and $\theta$ are independent and $\theta$ has exactly the density $p(\theta)$ (per radian): uniform, $1/2\pi$, by default; a von Mises, $\exp[\kappa(\cos(\theta - \mu) - 1)] / (2\pi\, \mathrm{i0e}(\kappa))$, given `mean` and `kappa`; or, with `axial=True`, the von Mises of $2\theta$, so that $\theta$ and $\theta + 180°$ are equally likely. The radius is a nuisance with a ring prior whose mode is the unit circle, which keeps a maximum a posteriori fit away from the origin, where $\theta$ is undefined. The ring width $s$ (`ring_width`, default 0.25) does not change the prior on $\theta$.

Every term has a least-squares form for Levenberg–Marquardt and [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass], and the density is normalized in $\mathbb{R}^2$, so evidences stay normalized. The ring's residual is $(r - 1)/s$. A von Mises enters as a chord, $\sqrt{\kappa}\,(\hat v - \hat m)$ with $\hat v = v / r$ and $\hat m = (\cos\mu, \sin\mu)$; half its square is $\kappa(1 - \cos(\theta - \mu))$, the von Mises exponent, just as virgil's phase residuals $2\sin(\Delta/2)/\sigma$ are chords with $\kappa = 1/\sigma^2$ (see [`whitened_residuals`][virgil.likelihood.whitened_residuals]). The axial chord is the same on the doubled direction, $(\cos 2\theta, \sin 2\theta) = (x^2 - y^2, 2xy)/r^2$.

The construction follows Octofitter's `UniformCircular` (Thompson et al. 2023, AJ 166, 164) and exoplanet's `Angle` (Foreman-Mackey et al. 2021, JOSS 6, 3285), which sample $v \sim N(0, I)$ and so peak at the origin; the ring and chord priors are virgil's own.

### Azimuthal harmonics {#azimuthal-harmonics}

At each radius, every azimuthal profile of a [`MultipoleDisk`][virgil.models.MultipoleDisk] is a real function of the azimuth $\theta$ (in the disk plane, from the projected major axis towards `pa + 90`) with the Fourier series

$$
\frac{I(\theta)}{I_0} = \sum_{m=-M}^{M} I_m e^{im\theta}, \qquad I_0 = 1, \quad I_{-m} = \bar I_m,
$$

whose *image harmonics* $I_m$ are complex, $m = 0 \dots M$, normalized to the azimuthal mean. Every form maps to and from them with [`to_image_harmonics`][virgil.harmonics.to_image_harmonics] and [`from_image_harmonics`][virgil.harmonics.from_image_harmonics] (or `Harmonics.convert`):

- **linear**: $1 + \sum_m c_m \cos m\theta + s_m \sin m\theta$, so $c_m = 2\,\mathrm{Re}\,I_m$ and $s_m = -2\,\mathrm{Im}\,I_m$ (exact);
- **positive**: $|h(e^{i\theta})|^2 / \sum_j |h_j|^2$ with $h(z) = 1 + h_1 z + \dots + h_M z^M$, whose roots all lie outside the unit circle. It is built from complex reflection coefficients $k_j$ with $|k_j| < 1$ by the Levinson step-up recursion $h^{(j)}_i = h^{(j-1)}_i + k_j \overline{h^{(j-1)}_{j-i}}$, with $k = w \tanh|w| / |w|$ from unconstrained latents $w_j \in \mathbb{C}$. Then $I_m = \sum_j h_{j+m} \bar h_j / \sum_j |h_j|^2$ (exact); the inverse is a spectral factorization (`from_image_harmonics`);
- **von Mises**, $\exp[\kappa\cos(\theta - \theta_0)]$, truncated at the model's degree;
- the legacy amplitude and azimuth pairs `az_amps`/`az_pas` of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim] and [`TruncatedCone`][virgil.models.TruncatedCone].

Use the conversions to report models in one convention, warm-start a positive-form fit from a linear one, or compare models. Report image harmonics, not the positive form's internal reflection coefficients.

References: Dumitrescu 2007, *Positive Trigonometric Polynomials and Signal Processing Applications* (Springer, doi:10.1007/978-1-4020-5125-8), for the Fejér–Riesz theorem, spectral factorization and the Schur–Levinson recursion; Barndorff-Nielsen & Schou 1973, J. Multivariate Anal. 3, 408 (doi:10.1016/0047-259X(73)90030-4), for the unconstrained parametrization of a minimum-phase polynomial by reflection (partial autocorrelation) coefficients, here with a complex tanh map.

### Hierarchical scales {#hierarchical-scales}

[`hierarchical_scales`][virgil.priors.hierarchical_scales] draws a set of scale factors, say one error scale per night, from a log-normal population, $\log s_k \sim N(\log \mu, \sigma^2)$, sampling the median $\mu$ and spread $\sigma$ under log-uniform priors. The population pulls poorly constrained members towards the typical value, and $\mu$ and $\sigma$ tell you how well the stated errors describe the instrument.

The **centred** form (the default) samples $u_k = \log s_k$ and adds the population density $N(u_k \mid \log\mu, \sigma)$. Use it when the data measure the members well, such as error scales of epochs with tens of closure phases or more: $u_k$ is then nearly independent of the hyperparameters. The **non-centred** form samples $z_k \sim N(0, 1)$ with $s_k = \mu\,e^{\sigma z_k}$; use it when the data barely constrain the members. With well-measured members it makes a curved ridge, $z_k \propto 1/\sigma$, on which NUTS diverges (in a test with eight members measured to 7%, 10 divergences in 4000 draws against none centred). The [hierarchical inference](hierarchical_inference.md) tutorial applies the same idea to a population of fluxes.

## Priors and the MAP

A maximum a posteriori point is not invariant under a change of variables, because a density picks up a Jacobian. The `LogUniform` density $1/x$ peaks at the lower bound, so a fit in $x$ would drag every scale down, although the prior prefers no scale.

So [`fit`][virgil.fitting.fit] optimizes each parameter with an invariant prior in its *flat coordinate*, where its prior is uniform: $\log x$ for `LogUniform(a, b)`, on $[\log a, \log b]$; $\cos i$ for an isotropic inclination; $\sin(\mathrm{lat})$ for an isotropic latitude (any prior with a `flat_coordinate()` method); and the parameter itself for `Uniform`. There the prior is constant, so the MAP is the likelihood maximum (times any other priors) within the prior's range, and Levenberg–Marquardt applies. Other priors (`Normal`, `Beta`, `HalfNormal`, ...) are evaluated in the model's own parameters, and fitted values are always reported in them.

Take a Gaussian approximation at the fit in the same flat coordinate and carry it to the model's parameters by the delta method, $\sigma_x = |{\rm d}x/{\rm d}u|\,\sigma_u$ (for a log-uniform scale, $\sigma_x = x\,\sigma_{\log x}$). [`laplace_cov`][virgil.inference.laplace_cov] and [`observed_information`][virgil.inference.observed_information] are curvatures of the likelihood alone, in the model's parameters, and do not depend on this. By default [`numpyro_model`][virgil.likelihood.numpyro_model] lets numpyro map each prior's support to NUTS's unconstrained coordinates. With `flat_coordinates=True` it samples in `fit`'s flat coordinates instead (NUTS moves in the logit of $u$ on its interval, where the prior is a logistic density), and [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] takes the same `flat_coordinates` to give its mass matrix in those coordinates. That matrix includes each coordinate's prior and transform curvature (the logistic transform adds $2\sigma(1-\sigma)$ in the logit's coordinate), as the sampler's potential does. Pass the same value to both: a mismatched matrix is never adapted, and [`chain_init_params`][virgil.likelihood.chain_init_params] (given `mass=`) raises on it. Samples are always in the model's own parameters, under the same site names.

## Precision

virgil never switches on JAX's 64-bit mode globally. Library code runs in float32 by default and is correct in float64 too. Fitting entry points such as [`fit`][virgil.fitting.fit] optimize in a local float64 context (`virgil._precision.run_in`), casting model, data and priors on the way in. Pass `dtype="float32"` to `fit` for a faster, less precise run.

Float32 is why absolute times are stored as `dt`. Matrix products and Fourier transforms over pixels use the highest matmul precision, because on A100 and H100 GPUs the default silently uses TF32, with a relative error of about $10^{-3}$.

### Fourier transform of pixel images

An [`Image`][virgil.models.Image] (and the `FlaredDisk` family) computes visibilities at irregular uv points either as an exact sum over pixels or with a non-uniform FFT (nufftax). The argument `ft` is `"auto"` (default), `"dft"` or `"nufft"`. With `"auto"` virgil uses the NUFFT only where it is faster: on a CPU for images of 64² pixels or larger with at least $\max(500, 2N)$ points ($N$ the larger side), and on a GPU for images of 128² or larger with 3000 or more points; otherwise it uses the exact sum. The NUFFT agrees with the sum to $10^{-5}$ in float32 and $10^{-12}$ in float64 of the zero-baseline visibility (the sum of the pixels), not of each visibility: a faint, low-$|V|$ point can be wrong by more of its own value, so pass `ft="dft"` when that matters. Data on a regular uv lattice, such as AMI (`data.uv_grid`), use the exact matrix Fourier transform whatever `ft` says, when the pixel grid's `rotation_deg` matches the lattice's; a `FlaredDisk` follows each data set's lattice by default (`rotation_deg="grid"`). Its pixels are then in the lattice's frame on lattice data and North up elsewhere (`disk.model(u, v, wavel)`, bandwidth smearing, data without a `uv_grid`), so the two paths differ by the pixel discretization, not the transform; a numeric `rotation_deg` fixes the frame for every path. `Rotated` does not pass the lattice on: its angle is traced inside fits.

[`set_ft_defaults`][virgil._geometry.set_ft_defaults] changes these defaults: the thresholds, the tolerances, and what `"auto"` means (`set_ft_defaults(ft="dft")` makes every model exact; `ft="nufft"` always uses the NUFFT). It returns the previous settings, so `set_ft_defaults(**old)` restores them. The settings are read when a function is traced, so a jitted function that has already compiled keeps its old choice. `jax.hessian` and nested `vmap` through the NUFFT fail in nufftax 0.6.1; use `ft="dft"` (or `virgil.set_ft_defaults(ft="dft")`) there.

## Preconditioning image fits

An image fit with a weak penalty (maximum entropy at a small weight) is badly conditioned: the faint pixels are almost unconstrained, so L-BFGS takes many tiny steps along them. `fit(..., precondition="diag")` rescales each coordinate by the square root of the diagonal of the Gauss–Newton curvature ([`gauss_newton_diag`][virgil.fitting.gauss_newton_diag]) for L-BFGS and Adam. The optimum is the same. In the two L-curves measured it took 30–40% fewer steps in total (more at a tight `gtol`), and at the weakest weight it sometimes converges where the plain fit reaches the step limit. LM already uses JᵀJ and ignores it. The option is provisional (see [stability](stability.md)).

```python
import jax
import jax.numpy as jnp

from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.imaging import MaxEntropy, image_priors
from virgil.models import Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

truth = System(star=PointSource(), env=Image.from_brightness(gaussian_blob(16, 12.0, 25.0, dra=20.0), 12.0, flux=0.1))
data = OIData(ami_grid_record(pitch_m=0.5)).with_model(truth, key=jax.random.PRNGKey(1))
start = System(star=PointSource(), env=Image(jnp.zeros((16, 16)), 12.0, flux=0.1))

result = fit(start, image_priors(start), data, [MaxEntropy(10.0, path="env")], precondition="diag")
result.info["precondition_rounds"]
```

`info["precondition_rounds"]` has one entry per L-BFGS round (one for Adam, none for LM): `decades`, the spread of the curvature before it is floored (the largest of any coordinate block), and `floored`, the fraction of coordinates held at the floor (1e-3 of the largest in their block). A spread of several decades is where it helps; a spread near zero means the plain fit was already well conditioned. `gtol`, `max_step_size` and `info["grad_norm"]` keep their meaning in the unscaled coordinates.

!!! warning "float32"
    At weak weights a float32 fit can still stall: the line search meets the noise of the float32 model evaluation, which rescaling does not remove. Use `dtype="float64"` (the default) for a sweep that reaches small weights.
