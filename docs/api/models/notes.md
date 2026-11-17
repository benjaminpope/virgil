# Model notes

Derivations, conventions and accuracy checks for the models in
[`virgil.models`](index.md) and [`virgil.fields`](../fields.md), kept out of
their docstrings. Each section is linked from its class.

## GaussianArc

The brightness is a circle of `radius` about the centre of curvature,
convolved with a circular Gaussian of FWHM `width` and weighted along it by a
von Mises function of the position angle $\varphi$ seen from the centre,
$\exp[\kappa\cos(\varphi - \mathrm{pa})]$. The concentration
$\kappa = (2\sqrt{2\ln 2}\,\mathrm{radius} / \mathrm{length})^2$ makes
`length` the FWHM of the weight along the arc, as an arc length, for an arc
short against its radius. For a large radius the model tends to an elliptical
Gaussian with FWHMs `hypot(length, width)` along the arc and `width` across
it; as `length` grows it tends to a uniform ring.

The weight is the periodic version of a Gaussian in arc length. Before 0.4.0
it was a Gaussian taken once round the circle and cut off at the antipode of
`pa`; the two agree for arcs short against the radius and differ when
`length` is comparable to $\pi\,\mathrm{radius}$, so refit if you relied on
long arcs.

The visibility is exact up to the truncation of the von Mises harmonics at
`degree`, chosen when the arc is built from `radius` and `length` so that the
neglected harmonics are below $10^{-6}$ of the mean. Arcs narrower than that
allows (more than 256 harmonics: `length / radius` below about 0.05, a FWHM of
about 3 degrees) are instead integrated along the arc at `nodes` points, as
the image is, at a cost independent of the width; `quadrature` says which was
chosen. Pass `degree` to force the series. Fitting `length` to much narrower
arcs than the starting ones needs a larger `degree`, and `is_physical` is
false once the series drops more than 1e-4 of its power.

## ArchimedeanSpiral

The arm is $r = |\mathrm{pitch}|\,\theta/2\pi$ with $\theta$ in radians from 0 to
$2\pi\,\mathrm{turns}$, in a plane whose azimuth $\alpha$ is measured from the
projected major axis in the sense of position angle: $\alpha = \alpha_0 +
\mathrm{sign}(\mathrm{pitch})\,\theta$. The plane is inclined as for
[`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim] (the minor axis
shortened by $\cos(\mathrm{inc})$, the major axis at `node`), and $\alpha_0$ is
fixed so that the arm leaves the centre at sky position angle `pa`. Along the
arm the brightness per unit arc length is $\exp(-\mathrm{fade}\,\theta/2\pi)$,
and the whole is blurred by a Gaussian of FWHM `width` that is round in the
arm plane. The geometry (a linear, Archimedean spiral of a wind carried
outward by an orbiting binary) is that fitted to WR 104 by Tuthill et al.
(1999, 2008) and used for Apep by Han et al. (2020); the brightness law is not
taken from them. Tuthill et al. (2008) measure a profile
along WR 104 that rises, peaks about 90 degrees downstream, then declines
roughly monotonically, and no closed form is given. A one-parameter
exponential fade is a modelling choice; `fade = 0` is a uniform arm.

The sky positions of the $N$ nodes $\theta_k$ (equally spaced in $\theta$) are
$\mathbf{x}_k$, with weights $w_k \propto \exp(-\mathrm{fade}\,\theta_k/2\pi)
\sqrt{1+\theta_k^2}$ (the arc-length element, so the nodes sample the arm
uniformly in length) and half weight at the two ends (trapezoidal rule). Then

$$V(\mathbf{q}) = \Big[\sum_k w_k\, e^{-2\pi i\,\mathbf{q}\cdot\mathbf{x}_k}\Big]
\exp\!\Big[-\frac{\pi^2\,\mathrm{width}^2\,|\mathbf{q}'|^2}{4\ln 2}\Big],$$

where $\mathbf{q}'$ is $\mathbf{q}$ rotated and shortened into the arm plane,
and the image is the same Gaussian blobs summed on the pixel grid. The image
floors `width` at one pixel.

**Choosing `nodes`.** A chain of Gaussians a distance $d$ apart has the
transform of the continuous ridge plus copies shifted by $1/d$ along the
arm. At a baseline $q$ the first copy is down by
$\exp[-3.56\,(\mathrm{width}/d - \mathrm{width}\,q)^2]$, which is below
$10^{-5}$ for $\mathrm{width}/d \ge \mathrm{width}\,q + 2$. The default places
nodes at most $\mathrm{width}/4$ apart (in arc length, the largest spacing
being at the end of the arm), safe up to $\mathrm{width}\,q = 2$, where the arm's
own transform is already $10^{-6}$; pass a larger `nodes` for the longest
baselines of a very narrow arm, in proportion to $\mathrm{width}\,q_\mathrm{max}+2$.
The trapezoidal rule also leaves an end-point error of about
$2\pi q h/(6N)$ with $h$ the node spacing, below $10^{-3}$ at the default.
Because `nodes` is static, give it explicitly when the shape parameters are
traced (the default is computed from their starting values), and re-create
the model with a larger value if a fit lengthens the arm substantially.

## TruncatedCone

**Rings.** The cone is a stack of rings about its axis. A ring of radius ρ in
the plane perpendicular to an axis tilted β out of the sky projects to an
ellipse, whose visibility is $J_0(2\pi \rho q)$ with
$q^2 = q_\perp^2 + (\mathrm{ratio}\,q_\parallel \sin\beta)^2$, where
$q_\parallel$ and $q_\perp$ are the spatial frequencies along and across the
projected axis, times the phase of the ring's centre, which lies
$(s\cos\alpha - \mathrm{tip})\cos\beta$ along the projected axis. The rings
are weighted by the area element (∝ ρ) and the emissivity, and integrated over
$s$ from $s_0$ to $s_0 + 5\,\mathrm{length}$ (the last 0.7 % of the flux is
dropped) by a Gauss-Legendre rule on `n_rings` rings.

**Azimuthal modulation.** `az_amps` and `az_pas` brighten one side of the
walls, e.g. the leading edge of a colliding-wind shock that the orbit sweeps
round. Every ring's brightness is multiplied by
$1 + \sum_m A_m \cos(m(\varphi - \varphi_m))$ in its own azimuth φ, the same
modulation along the whole length of the cone. As for
[`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim], azimuths are
measured in the plane of the ring, in the same sense as position angle.
φ = `pa + 90` and `pa - 90` are the two walls seen across the projected axis,
at those position angles on the sky; φ = `pa` is the side of each ring that
projects towards `pa` on the sky when `tilt > 0` (towards `pa + 180` when
`tilt < 0`). By the Jacobi-Anger expansion a ring adds
$A_m (-i)^m J_m(2\pi\rho q)\cos(m(\theta - \varphi_m))$ to $J_0$, with θ the
direction of the spatial frequency in the ring's plane, so the modulation
stays analytic. A modulated cone no longer looks the same at `tilt` and
`-tilt`: the image at `-tilt` is the one at `tilt` with `az_pas` mirrored to
`2 pa + 180 - az_pas`, which leaves the walls unchanged. Bound to an orbit's
frame by [`Attached`][virgil.models.Attached] (`bind={"az_pas": ...}`), a sky
position angle is deprojected into the rings' azimuth, so the bright side
points at it on the sky.

**Choosing `n_rings`.** The quadrature converges spectrally, not at second
order as the midpoint rule it replaced did: the error is negligible once the
nodes resolve the fringes, and does not change with more. Fringes of the
rings' centres and radii run at most
$q\max(\sin\alpha\max(1,\mathrm{ratio}), \cos\alpha\cos\beta)$ cycles per
milliarcsecond of slant distance, so about $\pi q\,5\,\mathrm{length}$ times
that factor nodes are safe (fewer often do). For `length` 13.8 mas and
`alpha` 62.5°, 24 nodes agree with a converged midpoint stack to 1e-5 for
baselines out to 0.1 cycles/mas, and 48 to 1e-8 out to 0.3 cycles/mas (where
256 midpoint rings are still off by 1e-5 and 64 by 5e-4), so the default of
48 suits baselines to about 130 m at 2.2 µm. Too few nodes is not a smooth
loss of accuracy: the error stays large and then drops sharply, so check by
doubling `n_rings`: if |Δχ²| ≳ 1 per dataset (|Δ log L| ≳ 0.5), use more. The
cost is linear in `n_rings` and about half the midpoint stack's at the same
accuracy.

## GravityDarkenedStar

**Grey mode** (`t_pole=None`, Dholakia's model): each triangle is weighted by
its bolometric flux times its projected area, the same at every wavelength, so
wavelength enters only through `u / wavel`.

**Chromatic mode** (`t_pole` set): each triangle has temperature
$T = T_\mathrm{pole}\,T_\mathrm{eff}/T_\mathrm{eff,pole}$ from the ELR11
gravity darkening, radiates the Planck function $B_\lambda(T)$, and is weighted
by its projected area times that, at each sample's own wavelength. The hot
pole and cool equator then have a contrast that rises towards short
wavelengths. Limb darkening and the bandwidth smearing of the star's own
structure are not modelled (opt-in smearing of a component's offset is
described under [`smeared_model`][virgil.models.smeared_model]). The weights
have shape `(n_samples, n_triangles)`, so memory is about 200 MB of complex64
for $10^4$ samples at the default `n_lat`.

In chromatic mode the star supplies its own spectrum to a
[`System`][virgil.models.System]: its weight is `flux` times the summed Planck
flux of its visible surface, relative to that at `wavel0`. A companion then
gets a physically consistent flux ratio at every wavelength with no separate
stellar spectrum, and `flux` must be a number, not a
[`Spectrum`][virgil.spectra.Spectrum], which would count the spectrum twice:

```python
star = GravityDarkenedStar(1.0, omega=0.9, inc=45.0, t_pole=9000.0)
companion = PointSource(flux=BlackBody(0.01, 3000.0))
system = System(star=star, companion=companion)
```

**Dholakia's parameters.** His `ELR_Model` parameters map onto these as follows
(his angles are in radians):

| his | here |
| --- | --- |
| `diam` | `diam_eq` (his `r_eq` is `diam_eq / 2`) |
| `omega` | `omega` |
| `inc` (0 = equator-on) | `90 - inc` degrees |
| `obl` | `pa` degrees |

**Choosing `n_lat`.** The mesh is a second-order quadrature: the error in the
visibility falls as `1 / n_lat**2`, so each doubling of `n_lat` cuts it by
about 4. An independent ELR11 reference (virgil-validation) measured it. In
grey mode the error is 1.4-3.4e-4 in visibility at `n_lat = 128`, falling 4x
per doubling. At the default `n_lat = 32` it is 2-6e-3 for fast rotators. In
chromatic mode (`t_pole` 9000 K) it is 1.3-1.4e-4 at 0.7, 1.65 and 2.2 µm,
also at `n_lat = 128`. The default suits most data, but well-measured data
(e.g. GRAVITY) can be sensitive to errors of this size.

To check, evaluate or refit at the best fit with `n_lat` doubled and compare
χ²: if |Δχ²| ≳ 1 per dataset, use more latitude rings, and double again until
the change is below that. The cost is O(`n_lat`$^2$) per baseline.

## ModulatedGaussianRim

In polar coordinates $(r, \phi)$ in the plane of the rim, the thin ring is
$\delta(r - \mathrm{diam}/2)\left(1 + \sum_{m=1}^{n} A_m
\cos(m(\phi - \mathrm{pa}_m))\right)$, convolved with an isotropic Gaussian of
FWHM `fwhm` in that same plane. The in-plane azimuth $\phi$ is counted in the
same sense as position angle, with $\phi = \mathrm{pa}$ along the major axis.
This face-on image is then compressed by $\cos(\mathrm{inc})$ along the minor
axis, so on the sky the blur is an elliptical Gaussian with FWHM `fwhm` along
the major axis and `fwhm * cos(inc)` along the minor axis. An unmodulated rim
therefore has the same peak brightness all the way round, without bright
ansae at the ends of the major axis.

So `az_pas` are in-plane (deprojected) angles. A point at in-plane azimuth
$\phi$ appears at the on-sky position angle $\theta$ with
$\tan(\theta - \mathrm{pa}) = \cos(\mathrm{inc})\tan(\phi - \mathrm{pa})$, in
the same quadrant. The two agree for a face-on rim and along the major and
minor axes; otherwise a peak at `az_pas` appears on the sky at a position
angle closer to the major axis.

## FlaredDisk

The brightness follows Eqs. 2–9 of Blakely et al. (2024). Sky offsets are
rotated so that $x$ runs along the major axis and $y$ along the minor axis,
positive towards the near side, and $y$ is divided by $\cos i$ to give
mid-plane coordinates. The surface height
$z = h\,r_0\,(\rho/r_0)^\beta$, with $\rho = \sqrt{x^2 + y^2}$ and
$h$ = `aspect`, raises the apparent radius to
$r = \sqrt{x^2 + (y + z\sin i)^2 + z^2}$, which shifts the ring towards the far
side. The brightness is

$$I(r, \theta) = \left(f(\theta) + A_s/A_a\right)
\exp\left(-\frac{(r - r_0)^2}{2\sigma_r^2}\right)
\frac{1}{2}\left(1 + \mathrm{erf}\left(
\frac{\alpha (r - r_0)}{\sqrt{2}\sigma_r}\right)\right),$$

where $\theta = \arctan(x / y)$ is the mid-plane azimuth from the near-side
minor axis and $f$ the phase function. The paper gives the height as
$H_{100}$ (au) at 100 au, which is
`aspect` $= (H_{100} / 100\,\mathrm{au})(r_0 / 100\,\mathrm{au})^{\beta - 1}$
with $r_0$ in au; its fitted fluxes $A_a, A_s$ are absolute, and here only
their ratio and the disk's total `flux` enter.

## Harmonics

The visibility of a [`MultipoleDisk`][virgil.models.MultipoleDisk] needs only
the orders `0..M`, from one call to `bessel_jn`.

**`form="positive"`.** $I/I_0 = |h(e^{i\theta})|^2 / \sum_j |h_j|^2$, with
$h(z) = 1 + h_1 z + \dots + h_M z^M$. By the Fejér-Riesz theorem every
non-negative trigonometric polynomial of degree `M` is of this form, so the
image is positive for every parameter value and every positive image of degree
`M` is reachable. The polynomial has a root-flip ambiguity (reflecting a root
through the unit circle leaves the image unchanged, giving up to $2^M$ copies
of every solution), which the parametrization removes by taking the
minimum-phase factor, built from complex reflection coefficients $k_j$
($|k_j| < 1$) by the step-up recursion
$h^{(j)}_i = h^{(j-1)}_i + k_j \overline{h^{(j-1)}_{j-i}}$, with
$k_j = w_j \tanh|w_j| / |w_j|$ from the unconstrained latents
$w_j = \mathrm{re}[j - 1] + i\,\mathrm{im}[j - 1]$ (Cartesian, finite
gradients at $w = 0$, a uniform ring). There is no label switching and no
constraint, so optimizers and samplers see a smooth unimodal surface. The
latents are an internal coordinate: for weak asymmetry $h_m \approx k_m$, so
$I \approx I_0 (1 + 2\,\mathrm{Re} \sum_m h_m e^{im\theta})$, but the
quantities to report are the image harmonics,
$I_m = \sum_j h_{j+m} \bar h_j / \sum_j |h_j|^2$
([`image_harmonics`][virgil.models.Harmonics.image_harmonics]).

**`form="linear"`.** $I/I_0 = 1 + \sum_m c_m(r)\cos m\theta + s_m(r)\sin m\theta$,
exactly linear in the Cartesian coefficients (no funnel as an amplitude goes to
zero and no wrap of a phase; a Gaussian prior on them is a uniform prior on the
phase). It reproduces `ModulatedGaussianRim` and `TruncatedCone`, but does not
keep the image positive: the sum of the amplitudes,
`sum(sqrt(c_m**2 + s_m**2)) <= 1`, guarantees it (an rms of the coefficients at
most 1 does not).

**References.** The positive form is the Fejér-Riesz spectral factorization of
a non-negative trigonometric polynomial (Dumitrescu 2007, *Positive
Trigonometric Polynomials and Signal Processing Applications*, Springer), with
the minimum-phase factor parametrized without constraint by reflection
coefficients (Barndorff-Nielsen & Schou 1973, J. Multivariate Anal. 3, 408).

## VonMises

The profile $\exp[\kappa\cos(\theta - \theta_0)]$ has the analytic harmonics
$c_m + i s_m \propto [I_m(\kappa)/I_0(\kappa)]\,e^{-im\theta_0}$, with $I_m$ the
modified Bessel function; this fixes the sign convention of `theta0`. They are
computed by a small FFT of $\exp[\kappa(\cos\theta_j - 1)]$, so no special
function is needed, and are differentiable in $\kappa$ and $\theta_0$. The FFT is
accurate to the aliasing error $I_{n_{\rm fft}-m}/I_0$, negligible for $\kappa$
below about `n_fft**2 / 30`. The brightness is the truncated series, not the von
Mises function itself.

## RadialGP

A non-parametric profile of `n_r` values is exactly a quadrature of the Hankel
transform, which is why the nodes are the disk's Gauss-Legendre nodes.

**Priors.** The invariant priors of `sigma` and `ell` are log-uniform on stated
bounds (`dist.LogUniform`), which also make the evidence proper. Fitted by MAP
they are biased, so fix them, sample them, or choose them by the evidence of
[`log_evidence`][virgil.imaging.log_evidence] at several values (that evidence
is over the latents, conditional on the hyperparameters).

**What `ell` means.** The Matérn length is a correlation length of $g$ in
milliarcseconds, not the width of a feature. A squared-exponential bump with
the same curvature at the peak as the Matérn-$\nu$ kernel has width
$\ell\sqrt{(\nu - 1)/\nu}$, and the prior's rms slope of $g$ is
$\sigma\sqrt{\nu/(\nu - 1)}/\ell$ per milliarcsecond. For the default
$\nu = 3/2$ these are $\ell/\sqrt3$ (so $\ell = 1.5$ mas is a Gaussian $\sigma$
of about 0.9 mas) and $\sqrt3\,\sigma/\ell$; for $\nu = 1/2$ $g$ is not
differentiable and there is no finite slope. A large `sigma` with a short `ell`
is a very rough prior (a short `ell` also gives many maxima, which `sigma` does
not change). Visibilities cannot constrain structure finer than the half-fringe
$\lambda/2B$ of the longest baseline, so below that scale, and where the data
have no leverage (the inner edge, the outer tail), the posterior is the prior.
A modest evidence difference between two `ell` values means that the data do
not choose between them: report `ell` as a prior choice.

**References.** Whitened Gaussian-process latents at radial nodes,
$g = \mu + L z$, sampled by NUTS, and a Gaussian process in the radius for each
quantity follow FRAP (Yoshida et al. 2026, ApJ 1007, 198, arXiv:2607.12618),
and the Gaussian-process prior on a radial profile frank (Jennings et al. 2020,
MNRAS 495, 3209). The code is a new implementation; none was copied.

## MultipoleKernel

The variance of the order-$m$ coefficients is

$$
\mathrm{Var}[c_m] = \sigma^2 \frac{S(m)}{S(1)}, \qquad
S(m) = \left[1 + \frac{1}{2\nu_\theta}
\left(\frac{m\,\ell_\theta}{\bar r}\right)^2\right]^{-\nu_\theta - 1/2},
$$

($S(m) \propto e^{-(m\ell_\theta/\bar r)^2/2}$ for $\nu_\theta = \infty$), with
$\bar r$ the mean radius of the range, so `ell_theta` is on the same scale as
`ell_r`. The prior is diagonal in $m$ (no extra cost), and $\sigma$ is the
standard deviation of the first-order coefficients. The positive form's latents
carry the same prior, so the variance falling with $j$ plays the part of
falling with $m$. Coefficients are interpolated linearly in $r$.

A separate Gaussian process in the radius for each quantity follows FRAP
(Yoshida et al. 2026, ApJ 1007, 198, arXiv:2607.12618).

## MultipoleDisk

**Symmetry.** Without `harmonics` or `height` the disk is azimuthally symmetric
and point-symmetric (whatever `inc`), so when centred its visibilities are real
and its closure phases are 0 or 180 degrees (the visibility changes sign past
its nulls); other values appear with an offset `dra`/`ddec`, an odd harmonic, a
height, or in a [`System`][virgil.models.System] with other components. Light
on scales much larger than the shortest baseline's resolution appears only as a
loss of visibility, which [`Resolved`][virgil.models.Resolved] also models: the
data constrain their sum.

**Choosing `n_r`.** The transform needs enough nodes to sample the Bessel
oscillations across the profile's range: about
$\pi q_{\max}(r_{\rm out} - r_{\rm in})$, plus a dozen. A height needs `n_r` to
resolve the extra phase $2\pi q h\sin(\mathrm{inc})$ as well. Rendering a disk
with a `height` splats rings on a fine radial grid and needs concrete
parameters.

**Precision.** A profile's reference nodes are built in the default precision
when the disk is made, so to evaluate in float64 construct the disk inside
`jax.enable_x64(True)`.

**Where the options live.** Profiles and
[`RadialGP`][virgil.models.RadialGP] (a non-parametric profile, see its notes
on what `ell` means); azimuthal structure in
[`Harmonics`][virgil.models.Harmonics], [`VonMises`][virgil.models.VonMises] and
the priors of [`MultipoleKernel`][virgil.models.MultipoleKernel], converted by
[`virgil.harmonics`][virgil.harmonics]; height in
[`Cone`][virgil.models.Cone] and [`PowerLawHeight`][virgil.models.PowerLawHeight];
priors in [`gp_priors`][virgil.models.gp_priors]; and a residual aligned with
the disk in [`MultipoleField`][virgil.fields.MultipoleField].

**The quadrature visibility.** [`multipole_disk_cvis`][virgil.models.multipole_disk_cvis]
sums thin rings at nodes $r_k$ with weights $w_k$ after undoing `inc` and `pa`.
Ring $k$ has brightness $1 + \sum_m c_{mk}\cos m\theta + s_{mk}\sin m\theta$ in
the disk plane, with $\theta$ measured from the projected major axis towards East,
so

$$
V = 2\pi \sum_{|m| \le M} (-i)^m e^{im\phi} \sum_k w_k I_m(r_k)
    J_m(2\pi q r_k),
$$

with $I_{\pm m} = (c_m \mp i s_m)/2$, $I_0 = 1$, and $q$, $\phi$ the length and
direction of the spatial frequency in the disk plane. The sum is normalized
($V(0) = 1$) and `blur_fwhm` multiplies it as a Gaussian envelope. A `height`
lifts ring $k$ along the disk's normal, shifting it on the sky by
$h_k\sin(\mathrm{inc})$ towards `pa + 90` as a phase factor; the surface is thin
and nothing is occulted. Large problems are evaluated in chunks over the samples.

**References.** The Hankel-transform representation of a radial profile on a
fixed set of nodes follows frank (Jennings et al. 2020, MNRAS 495, 3209), and
the parametric profiles, uv deprojection and Gaussian envelope follow vis-r
(Kennedy 2025, RASTI 4, rzaf004). The code is a new implementation; none was
copied.

## ExpTail

A Gauss-Laguerre rule covers the oscillating tail less efficiently than a
Gauss-Legendre rule covers a finite range, and its accuracy depends on
$2\pi q L$ at the longest baseline $q$. The rule of thumb $16 \cdot 2\pi q L$
nodes (24 nodes at $2\pi q L = 1$, 100 at 5) is optimistic when the tail is
well resolved: in a benchmark of a widely resolved tail the largest error in
$V$ was about $2 \times 10^{-4}$ at 128 nodes and about $4 \times 10^{-5}$ at
the 150-node cap, not $10^{-5}$. For a tail resolved much more finely, prefer a
[`RadialPowerLaw`][virgil.models.RadialPowerLaw] with a soft edge, or give a
`cutoff`: the tail is cut at $r_{\rm in} + \mathrm{cutoff}\, L$ (the light beyond
is dropped; a cutoff of 5 drops 0.7 % of the flux of a thin disk) and
integrated by a Gauss-Legendre rule, which needs about
$\pi q\, \mathrm{cutoff}\, L + 12$ nodes. See
[`MultipoleDisk.suggest_n_r`][virgil.models.MultipoleDisk.suggest_n_r].

## Bandwidth smearing

[`smeared_model`][virgil.models.smeared_model] averages over a boxcar channel of
full width $\Delta\lambda$ the visibility of a source at offset $\mathbf{s}$. This
multiplies the component's offset phasor by the real envelope
$\mathrm{sinc}(\pi\,\Delta\lambda\,(u\,\mathrm{dra} + v\,\mathrm{ddec})/\lambda^2)$, with
$u$, $v$ in metres and the offsets in radians. It is applied to each component of a
[`System`][virgil.models.System] before the components are weighted and summed
(offsets add down nested systems), not to the final visibility, so the flux weights
and the zero-offset component are unchanged. The model is the first-order one: the
weights and the shape of each component are taken at the channel centre, and an
extended component is smeared as if it sat at its centre offset (its internal
structure is not smeared). A component at zero offset is exactly unaffected.
[`Resolved`][virgil.models.Resolved] pieces have no offset to smear.

## MultipoleField

The prior on the field $\eta$ is separable,

$$\mathrm{Cov}[\eta(r, \theta), \eta(r', \theta')] =
\sigma^2\, k_r(|r - r'|;\, \ell_r)\, k_\theta(\Delta\theta;\, \ell_\theta),$$

with a Matérn kernel in radius and a periodic Matérn kernel in azimuth given by
the spectral density that [`MultipoleKernel`][virgil.models.MultipoleKernel]
gives the harmonics of a [`MultipoleDisk`][virgil.models.MultipoleDisk],
$S(m) \propto [1 + (m\,\ell_\theta)^2 / 2\nu_\theta]^{-\nu_\theta - 1/2}$ at the
integer order $m$ ($\ell_\theta$ in radians, the angle `ell_theta / r` of a
`MultipoleKernel`).

The latents are whitened: $\eta = \sigma L_r Z F_\theta$ with $L_r$ the Cholesky
factor of the radial correlation and $F_\theta$ the (circulant) square root of
the azimuthal one, scaled so that $\sigma$ is the standard deviation of the
field on the grid. Beyond the polar grid (outside `r_max_mas`, or inside
`r_min_mas`) $\eta$ falls to `outside`, so those pixels are dark and the
Image's flux goes to the grid's ellipse. The azimuth is measured from the
projected major axis towards `pa + 90`, as for `MultipoleDisk`'s harmonics.
Giving $\sigma$ and the two lengths bounded log-uniform priors makes the
evidence proper.

**Regularity at the centre.** With `r_min_mas=0` the field must be
single-valued at $r=0$. A smooth 2D field has Fourier coefficients
$a_m(r)\sim r^{|m|}$ (the Zernike radial polynomials, Born & Wolf, *Principles
of Optics*, section 9.2), so order $m$ of $\eta$ is multiplied by
$\tanh(r/\ell_r)^{|m|}$, with $\ell_r$ the radial correlation length, the
scale on which the field varies (a physical length, so the prior does not
depend on the grid). The variance of order $m$ is $\tanh(r/\ell_r)^{2|m|}$ of
$\sigma^2$: it is within $4|m|e^{-10}$ of $\sigma^2$ beyond five correlation
lengths, but high orders recover slowly (for $n_\theta = 64$, order 32 keeps
about 10 per cent of its variance at $r = 2\ell_r$). The default
$\ell_r$ is five node spacings, so the region where the prior is not
stationary is about five spacings, not one, and a fitted $\ell_r$ also sets
how much $m \ge 1$ structure the inner disk can carry; the latents stay whitened and only $m=0$ is free at $r=0$. It is
applied only when `r_min_mas = 0`; an annulus keeps the plain prior.

**Geometry.** By default the geometry parameters belong to the field: they are
copies of a `MultipoleDisk`'s (see
[`from_disk`][virgil.fields.MultipoleField.from_disk]), and if both are fitted
the caller must sample the same values for both. With `geometry_from="disk"`
they are tied instead. The field then has no `inc`, `pa`, `dra` or `ddec`
parameters (they are `None`: nothing to fit, no path to set), and when the
enclosing `System` is evaluated (`model`, `render`, the likelihood, `fit`,
`log_evidence`) it takes the geometry of its sibling `disk`, in the Image's grid
frame: `inc` and `pa - rotation_deg` of the disk, and the disk's centre relative
to the Image's, along the Image's rotated axes. The gradient therefore flows
into the disk's parameters. The ties are resolved by
[`System`][virgil.models.System], so they hold for siblings of one System (not
for a disk in another nested System), and the Image is tied only inside a
System: evaluated alone (`Image.brightness`, or a regularizer given the bare
Image) the field uses the fallback geometry it was built with, which is also
what `from_disk` stores. `System.resolved` returns the System with the ties
baked in, for code that reads the pixels directly. The pixel scale comes from
the Image.

**References.** The whitened Gaussian-process latents of a radial profile follow
FRAP (Yoshida et al. 2026, arXiv:2607.12618), and the polar-grid representation
of a disk's brightness follows frank (Jennings et al. 2020, MNRAS 495, 3209).
The code is a new implementation; none was copied.
