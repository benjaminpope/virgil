# `virgil.imaging`

Regularizers, priors and helpers for image reconstruction with
[`Image`](models/classes/image.md). Ways to choose a regularization weight are the
L-curve's corner, the discrepancy principle, classic MaxEnt
(`LCurve.classic_maxent`), the degrees-of-freedom-corrected discrepancy
(`LCurve.dof_discrepancy`) and, for
Gaussian-field images, the Laplace evidence (`log_evidence`).

## Methods {#methods}

### CLEAN {#clean}

Högbom's CLEAN (Högbom 1974, A&AS 15, 417) finds the peak of the residual
dirty image and adds a fraction (the loop gain) of a point source there. For
data linear in the image, that residual is proportional to $-\partial\chi^2/\partial c_p$,
the gradient of χ² with respect to the flux of a point at each pixel. `clean`
uses that gradient directly, so it works for closure, kernel and DISCO phases
and V² as well. With $J e_p$ the change in the whitened residuals per unit
flux at pixel $p$, each iteration picks the pixel whose Gauss–Newton step
lowers χ² most, the largest $g_p^2/|J e_p|^2$ with $g_p<0$, and adds `gain`
times that step, $-g_p/(2|J e_p|^2)$. This is matching pursuit; for linear
data, where $|J e_p|$ is the same everywhere, it is exactly Högbom's CLEAN.
The norms $|J e_p|$ are computed once, at the start, with one
Jacobian–vector product per pixel. Normalizing by them matters next to an
analytic star, where flux in a pixel is nearly the same as the star's: the
gradient there is small, but so is $|J e_p|$, and an unnormalized search
would pile flux beside the star.

Components are added to a fixed `base` scene, with fluxes relative to its
(for a `System` base, to its total, as siblings of its components). They are
grey unless a `spectrum` is given, as the image is in SPARCO. Without a
base, the components alone make the image, starting from one at the centre
of the grid (closure phases do not fix the position, so this is also the
anchor).

Each iteration only adds flux, so an early step that overshoots cannot be
undone by later ones. Every `refit_every` iterations a **major cycle** (Clark
1980, A&A 89, 377; Cotton–Schwab, Schwab 1984, AJ 89, 1076) refits the fluxes
of all the components at once by non-negative least squares on the
linearized residuals: flux can move between components, and components whose
flux falls to zero are removed. A last major cycle runs when CLEAN stops at
the target or at `max_iterations`. The base is fixed unless `base_priors`
names some of its parameters; each major cycle then fits those together with
the fluxes with `fit`, since fitting them in turn would let the components
absorb an error in the base, or the base absorb flux it does not model. A
companion much closer to the star than λ/B has its flux and separation
nearly degenerate, whatever fits it.

**Multi-scale CLEAN** (Cornwell 2008, IEEE JSTSP 2, 793,
[arXiv:0806.2228](https://arxiv.org/abs/0806.2228); Rau & Cornwell 2011,
[arXiv:1106.2745](https://arxiv.org/abs/1106.2745)): with several
`scales_mas`, each component is a pixel convolved with a circular Gaussian of
FWHM $s$ ($s=0$ is a point), and the image is $\sum_s G_s * F_s$. The search
runs over every (scale, pixel) with the same score $g^2/|J e|^2$, and the
major cycles refit all scales together. Each shape is cut to the `support`
and renormalized there, so a component's flux is the flux it puts in the
image. Cornwell multiplies the peak residual at each scale by a bias
$1 - 0.6\,s/s_{\max}$ to favour small scales; the score here is the χ²
decrease of each component's step, which does not depend on its size, so no
bias is needed. `scale_bias` applies one, $1-b\,s/s_{\max}$, if the broad
components grab flux that belongs to points.

Iteration stops when χ² per data point reaches `target_chi2_red` (the
discrepancy principle; it relies on correct error bars), after
`max_iterations`, or when χ² has **stalled**: no pixel lowers it, or it has
fallen by less than a fraction `stall_tolerance` over the last `stall_window`
iterations, and a major cycle does not help. That happens when noise puts the
truth's own χ² per point above the target; `chi2_red` shows how close it
came.

### Evidence {#evidence}

With a Gaussian-field image, whitened latents $z\sim N(0,I)$ and Jacobian $J$
of the whitened residuals with respect to them, the Laplace evidence at the
MAP is

$$\log Z \approx -\tfrac12\chi^2 - L - \tfrac12|z|^2 - \tfrac12\log\det(I+J^\top J),$$

and the Laplace posterior of the latents is $N(z_0,(I+J^\top J)^{-1})$. The
log determinant is $\sum\log(1+s^2)$ over the singular values $s$ of $J$,
computed from the singular values of $J$ (QR-reduced when far from square):
the Gram matrix's rounding error can exceed the identity on high
signal-to-noise data and leave $I+J^\top J$ numerically indefinite. With the
thin SVD $J=USV^\top$ a posterior draw is
$z_0 + V\,\mathrm{diag}((1+s^2)^{-1/2})\,\varepsilon_1 + (I-VV^\top)\varepsilon_2$.

$-\tfrac12\chi^2 - L$ is
[`model_loglike`][virgil.likelihood.model_loglike] at the MAP less its
data-only normalization, $-\sum\log\sigma - \tfrac n2\log 2\pi$ (in its von
Mises form for unprojected phases, and with the correlated closure phases'
determinant), which is the same for every model and hyperparameter on a
given dataset and cancels in differences of $\log Z$. For plain data
$L=0$. Otherwise $L$ is the rest of the likelihood's normalizer: for data
with calibration gains or closure offsets (which the likelihood marginalizes)
it is $\tfrac12\log\det$ of their covariance factor, and for extra
observables the log ratio of effective to quoted errors. It is evaluated at
the MAP as the likelihood evaluates it, so that evidences for data with
different gain or offset widths compare; its curvature with respect to $z$
is left out of $J$, as Gauss–Newton leaves out the residuals' own second
derivatives.

### Error scale {#error-scale}

In MacKay's evidence framework (MacKay 1992, eqs. 4.9–4.10; Bishop 2006,
eqs. 3.91–3.95) the noise level is a hyperparameter. Write the noise
precision as $\beta=1/s^2$, so that the likelihood is $\exp(-\beta\chi^2/2)$
with χ² on the quoted errors. The Laplace evidence is maximized when

$$\frac1\beta = s^2 = \frac{\chi^2}{N-\gamma},
\qquad \gamma=\sum_i\frac{\beta\lambda_i}{1+\beta\lambda_i},$$

where $N$ is the number of independent data and the $\lambda_i$ are the
eigenvalues of the Gauss–Newton curvature $J^\top J$ in the field's whitened
latents, in which the prior's curvature is the identity. $\gamma$ is the
effective number of parameters the data measure: a direction with
$\beta\lambda\gg1$ is fixed by the data and counts as one, one with
$\beta\lambda\ll1$ is fixed by the prior and counts as none. It is solved for
$\beta$ by Newton's method, which converges monotonically from $\beta=0$
because $\gamma$ is concave in $\beta$. Each measured parameter uses up one
datum's worth of scatter, so χ² ≈ N − γ for honest error bars, and the
ordinary $s^2=\chi^2/N$ is biased low for the same reason as the 1/N estimate
of a sample variance. Only the field's latents are counted in $\gamma$; each
other measured parameter raises $s$ by about 1/(2N), negligible in SPARCO fits.

**One scale per observable.** A single $s$ assumes every error bar is off by
the same factor. The V² and closure phases of one instrument are calibrated
differently: in a MATISSE N-band contest file the V² gave χ² per point 0.005
and the closure phases 0.49. With `by_observable=True` each kind of observable
$b$ (`"vis"`, `"phi"` and each kind in `extras`, pooled over the datasets) gets
its own precision $\beta_b$ on its rows of the residual vector, and the
evidence is maximized where, for every block,

$$\frac1{\beta_b}=s_b^2=\frac{\chi_b^2}{N_b-\gamma_b},
\qquad \gamma_b=\beta_b\operatorname{tr}(A^{-1}G_b),
\qquad A=I+\sum_b\beta_bG_b,$$

with $G_b=J_b^\top J_b$ from block $b$'s rows. $\gamma_b$ is the sum over the
block's rows of the diagonal of the hat matrix $B^{1/2}JA^{-1}J^\top B^{1/2}$
($B$ the diagonal matrix of each row's $\beta_b$): the share of the measured
parameters that block pays for. The blocks are coupled, so the equations are
solved by iterating $\beta_b\leftarrow N_b/(\chi_b^2+\gamma_b/\beta_b)$ (the
same fixed point as MacKay's, but monotone, so it cannot oscillate) from
$\beta_b=N_b/\chi_b^2$ to a relative change of $10^{-10}$. With one block it
reproduces the single scale. $N_b$ counts independent data: closure phases
from four or more telescopes count their independent combinations, and their
periodic penalty residuals belong to the block's χ² and $\gamma_b$ but are not
counted as data. The blocks' covariances must be $s_b^2D$: calibration gains,
closure-phase offsets, marginalized flux scales or differential phases with a
finite `prior_width` add nuisance covariance that does not scale with the
quoted errors, and raise a `ValueError`. Prefer the single scale when a block
has few data: $s_b$ scatters by about $1/\sqrt{2(N_b-\gamma_b)}$.

References: D. J. C. MacKay (1992), "Bayesian interpolation", Neural
Computation 4, 415 ([doi](https://doi.org/10.1162/neco.1992.4.3.415)); C. M.
Bishop (2006), *Pattern Recognition and Machine Learning*, §3.5; S. F. Gull
(1989), "Developments in maximum entropy data analysis", in *Maximum Entropy
and Bayesian Methods*, Kluwer, 53–71, for the same $N-\gamma$ argument applied
to maximum entropy (`LCurve.classic_maxent`).

::: virgil.imaging
    options:
      members:
        - TSV
        - TV
        - MaxEntropy
        - Laplacian
        - StarletL1
        - LogSum
        - starlet
        - clean
        - CleanResult
        - Centroid
        - starting_image
        - dirty_image
        - image_priors
        - nyquist_pixel_scale
        - field_of_view
        - image_field
        - second_moment_fwhm
        - envelope_fwhm
        - beam
        - Beam
        - convolve_beam
        - l_curve
        - LCurve
        - DofDiscrepancy
        - log_evidence
        - laplace_samples
        - error_scale
        - chi2_by_observable
        - short_baseline_residual
        - residual_image
        - diagnose
        - Diagnosis

::: virgil._geometry
    options:
      members:
        - pixel_offsets
        - set_ft_defaults
