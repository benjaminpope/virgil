# Gaussian-process priors and information field theory

virgil's Gaussian-process (GP) image prior, [`GaussianField`][virgil.fields.GaussianField], takes its main ideas from information field theory (IFT). IFT was developed by Torsten Enßlin's group, who implement it in the package NIFTy. IFT formulates Bayesian inference on fields in the language of statistical field theory, with terms such as Hamiltonian, propagator, source and free energy. That framing connects inference to methods from physics, such as perturbation theory, and it extends naturally to non-Gaussian and nonlinear problems. For Gaussian priors, though, many of its objects are the same as those of Gaussian-process regression, which statistics and machine learning describe in different terms (e.g. Rasmussen & Williams 2006). A reader who knows one vocabulary can therefore find it difficult to recognise the same ideas in the other, and to see how virgil relates to both. This page translates between them.

This page is for two kinds of reader: those who know GPs and want to follow the IFT papers, and those who met IFT first and want to see what virgil does in GP terminology. It covers:

1. what virgil does, in GP terminology;
2. a dictionary from IFT terms to GP terminology, with the corresponding virgil code;
3. where virgil differs from NIFTy;
4. what to read next.

For a worked example, see [Imaging, part 3](imaging_gp.md).

**In brief.** virgil puts a Gaussian-process prior on the logarithm of the image's brightness. The prior's covariance is a Matérn-like kernel in the image, with an amplitude σ and a correlation length ℓ in milliarcseconds. To make the computation cheap, the covariance is applied in the image's cosine (DCT) basis, where it is diagonal. The fitted parameters are standard-normal latent variables that this covariance maps to the image. σ and ℓ are chosen by maximizing the Bayesian evidence. In IFT's vocabulary, these are a "signal field" with a "power spectrum", written in "standardized coordinates", with hyperparameters chosen from the "partition function".

## What virgil does

### The image

An [`Image`][virgil.models.Image] has a log-brightness $\eta_i$ in each pixel $i$. The pixel fluxes are

$$b_i = F \, \frac{e^{\eta_i}}{\sum_j e^{\eta_j}},$$

the softmax of η, with the sum taken over the pixels in the image's support. F is the image's total flux (the `flux` parameter). Two consequences follow:

- every pixel is positive, whatever the value of η;
- adding the same constant to every $\eta_i$ leaves the image unchanged.

### The prior

`GaussianField` makes η a Gaussian process about a template image μ:

$$\eta \sim \mathcal{N}\left(\log\left(\frac{\mu}{\max\mu} + \epsilon\right),\ \Sigma\right).$$

The template μ is any positive image you supply (`mean`), such as a Gaussian of roughly the right size. ε (`mean_floor`, default $10^{-3}$) keeps the logarithm finite where the template is zero. With no template, the prior mean is zero, which is a flat image.

The covariance Σ is **approximately stationary**: away from the image's edges, the covariance of the log-brightness in two pixels depends only on the separation between them, not on where they are. Near the edges it departs from this, because the edges are reflecting (see below). It has two hyperparameters, and both are defined in the image:

- **σ** (`sigma`) is the typical size of the departures of η from the template, in units of log-brightness. Precisely, σ² is the variance of η averaged over all pixels. A departure of 1 in η changes a pixel's brightness, relative to the rest of the image, by a factor of e ≈ 2.7.
- **ℓ** (`length_mas`) is the correlation length in milliarcseconds. Pixels much closer together than ℓ brighten and fade together. Pixels much farther apart than ℓ vary independently.

The kernel is close to a Matérn kernel; the next section explains the difference. ℓ is the Matérn length scale 1/κ, which is not the separation at which the correlation falls to 1/e. For the default `order=2`, the continuum kernel's correlation is about 0.6 at a separation of ℓ, about 0.3 at 2ℓ and about 0.1 at 3ℓ. A prior with ℓ smaller than the interferometer's resolution allows structure the data cannot constrain; the evidence (below) is what decides.

### Computing with it

An n × m image has nm pixels, so Σ is an nm × nm matrix. Storing it, let alone inverting it, is expensive. A stationary covariance avoids this, because it is diagonal in a Fourier basis, and virgil's nearly stationary one is exactly diagonal in a cosine basis. Its diagonal in that basis is the **power spectrum**, which is the Fourier transform of the kernel.

virgil uses the cosine transform (DCT-II) rather than the FFT. The FFT treats the image as periodic, so the left edge is adjacent to the right edge and strongly correlated with it. The DCT is the Fourier series of the image reflected at its edges, so the image does not wrap around: opposite edges are no longer adjacent, and are only as correlated as their distance apart allows. The price is that this is the covariance of a Laplacian with reflecting (Neumann) boundaries. It is translation-invariant only far from the edges: near an edge, a pixel is correlated with its own mirror image across it, so its variance and correlations differ slightly from those in the interior. Writing C for the orthonormal DCT matrix, the covariance is

$$\Sigma = C^\top \mathrm{diag}(S)\, C, \qquad S_{jk} \propto \left(\frac{1}{\ell^2} + \lambda_{jk}\right)^{-\mathrm{order}}.$$

Here (j, k) labels a cosine mode, and $\lambda_{jk}$ is the corresponding eigenvalue of the grid's Laplacian (the discrete second derivative). For an n × m grid of pixels h mas across,

$$\lambda_{jk} = \left(\frac{2}{h}\right)^2 \left[\sin^2\frac{\pi j}{2n} + \sin^2\frac{\pi k}{2m}\right].$$

For modes much coarser than a pixel, $\lambda_{jk} \approx q^2$, where q is the mode's angular wavenumber (in radians per mas). S is then the continuum Matérn spectrum, $(1/\ell^2 + q^2)^{-\mathrm{order}}$. At the finest scales the sines make S differ slightly from the continuum spectrum, which is why the kernel is "Matérn-like" rather than exactly Matérn.

Two adjustments complete S. Both are made in [`field_spectrum`][virgil.fields.field_spectrum]:

- **Normalization.** S is rescaled so that the variance of η, averaged over pixels, is σ². This uses the fact that the DCT is orthonormal, so that the sum of the per-pixel variances equals the sum of the $S_{jk}$.
- **The constant mode.** $S_{00}$, the variance of the mode that raises every pixel equally, is set to zero, because the softmax ignores it.

Read in this way, S is not a second prior that acts "in frequency space". It is the same image-space kernel, written in the basis where its covariance matrix is diagonal. σ sets the overall height of S. ℓ sets its knee: S is flat for wavenumbers q ≲ 1/ℓ and falls as $q^{-2\,\mathrm{order}}$ above it, so structure finer than about ℓ is suppressed.

This Fourier space belongs to the image. It is not the interferometer's (u, v) plane, and the prior does not depend on which baselines were observed.

### Whitening

The parameters that virgil fits are not the log-brightnesses η. They are **latent variables** z, one per cosine mode, each with an independent standard-normal prior. The field maps them to η by scaling each mode by its prior standard deviation $\sqrt{S_{jk}}$ and transforming back to pixels:

$$\eta = \log\left(\frac{\mu}{\max\mu} + \epsilon\right) + C^\top\left(\sqrt{S} \odot z\right),$$

where ⊙ is elementwise multiplication. If z ~ N(0, I), then η has exactly the prior above. Statisticians call this the **non-centred parameterization**; the machine-learning literature calls it **whitening**. It has three practical benefits:

- **The MAP fit is least squares.** The prior's negative log density is $\tfrac{1}{2}\lVert z\rVert^2$, a sum of squares, just like the data's $\tfrac{1}{2}\chi^2$. The whole objective is therefore a nonlinear least-squares problem, and [`fit`][virgil.fitting.fit] solves it with the Levenberg–Marquardt (LM) algorithm, which converges in a few dozen steps.
- **The hyperparameters do not change the prior on the fitted parameters.** σ and ℓ appear only in the map from z to η, not in the prior on z. This is what makes it practical to sample σ and ℓ together with z. In the alternative "centred" form, the prior on η itself depends on σ and ℓ, and that coupling creates the funnel-shaped posteriors that defeat samplers.
- **Directions the data do not constrain are already well scaled.** In those directions the posterior of z equals its prior, N(0, 1), so it has unit width in every such direction. This helps samplers such as NUTS (`likelihood.numpyro_model`), which work best when the posterior has a similar width in every direction.

### Choosing σ and ℓ

The **evidence**, or marginal likelihood, is the probability of the data given σ and ℓ, with the image integrated over:

$$Z(\sigma, \ell) = p(\mathrm{data} \mid \sigma, \ell) = \int p(\mathrm{data} \mid z)\, p(z)\, dz.$$

It rewards hyperparameters under which the observed data are probable. A prior too tight to fit the data scores badly, and so does a prior so loose that it spreads its probability over many images the data rule out. [`log_evidence`][virgil.imaging.log_evidence] evaluates it with the Laplace approximation, which approximates the posterior by a Gaussian at the MAP:

$$\log Z \approx -\tfrac{1}{2}\chi^2 - \tfrac{1}{2}\lVert z\rVert^2 - \tfrac{1}{2}\log\det\left(I + J^\top J\right),$$

up to a constant that is the same for every σ and ℓ. Each quantity is evaluated at the MAP:

- χ² is the sum of the squared whitened residuals, (model − data)/error;
- $\lVert z\rVert^2$ is the prior penalty;
- J is the Jacobian of the whitened residuals with respect to z, so $J^\top J$ measures how strongly the data constrain each direction of z. The log-determinant is the Occam factor, which penalizes a prior that leaves many directions for the data to fix.

The approximation is exact for a linear model with Gaussian noise. Choosing the σ and ℓ that maximize Z is called type-II maximum likelihood, empirical Bayes, or MacKay's evidence framework. [`error_scale`][virgil.imaging.error_scale] applies the same framework to the noise level, to check whether the error bars are too large or too small; with `by_observable=True` it gives one scale per kind of observable (V², closure phases, fluxes), for data whose blocks are mis-calibrated by different factors. Alternatively, σ and ℓ can be given priors and sampled together with z.

### Relation to TSV and Gaussian Markov random fields

Some imaging codes (e.g. Tiede et al. 2026, HIBI) use Gaussian Markov random field (GMRF) priors. These are usually described by "neighbouring pixels are correlated", which can make them look different from a kernel with a correlation length. They are not different. A GMRF is specified by its **precision matrix** Q = Σ⁻¹, the inverse of the covariance. Q is sparse: each pixel is coupled to its neighbours only. Its inverse Σ is dense: every pair of pixels is correlated, by an amount that decays with separation over a correlation length. "Only neighbours are coupled" describes Q, not Σ.

virgil's prior is a GMRF of this kind. Write κ = 1/ℓ, and let L be the grid's Laplacian matrix, defined by

$$\eta^\top L\, \eta = \frac{1}{h^2}\sum_{\text{neighbouring pairs } (i, j)} (\eta_i - \eta_j)^2,$$

where the sum runs over horizontally and vertically adjacent pixels inside the image. This sum is the **total squared variation** (TSV) of η. Then:

- **With `order=1`,** Q is proportional to κ²I + L. With the prior mean $\bar\eta = \log(\mu/\max\mu + \epsilon)$ from the template, the negative log prior, $\tfrac{1}{2}(\eta - \bar\eta)^\top Q\, (\eta - \bar\eta)$, is therefore a weighted sum of an L2 penalty, $\kappa^2 \sum_i (\eta_i - \bar\eta_i)^2$, and the TSV of the departure η − η̄. The penalty acts on departures from the template, not on η itself; with no template, η̄ = 0. The DCT diagonalizes L exactly, with eigenvalues $\lambda_{jk}$, because both treat the image edges as reflecting. The equivalence is therefore exact, apart from the removed constant mode and the overall σ normalization.
- **With `order=2`** (the default), Q is proportional to (κ²I + L)², which couples each pixel to its neighbours' neighbours. It is still sparse.

This is the link between Matérn kernels and GMRFs found by Lindgren, Rue & Lindström (2011). They showed that a Matérn field is the solution of a stochastic partial differential equation (SPDE),

$$(\kappa^2 - \nabla^2)^{\mathrm{order}/2}\, \eta = \text{white noise},$$

and that discretizing the SPDE on a grid gives a sparse GMRF. The kernel's smoothness parameter is ν = order − d/2, where d = 2 is the dimension of the image. The default `order=2` therefore gives ν = 1. `order=1` gives ν = 0, an edge case. In the continuum, a ν = 0 field in two dimensions has infinite variance at every point: the variance grows logarithmically as the pixels shrink. virgil's σ normalization hides this, but the price is that the prior's correlation at a fixed separation then depends on the pixel size as well as on ℓ.

## A dictionary of IFT terms

The IFT papers write s for the signal (the unknown field), d for the data, R for the response (the forward model) and N for the noise covariance. For a linear model they write d = Rs + n, with prior s ~ N(0, S) and noise n ~ N(0, N). Two symbols mean different things on this page:

- IFT's **S** is the prior covariance matrix, which this page calls **Σ**. This page's S is the diagonal of Σ in the DCT basis.
- IFT's **D** is the posterior covariance. This page uses C, not D, for the DCT.

IFT writes † for the adjoint, which for real matrices is the transpose ᵀ.

The "GP terminology" column follows Rasmussen & Williams (2006) where that book covers the concept, and the wider statistics and machine-learning literature otherwise (for example for variational inference). The "In virgil" column names the code that implements or corresponds to each object. In it, `env` stands for the name of an `Image` component in a `System`, and `field` for its `GaussianField`, so that `env.log_brightness` is `field`.

| IFT / NIFTy term | GP terminology | In virgil |
|---|---|---|
| signal field, s | the unknown function; the latent function of a GP | the log-brightness η: the array `env.eta`, which calls `field.evaluate(pixel_scale_mas)` |
| response, R | forward model, measurement operator | `OIData.model(scene)`, which computes the observables of a `SourceModel` |
| signal covariance, S | prior covariance matrix, kernel matrix | Σ. Never formed as a matrix; it is set by `field.sigma`, `field.length_mas` and `field.order` |
| power spectrum, $P_s(k)$ | spectral density of a stationary kernel | $S_{jk}$: the array returned by `field_spectrum(shape, pixel_scale_mas, sigma, length_mas, order)` |
| harmonic space; harmonic partner | Fourier space; Fourier basis | the DCT-II coefficients, the space in which `field.latent` lives (the image's own Fourier space, not the (u, v) plane) |
| amplitude operator, A, with S = AA† | a square root of the covariance (like a Cholesky factor) | $C^\top \mathrm{diag}(\sqrt{S})$: in `field.evaluate`, the latents are multiplied by the square root of `field_spectrum(...)` and transformed with `idctn(..., type=2, norm="ortho")` |
| standardized coordinates, excitations, ξ | whitened latents; non-centred parameterization | z: the array `field.latent`, fitted at the path `"env.log_brightness.latent"` with standard-normal priors from `image_priors(scene)` |
| zero mode; offset | the mean level, the k = 0 Fourier component | `field_spectrum` sets `spectrum[0, 0]` to zero; the image's overall level is set by `env.flux` instead |
| information Hamiltonian, H(d, s) = −ln p(d, s) | negative log joint density; the loss function | the loss that `fit` minimizes, reported as `FitResult.info["loss"]`: half the sum of squares of `whitened_residuals(scene, data)` and of `field.latent`, up to a constant |
| partition function, Z(d) | evidence, marginal likelihood p(d) | `log_evidence(scene, data)` returns its logarithm, in the Laplace approximation |
| classical solution; minimum of H | MAP (maximum a posteriori) estimate | `fit(scene, priors, data)`, which returns the MAP as `FitResult.model` and `FitResult.values` |
| information source, j = R†N⁻¹d | back-projected, noise-weighted data; for a linear response, a dirty image | not formed. The closest analogue is `dirty_image(data, npix, pixel_scale_mas)`, which weights the data uniformly rather than by their errors |
| information propagator, D = (S⁻¹ + R†N⁻¹R)⁻¹ | posterior covariance | not returned for the latents. Its Laplace approximation, $(I + J^\top J)^{-1}$, is what `log_evidence` and `error_scale` use, through the determinant and eigenvalues of $J^\top J$ |
| Wiener filter, m = Dj | GP posterior mean; kriging | `FitResult.model`, the MAP. The MAP equals the posterior mean only for a linear model with Gaussian noise, and virgil's model is nonlinear |
| (Gibbs) free energy | variational free energy, the negative evidence lower bound (ELBO) | not computed |
| Fisher metric, M = J†N⁻¹J + 1 | Gauss–Newton curvature of the loss: the data's part plus the prior's (the identity, in whitened coordinates) | $J^\top J + I$, where J is the Jacobian of `whitened_residuals` with respect to `field.latent`. It is the matrix in each step of `fit(..., method="lm")`; because `whitened_residuals` already divides by the errors, N⁻¹ does not appear |
| MGVI (metric Gaussian variational inference) | variational inference with a Gaussian approximation whose covariance is the inverse Fisher metric | not implemented. virgil uses `fit` with the Laplace approximation, or NUTS through `numpyro_model` |
| geoVI (geometric variational inference) | variational inference after a nonlinear change of coordinates that makes the posterior closer to Gaussian | not implemented |
| critical filter | empirical-Bayes estimate of the power spectrum | a grid of `fit` and `log_evidence` over `sigma` and `length_mas`, as in [Imaging, part 3](imaging_gp.md); the spectral shape is fixed |
| correlated field model | a GP whose power spectrum is itself unknown and inferred, under a hierarchical prior | not implemented; `field_spectrum` has a fixed Matérn-like shape |
| log-normal field, s = e^φ | the exponential of a GP, used for positive quantities | `env.brightness`, the softmax of `env.eta` (unit sum; multiplied by `env.flux` in the visibilities) |
| Feynman diagrams; interaction terms | a perturbative expansion of a non-Gaussian posterior about a Gaussian one | not used |

Two terms are easy to misread:

- **Field.** A field is a function on a continuous domain, such as the sky. In practice it is discretized on a grid. A Gaussian random field and a Gaussian process are the same mathematical object, a probability distribution over functions. "Random field" is the name used in spatial statistics and physics, "Gaussian process" the one used in machine learning.
- **Free energy.** In statistical mechanics the free energy is −ln Z, minus the log of the partition function. IFT borrows the term. Variational inference approximates the posterior p(s | d) by a simpler distribution q, by minimizing $\mathrm{KL}(q \,\Vert\, p(s \mid d)) - \ln Z$, which is the negative ELBO. That quantity is also called the variational free energy. Since the KL divergence is never negative, it is at least −ln Z, and equal to it when q is the exact posterior.

## Where virgil differs from NIFTy

virgil takes two ideas from IFT: whitened latents, and a stationary field computed in a Fourier basis. It does not use the NIFTy package. Compared with NIFTy's standard prior for images, the "correlated field model", it differs in four ways:

| | virgil | NIFTy's correlated field |
|---|---|---|
| Spectrum | a fixed Matérn-like shape with two hyperparameters, σ and ℓ (`field_spectrum`) | inferred from the data: a power law plus smooth departures from it, controlled by hyperparameters named `fluctuations`, `flexibility`, `asperity` and `loglogavgslope` |
| Basis and edges | DCT-II, with reflecting edges (`idctn(..., type=2, norm="ortho")` in `field.evaluate`) | FFT, which is periodic, usually with zero-padding to keep the edges apart |
| Hyperparameters | chosen by the evidence on a grid (`log_evidence`), or sampled with NUTS (`numpyro_model`) | inferred together with the field, usually by MGVI or geoVI |
| Posterior | the MAP (`fit`) with a Laplace approximation (a Gaussian centred on the MAP), or NUTS samples (`numpyro_model`) | samples from MGVI or geoVI |

A fixed spectral shape has fewer hyperparameters, so it is quicker to fit and easier to interpret. A learned spectrum can adapt to structure on several scales at once, such as a compact core within a diffuse halo.

The linear algebra is the same in both. In whitened coordinates the prior's curvature is the identity, so both codes build the matrix $J^\top J + I$. Each Levenberg–Marquardt step solves a system in this matrix to move towards the MAP, and `log_evidence` takes its log-determinant. MGVI uses its inverse as the covariance of a Gaussian approximation to the posterior, and draws samples from it.

## Further reading

The GP references come first because they use the terminology of the rest of this page. Reading them first makes it easier to map the IFT papers onto it.

**Gaussian processes**

- Rasmussen & Williams 2006, *Gaussian Processes for Machine Learning*, MIT Press, [free online](https://gaussianprocess.org/gpml/). Chapter 2 covers GP regression, which IFT calls the Wiener filter. §4.2 covers the Matérn kernels, and chapter 5 the marginal likelihood and the choice of hyperparameters.

**GMRFs and the SPDE link**

- Lindgren, Rue & Lindström 2011, JRSS B 73, 423. This paper shows that Matérn fields are solutions of an SPDE, and that discretizing the SPDE gives a sparse GMRF. It connects the "neighbouring pixels" and "power spectrum" descriptions of the same prior.
- Rue & Held 2005, *Gaussian Markov Random Fields: Theory and Applications*, Chapman & Hall/CRC. The standard textbook on GMRFs.

**Evidence and hyperparameters**

- MacKay 1992, Neural Computation 4, 415, "Bayesian interpolation". The evidence framework used by `log_evidence` and `error_scale`.
- Bishop 2006, *Pattern Recognition and Machine Learning*, §3.5. A textbook account of the same framework.

**Whitening and the non-centred parameterization**

- Papaspiliopoulos, Roberts & Sköld 2007, Statistical Science 22, 59. Centred and non-centred parameterizations of hierarchical models, and when each works better.
- Betancourt & Girolami, [arXiv:1312.0906](https://arxiv.org/abs/1312.0906), "Hamiltonian Monte Carlo for hierarchical models". Why the centred form produces funnels that defeat samplers, and how the non-centred form avoids them.

**Information field theory**

Read these with the dictionary above to hand.

- Enßlin, Frommert & Kitaura 2009, PRD 80, 105005, [arXiv:0806.3474](https://arxiv.org/abs/0806.3474). The founding paper. It introduces the Hamiltonian, information source, propagator and Wiener filter.
- Enßlin 2019, Annalen der Physik 531, 1800127, [arXiv:1804.03350](https://arxiv.org/abs/1804.03350), "Information theory for fields". A review of IFT.
- Knollmüller & Enßlin 2019, [arXiv:1901.11033](https://arxiv.org/abs/1901.11033). MGVI.
- Frank, Leike & Enßlin 2021, Entropy 23, 853, [arXiv:2105.10470](https://arxiv.org/abs/2105.10470). geoVI.
- Edenhofer et al. 2024, JOSS 9, 6593, [arXiv:2402.16683](https://arxiv.org/abs/2402.16683). NIFTy.re, the JAX version of NIFTy, subtitled "a library for Gaussian processes and variational inference".
- The [NIFTy documentation](https://ift.pages.mpcdf.de/nifty/user/index.html), including its guide to the correlated field model.

**Imaging applications**

- Junklewitz et al. 2016, A&A 586, A76, [arXiv:1311.5282](https://arxiv.org/abs/1311.5282). RESOLVE, an IFT imaging algorithm for radio interferometry.
- Arras et al. 2022, Nature Astronomy 6, 259, [arXiv:2002.05218](https://arxiv.org/abs/2002.05218). Imaging M87* with resolve. The Methods section describes the correlated field model.
- Tiede et al. 2026, ApJ 997, 262, [arXiv:2511.17706](https://arxiv.org/abs/2511.17706). HIBI: GMRF priors for VLBI imaging, implemented in Comrade.jl.
- Thiébaut & Young 2017, JOSA A 34, 904, [arXiv:1708.08390](https://arxiv.org/abs/1708.08390). A tutorial on image reconstruction in optical interferometry, which treats quadratic regularizers as Gaussian priors.
