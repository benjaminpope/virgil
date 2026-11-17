# Contributions

`virgil` brings together many projects from one research team and ideas we liked in other packages. We used Claude Code (Opus 5.5) to join existing ideas in one framework, not to write everything from scratch, and GitHub Copilot's coding agent wrote much of the 2026 code: tests, review fixes, the Laplace uncertainties for grid searches and some tutorials. So the commit history does not show who deserves credit for which ideas. The [Development carbon](dev_carbon.md) page estimates the energy and carbon cost of this AI-assisted work on virgil and virgil-vlti.

## History

- **drpangloss.** Dori Blakely began the project as a set of functions for the [PDS 70 paper](https://arxiv.org/abs/2404.13032) (Blakely et al. 2025, AJ 169, 137). Benjamin Pope and Louis Desdoigts then rewrote it by hand as `drpangloss`. Louis wrote the fast JAX grid searches (the basis of `grid_fit`) and the first kernel-phase support. Dori wrote the first notebook of Bayesian upper limits. The name was a joke on Voltaire's *Candide* and Antoine Mérand's [CANDID](https://github.com/amerand/CANDID): Doctor Pangloss believes we live in the best of all possible worlds, much as CANDID with better optimizers would.
- **virgil.** By 2026 the joke no longer fitted, and the team took up Benjamin Pope's suggestion of `virgil`, with Jonah Goldfine's acronym: the Versatile Interferometric Reconstruction and Gradient-based Inference Library. In Dante's *Divine Comedy*, Virgil guides the poet through the underworld; the README explains why that suits image reconstruction.

## People and their projects

- **Louis Desdoigts.**
    - Every model is built on his [zodiax](https://github.com/LouisDesdoigts/zodiax), whose dot-paths (`"comp.flux"`) are the public fitting interface.
    - `virgil.amigo` reads the mixed-DISCO format of his AMIGO pipeline ([Desdoigts et al. 2026](https://arxiv.org/abs/2510.09806), PASA 43, e075). `virgil.coverage` simulates data in that form, following AMIGO's latent visibility basis.
    - The matrix Fourier transform was checked against the one in his [dLux](https://github.com/LouisDesdoigts/dLux) ([Desdoigts et al. 2024](https://arxiv.org/abs/2406.08703)).
- **Shashank Dholakia.**
    - He made `drpangloss` handle spectrally dispersed data in 2024.
    - `GravityDarkenedStar` comes from his unpublished thesis work on rapidly rotating stars. Its equations are those of [Espinosa Lara & Rieutord (2011)](https://arxiv.org/abs/1109.3038); its implementation follows Aaron Dotter's [GDit](https://github.com/aarondotter/GDit).
    - virgil has an interface to his [harmonix](https://github.com/shashankdholakia/harmonix/) ([Dholakia & Pope 2025](https://arxiv.org/abs/2509.25433)).
    - The JAX translation of the [CEPHES](https://www.netlib.org/cephes/) Bessel functions, now the separate package [jaxbessel](https://github.com/benjaminpope/jaxbessel), is adapted from harmonix.
- **Max Charles.** The image reconstruction is largely a port and extension of the visibility-based part of his `dorito`, written for [his NIRISS AMI paper](https://arxiv.org/abs/2510.10924) (PASA 43, e048, 2026). Some of the inspiration for `dorito` came from Ian Czekala's [MPoL](https://mpol-dev.github.io/MPoL/).
- **Jonah Goldfine.** The parametric scenes (`virgil.scenes`) come from his [frito](https://github.com/JonahDG/frito), together with geometric primitives from his unpublished work on PDS 70.
- **Toon De Prins.**
    - He contributed the inclined, azimuthally modulated rim (`ModulatedGaussianRim`) and the elliptical transforms, from his work on post-AGB disks.
    - He reviewed and fixed the 0.4.0 release, including the pipeline layer.
- **Dori Blakely.** Besides starting the project, Dori contributed the flared-disk geometry from the PDS 70 paper.
- **Frantz Martinache.** The binary visibility function `cvis_binary` is adapted from his [xara](https://github.com/fmartinache/xara) (formerly pysco; [Martinache 2010](https://arxiv.org/abs/1009.3933)).

Claude Code made the harmonix, ELR, dorito, scenes and Gaussian-process inclusions.

## Methods

Docstrings cite these where they are used. The [References](references.md) page lists every source by module, and `references/` in the repository has the BibTeX.

**Data and observables**

- OIFITS: [Pauls et al. (2005)](https://doi.org/10.1086/444523) and OIFITS 2, [Duvert, Young & Hummel (2017)](https://arxiv.org/abs/1510.04556).
- Legacy OIFITS tools (`virgil.legacy`): from [ImPlaneIA](https://github.com/anand0xff/ImPlaneIA) by Anand Sivaramakrishnan and collaborators.
- NIRISS mask geometry: [Sivaramakrishnan et al. (2023)](https://arxiv.org/abs/2210.17434).
- Kernel phase: [Martinache (2010)](https://arxiv.org/abs/1009.3933).
- Correlated closure phases: [Kammerer et al. (2020)](https://arxiv.org/abs/2011.01209). Closure phases that share a baseline are correlated (±1/3 for equal noise), so virgil keeps only the independent combinations and uses their covariance in the likelihood.
- Telescope gains, with the covariance taken from the model: [Lachaume (2021)](https://arxiv.org/abs/2104.07082).
- Error floors: as in PMOIRED ([Mérand 2022](https://arxiv.org/abs/2207.11047)).
- Matrix Fourier transform: [Soummer et al. (2007)](https://arxiv.org/abs/0711.0368).

**Source models**

- Limb-darkened disks: Quirrenbach et al. (1996), eq. 3, with [Kipping (2013)](https://arxiv.org/abs/1308.0009) q₁, q₂ coefficients.
- Gravity darkening: [Espinosa Lara & Rieutord (2011)](https://arxiv.org/abs/1109.3038).
- Radial profiles of disks by quadrature (`MultipoleDisk`): the Fourier–Bessel (Hankel) representation of a radial brightness profile on a set of nodes, from [frank](https://github.com/discsim/frank) ([Jennings et al. 2020](https://arxiv.org/abs/2005.07709)); parametric profiles (Gaussian rings and gaps, power laws with soft edges), the deprojection of the uv plane and a Gaussian envelope, from [vis-r](https://github.com/drgmk/vis-r) ([Kennedy 2025](https://arxiv.org/abs/2502.17619)). virgil reimplements the ideas for optical and infrared observables and copies no code.
- Non-parametric radial profiles as Gaussian processes (`RadialGP`, `RadialGPFunction`, `MultipoleKernel`): whitened latents on radial nodes, $g = \mu + Lz$ with a dense kernel, fitted and sampled with automatic differentiation, and a separate radial process per physical quantity, from [FRAP](https://github.com/tomyoshida/frap) ([Yoshida et al. 2026](https://arxiv.org/abs/2607.12618)). virgil reimplements the ideas (Matérn kernels, a profile that is the exponential of the process, processes for the azimuthal harmonics and the spectral index) for optical and infrared observables and copies no code.
- Chromatic scenes of stars and environment, each with its own spectrum: SPARCO ([Kluska et al. 2014](https://arxiv.org/abs/1403.3343)); blackbody dust as in [Hillen et al. (2016)](https://doi.org/10.1051/0004-6361/201628125).

**Detection and limits**

- Detection limits: CANDID ([Gallenne et al. 2015](https://arxiv.org/abs/1505.02715)) and [Absil et al. (2011)](https://arxiv.org/abs/1110.1178).
- Bayesian upper limits: [Ruffio et al. (2018)](https://arxiv.org/abs/1809.08261).
- Likelihood-ratio tests: [Wilks' theorem](https://en.wikipedia.org/wiki/Wilks%27_theorem) and [Chernoff (1954)](https://doi.org/10.1214/aoms/1177728725) on parameter boundaries.
- Trials factor: the [look-elsewhere effect](https://en.wikipedia.org/wiki/Look-elsewhere_effect) ([Gross & Vitells 2010](https://arxiv.org/abs/1005.1891)).
- Detection statistics compared by injection-recovery [ROC curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) ([Fawcett 2006](https://doi.org/10.1016/j.patrec.2005.10.010)).

**Orbits**

- Linear marginalization of the Thiele–Innes constants: [Hartkopf, McAlister & Franz (1989)](https://doi.org/10.1086/115193), [Wright & Howard (2009)](https://arxiv.org/abs/0904.3725), [Lucy (2014)](https://arxiv.org/abs/1309.2868), [Luger, Foreman-Mackey & Hogg (2017)](https://arxiv.org/abs/1710.11136).
- Period resolution: from The Joker ([Price-Whelan et al. 2017](https://arxiv.org/abs/1610.07602)).
- Uniform angle priors and calibration terms: from Octofitter ([Thompson et al. 2023](https://arxiv.org/abs/2402.01971)).
- Compared with, but not taken from: orbitize! ([Blunt et al. 2020](https://arxiv.org/abs/1910.01756)) and orvara ([Brandt et al. 2021](https://arxiv.org/abs/2105.11671)).

**Imaging**

- Maximum entropy: the "historic" MaxEnt of Skilling & Bryan (1984), and the choice of its weight by "classic" MaxEnt (Gull 1989; [Skilling 1989](https://doi.org/10.1007/978-94-015-7860-8_3)).
- Evidence for regularization weights and re-estimated error bars: [MacKay (1992)](https://doi.org/10.1162/neco.1992.4.3.415).
- Choosing the weight from the L-curve corner: [Hansen & O'Leary (1993)](https://doi.org/10.1137/0914086).
- Total variation: [Rudin, Osher & Fatemi (1992)](https://doi.org/10.1016/0167-2789(92)90242-F); TSV and L1 as in [Honma et al. (2014)](https://arxiv.org/abs/1407.2422), [Akiyama et al. (2017)](https://arxiv.org/abs/1702.07361) and [Kuramochi et al. (2018)](https://arxiv.org/abs/1802.05783).
- Log-sum sparsity: [Candès, Wakin & Boyd (2008)](https://arxiv.org/abs/0711.1612), as in SQUEEZE ([Baron, Monnier & Kloppenburg 2010](https://doi.org/10.1117/12.857364)).
- Starlet sparsity: Starck, Murtagh & Fadili (2010).
- CLEAN: Högbom (1974), multiscale CLEAN ([Cornwell 2008](https://arxiv.org/abs/0806.2228)) and multi-frequency CLEAN ([Rau & Cornwell 2011](https://arxiv.org/abs/1106.2745)).
- Image-quality metrics: from the interferometric imaging beauty contests ([Malbet et al. 2010](https://arxiv.org/abs/1007.4473); [Baron et al. 2012](https://arxiv.org/abs/1207.7141)).
- Review of optical-interferometric imaging: [Thiébaut & Young (2017)](https://arxiv.org/abs/1708.08390).
- Gaussian-process image priors: from information field theory ([Enßlin, Frommert & Kitaura 2009](https://arxiv.org/abs/0806.3474)) and the Enßlin group's [NIFTy](https://ift.pages.mpcdf.de/nifty/user/index.html) ([Arras et al. 2021](https://arxiv.org/abs/2002.05218); [Edenhofer et al. 2024](https://arxiv.org/abs/2402.16683)).
- Imaging ensembles: [Drevon et al. (2026)](https://arxiv.org/abs/2609.15365).

**Inference**

- Sampling: the No-U-Turn Sampler ([Hoffman & Gelman 2014](https://arxiv.org/abs/1111.4246)) through numpyro.
- Variational inference:
    - ADVI ([Kucukelbir et al. 2017](https://arxiv.org/abs/1603.00788));
    - flow guides, as inverse autoregressive flows ([Kingma et al. 2016](https://arxiv.org/abs/1606.04934)) and block neural autoregressive flows ([De Cao, Titov & Aziz 2019](https://arxiv.org/abs/1904.04676));
    - PSIS diagnostics ([Vehtari et al. 2024](https://arxiv.org/abs/1507.02646)).
- Default priors: [Jeffreys (1946)](https://doi.org/10.1098/rspa.1946.0056).

## Software

`virgil` is built on [JAX](https://github.com/jax-ml/jax) and on Patrick Kidger's [equinox](https://github.com/patrick-kidger/equinox), [optimistix](https://github.com/patrick-kidger/optimistix) and [lineax](https://github.com/patrick-kidger/lineax), with [optax](https://github.com/google-deepmind/optax), [numpyro](https://github.com/pyro-ppl/numpyro), [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) and [jaxbessel](https://github.com/benjaminpope/jaxbessel). [AMICAL](https://github.com/SAIL-Labs/AMICAL) reduces the ground-based aperture-masking data used in the examples.
