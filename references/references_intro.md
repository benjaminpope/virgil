# References and provenance

Where each idea or piece of code in virgil came from: a paper, a package or a
person. This page is generated from `references/provenance.toml`, the single
list that every docstring and docs page cites; BibTeX is in
`references/virgil.bib`. The [Contributions](contributors.md) page tells the
history of the project in prose.

"Ported" means code was adapted; "after" means the idea and its equations
were reimplemented from the paper or the package's documentation, with no
code copied. An entry is marked verified when its arXiv ID or DOI was checked
against the arXiv abstract page, Crossref or NASA ADS. Entries marked "Not yet verified" have not had that check: treat
their identifiers and attributions as unconfirmed until the mark is removed.

## Origin of the package

- Functions for the PDS 70 JWST/AMI analysis by Dori Blakely
  ([Blakely et al. 2024](#blakely2024)), rewritten by Benjamin Pope and Louis
  Desdoigts as `drpangloss`. Louis Desdoigts wrote the JAX grid-search syntax
  and the first kernel-phase support; everything is built on his
  [zodiax](#zodiax) (the dot-path fitting interface). The name is from
  Voltaire's *Candide*, after CANDID ([Gallenne et al. 2015](#gallenne2015)).
- Jonah Goldfine: the acronym, and the parametric scenes in `virgil.scenes`
  (from his [frito](#frito)).
- Toon De Prins: `ModulatedGaussianRim` (an azimuthally modulated thin ring
  times a Gaussian envelope) and the elliptical (u, v) transform helpers,
  from his post-AGB disk work. The literature for such rim models is
  [Lazareff et al. (2017)](#lazareff2017) and [Hillen et al. (2016)](#hillen2016).
- Grid searches (`virgil.grid_fit`): Dori Blakely and Louis Desdoigts, in the
  `drpangloss` era. Only `linear_flux_grid` notes its equivalence to
  fouriever's `lincmap`, as a comparison.
- Shashank Dholakia: the gravity-darkened star (equations from
  [Espinosa Lara & Rieutord 2011](#espinosalara2011); the implementation is
  Aaron Dotter's [GDit](#gdit), ported to JAX in his jax-interferometry), the
  spherical-harmonic star ([harmonix](#harmonix)), the Bessel functions
  (CEPHES, via harmonix, now [jaxbessel](#jaxbessel)) and
  spectrally dispersed data.
- Max Charles: the image reconstruction in `virgil.imaging` is a port and
  extension of the visibility part of [dorito](#dorito), with ideas from Ian
  Czekala's [MPoL](#mpol).
- Louis Desdoigts: `virgil.amigo` reads the mixed-DISCO format of
  [AMIGO](#amigo); `virgil.coverage` follows its latent visibility basis; the
  matrix Fourier transform was checked against [dLux](#dlux).
