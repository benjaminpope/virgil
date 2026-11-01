<!-- AUTO-GENERATED from references/provenance.toml and references/references_intro.md by scripts/sync_references_page.py. Edit those, not this file. -->

# References and provenance

Where each idea or piece of code in virgil came from: a paper, a package or a
person. This page is generated from `references/provenance.toml`, the single
list that every docstring and docs page cites; BibTeX is in
`references/virgil.bib`. The [Contributions](contributors.md) page tells the
history of the project in prose.

"Ported" means code was adapted; "after" means the idea and its equations
were reimplemented from the paper or the package's documentation, with no
code copied. An entry is marked verified when its arXiv ID or DOI was checked
against the arXiv abstract page, Crossref or NASA ADS.

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

## Sources by module

Each row is a place where code in virgil follows a paper or package. *ported*: code was adapted. *after*: the idea and equations were reimplemented from the paper or the package's documentation, with no code copied.

### `virgil._closure`

| Symbol | Source | Relation |
|---|---|---|
| `ClosureNoise` | [Kammerer et al. (2020)](#kammerer2020), [arXiv:2011.01209](https://arxiv.org/abs/2011.01209), [ADS](https://ui.adsabs.harvard.edu/abs/2020A&A...644A.110K/abstract) | after |

### `virgil._elr`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Espinosa Lara & Rieutord (2011)](#espinosalara2011), [arXiv:1109.3038](https://arxiv.org/abs/1109.3038), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...533A..43E/abstract), [doi:10.1051/0004-6361/201117252](https://doi.org/10.1051/0004-6361/201117252) | after |
| `(module)` | [Dotter](#gdit), [GitHub](https://github.com/aarondotter/GDit) | ported |
| `(module)` | [Dholakia](#jaxinterferometry), [GitHub](https://github.com/shashankdholakia/jax-interferometry) | ported |

### `virgil._flat`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Hogg, Myers & Bovy (2010)](#hogg2010ecc), [arXiv:1008.4146](https://arxiv.org/abs/1008.4146), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2166H/abstract) | after |

### `virgil._geometry`

| Symbol | Source | Relation |
|---|---|---|
| `grid_visibilities` | [Desdoigts et al. (2024)](#dlux), [arXiv:2406.08703](https://arxiv.org/abs/2406.08703), [ADS](https://ui.adsabs.harvard.edu/abs/2023JATIS...9b8007D/abstract), [GitHub](https://github.com/LouisDesdoigts/dLux) | after |
| `grid_visibilities` | [Soummer et al. (2007)](#soummer2007), [arXiv:0711.0368](https://arxiv.org/abs/0711.0368), [ADS](https://ui.adsabs.harvard.edu/abs/2007OExpr..1515935S/abstract) | after |

### `virgil._linear`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Luger, Foreman-Mackey & Hogg (2017)](#luger2017linear), [arXiv:1710.11136](https://arxiv.org/abs/1710.11136), [ADS](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract), [doi:10.3847/2515-5172/aa96b5](https://doi.org/10.3847/2515-5172/aa96b5) | after |

### `virgil._nufft_spread`

| Symbol | Source | Relation |
|---|---|---|
| `evaluate` | [Barnett et al. (2019)](#barnett2019), [arXiv:1808.06736](https://arxiv.org/abs/1808.06736), [ADS](https://ui.adsabs.harvard.edu/abs/2019SJSC...41C.479B/abstract), [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X) | after |
| `evaluate` | [Lee & Greengard (2005)](#leegreengard2005), [ADS](https://ui.adsabs.harvard.edu/abs/2005JCoPh.206....1L/abstract), [doi:10.1016/j.jcp.2004.12.004](https://doi.org/10.1016/j.jcp.2004.12.004) | after |
| `spread` | [Oudoumanessah & Iollo](#nufftax), [GitHub](https://github.com/GragasLab/nufftax) | after |

### `virgil._orbit_coordinates`

| Symbol | Source | Relation |
|---|---|---|
| `sample_z` | [Moran (1975)](#moran1975), [doi:10.1017/s0021900200047720](https://doi.org/10.1017/s0021900200047720) | after |
| `ti_to_z` | [Goldstein, Poole & Safko (2002)](#goldstein2002) | after |
| `z_log_prior` | [Miles (1965)](#miles1965), [doi:10.1093/biomet/52.3-4.636](https://doi.org/10.1093/biomet/52.3-4.636) | after |
| `zeta_to_eccentricity` | [Ford (2006)](#ford2006), [arXiv:astro-ph/0512634](https://arxiv.org/abs/astro-ph/0512634), [ADS](https://ui.adsabs.harvard.edu/abs/2006ApJ...642..505F/abstract), [doi:10.1086/500802](https://doi.org/10.1086/500802) | after |

### `virgil.aliases`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Jeffreys (1946)](#jeffreys1946), [ADS](https://ui.adsabs.harvard.edu/abs/1946RSPSA.186..453J/abstract), [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056) | after |
| `(module)` | [Lucy (2014)](#lucy2014), [arXiv:1309.2868](https://arxiv.org/abs/1309.2868), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...563A.126L/abstract) | after |
| `(module)` | [MacKay (1992)](#mackay1992), [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415) | after |
| `(module)` | [Price-Whelan et al. (2017)](#pricewhelan2017), [arXiv:1610.07602](https://arxiv.org/abs/1610.07602), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...20P/abstract) | after |
| `(module)` | [VanderPlas (2017)](#vanderplas2018), [arXiv:1703.09824](https://arxiv.org/abs/1703.09824), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract) | after |
| `fit_orbit_aliases` | [virgil (own design) (2026)](#virgil_alias_search) | after |

### `virgil.amigo`

| Symbol | Source | Relation |
|---|---|---|
| `load_oi_data` | [Desdoigts et al. (2025)](#amigo), [arXiv:2510.09806](https://arxiv.org/abs/2510.09806), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251009806D/abstract) | after |

### `virgil.angles`

| Symbol | Source | Relation |
|---|---|---|
| `AngleVector` | [Thompson et al. (2023)](#thompson2023), [arXiv:2402.01971](https://arxiv.org/abs/2402.01971), [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....166..164T/abstract), [doi:10.3847/1538-3881/acf5cc](https://doi.org/10.3847/1538-3881/acf5cc), [GitHub](https://github.com/sefffal/Octofitter.jl) | after |

### `virgil.arcs`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [virgil (own design) (2026)](#virgil_short_arcs) | after |

### `virgil.coverage`

| Symbol | Source | Relation |
|---|---|---|
| `ami_grid_record` | [Desdoigts et al. (2025)](#amigo), [arXiv:2510.09806](https://arxiv.org/abs/2510.09806), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251009806D/abstract) | after |
| `mask_transfer` | [Sivaramakrishnan et al. (2022)](#sivaramakrishnan2023), [arXiv:2210.17434](https://arxiv.org/abs/2210.17434), [ADS](https://ui.adsabs.harvard.edu/abs/2023PASP..135a5003S/abstract) | after |

### `virgil.detection`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Gross & Vitells (2010)](#gross2010), [arXiv:1005.1891](https://arxiv.org/abs/1005.1891), [ADS](https://ui.adsabs.harvard.edu/abs/2010EPJC...70..525G/abstract) | after |
| `DetectionMC` | [Fawcett (2006)](#fawcett2006), [ADS](https://ui.adsabs.harvard.edu/abs/2006PaReL..27..861F/abstract), [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010) | after |
| `bootstrap_null` | [Efron (1979)](#efron1979), [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) | after |
| `local_nsigma` | [Chernoff (1954)](#chernoff1954), [ADS](https://ui.adsabs.harvard.edu/abs/1954AnnMS..25..573C/abstract), [doi:10.1214/aoms/1177728725](https://doi.org/10.1214/aoms/1177728725) | after |
| `local_nsigma` | [Wilks (1938)](#wilks1938), [doi:10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360) | after |
| `rescale_errors` | [Kammerer et al. (2020)](#kammerer2020), [arXiv:2011.01209](https://arxiv.org/abs/2011.01209), [ADS](https://ui.adsabs.harvard.edu/abs/2020A&A...644A.110K/abstract) | after |

### `virgil.ensemble`

| Symbol | Source | Relation |
|---|---|---|
| `combine` | [Drevon et al. (2026)](#drevon2026), [arXiv:2609.15365](https://arxiv.org/abs/2609.15365), [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260915365D/abstract) | after |
| `ensemble` | [Drevon et al. (2026)](#drevon2026), [arXiv:2609.15365](https://arxiv.org/abs/2609.15365), [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260915365D/abstract) | after |

### `virgil.epochs`

| Symbol | Source | Relation |
|---|---|---|
| `marginal_loglike` | [MacKay (1992)](#mackay1992), [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415) | after |

### `virgil.fields`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Enßlin, Frommert & Kitaura (2008)](#ensslin2009), [arXiv:0806.3474](https://arxiv.org/abs/0806.3474), [ADS](https://ui.adsabs.harvard.edu/abs/2009PhRvD..80j5005E/abstract) | after |
| `GaussianField` | [Arras et al. (2020)](#arras2021), [arXiv:2008.11435](https://arxiv.org/abs/2008.11435), [ADS](https://ui.adsabs.harvard.edu/abs/2021A&A...646A..84A/abstract) | after |
| `GaussianField` | [Arras et al. (2020)](#arras2022), [arXiv:2002.05218](https://arxiv.org/abs/2002.05218), [ADS](https://ui.adsabs.harvard.edu/abs/2022NatAs...6..259A/abstract) | after |
| `GaussianField` | [Edenhofer et al. (2024)](#edenhofer2024), [arXiv:2402.16683](https://arxiv.org/abs/2402.16683), [ADS](https://ui.adsabs.harvard.edu/abs/2024JOSS....9.6593E/abstract) | after |
| `GaussianField` | [Enßlin (2018)](#ensslin2019), [arXiv:1804.03350](https://arxiv.org/abs/1804.03350), [ADS](https://ui.adsabs.harvard.edu/abs/2019AnP...53100127E/abstract) | after |
| `GaussianField` | [Enßlin group (Arras, Edenhofer et al.)](#nifty), [link](https://ift.pages.mpcdf.de/nifty/user/index.html) | after |
| `GaussianField` | [Selig et al. (2013)](#selig2013), [arXiv:1301.4499](https://arxiv.org/abs/1301.4499), [ADS](https://ui.adsabs.harvard.edu/abs/2013A&A...554A..26S/abstract) | after |

### `virgil.fitting`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Kingma & Ba (2014)](#kingma2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980), [ADS](https://ui.adsabs.harvard.edu/abs/2014arXiv1412.6980K/abstract) | after |
| `(module)` | [Liu & Nocedal (1989)](#liu1989), [ADS](https://ui.adsabs.harvard.edu/abs/1989MatPr..45..503L/abstract), [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116) | after |
| `fit` | [Charles et al. (2026)](#dorito), [arXiv:2510.10924](https://arxiv.org/abs/2510.10924), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASA...43...48C/abstract) | ported |
| `fit` | [Hutchinson (1989)](#hutchinson1989), [ADS](https://ui.adsabs.harvard.edu/abs/1989CoSSC..18.1059H/abstract), [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806) | after |
| `fit` | [Skilling & Bryan (1984)](#skillingbryan1984), [ADS](https://ui.adsabs.harvard.edu/abs/1984MNRAS.211..111S/abstract) | after |
| `gauss_newton_diag` | [Bekas, Kokiopoulou & Saad (2007)](#bekas2007), [ADS](https://ui.adsabs.harvard.edu/abs/2007ApNM...57.1214B/abstract), [doi:10.1016/j.apnum.2007.01.003](https://doi.org/10.1016/j.apnum.2007.01.003) | after |
| `gauss_newton_diag` | [Cornwell & Evans (1985)](#cornwellevans1985), [ADS](https://ui.adsabs.harvard.edu/abs/1985A&A...143...77C/abstract) | after |
| `gauss_newton_diag` | [Hutchinson (1989)](#hutchinson1989), [ADS](https://ui.adsabs.harvard.edu/abs/1989CoSSC..18.1059H/abstract), [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806) | after |
| `gauss_newton_diag` | [Martens (2010)](#martens2010), [link](https://icml.cc/Conferences/2010/papers/458.pdf) | after |
| `gauss_newton_diag` | [Schraudolph (2002)](#schraudolph2002), [doi:10.1162/08997660260028683](https://doi.org/10.1162/08997660260028683) | after |
| `gauss_newton_diag` | [Skilling & Bryan (1984)](#skillingbryan1984), [ADS](https://ui.adsabs.harvard.edu/abs/1984MNRAS.211..111S/abstract) | after |

### `virgil.gains`

| Symbol | Source | Relation |
|---|---|---|
| `GainModes` | [Lachaume (2021)](#lachaume2021), [arXiv:2104.07082](https://arxiv.org/abs/2104.07082), [ADS](https://ui.adsabs.harvard.edu/abs/2021PASA...38...29L/abstract), [doi:10.1017/pasa.2021.20](https://doi.org/10.1017/pasa.2021.20) | after |

### `virgil.grid_fit`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Jeffreys (1946)](#jeffreys1946), [ADS](https://ui.adsabs.harvard.edu/abs/1946RSPSA.186..453J/abstract), [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056) | after |
| `linear_flux_grid` | [Gallenne et al. (2015)](#gallenne2015), [arXiv:1505.02715](https://arxiv.org/abs/1505.02715), [ADS](https://ui.adsabs.harvard.edu/abs/2015A&A...579A..68G/abstract) | after |
| `linear_flux_grid` | [Luger, Foreman-Mackey & Hogg (2017)](#luger2017linear), [arXiv:1710.11136](https://arxiv.org/abs/1710.11136), [ADS](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract), [doi:10.3847/2515-5172/aa96b5](https://doi.org/10.3847/2515-5172/aa96b5) | after |

### `virgil.harmonics`

| Symbol | Source | Relation |
|---|---|---|
| `from_image_harmonics` | [Dumitrescu (2007)](#dumitrescu2007), [doi:10.1007/978-1-4020-5125-8](https://doi.org/10.1007/978-1-4020-5125-8) | after |
| `reflection_coefficients` | [Barndorff-Nielsen & Schou (1973)](#barndorff-nielsen1973), [doi:10.1016/0047-259X(73)90030-4](https://doi.org/10.1016/0047-259X(73)90030-4) | after |

### `virgil.imaging`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Tiede et al. (2025)](#tiede2025), [arXiv:2511.17706](https://arxiv.org/abs/2511.17706), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ...997..262T/abstract) | after |
| `Image` | [Charles et al. (2026)](#dorito), [arXiv:2510.10924](https://arxiv.org/abs/2510.10924), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASA...43...48C/abstract) | ported |
| `LCurve.classic_maxent` | [Gull (1989)](#gull1989), [doi:10.1007/978-94-015-7860-8_4](https://doi.org/10.1007/978-94-015-7860-8_4) | after |
| `LCurve.classic_maxent` | [Skilling (1989)](#skilling1989), [doi:10.1007/978-94-015-7860-8_3](https://doi.org/10.1007/978-94-015-7860-8_3) | after |
| `Laplacian` | [Thiébaut & Young (2017)](#thiebaut2017), [arXiv:1708.08390](https://arxiv.org/abs/1708.08390), [ADS](https://ui.adsabs.harvard.edu/abs/2017JOSAA..34..904T/abstract) | after |
| `LogSum` | [Baron, Monnier & Kloppenburg (2010)](#baron2010squeeze), [ADS](https://ui.adsabs.harvard.edu/abs/2010SPIE.7734E..2IB/abstract), [doi:10.1117/12.857364](https://doi.org/10.1117/12.857364) | after |
| `LogSum` | [Candès, Wakin & Boyd (2007)](#candes2008), [arXiv:0711.1612](https://arxiv.org/abs/0711.1612), [ADS](https://ui.adsabs.harvard.edu/abs/2008JFAA...14..877C/abstract) | after |
| `MaxEntropy` | [Baron & Young (2008)](#baron2008), [ADS](https://ui.adsabs.harvard.edu/abs/2008SPIE.7013E..3XB/abstract), [doi:10.1117/12.789115](https://doi.org/10.1117/12.789115) | after |
| `MaxEntropy` | [Skilling & Bryan (1984)](#skillingbryan1984), [ADS](https://ui.adsabs.harvard.edu/abs/1984MNRAS.211..111S/abstract) | after |
| `StarletL1` | [Akiyama et al. (2017)](#akiyama2017), [arXiv:1702.07361](https://arxiv.org/abs/1702.07361), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...838....1A/abstract) | after |
| `StarletL1` | [Honma, Akiyama, Uemura & Ikeda (2014)](#honma2014), [arXiv:1407.2422](https://arxiv.org/abs/1407.2422), [ADS](https://ui.adsabs.harvard.edu/abs/2014PASJ...66...95H/abstract), [doi:10.1093/pasj/psu070](https://doi.org/10.1093/pasj/psu070) | after |
| `TSV` | [Akiyama et al. (2017)](#akiyama2017), [arXiv:1702.07361](https://arxiv.org/abs/1702.07361), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...838....1A/abstract) | after |
| `TSV` | [Kuramochi et al. (2018)](#kuramochi2018), [arXiv:1802.05783](https://arxiv.org/abs/1802.05783), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...56K/abstract), [doi:10.3847/1538-4357/aab6b5](https://doi.org/10.3847/1538-4357/aab6b5) | after |
| `TV` | [Chael et al. (2018)](#chael2018), [arXiv:1803.07088](https://arxiv.org/abs/1803.07088), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...857...23C/abstract), [doi:10.3847/1538-4357/aab6a8](https://doi.org/10.3847/1538-4357/aab6a8), [GitHub](https://github.com/achael/eht-imaging) | after |
| `TV` | [Rudin, Osher & Fatemi (1992)](#rudin1992), [ADS](https://ui.adsabs.harvard.edu/abs/1992PhyD...60..259R/abstract), [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F) | after |
| `clean` | [Cornwell (2008)](#cornwell2008), [arXiv:0806.2228](https://arxiv.org/abs/0806.2228), [ADS](https://ui.adsabs.harvard.edu/abs/2008ISTSP...2..793C/abstract), [doi:10.1109/jstsp.2008.2006388](https://doi.org/10.1109/jstsp.2008.2006388) | after |
| `clean` | [Högbom (1974)](#hogbom1974), [ADS](https://ui.adsabs.harvard.edu/abs/1974A&AS...15..417H/abstract) | after |
| `clean` | [Rau & Cornwell (2011)](#rau2011), [arXiv:1106.2745](https://arxiv.org/abs/1106.2745), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...532A..71R/abstract), [doi:10.1051/0004-6361/201117104](https://doi.org/10.1051/0004-6361/201117104) | after |
| `diagnose` | [Thiébaut & Young (2017)](#thiebaut2017), [arXiv:1708.08390](https://arxiv.org/abs/1708.08390), [ADS](https://ui.adsabs.harvard.edu/abs/2017JOSAA..34..904T/abstract) | after |
| `error_scale` | [MacKay (1992)](#mackay1992), [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415) | after |
| `l_curve` | [Hansen & O'Leary (1993)](#hansen1993), [ADS](https://ui.adsabs.harvard.edu/abs/1993SJSC...14.1487H/abstract), [doi:10.1137/0914086](https://doi.org/10.1137/0914086) | after |
| `laplace_samples` | [Knollmüller & Enßlin (2019)](#knollmuller2019), [arXiv:1901.11033](https://arxiv.org/abs/1901.11033), [ADS](https://ui.adsabs.harvard.edu/abs/2019arXiv190111033K/abstract) | after |
| `log_evidence` | [MacKay (1992)](#mackay1992), [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415) | after |
| `short_baseline_residual` | [Ireland, Monnier & Thureau (2006)](#ireland2006), [arXiv:2007.00716](https://arxiv.org/abs/2007.00716), [ADS](https://ui.adsabs.harvard.edu/abs/2006SPIE.6268E..1TI/abstract), [doi:10.1117/12.670940](https://doi.org/10.1117/12.670940) | after |
| `starlet` | [Starck, Murtagh & Fadili (2010)](#starck2010) | after |
| `starting_image` | [Charles et al. (2026)](#dorito), [arXiv:2510.10924](https://arxiv.org/abs/2510.10924), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASA...43...48C/abstract) | ported |

### `virgil.inference`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Betancourt (2017)](#betancourt2017), [arXiv:1701.02434](https://arxiv.org/abs/1701.02434), [ADS](https://ui.adsabs.harvard.edu/abs/2017arXiv170102434B/abstract) | after |
| `(module)` | [Hoffman & Gelman (2011)](#hoffman2014), [arXiv:1111.4246](https://arxiv.org/abs/1111.4246), [ADS](https://ui.adsabs.harvard.edu/abs/2011arXiv1111.4246H/abstract) | after |

### `virgil.legacy.oifits_implaneia`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Greenbaum et al. (2014)](#greenbaum2015), [arXiv:1411.3446](https://arxiv.org/abs/1411.3446), [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...798...68G/abstract) | after |
| `(module)` | [Greenbaum, Sivaramakrishnan et al.](#implaneia), [GitHub](https://github.com/anand0xff/ImPlaneIA) | ported |

### `virgil.likelihood`

| Symbol | Source | Relation |
|---|---|---|
| `group_outlier_mixture` | [Jeffreys (1946)](#jeffreys1946), [ADS](https://ui.adsabs.harvard.edu/abs/1946RSPSA.186..453J/abstract), [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056) | after |
| `inflated_errors` | [Hogg, Bovy & Lang (2010)](#hogg2010recipes), [arXiv:1008.4686](https://arxiv.org/abs/1008.4686), [ADS](https://ui.adsabs.harvard.edu/abs/2010arXiv1008.4686H/abstract) | after |

### `virgil.limits`

| Symbol | Source | Relation |
|---|---|---|
| `absil_limits` | [Absil et al. (2011)](#absil2011), [arXiv:1110.1178](https://arxiv.org/abs/1110.1178), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...535A..68A/abstract) | after |
| `injection_limits` | [Gallenne et al. (2015)](#gallenne2015), [arXiv:1505.02715](https://arxiv.org/abs/1505.02715), [ADS](https://ui.adsabs.harvard.edu/abs/2015A&A...579A..68G/abstract) | after |
| `nsigma` | [Absil et al. (2011)](#absil2011), [arXiv:1110.1178](https://arxiv.org/abs/1110.1178), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...535A..68A/abstract) | after |
| `ruffio_upperlimit` | [Ruffio et al. (2018)](#ruffio2018), [arXiv:1809.08261](https://arxiv.org/abs/1809.08261), [ADS](https://ui.adsabs.harvard.edu/abs/2018AJ....156..196R/abstract) | after |

### `virgil.metrics`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Baron et al. (2012)](#baron2012), [arXiv:1207.7141](https://arxiv.org/abs/1207.7141), [ADS](https://ui.adsabs.harvard.edu/abs/2012SPIE.8445E..1EB/abstract) | after |
| `(module)` | [Malbet et al. (2010)](#malbet2010), [arXiv:1007.4473](https://arxiv.org/abs/1007.4473), [ADS](https://ui.adsabs.harvard.edu/abs/2010SPIE.7734E..2NM/abstract) | after |
| `lawson_sigma_over_peak` | [Lawson, Cotton & Hummel (2004)](#lawson2004), [ADS](https://ui.adsabs.harvard.edu/abs/2004SPIE.5491..886L/abstract), [doi:10.1117/12.550710](https://doi.org/10.1117/12.550710) | after |
| `rms_convolved` | [Cotton, Monnier & Baron (2008)](#cotton2008), [ADS](https://ui.adsabs.harvard.edu/abs/2008SPIE.7013E..1NC/abstract), [doi:10.1117/12.788903](https://doi.org/10.1117/12.788903) | after |

### `virgil.models`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Pope et al.](#jaxbessel), [GitHub](https://github.com/benjaminpope/jaxbessel) | ported |
| `(module)` | [Desdoigts](#zodiax), [GitHub](https://github.com/LouisDesdoigts/zodiax) | after |
| `FlaredDisk` | [Blakely et al. (2024)](#blakely2024), [arXiv:2404.13032](https://arxiv.org/abs/2404.13032), [ADS](https://ui.adsabs.harvard.edu/abs/2025AJ....169..137B/abstract) | after |
| `GravityDarkenedStar` | [Espinosa Lara & Rieutord (2011)](#espinosalara2011), [arXiv:1109.3038](https://arxiv.org/abs/1109.3038), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...533A..43E/abstract), [doi:10.1051/0004-6361/201117252](https://doi.org/10.1051/0004-6361/201117252) | after |
| `Harmonics` | [Barndorff-Nielsen & Schou (1973)](#barndorff-nielsen1973), [doi:10.1016/0047-259X(73)90030-4](https://doi.org/10.1016/0047-259X(73)90030-4) | after |
| `Harmonics` | [Dumitrescu (2007)](#dumitrescu2007), [doi:10.1007/978-1-4020-5125-8](https://doi.org/10.1007/978-1-4020-5125-8) | after |
| `HarmonixModel` | [Dholakia & Pope (2025)](#harmonix), [arXiv:2509.25433](https://arxiv.org/abs/2509.25433), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASP..138e4504D/abstract), [GitHub](https://github.com/shashankdholakia/harmonix) | after |
| `HarmonixModel` | [Luger et al. (2018)](#starry), [arXiv:1810.06559](https://arxiv.org/abs/1810.06559), [ADS](https://ui.adsabs.harvard.edu/abs/2019AJ....157...64L/abstract) | after |
| `Image` | [Czekala et al. (2025)](#mpol), [arXiv:2502.00100](https://arxiv.org/abs/2502.00100), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250200100C/abstract), [GitHub](https://github.com/mpol-dev/MPoL), [link](https://mpol-dev.github.io/MPoL/) | after |
| `LimbDarkenedDisk` | [Dholakia & Pope (2025)](#harmonix), [arXiv:2509.25433](https://arxiv.org/abs/2509.25433), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASP..138e4504D/abstract), [GitHub](https://github.com/shashankdholakia/harmonix) | after |
| `ModulatedGaussianRim` | [Lazareff et al. (2016)](#lazareff2017), [arXiv:1611.08428](https://arxiv.org/abs/1611.08428), [ADS](https://ui.adsabs.harvard.edu/abs/2017A&A...599A..85L/abstract) | after |
| `MultipoleDisk` | [Jennings et al. (2020)](#frank), [arXiv:2005.07709](https://arxiv.org/abs/2005.07709), [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract), [GitHub](https://github.com/discsim/frank) | after |
| `MultipoleDisk` | [Kennedy (2025)](#visr), [arXiv:2502.17619](https://arxiv.org/abs/2502.17619), [ADS](https://ui.adsabs.harvard.edu/abs/2025RASTI...4....4D/abstract), [GitHub](https://github.com/drgmk/vis-r) | after |
| `MultipoleKernel` | [Yoshida et al. (2026)](#frap), [arXiv:2607.12618](https://arxiv.org/abs/2607.12618), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1007..198Y/abstract), [doi:10.3847/1538-4357/ae89ad](https://doi.org/10.3847/1538-4357/ae89ad), [GitHub](https://github.com/tomyoshida/frap) | after |
| `QuadraticLimbDarkenedDisk` | [Kipping (2013)](#kipping2013), [arXiv:1308.0009](https://arxiv.org/abs/1308.0009), [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract), [doi:10.1093/mnras/stt1435](https://doi.org/10.1093/mnras/stt1435) | after |
| `RadialGP` | [Yoshida et al. (2026)](#frap), [arXiv:2607.12618](https://arxiv.org/abs/2607.12618), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1007..198Y/abstract), [doi:10.3847/1538-4357/ae89ad](https://doi.org/10.3847/1538-4357/ae89ad), [GitHub](https://github.com/tomyoshida/frap) | after |
| `RadialGPFunction` | [Yoshida et al. (2026)](#frap), [arXiv:2607.12618](https://arxiv.org/abs/2607.12618), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1007..198Y/abstract), [doi:10.3847/1538-4357/ae89ad](https://doi.org/10.3847/1538-4357/ae89ad), [GitHub](https://github.com/tomyoshida/frap) | after |
| `SquareRootLimbDarkenedDisk` | [Díaz-Cordovés & Giménez (1992)](#diazcordoves1992), [ADS](https://ui.adsabs.harvard.edu/abs/1992A&A...259..227D/abstract) | after |
| `SquareRootLimbDarkenedDisk` | [Kipping (2013)](#kipping2013), [arXiv:1308.0009](https://arxiv.org/abs/1308.0009), [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract), [doi:10.1093/mnras/stt1435](https://doi.org/10.1093/mnras/stt1435) | after |
| `System` | [Kluska et al. (2014)](#kluska2014), [arXiv:1403.3343](https://arxiv.org/abs/1403.3343), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...564A..80K/abstract) | after |
| `cvis_binary` | [Ireland (2013)](#ireland2013), [arXiv:1301.6205](https://arxiv.org/abs/1301.6205), [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract) | after |
| `cvis_binary` | [Martinache (2010)](#xara), [arXiv:1009.3933](https://arxiv.org/abs/1009.3933), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...724..464M/abstract), [GitHub](https://github.com/fmartinache/xara) | ported |
| `cvis_limb_darkened_disk` | [Quirrenbach et al. (1996)](#quirrenbach1996), [ADS](https://ui.adsabs.harvard.edu/abs/1996A&A...312..160Q/abstract) | after |
| `multipole_disk_cvis` | [Jennings et al. (2020)](#frank), [arXiv:2005.07709](https://arxiv.org/abs/2005.07709), [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract), [GitHub](https://github.com/discsim/frank) | after |
| `multipole_disk_cvis` | [Kennedy (2025)](#visr), [arXiv:2502.17619](https://arxiv.org/abs/2502.17619), [ADS](https://ui.adsabs.harvard.edu/abs/2025RASTI...4....4D/abstract), [GitHub](https://github.com/drgmk/vis-r) | after |

### `virgil.oidata`

| Symbol | Source | Relation |
|---|---|---|
| `OIData.with_error_floor` | [Mérand (2022)](#merand2022), [arXiv:2207.11047](https://arxiv.org/abs/2207.11047), [ADS](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1NM/abstract), [GitHub](https://github.com/amerand/pmoired) | after |
| `OIData.with_north_angle` | [Thompson et al. (2023)](#thompson2023), [arXiv:2402.01971](https://arxiv.org/abs/2402.01971), [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....166..164T/abstract), [doi:10.3847/1538-3881/acf5cc](https://doi.org/10.3847/1538-3881/acf5cc), [GitHub](https://github.com/sefffal/Octofitter.jl) | after |

### `virgil.oifits`

| Symbol | Source | Relation |
|---|---|---|
| `read_oifits` | [Duvert, Young & Hummel (2015)](#duvert2017), [arXiv:1510.04556](https://arxiv.org/abs/1510.04556), [ADS](https://ui.adsabs.harvard.edu/abs/2017A&A...597A...8D/abstract), [doi:10.1051/0004-6361/201526405](https://doi.org/10.1051/0004-6361/201526405) | after |
| `read_oifits` | [Pauls et al. (2005)](#pauls2005), [arXiv:astro-ph/0508185](https://arxiv.org/abs/astro-ph/0508185), [ADS](https://ui.adsabs.harvard.edu/abs/2005PASP..117.1255P/abstract), [doi:10.1086/444523](https://doi.org/10.1086/444523) | after |
| `write_oifits` | [Duvert, Young & Hummel (2015)](#duvert2017), [arXiv:1510.04556](https://arxiv.org/abs/1510.04556), [ADS](https://ui.adsabs.harvard.edu/abs/2017A&A...597A...8D/abstract), [doi:10.1051/0004-6361/201526405](https://doi.org/10.1051/0004-6361/201526405) | after |

### `virgil.orbit_cells`

| Symbol | Source | Relation |
|---|---|---|
| `position_leg` | [Luger, Foreman-Mackey & Hogg (2017)](#luger2017linear), [arXiv:1710.11136](https://arxiv.org/abs/1710.11136), [ADS](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract), [doi:10.3847/2515-5172/aa96b5](https://doi.org/10.3847/2515-5172/aa96b5) | after |

### `virgil.orbits`

| Symbol | Source | Relation |
|---|---|---|
| `KeplerOrbit` | [Hartkopf, McAlister & Franz (1989)](#hartkopf1989), [ADS](https://ui.adsabs.harvard.edu/abs/1989AJ.....98.1014H/abstract), [doi:10.1086/115193](https://doi.org/10.1086/115193) | after |
| `KeplerOrbit` | [Wright & Howard (2009)](#wright2009), [arXiv:0904.3725](https://arxiv.org/abs/0904.3725), [ADS](https://ui.adsabs.harvard.edu/abs/2009ApJS..182..205W/abstract) | after |
| `KeplerOrbit.from_position_angle` | [Thompson et al. (2023)](#thompson2023), [arXiv:2402.01971](https://arxiv.org/abs/2402.01971), [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....166..164T/abstract), [doi:10.3847/1538-3881/acf5cc](https://doi.org/10.3847/1538-3881/acf5cc), [GitHub](https://github.com/sefffal/Octofitter.jl) | after |
| `PositionData.term` | [Thompson et al. (2023)](#thompson2023), [arXiv:2402.01971](https://arxiv.org/abs/2402.01971), [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....166..164T/abstract), [doi:10.3847/1538-3881/acf5cc](https://doi.org/10.3847/1538-3881/acf5cc), [GitHub](https://github.com/sefffal/Octofitter.jl) | after |
| `_thiele_innes_fit` | [Thiele (1883)](#thiele1883), [ADS](https://ui.adsabs.harvard.edu/abs/1883AN....104..245T/abstract) | after |
| `period_grid` | [Lucy (2014)](#lucy2014), [arXiv:1309.2868](https://arxiv.org/abs/1309.2868), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...563A.126L/abstract) | after |
| `starting_orbits` | [Lucy (2014)](#lucy2014), [arXiv:1309.2868](https://arxiv.org/abs/1309.2868), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...563A.126L/abstract) | after |
| `starting_orbits` | [Luger, Foreman-Mackey & Hogg (2017)](#luger2017linear), [arXiv:1710.11136](https://arxiv.org/abs/1710.11136), [ADS](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract), [doi:10.3847/2515-5172/aa96b5](https://doi.org/10.3847/2515-5172/aa96b5) | after |
| `starting_orbits` | [Price-Whelan et al. (2017)](#pricewhelan2017), [arXiv:1610.07602](https://arxiv.org/abs/1610.07602), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...20P/abstract) | after |

### `virgil.pipeline._checks`

| Symbol | Source | Relation |
|---|---|---|
| `r_hat` | [Vehtari et al. (2021)](#vehtari2021), [arXiv:1903.08008](https://arxiv.org/abs/1903.08008), [ADS](https://ui.adsabs.harvard.edu/abs/2021BayAn..16..667V/abstract) | after |

### `virgil.pipeline.star`

| Symbol | Source | Relation |
|---|---|---|
| `_compare` | [Chernoff (1954)](#chernoff1954), [ADS](https://ui.adsabs.harvard.edu/abs/1954AnnMS..25..573C/abstract), [doi:10.1214/aoms/1177728725](https://doi.org/10.1214/aoms/1177728725) | after |
| `_compare` | [Self & Liang (1987)](#selfliang1987), [ADS](https://ui.adsabs.harvard.edu/abs/1987JASA...82..605S/abstract), [doi:10.1080/01621459.1987.10478472](https://doi.org/10.1080/01621459.1987.10478472) | after |

### `virgil.priors`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Hogg, Myers & Bovy (2010)](#hogg2010ecc), [arXiv:1008.4146](https://arxiv.org/abs/1008.4146), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2166H/abstract) | after |
| `(module)` | [Jeffreys (1946)](#jeffreys1946), [ADS](https://ui.adsabs.harvard.edu/abs/1946RSPSA.186..453J/abstract), [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056) | after |
| `hierarchical_scales` | [Betancourt & Girolami (2013)](#betancourt2013), [arXiv:1312.0906](https://arxiv.org/abs/1312.0906), [ADS](https://ui.adsabs.harvard.edu/abs/2013arXiv1312.0906B/abstract) | after |

### `virgil.radio`

| Symbol | Source | Relation |
|---|---|---|
| `CalibrationNuisance` | [Arras et al. (2019)](#arras2019), [arXiv:1903.11169](https://arxiv.org/abs/1903.11169), [ADS](https://ui.adsabs.harvard.edu/abs/2019A&A...627A.134A/abstract) | after |
| `CalibrationNuisance` | [Blackburn et al. (2020)](#blackburn2020), [arXiv:1910.02062](https://arxiv.org/abs/1910.02062), [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...894...31B/abstract) | after |
| `CalibrationNuisance` | [Broderick et al. (2020)](#broderick2020), [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...897..139B/abstract), [doi:10.3847/1538-4357/ab91a4](https://doi.org/10.3847/1538-4357/ab91a4) | after |
| `CalibrationNuisance` | [Cornwell & Wilkinson (1981)](#cornwellwilkinson1981), [ADS](https://ui.adsabs.harvard.edu/abs/1981MNRAS.196.1067C/abstract), [doi:10.1093/mnras/196.4.1067](https://doi.org/10.1093/mnras/196.4.1067) | after |
| `CalibrationNuisance` | [Czekala et al. (2025)](#mpol), [arXiv:2502.00100](https://arxiv.org/abs/2502.00100), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250200100C/abstract), [GitHub](https://github.com/mpol-dev/MPoL), [link](https://mpol-dev.github.io/MPoL/) | after |
| `CalibrationNuisance` | [Zawadzki et al. (2023)](#zawadzki2023), [arXiv:2209.11813](https://arxiv.org/abs/2209.11813), [ADS](https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z/abstract), [doi:10.1088/1538-3873/acdf84](https://doi.org/10.1088/1538-3873/acdf84) | after |
| `ComplexVisData` | [Pawitan (2001)](#pawitan2001), [doi:10.1093/oso/9780198507659.001.0001](https://doi.org/10.1093/oso/9780198507659.001.0001) | after |
| `ComplexVisData` | [Zawadzki et al. (2023)](#zawadzki2023), [arXiv:2209.11813](https://arxiv.org/abs/2209.11813), [ADS](https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z/abstract), [doi:10.1088/1538-3873/acdf84](https://doi.org/10.1088/1538-3873/acdf84) | after |
| `ComplexVisData._imaging_weights` | [Briggs (1995)](#briggs1995), [ADS](https://ui.adsabs.harvard.edu/abs/1995PhDT.......238B/abstract) | after |

### `virgil.radio_clean`

| Symbol | Source | Relation |
|---|---|---|
| `_minor_cycle` | [Clark (1980)](#clark1980), [ADS](https://ui.adsabs.harvard.edu/abs/1980A&A....89..377C/abstract) | after |
| `_minor_cycle` | [Cornwell (2008)](#cornwell2008), [arXiv:0806.2228](https://arxiv.org/abs/0806.2228), [ADS](https://ui.adsabs.harvard.edu/abs/2008ISTSP...2..793C/abstract), [doi:10.1109/jstsp.2008.2006388](https://doi.org/10.1109/jstsp.2008.2006388) | after |
| `clean` | [Clark (1980)](#clark1980), [ADS](https://ui.adsabs.harvard.edu/abs/1980A&A....89..377C/abstract) | after |
| `clean` | [Högbom (1974)](#hogbom1974), [ADS](https://ui.adsabs.harvard.edu/abs/1974A&AS...15..417H/abstract) | after |
| `clean` | [Offringa et al. (2014)](#offringa2014), [arXiv:1407.1943](https://arxiv.org/abs/1407.1943), [ADS](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..606O/abstract) | after |
| `clean` | [Schwab (1984)](#schwab1984), [ADS](https://ui.adsabs.harvard.edu/abs/1984AJ.....89.1076S/abstract) | after |

### `virgil.radio_toeplitz`

| Symbol | Source | Relation |
|---|---|---|
| `ToeplitzVisData` | [Barnett et al. (2019)](#barnett2019), [arXiv:1808.06736](https://arxiv.org/abs/1808.06736), [ADS](https://ui.adsabs.harvard.edu/abs/2019SJSC...41C.479B/abstract), [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X) | after |
| `ToeplitzVisData` | [Fessler et al. (2005)](#fessler2005), [ADS](https://ui.adsabs.harvard.edu/abs/2005ITSP...53.3393F/abstract), [doi:10.1109/TSP.2005.853152](https://doi.org/10.1109/TSP.2005.853152) | after |
| `ToeplitzVisData` | [Lee & Greengard (2005)](#leegreengard2005), [ADS](https://ui.adsabs.harvard.edu/abs/2005JCoPh.206....1L/abstract), [doi:10.1016/j.jcp.2004.12.004](https://doi.org/10.1016/j.jcp.2004.12.004) | after |
| `ToeplitzVisData` | [Oudoumanessah & Iollo](#nufftax), [GitHub](https://github.com/GragasLab/nufftax) | after |

### `virgil.scenes`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Goldfine](#frito), [GitHub](https://github.com/JonahDG/frito) | ported |

### `virgil.smc`

| Symbol | Source | Relation |
|---|---|---|
| `bridge_smc` | [Cabezas et al. (2024)](#cabezas2024), [arXiv:2402.10797](https://arxiv.org/abs/2402.10797), [GitHub](https://github.com/blackjax-devs/blackjax) | after |
| `bridge_smc` | [Du & Guyader (2021)](#du2021), [arXiv:1909.13602](https://arxiv.org/abs/1909.13602), [doi:10.1214/20-AAP1611](https://doi.org/10.1214/20-AAP1611) | after |
| `bridge_smc` | [Lee & Whiteley (2018)](#lee2018), [arXiv:1509.00394](https://arxiv.org/abs/1509.00394), [doi:10.1093/biomet/asy028](https://doi.org/10.1093/biomet/asy028) | after |
| `tempered_smc` | [Cabezas et al. (2024)](#cabezas2024), [arXiv:2402.10797](https://arxiv.org/abs/2402.10797), [GitHub](https://github.com/blackjax-devs/blackjax) | after |
| `tempered_smc` | [Du & Guyader (2021)](#du2021), [arXiv:1909.13602](https://arxiv.org/abs/1909.13602), [doi:10.1214/20-AAP1611](https://doi.org/10.1214/20-AAP1611) | after |
| `tempered_smc` | [Lee & Whiteley (2018)](#lee2018), [arXiv:1509.00394](https://arxiv.org/abs/1509.00394), [doi:10.1093/biomet/asy028](https://doi.org/10.1093/biomet/asy028) | after |

### `virgil.spectra`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [Hillen et al. (2016)](#hillen2016), [arXiv:1603.03023](https://arxiv.org/abs/1603.03023), [ADS](https://ui.adsabs.harvard.edu/abs/2016A&A...588L...1H/abstract), [doi:10.1051/0004-6361/201628125](https://doi.org/10.1051/0004-6361/201628125) | after |
| `Nodes` | [Mérand (2022)](#merand2022), [arXiv:2207.11047](https://arxiv.org/abs/2207.11047), [ADS](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1NM/abstract), [GitHub](https://github.com/amerand/pmoired) | after |
| `PowerLaw` | [Kluska et al. (2014)](#kluska2014), [arXiv:1403.3343](https://arxiv.org/abs/1403.3343), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...564A..80K/abstract) | after |

### `virgil.svi`

| Symbol | Source | Relation |
|---|---|---|
| `variational` | [De Cao, Titov & Aziz (2019)](#decao2019), [arXiv:1904.04676](https://arxiv.org/abs/1904.04676), [ADS](https://ui.adsabs.harvard.edu/abs/2019arXiv190404676D/abstract) | after |
| `variational` | [Frank, Leike & Enßlin (2021)](#frank2021), [arXiv:2105.10470](https://arxiv.org/abs/2105.10470), [ADS](https://ui.adsabs.harvard.edu/abs/2021Entrp..23..853F/abstract) | after |
| `variational` | [Kingma et al. (2016)](#kingma2016), [arXiv:1606.04934](https://arxiv.org/abs/1606.04934), [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160604934K/abstract) | after |
| `variational` | [Kucukelbir et al. (2016)](#kucukelbir2017), [arXiv:1603.00788](https://arxiv.org/abs/1603.00788), [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160300788K/abstract) | after |
| `variational` | [Vehtari et al. (2015)](#vehtari2024), [arXiv:1507.02646](https://arxiv.org/abs/1507.02646), [ADS](https://ui.adsabs.harvard.edu/abs/2015arXiv150702646V/abstract) | after |

### `virgil.tempering`

| Symbol | Source | Relation |
|---|---|---|
| `(module)` | [virgil (own design) (2026)](#virgil_tempering) | after |

## Compared, not ported

Studied for comparison; nothing was taken from them.

| Source | Note |
|---|---|
| [Blunt et al. (2017)](#blunt2017), [arXiv:1703.10653](https://arxiv.org/abs/1703.10653), [ADS](https://ui.adsabs.harvard.edu/abs/2017AJ....153..229B/abstract) | OFTI |
| [Blunt et al. (2019)](#blunt2020), [arXiv:1910.01756](https://arxiv.org/abs/1910.01756), [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....159...89B/abstract), [GitHub](https://github.com/sblunt/orbitize) | orbitize!; AJ 159, 89 (2020) |
| [Blunt et al. (2024)](#blunt2024), [arXiv:2409.11573](https://arxiv.org/abs/2409.11573), [ADS](https://ui.adsabs.harvard.edu/abs/2024JOSS....9.6756B/abstract), [doi:10.21105/joss.06756](https://doi.org/10.21105/joss.06756) | orbitize!; JOSS 9, 6756 |
| [Brandt et al. (2021)](#brandt2021), [arXiv:2105.11671](https://arxiv.org/abs/2105.11671), [ADS](https://ui.adsabs.harvard.edu/abs/2021AJ....162..186B/abstract), [GitHub](https://github.com/t-brandt/orvara) | orvara; AJ 162, 186 |
| [CASA Team (2022)](#casa_tclean), [link](https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html) | Iteration Control, global stopping criteria: divergence 1 is (PeakRes - PrevPeakRes) / PrevPeakRes > 3.0, divergence 2 is (PeakRes - MinPeakRes) / MinPeakRes > 3.0, both over the whole image ignoring the mask (stop codes 5 and 6); minor cycle: (currentPeak - prevPeak) / prevPeak > 0.1, stop code 4 saying 'from minimum'. The page is inconsistent on the minor-cycle reference; virgil uses the running minimum per scale (virgil's choice). Read 2026-10-24 |
| [Kammerer](#fouriever), [GitHub](https://github.com/kammerje/fouriever) | comparison only: compare linear_flux_grid with its lincmap (closure/kernel phases only); no paper found |
| [Mawet et al. (2014)](#mawet2014), [arXiv:1407.2247](https://arxiv.org/abs/1407.2247), [ADS](https://ui.adsabs.harvard.edu/abs/2014ApJ...792...97M/abstract) | ApJ 792, 97; the small-sample false-alarm correction, not applied in injection_recovery |
| [Mérand et al.](#candid), [GitHub](https://github.com/amerand/CANDID) | algorithm paper: gallenne2015 |

## Background literature

Cited in the documentation as motivation or terminology; not a source of virgil's methods.

| Source | Note |
|---|---|
| [Abramowitz & Stegun (1964)](#abramowitz1964), [link](https://nvlpubs.nist.gov/nistpubs/Legacy/MP/nbsmiscellaneouspub346.pdf) | 25.4.38: Gauss-Chebyshev quadrature (first kind), exact for polynomials of degree 2n-1. |
| [Bishop (2006)](#bishop2006) | Springer textbook, section 3.5 (evidence approximation for regularization weights) |
| [Born & Wolf (1999)](#born1999), [doi:10.1017/CBO9781139644181](https://doi.org/10.1017/CBO9781139644181) | section 9.2: Zernike radial polynomials, R_n^m(r) ~ r^\|m\|, the regularity of a smooth field at the origin |
| [Currie et al. (2019)](#currie2019), [arXiv:1905.04322](https://arxiv.org/abs/1905.04322), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...877L...3C/abstract), [doi:10.3847/2041-8213/ab1b42](https://doi.org/10.3847/2041-8213/ab1b42) | ApJL 877, L3; motivates fitting a disk and a planet together |
| [Elsworth et al. (2025)](#elsworth2025), [arXiv:2508.15734](https://arxiv.org/abs/2508.15734), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250815734E/abstract) | Google's measured 0.24 Wh per median Gemini Apps text prompt; a cross-check for the Gemini figure in the development-carbon estimate (docs/dev_carbon.md) |
| [Junklewitz et al. (2016)](#junklewitz2016), [arXiv:1311.5282](https://arxiv.org/abs/1311.5282), [ADS](https://ui.adsabs.harvard.edu/abs/2016A&A...586A..76J/abstract) | A&A 586, A76; an IFT imaging algorithm for radio interferometry |
| [Kraus & Ireland (2012)](#kraus2012), [arXiv:1110.3808](https://arxiv.org/abs/1110.3808), [ADS](https://ui.adsabs.harvard.edu/abs/2012ApJ...745....5K/abstract), [doi:10.1088/0004-637X/745/1/5](https://doi.org/10.1088/0004-637X/745/1/5) | ApJ 745, 5; motivates fitting a disk and a planet together |
| [Lannelongue, Grealey & Inouye (2021)](#lannelongue2021), [ADS](https://ui.adsabs.harvard.edu/abs/2021AdvSc...800707L/abstract), [doi:10.1002/advs.202100707](https://doi.org/10.1002/advs.202100707) | Advanced Science 8, 2100707; the method behind the development-carbon estimate (docs/dev_carbon.md) |
| [Lindgren, Rue & Lindström (2011)](#lindgren2011), [doi:10.1111/j.1467-9868.2011.00777.x](https://doi.org/10.1111/j.1467-9868.2011.00777.x) | JRSS B 73, 423; Matérn fields as solutions of an SPDE |
| [Monnier et al. (2007)](#monnier2007), [arXiv:0706.0867](https://arxiv.org/abs/0706.0867), [ADS](https://ui.adsabs.harvard.edu/abs/2007Sci...317..342M/abstract), [doi:10.1126/science.1143205](https://doi.org/10.1126/science.1143205) | Science 317, 342; the Altair-like truth of notebooks/mwe/mwe_elr_chara.ipynb is inspired by it, not a reproduction |
| [Papaspiliopoulos, Roberts & Sköld (2007)](#papaspiliopoulos2007), [ADS](https://ui.adsabs.harvard.edu/abs/2007StaSc..2200014P/abstract), [doi:10.1214/088342307000000014](https://doi.org/10.1214/088342307000000014) | Statistical Science 22, 59; centred and non-centred parameterizations |
| [Rasmussen & Williams (2006)](#rasmussen2006), [doi:10.7551/mitpress/3206.001.0001](https://doi.org/10.7551/mitpress/3206.001.0001), [link](https://gaussianprocess.org/gpml/) | MIT Press (book); GP terminology, Matérn kernels, marginal likelihood; A.9 matrix inversion and determinant lemmas (arrow_at) |
| [Roettenbacher et al. (2016)](#roettenbacher2016), [arXiv:1709.10107](https://arxiv.org/abs/1709.10107), [ADS](https://ui.adsabs.harvard.edu/abs/2016Natur.533..217R/abstract), [doi:10.1038/nature17444](https://doi.org/10.1038/nature17444) | Nature 533, 217; CHARA/MIRC starspot imaging that motivates HarmonixModel |
| [Rue & Held (2005)](#rue2005), [doi:10.1201/9780203492024](https://doi.org/10.1201/9780203492024) | Chapman & Hall/CRC (book) |
| [Sallum et al. (2015)](#sallum2015), [arXiv:1511.07456](https://arxiv.org/abs/1511.07456), [ADS](https://ui.adsabs.harvard.edu/abs/2015Natur.527..342S/abstract), [doi:10.1038/nature15761](https://doi.org/10.1038/nature15761) | Nature 527, 342; motivates fitting a disk and a planet together |
| [van Hamme (1993)](#vanhamme1993), [ADS](https://ui.adsabs.harvard.edu/abs/1993AJ....106.2096V/abstract), [doi:10.1086/116788](https://doi.org/10.1086/116788) | AJ 106, 2096; the quadratic law the square-root law is compared with |

## Other background sources

Heritage and context without a code symbol of their own.

| Source | Note |
|---|---|
| [Martinache (2010)](#martinache2010), [arXiv:1009.3933](https://arxiv.org/abs/1009.3933), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...724..464M/abstract) | ApJ 724, 464 (2010) |

## In the bibliography only

Cited by the paper, not a source of virgil's methods.

| Source | Note |
|---|---|
| [Beskos, Pillai, Roberts, Sanz-Serna & Stuart (2013)](#beskos2013), [doi:10.3150/12-bej414](https://doi.org/10.3150/12-bej414) | Bernoulli 19, 1501; acceptance target 0.651 |
| [Buchholz, Chopin & Jacob (2021)](#buchholz2021), [arXiv:1808.07730](https://arxiv.org/abs/1808.07730), [ADS](https://ui.adsabs.harvard.edu/abs/2018arXiv180807730B/abstract), [doi:10.1214/20-ba1222](https://doi.org/10.1214/20-ba1222) | Bayesian Anal. 16, 745; fixed-L HMC kernels tuned per stage |
| [Chavez et al. (2026)](#chavez2026), [arXiv:2606.04094](https://arxiv.org/abs/2606.04094), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1005..132C/abstract) |  |
| [Del Moral, Doucet & Jasra (2006)](#delmoral2006), [doi:10.1111/j.1467-9868.2006.00553.x](https://doi.org/10.1111/j.1467-9868.2006.00553.x) | JRSSB 68, 411; SMC samplers and the normalizing-constant estimate, Z as the product of the incremental weights |
| [Foreman-Mackey et al. (2021)](#foreman-mackey2021), [arXiv:2105.01994](https://arxiv.org/abs/2105.01994), [ADS](https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3285F/abstract), [doi:10.21105/joss.03285](https://doi.org/10.21105/joss.03285) | JOSS 6, 3285 |
| [Genz & Malik (1980)](#genz1980), [doi:10.1016/0771-050X(80)90039-X](https://doi.org/10.1016/0771-050X(80)90039-X) | J. Comput. Appl. Math. 6, 295; the degree-7 rule with an embedded degree-5 rule and fourth-difference split axis, for the outlier floor b_k |
| [Hattori et al. (2024)](#jaxoplanet), [ADS](https://ui.adsabs.harvard.edu/abs/2024zndo..10736936H/abstract), [doi:10.5281/zenodo.10736936](https://doi.org/10.5281/zenodo.10736936), [GitHub](https://github.com/exoplanet-dev/jaxoplanet) | Zenodo; https://github.com/exoplanet-dev/jaxoplanet |
| [Jasra, Stephens, Doucet & Tsagaris (2011)](#jasra2011), [doi:10.1111/j.1467-9469.2010.00723.x](https://doi.org/10.1111/j.1467-9469.2010.00723.x) | Scand. J. Stat. 38, 1 (online December 2010); ESS-adaptive tempering schedules |
| [Kustaanheimo & Stiefel (1965)](#kustaanheimo1965), [doi:10.1515/crll.1965.218.204](https://doi.org/10.1515/crll.1965.218.204) | J. reine angew. Math. 218, 204; regularizing maps of Kepler motion |
| [Levi-Civita (1920)](#levicivita1920), [doi:10.1007/BF02404416](https://doi.org/10.1007/BF02404416) | Acta Math. 42, 99; the planar regularizing map |
| [Mengersen & Tweedie (1996)](#mengersen1996), [doi:10.1214/aos/1033066201](https://doi.org/10.1214/aos/1033066201) | Ann. Stat. 24, 101; independence Metropolis-Hastings is uniformly ergodic iff the weight is bounded |
| [Phipson & Smyth (2010)](#phipson2010), [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585) | Stat. Appl. Genet. Mol. Biol. 9, 39; (1 + #)/(1 + n) permutation p-value |
| [Price-Whelan et al. (2020)](#pricewhelan2020), [arXiv:2002.00014](https://arxiv.org/abs/2002.00014), [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...895....2P/abstract) | ApJ 895, 2; The Joker at scale |
| [Shoemake (1992)](#shoemake1992), [doi:10.1016/b978-0-08-050755-2.50036-1](https://doi.org/10.1016/b978-0-08-050755-2.50036-1) | Graphics Gems III, 124; unit quaternions and the Haar measure of SO(3) |
| [Soulain et al.](#amical), [GitHub](https://github.com/SAIL-Labs/AMICAL) | separate package, bibliography only; paper reference to add (SPIE 2020) |
| [Surjanovic, Biron-Lattes, Tiede et al. (2023)](#surjanovic2023), [arXiv:2308.09769](https://arxiv.org/abs/2308.09769), [ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv230809769S/abstract) | parallel tempering prior art next to Octofitter |
| [Zhou, Johansen & Aston (2016)](#zhou2016), [doi:10.1080/10618600.2015.1060885](https://doi.org/10.1080/10618600.2015.1060885) | JCGS 25, 701; conditional ESS for the next temperature |

## All sources

Alphabetical, one anchor per bibkey. Docstrings link here as `../references.md#<bibkey>`.

- <a id="abramowitz1964"></a>**Abramowitz & Stegun (1964)**. Handbook of Mathematical Functions. [link](https://nvlpubs.nist.gov/nistpubs/Legacy/MP/nbsmiscellaneouspub346.pdf). 25.4.38: Gauss-Chebyshev quadrature (first kind), exact for polynomials of degree 2n-1. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="absil2011"></a>**Absil et al. (2011)**. Searching for faint companions with VLTI/PIONIER. I. Method and first results. [arXiv:1110.1178](https://arxiv.org/abs/1110.1178), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...535A..68A/abstract).
- <a id="akiyama2017"></a>**Akiyama et al. (2017)**. Imaging the Schwarzschild-radius-scale Structure of M87 with the Event Horizon Telescope using Sparse Modeling. [arXiv:1702.07361](https://arxiv.org/abs/1702.07361), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...838....1A/abstract). ApJ 838, 1.
- <a id="arras2019"></a>**Arras et al. (2019)**. Unified Radio Interferometric Calibration and Imaging with Joint Uncertainty Quantification. [arXiv:1903.11169](https://arxiv.org/abs/1903.11169), [ADS](https://ui.adsabs.harvard.edu/abs/2019A&A...627A.134A/abstract). A&A 627, A134 (2019); joint calibration and imaging (resolve), gain priors.
- <a id="arras2021"></a>**Arras et al. (2020)**. Comparison of classical and Bayesian imaging in radio interferometry: Cygnus A with CLEAN and resolve. [arXiv:2008.11435](https://arxiv.org/abs/2008.11435), [ADS](https://ui.adsabs.harvard.edu/abs/2021A&A...646A..84A/abstract). A&A 646, A84 (2021).
- <a id="arras2022"></a>**Arras et al. (2020)**. Variable structures in M87* from space, time and frequency resolved interferometry. [arXiv:2002.05218](https://arxiv.org/abs/2002.05218), [ADS](https://ui.adsabs.harvard.edu/abs/2022NatAs...6..259A/abstract). Nature Astron. 6, 259 (2022).
- <a id="barndorff-nielsen1973"></a>**Barndorff-Nielsen & Schou (1973)**. On the parametrization of autoregressive models by partial autocorrelations. [doi:10.1016/0047-259X(73)90030-4](https://doi.org/10.1016/0047-259X(73)90030-4). J. Multivariate Anal. 3, 408-419; the unconstrained reflection-coefficient parametrization of a minimum-phase polynomial. Standard mathematics, credited here and in docstrings, not on the contributors page.
- <a id="barnett2019"></a>**Barnett et al. (2019)**. A parallel nonuniform fast Fourier transform library based on an "exponential of semicircle" kernel. [arXiv:1808.06736](https://arxiv.org/abs/1808.06736), [ADS](https://ui.adsabs.harvard.edu/abs/2019SJSC...41C.479B/abstract), [doi:10.1137/18M120885X](https://doi.org/10.1137/18M120885X). SIAM J. Sci. Comput. 41, C479 (FINUFFT); the ES kernel, and ψ̂ at non-integer frequencies by quadrature (Sect. 3.3).
- <a id="baron2008"></a>**Baron & Young (2008)**. Image reconstruction at Cambridge University. [ADS](https://ui.adsabs.harvard.edu/abs/2008SPIE.7013E..3XB/abstract), [doi:10.1117/12.789115](https://doi.org/10.1117/12.789115). BSMEM; SPIE 7013.
- <a id="baron2012"></a>**Baron et al. (2012)**. The 2012 Interferometric Imaging Beauty Contest. [arXiv:1207.7141](https://arxiv.org/abs/1207.7141), [ADS](https://ui.adsabs.harvard.edu/abs/2012SPIE.8445E..1EB/abstract). SPIE 8445.
- <a id="baron2010squeeze"></a>**Baron, Monnier & Kloppenburg (2010)**. A novel image reconstruction software for optical/infrared interferometry. [ADS](https://ui.adsabs.harvard.edu/abs/2010SPIE.7734E..2IB/abstract), [doi:10.1117/12.857364](https://doi.org/10.1117/12.857364). SQUEEZE; SPIE 7734.
- <a id="bekas2007"></a>**Bekas, Kokiopoulou & Saad (2007)**. An estimator for the diagonal of a matrix. [ADS](https://ui.adsabs.harvard.edu/abs/2007ApNM...57.1214B/abstract), [doi:10.1016/j.apnum.2007.01.003](https://doi.org/10.1016/j.apnum.2007.01.003). Appl. Numer. Math. 57, 1214; the estimator mean(v ⊙ Hv) over Rademacher v for the regularizer Hessian diagonal.
- <a id="beskos2013"></a>**Beskos, Pillai, Roberts, Sanz-Serna & Stuart (2013)**. Optimal tuning of the hybrid Monte Carlo algorithm. [doi:10.3150/12-bej414](https://doi.org/10.3150/12-bej414). Bernoulli 19, 1501; acceptance target 0.651.
- <a id="betancourt2017"></a>**Betancourt (2017)**. A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv:1701.02434](https://arxiv.org/abs/1701.02434), [ADS](https://ui.adsabs.harvard.edu/abs/2017arXiv170102434B/abstract).
- <a id="betancourt2013"></a>**Betancourt & Girolami (2013)**. Hamiltonian Monte Carlo for Hierarchical Models. [arXiv:1312.0906](https://arxiv.org/abs/1312.0906), [ADS](https://ui.adsabs.harvard.edu/abs/2013arXiv1312.0906B/abstract).
- <a id="bishop2006"></a>**Bishop (2006)**. Pattern Recognition and Machine Learning. Springer textbook, section 3.5 (evidence approximation for regularization weights). *Not yet verified against arXiv, Crossref or ADS.*
- <a id="blackburn2020"></a>**Blackburn et al. (2020)**. Closure statistics in interferometric data. [arXiv:1910.02062](https://arxiv.org/abs/1910.02062), [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...894...31B/abstract). ApJ 894, 31 (2020); closure equivalence with gains marginalized, finite gain priors.
- <a id="blakely2024"></a>**Blakely et al. (2024)**. The James Webb Interferometer: Space-based interferometric detections of PDS 70 b and c at 3.8 and 4.8 microns. [arXiv:2404.13032](https://arxiv.org/abs/2404.13032), [ADS](https://ui.adsabs.harvard.edu/abs/2025AJ....169..137B/abstract). AJ 169, 137 (2025).
- <a id="blunt2017"></a>**Blunt et al. (2017)**. Orbits for the Impatient: A Bayesian Rejection Sampling Method for Quickly Fitting the Orbits of Long-Period Exoplanets. [arXiv:1703.10653](https://arxiv.org/abs/1703.10653), [ADS](https://ui.adsabs.harvard.edu/abs/2017AJ....153..229B/abstract). OFTI.
- <a id="blunt2020"></a>**Blunt et al. (2019)**. orbitize!: A Comprehensive Orbit-fitting Software Package for the High-contrast Imaging Community. [arXiv:1910.01756](https://arxiv.org/abs/1910.01756), [ADS](https://ui.adsabs.harvard.edu/abs/2020AJ....159...89B/abstract), [GitHub](https://github.com/sblunt/orbitize). orbitize!; AJ 159, 89 (2020).
- <a id="blunt2024"></a>**Blunt et al. (2024)**. orbitize! v3: Orbit fitting for the High-contrast Imaging Community. [arXiv:2409.11573](https://arxiv.org/abs/2409.11573), [ADS](https://ui.adsabs.harvard.edu/abs/2024JOSS....9.6756B/abstract), [doi:10.21105/joss.06756](https://doi.org/10.21105/joss.06756). orbitize!; JOSS 9, 6756.
- <a id="born1999"></a>**Born & Wolf (1999)**. Principles of Optics, 7th edition. [doi:10.1017/CBO9781139644181](https://doi.org/10.1017/CBO9781139644181). section 9.2: Zernike radial polynomials, R_n^m(r) ~ r^|m|, the regularity of a smooth field at the origin.
- <a id="brandt2021"></a>**Brandt et al. (2021)**. orvara: An Efficient Code to Fit Orbits using Radial Velocity, Absolute, and/or Relative Astrometry. [arXiv:2105.11671](https://arxiv.org/abs/2105.11671), [ADS](https://ui.adsabs.harvard.edu/abs/2021AJ....162..186B/abstract), [GitHub](https://github.com/t-brandt/orvara). orvara; AJ 162, 186.
- <a id="briggs1995"></a>**Briggs (1995)**. High fidelity deconvolution of moderately resolved sources. [ADS](https://ui.adsabs.harvard.edu/abs/1995PhDT.......238B/abstract). PhD thesis, New Mexico Institute of Mining and Technology; the robust weighting; ADS 1995PhDT.......238B. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="broderick2020"></a>**Broderick et al. (2020)**. THEMIS: A Parameter Estimation Framework for the Event Horizon Telescope. [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...897..139B/abstract), [doi:10.3847/1538-4357/ab91a4](https://doi.org/10.3847/1538-4357/ab91a4). ApJ 897, 139 (2020); Laplace marginalization of station gains.
- <a id="buchholz2021"></a>**Buchholz, Chopin & Jacob (2021)**. Adaptive Tuning of Hamiltonian Monte Carlo Within Sequential Monte Carlo. [arXiv:1808.07730](https://arxiv.org/abs/1808.07730), [ADS](https://ui.adsabs.harvard.edu/abs/2018arXiv180807730B/abstract), [doi:10.1214/20-ba1222](https://doi.org/10.1214/20-ba1222). Bayesian Anal. 16, 745; fixed-L HMC kernels tuned per stage.
- <a id="cabezas2024"></a>**Cabezas et al. (2024)**. BlackJAX: Composable Bayesian inference in JAX. [arXiv:2402.10797](https://arxiv.org/abs/2402.10797), [GitHub](https://github.com/blackjax-devs/blackjax). the adaptive tempered SMC, HMC and pretuning kernels (blackjax >= 1.7.1) that virgil.smc wraps.
- <a id="candes2008"></a>**Candès, Wakin & Boyd (2007)**. Enhancing Sparsity by Reweighted L1 Minimization. [arXiv:0711.1612](https://arxiv.org/abs/0711.1612), [ADS](https://ui.adsabs.harvard.edu/abs/2008JFAA...14..877C/abstract). J. Fourier Anal. Appl. 14, 877 (2008).
- <a id="casa_tclean"></a>**CASA Team (2022)**. CASAdocs, Synthesis Imaging: Iteration Control (stopping criteria and stop codes). [link](https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html). Iteration Control, global stopping criteria: divergence 1 is (PeakRes - PrevPeakRes) / PrevPeakRes > 3.0, divergence 2 is (PeakRes - MinPeakRes) / MinPeakRes > 3.0, both over the whole image ignoring the mask (stop codes 5 and 6); minor cycle: (currentPeak - prevPeak) / prevPeak > 0.1, stop code 4 saying 'from minimum'. The page is inconsistent on the minor-cycle reference; virgil uses the running minimum per scale (virgil's choice). Read 2026-10-24. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="chael2018"></a>**Chael et al. (2018)**. Interferometric Imaging Directly with Closure Phases and Closure Amplitudes. [arXiv:1803.07088](https://arxiv.org/abs/1803.07088), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...857...23C/abstract), [doi:10.3847/1538-4357/aab6a8](https://doi.org/10.3847/1538-4357/aab6a8), [GitHub](https://github.com/achael/eht-imaging). ApJ 857, 23; eht-imaging.
- <a id="dorito"></a>**Charles et al. (2026)**. Image reconstruction with the JWST Interferometer. [arXiv:2510.10924](https://arxiv.org/abs/2510.10924), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASA...43...48C/abstract). PASA 43, e048 (2026); fit(post_step=): they convolve the log-parameter image with a Gaussian (sigma ~0.25 pix) and renormalise every fitting epoch, to stop flux collapsing into a few pixels.
- <a id="chavez2026"></a>**Chavez et al. (2026)**. Multifrequency Synthesis via CHIBI: Colorful Hierarchical Interferometric Bayesian Imaging. [arXiv:2606.04094](https://arxiv.org/abs/2606.04094), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1005..132C/abstract).
- <a id="chernoff1954"></a>**Chernoff (1954)**. On the Distribution of the Likelihood Ratio. [ADS](https://ui.adsabs.harvard.edu/abs/1954AnnMS..25..573C/abstract), [doi:10.1214/aoms/1177728725](https://doi.org/10.1214/aoms/1177728725).
- <a id="clark1980"></a>**Clark (1980)**. An efficient implementation of the algorithm 'CLEAN'. [ADS](https://ui.adsabs.harvard.edu/abs/1980A&A....89..377C/abstract). A&A 89, 377; bibcode 1980A&A....89..377C; PSF-window minor cycles and the cycle threshold. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="cornwell2008"></a>**Cornwell (2008)**. Multi-Scale CLEAN deconvolution of radio synthesis images. [arXiv:0806.2228](https://arxiv.org/abs/0806.2228), [ADS](https://ui.adsabs.harvard.edu/abs/2008ISTSP...2..793C/abstract), [doi:10.1109/jstsp.2008.2006388](https://doi.org/10.1109/jstsp.2008.2006388). IEEE JSTSP 2, 793.
- <a id="cornwellevans1985"></a>**Cornwell & Evans (1985)**. A simple maximum entropy deconvolution algorithm. [ADS](https://ui.adsabs.harvard.edu/abs/1985A&A...143...77C/abstract). A&A 143, 77; the entropy metric as the natural preconditioner in MEM. No DOI or arXiv ID: checked on ADS. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="cornwellwilkinson1981"></a>**Cornwell & Wilkinson (1981)**. A new method for making maps with unstable radio interferometers. [ADS](https://ui.adsabs.harvard.edu/abs/1981MNRAS.196.1067C/abstract), [doi:10.1093/mnras/196.4.1067](https://doi.org/10.1093/mnras/196.4.1067). MNRAS 196, 1067; self-calibration with a reference antenna fixing the gauge (the reference observation's gain 1).
- <a id="cotton2008"></a>**Cotton, Monnier & Baron (2008)**. 2008 imaging beauty contest. [ADS](https://ui.adsabs.harvard.edu/abs/2008SPIE.7013E..1NC/abstract), [doi:10.1117/12.788903](https://doi.org/10.1117/12.788903). Proc. SPIE 7013.
- <a id="currie2019"></a>**Currie et al. (2019)**. No Clear, Direct Evidence for Multiple Protoplanets Orbiting LkCa 15: LkCa 15 bcd are Likely Inner Disk Signals. [arXiv:1905.04322](https://arxiv.org/abs/1905.04322), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...877L...3C/abstract), [doi:10.3847/2041-8213/ab1b42](https://doi.org/10.3847/2041-8213/ab1b42). ApJL 877, L3; motivates fitting a disk and a planet together.
- <a id="mpol"></a>**Czekala et al. (2025)**. Million Points of Light (MPoL): a PyTorch library for radio interferometric imaging and inference. [arXiv:2502.00100](https://arxiv.org/abs/2502.00100), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250200100C/abstract), [GitHub](https://github.com/mpol-dev/MPoL), [link](https://mpol-dev.github.io/MPoL/). software paper; the Gaussian base layer (GaussConvFourier) behind blur_fwhm_mas; ADS bibcode checked.
- <a id="decao2019"></a>**De Cao, Titov & Aziz (2019)**. Block Neural Autoregressive Flow. [arXiv:1904.04676](https://arxiv.org/abs/1904.04676), [ADS](https://ui.adsabs.harvard.edu/abs/2019arXiv190404676D/abstract). BNAF.
- <a id="delmoral2006"></a>**Del Moral, Doucet & Jasra (2006)**. Sequential Monte Carlo Samplers. [doi:10.1111/j.1467-9868.2006.00553.x](https://doi.org/10.1111/j.1467-9868.2006.00553.x). JRSSB 68, 411; SMC samplers and the normalizing-constant estimate, Z as the product of the incremental weights.
- <a id="zodiax"></a>**Desdoigts**. zodiax. [GitHub](https://github.com/LouisDesdoigts/zodiax). no paper. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="dlux"></a>**Desdoigts et al. (2024)**. Differentiable Optics with dLux I: Deep calibration of Flat Field and Phase Retrieval. [arXiv:2406.08703](https://arxiv.org/abs/2406.08703), [ADS](https://ui.adsabs.harvard.edu/abs/2023JATIS...9b8007D/abstract), [GitHub](https://github.com/LouisDesdoigts/dLux). matrix Fourier transform cross-check; part II is 2406.08704.
- <a id="amigo"></a>**Desdoigts et al. (2025)**. AMIGO: a Data-Driven Calibration of the JWST Interferometer. [arXiv:2510.09806](https://arxiv.org/abs/2510.09806), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251009806D/abstract). PASA 43, e075 (2026).
- <a id="jaxinterferometry"></a>**Dholakia**. jax-interferometry. [GitHub](https://github.com/shashankdholakia/jax-interferometry). the original ELR code (core/ELR.py, core/utils.py), run unmodified at commit 70689ed3dba338d59c98d02e8126a07a3b4e86da by scripts/make_elr_golden.py to produce the golden fixture the port is checked against. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="harmonix"></a>**Dholakia & Pope (2025)**. Analytic Interferometry of Rotating Stellar Surfaces. [arXiv:2509.25433](https://arxiv.org/abs/2509.25433), [ADS](https://ui.adsabs.harvard.edu/abs/2026PASP..138e4504D/abstract), [GitHub](https://github.com/shashankdholakia/harmonix).
- <a id="gdit"></a>**Dotter**. GDit. [GitHub](https://github.com/aarondotter/GDit). ELR equations ported via Dholakia from GDit. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="drevon2026"></a>**Drevon et al. (2026)**. PYRA and MYTHRA: new statistical analysis for image reconstruction. [arXiv:2609.15365](https://arxiv.org/abs/2609.15365), [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260915365D/abstract). arXiv v1 is 2026.
- <a id="du2021"></a>**Du & Guyader (2021)**. Variance estimation in adaptive sequential Monte Carlo. [arXiv:1909.13602](https://arxiv.org/abs/1909.13602), [doi:10.1214/20-AAP1611](https://doi.org/10.1214/20-AAP1611). Ann. Appl. Probab. 31, 1021; extends the Eve-index variance estimator to SMC whose tempering schedule is chosen adaptively from the particles.
- <a id="dumitrescu2007"></a>**Dumitrescu (2007)**. Positive Trigonometric Polynomials and Signal Processing Applications. [doi:10.1007/978-1-4020-5125-8](https://doi.org/10.1007/978-1-4020-5125-8). Springer (book); the Fejér-Riesz theorem, spectral factorization and the Schur-Levinson recursion behind the positive form of Harmonics. Standard mathematics, credited here and in docstrings, not on the contributors page.
- <a id="duvert2017"></a>**Duvert, Young & Hummel (2015)**. OIFITS 2: the 2nd version of the Data Exchange Standard for Optical (Visible/IR) Interferometry. [arXiv:1510.04556](https://arxiv.org/abs/1510.04556), [ADS](https://ui.adsabs.harvard.edu/abs/2017A&A...597A...8D/abstract), [doi:10.1051/0004-6361/201526405](https://doi.org/10.1051/0004-6361/201526405). A&A 597, A8 (2017).
- <a id="diazcordoves1992"></a>**Díaz-Cordovés & Giménez (1992)**. A new nonlinear approximation to the limb-darkening of hot stars. [ADS](https://ui.adsabs.harvard.edu/abs/1992A&A...259..227D/abstract). A&A 259, 227; the square-root limb-darkening law. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="edenhofer2024"></a>**Edenhofer et al. (2024)**. Re-Envisioning Numerical Information Field Theory (NIFTy.re): A Library for Gaussian Processes and Variational Inference. [arXiv:2402.16683](https://arxiv.org/abs/2402.16683), [ADS](https://ui.adsabs.harvard.edu/abs/2024JOSS....9.6593E/abstract). JOSS 9, 6593.
- <a id="efron1979"></a>**Efron (1979)**. Bootstrap Methods: Another Look at the Jackknife. [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).
- <a id="elsworth2025"></a>**Elsworth et al. (2025)**. Measuring the environmental impact of delivering AI at Google Scale. [arXiv:2508.15734](https://arxiv.org/abs/2508.15734), [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250815734E/abstract). Google's measured 0.24 Wh per median Gemini Apps text prompt; a cross-check for the Gemini figure in the development-carbon estimate (docs/dev_carbon.md).
- <a id="ensslin2019"></a>**Enßlin (2018)**. Information theory for fields. [arXiv:1804.03350](https://arxiv.org/abs/1804.03350), [ADS](https://ui.adsabs.harvard.edu/abs/2019AnP...53100127E/abstract). Annalen der Physik 531, 1800127 (2019).
- <a id="nifty"></a>**Enßlin group (Arras, Edenhofer et al.)**. NIFTy: Numerical Information Field Theory. [link](https://ift.pages.mpcdf.de/nifty/user/index.html). papers: edenhofer2024, arras2021, ensslin2009. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="ensslin2009"></a>**Enßlin, Frommert & Kitaura (2008)**. Information field theory for cosmological perturbation reconstruction and non-linear signal analysis. [arXiv:0806.3474](https://arxiv.org/abs/0806.3474), [ADS](https://ui.adsabs.harvard.edu/abs/2009PhRvD..80j5005E/abstract). Phys. Rev. D 80, 105005 (2009).
- <a id="espinosalara2011"></a>**Espinosa Lara & Rieutord (2011)**. Gravity darkening in rotating stars. [arXiv:1109.3038](https://arxiv.org/abs/1109.3038), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...533A..43E/abstract), [doi:10.1051/0004-6361/201117252](https://doi.org/10.1051/0004-6361/201117252). A&A 533, A43.
- <a id="fawcett2006"></a>**Fawcett (2006)**. An introduction to ROC analysis. [ADS](https://ui.adsabs.harvard.edu/abs/2006PaReL..27..861F/abstract), [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010).
- <a id="fessler2005"></a>**Fessler et al. (2005)**. Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity. [ADS](https://ui.adsabs.harvard.edu/abs/2005ITSP...53.3393F/abstract), [doi:10.1109/TSP.2005.853152](https://doi.org/10.1109/TSP.2005.853152). IEEE Trans. Signal Process. 53, 3393; the Toeplitz embedding of the normal operator AᴴWA on a 2N grid.
- <a id="ford2006"></a>**Ford (2006)**. Improving the Efficiency of Markov Chain Monte Carlo for Analyzing the Orbits of Extrasolar Planets. [arXiv:astro-ph/0512634](https://arxiv.org/abs/astro-ph/0512634), [ADS](https://ui.adsabs.harvard.edu/abs/2006ApJ...642..505F/abstract), [doi:10.1086/500802](https://doi.org/10.1086/500802). ApJ 642, 505; the (sqrt(e) cos omega, sqrt(e) sin omega) parameterization, uniform in e; zeta is its half-angle variant.
- <a id="foreman-mackey2021"></a>**Foreman-Mackey et al. (2021)**. exoplanet: Gradient-based probabilistic inference for exoplanet data & other astronomical time series. [arXiv:2105.01994](https://arxiv.org/abs/2105.01994), [ADS](https://ui.adsabs.harvard.edu/abs/2021JOSS....6.3285F/abstract), [doi:10.21105/joss.03285](https://doi.org/10.21105/joss.03285). JOSS 6, 3285.
- <a id="frank2021"></a>**Frank, Leike & Enßlin (2021)**. Geometric variational inference. [arXiv:2105.10470](https://arxiv.org/abs/2105.10470), [ADS](https://ui.adsabs.harvard.edu/abs/2021Entrp..23..853F/abstract). geoVI.
- <a id="gallenne2015"></a>**Gallenne et al. (2015)**. Robust high-contrast companion detection from interferometric observations. The CANDID algorithm and an application to six binary stars. [arXiv:1505.02715](https://arxiv.org/abs/1505.02715), [ADS](https://ui.adsabs.harvard.edu/abs/2015A&A...579A..68G/abstract). CANDID.
- <a id="genz1980"></a>**Genz & Malik (1980)**. Remarks on algorithm 006: An adaptive algorithm for numerical integration over an N-dimensional rectangular region. [doi:10.1016/0771-050X(80)90039-X](https://doi.org/10.1016/0771-050X(80)90039-X). J. Comput. Appl. Math. 6, 295; the degree-7 rule with an embedded degree-5 rule and fourth-difference split axis, for the outlier floor b_k.
- <a id="frito"></a>**Goldfine**. frito. [GitHub](https://github.com/JonahDG/frito). *Not yet verified against arXiv, Crossref or ADS.*
- <a id="goldstein2002"></a>**Goldstein, Poole & Safko (2002)**. Classical Mechanics (3rd ed.). Addison-Wesley, section 4.5; the Cayley-Klein parameters of a rotation in Euler angles. A textbook with no DOI or ADS record: unverified. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="greenbaum2015"></a>**Greenbaum et al. (2014)**. An image-plane algorithm for JWST's non-redundant aperture mask data. [arXiv:1411.3446](https://arxiv.org/abs/1411.3446), [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...798...68G/abstract). ImPlaneIA; ApJ 798, 68 (2015).
- <a id="implaneia"></a>**Greenbaum, Sivaramakrishnan et al.**. ImPlaneIA. [GitHub](https://github.com/anand0xff/ImPlaneIA). paper: greenbaum2015. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="gross2010"></a>**Gross & Vitells (2010)**. Trial factors for the look elsewhere effect in high energy physics. [arXiv:1005.1891](https://arxiv.org/abs/1005.1891), [ADS](https://ui.adsabs.harvard.edu/abs/2010EPJC...70..525G/abstract).
- <a id="gull1989"></a>**Gull (1989)**. Developments in Maximum Entropy Data Analysis. [doi:10.1007/978-94-015-7860-8_4](https://doi.org/10.1007/978-94-015-7860-8_4). in Maximum Entropy and Bayesian Methods, Cambridge 1988; the DOI chapter title is the Gull one; Skilling 1989 shares the volume.
- <a id="hansen1993"></a>**Hansen & O'Leary (1993)**. The Use of the L-Curve in the Regularization of Discrete Ill-Posed Problems. [ADS](https://ui.adsabs.harvard.edu/abs/1993SJSC...14.1487H/abstract), [doi:10.1137/0914086](https://doi.org/10.1137/0914086).
- <a id="hartkopf1989"></a>**Hartkopf, McAlister & Franz (1989)**. Binary star orbits from speckle interferometry. II. Combined visual-spectroscopic orbits of 31 close binaries. [ADS](https://ui.adsabs.harvard.edu/abs/1989AJ.....98.1014H/abstract), [doi:10.1086/115193](https://doi.org/10.1086/115193). AJ 98, 1014; bibcode 1989AJ.....98.1014H.
- <a id="jaxoplanet"></a>**Hattori et al. (2024)**. jaxoplanet. [ADS](https://ui.adsabs.harvard.edu/abs/2024zndo..10736936H/abstract), [doi:10.5281/zenodo.10736936](https://doi.org/10.5281/zenodo.10736936), [GitHub](https://github.com/exoplanet-dev/jaxoplanet). Zenodo; https://github.com/exoplanet-dev/jaxoplanet.
- <a id="hillen2016"></a>**Hillen et al. (2016)**. Imaging the dust sublimation front of a circumbinary disk. [arXiv:1603.03023](https://arxiv.org/abs/1603.03023), [ADS](https://ui.adsabs.harvard.edu/abs/2016A&A...588L...1H/abstract), [doi:10.1051/0004-6361/201628125](https://doi.org/10.1051/0004-6361/201628125).
- <a id="hoffman2014"></a>**Hoffman & Gelman (2011)**. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. [arXiv:1111.4246](https://arxiv.org/abs/1111.4246), [ADS](https://ui.adsabs.harvard.edu/abs/2011arXiv1111.4246H/abstract). NUTS; JMLR 15, 1593 (2014).
- <a id="hogg2010recipes"></a>**Hogg, Bovy & Lang (2010)**. Data analysis recipes: Fitting a model to data. [arXiv:1008.4686](https://arxiv.org/abs/1008.4686), [ADS](https://ui.adsabs.harvard.edu/abs/2010arXiv1008.4686H/abstract).
- <a id="hogg2010ecc"></a>**Hogg, Myers & Bovy (2010)**. Inferring the eccentricity distribution. [arXiv:1008.4146](https://arxiv.org/abs/1008.4146), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2166H/abstract).
- <a id="honma2014"></a>**Honma, Akiyama, Uemura & Ikeda (2014)**. Super-resolution imaging with radio interferometry using sparse modeling. [arXiv:1407.2422](https://arxiv.org/abs/1407.2422), [ADS](https://ui.adsabs.harvard.edu/abs/2014PASJ...66...95H/abstract), [doi:10.1093/pasj/psu070](https://doi.org/10.1093/pasj/psu070). PASJ 66, 95.
- <a id="hutchinson1989"></a>**Hutchinson (1989)**. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. [ADS](https://ui.adsabs.harvard.edu/abs/1989CoSSC..18.1059H/abstract), [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806). Commun. Statist. Simul. Comput. 18, 1059; the trace estimator whose Rademacher probes the regularizer part of the diagonal follows (the JᵀJ part uses Gaussian (Jᵀu)² sketches).
- <a id="hogbom1974"></a>**Högbom (1974)**. Aperture synthesis with a non-regular distribution of interferometer baselines. [ADS](https://ui.adsabs.harvard.edu/abs/1974A&AS...15..417H/abstract). A&AS 15, 417; bibcode 1974A&AS...15..417H. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="ireland2013"></a>**Ireland (2013)**. Phase Errors in Diffraction-Limited Imaging: Contrast Limits for Sparse Aperture Masking. [arXiv:1301.6205](https://arxiv.org/abs/1301.6205), [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.433.1718I/abstract). MNRAS 433, 1718.
- <a id="ireland2006"></a>**Ireland, Monnier & Thureau (2006)**. Monte-Carlo imaging for optical interferometry. [arXiv:2007.00716](https://arxiv.org/abs/2007.00716), [ADS](https://ui.adsabs.harvard.edu/abs/2006SPIE.6268E..1TI/abstract), [doi:10.1117/12.670940](https://doi.org/10.1117/12.670940). Proc. SPIE 6268 (arXiv posting dated 2020, id checked in ADS); MACIM, whose test of the shortest baselines the large-scale residual check follows.
- <a id="jasra2011"></a>**Jasra, Stephens, Doucet & Tsagaris (2011)**. Inference for Lévy-Driven Stochastic Volatility Models via Adaptive Sequential Monte Carlo. [doi:10.1111/j.1467-9469.2010.00723.x](https://doi.org/10.1111/j.1467-9469.2010.00723.x). Scand. J. Stat. 38, 1 (online December 2010); ESS-adaptive tempering schedules.
- <a id="jeffreys1946"></a>**Jeffreys (1946)**. An invariant form for the prior probability in estimation problems. [ADS](https://ui.adsabs.harvard.edu/abs/1946RSPSA.186..453J/abstract), [doi:10.1098/rspa.1946.0056](https://doi.org/10.1098/rspa.1946.0056).
- <a id="frank"></a>**Jennings et al. (2020)**. Frankenstein: fitting the visibilities of protoplanetary disc and other radially symmetric sources. [arXiv:2005.07709](https://arxiv.org/abs/2005.07709), [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract), [GitHub](https://github.com/discsim/frank). MNRAS 495, 3209 (2020); github.com/discsim/frank. Ideas only, no code copied (LGPL).
- <a id="junklewitz2016"></a>**Junklewitz et al. (2016)**. RESOLVE: A new algorithm for aperture synthesis imaging of extended emission in radio astronomy. [arXiv:1311.5282](https://arxiv.org/abs/1311.5282), [ADS](https://ui.adsabs.harvard.edu/abs/2016A&A...586A..76J/abstract). A&A 586, A76; an IFT imaging algorithm for radio interferometry.
- <a id="fouriever"></a>**Kammerer**. fouriever. [GitHub](https://github.com/kammerje/fouriever). comparison only: compare linear_flux_grid with its lincmap (closure/kernel phases only); no paper found. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="kammerer2020"></a>**Kammerer et al. (2020)**. Increasing the achievable contrast of infrared interferometry with an error correlation model. [arXiv:2011.01209](https://arxiv.org/abs/2011.01209), [ADS](https://ui.adsabs.harvard.edu/abs/2020A&A...644A.110K/abstract).
- <a id="visr"></a>**Kennedy (2025)**. vis-r: a parametric visibility modelling code for radial profiles of discs. [arXiv:2502.17619](https://arxiv.org/abs/2502.17619), [ADS](https://ui.adsabs.harvard.edu/abs/2025RASTI...4....4D/abstract), [GitHub](https://github.com/drgmk/vis-r). RASTI (2025); github.com/drgmk/vis-r. Ideas only, no code copied.
- <a id="kingma2015"></a>**Kingma & Ba (2014)**. Adam: A Method for Stochastic Optimization. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980), [ADS](https://ui.adsabs.harvard.edu/abs/2014arXiv1412.6980K/abstract).
- <a id="kingma2016"></a>**Kingma et al. (2016)**. Improving Variational Inference with Inverse Autoregressive Flow. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934), [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160604934K/abstract). IAF.
- <a id="kipping2013"></a>**Kipping (2013)**. Efficient, uninformative sampling of limb darkening coefficients for two-parameter laws. [arXiv:1308.0009](https://arxiv.org/abs/1308.0009), [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract), [doi:10.1093/mnras/stt1435](https://doi.org/10.1093/mnras/stt1435).
- <a id="kluska2014"></a>**Kluska et al. (2014)**. SPARCO: a semi-parametric approach for image reconstruction of chromatic objects. [arXiv:1403.3343](https://arxiv.org/abs/1403.3343), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...564A..80K/abstract).
- <a id="knollmuller2019"></a>**Knollmüller & Enßlin (2019)**. Metric Gaussian Variational Inference. [arXiv:1901.11033](https://arxiv.org/abs/1901.11033), [ADS](https://ui.adsabs.harvard.edu/abs/2019arXiv190111033K/abstract).
- <a id="kraus2012"></a>**Kraus & Ireland (2012)**. LkCa 15: A Young Exoplanet Caught at Formation?. [arXiv:1110.3808](https://arxiv.org/abs/1110.3808), [ADS](https://ui.adsabs.harvard.edu/abs/2012ApJ...745....5K/abstract), [doi:10.1088/0004-637X/745/1/5](https://doi.org/10.1088/0004-637X/745/1/5). ApJ 745, 5; motivates fitting a disk and a planet together.
- <a id="kucukelbir2017"></a>**Kucukelbir et al. (2016)**. Automatic Differentiation Variational Inference. [arXiv:1603.00788](https://arxiv.org/abs/1603.00788), [ADS](https://ui.adsabs.harvard.edu/abs/2016arXiv160300788K/abstract). ADVI; JMLR 18, 14 (2017).
- <a id="kuramochi2018"></a>**Kuramochi et al. (2018)**. Superresolution Interferometric Imaging with Sparse Modeling Using Total Squared Variation: Application to Imaging the Black Hole Shadow. [arXiv:1802.05783](https://arxiv.org/abs/1802.05783), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJ...858...56K/abstract), [doi:10.3847/1538-4357/aab6b5](https://doi.org/10.3847/1538-4357/aab6b5). ApJ 858, 56.
- <a id="kustaanheimo1965"></a>**Kustaanheimo & Stiefel (1965)**. Perturbation theory of Kepler motion based on spinor regularization. [doi:10.1515/crll.1965.218.204](https://doi.org/10.1515/crll.1965.218.204). J. reine angew. Math. 218, 204; regularizing maps of Kepler motion.
- <a id="lachaume2021"></a>**Lachaume (2021)**. Bias-free model fitting of correlated data in interferometry. [arXiv:2104.07082](https://arxiv.org/abs/2104.07082), [ADS](https://ui.adsabs.harvard.edu/abs/2021PASA...38...29L/abstract), [doi:10.1017/pasa.2021.20](https://doi.org/10.1017/pasa.2021.20). PASA 38, e033 (2021).
- <a id="lannelongue2021"></a>**Lannelongue, Grealey & Inouye (2021)**. Green Algorithms: Quantifying the Carbon Footprint of Computation. [ADS](https://ui.adsabs.harvard.edu/abs/2021AdvSc...800707L/abstract), [doi:10.1002/advs.202100707](https://doi.org/10.1002/advs.202100707). Advanced Science 8, 2100707; the method behind the development-carbon estimate (docs/dev_carbon.md).
- <a id="lawson2004"></a>**Lawson, Cotton & Hummel (2004)**. An interferometry imaging beauty contest. [ADS](https://ui.adsabs.harvard.edu/abs/2004SPIE.5491..886L/abstract), [doi:10.1117/12.550710](https://doi.org/10.1117/12.550710). Proc. SPIE 5491, 886; Eq. 2.
- <a id="lazareff2017"></a>**Lazareff et al. (2016)**. Structure of Herbig AeBe disks at the milliarcsecond scale: A statistical survey in the H band using PIONIER-VLTI. [arXiv:1611.08428](https://arxiv.org/abs/1611.08428), [ADS](https://ui.adsabs.harvard.edu/abs/2017A&A...599A..85L/abstract). A&A 599, A85 (2017).
- <a id="leegreengard2005"></a>**Lee & Greengard (2005)**. The type 3 nonuniform FFT and its applications. [ADS](https://ui.adsabs.harvard.edu/abs/2005JCoPh.206....1L/abstract), [doi:10.1016/j.jcp.2004.12.004](https://doi.org/10.1016/j.jcp.2004.12.004). J. Comput. Phys. 206, 1; a type-1 spread grid evaluated off the integer lattice (the free offsets).
- <a id="lee2018"></a>**Lee & Whiteley (2018)**. Variance estimation in the particle filter. [arXiv:1509.00394](https://arxiv.org/abs/1509.00394), [doi:10.1093/biomet/asy028](https://doi.org/10.1093/biomet/asy028). Biometrika 105, 609; the single-run variance estimator of the normalizing constant from Eve (time-0 ancestor) indices, eq. 4 with phi = 1; needs multinomial resampling.
- <a id="levicivita1920"></a>**Levi-Civita (1920)**. Sur la régularisation du problème des trois corps. [doi:10.1007/BF02404416](https://doi.org/10.1007/BF02404416). Acta Math. 42, 99; the planar regularizing map.
- <a id="lindgren2011"></a>**Lindgren, Rue & Lindström (2011)**. An Explicit Link between Gaussian Fields and Gaussian Markov Random Fields: The Stochastic Partial Differential Equation Approach. [doi:10.1111/j.1467-9868.2011.00777.x](https://doi.org/10.1111/j.1467-9868.2011.00777.x). JRSS B 73, 423; Matérn fields as solutions of an SPDE.
- <a id="liu1989"></a>**Liu & Nocedal (1989)**. On the limited memory BFGS method for large scale optimization. [ADS](https://ui.adsabs.harvard.edu/abs/1989MatPr..45..503L/abstract), [doi:10.1007/BF01589116](https://doi.org/10.1007/BF01589116). L-BFGS.
- <a id="lucy2014"></a>**Lucy (2014)**. Mass estimates for visual binaries with incomplete orbits. [arXiv:1309.2868](https://arxiv.org/abs/1309.2868), [ADS](https://ui.adsabs.harvard.edu/abs/2014A&A...563A.126L/abstract). A&A 563, A126.
- <a id="starry"></a>**Luger et al. (2018)**. STARRY: Analytic Occultation Light Curves. [arXiv:1810.06559](https://arxiv.org/abs/1810.06559), [ADS](https://ui.adsabs.harvard.edu/abs/2019AJ....157...64L/abstract). AJ 157, 64 (2019).
- <a id="luger2017linear"></a>**Luger, Foreman-Mackey & Hogg (2017)**. Linear Models for Systematics and Nuisances. [arXiv:1710.11136](https://arxiv.org/abs/1710.11136), [ADS](https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract), [doi:10.3847/2515-5172/aa96b5](https://doi.org/10.3847/2515-5172/aa96b5). RNAAS 1, 7.
- <a id="mackay1992"></a>**MacKay (1992)**. Bayesian Interpolation. [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415). Neural Comput. 4, 415.
- <a id="malbet2010"></a>**Malbet et al. (2010)**. The 2010 Interferometric Imaging Beauty Contest. [arXiv:1007.4473](https://arxiv.org/abs/1007.4473), [ADS](https://ui.adsabs.harvard.edu/abs/2010SPIE.7734E..2NM/abstract). SPIE 7734.
- <a id="martens2010"></a>**Martens (2010)**. Deep learning via Hessian-free optimization. [link](https://icml.cc/Conferences/2010/papers/458.pdf). ICML 27, 735-742; the Gauss-Newton matrix as the curvature matrix. No DOI or arXiv ID, not on ADS: title and author read from the ICML 2010 PDF. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="martinache2010"></a>**Martinache (2010)**. Kernel-Phase in Fizeau Inteferometry. [arXiv:1009.3933](https://arxiv.org/abs/1009.3933), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...724..464M/abstract). ApJ 724, 464 (2010).
- <a id="xara"></a>**Martinache (2010)**. Kernel-Phase in Fizeau Inteferometry. [arXiv:1009.3933](https://arxiv.org/abs/1009.3933), [ADS](https://ui.adsabs.harvard.edu/abs/2010ApJ...724..464M/abstract), [GitHub](https://github.com/fmartinache/xara). xara was formerly pysco; cvis_binary is adapted from it.
- <a id="mawet2014"></a>**Mawet et al. (2014)**. Fundamental limitations of high contrast imaging set by small sample statistics. [arXiv:1407.2247](https://arxiv.org/abs/1407.2247), [ADS](https://ui.adsabs.harvard.edu/abs/2014ApJ...792...97M/abstract). ApJ 792, 97; the small-sample false-alarm correction, not applied in injection_recovery.
- <a id="mengersen1996"></a>**Mengersen & Tweedie (1996)**. Rates of convergence of the Hastings and Metropolis algorithms. [doi:10.1214/aos/1033066201](https://doi.org/10.1214/aos/1033066201). Ann. Stat. 24, 101; independence Metropolis-Hastings is uniformly ergodic iff the weight is bounded.
- <a id="miles1965"></a>**Miles (1965)**. On random rotations in R^3. [doi:10.1093/biomet/52.3-4.636](https://doi.org/10.1093/biomet/52.3-4.636). Biometrika 52, 636-639; the isotropic (Haar) distribution of rotations: Omega and omega uniform, cos i uniform. Not on ADS: checked on Crossref.
- <a id="monnier2007"></a>**Monnier et al. (2007)**. Imaging the Surface of Altair. [arXiv:0706.0867](https://arxiv.org/abs/0706.0867), [ADS](https://ui.adsabs.harvard.edu/abs/2007Sci...317..342M/abstract), [doi:10.1126/science.1143205](https://doi.org/10.1126/science.1143205). Science 317, 342; the Altair-like truth of notebooks/mwe/mwe_elr_chara.ipynb is inspired by it, not a reproduction.
- <a id="moran1975"></a>**Moran (1975)**. Quaternions, Haar measure and the estimation of a palaeomagnetic rotation. [doi:10.1017/s0021900200047720](https://doi.org/10.1017/s0021900200047720). J. Appl. Prob. 12, 295-301; Haar measure on SO(3) is the uniform measure on S^3, covered 2:1 by unit quaternions. Not on ADS: checked on Crossref.
- <a id="merand2022"></a>**Mérand (2022)**. Flexible Spectro Interferometric modelling of OIFITS data with PMOIRED. [arXiv:2207.11047](https://arxiv.org/abs/2207.11047), [ADS](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1NM/abstract), [GitHub](https://github.com/amerand/pmoired). PMOIRED.
- <a id="candid"></a>**Mérand et al.**. CANDID: Companion Analysis and Non-Detection in Interferometric Data. [GitHub](https://github.com/amerand/CANDID). algorithm paper: gallenne2015. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="offringa2014"></a>**Offringa et al. (2014)**. WSCLEAN: an implementation of a fast, generic wide-field imager for radio astronomy. [arXiv:1407.1943](https://arxiv.org/abs/1407.1943), [ADS](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..606O/abstract). MNRAS 444, 606; bibcode 2014MNRAS.444..606O; the WSClean argument conventions radio_clean follows.
- <a id="nufftax"></a>**Oudoumanessah & Iollo**. nufftax. [GitHub](https://github.com/GragasLab/nufftax). no paper; its spreader and kernel parameters (private API, nufftax >= 0.6.1). *Not yet verified against arXiv, Crossref or ADS.*
- <a id="papaspiliopoulos2007"></a>**Papaspiliopoulos, Roberts & Sköld (2007)**. A General Framework for the Parametrization of Hierarchical Models. [ADS](https://ui.adsabs.harvard.edu/abs/2007StaSc..2200014P/abstract), [doi:10.1214/088342307000000014](https://doi.org/10.1214/088342307000000014). Statistical Science 22, 59; centred and non-centred parameterizations.
- <a id="pauls2005"></a>**Pauls et al. (2005)**. A Data Exchange Standard for Optical (Visible/IR) Interferometry. [arXiv:astro-ph/0508185](https://arxiv.org/abs/astro-ph/0508185), [ADS](https://ui.adsabs.harvard.edu/abs/2005PASP..117.1255P/abstract), [doi:10.1086/444523](https://doi.org/10.1086/444523). OIFITS 1; PASP 117, 1255.
- <a id="pawitan2001"></a>**Pawitan (2001)**. In All Likelihood: Statistical Modelling and Inference Using Likelihood. [doi:10.1093/oso/9780198507659.001.0001](https://doi.org/10.1093/oso/9780198507659.001.0001). Oxford University Press (book); the profile likelihood of the noise scale, sigma_hat = sqrt(chi2 / 2N). Not on ADS: checked on Crossref.
- <a id="phipson2010"></a>**Phipson & Smyth (2010)**. Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585). Stat. Appl. Genet. Mol. Biol. 9, 39; (1 + #)/(1 + n) permutation p-value.
- <a id="jaxbessel"></a>**Pope et al.**. jaxbessel. [GitHub](https://github.com/benjaminpope/jaxbessel). CEPHES (Moshier, https://www.netlib.org/cephes/) translation, adapted from harmonix; the Cephes credit lives here and in the jaxbessel repository. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="pricewhelan2017"></a>**Price-Whelan et al. (2017)**. The Joker: A custom Monte Carlo sampler for binary-star and exoplanet radial velocity data. [arXiv:1610.07602](https://arxiv.org/abs/1610.07602), [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...20P/abstract). ApJ 837, 20.
- <a id="pricewhelan2020"></a>**Price-Whelan et al. (2020)**. Close Binary Companions to APOGEE DR16 Stars: 20,000 Binary-star Systems Across the Color-Magnitude Diagram. [arXiv:2002.00014](https://arxiv.org/abs/2002.00014), [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJ...895....2P/abstract). ApJ 895, 2; The Joker at scale.
- <a id="quirrenbach1996"></a>**Quirrenbach et al. (1996)**. Angular diameter and limb darkening of Arcturus. [ADS](https://ui.adsabs.harvard.edu/abs/1996A&A...312..160Q/abstract). A&A 312, 160; bibcode 1996A&A...312..160Q. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="rasmussen2006"></a>**Rasmussen & Williams (2006)**. Gaussian Processes for Machine Learning. [doi:10.7551/mitpress/3206.001.0001](https://doi.org/10.7551/mitpress/3206.001.0001), [link](https://gaussianprocess.org/gpml/). MIT Press (book); GP terminology, Matérn kernels, marginal likelihood; A.9 matrix inversion and determinant lemmas (arrow_at).
- <a id="rau2011"></a>**Rau & Cornwell (2011)**. A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry. [arXiv:1106.2745](https://arxiv.org/abs/1106.2745), [ADS](https://ui.adsabs.harvard.edu/abs/2011A&A...532A..71R/abstract), [doi:10.1051/0004-6361/201117104](https://doi.org/10.1051/0004-6361/201117104). A&A 532, A71.
- <a id="roettenbacher2016"></a>**Roettenbacher et al. (2016)**. No Sun-like dynamo on the active star ζ Andromedae from starspot asymmetry. [arXiv:1709.10107](https://arxiv.org/abs/1709.10107), [ADS](https://ui.adsabs.harvard.edu/abs/2016Natur.533..217R/abstract), [doi:10.1038/nature17444](https://doi.org/10.1038/nature17444). Nature 533, 217; CHARA/MIRC starspot imaging that motivates HarmonixModel.
- <a id="rudin1992"></a>**Rudin, Osher & Fatemi (1992)**. Nonlinear total variation based noise removal algorithms. [ADS](https://ui.adsabs.harvard.edu/abs/1992PhyD...60..259R/abstract), [doi:10.1016/0167-2789(92)90242-F](https://doi.org/10.1016/0167-2789(92)90242-F). Physica D 60, 259.
- <a id="rue2005"></a>**Rue & Held (2005)**. Gaussian Markov Random Fields: Theory and Applications. [doi:10.1201/9780203492024](https://doi.org/10.1201/9780203492024). Chapman & Hall/CRC (book).
- <a id="ruffio2018"></a>**Ruffio et al. (2018)**. A Bayesian Framework for Exoplanet Direct Detection and Non-Detection. [arXiv:1809.08261](https://arxiv.org/abs/1809.08261), [ADS](https://ui.adsabs.harvard.edu/abs/2018AJ....156..196R/abstract).
- <a id="sallum2015"></a>**Sallum et al. (2015)**. Accreting protoplanets in the LkCa 15 transition disk. [arXiv:1511.07456](https://arxiv.org/abs/1511.07456), [ADS](https://ui.adsabs.harvard.edu/abs/2015Natur.527..342S/abstract), [doi:10.1038/nature15761](https://doi.org/10.1038/nature15761). Nature 527, 342; motivates fitting a disk and a planet together.
- <a id="schraudolph2002"></a>**Schraudolph (2002)**. Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent. [doi:10.1162/08997660260028683](https://doi.org/10.1162/08997660260028683). Neural Comput. 14, 1723; Gauss-Newton matrix-vector products by forward-mode differentiation. Not on ADS: checked on Crossref.
- <a id="schwab1984"></a>**Schwab (1984)**. Relaxing the isoplanatism assumption in self-calibration; applications to low-frequency radio interferometry. [ADS](https://ui.adsabs.harvard.edu/abs/1984AJ.....89.1076S/abstract). AJ 89, 1076; bibcode 1984AJ.....89.1076S; the Cotton-Schwab major cycle (residual from the visibilities). Cotton is cited there as a private communication. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="selfliang1987"></a>**Self & Liang (1987)**. Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. [ADS](https://ui.adsabs.harvard.edu/abs/1987JASA...82..605S/abstract), [doi:10.1080/01621459.1987.10478472](https://doi.org/10.1080/01621459.1987.10478472). Nonstandard conditions: a parameter on the boundary, or a nuisance parameter unidentified under the null.
- <a id="selig2013"></a>**Selig et al. (2013)**. NIFTY - Numerical Information Field Theory - a versatile Python library for signal inference. [arXiv:1301.4499](https://arxiv.org/abs/1301.4499), [ADS](https://ui.adsabs.harvard.edu/abs/2013A&A...554A..26S/abstract).
- <a id="shoemake1992"></a>**Shoemake (1992)**. Uniform Random Rotations. [doi:10.1016/b978-0-08-050755-2.50036-1](https://doi.org/10.1016/b978-0-08-050755-2.50036-1). Graphics Gems III, 124; unit quaternions and the Haar measure of SO(3).
- <a id="sivaramakrishnan2023"></a>**Sivaramakrishnan et al. (2022)**. The Near Infrared Imager and Slitless Spectrograph for the James Webb Space Telescope. IV. Aperture Masking Interferometry. [arXiv:2210.17434](https://arxiv.org/abs/2210.17434), [ADS](https://ui.adsabs.harvard.edu/abs/2023PASP..135a5003S/abstract). PASP 135, 015003 (2023).
- <a id="skilling1989"></a>**Skilling (1989)**. Classic Maximum Entropy. [doi:10.1007/978-94-015-7860-8_3](https://doi.org/10.1007/978-94-015-7860-8_3). Maximum Entropy and Bayesian Methods, Cambridge 1988. docs/contributors.md links its Skilling 1989 entry to DOI ..._4, which Crossref gives as the Gull 1989 chapter.
- <a id="skillingbryan1984"></a>**Skilling & Bryan (1984)**. Maximum entropy image reconstruction: general algorithm. [ADS](https://ui.adsabs.harvard.edu/abs/1984MNRAS.211..111S/abstract). MNRAS 211, 111; bibcode 1984MNRAS.211..111S; precondition="diag" generalizes its entropy metric to the local curvature of data and regularizers. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="amical"></a>**Soulain et al.**. AMICAL: aperture masking interferometry calibration and analysis. [GitHub](https://github.com/SAIL-Labs/AMICAL). separate package, bibliography only; paper reference to add (SPIE 2020). *Not yet verified against arXiv, Crossref or ADS.*
- <a id="soummer2007"></a>**Soummer et al. (2007)**. Fast computation of Lyot-style coronagraph propagation. [arXiv:0711.0368](https://arxiv.org/abs/0711.0368), [ADS](https://ui.adsabs.harvard.edu/abs/2007OExpr..1515935S/abstract). matrix Fourier transform.
- <a id="starck2010"></a>**Starck, Murtagh & Fadili (2010)**. Sparse Image and Signal Processing: Wavelets, Curvelets, Morphological Diversity. book (CUP). *Not yet verified against arXiv, Crossref or ADS.*
- <a id="surjanovic2023"></a>**Surjanovic, Biron-Lattes, Tiede et al. (2023)**. Pigeons.jl: Distributed Sampling From Intractable Distributions. [arXiv:2308.09769](https://arxiv.org/abs/2308.09769), [ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv230809769S/abstract). parallel tempering prior art next to Octofitter.
- <a id="thiele1883"></a>**Thiele (1883)**. Neue Methode zur Berechung von Doppelsternbahnen. [ADS](https://ui.adsabs.harvard.edu/abs/1883AN....104..245T/abstract). Astron. Nachr. 104, 245; title as spelled by ADS. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="thiebaut2017"></a>**Thiébaut & Young (2017)**. Principles of image reconstruction in optical interferometry: tutorial. [arXiv:1708.08390](https://arxiv.org/abs/1708.08390), [ADS](https://ui.adsabs.harvard.edu/abs/2017JOSAA..34..904T/abstract). JOSA A 34, 904.
- <a id="thompson2023"></a>**Thompson et al. (2023)**. Octofitter: Fast, Flexible, and Accurate Orbit Modelling to Detect Exoplanets. [arXiv:2402.01971](https://arxiv.org/abs/2402.01971), [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....166..164T/abstract), [doi:10.3847/1538-3881/acf5cc](https://doi.org/10.3847/1538-3881/acf5cc), [GitHub](https://github.com/sefffal/Octofitter.jl). AJ 166, 164 (2023); the arXiv preprint was posted in February 2024, after publication.
- <a id="tiede2025"></a>**Tiede et al. (2025)**. Hierarchical Interferometric Bayesian Imaging. [arXiv:2511.17706](https://arxiv.org/abs/2511.17706), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ...997..262T/abstract). HIBI/Comrade.
- <a id="vanhamme1993"></a>**van Hamme (1993)**. New Limb-Darkening Coefficients for Modeling Binary Star Light Curves. [ADS](https://ui.adsabs.harvard.edu/abs/1993AJ....106.2096V/abstract), [doi:10.1086/116788](https://doi.org/10.1086/116788). AJ 106, 2096; the quadratic law the square-root law is compared with.
- <a id="vanderplas2018"></a>**VanderPlas (2017)**. Understanding the Lomb-Scargle Periodogram. [arXiv:1703.09824](https://arxiv.org/abs/1703.09824), [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract). ApJS 236, 16 (2018).
- <a id="vehtari2024"></a>**Vehtari et al. (2015)**. Pareto Smoothed Importance Sampling. [arXiv:1507.02646](https://arxiv.org/abs/1507.02646), [ADS](https://ui.adsabs.harvard.edu/abs/2015arXiv150702646V/abstract). JMLR 25, 72 (2024).
- <a id="vehtari2021"></a>**Vehtari et al. (2021)**. Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. [arXiv:1903.08008](https://arxiv.org/abs/1903.08008), [ADS](https://ui.adsabs.harvard.edu/abs/2021BayAn..16..667V/abstract). Bayesian Analysis 16, 667; R-hat is trusted only with bulk ESS above 100 per chain.
- <a id="virgil_tempering"></a>**virgil (own design) (2026)**. Tempering an evidence for correlated spectral channels. No published source. The effective-channel factor from neighbouring-channel residual covariance is virgil's own method; provisional, and no test checks that tempered evidences are calibrated. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="virgil_short_arcs"></a>**virgil (own design) (2026)**. Short-arc motion constraints for the orbit period search. No published source. Provisional; not independently validated. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="virgil_alias_search"></a>**virgil (own design) (2026)**. Period-alias band search for sparsely sampled orbits. No published source. The search over cycle-count bands, its nats thresholds and its night-outlier mixture (marginalized rate, Beta(1/2, 1/2) (Jeffreys)) are virgil's own design, provisional and not independently validated end to end. *Not yet verified against arXiv, Crossref or ADS.*
- <a id="wilks1938"></a>**Wilks (1938)**. The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses. [doi:10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360).
- <a id="wright2009"></a>**Wright & Howard (2009)**. Efficient fitting of multiplanet Keplerian models to radial velocity and astrometry data. [arXiv:0904.3725](https://arxiv.org/abs/0904.3725), [ADS](https://ui.adsabs.harvard.edu/abs/2009ApJS..182..205W/abstract). ApJS 182, 205.
- <a id="frap"></a>**Yoshida et al. (2026)**. Direct Retrieval of Protoplanetary Disk Dust Properties Using Auto-differentiable Gaussian Processes and Their Application to the HD 169142 Disk. [arXiv:2607.12618](https://arxiv.org/abs/2607.12618), [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ..1007..198Y/abstract), [doi:10.3847/1538-4357/ae89ad](https://doi.org/10.3847/1538-4357/ae89ad), [GitHub](https://github.com/tomyoshida/frap). ApJ 1007, 198 (2026); github.com/tomyoshida/frap. Ideas only, no code copied: whitened Gaussian-process latents (g = mu + L z) at radial nodes with a dense kernel, sampled by NUTS, and a radial GP per physical quantity.
- <a id="zawadzki2023"></a>**Zawadzki et al. (2023)**. Regularized Maximum Likelihood Image Synthesis and Validation for ALMA Continuum Observations of Protoplanetary Disks. [arXiv:2209.11813](https://arxiv.org/abs/2209.11813), [ADS](https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z/abstract), [doi:10.1088/1538-3873/acdf84](https://doi.org/10.1088/1538-3873/acdf84). PASP 135, 064503 (MPoL); the RML objective for ALMA visibilities that virgil.radio reproduces (design register A).
- <a id="zhou2016"></a>**Zhou, Johansen & Aston (2016)**. Toward Automatic Model Comparison: An Adaptive Sequential Monte Carlo Approach. [doi:10.1080/10618600.2015.1060885](https://doi.org/10.1080/10618600.2015.1060885). JCGS 25, 701; conditional ESS for the next temperature.
