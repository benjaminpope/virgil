# `virgil.priors`

Priors that are invariant measures but not plain numpyro distributions: the
isotropic orientation priors, uniform in cos i (orbit inclination) and in
sin(lat) (a position on a sphere). Locations take `Uniform`, scales take
`LogUniform`, and longitudes and nodes are uniform in their angles. See
[Which priors?](../conventions.md#which-priors) for the full table and the reasons.
`hierarchical_scales` draws a set of scales, such as one error scale per
epoch, from a log-normal population with log-uniform hyperpriors, for use as
`noise=` terms tied to the parameters.

::: virgil.priors
    options:
      members:
        - IsotropicInclination
        - IsotropicLatitude
        - hierarchical_scales
        - PopulationScale
