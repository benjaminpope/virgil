# Composition: `System` and building blocks

See the [Composing Models](../../../composition.md) tutorial for usage.

::: virgil.models.System
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - model
        - render

::: virgil.models.Component
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.PointSource
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GaussianDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.EllipticalGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GaussianArc
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.TruncatedCone
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.UniformDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.LimbDarkenedDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.QuadraticLimbDarkenedDisk
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - from_u

::: virgil.models.SquareRootLimbDarkenedDisk
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - from_cd

::: virgil.models.EllipticalLimbDarkenedDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.GravityDarkenedStar
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - plot_surface

::: virgil.models.ModulatedGaussianRim
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDisk
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskHG
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskGaussian
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.FlaredDiskPowerLaw
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.Resolved
    options:
      show_root_heading: true
      heading_level: 2
      members: false

::: virgil.models.Rotated
    options:
      show_root_heading: true
      heading_level: 2

::: virgil.models.Attached
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - at

Models are fitted through [`virgil.likelihood`](../../likelihood.md)
(`build_model`, `loglike`, `numpyro_model`).
