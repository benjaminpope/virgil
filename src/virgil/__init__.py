"""virgil: the Versatile Interferometric Reconstruction and Gradient-based Inference Library.

The everyday names are available at the top level::

    from virgil import OIData, System, PointSource, likelihood_grid

Modules:

* [`oidata`][virgil.oidata] and [`oifits`][virgil.oifits]: data and
  OIFITS files; [`amigo`][virgil.amigo]: AMIGO mixed-DISCO products.
* [`models`][virgil.models]: source models and their visibilities.
* [`likelihood`][virgil.likelihood]: likelihoods and numpyro models.
* [`fitting`][virgil.fitting]: `fit`, maximum a posteriori fits with
  Levenberg–Marquardt, L-BFGS or Adam.
* [`imaging`][virgil.imaging]: regularisers and helpers for image
  reconstruction; [`scenes`][virgil.scenes]: synthetic truth images;
  [`coverage`][virgil.coverage]: synthetic coverage and noise.
* [`inference`][virgil.inference]: Laplace and Fisher curvature.
* [`grid_fit`][virgil.grid_fit]: grid searches.
* [`limits`][virgil.limits]: contrast limits and flux/contrast/Δmag
  conversions.
* [`spectra`][virgil.spectra]: wavelength-dependent fluxes.
* [`plotting`][virgil.plotting]: figures.

The legacy ImPlaneIA tools in ``virgil.legacy`` are not imported here.
"""

import importlib.metadata as _metadata

name = "virgil"

try:
    __version__ = _metadata.version("virgil-astro")
except _metadata.PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = "unknown"

from . import (  # noqa: E402
    amigo,
    coverage,
    fields,
    fitting,
    grid_fit,
    imaging,
    inference,
    likelihood,
    limits,
    models,
    oidata,
    oifits,
    orbits,
    plotting,
    scenes,
    simulate,
    spectra,
)
from ._geometry import pixel_offsets  # noqa: E402
from .fields import GaussianField  # noqa: E402
from .fitting import fit  # noqa: E402
from .grid_fit import (  # noqa: E402
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from .inference import fisher, laplace_cov  # noqa: E402
from .likelihood import (  # noqa: E402
    build_model,
    inflated_errors,
    loglike,
    model_loglike,
    numpyro_model,
    whitened_residuals,
)
from .limits import (  # noqa: E402
    absil_limits,
    contrast_to_flux,
    delta_mag_to_flux,
    flux_to_contrast,
    flux_to_delta_mag,
    radial_profile,
    ruffio_upperlimit,
)
from .models import (  # noqa: E402
    Attached,
    BinaryModelAngular,
    BinaryModelCartesian,
    EllipticalGaussian,
    FlaredDiskGaussian,
    FlaredDiskHG,
    FlaredDiskPowerLaw,
    GaussianArc,
    GaussianDisk,
    GravityDarkenedStar,
    HarmonixModel,
    Image,
    LimbDarkenedDisk,
    ModulatedGaussianRim,
    PointSource,
    QuadraticLimbDarkenedDisk,
    Resolved,
    SourceModel,
    SquareRootLimbDarkenedDisk,
    System,
    TruncatedCone,
    UniformDisk,
    circular_support,
)
from .oidata import OIData  # noqa: E402
from .oifits import read_oifits, write_oifits  # noqa: E402
from .orbits import (  # noqa: E402
    AxialVonMises,
    KeplerOrbit,
    PositionData,
    RVData,
    StateVectorOrbit,
    ThieleInnesOrbit,
    distance_pc,
    starting_orbits,
    total_mass,
)
from .spectra import (  # noqa: E402
    BlackBody,
    GaussianLine,
    LorentzianLine,
    Nodes,
    PowerLaw,
    Sum,
)


__all__ = [
    "Attached",
    "AxialVonMises",
    "BinaryModelAngular",
    "BinaryModelCartesian",
    "BlackBody",
    "EllipticalGaussian",
    "FlaredDiskGaussian",
    "FlaredDiskHG",
    "FlaredDiskPowerLaw",
    "GaussianArc",
    "GaussianDisk",
    "GaussianField",
    "GaussianLine",
    "GravityDarkenedStar",
    "HarmonixModel",
    "Image",
    "KeplerOrbit",
    "LimbDarkenedDisk",
    "LorentzianLine",
    "ModulatedGaussianRim",
    "Nodes",
    "OIData",
    "PointSource",
    "PositionData",
    "RVData",
    "PowerLaw",
    "QuadraticLimbDarkenedDisk",
    "Resolved",
    "SourceModel",
    "StateVectorOrbit",
    "SquareRootLimbDarkenedDisk",
    "Sum",
    "System",
    "ThieleInnesOrbit",
    "TruncatedCone",
    "UniformDisk",
    "absil_limits",
    "circular_support",
    "best_grid_point",
    "build_model",
    "contrast_to_flux",
    "delta_mag_to_flux",
    "distance_pc",
    "fit",
    "fisher",
    "inflated_errors",
    "flux_to_contrast",
    "flux_to_delta_mag",
    "laplace_cov",
    "laplace_flux_uncertainty_grid",
    "likelihood_grid",
    "loglike",
    "model_loglike",
    "numpyro_model",
    "optimized_flux_grid",
    "optimized_likelihood_grid",
    "pixel_offsets",
    "radial_profile",
    "read_oifits",
    "ruffio_upperlimit",
    "starting_orbits",
    "total_mass",
    "whitened_residuals",
    "write_oifits",
]
