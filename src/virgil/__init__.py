"""virgil: the Versatile Interferometric Reconstruction and Gradient-based Inference Library.

The everyday names are available at the top level::

    from virgil import OIData, System, PointSource, likelihood_grid

Modules:

* Data: [`oidata`][virgil.oidata], [`amigo`][virgil.amigo] (mixed-DISCO products).
* Models and inference: [`models`][virgil.models],
  [`likelihood`][virgil.likelihood], [`fitting`][virgil.fitting] (`fit`),
  [`svi`][virgil.svi] (`variational`), [`inference`][virgil.inference]
  (Laplace, Fisher), [`priors`][virgil.priors], [`gains`][virgil.gains],
  [`spectra`][virgil.spectra].
* Imaging: [`imaging`][virgil.imaging] (regularizers and helpers),
  [`ensemble`][virgil.ensemble], [`scenes`][virgil.scenes],
  [`metrics`][virgil.metrics], [`coverage`][virgil.coverage].
* Searches and limits: [`grid_fit`][virgil.grid_fit],
  [`limits`][virgil.limits], [`detection`][virgil.detection].
* Orbits: [`orbits`][virgil.orbits], [`epochs`][virgil.epochs].
* [`simulate`][virgil.simulate], [`plotting`][virgil.plotting],
  [`oifits`][virgil.oifits] (with ``read_oifits`` and ``write_oifits``) and
  [`pipeline`][virgil.pipeline] (provisional; imported on first use, which
  keeps matplotlib out of ``import virgil``; ``pipeline`` needs
  the ``pipeline`` extra).

The legacy ImPlaneIA tools in ``virgil.legacy`` are not imported here.

Everything public is **stable** (a deprecation cycle before any change)
except **provisional** features listed in a module's ``__provisional__``
tuple (new in 0.4: orbit search, ``MultipoleDisk``, pipelines, GP ensembles,
SVI), which may change in any 0.4.x release; see the Stability page.
"""

import importlib.metadata as _metadata

try:
    __version__ = _metadata.version("virgil-astro")
except _metadata.PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = "unknown"

from . import (  # noqa: E402
    aliases,
    amigo,
    coverage,
    detection,
    ensemble,
    epochs,
    fields,
    fitting,
    gains,
    grid_fit,
    harmonics,
    imaging,
    inference,
    likelihood,
    limits,
    metrics,
    models,
    oidata,
    orbits,
    priors,
    radio,
    scenes,
    simulate,
    spectra,
    svi,
)
from ._geometry import pixel_offsets, set_ft_defaults  # noqa: E402
from .detection import detection_statistics  # noqa: E402
from .epochs import Epochs, rank_orbits, start_from_positions  # noqa: E402
from .fields import GaussianField, MultipoleField  # noqa: E402
from .fitting import fit  # noqa: E402
from .grid_fit import (  # noqa: E402
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    linear_flux_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from .inference import (  # noqa: E402
    fisher,
    laplace_cov,
    observed_information,
)
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
    injection_limits,
    radial_profile,
    ruffio_upperlimit,
)
from .models import (  # noqa: E402
    Attached,
    BinaryModelAngular,
    BinaryModelCartesian,
    Cone,
    EllipticalGaussian,
    EllipticalLimbDarkenedDisk,
    ExpTail,
    FlaredDiskGaussian,
    FlaredDiskHG,
    FlaredDiskPowerLaw,
    Gap,
    GaussianArc,
    GaussianDisk,
    GaussianRing,
    GravityDarkenedStar,
    HarmonixModel,
    Harmonics,
    Image,
    LimbDarkenedDisk,
    LinearImage,
    ModulatedGaussianRim,
    OrbitalBinary,
    PointSource,
    MultipoleDisk,
    MultipoleKernel,
    PowerLawHeight,
    QuadraticLimbDarkenedDisk,
    RadialGP,
    RadialGPFunction,
    RadialLinear,
    RadialPower,
    RadialPowerLaw,
    RadialSum,
    Resolved,
    SourceModel,
    SquareRootLimbDarkenedDisk,
    System,
    ThinRing,
    TruncatedCone,
    UniformDisk,
    VonMises,
    circular_support,
    gp_priors,
    multipole_disk_cvis,
)
from .oidata import OIData  # noqa: E402
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
from .radio import (  # noqa: E402
    CalibrationNuisance,
    ComplexVisData,
    dartboard_folds,
    primary_beam,
)
from .spectra import (  # noqa: E402
    BlackBody,
    GaussianLine,
    LorentzianLine,
    Nodes,
    PowerLaw,
    Scaled,
    Sum,
)


_LAZY = {  # imported on first use: matplotlib is slow to load; astropy follows once oidata stops importing oifits
    "pipeline": ("pipeline", None),
    "plotting": ("plotting", None),
    "oifits": ("oifits", None),
    "read_oifits": ("oifits", "read_oifits"),
    "write_oifits": ("oifits", "write_oifits"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        mod = importlib.import_module(f".{module}", __name__)
        return mod if attr is None else getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Attached",
    "AxialVonMises",
    "BinaryModelAngular",
    "BinaryModelCartesian",
    "BlackBody",
    "Cone",
    "EllipticalGaussian",
    "EllipticalLimbDarkenedDisk",
    "Epochs",
    "ExpTail",
    "FlaredDiskGaussian",
    "FlaredDiskHG",
    "FlaredDiskPowerLaw",
    "Gap",
    "GaussianArc",
    "GaussianDisk",
    "GaussianField",
    "MultipoleField",
    "GaussianRing",
    "GaussianLine",
    "GravityDarkenedStar",
    "HarmonixModel",
    "Image",
    "LinearImage",
    "KeplerOrbit",
    "Harmonics",
    "LimbDarkenedDisk",
    "LorentzianLine",
    "ModulatedGaussianRim",
    "Nodes",
    "OIData",
    "OrbitalBinary",
    "PointSource",
    "MultipoleDisk",
    "MultipoleKernel",
    "PowerLawHeight",
    "PositionData",
    "CalibrationNuisance",
    "ComplexVisData",
    "dartboard_folds",
    "primary_beam",
    "RVData",
    "PowerLaw",
    "QuadraticLimbDarkenedDisk",
    "RadialGP",
    "RadialGPFunction",
    "RadialLinear",
    "RadialPower",
    "RadialPowerLaw",
    "RadialSum",
    "Resolved",
    "SourceModel",
    "StateVectorOrbit",
    "SquareRootLimbDarkenedDisk",
    "Scaled",
    "Sum",
    "System",
    "ThieleInnesOrbit",
    "ThinRing",
    "TruncatedCone",
    "UniformDisk",
    "VonMises",
    "absil_limits",
    "injection_limits",
    "circular_support",
    "best_grid_point",
    "build_model",
    "contrast_to_flux",
    "delta_mag_to_flux",
    "detection_statistics",
    "distance_pc",
    "fit",
    "inflated_errors",
    "flux_to_contrast",
    "flux_to_delta_mag",
    "laplace_cov",
    "laplace_flux_uncertainty_grid",
    "likelihood_grid",
    "loglike",
    "model_loglike",
    "numpyro_model",
    "observed_information",
    "linear_flux_grid",
    "optimized_flux_grid",
    "optimized_likelihood_grid",
    "pixel_offsets",
    "set_ft_defaults",
    "gp_priors",
    "multipole_disk_cvis",
    "radial_profile",
    "rank_orbits",
    "read_oifits",
    "ruffio_upperlimit",
    "start_from_positions",
    "starting_orbits",
    "total_mass",
    "whitened_residuals",
    "write_oifits",
]
