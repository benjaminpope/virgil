"""Components built inside ``jax.jit`` from concrete shapes and traced flux.

A fit's model function often builds a component from fixed shape parameters
(concrete JAX arrays) and a traced flux. The parameter checks must not stage
those concrete values into tracers.
"""

import jax
import jax.numpy as np
import pytest

from virgil.models import (
    EllipticalGaussian,
    EllipticalLimbDarkenedDisk,
    FlaredDiskHG,
    GaussianArc,
    GravityDarkenedStar,
    Image,
    ModulatedGaussianRim,
    TruncatedCone,
    UniformDisk,
)
from virgil.spectra import Nodes

WAVEL = np.array([1.5e-6, 1.65e-6, 1.8e-6])
U = np.array([1e6, 2e6, 3e6, 4e6])
V = np.array([2e6, -1e6, 0.5e6, 3e6])

# class, constructor arguments, attributes of an instance to rebuild from
CASES = [
    (UniformDisk, dict(diam=3.0), ("diam",)),
    (
        EllipticalGaussian,
        dict(fwhm=3.0, ratio=0.6, pa=20.0),
        ("fwhm", "ratio", "pa"),
    ),
    (
        EllipticalLimbDarkenedDisk,
        dict(diam=3.0, ratio=0.6, pa=20.0, u=[0.5]),
        ("diam", "ratio", "pa", "u"),
    ),
    (
        GaussianArc,
        dict(radius=6.0, width=1.0, length=4.0, pa=10.0, nodes=16),
        ("radius", "width", "length", "pa", "nodes"),
    ),
    (
        TruncatedCone,
        dict(
            tip=11.9,
            alpha=62.5,
            s0=7.8,
            length=13.8,
            width=1.7,
            tilt=0.34,
            pa=96.5,
            n_rings=8,
        ),
        ("tip", "alpha", "s0", "length", "width", "tilt", "pa", "n_rings"),
    ),
    (
        GravityDarkenedStar,
        dict(diam_eq=2.0, omega=0.3, inc=60.0, pa=15.0, n_lat=8),
        ("diam_eq", "omega", "inc", "pa", "n_lat"),
    ),
    (
        ModulatedGaussianRim,
        dict(
            diam=8.0,
            fwhm=2.0,
            inc=40.0,
            pa=30.0,
            az_amps=[0.3, 0.1],
            az_pas=[10.0, 50.0],
        ),
        ("diam", "fwhm", "inc", "pa", "az_amps", "az_pas"),
    ),
    (
        Image,
        dict(
            log_brightness=np.zeros((8, 8)).at[3, 4].set(1.0),
            pixel_scale_mas=0.5,
        ),
        ("log_brightness", "pixel_scale_mas"),
    ),
    (
        FlaredDiskHG,
        dict(
            g=0.3,
            radius=5.0,
            fwhm=1.0,
            inc=40.0,
            pa=30.0,
            npix=32,
            pixel_scale_mas=0.5,
        ),
        ("g", "radius", "fwhm", "inc", "pa", "npix", "pixel_scale_mas"),
    ),
]


@pytest.mark.parametrize(
    "cls,kwargs,attrs", CASES, ids=[c[0].__name__ for c in CASES]
)
def test_component_builds_under_jit(cls, kwargs, attrs):
    shape = cls(**kwargs)
    fixed = {name: getattr(shape, name) for name in attrs}

    def build(flux):
        return cls(**fixed, flux=Nodes(flux, WAVEL))

    flux = np.array([1.0, 2.0, 3.0])
    eager = build(flux).model(U, V, 1.65e-6)
    jitted = jax.jit(lambda f: build(f).model(U, V, 1.65e-6))(flux)
    # Loose enough for float32 differences between eager and compiled code.
    assert np.allclose(
        eager, jitted, rtol=0.0, atol=1e-2 * np.abs(eager).max()
    )
