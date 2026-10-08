"""Synthetic uv coverage and noise, for simulating data from a known truth.

Each function returns data whose values are zero, to be filled from a model
with [`OIData.with_model`][virgil.oidata.OIData.with_model]. They are
computed on the fly, so tests and tutorials need no large data files.

* :func:`ami_grid_record`: AMI as AMIGO represents it, with complex
  visibilities on a fine uv grid and a projection onto an orthonormal basis
  of modes weighted by the mask's splodges (an AMIGO mixed-DISCO record).
* :func:`nrm_oidata`: the same mask as a classical non-redundant masking
  observation, with squared visibilities at the splodge centres and closure
  phases (as in AMICAL).
* :func:`vlti_oidata`: a long-baseline observation, V² and closure phases
  along Earth-rotation tracks in several spectral channels (VLTI/MATISSE
  with the four UTs by default).
"""

import itertools

import numpy as onp

from .oidata import OIData, cp_indices

# Hole centres of the JWST/NIRISS 7-hole AMI mask, projected onto the primary
# mirror, in metres (published mask geometry, e.g. Sivaramakrishnan et al.
# 2023, arXiv:2210.17434).
NIRISS_AMI_HOLES = onp.array(
    [
        [0.0, -2.64],
        [-2.28631, 0.0],
        [2.28631, -1.32],
        [-2.28631, 1.32],
        [-1.14315, 1.98],
        [2.28631, 1.32],
        [1.14315, 1.98],
    ]
)
NIRISS_AMI_HOLE_DIAMETER = 0.82  # metres, across the flats of the hexagons


def _rotate(u, v, rotation_deg):
    c, s = (
        onp.cos(onp.radians(rotation_deg)),
        onp.sin(onp.radians(rotation_deg)),
    )
    return c * u + s * v, -s * u + c * v


def _baselines(holes):
    """Baseline vectors (j minus i) for all hole pairs ``i < j``."""
    pairs = list(itertools.combinations(range(len(holes)), 2))
    return pairs, onp.array([holes[j] - holes[i] for i, j in pairs])


def _overlap(distance, diameter):
    """Area of overlap of two discs of ``diameter`` whose centres are apart."""
    r = 0.5 * diameter
    d = onp.minimum(onp.abs(distance), diameter)
    return 2 * r**2 * onp.arccos(d / (2 * r)) - 0.5 * d * onp.sqrt(
        4 * r**2 - d**2
    )


def mask_transfer(
    u, v, holes=NIRISS_AMI_HOLES, diameter=NIRISS_AMI_HOLE_DIAMETER
):
    """The modulus of an aperture mask's optical transfer function.

    The autocorrelation of the pupil, for circular holes of the given
    diameter, normalized to one at zero baseline: each hole pair gives a
    "splodge" around its baseline and the opposite one, and all holes
    together the central splodge.

    Parameters
    ----------
    u, v : array-like
        Baselines in metres, in the frame of the mask.
    holes : array-like, shape (n_holes, 2), optional
        Hole centres in metres (default: the NIRISS AMI mask).
    diameter : float, optional
        Hole diameter in metres (NIRISS: hexagons about 0.82 m across).
    """
    u, v = onp.asarray(u, float), onp.asarray(v, float)
    _, baselines = _baselines(onp.asarray(holes, float))
    mtf = len(holes) * _overlap(onp.hypot(u, v), diameter)
    for bu, bv in baselines:
        mtf = mtf + _overlap(onp.hypot(u - bu, v - bv), diameter)
        mtf = mtf + _overlap(onp.hypot(u + bu, v + bv), diameter)
    return mtf / (len(holes) * _overlap(0.0, diameter))


def ami_grid_record(
    wavelength_m=4.3e-6,
    pitch_m=0.3,
    rotation_deg=0.0,
    sigma=1e-4,
    keep=0.99,
    holes=NIRISS_AMI_HOLES,
    diameter=NIRISS_AMI_HOLE_DIAMETER,
):
    """A simulated AMI record in AMIGO's form: a uv grid and a mode basis.

    This follows the construction of AMIGO's latent visibility basis
    (Desdoigts et al. 2025, arXiv:2510.09806) in simplified form. The
    complex visibilities live on the half-plane of a square uv grid (as on
    the detector's Fourier grid, rotated on the sky by ``rotation_deg``, the
    parallactic angle). Their information is weighted by the mask's transfer
    function: a cell where the splodges have transfer ``m`` measures the
    log-amplitude and phase of the visibility with error ``sigma / m``. The
    responses to the source's flux (a constant log-amplitude) and position
    (phases linear in u and v) are projected out, and an SVD of what remains
    gives orthonormal modes, kept in order of precision until a fraction
    ``keep`` of the total precision is retained. The modes are the DISCO
    coefficients of the record, with independent errors.

    The coefficients are zero: fill them with
    [`OIData.with_model`][virgil.oidata.OIData.with_model].

    Parameters
    ----------
    wavelength_m : float, optional
        Wavelength in metres (default 4.3 µm, like F430M).
    pitch_m : float, optional
        Grid spacing in metres. AMIGO products are sampled more finely (ν
        Hor: 0.2165 m); a coarser grid keeps simulations small.
    rotation_deg : float, optional
        Position angle, North to East, of the grid's "up" axis.
    sigma : float, optional
        Error of a cell's log-amplitude and phase where the transfer is one
        (ν Hor's DISCO errors are ~5e-5 to 3e-4).
    keep : float, optional
        Fraction of the precision kept (AMIGO uses 0.99).
    holes, diameter : optional
        The mask (default: NIRISS AMI).

    Returns
    -------
    dict
        A record for [`OIData`][virgil.oidata.OIData].
    """
    holes = onp.asarray(holes, float)
    _, baselines = _baselines(holes)
    reach = onp.max(onp.hypot(*baselines.T)) + diameter
    n = int(onp.ceil(reach / pitch_m))
    col, row = onp.meshgrid(onp.arange(-n, n + 1), onp.arange(-n, n + 1))
    col, row = col.ravel(), row.ravel()
    half = (row > 0) | ((row == 0) & (col > 0))
    grid_u, grid_v = pitch_m * col[half], pitch_m * row[half]
    weight = mask_transfer(grid_u, grid_v, holes, diameter) / sigma
    inside = weight > 0.0
    grid_u, grid_v, weight = grid_u[inside], grid_v[inside], weight[inside]
    npts = grid_u.size
    # The record stores -u, -v (see amigo.mixed_disco_fields).
    u, v = _rotate(grid_u, grid_v, rotation_deg)
    u, v = -u, -v
    # Whitened measurement of (log|V|, arg V) at each cell, and the
    # responses to a change of flux and of position, which the data cannot
    # be trusted to constrain.
    whiten = onp.concatenate([weight, weight])
    nuisance = onp.zeros((2 * npts, 3))
    nuisance[:npts, 0] = weight
    nuisance[npts:, 1], nuisance[npts:, 2] = weight * u, weight * v
    q = onp.linalg.qr(nuisance)[0]
    projected = onp.diag(whiten) - q @ (q.T @ onp.diag(whiten))
    _, precision, modes = onp.linalg.svd(projected, full_matrices=False)
    fraction = onp.cumsum(precision**2) / onp.sum(precision**2)
    nmodes = int(onp.searchsorted(fraction, keep) + 1)
    modes, precision = modes[:nmodes], precision[:nmodes]
    return {
        "u": u,
        "v": v,
        "wavelength_m": onp.asarray(wavelength_m, dtype=float),
        "disco_coefficients": onp.zeros(nmodes),
        "disco_sigma": 1.0 / precision,
        "disco_logamp_model_operator": modes[:, :npts],
        "disco_phase_model_operator": modes[:, npts:],
    }


def nrm_oidata(
    wavelength_m=4.3e-6,
    rotation_deg=0.0,
    sigma_v2=0.01,
    sigma_cp_deg=0.5,
    holes=NIRISS_AMI_HOLES,
):
    """A simulated non-redundant masking observation: V² and closure phases.

    The classical reduction of aperture masking (as in AMICAL), with one
    squared visibility at the centre of each splodge, that is at each
    baseline between two holes, and the closure phases of every triangle of
    holes. For the 7-hole NIRISS mask that is 21 V² and 35 closure phases.

    Parameters
    ----------
    wavelength_m : float, optional
        Wavelength in metres.
    rotation_deg : float, optional
        Position angle, North to East, of the mask's "up" axis on the sky.
    sigma_v2 : float, optional
        Error of each squared visibility.
    sigma_cp_deg : float, optional
        Error of each closure phase, in degrees (AMI on sky: ~0.1–1°).
    holes : array-like, optional
        Hole centres in metres (default: NIRISS AMI).

    Returns
    -------
    OIData
        Data with zero values, for
        [`with_model`][virgil.oidata.OIData.with_model].
    """
    holes = onp.asarray(holes, float)
    pairs, baselines = _baselines(holes)
    u, v = _rotate(baselines[:, 0], baselines[:, 1], rotation_deg)
    triangles = list(itertools.combinations(range(len(holes)), 3))
    i1, i2, i3 = cp_indices(pairs, triangles)
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": float(wavelength_m),
            "vis": onp.ones(len(pairs)),
            "d_vis": onp.full(len(pairs), float(sigma_v2)),
            "phi": onp.zeros(len(triangles)),
            "d_phi": onp.full(len(triangles), float(sigma_cp_deg)),
            "phi_unit": "deg",
            "i_cps1": i1,
            "i_cps2": i2,
            "i_cps3": i3,
        }
    )


# The four VLTI Unit Telescopes in local (East, North) ground coordinates,
# in metres (ESO's station positions, already converted from the platform's
# (P, Q) frame). Baselines run from 46.6 m (UT2-UT3) to 130.2 m (UT1-UT4).
VLTI_UTS = onp.array(
    [
        [-9.925, -20.335],
        [14.887, 30.502],
        [44.915, 66.183],
        [103.306, 43.999],
    ]
)
PARANAL_LATITUDE_DEG = -24.6276


def vlti_oidata(
    stations=VLTI_UTS,
    declination_deg=-50.0,
    hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
    wavelengths_m=onp.linspace(3.0e-6, 4.0e-6, 11),
    sigma_v2=0.03,
    sigma_cp_deg=1.0,
    latitude_deg=PARANAL_LATITUDE_DEG,
    nights_mjd=None,
):
    """A simulated long-baseline observation: V² and closure phases.

    Every pair of telescopes is observed at each hour angle, as the Earth's
    rotation carries the baselines along tracks in the uv plane, in every
    wavelength channel; closure phases come from every triangle at each
    hour angle. The defaults resemble VLTI/MATISSE in the L band at low
    spectral resolution with the four UTs. MATISSE's LOW mode (R ~ 30)
    smears structure further than about R λ / B from the field centre
    (~170 mas here), which virgil does not model.

    Parameters
    ----------
    stations : array-like, shape (n_telescopes, 2), optional
        Telescope positions (East, North) in metres (default: the UTs).
    declination_deg : float, optional
        The target's declination.
    hour_angles_h : sequence of float, optional
        Hour angles of the snapshots, in hours.
    wavelengths_m : array-like, optional
        The spectral channels, in metres.
    sigma_v2 : float, optional
        Error of each squared visibility.
    sigma_cp_deg : float, optional
        Error of each closure phase, in degrees.
    latitude_deg : float, optional
        The array's latitude (default: Paranal).
    nights_mjd : sequence of float, optional
        Repeat the snapshots on each of these nights, given as the MJD of
        the target's transit; each snapshot is then a frame at
        ``night + hour_angle / 24`` (for orbits and other scenes that
        change with time). By default the data have no times.

    Returns
    -------
    OIData
        Data with zero values, for
        [`with_model`][virgil.oidata.OIData.with_model]. Each snapshot is a
        frame, and each baseline carries its station pair (numbered from
        1), so [`with_gains`][virgil.oidata.OIData.with_gains] applies.
    """
    stations = onp.asarray(stations, float)
    n = len(stations)
    pairs = list(itertools.combinations(range(n), 2))
    triangles = list(itertools.combinations(range(n), 3))
    lat, dec = onp.radians(latitude_deg), onp.radians(declination_deg)
    us, vs, mjd, all_pairs, all_triangles = [], [], [], [], []
    frame, station_pairs = [], []
    nights = [None] if nights_mjd is None else list(nights_mjd)
    snapshots = [(night, hour) for night in nights for hour in hour_angles_h]
    for epoch, (night, hour) in enumerate(snapshots):
        h = onp.radians(15.0 * hour)
        for i, j in pairs:
            east, north = stations[j] - stations[i]
            # Local (East, North, up=0) to equatorial, then to (u, v)
            # (Thompson, Moran & Swenson, ch. 4).
            x, y, z = -onp.sin(lat) * north, east, onp.cos(lat) * north
            us.append(onp.sin(h) * x + onp.cos(h) * y)
            vs.append(
                -onp.sin(dec) * onp.cos(h) * x
                + onp.sin(dec) * onp.sin(h) * y
                + onp.cos(dec) * z
            )
            all_pairs.append((n * epoch + i, n * epoch + j))
            frame.append(epoch)
            station_pairs.append((i + 1, j + 1))
            mjd.append(None if night is None else night + hour / 24.0)
        all_triangles += [
            tuple(n * epoch + t for t in tri) for tri in triangles
        ]
    i1, i2, i3 = cp_indices(all_pairs, all_triangles)
    wavelengths = onp.asarray(wavelengths_m, float)
    n_bl, n_cp, n_wl = len(us), len(all_triangles), wavelengths.size
    times = {} if nights_mjd is None else {"mjd": onp.asarray(mjd, float)}
    return OIData(
        {
            **times,
            "frame": onp.asarray(frame),
            "stations": onp.asarray(station_pairs),
            "u": onp.asarray(us),
            "v": onp.asarray(vs),
            "wavel": wavelengths,
            "vis": onp.ones((n_bl, n_wl)),
            "d_vis": onp.full((n_bl, n_wl), float(sigma_v2)),
            "phi": onp.zeros((n_cp, n_wl)),
            "d_phi": onp.full((n_cp, n_wl), float(sigma_cp_deg)),
            "phi_unit": "deg",
            "i_cps1": i1,
            "i_cps2": i2,
            "i_cps3": i3,
        }
    )
