"""Position-angle round trips through OIFITS files (design
orbit_scene_joint_fitting.md §5.3.1).

Closure-phase conventions (baseline order in OI_T3, the sign of u and v, the
choice of reference star) each flip a binary by 180°. A binary with unequal
fluxes is written to OIFITS in an instrument's layout, read back, and found
by a grid search over the whole field: it must come back within 1° of its
true position angle, not 180° away, and fainter than the primary.
"""

import numpy as onp
import pytest

from virgil.coverage import NIRISS_AMI_HOLES, VLTI_UTS
from virgil.grid_fit import best_grid_point, likelihood_grid
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData, cp_indices
from virgil.oifits import write_oifits

TRUTH = BinaryModelCartesian(dra=-3.0, ddec=5.0, flux=0.2)  # PA ≈ 329°
TRUE_PA = onp.degrees(onp.arctan2(-3.0, 5.0)) % 360


def _tables(stations, pairs, triangles, waves, hour_angles, insname):
    """OIFITS tables sampling TRUTH, one frame per hour angle.

    ``pairs`` and ``triangles`` are the instrument's STA_INDEX values, in its
    own order; UCOORD/VCOORD run from the first station to the second.
    """
    index = {s: k for k, s in enumerate(sorted(set(pairs.ravel())))}
    vis2, t3 = [], []
    for k, angle in enumerate(onp.deg2rad(hour_angles)):
        rot = onp.array(
            [
                [onp.cos(angle), -onp.sin(angle)],
                [onp.sin(angle), onp.cos(angle)],
            ]
        )
        xy = stations @ rot.T

        def baseline(a, b):
            return xy[index[b]] - xy[index[a]]

        uv = onp.array([baseline(a, b) for a, b in pairs])
        cvis = onp.asarray(
            TRUTH.model(uv[:, :1], uv[:, 1:], onp.asarray(waves)[None, :])
        )
        i1, i2, i3 = cp_indices(pairs, triangles)
        uv1 = onp.array([baseline(a, b) for a, b, _ in triangles])
        uv2 = onp.array([baseline(b, c) for _, b, c in triangles])
        mjd = 60000.0 + k / 24
        vis2.append((uv, onp.abs(cvis) ** 2, mjd))
        t3.append((uv1, uv2, onp.angle(cvis[i1] * cvis[i2] / cvis[i3]), mjd))

    def stack(rows, column):
        return onp.concatenate([r[column] for r in rows])

    return {
        "info": {"TARGET": "BIN", "INSTRUME": insname, "INSNAME": insname},
        "OI_WAVELENGTH": {"EFF_WAVE": onp.asarray(waves), "EFF_BAND": 1e-8},
        "OI_VIS2": {
            "VIS2DATA": stack(vis2, 1),
            "VIS2ERR": onp.full(stack(vis2, 1).shape, 0.01),
            "UCOORD": stack(vis2, 0)[:, 0],
            "VCOORD": stack(vis2, 0)[:, 1],
            "STA_INDEX": onp.concatenate([pairs] * len(hour_angles)),
            "MJD": onp.repeat([r[2] for r in vis2], len(pairs)),
        },
        "OI_T3": {
            "T3PHI": onp.rad2deg(stack(t3, 2)),
            "T3PHIERR": onp.full(stack(t3, 2).shape, 0.5),
            "U1COORD": stack(t3, 0)[:, 0],
            "V1COORD": stack(t3, 0)[:, 1],
            "U2COORD": stack(t3, 1)[:, 0],
            "V2COORD": stack(t3, 1)[:, 1],
            "STA_INDEX": onp.concatenate([triangles] * len(hour_angles)),
            "MJD": onp.repeat([r[3] for r in t3], len(triangles)),
        },
    }


def _gravity_layout():
    # A real GRAVITY product's station numbers and row order (UT1–4 as
    # STA_INDEX 1, 18, 23, 28; baselines from the highest index).
    pairs = onp.array(
        [[28, 23], [28, 18], [28, 1], [23, 18], [23, 1], [18, 1]]
    )
    triangles = onp.array(
        [[28, 23, 18], [28, 23, 1], [28, 18, 1], [23, 18, 1]]
    )
    stations = VLTI_UTS[[0, 1, 2, 3]]  # sorted STA_INDEX 1, 18, 23, 28
    waves = onp.linspace(2.0e-6, 2.4e-6, 5)
    return stations, pairs, triangles, waves, (-40.0, 0.0, 40.0), "GRAVITY_SC"


def _mask_layout():
    # A 7-hole mask, baselines and triangles in increasing hole order.
    holes = onp.arange(1, 8)
    pairs = onp.array([[a, b] for a in holes for b in holes if a < b])
    triangles = onp.array(
        [[a, b, c] for a in holes for b in holes for c in holes if a < b < c]
    )
    return (
        NIRISS_AMI_HOLES * 1.0,
        pairs,
        triangles,
        [4.8e-6],
        (0.0, 30.0),
        "MASK",
    )


def _found(tables, path):
    """Grid-search position angle and flux of the binary in a written file."""
    data = OIData(write_oifits(tables, path))
    axis = onp.linspace(-8.0, 8.0, 65)
    samples = {
        "dra": axis,
        "ddec": axis,
        "flux": onp.array([0.05, 0.2, 0.6, 0.9]),
    }
    best = best_grid_point(likelihood_grid(data, TRUTH, samples), samples)
    return onp.degrees(onp.arctan2(best["dra"], best["ddec"])) % 360, best[
        "flux"
    ]


def _miss(pa, want):
    return abs((pa - want + 180) % 360 - 180)


@pytest.mark.parametrize("layout", [_gravity_layout, _mask_layout])
def test_a_binary_comes_back_at_its_position_angle(tmp_path, layout):
    stations, pairs, triangles, waves, hour_angles, insname = layout()
    if insname == "MASK":
        # Scale the mask's baselines so the companion is resolved.
        stations = stations * 20.0
    tables = _tables(stations, pairs, triangles, waves, hour_angles, insname)
    pa, flux = _found(tables, tmp_path / "binary.oifits")
    assert _miss(pa, TRUE_PA) < 1.0
    # Fluxes are searched below 1 only: a binary with flux f at r has the
    # same V² and closure phases as one with flux 1/f at -r (the other star
    # as the primary), so f < 1 is the convention, and it is the PA that
    # shows a flip.
    assert flux == pytest.approx(0.2)
    # The test can see a flip: negated closure phases put the companion on
    # the other side.
    tables["OI_T3"]["T3PHI"] = -tables["OI_T3"]["T3PHI"]
    pa, _ = _found(tables, tmp_path / "flipped.oifits")
    assert _miss(pa, TRUE_PA + 180.0) < 1.0
