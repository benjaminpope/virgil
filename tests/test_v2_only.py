"""Visibility-only data: V² (or amplitudes) with no phases at all.

Some datasets have no usable phases. virgil reads and fits them with an
empty phase block, so every likelihood, fit, uncertainty and grid uses the
visibilities alone.
"""

import warnings

import jax
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.fitting import fit
from virgil.grid_fit import likelihood_grid
from virgil.imaging import dirty_image
from virgil.inference import laplace_cov
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian, UniformDisk
from virgil.oidata import OIData
from virgil.oifits import read_oifits, write_oifits

STATIONS = onp.array([[0.0, 0.0], [32.0, 2.0], [14.0, 26.0], [-11.0, 18.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
WAVES = onp.array([1.6e-6, 1.8e-6, 2.0e-6])
DISK = UniformDisk(2.5)


def _baselines():
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    return delta[:, 0], delta[:, 1]


def _v2_tables(model=DISK, noise=None):
    u, v = _baselines()
    cvis = onp.asarray(model.model(u[:, None], v[:, None], WAVES[None, :]))
    v2 = onp.abs(cvis) ** 2
    if noise is not None:
        v2 = v2 + noise
    return {
        "info": {"TARGET": "STAR", "INSTRUME": "TEST", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": WAVES, "EFF_BAND": 0.1e-6},
        "OI_VIS2": {
            "VIS2DATA": v2,
            "VIS2ERR": onp.full(v2.shape, 2e-3),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": PAIRS,
        },
    }


def _v2_dict(model=DISK, **extra):
    u, v = _baselines()
    cvis = onp.asarray(model.model(u[:, None], v[:, None], WAVES[None, :]))
    return {
        "u": u,
        "v": v,
        "wavel": WAVES,
        "vis": onp.abs(cvis) ** 2,
        "d_vis": onp.full(cvis.shape, 2e-3),
        **extra,
    }


def test_a_file_without_phases_is_read(tmp_path):
    path = write_oifits(_v2_tables(), tmp_path / "v2.oifits")
    record = read_oifits(path)
    assert record["phi"].size == 0 and not record["cp_flag"]
    data = OIData(path)
    assert not data.has_phases
    observed, errors = data.flatten_data()
    assert observed.size == errors.size == len(PAIRS) * WAVES.size
    assert data.model(DISK).shape == observed.shape
    assert data.n_independent == observed.size
    assert onp.allclose(whitened_residuals(DISK, data), 0.0, atol=1e-4)


@pytest.mark.parametrize("phases", ["absent", "empty"])
def test_a_dictionary_without_phases(phases):
    extra = {} if phases == "absent" else {"phi": [], "d_phi": []}
    data = OIData(_v2_dict(**extra))
    assert not data.has_phases
    assert data.model(DISK).shape == data.flatten_data()[0].shape
    assert onp.isfinite(model_loglike(DISK, data))


def test_data_with_phases_still_have_them():
    u, v = _baselines()
    cvis = onp.asarray(DISK.model(u, v, 2.0e-6))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": 2.0e-6,
            "vis": onp.abs(cvis) ** 2,
            "d_vis": onp.full(u.size, 2e-3),
            "phi": onp.angle(cvis),
            "d_phi": onp.full(u.size, 1e-2),
        }
    )
    assert data.has_phases and data.phi.size == u.size


def test_a_fit_recovers_a_diameter_from_v2_alone():
    """A uniform-disk diameter, with its Laplace uncertainty, from noisy V²
    alone; and the fit's chi-squared counts only the visibilities."""
    rng = onp.random.default_rng(3)
    with jax.enable_x64(True):
        record = _v2_dict()
        record["vis"] = record["vis"] + 2e-3 * rng.standard_normal(
            record["vis"].shape
        )
        data = OIData(record)
        result = fit(UniformDisk(2.0), {"diam": dist.Uniform(0.1, 10.0)}, data)
        cov = laplace_cov(
            onp.array([float(result.values["diam"])]),
            ["diam"],
            UniformDisk,
            data,
        )
        sigma = float(onp.sqrt(onp.asarray(cov)[0, 0]))
    assert result.info["converged"]
    assert abs(float(result.values["diam"]) - 2.5) < 5 * sigma
    assert 0 < sigma < 0.1
    assert result.info["ndata"][0] == len(PAIRS) * WAVES.size


def test_all_closure_phases_flagged_is_the_same_as_no_phases(tmp_path):
    """Flagging every closure phase gives exactly the V²-only likelihood."""
    from virgil.oidata import cp_indices

    model = BinaryModelCartesian(4.0, -3.0, 0.1)
    with_t3 = _v2_dict(model)
    u, v = _baselines()
    tris = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
    i1, i2, i3 = cp_indices(PAIRS, tris)
    with_t3.update(
        phi=onp.zeros((len(tris), WAVES.size)),
        d_phi=onp.full((len(tris), WAVES.size), 1e-2),
        phi_flag=onp.ones((len(tris), WAVES.size), bool),
        i_cps1=i1,
        i_cps2=i2,
        i_cps3=i3,
    )
    with pytest.warns(UserWarning, match="Every closure phase is flagged"):
        flagged = OIData(with_t3)
    plain = OIData(_v2_dict(model))
    probe = BinaryModelCartesian(5.0, -2.0, 0.08)
    assert onp.isclose(
        model_loglike(probe, flagged), model_loglike(probe, plain)
    )


def test_a_grid_search_runs_on_v2_alone():
    """V² alone cannot tell a companion from its mirror image, but a
    likelihood grid still runs and peaks at one of the two."""
    model = BinaryModelCartesian(4.0, -3.0, 0.1)
    data = OIData(_v2_dict(model))
    axis = onp.linspace(-6, 6, 13)
    grid = likelihood_grid(
        BinaryModelCartesian, data, {"dra": axis, "ddec": axis, "flux": [0.1]}
    )[..., 0]
    grid = onp.asarray(grid)
    assert onp.all(onp.isfinite(grid))
    i, j = onp.unravel_index(onp.argmax(grid), grid.shape)
    peak = axis[[i, j]]
    # the companion or its point reflection, and no other sign pattern
    assert onp.allclose(peak, [4.0, -3.0]) or onp.allclose(peak, [-4.0, 3.0])


def test_epochs_split_without_phases():
    record = _v2_dict()
    n = record["vis"].size
    record["mjd"] = onp.repeat([60000.0, 60010.0], [n // 2, n - n // 2])
    data = OIData(record)
    parts = data.split_by_epoch()
    assert len(parts) == 2 and all(not p.has_phases for p in parts)
    assert sum(p.flatten_data()[0].size for p in parts) == n


def test_a_dirty_image_needs_phases():
    with pytest.raises(ValueError, match="visibility-only data"):
        dirty_image(OIData(_v2_dict()), 16, 1.0)


def test_files_with_and_without_phases_are_not_merged(tmp_path):
    from virgil.oidata import cp_indices  # noqa: F401

    v2_only = write_oifits(_v2_tables(), tmp_path / "v2.oifits")
    tables = _v2_tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.ones((len(PAIRS), WAVES.size)),
        "VISAMPERR": onp.full((len(PAIRS), WAVES.size), 0.01),
        "VISPHI": onp.zeros((len(PAIRS), WAVES.size)),
        "VISPHIERR": onp.full((len(PAIRS), WAVES.size), 1.0),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    with_phases = write_oifits(tables, tmp_path / "vis.oifits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="Some files have phases"):
            read_oifits([v2_only, with_phases])


def test_phases_without_their_errors_are_refused():
    with pytest.raises(ValueError, match="d_phi"):
        OIData(_v2_dict(phi=onp.zeros((len(PAIRS), WAVES.size))))
    with pytest.raises(ValueError, match="d_phi"):
        OIData(_v2_dict(d_phi=onp.ones((len(PAIRS), WAVES.size))))


def test_closure_phases_alone_all_flagged_leave_no_data():
    """A closure-phase-only dataset (no visibilities) with every closure
    phase flagged has no data at all: refused, not fitted to the prior."""
    from virgil.oidata import cp_indices

    tris = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
    i1, i2, i3 = cp_indices(PAIRS, tris)
    u, v = _baselines()
    record = {
        "u": u,
        "v": v,
        "wavel": 2.0e-6,
        "vis": onp.zeros(0),
        "d_vis": onp.zeros(0),
        "phi": onp.zeros(len(tris)),
        "d_phi": onp.full(len(tris), 0.01),
        "phi_flag": onp.ones(len(tris), bool),
        "i_cps1": i1,
        "i_cps2": i2,
        "i_cps3": i3,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No unflagged data"):
            OIData(record)


def test_an_operator_that_projects_everything_away_is_refused():
    """Phases only, through an operator with no non-zero rows: the
    projection drops every observable, which must be refused too."""
    u, v = _baselines()
    record = {
        "u": u,
        "v": v,
        "wavel": 2.0e-6,
        "vis": onp.zeros(0),
        "d_vis": onp.zeros(0),
        "phi": onp.zeros(u.size),
        "d_phi": onp.full(u.size, 0.01),
        "phi_mat": onp.zeros((2, u.size)),
    }
    with pytest.raises(ValueError, match="No data left"):
        OIData(record)
