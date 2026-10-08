"""Pipeline IO: model specs without pickles, processed OIFITS, atomic JSON."""

import json

import numpy as onp
import pytest

pytest.importorskip("h5py")

from virgil import (  # noqa: E402
    BinaryModelAngular,
    GaussianDisk,
    OIData,
    PointSource,
    System,
    write_oifits,
)
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.pipeline import SCHEMA, _io  # noqa: E402
from tests.test_oifits import _tables  # noqa: E402

NUHOR = "data/NuHor_F480M.oifits"
TRUTH = BinaryModelAngular(sep=150.0, pa=40.0, flux=0.02)


def _same_model(a, b):
    import jax

    leaves_a, tree_a = jax.tree_util.tree_flatten(a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(b)
    assert tree_a == tree_b
    for x, y in zip(leaves_a, leaves_b):
        onp.testing.assert_array_equal(onp.asarray(x), onp.asarray(y))


@pytest.mark.parametrize(
    "model",
    [
        TRUTH,
        System(
            star=PointSource(),
            disk=GaussianDisk(sigma=3.0, flux=0.2),
            comp=PointSource(dra=40.0, ddec=-10.0, flux=0.01),
        ),
    ],
    ids=["binary", "system"],
)
def test_model_spec_round_trip(tmp_path, model):
    _io.save_model(tmp_path, model, {"flux": onp.asarray(0.5)}, ["flux"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["rebuildable"] and manifest["schema"] == SCHEMA
    rebuilt = _io.load_model(tmp_path)
    assert type(rebuilt) is type(model)
    _same_model(rebuilt, model)
    assert _io.load_model_values(tmp_path) == {"flux": 0.5}
    # No pickles anywhere: the arrays load with allow_pickle=False.
    with onp.load(tmp_path / "values.npz", allow_pickle=False) as f:
        assert f.files


def test_model_spec_refuses_non_virgil_classes(tmp_path):
    import equinox as eqx

    class Custom(eqx.Module):
        x: float = 1.0

    _io.save_model(tmp_path, Custom(), {"x": onp.asarray(2.0)}, ["x"])
    with pytest.raises(_io.ModelSpecError, match="model_values"):
        _io.load_model(tmp_path)
    assert _io.load_model_values(tmp_path) == {"x": 2.0}


def _round_trip(data, tmp_path):
    tables, reason = _io.oidata_tables(data)
    assert reason is None
    back = OIData(write_oifits(tables, tmp_path / "processed.oifits"))
    for name in ("u", "v", "vis", "d_vis", "phi", "d_phi"):
        onp.testing.assert_allclose(
            onp.asarray(getattr(back, name)),
            onp.asarray(getattr(data, name)),
            rtol=1e-6,
            atol=1e-12,
        )
    onp.testing.assert_allclose(
        onp.broadcast_to(back.wavel, back.u.shape),
        onp.broadcast_to(data.wavel, data.u.shape),
    )
    onp.testing.assert_allclose(
        float(model_loglike(TRUTH, back)),
        float(model_loglike(TRUTH, data)),
        rtol=1e-5,
    )
    return back


def test_processed_oifits_round_trip_closure_phases(tmp_path):
    back = _round_trip(OIData(NUHOR), tmp_path)
    assert back.cp_flag


def test_processed_oifits_round_trip_several_channels(tmp_path):
    data = OIData(
        write_oifits(_tables(waves=(4.6e-6, 5.0e-6)), tmp_path / "in.fits")
    )
    _round_trip(data, tmp_path)


def test_processed_oifits_round_trip_absolute_phases(tmp_path):
    rng = onp.random.default_rng(0)
    u, v = rng.uniform(-6, 6, (2, 12))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": [4.8e-6],
            "vis": onp.ones(12),
            "d_vis": onp.full(12, 0.01),
            "phi": rng.normal(0, 0.01, 12),
            "d_phi": onp.full(12, 0.01),
            "cp_flag": False,
        }
    )
    back = _round_trip(data, tmp_path)
    assert not back.cp_flag


def test_processed_oifits_reports_unsupported_layouts():
    data = OIData(NUHOR)
    flagged = data.with_error_floor()  # unchanged copy
    assert _io.oidata_tables(flagged)[1] is None
    import equinox as eqx

    no_stations = eqx.tree_at(
        lambda d: d.stations, data, None, is_leaf=lambda x: x is None
    )
    tables, reason = _io.oidata_tables(no_stations)
    assert tables is None and "station" in reason


def test_json_is_strict_and_atomic(tmp_path):
    path = tmp_path / "x.json"
    _io.write_json(
        path, {"a": float("nan"), "b": onp.arange(3), "c": onp.float32(1.5)}
    )
    assert json.loads(path.read_text()) == {
        "a": None,
        "b": [0, 1, 2],
        "c": 1.5,
    }
    with pytest.raises(TypeError):
        _io.write_json(path, {"bad": object()})
    # The failed write left the old file and no temporary file behind.
    assert json.loads(path.read_text())["c"] == 1.5
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_load_refuses_other_schemas(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"schema": "other-v9"}))
    with pytest.raises(ValueError, match="other-v9"):
        _io.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        _io.load(tmp_path / "missing")


def test_core_import_does_not_need_the_extra(monkeypatch):
    import subprocess
    import sys

    code = (
        "import sys, virgil; "
        "assert 'virgil.pipeline' not in sys.modules; "
        "assert 'h5py' not in sys.modules; "
        "virgil.pipeline; assert 'virgil.pipeline.binary' not in sys.modules"
    )
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)

    from virgil.pipeline import BinaryPipeline

    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[pipeline\]"):
        BinaryPipeline(OIData(NUHOR))


def test_data_fingerprint_covers_every_field():
    """The resume check must notice any change to the data, not just to the
    observables (the old fingerprint hashed twelve arrays and three flags)."""
    import copy

    import jax.numpy as jnp
    from virgil.coverage import vlti_oidata

    def replace(data, **fields):
        other = copy.copy(data)
        for name, value in fields.items():
            object.__setattr__(other, name, value)
        return other

    def build():
        return vlti_oidata(
            hour_angles_h=(-1.0, 0.0),
            wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 3),
        )

    data = build()
    assert data.stations is not None
    base = _io.data_fingerprint(data)
    assert _io.data_fingerprint(build()) == base

    changed = {
        "stations": replace(data, stations=data.stations + 1),
        "t_ref": replace(data, t_ref=(data.t_ref or 0.0) + 1.0),
        "dt": replace(data, dt=jnp.full(data.u.shape, 0.5)),
        "frame": replace(data, frame=data.frame + 1),
        "gains": data.with_gains(telescope=0.01),
        "phase_offsets": data.with_closure_offsets(baseline=0.01),
    }
    for name, other in changed.items():
        assert _io.data_fingerprint(other) != base, name
