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
from virgil.oifits import _oidata_tables  # noqa: E402
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

    def refused(module, cls_name, match):
        spec = {"$module": {"module": module, "class": cls_name, "fields": {}}}
        with pytest.raises(_io.ModelSpecError, match=match):
            _io.model_from_spec(spec, {})

    # Modules outside the allow-list are refused before anything is imported.
    for module, cls_name in (
        ("os", "system"),
        ("subprocess", "Popen"),
        ("virgil.oidata", "OIData"),
        ("virgil.models", "_Private"),
    ):
        refused(module, cls_name, "allowed virgil model modules")
    # Allowed module, but the name is not a model class.
    refused("virgil.models", "Any", "public virgil model class")
    refused("virgil.models", "np", "public virgil model class")


def _round_trip(data, tmp_path):
    back = OIData(write_oifits(data, tmp_path / "processed.oifits"))
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


def test_processed_oifits_round_trip_nrm_oidata(tmp_path):
    from virgil.coverage import nrm_oidata

    back = _round_trip(nrm_oidata(), tmp_path)
    assert back.cp_flag and back.phi.size == 35


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


def test_processed_oifits_reports_unsupported_layouts(tmp_path):
    assert _oidata_tables(OIData(NUHOR))[1] is None
    amplitudes = OIData(
        {
            "u": onp.array([3.0, -2.0, 1.0]),
            "v": onp.array([1.0, 2.0, -3.0]),
            "wavel": [4.8e-6],
            "vis": onp.ones(3),
            "d_vis": onp.full(3, 0.01),
            "v2_flag": False,
            "cp_flag": False,
        }
    )
    reason = "visibility amplitudes rather than V²"
    assert _oidata_tables(amplitudes) == (None, reason)
    with pytest.raises(ValueError, match=reason):
        write_oifits(amplitudes, tmp_path / "x.oifits")


def test_processed_oifits_refuses_mischained_triangles():
    """OI_T3 stores the triangle ABC; virgil's closure phase is
    φ[i_cps1] + φ[i_cps2] - φ[i_cps3] over AB, BC and AC, so legs that do
    not chain that way must not be written as if they did."""
    import equinox as eqx

    data = OIData(NUHOR)
    assert _oidata_tables(data)[1] is None
    irregular = (None, "irregular closure-phase layout")
    # Legs 1 and 2 swapped: BC then AB.
    swapped = eqx.tree_at(
        lambda d: (d.i_cps1, d.i_cps2), data, (data.i_cps2, data.i_cps1)
    )
    assert _oidata_tables(swapped) == irregular
    # The third leg is not the baseline AC.
    other = eqx.tree_at(lambda d: d.i_cps3, data, data.i_cps1)
    assert _oidata_tables(other) == irregular
    # Without stations the same legs would give a baseline from a station to
    # itself, which is no valid OI_ARRAY pair.
    bare = eqx.tree_at(
        lambda d: d.stations, data, None, is_leaf=lambda x: x is None
    )
    assert _oidata_tables(bare)[1] is None
    assert (
        _oidata_tables(eqx.tree_at(lambda d: d.i_cps3, bare, bare.i_cps1))
        == irregular
    )


def test_result_data_reports_why_the_processed_oifits_is_missing(tmp_path):
    """The reason is in the load stage's report, not in run.json."""
    (tmp_path / "run.json").write_text(
        json.dumps({"schema": SCHEMA, "stages": {"load": {}}})
    )
    report = tmp_path / "stages" / "load" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"processed_oifits_skipped": "extra observables"})
    )
    with pytest.raises(FileNotFoundError, match="extra observables"):
        _io.load(tmp_path).data()


def test_spec_digest_covers_dtype_and_shape():
    spec = {"x": 1}
    digests = [
        _io.spec_digest(spec, {"leaf_0": array})
        for array in (
            onp.zeros(4, dtype=onp.float32),
            onp.zeros(4, dtype=onp.int32),  # same bytes, other dtype
            onp.zeros((2, 2), dtype=onp.float32),  # same bytes, other shape
            onp.zeros(4, dtype=onp.float32),
        )
    ]
    assert len(set(digests)) == 3 and digests[0] == digests[3]


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

    import jax.numpy as np
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
        "dt": replace(data, dt=np.full(data.u.shape, 0.5)),
        "frame": replace(data, frame=data.frame + 1),
        "gains": data.with_gains(telescope=0.01),
        "phase_offsets": data.with_closure_offsets(baseline=0.01),
    }
    for name, other in changed.items():
        assert _io.data_fingerprint(other) != base, name
