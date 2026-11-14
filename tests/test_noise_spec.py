"""Tests for declarative instrument noise specifications (virgil.noise_spec)."""

from dataclasses import dataclass
import json

import numpy as onp
import pytest

from tests._shared import PAIRS, STATIONS, TRIANGLES
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.noise_spec import (
    SCHEMA,
    apply_noise_spec,
    instrument_match,
    load_noise_spec,
    validate_noise_spec,
)
from virgil.oidata import OIData, cp_indices
from virgil.oifits import build_hdulist

SCENE = BinaryModelCartesian(dra=2.5, ddec=-1.5, flux=0.25)
WAVES = onp.linspace(2.150e-6, 2.180e-6, 8)
LINE_WINDOW = [(2.162e-6, 2.170e-6)]


def _base_spec(**overrides):
    spec = {
        "schema": SCHEMA,
        "instrument": "PIONIER",
        "white": {"vis": 0.32, "phi": 0.65},
        "gains": {"telescope": 0.0055, "baseline": 0.0045, "chromatic": None},
        "closure_offsets": {"triangle": 0.0036, "baseline": None},
        "common_mode": {"visphi": None},
        "provenance": {
            "run": "pair_test:11_pairs",
            "date": "2026-10-04",
            "bound": "lower",
        },
    }
    spec.update(overrides)
    return spec


def _multi_channel_oidata(n_frames=2, extras=()):
    """Physical 4-telescope multi-channel dataset across ``n_frames``."""
    rng = onp.random.default_rng(7)
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    angles = onp.deg2rad(15.0 * onp.arange(n_frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )
    cvis = onp.asarray(SCENE.model(u[:, None], v[:, None], WAVES[None, :]))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n_b = len(PAIRS)
    s1 = onp.concatenate([i1 + f * n_b for f in range(n_frames)])
    s2 = onp.concatenate([i2 + f * n_b for f in range(n_frames)])
    s3 = onp.concatenate([i3 + f * n_b for f in range(n_frames)])
    cp = onp.angle(cvis[s1] * cvis[s2] / cvis[s3])
    t3amp = onp.abs(cvis[s1] * cvis[s2] * cvis[s3])
    d_v2 = onp.full(cvis.shape, 0.01)
    d_cp_deg = onp.full(cp.shape, 0.5)
    v2_obs = onp.abs(cvis) ** 2 + rng.normal(0.0, 0.004, cvis.shape)
    cp_deg = onp.rad2deg(cp) + rng.normal(0.0, 0.25, cp.shape)
    row_mjd = onp.repeat(60000.0 + onp.arange(n_frames) / 24.0, n_b)
    t3_mjd = onp.repeat(60000.0 + onp.arange(n_frames) / 24.0, len(TRIANGLES))
    tables = {
        "info": {"TARGET": "SYNTH", "INSTRUME": "PIONIER", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": WAVES, "EFF_BAND": 1e-9},
        "OI_VIS2": {
            "VIS2DATA": v2_obs,
            "VIS2ERR": d_v2,
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": onp.tile(PAIRS, (n_frames, 1)),
            "MJD": row_mjd,
        },
        "OI_VIS": {
            "VISAMP": onp.abs(cvis),
            "VISAMPERR": onp.full(cvis.shape, 0.005),
            "VISPHI": onp.rad2deg(onp.angle(cvis))
            + rng.normal(0.0, 0.15, cvis.shape),
            "VISPHIERR": onp.full(cvis.shape, 0.2),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": onp.tile(PAIRS, (n_frames, 1)),
            "MJD": row_mjd,
            "AMPTYP": "absolute",
            "PHITYP": "differential",
        },
        "OI_T3": {
            "T3AMP": t3amp + rng.normal(0.0, 0.002, t3amp.shape),
            "T3AMPERR": onp.full(t3amp.shape, 0.005),
            "T3PHI": cp_deg,
            "T3PHIERR": d_cp_deg,
            "U1COORD": u[s1],
            "V1COORD": v[s1],
            "U2COORD": u[s2],
            "V2COORD": v[s2],
            "STA_INDEX": onp.tile(TRIANGLES, (n_frames, 1)),
            "MJD": t3_mjd,
        },
    }
    return OIData(build_hdulist(tables), extras=extras)


def test_load_noise_spec_resolves_dict_json_and_protocol(tmp_path):
    raw = _base_spec()
    from_dict = load_noise_spec(raw)
    assert from_dict["schema"] == SCHEMA
    assert from_dict["gains"]["chromatic"] is None

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    assert load_noise_spec(path) == from_dict
    assert load_noise_spec(str(path)) == from_dict

    @dataclass
    class _StubFit:
        payload: dict

        def to_noise_spec(self):
            return self.payload

    assert load_noise_spec(_StubFit(raw)) == from_dict

    with pytest.raises(TypeError, match="to_noise_spec"):
        load_noise_spec(42)


def test_validate_noise_spec_normalizes_minimal_diffphase_and_null_bound():
    minimal = {
        "schema": SCHEMA,
        "white": {"visphi": onp.float64(0.85)},
        "common_mode": {"visphi": onp.float64(0.012)},
        "provenance": {
            "tool": "virgil-vlti",
            "run": "diffphase_pairs:8",
            "date": "2026-10-18",
            "bound": None,
        },
    }
    norm = validate_noise_spec(minimal)
    assert norm["instrument"] is None
    assert norm["white"] == {"visphi": 0.85}
    assert norm["gains"] == {
        "telescope": None,
        "baseline": None,
        "chromatic": None,
    }
    assert norm["closure_offsets"] == {"triangle": None, "baseline": None}
    assert norm["common_mode"] == {"visphi": 0.012}
    assert norm["provenance"]["bound"] is None
    assert json.loads(json.dumps(norm, sort_keys=True)) == norm


def test_validate_noise_spec_lists_all_problems_at_once():
    bad = {
        "schema": "virgil.noise_spec/2",
        "instrument": "   ",
        "unexpected_top": 1,
        "white": {"vis": -0.1, "v2": 0.5, "phi": True},
        "gains": {"telescope": 1.5, "modes": 0.01},
        "closure_offsets": {"triangle": 0.4, "unknown": 0.01},
        "common_mode": {"visphi": 0.8, "extra": 0.1},
        "provenance": {"run": "", "bound": "upper"},
    }
    with pytest.raises(ValueError) as exc_info:
        validate_noise_spec(bad)
    msg = str(exc_info.value)
    for expected in (
        "unsupported schema version 'virgil.noise_spec/2'",
        "unknown top-level keys",
        "instrument must be a non-empty string",
        "white has unknown observable keys ['v2']",
        "white.vis must be a finite scale > 0",
        "white.phi must be a finite scale > 0",
        "gains has unknown keys ['modes']",
        "gains.telescope",
        "closure_offsets has unknown keys ['unknown']",
        "closure_offsets.triangle",
        "common_mode has unknown keys ['extra']",
        "common_mode.visphi",
        "provenance.run",
        "provenance.date",
        "provenance.bound",
    ):
        assert expected in msg


def test_validate_noise_spec_warns_on_unit_slips_without_rejecting():
    spec = _base_spec(
        gains={"telescope": 0.12, "baseline": None, "chromatic": None},
        closure_offsets={"triangle": 0.05, "baseline": None},
        common_mode={"visphi": 0.08},
    )
    with pytest.warns(UserWarning) as record:
        out = validate_noise_spec(spec)
    texts = [str(w.message) for w in record]
    assert any("gains.telescope" in t and "V²" in t for t in texts)
    assert any("closure_offsets.triangle" in t and "rad" in t for t in texts)
    assert any("common_mode.visphi" in t and "rad" in t for t in texts)
    assert out["gains"]["telescope"] == 0.12


def test_instrument_match_delimiter_prefix_and_mismatch_warnings():
    spec_grav = _base_spec(instrument="GRAVITY")
    assert instrument_match({"insname": "GRAVITY"}, spec_grav)
    assert instrument_match({"insname": "GRAVITY_SC"}, spec_grav)
    assert instrument_match(
        {"insname": ("GRAVITY_SC_P1", "GRAVITY-FT")}, spec_grav
    )
    assert instrument_match(
        {"insname": "MATISSE_LM"}, _base_spec(instrument="MATISSE")
    )
    assert instrument_match({"insname": ""}, spec_grav)
    assert instrument_match({}, spec_grav)
    assert instrument_match(
        {"insname": "GRAVITY_SC"}, _base_spec(instrument=None)
    )

    # Reject reverse substrings and non-delimiter prefix collisions.
    for data_ins, spec_ins in (
        ("GRAVITY_SC", "PIONIER"),
        ("GRAVITY_SC", "GRAVITY_FT"),
        ("GRAVITY", "GRAVITY_FT"),
        ("N", "PIONIER"),
        ("H", "CHARA"),
        ("VIS", "VISIR"),
    ):
        with pytest.warns(UserWarning, match=f"{spec_ins}.*{data_ins}"):
            assert not instrument_match(
                {"insname": data_ins}, _base_spec(instrument=spec_ins)
            )


@pytest.mark.validates(
    "virgil.noise_spec.apply_noise_spec",
    roots=["self-consistency"],
    kind="check",
)
def test_apply_noise_spec_equals_manual_chain_and_is_instrument_invariant():
    data = _multi_channel_oidata(n_frames=2, extras=("t3amp",))
    spec = _base_spec(
        white={"vis": 0.32, "phi": 0.65, "t3amp": 0.80, "visphi": 0.90},
        gains={"telescope": 0.0055, "baseline": 0.0045, "chromatic": 0.002},
        closure_offsets={"triangle": 0.0036, "baseline": 0.002},
    )
    applied = apply_noise_spec(data, spec, insname="PIONIER")
    manual = (
        data.with_error_scale({"vis": 0.32, "phi": 0.65, "t3amp": 0.80})
        .with_gains(telescope=0.0055, baseline=0.0045, chromatic=0.002)
        .with_closure_offsets(triangle=0.0036, baseline=0.002)
    )

    onp.testing.assert_allclose(applied.d_vis, manual.d_vis, rtol=0, atol=1e-7)
    onp.testing.assert_allclose(applied.d_phi, manual.d_phi, rtol=0, atol=1e-7)
    onp.testing.assert_allclose(
        applied.extras[0].errors, manual.extras[0].errors, rtol=0, atol=1e-7
    )
    onp.testing.assert_allclose(
        whitened_residuals(SCENE, applied),
        whitened_residuals(SCENE, manual),
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(model_loglike(SCENE, applied)) == pytest.approx(
        float(model_loglike(SCENE, manual)), rel=1e-6
    )

    # Identical likelihood whatever instrument string the spec carries, while
    # emitting UserWarning when insname disagrees.
    with pytest.warns(UserWarning, match="MIRC-X.*PIONIER"):
        other_inst = apply_noise_spec(
            data,
            _base_spec(**{**spec, "instrument": "MIRC-X"}),
            insname="PIONIER",
        )
    assert float(model_loglike(SCENE, applied)) == pytest.approx(
        float(model_loglike(SCENE, other_inst)), rel=1e-12
    )

    # Zero or None widths leave gains and phase_offsets unattached.
    white_only = apply_noise_spec(
        data,
        _base_spec(
            gains={"telescope": 0.0, "baseline": None, "chromatic": None},
            closure_offsets={"triangle": None, "baseline": 0.0},
        ),
    )
    assert white_only.gains is None
    assert white_only.phase_offsets is None


def test_apply_noise_spec_skips_absent_gain_and_offset_observables():
    data = _multi_channel_oidata(n_frames=2)
    spec = _base_spec()

    vis_only = data.select(observables="vis")
    vis_applied = apply_noise_spec(vis_only, spec)
    assert vis_applied.gains is not None
    assert vis_applied.phase_offsets is None
    onp.testing.assert_allclose(
        vis_applied.d_vis, vis_only.d_vis * 0.32, rtol=0, atol=1e-7
    )

    phi_only = data.select(observables="phi")
    phi_applied = apply_noise_spec(phi_only, spec)
    assert phi_applied.gains is None
    assert phi_applied.phase_offsets is not None
    onp.testing.assert_allclose(
        phi_applied.d_phi, phi_only.d_phi * 0.65, rtol=0, atol=1e-7
    )


def test_apply_noise_spec_refuses_stacking_upfront():
    data = _multi_channel_oidata(n_frames=2)
    spec = _base_spec()
    applied = apply_noise_spec(data, spec)
    white_only = _base_spec(gains=None, closure_offsets=None)

    with pytest.raises(ValueError, match="refuses to stack"):
        apply_noise_spec(applied, white_only)

    offsets_applied = apply_noise_spec(data, _base_spec(gains=None))
    with pytest.raises(ValueError, match="refuses to stack"):
        apply_noise_spec(offsets_applied, white_only)

    dp_prior = _multi_channel_oidata(
        n_frames=2, extras=("visphi",)
    ).with_continuum(lines=LINE_WINDOW, prior_width=0.05)
    with pytest.raises(ValueError, match="differential-phase nuisance modes"):
        apply_noise_spec(dp_prior, white_only)


def test_apply_noise_spec_common_mode_visphi_raises_until_supported():
    data = _multi_channel_oidata(
        n_frames=2, extras=("visphi",)
    ).with_continuum(lines=LINE_WINDOW)
    spec = _base_spec(
        white={"visphi": 0.75},
        gains=None,
        closure_offsets=None,
        common_mode={"visphi": 0.015},
    )

    # Absent visphi on data skips common_mode.visphi cleanly.
    v2_cp_only = _multi_channel_oidata(n_frames=2)
    assert apply_noise_spec(v2_cp_only, spec).gains is None

    # Null/zero common_mode.visphi rescales white.visphi cleanly.
    scaled_only = apply_noise_spec(
        data, _base_spec(**{**spec, "common_mode": {"visphi": 0.0}})
    )
    onp.testing.assert_allclose(
        scaled_only.extras[0].errors,
        data.extras[0].errors * 0.75,
        rtol=0,
        atol=1e-7,
    )

    with pytest.raises(NotImplementedError, match="common_mode.visphi"):
        apply_noise_spec(data, spec)
