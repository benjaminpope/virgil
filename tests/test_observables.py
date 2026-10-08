"""Extra observables: OI_FLUX, T3AMP, VISAMP and differential VISPHI."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.likelihood import (
    _whitened_and_errors,
    flux_scale_posterior,
    inflated_errors,
    model_loglike,
    whitened_residuals,
)
from virgil.models import PointSource, System
from virgil.observables import (
    DifferentialPhase,
    FluxSpectrum,
    continuum_operator,
)
from virgil.oidata import OIData, cp_indices
from virgil.oifits import build_hdulist, read_oifits, write_oifits
from virgil.spectra import GaussianLine, PowerLaw, Sum
from virgil._utils import mas2rad
from tests._shared import PAIRS, TRIANGLES  # noqa: E402

STATIONS = onp.array([[0.0, 0.0], [32.0, 2.0], [14.0, 26.0], [-11.0, 18.0]])
WAVES = onp.linspace(2.150e-6, 2.180e-6, 12)
LINE = (2.163e-6, 2.169e-6)


def _baselines(pairs):
    delta = STATIONS[pairs[:, 1] - 1] - STATIONS[pairs[:, 0] - 1]
    return delta[:, 0], delta[:, 1]


def _scene(dra=0.5, ddec=0.2, amplitude=0.5):
    """A star and a fainter component with a Brγ-like emission line."""
    line = GaussianLine(amplitude=amplitude, line_wavel=2.166e-6, fwhm=3e-9)
    flux = Sum(cont=PowerLaw(0.1, 0.0, wavel0=2.166e-6), brg=line)
    return System(
        star=PointSource(), blob=PointSource(dra=dra, ddec=ddec, flux=flux)
    )


def _tables(scene=None, scale=3.7, amptyp="absolute", reverse=False):
    """Noiseless OIFITS tables of ``scene`` with every table filled."""
    scene = _scene() if scene is None else scene
    u, v = _baselines(PAIRS)
    cvis = onp.asarray(scene.model(u[:, None], v[:, None], WAVES[None, :]))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    cp = onp.angle(cvis[i1] * cvis[i2] / cvis[i3])
    t3amp = onp.abs(cvis[i1] * cvis[i2] * cvis[i3])
    u1, v1 = _baselines(TRIANGLES[:, [0, 1]])
    u2, v2 = _baselines(TRIANGLES[:, [1, 2]])
    flux = scale * onp.asarray(scene.total_spectrum(WAVES))
    amp = onp.abs(cvis)
    if amptyp == "correlated flux":
        amp = amp * flux
    vis_pairs, vis_u, vis_v, phase = PAIRS.copy(), u.copy(), v.copy(), cvis
    if reverse:  # the first baseline stored as (2, 1)
        vis_pairs[0] = vis_pairs[0, ::-1]
        vis_u[0], vis_v[0] = -u[0], -v[0]
        phase = cvis.copy()
        phase[0] = onp.conj(cvis[0])
    return {
        "info": {"TARGET": "X", "INSTRUME": "TEST", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": WAVES, "EFF_BAND": 1e-9},
        "OI_VIS2": {
            "VIS2DATA": onp.abs(cvis) ** 2,
            "VIS2ERR": onp.full(cvis.shape, 1e-3),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": PAIRS,
        },
        "OI_VIS": {
            "VISAMP": amp,
            "VISAMPERR": onp.full(cvis.shape, 1e-3),
            "VISPHI": onp.rad2deg(onp.angle(phase)),
            "VISPHIERR": onp.full(cvis.shape, 0.2),
            "UCOORD": vis_u,
            "VCOORD": vis_v,
            "STA_INDEX": vis_pairs,
            "AMPTYP": amptyp,
            "PHITYP": "differential",
        },
        "OI_T3": {
            "T3AMP": t3amp,
            "T3AMPERR": onp.full(cp.shape, 1e-3),
            "T3PHI": onp.rad2deg(cp),
            "T3PHIERR": onp.full(cp.shape, 0.5),
            "U1COORD": u1,
            "V1COORD": v1,
            "U2COORD": u2,
            "V2COORD": v2,
            "STA_INDEX": TRIANGLES,
        },
        "OI_FLUX": {
            "FLUXDATA": onp.tile(flux, (4, 1)),
            "FLUXERR": onp.full((4, WAVES.size), 0.01),
            "STA_INDEX": onp.arange(1, 5),
        },
    }


ALL = ("flux", "t3amp", "visamp", "visphi")
# A stated prior on the grey scale (the tables' true scale is 3.7): broad,
# and not taken from the data.
SCALE_PRIOR = (3.0, 3.0)


# === READERS AND WRITER ===


def test_extras_are_off_by_default():
    record = read_oifits(build_hdulist(_tables()))
    assert not any(k in record for k in ("flux", "t3amp", "visamp", "visphi"))
    assert OIData(record).extras == ()


def test_flux_round_trip(tmp_path):
    tables = _tables()
    path = write_oifits(tables, tmp_path / "flux.fits")
    record = read_oifits(path, extras=("flux",))["flux"]
    onp.testing.assert_allclose(
        record["value"], tables["OI_FLUX"]["FLUXDATA"].reshape(-1)
    )
    onp.testing.assert_allclose(record["wavel"], onp.tile(WAVES, 4))
    onp.testing.assert_array_equal(
        record["station"], onp.repeat([1, 2, 3, 4], 12)
    )
    onp.testing.assert_array_equal(
        record["row"], onp.repeat(onp.arange(4), 12)
    )
    nflux = read_oifits(path, extras=("nflux",))["nflux"]
    onp.testing.assert_allclose(nflux["value"], record["value"])
    with pytest.raises(ValueError, match="not both"):
        read_oifits(path, extras=("flux", "nflux"))


def test_unused_oi_flux_does_not_affect_default_target_selection():
    hdul = build_hdulist(_tables())
    hdul["OI_FLUX"].data["TARGET_ID"] = 2
    record = read_oifits(hdul)
    assert record["u"].size == PAIRS.shape[0] * WAVES.size
    with pytest.raises(ValueError, match="several targets"):
        read_oifits(hdul, extras=("flux",))


def test_t3amp_round_trip(tmp_path):
    tables = _tables()
    path = write_oifits(tables, tmp_path / "t3.fits")
    record = read_oifits(path, extras=("t3amp",))
    onp.testing.assert_allclose(
        record["t3amp"]["value"], tables["OI_T3"]["T3AMP"].reshape(-1)
    )
    # Aligned with the closure phases, so the model reproduces the data.
    data = OIData(record)
    (block,) = data.extras
    cvis = _scene().model(data.u, data.v, data.wavel)
    onp.testing.assert_allclose(
        block.predict(_scene(), cvis), block.values, rtol=1e-5
    )


@pytest.mark.parametrize("amptyp", ["absolute", "correlated flux"])
def test_visamp_round_trip(amptyp, tmp_path):
    tables = _tables(amptyp=amptyp)
    path = write_oifits(tables, tmp_path / "vis.fits")
    record = read_oifits(path, extras=("visamp",))
    assert record["visamp"]["amptyp"] == amptyp
    onp.testing.assert_allclose(
        record["visamp"]["value"], tables["OI_VIS"]["VISAMP"].reshape(-1)
    )
    data = OIData(record)
    if amptyp == "correlated flux":
        data = data.with_flux_scale(scale=SCALE_PRIOR)
    (block,) = data.extras
    assert block.kind == ("visamp" if amptyp == "absolute" else "corrflux")
    # The model at the matched samples reproduces the data (up to the
    # grey scale, recovered for correlated fluxes).
    r = whitened_residuals(_scene(), data)
    # The values themselves round-trip exactly (float64 columns, checked
    # above). This compares two float32 evaluations of the model, at the
    # fixture's (6, 12) layout and at the reader's flat samples. Near the
    # 3 nm line, λ − λ_line loses ~4 digits to cancellation in float32
    # (ulp(λ) ≈ 2e-13 m), so XLA's op order changes the line flux by ~3e-5
    # (measured 0.035σ on Linux CI, 5e-4σ on macOS). Correlated fluxes of
    # ~5 with σ = 1e-3 (S/N 5000) see that; |V| with σ = 1e-3 does not.
    limit = 0.1 if amptyp == "correlated flux" else 1e-2
    assert float(np.max(np.abs(r))) < limit
    if amptyp == "correlated flux":
        scale = flux_scale_posterior(_scene(), data)["corrflux"]["scale"]
        onp.testing.assert_allclose(scale, 3.7, rtol=1e-4)


def test_visphi_round_trip_with_a_reversed_baseline(tmp_path):
    tables = _tables(reverse=True)
    path = write_oifits(tables, tmp_path / "phi.fits")
    record = read_oifits(path, extras=("visphi",))
    visphi = record["visphi"]
    # Phases are stored in the orientation of the matched V² samples.
    cvis = onp.asarray(
        _scene().model(record["u"], record["v"], record["wavel"])
    )
    onp.testing.assert_allclose(
        # float32 model phases of two layouts (see test_visamp_round_trip):
        # 2e-6 rad on Linux CI, 1e-8 on macOS; VISPHIERR is 3.5e-3 rad.
        visphi["value"],
        onp.angle(cvis[visphi["sample"]]),
        atol=5e-6,
    )
    assert record["u"].size == PAIRS.shape[0] * WAVES.size  # no new samples


def test_visphi_without_closure_phases_and_differential_refusal():
    tables = _tables()
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    with pytest.raises(ValueError, match="PHITYP"):
        read_oifits(hdul)
    data = OIData(read_oifits(hdul, extras=("visphi",)))
    assert not data.has_phases
    (block,) = data.extras
    assert not block.closure_free  # nothing to double count


def test_correlated_flux_without_vis2_gives_flagged_samples():
    tables = _tables(amptyp="correlated flux")
    del tables["OI_VIS2"]
    hdul = build_hdulist(tables)
    with pytest.raises(ValueError, match="AMPTYP"):
        read_oifits(hdul)
    data = OIData(read_oifits(hdul, extras=("visamp",)))
    assert data.vis.size == 0 and data.extras[0].kind == "corrflux"


def test_extras_concatenate_over_files(tmp_path):
    paths = [
        write_oifits(_tables(), tmp_path / f"f{i}.fits") for i in range(2)
    ]
    one = read_oifits(paths[0], extras=ALL)
    two = read_oifits(paths, extras=ALL)
    n = one["u"].size
    onp.testing.assert_array_equal(
        two["visphi"]["sample"][-one["visphi"]["sample"].size :],
        one["visphi"]["sample"] + n,
    )
    data = OIData(two)
    assert data.n_independent == 2 * OIData(one).n_independent


# === DIFFERENTIAL PHASE ===


def _visphi_data(scene, waves, pairs, closure_free=False, **windows):
    """A dict-built OIData with differential phases only (one frame)."""
    delta = STATIONS[pairs[:, 1] - 1] - STATIONS[pairs[:, 0] - 1]
    u, v = delta[:, 0] * 4, delta[:, 1] * 4  # up to 160 m
    n = pairs.shape[0] * waves.size
    cvis = onp.asarray(scene.model(u[:, None], v[:, None], waves[None, :]))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": waves,
            "vis": onp.full((len(u), waves.size), onp.nan),
            "d_vis": onp.ones((len(u), waves.size)),
            "frame": onp.zeros(len(u), int),
            "stations": pairs,
            "visphi": {
                "sample": onp.arange(n),
                "value": onp.angle(cvis).reshape(-1),
                "error": onp.full(n, 0.01),
            },
        }
    )
    return data.with_continuum(**windows) if windows else data


def _photocentre_phase(scene, u, v, wavel):
    """−2π u·p(λ)/λ for the flux-weighted photocentre p(λ)."""
    weights = [onp.asarray(c._weight(wavel)) for c in scene.parts]
    total = sum(weights)
    px = sum(w * float(c.dra) for w, c in zip(weights, scene.parts)) / total
    py = sum(w * float(c.ddec) for w, c in zip(weights, scene.parts)) / total
    return -2.0 * onp.pi * (u * px + v * py) * mas2rad / wavel


def _block_operator(block, f=0):
    """The dense operator Qᵀ ⊗ W of frame ``f`` on (baseline, channel)."""
    return onp.kron(onp.asarray(block.q[f]), onp.asarray(block.w[f]))


def test_unresolved_line_matches_the_photocentre_shift():
    scene = _scene(dra=0.05, ddec=-0.03)  # ≪ λ/B ≈ 3 mas
    data = _visphi_data(scene, WAVES, PAIRS, lines=[LINE])
    (block,) = data.extras
    cvis = scene.model(data.u, data.v, data.wavel)
    predicted = onp.asarray(block.predict(scene, cvis))
    u = onp.asarray(data.u).reshape(-1, WAVES.size)
    v = onp.asarray(data.v).reshape(-1, WAVES.size)
    phase = _photocentre_phase(scene, u, v, WAVES[None, :])
    expected = (_block_operator(block) @ phase.reshape(-1))[block.keep]
    assert onp.max(onp.abs(expected)) > 1e-3  # the line moves the phase
    onp.testing.assert_allclose(predicted, expected, rtol=0.02, atol=2e-5)


def test_resolved_binary_needs_the_exact_phase():
    # A 28 mas binary on 160 m baselines (λ/B ≈ 3 mas), the line in the
    # companion: the photocentre formula fails, the exact arg V does not.
    scene = _scene(dra=20.0, ddec=-19.6, amplitude=1.5)
    data = _visphi_data(scene, WAVES, PAIRS, lines=[LINE])
    (block,) = data.extras
    cvis = onp.asarray(scene.model(data.u, data.v, data.wavel))
    predicted = onp.asarray(block.predict(scene, cvis))
    # Independent reference: unwrap arg V along wavelength with NumPy and
    # subtract an offset + delay fitted to the continuum channels.
    phases = onp.unwrap(onp.angle(cvis).reshape(-1, WAVES.size), axis=1)
    sigma = 1.0 / WAVES
    cont = ~((WAVES >= LINE[0]) & (WAVES <= LINE[1]))
    design = onp.column_stack([onp.ones(cont.sum()), sigma[cont]])
    coef = onp.linalg.lstsq(design, phases[:, cont].T, rcond=None)[0]
    exact = phases - (coef[0][:, None] + coef[1][:, None] * sigma[None, :])
    line = ~cont
    w = onp.asarray(block.w[0])[:, : WAVES.size]
    # The block keeps W = R M, a rotation R of the line rows M of N.
    n_op = onp.eye(WAVES.size) - continuum_operator(WAVES, cont, 1)
    rotation = w @ onp.linalg.pinv(n_op[line])
    reference = (exact[:, line] @ rotation.T).reshape(-1)
    onp.testing.assert_allclose(predicted, reference, atol=1e-4)
    u = onp.asarray(data.u).reshape(-1, WAVES.size)
    v = onp.asarray(data.v).reshape(-1, WAVES.size)
    photocentre = _photocentre_phase(scene, u, v, WAVES[None, :])
    approx = (_block_operator(block) @ photocentre.reshape(-1))[block.keep]
    # The photocentre formula is off by far more than the 0.01 rad errors.
    assert onp.max(onp.abs(approx - predicted)) > 0.3


def test_continuum_operator_removes_offset_and_delay():
    cont = ~((WAVES >= LINE[0]) & (WAVES <= LINE[1]))
    n_op = onp.eye(WAVES.size) - continuum_operator(WAVES, cont, 1)
    onp.testing.assert_allclose(n_op @ onp.ones(WAVES.size), 0, atol=1e-10)
    onp.testing.assert_allclose(n_op @ (2e-7 / WAVES), 0, atol=1e-8)
    # The fit uses only continuum channels: a spike in the line survives.
    spike = onp.where(~cont, 1.0, 0.0)
    onp.testing.assert_allclose((n_op @ spike)[~cont], 1.0)
    with pytest.raises(ValueError, match="continuum channels"):
        continuum_operator(WAVES, onp.eye(WAVES.size)[0] > 0, 1)


def test_visphi_and_closure_phases_count_each_constraint_once():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    data = data.with_continuum(lines=[LINE])
    (block,) = data.extras
    assert block.closure_free
    n_line = int(((WAVES >= LINE[0]) & (WAVES <= LINE[1])).sum())
    # Per line channel: 3 independent closure phases (4 telescopes) and
    # N - 1 = 3 closure-free differential phases: 6 = all 6 baselines.
    assert block.n_independent == 3 * n_line
    q = onp.asarray(block.q[0])
    t = onp.zeros((4, 6))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    for k in range(4):
        t[k, i1[k]] += 1
        t[k, i2[k]] += 1
        t[k, i3[k]] -= 1
    # The rows follow the stations of the frame's baselines.
    onp.testing.assert_allclose(q @ t.T, 0.0, atol=1e-12)
    assert onp.linalg.matrix_rank(onp.vstack([t, q])) == 6
    assert data.n_independent == 72 + 36 + 3 * n_line


def test_visphi_whitening_matches_a_dense_covariance():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    data = data.with_continuum(lines=[LINE])
    (block,) = data.extras
    rng = onp.random.default_rng(1)
    errors = rng.uniform(0.005, 0.02, block.errors.shape)
    block = block.with_errors(errors)
    other = _scene(dra=0.6, ddec=0.1)
    cvis = other.model(data.u, data.v, data.wavel)
    white, effective = block.whiten(
        block.predict(other, cvis), block.data(), None
    )
    # Dense reference, from scratch: K = Qᵀ ⊗ M with the unrotated line
    # rows M of N, acting on the per-sample phase residuals.
    cont = ~((WAVES >= LINE[0]) & (WAVES <= LINE[1]))
    m = (onp.eye(WAVES.size) - continuum_operator(WAVES, cont, 1))[~cont]
    q = onp.asarray(block.q[0])
    grid = onp.asarray(block.grid[0])
    k = onp.kron(q, m)
    model_phase = onp.unwrap(onp.angle(onp.asarray(cvis)[block.sample][grid]))
    data_phase = onp.unwrap(onp.asarray(block.values)[grid])
    resid = k @ (model_phase - data_phase).reshape(-1)
    cov = k @ onp.diag(errors[grid].reshape(-1) ** 2) @ k.T
    chi2 = resid @ onp.linalg.solve(cov, resid)
    onp.testing.assert_allclose(float(np.sum(white**2)), chi2, rtol=1e-3)
    # The effective errors carry ½ log det of the covariance, up to the
    # rotation's (constant) Jacobian, which a ratio of two models cancels:
    # compare against a second set of errors.
    _, slogdet = onp.linalg.slogdet(cov)
    cov2 = k @ onp.diag((2 * errors[grid]).reshape(-1) ** 2) @ k.T
    _, slogdet2 = onp.linalg.slogdet(cov2)
    _, effective2 = block.with_errors(2 * errors).whiten(
        block.predict(other, cvis), block.data(), None
    )
    onp.testing.assert_allclose(
        float(np.sum(np.log(effective2)) - np.sum(np.log(effective))),
        0.5 * (slogdet2 - slogdet),
        rtol=1e-4,
    )


# === SPECTRA ===


def test_flux_grey_scale_is_recovered_and_marginalized():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("flux",)))
    data = data.with_flux_scale(scale=SCALE_PRIOR)
    (block,) = [b for b in data.extras if b.kind == "flux"]
    scene = _scene()
    post = flux_scale_posterior(scene, data)["flux"]
    onp.testing.assert_allclose(post["scale"], 3.7, rtol=1e-4)
    # The marginal likelihood equals a dense Gaussian with the scale
    # integrated out: covariance D + A Λ Aᵀ about A μ.
    prediction = block.predict(scene, None)
    white, effective = block.whiten(prediction, block.values, block.errors)
    template = onp.asarray(prediction) / onp.asarray(block.mu)[block.group]
    var = (block.widths[0] * block.mu[0]) ** 2
    cov = onp.diag(onp.asarray(block.errors) ** 2) + var * onp.outer(
        template, template
    )
    resid = onp.asarray(prediction - block.values)
    chi2 = resid @ onp.linalg.solve(cov, resid)
    onp.testing.assert_allclose(float(np.sum(white**2)), chi2, rtol=1e-3)
    _, logdet = onp.linalg.slogdet(cov)
    onp.testing.assert_allclose(
        float(np.sum(np.log(effective))), 0.5 * logdet, rtol=1e-4
    )


def test_flux_scale_per_row_and_polynomial():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("flux",)))
    data = data.with_flux_scale(scale=SCALE_PRIOR)
    per_row = data.with_flux_scale(per="station", poly_order=1)
    (block,) = per_row.extras
    assert block.members.shape[0] == 4 and len(block.widths) == 2
    post = flux_scale_posterior(_scene(), per_row)["flux"]
    onp.testing.assert_allclose(post["scale"], 3.7, rtol=1e-3)
    onp.testing.assert_allclose(post["mean"][:, 1], 0.0, atol=1e-3)
    # A chromatic calibration error is absorbed by the polynomial mode.
    tilt = onp.linspace(0.9, 1.1, WAVES.size)
    tilted = eqx_values(block, block.values * onp.tile(tilt, 4))
    grey = eqx_values(data.extras[0], tilted.values)
    chi_grey = float(
        np.sum(
            grey.whiten(
                grey.predict(_scene(), None), grey.values, grey.errors
            )[0]
            ** 2
        )
    )
    chi_poly = float(
        np.sum(
            tilted.whiten(
                tilted.predict(_scene(), None), tilted.values, tilted.errors
            )[0]
            ** 2
        )
    )
    assert chi_poly < 0.1 * chi_grey


def eqx_values(block, values):
    return block.rebuild(values=onp.asarray(values))


def test_nflux_is_the_continuum_normalized_total_spectrum():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("nflux",)))
    cont = [(2.150e-6, 2.160e-6), (2.172e-6, 2.180e-6)]
    data = data.with_continuum(cont)
    (block,) = data.extras
    scene = _scene()
    total = onp.asarray(scene.total_spectrum(WAVES))
    in_cont = (WAVES <= 2.160e-6) | (WAVES >= 2.172e-6)
    expected = onp.tile(total / total[in_cont].mean(), 4)
    onp.testing.assert_allclose(
        block.predict(scene, None), expected, rtol=1e-5
    )
    assert block.mu[0] == 1.0
    # The line's equivalent width is what NFLUX measures: a model without
    # the line fits worse than the true one.
    flat = _scene(amplitude=0.0)
    assert float(np.sum(whitened_residuals(flat, data) ** 2)) > 100 * (
        float(np.sum(whitened_residuals(scene, data) ** 2)) + 1.0
    )


# === ERROR FLOORS AND BOOKKEEPING ===


def test_error_floors_leave_the_closure_penalty_rows_alone():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    floor = onp.deg2rad(2.0)  # above the 0.5° closure-phase errors
    floored = data.with_error_floor(absolute={"phi": floor, "visphi": 0.05})
    onp.testing.assert_allclose(floored.d_phi, floor, rtol=1e-6)
    assert floored.n_residuals == data.n_residuals
    other = _scene(dra=0.8)
    white, errors = _whitened_and_errors(other, floored, {})
    n_vis, n_cp = floored.vis.size, floored.phi.size
    k = floored.cp_noise.size
    penalty = slice(n_vis + k, n_vis + k + n_cp)
    onp.testing.assert_allclose(errors[penalty], 1.0 / onp.sqrt(2.0 * onp.pi))
    # The whitened rows use the floored errors, as if they had been given.
    manual = eqx_set_phi_errors(data, floor)
    w2, e2 = _whitened_and_errors(other, manual, {})
    n_main = n_vis + k + n_cp
    onp.testing.assert_allclose(white[:n_main], w2[:n_main], rtol=1e-5)
    onp.testing.assert_allclose(errors[:n_main], e2[:n_main], rtol=1e-5)
    onp.testing.assert_allclose(floored.extras[0].errors, 0.05, rtol=1e-6)


def eqx_set_phi_errors(data, value):
    import equinox as eqx

    return eqx.tree_at(
        lambda d: d.d_phi, data, np.full_like(data.d_phi, value)
    )


def test_error_floors_take_the_maximum_and_check_their_names():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("flux",)))
    floored = data.with_error_floor(
        absolute={"vis": 2e-3}, relative={"vis": 0.01, "flux": 0.01}
    )
    expected = onp.maximum(
        onp.maximum(onp.asarray(data.d_vis), 2e-3), 0.01 * onp.abs(data.vis)
    )
    onp.testing.assert_allclose(floored.d_vis, expected, rtol=1e-6)
    flux = data.extras[0]
    onp.testing.assert_allclose(
        floored.extras[0].errors,
        onp.maximum(flux.errors, 0.01 * onp.abs(flux.values)),
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="No observables"):
        data.with_error_floor(absolute={"t3amp": 0.1})
    with pytest.raises(ValueError, match="phases"):
        data.with_error_floor(relative={"phi": 0.1})


def test_inflated_errors_where_and_combine():
    data = OIData(read_oifits(build_hdulist(_tables())))
    prediction = data.model(_scene(dra=0.7))
    as_floor = inflated_errors(
        data, prediction, vis_error=2e-3, where="data", combine="max"
    )
    onp.testing.assert_allclose(
        as_floor[: data.vis.size], onp.maximum(data.d_vis, 2e-3)
    )
    added = inflated_errors(data, prediction, vis_error=2e-3)
    onp.testing.assert_allclose(
        added[: data.vis.size], onp.hypot(data.d_vis, 2e-3), rtol=1e-6
    )
    onp.testing.assert_allclose(
        added[data.vis.size :], data.flatten_data()[1][data.vis.size :]
    )


def test_select_split_scale_and_simulate_keep_the_extras():
    data = (
        OIData(read_oifits(build_hdulist(_tables()), extras=ALL))
        .with_continuum(lines=[LINE])
        .with_flux_scale(scale=SCALE_PRIOR)
    )
    assert [b.kind for b in data.extras] == [
        "flux",
        "visamp",
        "t3amp",
        "visphi",
    ]
    part = data.select(2.160e-6, 2.180e-6)
    assert all(b.n_independent > 0 for b in part.extras)
    assert part.extras[0].wavel.min() >= 2.160e-6
    (whole,) = data.split_by_epoch()
    assert whole.n_independent == data.n_independent
    scaled = data.with_error_scale(2.0)
    onp.testing.assert_allclose(
        scaled.extras[3].errors, 2 * data.extras[3].errors
    )
    sim = data.with_model(_scene(), key=jax.random.PRNGKey(0))
    chi2 = float(np.sum(whitened_residuals(_scene(), sim) ** 2))
    assert 0.5 * data.n_residuals < chi2 < 1.5 * data.n_residuals
    noiseless = data.with_model(_scene())
    assert (
        float(np.max(np.abs(whitened_residuals(_scene(), noiseless)))) < 1e-2
    )
    assert bool(np.isfinite(model_loglike(_scene(), data)))
    assert data.has_model_covariance  # the flux scale


def test_error_scales_per_observable_cover_the_extras():
    # with_error_scale takes a factor per kind, extras included, and the
    # residual rows error_scale(by_observable=True) assigns to each kind
    # are the ones that scale with it.
    from virgil.imaging import _observable_blocks

    data = (
        OIData(read_oifits(build_hdulist(_tables()), extras=ALL))
        .with_continuum(lines=[LINE])
        .with_flux_scale(scale=SCALE_PRIOR)
    )
    scaled = data.with_error_scale({"visamp": 2.0, "phi": 3.0})
    kinds = [b.kind for b in data.extras]
    for kind, before, after in zip(kinds, data.extras, scaled.extras):
        factor = 2.0 if kind == "visamp" else 1.0
        onp.testing.assert_allclose(after.errors, factor * before.errors)
    onp.testing.assert_allclose(scaled.d_phi, 3.0 * data.d_phi)
    onp.testing.assert_allclose(scaled.d_vis, data.d_vis)
    rows, counts = _observable_blocks([data])
    assert list(rows) == ["vis", "phi", *kinds]
    assert sum(counts.values()) == data.n_independent
    assert sum(v.size for v in rows.values()) == data.n_residuals
    sim = data.with_model(_scene(), key=jax.random.PRNGKey(0))
    base = onp.asarray(whitened_residuals(_scene(), sim))
    for kind in rows:
        other = onp.asarray(
            whitened_residuals(_scene(), sim.with_error_scale({kind: 2.0}))
        )
        changed = onp.flatnonzero(~onp.isclose(other, base, rtol=1e-5))
        assert changed.size and set(changed) <= set(rows[kind])


def test_fit_refuses_least_squares_with_a_marginalized_scale():
    import numpyro.distributions as dist

    from virgil.fitting import fit

    data = OIData(read_oifits(build_hdulist(_tables()), extras=("flux",)))
    data = data.with_flux_scale(scale=SCALE_PRIOR)
    priors = {"blob.dra": dist.Uniform(0.0, 1.0)}
    with pytest.raises(TypeError, match="flux scales"):
        fit(_scene(), priors, data, method="lm", max_steps=1)


def test_flux_build_checks_its_inputs():
    with pytest.raises(ValueError, match="per must be"):
        FluxSpectrum.build(
            "flux", [1.0], [0.1], [2e-6], [0], [0], [1], per="night"
        )
    with pytest.raises(ValueError, match="two values"):
        DifferentialPhase.build(
            [0.0, 0.1],
            [0.1, 0.1],
            [0, 1],
            [2e-6, 2e-6],
            [0, 0],
            [[1, 2]] * 2,
            row=[0, 0],
        )


def test_likelihood_compiles_and_differentiates_with_traced_data():
    import equinox as eqx

    data = (
        OIData(read_oifits(build_hdulist(_tables()), extras=ALL))
        .with_continuum(lines=[LINE])
        .with_flux_scale(scale=SCALE_PRIOR)
    )
    scene = _scene(dra=0.6)
    eager = model_loglike(scene, data)
    jitted = eqx.filter_jit(model_loglike)(scene, data)
    onp.testing.assert_allclose(jitted, eager, rtol=1e-5)
    grads = eqx.filter_jit(eqx.filter_grad(model_loglike))(scene, data)
    assert bool(np.isfinite(grads.parts[1].dra))
    with jax.enable_x64(True):
        assert bool(np.isfinite(model_loglike(scene, data)))


def _dense_prior_reference(block, other, cvis, widths, errors):
    """χ² and log det of the finite-prior VISPHI block, from scratch."""
    grid = onp.asarray(block.grid[0])
    q = onp.asarray(block.q[0])
    sel = onp.asarray(block.w[0])  # selects the used channels
    k = onp.kron(q, sel)
    model_phase = onp.unwrap(onp.angle(onp.asarray(cvis)[block.sample][grid]))
    data_phase = onp.unwrap(onp.asarray(block.values)[grid])
    resid = k @ (model_phase - data_phase).reshape(-1)
    used = sel.argmax(axis=1)
    sigma = 1.0 / WAVES[used]
    x = (sigma - sigma.mean()) / (sigma.max() - sigma.min())
    basis = onp.column_stack([onp.ones_like(x), x])
    nb = q.shape[1]
    modes = onp.kron(q, basis) @ onp.kron(onp.eye(nb), onp.diag(widths))
    cov = k @ onp.diag(errors[grid].reshape(-1) ** 2) @ k.T + modes @ modes.T
    chi2 = resid @ onp.linalg.solve(cov, resid)
    return chi2, onp.linalg.slogdet(cov)[1]


def test_visphi_finite_prior_matches_a_dense_marginal():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    data = data.with_continuum(lines=[LINE], prior_width=(0.3, 0.5))
    (block,) = data.extras
    assert block.n_independent == 3 * WAVES.size  # every channel kept
    rng = onp.random.default_rng(2)
    errors = rng.uniform(0.005, 0.02, block.errors.shape)
    block = block.with_errors(errors)
    other = _scene(dra=0.6, ddec=0.1)
    cvis = other.model(data.u, data.v, data.wavel)
    white, effective = block.whiten(
        block.predict(other, cvis), block.data(), None
    )
    chi2, logdet = _dense_prior_reference(
        block, other, cvis, onp.array([0.3, 0.5]), errors
    )
    onp.testing.assert_allclose(float(np.sum(white**2)), chi2, rtol=1e-3)
    onp.testing.assert_allclose(
        float(np.sum(np.log(effective))), 0.5 * logdet, rtol=1e-3
    )


def test_finite_prior_preserves_the_phase_anchor():
    scene = _scene()
    data = _visphi_data(scene, WAVES, PAIRS, lines=[LINE])
    (block,) = data.with_continuum(lines=[LINE], prior_width=(0.3, 0.5)).extras
    expected = onp.unwrap(
        onp.asarray(block.values).reshape(len(PAIRS), WAVES.size), axis=1
    )
    onp.testing.assert_allclose(
        onp.asarray(block.data()).reshape(len(PAIRS), WAVES.size),
        expected,
        atol=1e-6,
    )
    cvis = scene.model(data.u, data.v, data.wavel)
    onp.testing.assert_allclose(
        block.predict(scene, cvis), block.data(), atol=1e-6
    )


def test_with_model_draws_marginalized_extra_modes():
    data = (
        OIData(
            read_oifits(build_hdulist(_tables()), extras=("flux", "visphi"))
        )
        .with_continuum(lines=[LINE], prior_width=(0.3, 0.5))
        .with_flux_scale(scale=SCALE_PRIOR)
    )
    scene = _scene()
    key = jax.random.PRNGKey(5)
    simulated = data.with_model(scene, key=key, noise_scale=0.0)
    repeated = data.with_model(scene, key=key, noise_scale=0.0)
    for original, draw in zip(data.extras, simulated.extras):
        if original.kind == "flux":
            prediction = original.predict(scene, None)
            assert float(np.linalg.norm(draw.values - prediction)) > 0.0
        elif original.kind == "visphi":
            cvis = data._cvis(scene)
            phases = np.angle(cvis)[original.sample]
            assert float(np.linalg.norm(draw.values - phases)) > 0.0
    for draw, again in zip(simulated.extras, repeated.extras):
        onp.testing.assert_array_equal(draw.values, again.values)


def test_visphi_broad_prior_tends_to_the_projection():
    with jax.enable_x64(True):
        data = OIData(
            read_oifits(build_hdulist(_tables()), extras=("visphi",))
        )
        other = _scene(dra=0.6, ddec=0.1)
        cvis = other.model(data.u, data.v, data.wavel)

        def chi2(block):
            white, _ = block.whiten(
                block.predict(other, cvis), block.data(), None
            )
            return float(np.sum(white**2))

        projected = data.extras[0]  # every channel continuum and line
        broad = data.with_continuum(prior_width=1e4).extras[0]
        onp.testing.assert_allclose(chi2(broad), chi2(projected), rtol=1e-4)


def test_the_grey_scale_prior_is_stated_not_taken_from_the_data():
    data = OIData(read_oifits(build_hdulist(_tables()), extras=("flux",)))
    with pytest.raises(ValueError, match="State the prior"):
        model_loglike(_scene(), data)
    with pytest.raises(ValueError, match="positive"):
        data.with_flux_scale(scale=(0.0, 1.0))
    with pytest.raises(ValueError, match="flat prior"):
        data.with_flux_scale(scale=(3.0, onp.inf))
    # The same prior whatever the data's level: rescaling the data by 10
    # leaves the block's prior unchanged.
    stated = data.with_flux_scale(scale=SCALE_PRIOR)
    (block,) = stated.extras
    louder = block.rebuild(values=10 * onp.asarray(block.values))
    assert louder.scale == block.scale == SCALE_PRIOR
    onp.testing.assert_array_equal(louder.mu, block.mu)


def test_with_model_draws_the_marginalized_modes_at_their_widths():
    # The grey scale and the finite-prior VISPHI offsets are drawn from
    # their priors, so simulations have the likelihood's covariance.
    data = OIData(
        read_oifits(build_hdulist(_tables()), extras=("flux", "visphi"))
    ).with_continuum(lines=[LINE], prior_width=(0.3, 0.5))
    data = data.with_flux_scale(scale=SCALE_PRIOR)
    flux, visphi = data.extras
    scene = _scene()
    cvis = data._cvis(scene)
    prediction = flux.predict(scene, None)
    phases = np.angle(cvis)[visphi.sample]
    keys = jax.random.split(jax.random.PRNGKey(7), 400)
    scales, offsets = [], []
    first = onp.asarray(visphi.grid[0, 0])  # one baseline of one frame
    x = onp.asarray(visphi.basis[0, :, 1])
    for key in keys:
        draw = flux.simulated(prediction, None, None, key=key)
        scales.append(float(np.mean(draw.values / prediction)))
        draw = visphi.simulated(None, cvis, None, key=key)
        shift = onp.asarray(draw.values - phases)[first]
        offsets.append(onp.polyfit(x[: shift.size], shift, 1))
    onp.testing.assert_allclose(onp.std(scales), flux.widths[0], rtol=0.15)
    onp.testing.assert_allclose(onp.mean(scales), 1.0, atol=0.15)
    spread = onp.std(onp.asarray(offsets), axis=0)[::-1]  # (offset, slope)
    onp.testing.assert_allclose(spread, [0.3, 0.5], rtol=0.15)


def test_visphi_cached_cholesky_gives_the_same_whitening():
    import equinox as eqx

    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    data = data.with_continuum(lines=[LINE])
    (block,) = data.extras
    assert block.chol is not None  # computed once, at build
    uncached = eqx.tree_at(
        lambda b: b.chol, block, None, is_leaf=lambda x: x is None
    )
    other = _scene(dra=0.6, ddec=0.1)
    cvis = other.model(data.u, data.v, data.wavel)
    prediction = block.predict(other, cvis)
    for a, b in zip(
        block.whiten(prediction, block.data(), None),
        uncached.whiten(prediction, block.data(), None),
    ):
        onp.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # New errors refresh the factor.
    rng = onp.random.default_rng(4)
    noisier = block.with_errors(rng.uniform(0.005, 0.03, block.errors.shape))
    fresh = eqx.tree_at(
        lambda b: b.chol, noisier, None, is_leaf=lambda x: x is None
    )
    for a, b in zip(
        noisier.whiten(prediction, block.data(), None),
        fresh.whiten(prediction, block.data(), None),
    ):
        onp.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # The likelihood, its jit and its gradient go through the cache.
    loglike = eqx.filter_jit(model_loglike)
    onp.testing.assert_allclose(
        loglike(other, data), model_loglike(other, data), rtol=1e-6
    )
    grads = eqx.filter_jit(eqx.filter_grad(model_loglike))(other, data)
    assert bool(np.isfinite(grads.parts[1].dra))


def test_visphi_cached_factor_matches_a_float64_rebuild():
    # A block built in float32, then cast to float64 for a fit: the cached
    # factor (formed in float64) matches the factor the fit would build.
    import equinox as eqx

    from virgil._precision import cast_tree

    data = OIData(read_oifits(build_hdulist(_tables()), extras=("visphi",)))
    data = data.with_continuum(lines=[LINE])
    with jax.enable_x64(True):
        data64 = cast_tree(data, "float64")
        (block,) = data64.extras
        uncached = eqx.tree_at(
            lambda b: b.chol, block, None, is_leaf=lambda x: x is None
        )
        other = _scene(dra=0.6, ddec=0.1)
        cvis = other.model(data64.u, data64.v, data64.wavel)
        prediction = block.predict(other, cvis)
        for a, b in zip(
            block.whiten(prediction, block.data(), None),
            uncached.whiten(prediction, block.data(), None),
        ):
            onp.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_select_observables_drops_extras_and_frees_the_visphi_closures():
    data = (
        OIData(read_oifits(build_hdulist(_tables()), extras=ALL))
        .with_continuum(lines=[LINE])
        .with_flux_scale(scale=SCALE_PRIOR)
    )
    n_line = int(((WAVES >= LINE[0]) & (WAVES <= LINE[1])).sum())
    # Without closure phases, the differential phases keep their closure
    # part: all 6 baselines per line channel, not N - 1 = 3.
    alone = data.select(observables="visphi")
    (block,) = alone.extras
    assert not block.closure_free and not alone.has_phases
    assert alone.vis.size == 0
    assert block.n_independent == 6 * n_line == alone.n_independent
    # With the closure phases kept, nothing changes.
    both = data.select(observables=("phi", "visphi"))
    assert both.extras[0].closure_free
    assert both.n_independent == 36 + 3 * n_line
    spectra = data.select(observables=("vis", "flux"))
    assert [b.kind for b in spectra.extras] == ["flux"]
    assert spectra.has_model_covariance  # the flux scale survives
    assert bool(np.isfinite(model_loglike(_scene(), alone)))
    sim = alone.with_model(_scene(), key=jax.random.PRNGKey(3))
    chi2 = float(np.sum(whitened_residuals(_scene(), sim) ** 2))
    assert 0.3 * alone.n_residuals < chi2 < 2.0 * alone.n_residuals


def test_select_observables_takes_the_gains_with_the_visibilities():
    data = OIData(read_oifits(build_hdulist(_tables()))).with_gains(
        telescope=0.05
    )
    assert data.select(observables="vis").gains is not None
    cp_only = data.select(observables="phi")
    assert cp_only.gains is None and not cp_only.has_model_covariance
    offsets = data.with_closure_offsets(baseline=0.01)
    with pytest.raises(ValueError, match="closure-phase offsets"):
        offsets.select(observables="vis")
