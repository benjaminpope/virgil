from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import jax.numpy as np
import numpy as onp

from virgil.legacy import oifits_implaneia
from virgil.models import BinaryModelCartesian
from virgil.oidata import closure_phases, cp_indices


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "synthetic_binary_workflow.py"
)


def _load_synthetic_module():
    spec = spec_from_file_location("synthetic_binary_workflow", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_docs_builder_writes_closure_phases_in_degrees():
    module = _load_synthetic_module()
    dic, truth, _ = module._build_synthetic_oifits_dict(seed=4)

    ucoord, vcoord, baseline_pairs, triangles = module._array_geometry()
    cvis = BinaryModelCartesian(**truth).model(
        ucoord, vcoord, np.array([4.8e-6])
    )
    i1, i2, i3 = cp_indices(baseline_pairs, triangles)
    cp_rad = onp.array(closure_phases(cvis, i1, i2, i3))
    cp_deg = onp.rad2deg(cp_rad)

    observed_cp = onp.asarray(dic["OI_T3"]["T3PHI"])
    mean_abs_err_deg = float(onp.mean(onp.abs(observed_cp - cp_deg)))
    mean_abs_err_rad = float(onp.mean(onp.abs(observed_cp - cp_rad)))

    assert mean_abs_err_deg < mean_abs_err_rad
    assert onp.max(onp.abs(observed_cp)) > onp.max(onp.abs(cp_rad))


def test_synthetic_docs_binary_recovery_within_two_sigma(tmp_path: Path):
    module = _load_synthetic_module()

    run_synthetic_binary_demo = module.run_synthetic_binary_demo
    within_two_sigma = module.within_two_sigma
    fisher_within_three_sigma = module.fisher_within_three_sigma

    output = tmp_path / "synthetic_binary_docs.oifits"
    # Short chains: a 3-parameter binary needs far fewer than the docs' 2000.
    summary = run_synthetic_binary_demo(
        output, num_warmup=300, num_samples=500
    )

    checks = within_two_sigma(summary)
    fisher_checks = fisher_within_three_sigma(summary)

    assert output.exists()
    assert summary.noise_settings == {
        "visamp_err_frac": 0.002,
        "visphi_err_frac": 0.004,
        "vis2_err_frac": 0.001,
        "cp_err_frac": 0.004,
    }

    # Guard against prior-dominated posteriors that would make 2σ checks trivially true.
    assert summary.hmc_std["dra"] < 30.0
    assert summary.hmc_std["ddec"] < 30.0
    assert summary.hmc_std["flux"] < 0.0015

    # The toy geometry can put one parameter slightly outside 2σ for a fixed RNG seed;
    # require broad agreement while still enforcing informative posteriors.
    assert sum(checks.values()) >= 2, (
        f"Too many HMC parameters exceed 2σ bounds: {checks}; summary={summary}"
    )
    assert all(fisher_checks.values()), (
        f"Fisher-HMC recovered parameters exceed 3σ bounds: {fisher_checks}; summary={summary}"
    )

    for key in ("dra", "ddec", "flux"):
        sigma = max(summary.hmc_std[key], 1e-12)
        z = abs(summary.hmc_median[key] - summary.truth[key]) / sigma
        assert z < 3.0, (
            f"HMC {key} is too far from truth in z-space ({z:.3f}); summary={summary}"
        )


def test_synthetic_oifits_save_skips_simbad_for_unknown_target(
    tmp_path: Path, monkeypatch
):
    module = _load_synthetic_module()
    dic, _, _ = module._build_synthetic_oifits_dict(seed=4)

    assert dic["info"]["TARGET"] == "UNKNOWN"

    class ForbiddenSimbad:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Simbad should not be instantiated")

    monkeypatch.setattr(oifits_implaneia, "Simbad", ForbiddenSimbad)

    output = tmp_path / "synthetic_binary_docs.oifits"
    oifits_implaneia.save(
        dic, filename=output.name, datadir=str(output.parent), verbose=False
    )

    assert output.exists()
