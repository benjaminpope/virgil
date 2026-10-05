"""The public layout of the package: top-level names and module boundaries."""

import ast
import sys
from pathlib import Path

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import pytest

import virgil
from virgil import BinaryModelCartesian, PointSource, System
from virgil.likelihood import posterior_predictive_summary
from virgil.plotting import plot_oidata_overview
from tests._test_data import oidata

SRC = Path(virgil.__file__).parent


def test_everyday_names_are_top_level():
    for name in virgil.__all__:
        assert hasattr(virgil, name), name


def test_version_is_the_installed_distribution_version():
    from importlib.metadata import version

    assert virgil.__version__ == version("virgil-astro")


def test_optional_plotting_dependencies_fail_with_install_hint(monkeypatch):
    # pandas and chainconsumer are in the "plots" extra, imported only by
    # the helpers that use them; without them the error says what to do.
    from virgil.plotting import (
        diagnostics_table_from_samples,
        plot_chainconsumer_diagnostics,
    )

    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setitem(sys.modules, "chainconsumer", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[plots\]"):
        diagnostics_table_from_samples(
            {"dra": [1.0], "ddec": [1.0], "flux": [1.0]}
        )
    with pytest.raises(ImportError, match=r"virgil-astro\[plots\]"):
        plot_chainconsumer_diagnostics({}, columns=["dra"])


def test_legacy_simbad_lookup_fails_with_install_hint(monkeypatch):
    from virgil.legacy import oifits_implaneia

    monkeypatch.setitem(sys.modules, "astroquery.simbad", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[legacy\]"):
        oifits_implaneia._simbad()


def test_limits_do_not_depend_on_grid_fit():
    # Both share the private _grid machinery instead.
    tree = ast.parse((SRC / "limits.py").read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "grid_fit" not in modules
    assert "_grid" in modules


def test_legacy_tools_are_not_imported_by_default():
    tree = ast.parse((SRC / "__init__.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "legacy" not in imported


def test_posterior_predictive_summary_works_for_any_model():
    rng = onp.random.default_rng(0)
    samples = {
        "comp.dra": 120.0 + rng.normal(size=20),
        "comp.ddec": -80.0 + rng.normal(size=20),
        "comp.flux": 2e-3 + 1e-4 * rng.normal(size=20),
    }
    template = System(star=PointSource(), comp=PointSource(flux=1e-3))
    summary = posterior_predictive_summary(samples, template, oidata)
    binary = posterior_predictive_summary(
        {key.split(".")[1]: value for key, value in samples.items()},
        BinaryModelCartesian,
        oidata,
    )
    assert summary["vis_mean"].shape == oidata.vis.shape
    assert summary["phi_std"].shape == oidata.phi.shape
    assert np.allclose(summary["phi_mean"], binary["phi_mean"], atol=1e-5)


def test_oidata_overview_has_three_panels():
    fig, axes = plot_oidata_overview(oidata)
    assert len(axes) == 3
    assert axes[2].get_ylabel() == "Closure phase (deg)"
    plt.close(fig)
