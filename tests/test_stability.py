"""Stability tiers: provisional names exist and say so in their docs."""

import importlib
import pkgutil

import pytest

import virgil
from virgil import _stability


def _modules_with_provisional():
    found = []
    for info in pkgutil.walk_packages(virgil.__path__, "virgil."):
        if info.name.startswith("virgil.legacy") or info.name.endswith(
            ("cli", "__main__")
        ):
            continue
        module = importlib.import_module(info.name)
        if hasattr(module, "__provisional__"):
            found.append(module)
    return found


MODULES = _modules_with_provisional()
NAMES = [
    (module.__name__, name)
    for module in MODULES
    for name in module.__provisional__
]


# The tiers, pinned. Changing a tier, or renaming a provisional name, means
# changing this table, docs/stability.md and a changelog fragment together.
EXPECTED = {
    "virgil.epochs": {"*"},
    "virgil.aliases": {"*"},
    "virgil.arcs": {"*"},
    "virgil.night_summaries": {"*"},
    "virgil.orbit_cells": {"*"},
    "virgil.orbit_search": {"*"},
    "virgil.tempering": {"*"},
    "virgil.svi": {"*"},
    "virgil.smc": {"*"},
    "virgil.metrics": {"*"},
    "virgil.radio": {"*"},
    "virgil.radio_clean": {"*"},
    "virgil.radio_toeplitz": {"*"},
    "virgil.crossval": {"*"},
    "virgil.pipeline": {"*"},
    "virgil.models": {
        "MultipoleDisk",
        "ThinRing",
        "GaussianRing",
        "RadialPowerLaw",
        "ExpTail",
        "Gap",
        "RadialSum",
        "RadialLinear",
        "RadialPower",
        "RadialGP",
        "RadialGPFunction",
        "MultipoleKernel",
        "Harmonics",
        "VonMises",
        "Cone",
        "PowerLawHeight",
        "multipole_disk_cvis",
        "gp_priors",
    },
    "virgil.fields": {"MultipoleField", "multipole_field_priors"},
    "virgil.ensemble": {
        "evidence_grid",
        "gp_members",
        "GPFit",
        "EvidenceGrid",
    },
    "virgil.imaging": {"clean", "CleanResult"},
    "virgil.fitting": {"gauss_newton_diag"},
    "virgil.plotting": {"plot_field_draws"},
    "virgil.likelihood": {
        "group_outlier_mixture",
        "group_outlier_probabilities",
    },
}


def test_provisional_lists_are_pinned():
    found = {
        module.__name__: set(module.__provisional__) for module in MODULES
    }
    assert found == EXPECTED


def test_provisional_is_a_tuple_of_strings():
    for module in MODULES:
        names = module.__provisional__
        assert isinstance(names, tuple), module.__name__
        assert all(isinstance(name, str) for name in names)
        assert len(set(names)) == len(names), module.__name__
        if "*" in names:
            assert names == ("*",), module.__name__


@pytest.mark.parametrize(("module", "name"), NAMES)
def test_provisional_name_exists_and_says_so(module, name):
    mod = importlib.import_module(module)
    if name == "*":
        obj = mod  # the whole module: its docstring carries the notice
    else:
        assert hasattr(mod, name), f"{module}.{name} is listed as provisional"
        obj = getattr(mod, name)
    doc = obj.__doc__ or ""
    # Docstrings are indented inside classes and functions.
    flat = "\n".join(line.strip() for line in doc.splitlines())
    expected = "\n".join(
        line.strip() for line in _stability.ADMONITION.splitlines()
    )
    assert expected in flat, f"{module}.{name} lacks the Provisional notice"


def test_pipelines_are_provisional():
    from virgil.pipeline import BinaryPipeline, StarPipeline
    from virgil.pipeline._core import _Pipeline

    assert _Pipeline.STABILITY == "provisional"
    for cls in (BinaryPipeline, StarPipeline):
        assert cls.STABILITY == "provisional"
        assert _stability.ADMONITION.splitlines()[0].strip() in cls.__doc__


def test_stability_module_imports_nothing():
    # ``scripts/sync_tutorial_docs.py`` loads it by path, without JAX.
    import ast
    from pathlib import Path

    tree = ast.parse(Path(_stability.__file__).read_text())
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]


PAGES = [
    "docs/orbit_fitting.md",
    "docs/multipole_disk_rings.md",
    "docs/imaging_clean.md",
    "docs/imaging_ensemble.md",
    "docs/variational_inference.md",
    "docs/pipeline.md",
    "docs/api/pipeline.md",
    "docs/api/models/classes/multipole_disk.md",
    "docs/radio_clean.md",
    "docs/api/radio_clean.md",
    "docs/api/radio_toeplitz.md",
    "docs/stability.md",
]


@pytest.mark.parametrize("page", PAGES)
def test_provisional_pages_carry_the_notice(page):
    from pathlib import Path

    root = Path(virgil.__file__).parents[2]
    text = (root / page).read_text()
    assert _stability.ADMONITION in text, page
