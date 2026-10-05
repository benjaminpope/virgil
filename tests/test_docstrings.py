import doctest
import importlib
import warnings
from pathlib import Path

import pytest

import virgil.models

SOURCES = sorted(Path(virgil.models.__file__).parent.rglob("*.py"))


def _module_names():
    """Every module of the package, so new examples are checked."""
    root = Path(virgil.models.__file__).parent
    names = []
    for path in SOURCES:
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        # The ImPlaneIA-derived tools need optional dependencies.
        if parts and parts[0] == "legacy":
            continue
        names.append(".".join(("virgil", *parts)))
    return names


@pytest.mark.parametrize("name", _module_names())
def test_docstring_examples_run(name):
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_compiles_without_warnings(path):
    # A LaTeX backslash ($\mu$) in a docstring that is not a raw string is
    # an invalid escape: Python warns on every fresh compile (as on OzSTAR,
    # where each pinned snapshot is compiled anew), and will one day refuse.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(), str(path), "exec")
