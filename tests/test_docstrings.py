import doctest
import warnings
from pathlib import Path

import pytest

import virgil.grid_fit
import virgil.models


@pytest.mark.parametrize(
    "module",
    [virgil.models, virgil.grid_fit],
    ids=lambda m: m.__name__,
)
def test_docstring_examples_run(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0


SOURCES = sorted(Path(virgil.models.__file__).parent.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_compiles_without_warnings(path):
    # A LaTeX backslash ($\mu$) in a docstring that is not a raw string is
    # an invalid escape: Python warns on every fresh compile (as on OzSTAR,
    # where each pinned snapshot is compiled anew), and will one day refuse.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(), str(path), "exec")
