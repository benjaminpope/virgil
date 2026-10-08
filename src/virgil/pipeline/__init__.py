"""Stable, scriptable pipelines on top of virgil.

A pipeline runs a standard analysis in named stages with simple defaults
that keywords override, and writes a standard run folder: ``run.json``
(configuration, inputs, provenance, status), ``summary.json`` (key numbers
and quality checks), HDF5 grids and samples, plots, the processed data as
OIFITS, the best-fit model, and an executed quicklook notebook. The folder
reloads into real virgil objects with [`load`][virgil.pipeline.load].

    from virgil.pipeline import BinaryPipeline, load

    res = BinaryPipeline(data, output="runs/hd1234").run()
    res = load("runs/hd1234")          # later, in a fresh session
    res.summary["companion"]["sep_mas"], res.checks

The class names, constructor keywords, stage names, ``Result`` accessors,
the output schema ``virgil-pipeline-run-v1`` and the keys of
``summary.json`` form a stable contract: they change only by addition.
The pipelines need the ``pipeline`` extra
(``pip install 'virgil-astro[pipeline]'``).
"""

import importlib

from ._checks import Check
from ._core import ConfigMismatchError, Stage
from ._io import SCHEMA, Result, load

_LAZY = {"BinaryPipeline": ".binary"}

__all__ = [
    "SCHEMA",
    "BinaryPipeline",
    "Check",
    "ConfigMismatchError",
    "Result",
    "Stage",
    "load",
]


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
