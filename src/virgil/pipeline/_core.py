"""The base class of the pipelines: settings, stages, resume and ``run.json``."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import importlib.metadata
import json
import platform
import shutil
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable

from . import _io
from ._checks import worst_status

SCHEMA = _io.SCHEMA

# Files and folders a run owns, removed by run(resume=False).
_OWNED = (
    "run.json",
    "summary.json",
    "grids.h5",
    "samples.h5",
    "quicklook.ipynb",
    "data",
    "models",
    "plots",
    "stages",
)


class ConfigMismatchError(ValueError):
    """Resuming a run whose stored configuration differs from this one."""


@dataclasses.dataclass(frozen=True)
class Stage:
    """One step of a pipeline.

    Attributes
    ----------
    name : str
        Stage name, as in the pipeline's ``STAGES``.
    run : callable
        ``run(pipeline) -> dict``: does the work, writes its outputs, and
        returns the stage report (JSON-ready key numbers), which is written
        to ``stages/<name>/report.json``.
    outputs : tuple of str
        Paths, relative to the run folder, the stage must leave behind. A
        completed stage whose outputs are all present is skipped on resume.
    """

    name: str
    run: Callable
    outputs: tuple = ()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance():
    """Versions and platform of this process, for ``run.json``."""
    import jax

    return {
        "virgil": _version("virgil-astro"),
        "jax": _version("jax"),
        "jaxlib": _version("jaxlib"),
        "numpyro": _version("numpyro"),
        "numpy": _version("numpy"),
        "equinox": _version("equinox"),
        "zodiax": _version("zodiax"),
        "h5py": _version("h5py"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64),
    }


def _normalize(settings):
    """Settings as JSON would return them (tuples become lists)."""
    return json.loads(json.dumps(_io.clean_json(settings)))


class _Pipeline:
    """Shared machinery of the pipelines; subclasses define the analysis.

    A subclass sets ``NAME``, ``STABILITY``, ``STAGES``, ``_DEFAULTS`` and
    implements ``_stages()`` (one [`Stage`][virgil.pipeline.Stage] per
    name in ``STAGES``), ``_validate(settings)``, ``_default_model()`` and
    ``_summarize(reports)``.
    """

    NAME = ""
    STABILITY = "stable"
    STAGES: tuple = ()
    _DEFAULTS: dict = {}
    _warned_provisional = False

    def __init__(self, data, model=None, *, output="run", **settings):
        self.data = data
        self.model = self._default_model() if model is None else model
        self.output = Path(output)
        self.settings = self._resolve(settings)
        _io.require_extra("h5py")
        self.inputs = getattr(self, "_input_paths", None)
        if (
            self.STABILITY == "provisional"
            and not type(self)._warned_provisional
        ):
            type(self)._warned_provisional = True
            warnings.warn(
                f"{type(self).__name__} is provisional: its settings and "
                "outputs may change while the virgil APIs it calls settle.",
                FutureWarning,
                stacklevel=2,
            )

    # --- settings -------------------------------------------------------

    @classmethod
    def defaults(cls):
        """The default settings, as a new dict."""
        return copy.deepcopy(cls._DEFAULTS)

    @classmethod
    def _resolve(cls, overrides):
        unknown = sorted(set(overrides) - set(cls._DEFAULTS))
        if unknown:
            raise TypeError(
                f"{cls.__name__} got unknown settings {unknown}; valid "
                f"settings are {sorted(cls._DEFAULTS)}."
            )
        settings = cls.defaults()
        settings.update(overrides)
        settings = _normalize(settings)
        cls._validate(settings)
        return settings

    @classmethod
    def _validate(cls, settings):
        """Raise ``ValueError`` for invalid settings."""

    @classmethod
    def from_oifits(cls, paths, model=None, *, output="run", **settings):
        """Build a pipeline from OIFITS files.

        Parameters
        ----------
        paths : str, os.PathLike or list of them
            OIFITS files, read together into one ``OIData``.
        model : SourceModel, optional
            Model template (default: the pipeline's canonical model).
        output : str or os.PathLike, optional
            Run folder.
        **settings
            Settings overriding the defaults.
        """
        from ..oidata import OIData

        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        data = OIData(
            [str(p) for p in paths] if len(paths) > 1 else str(paths[0])
        )
        pipeline = cls.__new__(cls)
        pipeline._input_paths = [str(Path(p).resolve()) for p in paths]
        cls.__init__(pipeline, data, model, output=output, **settings)
        return pipeline

    @classmethod
    def from_config(cls, path, model=None, **overrides):
        """Build a pipeline from a JSON config file.

        The file holds ``{"data": [OIFITS paths], "output": folder,
        "settings": {...}}``; paths are relative to the file. Settings
        resolve as defaults, then the file, then ``overrides`` (which may
        also give ``output``).
        """
        path = Path(path)
        config = _io.read_json(path)
        unknown = set(config) - {"pipeline", "data", "output", "settings"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}.")
        if config.get("pipeline", cls.NAME) != cls.NAME:
            raise ValueError(
                f"{path} is a {config['pipeline']!r} config, not {cls.NAME!r}."
            )
        data = config["data"]
        data = [data] if isinstance(data, str) else data
        paths = [path.parent / p for p in data]
        settings = dict(config.get("settings", {}))
        output = overrides.pop("output", config.get("output", "run"))
        settings.update(overrides)
        if not Path(output).is_absolute() and "output" in config:
            output = path.parent / output
        return cls.from_oifits(paths, model, output=output, **settings)

    # --- running --------------------------------------------------------

    def _stages(self):
        raise NotImplementedError

    def _default_model(self):
        raise NotImplementedError

    def _summarize(self, reports):
        """``(sections, checks)`` from the stage reports so far."""
        raise NotImplementedError

    def _input_record(self):
        record = {"data_sha256": _io.data_fingerprint(self.data), "files": []}
        for p in self.inputs or []:
            record["files"].append({"path": p, "sha256": _io.sha256_file(p)})
        return record

    def _model_digest(self):
        try:
            spec, arrays = _io.model_spec(self.model)
        except _io.ModelSpecError:
            return {"class": type(self.model).__name__, "digest": None}
        return {
            "class": type(self.model).__name__,
            "digest": _io.spec_digest(spec, arrays),
        }

    def _check_resume(self, record):
        stored = self.defaults()
        stored.update(record.get("config", {}))
        stored = _normalize(stored)
        diffs = [
            f"{key}: stored {stored.get(key)!r}, now {self.settings.get(key)!r}"
            for key in sorted(set(stored) | set(self.settings))
            if stored.get(key) != self.settings.get(key)
        ]
        if record["inputs"]["data_sha256"] != _io.data_fingerprint(self.data):
            diffs.append("the data differ")
        if record.get("model") != self._model_digest():
            diffs.append("the model template differs")
        if record.get("pipeline") != self.NAME:
            diffs.append(
                f"pipeline {record.get('pipeline')!r} != {self.NAME!r}"
            )
        if diffs:
            raise ConfigMismatchError(
                f"{self.output} holds a run with a different configuration "
                f"({'; '.join(diffs)}). Use a new output folder, or "
                "run(resume=False) to start this one over."
            )

    def _write_record(self, record):
        record["updated"] = _now()
        _io.write_json(self.output / "run.json", record)

    def _new_record(self):
        return {
            "schema": SCHEMA,
            "pipeline": self.NAME,
            "class": type(self).__name__,
            "stability": self.STABILITY,
            "status": "running",
            "error": None,
            "config": self.settings,
            "model": self._model_digest(),
            "inputs": self._input_record(),
            "provenance": provenance(),
            "stages": {},
            "created": _now(),
        }

    def run(self, through=None, *, resume=True):
        """Run the stages in order, up to and including ``through``.

        Parameters
        ----------
        through : str, optional
            Last stage to run (default: all of ``STAGES``). A later
            ``run()`` continues from there.
        resume : bool, optional
            Skip stages already completed in ``output`` (default True). The
            stored configuration, data and model template must match this
            pipeline's, after filling in the current defaults for settings
            the stored run lacks; otherwise
            [`ConfigMismatchError`][virgil.pipeline.ConfigMismatchError] is raised.
            ``False`` removes the run's previous outputs and starts over.

        Returns
        -------
        Result
            The reloaded outputs (see [`load`][virgil.pipeline.load]).
        """
        names = list(self.STAGES)
        if through is not None and through not in names:
            raise ValueError(
                f"through must be one of {names}; got {through!r}."
            )
        selected = (
            names if through is None else names[: names.index(through) + 1]
        )
        run_file = self.output / "run.json"
        if not resume:
            self._clear()
        if resume and run_file.exists():
            record = _io.read_json(run_file)
            self._check_resume(record)
            record["provenance"] = provenance()
        else:
            record = self._new_record()
        record["status"], record["error"] = "running", None
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_record(record)

        stages = {stage.name: stage for stage in self._stages()}
        rerun = False
        for name in selected:
            stage = stages[name]
            entry = record["stages"].get(name, {})
            done = entry.get("status") == "complete" and all(
                (self.output / p).exists() for p in stage.outputs
            )
            if done and not rerun:
                continue
            rerun = True  # later stages depend on this one
            for later in names[names.index(name) :]:
                record["stages"].pop(later, None)
                stale = self.output / "stages" / later / "report.json"
                if stale.exists():
                    stale.unlink()
            started, t0 = _now(), time.perf_counter()
            record["stages"][name] = {"status": "running", "started": started}
            if name == names[-1]:
                # The last stage (the quicklook) reports on a run whose
                # analysis is complete; a failure here still marks it failed.
                record["status"] = "complete"
            self._write_record(record)
            try:
                report = stage.run(self) or {}
                _io.write_json(
                    self.output / "stages" / name / "report.json", report
                )
            except BaseException as err:
                record["stages"][name].update(
                    status="failed", seconds=time.perf_counter() - t0
                )
                record["status"] = "failed"
                record["error"] = {
                    "stage": name,
                    "type": type(err).__name__,
                    "message": str(err),
                    "traceback": traceback.format_exc(),
                }
                self._write_record(record)
                raise
            record["stages"][name] = {
                "status": "complete",
                "started": started,
                "finished": _now(),
                "seconds": round(time.perf_counter() - t0, 3),
                "outputs": list(stage.outputs),
            }
            self._write_summary(record)
            self._write_record(record)

        complete = all(
            record["stages"].get(n, {}).get("status") == "complete"
            for n in names
        )
        record["status"] = "complete" if complete else "partial"
        self._write_summary(record)
        record["files"] = self._file_hashes()
        self._write_record(record)
        return _io.load(self.output)

    def _file_hashes(self):
        return {
            str(p.relative_to(self.output)): _io.sha256_file(p)
            for p in sorted(self.output.rglob("*"))
            if p.is_file()
            and p.name != "run.json"
            and not p.name.startswith(".")
        }

    def _reports(self):
        reports = {}
        for name in self.STAGES:
            path = self.output / "stages" / name / "report.json"
            if path.exists():
                reports[name] = _io.read_json(path)
        return reports

    def _write_summary(self, record):
        sections, checks = self._summarize(self._reports())
        summary = {
            "schema": SCHEMA,
            "pipeline": self.NAME,
            "stability": self.STABILITY,
            "status": record["status"],
            "worst_check": worst_status(checks),
            **sections,
            "checks": [c.to_dict() for c in checks],
        }
        _io.write_json(self.output / "summary.json", summary)

    def _clear(self):
        for name in _OWNED:
            path = self.output / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    def __repr__(self):
        return (
            f"{type(self).__name__}(output={str(self.output)!r}, "
            f"model={type(self.model).__name__})"
        )
