"""Reading and writing pipeline outputs, and the [`Result`][virgil.pipeline.Result] of a run.

Every file is written atomically (to a temporary file in the same folder,
then ``os.replace``), so an interrupted run never leaves a half-written
output behind. Nothing is pickled: arrays go to HDF5 or ``.npz`` (read with
``allow_pickle=False``), everything else to JSON, and models are stored as
a recursive spec of public virgil classes (see :func:`model_spec`).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from ._checks import Check

SCHEMA = "virgil-pipeline-run-v1"


def require_extra(module):
    """Import a dependency of the ``pipeline`` extra, or explain how to get it."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"virgil.pipeline needs {module.split('.')[0]}, which is not "
            "installed. Install it with: pip install 'virgil-astro[pipeline]'"
        ) from err


# === ATOMIC WRITES ===


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temporary path that replaces ``path`` on success.

    The temporary file keeps ``path``'s suffix, so writers that infer the
    format from it (matplotlib, numpy) work.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix
    )
    os.close(fd)
    try:
        yield Path(tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _clean_float(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "tolist"):
        return clean_json(np.asarray(value).tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable.")


def _clean_float(value):
    return value if math.isfinite(value) else None


def clean_json(value):
    """``value`` with NaN and infinities replaced by ``None`` (strict JSON)."""
    if isinstance(value, float):
        return _clean_float(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)) or hasattr(value, "tolist"):
        return clean_json(np.asarray(value).tolist())
    return value


def write_json(path, record):
    """Write ``record`` as indented, strict JSON, atomically."""
    text = json.dumps(
        clean_json(record),
        indent=2,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
    )
    with atomic_path(path) as tmp:
        tmp.write_text(text + "\n", encoding="utf-8")


def read_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    """SHA-256 of a file's bytes, as hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_fingerprint(data):
    """SHA-256 of the arrays that define an ``OIData``'s observables."""
    h = hashlib.sha256()
    for name in (
        "u",
        "v",
        "wavel",
        "vis",
        "d_vis",
        "phi",
        "d_phi",
        "i_cps1",
        "i_cps2",
        "i_cps3",
        "vis_mat",
        "phi_mat",
    ):
        value = getattr(data, name, None)
        h.update(name.encode())
        if value is not None:
            array = np.ascontiguousarray(np.asarray(value, dtype=float))
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
    h.update(f"{data.vis_mode}|{data.v2_flag}|{data.cp_flag}".encode())
    return h.hexdigest()


# === HDF5 ===


def write_h5(path, groups, attrs=None):
    """Write nested dicts of arrays to HDF5, atomically.

    ``groups`` maps group name to ``{dataset: array}``; a dataset may be
    given as ``(array, {attr: value})``. ``attrs`` maps group name (``"/"``
    for the file) to attributes; values that are not scalars or strings
    are stored as JSON strings.
    """
    h5py = require_extra("h5py")
    with atomic_path(path) as tmp:
        with h5py.File(tmp, "w") as f:
            for group_name, datasets in groups.items():
                group = f.require_group(group_name)
                for name, value in datasets.items():
                    meta = {}
                    if isinstance(value, tuple):
                        value, meta = value
                    ds = group.create_dataset(name, data=np.asarray(value))
                    for key, item in meta.items():
                        ds.attrs[key] = _h5_attr(item)
            for group_name, items in (attrs or {}).items():
                target = (
                    f if group_name == "/" else f.require_group(group_name)
                )
                for key, item in items.items():
                    target.attrs[key] = _h5_attr(item)


def _h5_attr(value):
    if isinstance(value, (str, int, float, bool, np.number)):
        return value
    return json.dumps(clean_json(value), default=_json_default)


def read_h5(path, group):
    """Read every dataset of ``group`` into a dict of numpy arrays."""
    h5py = require_extra("h5py")
    with h5py.File(path, "r") as f:
        if group not in f:
            return {}
        return {name: np.asarray(ds) for name, ds in f[group].items()}


def read_h5_attrs(path, group="/"):
    """Attributes of ``group``, with JSON strings decoded."""
    h5py = require_extra("h5py")
    with h5py.File(path, "r") as f:
        target = f if group == "/" else f[group]
        out = {}
        for key, value in target.attrs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    value = json.loads(value)
            elif isinstance(value, np.generic):
                value = value.item()
            out[key] = value
        return out


# === MODELS: RECURSIVE SPECS OF PUBLIC CLASSES, NO PICKLES ===


class ModelSpecError(ValueError):
    """A model cannot be stored as, or rebuilt from, a spec of public classes."""


def _public_class(obj):
    cls = type(obj)
    module = cls.__module__
    if not module.startswith("virgil.") or cls.__name__.startswith("_"):
        raise ModelSpecError(
            f"{module}.{cls.__qualname__} is not a public virgil class."
        )
    if getattr(importlib.import_module(module), cls.__name__, None) is not cls:
        raise ModelSpecError(
            f"{module}.{cls.__qualname__} is not importable by name."
        )
    return {"module": module, "class": cls.__name__}


def model_spec(model):
    """A JSON spec and arrays that rebuild ``model`` without pickling.

    The spec is recursive: each equinox module becomes its public class
    (module and name) with its dataclass fields, nested modules become
    nested specs, and arrays are stored by reference into the returned
    ``arrays`` dict (written to ``values.npz``). Only public virgil classes
    and plain values (numbers, strings, booleans, ``None``, tuples, lists,
    dicts of these) are accepted; anything else raises
    :class:`ModelSpecError`.

    Returns
    -------
    tuple
        ``(spec, arrays)``.
    """
    import dataclasses

    import equinox as eqx

    arrays = {}

    def encode(value):
        if isinstance(value, eqx.Module):
            node = _public_class(value)
            node["fields"] = {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
            return {"$module": node}
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return {"$float": repr(value)}
        if isinstance(value, (tuple, list)):
            key = "$tuple" if isinstance(value, tuple) else "$list"
            return {key: [encode(v) for v in value]}
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ModelSpecError("only dicts with string keys are stored.")
            return {"$dict": {k: encode(v) for k, v in value.items()}}
        if (
            isinstance(value, (np.ndarray, np.generic))
            or hasattr(value, "__jax_array__")
            or type(value).__module__.startswith("jax")
        ):
            name = f"leaf_{len(arrays)}"
            kind = (
                "numpy"
                if isinstance(value, (np.ndarray, np.generic))
                else "jax"
            )
            arrays[name] = np.asarray(value)
            return {"$array": name, "kind": kind}
        raise ModelSpecError(
            f"cannot store a {type(value).__name__} in a model spec."
        )

    return encode(model), arrays


def model_from_spec(spec, arrays):
    """Rebuild a model from :func:`model_spec` output."""
    import dataclasses

    import jax.numpy as jnp

    def decode(node):
        if not isinstance(node, dict):
            return node
        if "$module" in node:
            info = node["$module"]
            cls = getattr(
                importlib.import_module(info["module"]), info["class"]
            )
            fields = {f.name for f in dataclasses.fields(cls)}
            stored = set(info["fields"])
            if fields != stored:
                raise ModelSpecError(
                    f"{info['class']} now has fields {sorted(fields)} but the "
                    f"stored model has {sorted(stored)}."
                )
            obj = object.__new__(cls)
            for name, value in info["fields"].items():
                object.__setattr__(obj, name, decode(value))
            return obj
        if "$array" in node:
            array = np.asarray(arrays[node["$array"]])
            return jnp.asarray(array) if node.get("kind") == "jax" else array
        if "$float" in node:
            return float(node["$float"])
        if "$tuple" in node:
            return tuple(decode(v) for v in node["$tuple"])
        if "$list" in node:
            return [decode(v) for v in node["$list"]]
        if "$dict" in node:
            return {k: decode(v) for k, v in node["$dict"].items()}
        raise ModelSpecError(f"unknown spec node {sorted(node)}.")

    return decode(spec)


def spec_digest(spec, arrays):
    """A short hash of a model spec and its arrays, for resume checks."""
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arrays[name]).tobytes())
    return h.hexdigest()


def save_model(folder, model, values, params):
    """Write ``manifest.json`` and ``values.npz`` for a model.

    ``values`` maps parameter path to value (the fitted parameters,
    including any ``"<path>_vec"`` angle vectors); they are stored under
    ``param/<path>`` beside the spec's arrays.
    """
    folder = Path(folder)
    try:
        spec, arrays = model_spec(model)
        error = None
    except ModelSpecError as err:
        spec, arrays, error = None, {}, str(err)
    manifest = {
        "schema": SCHEMA,
        "class": type(model).__name__,
        "module": type(model).__module__,
        "params": list(params),
        "value_keys": list(values),
        "spec": spec,
        "rebuildable": spec is not None,
        "error": error,
    }
    store = {f"spec/{k}": v for k, v in arrays.items()}
    store.update({f"param/{k}": np.asarray(v) for k, v in values.items()})
    with atomic_path(folder / "values.npz") as tmp:
        np.savez(tmp, **store)
    write_json(folder / "manifest.json", manifest)


def load_model_values(folder):
    """The stored parameter values of a saved model, as path → array."""
    with np.load(Path(folder) / "values.npz", allow_pickle=False) as f:
        return {
            k[len("param/") :]: np.asarray(f[k])
            for k in f.files
            if k.startswith("param/")
        }


def load_model(folder):
    """Rebuild a model saved with :func:`save_model`, values set."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if not manifest["rebuildable"]:
        raise ModelSpecError(
            f"The {manifest['class']} model was not stored as a spec "
            f"({manifest['error']}); its parameter values are available "
            "from Result.model_values()."
        )
    with np.load(folder / "values.npz", allow_pickle=False) as f:
        arrays = {
            k[len("spec/") :]: np.asarray(f[k])
            for k in f.files
            if k.startswith("spec/")
        }
    return model_from_spec(manifest["spec"], arrays)


# === PROCESSED DATA AS OIFITS ===


def oidata_tables(data):
    """OIFITS tables for :func:`virgil.oifits.write_oifits` from ``OIData``.

    Supports the common layout: squared visibilities with closure phases
    (needing station indices) or absolute phases, one or more wavelength
    channels per baseline, nothing flagged or projected and no extra
    observables. Returns ``(tables, None)``, or ``(None, reason)`` when the
    data cannot be written faithfully.
    """
    if data.vis_mat is not None or data.phi_mat is not None:
        return None, "projected observables (vis_mat/phi_mat)"
    if data.uv_grid is not None:
        return None, "uv-grid (AMIGO DISCO) data"
    if data.extras:
        return None, "extra observables"
    if data.gains is not None or data.phase_offsets is not None:
        return None, "calibration gains or closure offsets"
    if data.vis_index is not None or data.phi_index is not None:
        return None, "flagged samples"
    if not data.v2_flag or data.vis_mode != "v2":
        return None, "visibility amplitudes rather than V²"
    u = np.asarray(data.u, dtype=float)
    v = np.asarray(data.v, dtype=float)
    n = u.size
    wavel = np.broadcast_to(np.asarray(data.wavel, dtype=float), u.shape)
    waves = np.asarray(list(dict.fromkeys(wavel.tolist())))
    n_w = waves.size
    if n % n_w:
        return None, "irregular wavelength layout"
    n_b = n // n_w
    grid = wavel.reshape(n_b, n_w)
    if not (
        np.all(grid == waves[None, :])
        and np.all(u.reshape(n_b, n_w) == u.reshape(n_b, n_w)[:, :1])
        and np.all(v.reshape(n_b, n_w) == v.reshape(n_b, n_w)[:, :1])
    ):
        return None, "irregular wavelength layout"
    ub, vb = u.reshape(n_b, n_w)[:, 0], v.reshape(n_b, n_w)[:, 0]
    vis = np.asarray(data.vis, dtype=float)
    if vis.size != n:
        return None, "visibilities do not match the samples"
    if data.stations is not None:
        stations = np.asarray(data.stations, dtype=int).reshape(n_b, n_w, 2)[
            :, 0
        ]
    elif data.cp_flag:
        return None, "closure phases without station indices"
    else:
        stations = np.stack(
            [2 * np.arange(n_b) + 1, 2 * np.arange(n_b) + 2], axis=1
        )
    mjd = None
    if data.dt is not None and data.t_ref is not None:
        mjd = (np.asarray(data.dt, dtype=float) + data.t_ref).reshape(
            n_b, n_w
        )[:, 0]
    tables = {
        "info": {"TARGET": "PIPELINE", "INSTRUME": "VIRGIL"},
        "OI_WAVELENGTH": {"EFF_WAVE": waves, "EFF_BAND": 0.0},
        "OI_VIS2": {
            "VIS2DATA": vis.reshape(n_b, n_w),
            "VIS2ERR": np.asarray(data.d_vis, float).reshape(n_b, n_w),
            "UCOORD": ub,
            "VCOORD": vb,
            "STA_INDEX": stations,
        },
    }
    if mjd is not None:
        tables["OI_VIS2"]["MJD"] = mjd
        tables["info"]["MJD"] = float(mjd[0])
    phi = np.rad2deg(np.asarray(data.phi, dtype=float))
    d_phi = np.rad2deg(np.asarray(data.d_phi, dtype=float))
    if phi.size == 0:
        return tables, None
    if not data.cp_flag:
        if phi.size != n:
            return None, "absolute phases do not match the samples"
        tables["OI_VIS"] = {
            "VISPHI": phi.reshape(n_b, n_w),
            "VISPHIERR": d_phi.reshape(n_b, n_w),
            "VISAMP": np.full((n_b, n_w), np.nan),
            "VISAMPERR": np.full((n_b, n_w), np.nan),
            "UCOORD": ub,
            "VCOORD": vb,
            "STA_INDEX": stations,
        }
        if mjd is not None:
            tables["OI_VIS"]["MJD"] = mjd
        return tables, None
    legs = [np.asarray(i, dtype=int) for i in (data.i_cps1, data.i_cps2)]
    if phi.size % n_w:
        return None, "irregular closure-phase layout"
    n_t = phi.size // n_w
    rows = [leg.reshape(n_t, n_w) // n_w for leg in legs]
    cols = [leg.reshape(n_t, n_w) % n_w for leg in legs]
    if not all(
        np.all(r == r[:, :1]) and np.all(c == np.arange(n_w)[None, :])
        for r, c in zip(rows, cols)
    ):
        return None, "irregular closure-phase layout"
    b1, b2 = rows[0][:, 0], rows[1][:, 0]
    triangles = np.stack(
        [stations[b1, 0], stations[b1, 1], stations[b2, 1]], axis=1
    )
    tables["OI_T3"] = {
        "T3PHI": phi.reshape(n_t, n_w),
        "T3PHIERR": d_phi.reshape(n_t, n_w),
        "U1COORD": ub[b1],
        "V1COORD": vb[b1],
        "U2COORD": ub[b2],
        "V2COORD": vb[b2],
        "STA_INDEX": triangles,
    }
    if mjd is not None:
        tables["OI_T3"]["MJD"] = mjd[b1]
    return tables, None


# === RESULTS ===


def lobe_text(summary):
    """The preferred model's diameter lobes in one line, or ``None``.

    Each lobe is its fitted diameter and its Δχ² over the best lobe.
    """
    fit = summary.get("fit") or {}
    rows = (fit.get("models", {}).get(fit.get("best"), {})).get("lobes")
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["delta_chi2"])
    return "; ".join(
        f"{r['diam_mas']:.4g} mas "
        + ("(best)" if i == 0 else f"(Δχ² {r['delta_chi2']:.3g})")
        for i, r in enumerate(rows)
    )


class Result:
    """The outputs of a pipeline run, reloaded from its folder.

    Nothing is recomputed: every accessor reads the files the run wrote,
    and returns plain numbers, numpy arrays or real virgil objects that can
    be refined by hand.

    Attributes
    ----------
    path : pathlib.Path
        The run folder.
    run : dict
        The contents of ``run.json``: schema, status, resolved config,
        inputs, provenance and per-stage timings.
    summary : dict
        The contents of ``summary.json``: key numbers by stage, and the
        checks as dicts.
    """

    def __init__(self, path):
        self.path = Path(path)
        run_file = self.path / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(f"{run_file} does not exist.")
        self.run = read_json(run_file)
        if self.run.get("schema") != SCHEMA:
            raise ValueError(
                f"{run_file} has schema {self.run.get('schema')!r}; this "
                f"version of virgil reads {SCHEMA!r}."
            )
        summary_file = self.path / "summary.json"
        self.summary = read_json(summary_file) if summary_file.exists() else {}

    def __repr__(self):
        return (
            f"Result({str(self.path)!r}, pipeline={self.run['pipeline']!r}, "
            f"status={self.status!r})"
        )

    @property
    def status(self):
        """Run status: ``"running"``, ``"partial"``, ``"complete"`` or ``"failed"``."""
        return self.run["status"]

    @property
    def checks(self):
        """The quality checks, as a list of [`Check`][virgil.pipeline.Check]."""
        return [Check.from_dict(c) for c in self.summary.get("checks", [])]

    def describe(self):
        """The key numbers of the summary as short text, one per line."""
        lines = [f"{self.run['class']} run in {self.path}: {self.status}"]
        for section, values in self.summary.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    lines.append(f"  {section}.{key} = {value:.6g}")
        lobes = lobe_text(self.summary)
        if lobes:
            lines.append(f"  lobes: {lobes}")
        for check in self.checks:
            lines.append(f"  [{check.status}] {check.name}: {check.message}")
        caught = self.summary.get("warnings", [])
        if caught:
            stages = ", ".join(sorted({w["stage"] for w in caught}))
            lines.append(
                f"  {len(caught)} warning(s) recorded in the {stages} "
                "stage(s) (see the checks, or Result.summary['warnings'])"
            )
        return "\n".join(lines)

    @property
    def plots(self):
        """Paths of the plots the run wrote, sorted by name."""
        return sorted((self.path / "plots").glob("*.png"))

    def data(self):
        """The data actually fitted, as an ``OIData``, from ``data/processed.oifits``."""
        from ..oidata import OIData

        path = self.path / "data" / "processed.oifits"
        if not path.exists():
            reason = (
                self.run.get("stages", {})
                .get("load", {})
                .get("report", {})
                .get("processed_oifits_skipped")
            )
            raise FileNotFoundError(
                f"{path} was not written"
                + (f" ({reason})" if reason else "")
                + "; rebuild the data from the inputs listed in "
                "Result.run['inputs']."
            )
        return OIData(path)

    def model(self, name=None):
        """A fitted model, a real virgil object with its values set.

        Parameters
        ----------
        name : str, optional
            For a run that fitted several models (``StarPipeline``), the
            model to rebuild; by default the preferred one, ``"best"``.
        """
        return load_model(self._model_folder(name))

    def model_values(self, name=None):
        """The fitted parameter values, as path → numpy array (see ``model``)."""
        return load_model_values(self._model_folder(name))

    def samples(self, group_by_chain=True, name=None):
        """Posterior samples by site, shaped ``(chain, draw)`` (or flat).

        ``name`` selects one of several fitted models, as in ``model``.
        """
        samples = read_h5(
            self._stage_file("samples.h5", "posterior"),
            self._group("posterior", name),
        )
        if group_by_chain:
            return samples
        return {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}

    def sample_stats(self, name=None):
        """NUTS diagnostics per transition, shaped ``(chain, draw)``."""
        return read_h5(
            self._stage_file("samples.h5", "posterior"),
            self._group("sample_stats", name),
        )

    def grid(self):
        """Grid results: ``"axes"`` (usable as ``grid=``) plus one map per name."""
        stage = "fit" if self.run["pipeline"] == "star" else "search"
        path = self._stage_file("grids.h5", stage)
        out = {"axes": read_h5(path, "axes")}
        for group in ("search", "limits", "scan"):
            out.update(read_h5(path, group))
        order = read_h5_attrs(path).get("axis_order")
        if order:
            out["axes"] = {k: out["axes"][k] for k in order}
        return out

    def fit_result(self, name=None):
        """The MAP fit as a virgil ``FitResult``, e.g. to continue with ``fit``.

        ``name`` selects one of several fitted models, as in ``model``.
        """
        from ..fitting import FitResult

        folder = self._model_folder(name)
        info = read_json(Path(folder) / "info.json")
        return FitResult(
            model=load_model(folder),
            values=load_model_values(folder),
            info=info,
        )

    def _model_folder(self, name):
        return self._stage_file(
            "models/best" if name is None else f"models/{name}", "fit"
        )

    @staticmethod
    def _group(group, name):
        return group if name is None else f"{group}_{name}"

    def _stage_file(self, relative, stage):
        path = self.path / relative
        if not path.exists():
            done = self.run.get("stages", {}).get(stage, {}).get("status")
            raise FileNotFoundError(
                f"{path} does not exist: the {stage!r} stage has status "
                f"{done or 'not run'}; run the pipeline through it first."
            )
        return path


def load(path):
    """Reload a pipeline run from its folder, without recomputing anything.

    Parameters
    ----------
    path : str or os.PathLike
        The run folder (the pipeline's ``output``).

    Returns
    -------
    Result
    """
    return Result(path)
