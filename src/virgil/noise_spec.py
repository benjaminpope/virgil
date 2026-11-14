"""Declarative instrument noise specifications (``virgil.noise_spec/1``).

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

An instrument package measures departures from independent quoted errors and
emits a plain dictionary, a JSON file, or an object implementing
``to_noise_spec() -> dict``. This module validates the specification and
applies it to an [`OIData`][virgil.oidata.OIData] through its existing
[`with_error_scale`][virgil.oidata.OIData.with_error_scale],
[`with_gains`][virgil.oidata.OIData.with_gains] and
[`with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets] methods;
see [conventions](../conventions.md#instrument-noise-specs).
"""

from collections.abc import Mapping
import copy
import inspect
import json
import math
from numbers import Real
import os
from pathlib import Path
import warnings

import equinox as eqx
import numpy as onp

from ._deprecate import external_stacklevel
from .observables import KINDS, DifferentialPhase, in_ranges

__all__ = [
    "apply_noise_spec",
    "instrument_match",
    "load_noise_spec",
    "validate_noise_spec",
]

__provisional__ = (
    "load_noise_spec",
    "validate_noise_spec",
    "apply_noise_spec",
    "instrument_match",
)

SCHEMA = "virgil.noise_spec/1"
SCHEMA_PREFIX = "virgil.noise_spec/"
WHITE_KEYS = ("vis", "phi", *KINDS)
GAIN_KEYS = ("telescope", "baseline", "chromatic")
OFFSET_KEYS = ("triangle", "baseline")
COMMON_MODE_KEYS = ("visphi",)
BOUND_VALUES = ("lower", "estimate")
_TOP_LEVEL_KEYS = (
    "schema",
    "instrument",
    "white",
    "gains",
    "closure_offsets",
    "common_mode",
    "provenance",
)
_GAIN_SPEC = (
    "gains",
    GAIN_KEYS,
    1.0,
    0.1,
    "on log |V|",
    "likely percent or V² unit slip "
    "(log |V| width is half the fractional V² width)",
)
_OFFSET_SPEC = (
    "closure_offsets",
    OFFSET_KEYS,
    0.3,
    0.02,
    "rad",
    "likely deg-vs-rad unit slip (closure_offsets widths must be in radians)",
)
_COMMON_MODE_SPEC = (
    "common_mode",
    COMMON_MODE_KEYS,
    0.5,
    0.05,
    "rad",
    "likely deg-vs-rad unit slip (common_mode widths must be in radians)",
)


def _finite_real(val):
    """Return True when ``val`` is a non-boolean finite real number."""
    return (
        not isinstance(val, bool)
        and isinstance(val, Real)
        and math.isfinite(float(val))
    )


def _validate_width_group(
    raw,
    group_name,
    allowed_keys,
    max_width,
    warn_above,
    unit_label,
    slip_hint,
    problems,
    warns,
):
    """Validate an optional group of nullable non-negative 1-sigma widths."""
    out = dict.fromkeys(allowed_keys, None)
    if raw is None:
        return out
    if not isinstance(raw, Mapping):
        problems.append(
            f"{group_name} must be a mapping or None, got {type(raw).__name__}"
        )
        return out
    unknown = sorted(set(raw) - set(allowed_keys), key=repr)
    if unknown:
        problems.append(
            f"{group_name} has unknown keys {unknown}; allowed keys are "
            f"{list(allowed_keys)}"
        )
    for key in allowed_keys:
        val = raw.get(key)
        if val is None:
            continue
        if not _finite_real(val) or not (0.0 <= float(val) <= max_width):
            problems.append(
                f"{group_name}.{key} must be None or a finite width in "
                f"[0, {max_width:g}] {unit_label}, got {val!r}"
            )
            continue
        fval = float(val)
        out[key] = fval
        if fval > warn_above:
            warns.append(
                f"{group_name}.{key}={fval:g} {unit_label} exceeds "
                f"{warn_above:g} {unit_label}; {slip_hint}."
            )
    return out


def validate_noise_spec(spec):
    """Validate and normalize a ``virgil.noise_spec/1`` mapping.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Checks the major schema version, rejects unknown keys outside
    ``provenance``, verifies value bounds across every group, and warns on
    plausible unit slips without mutating ``spec``.

    Parameters
    ----------
    spec : Mapping
        Raw noise specification dictionary.

    Returns
    -------
    dict
        Normalized copy with ``schema``, ``instrument``, ``white``, ``gains``,
        ``closure_offsets``, ``common_mode`` and ``provenance``.

    Raises
    ------
    ValueError
        Listing every validation problem found in ``spec``.
    """
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"Noise spec must be a mapping, got {type(spec).__name__}."
        )
    problems, warns = [], []

    schema = spec.get("schema")
    if schema is None:
        problems.append(f"missing required 'schema' ({SCHEMA!r})")
    elif not isinstance(schema, str):
        problems.append(
            f"schema must be the string {SCHEMA!r}, got {schema!r}"
        )
    elif schema != SCHEMA:
        kind = (
            "schema version" if schema.startswith(SCHEMA_PREFIX) else "schema"
        )
        problems.append(f"unsupported {kind} {schema!r}; expected {SCHEMA!r}")

    unknown_top = sorted(set(spec) - set(_TOP_LEVEL_KEYS), key=repr)
    if unknown_top:
        problems.append(
            f"unknown top-level keys {unknown_top}; allowed keys are "
            f"{list(_TOP_LEVEL_KEYS)}"
        )

    raw_inst = spec.get("instrument")
    instrument = None
    if raw_inst is not None:
        if isinstance(raw_inst, str) and raw_inst.strip():
            instrument = raw_inst.strip()
        else:
            problems.append(
                "instrument must be a non-empty string or None, got "
                f"{raw_inst!r}"
            )

    raw_white, white = spec.get("white"), {}
    if raw_white is not None:
        if not isinstance(raw_white, Mapping):
            problems.append(
                f"white must be a mapping or None, got "
                f"{type(raw_white).__name__}"
            )
        else:
            unknown_w = sorted(set(raw_white) - set(WHITE_KEYS), key=repr)
            if unknown_w:
                problems.append(
                    f"white has unknown observable keys {unknown_w}; "
                    f"allowed keys are {list(WHITE_KEYS)}"
                )
            for key in (k for k in WHITE_KEYS if k in raw_white):
                val = raw_white[key]
                if _finite_real(val) and float(val) > 0.0:
                    white[key] = float(val)
                else:
                    problems.append(
                        f"white.{key} must be a finite scale > 0, got {val!r}"
                    )

    gains = _validate_width_group(
        spec.get("gains"), *_GAIN_SPEC, problems, warns
    )
    closure_offsets = _validate_width_group(
        spec.get("closure_offsets"), *_OFFSET_SPEC, problems, warns
    )
    common_mode = _validate_width_group(
        spec.get("common_mode"), *_COMMON_MODE_SPEC, problems, warns
    )

    raw_prov, provenance = spec.get("provenance"), {}
    if not isinstance(raw_prov, Mapping):
        problems.append(
            "provenance must be a mapping with non-empty 'run' and 'date' "
            f"strings, got {raw_prov!r}"
        )
    else:
        for req in ("run", "date"):
            val = raw_prov.get(req)
            if not isinstance(val, str) or not val.strip():
                problems.append(
                    f"provenance.{req} must be a non-empty string, got {val!r}"
                )
        if "bound" in raw_prov and raw_prov["bound"] not in BOUND_VALUES:
            problems.append(
                f"provenance.bound must be one of {list(BOUND_VALUES)}, got "
                f"{raw_prov['bound']!r}"
            )
        provenance = copy.deepcopy(dict(raw_prov))

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(f"Invalid noise spec:\n  - {joined}")
    level = external_stacklevel()
    for msg in warns:
        warnings.warn(msg, UserWarning, stacklevel=level)
    return {
        "schema": SCHEMA,
        "instrument": instrument,
        "white": white,
        "gains": gains,
        "closure_offsets": closure_offsets,
        "common_mode": common_mode,
        "provenance": provenance,
    }


def load_noise_spec(obj):
    """Load and validate a ``virgil.noise_spec/1`` specification.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Parameters
    ----------
    obj : Mapping, str, os.PathLike, or object with ``to_noise_spec()``
        A specification dictionary, a path to its JSON file, or an object
        whose ``to_noise_spec()`` method returns the dictionary.

    Returns
    -------
    dict
        Normalized noise specification from
        [`validate_noise_spec`][virgil.noise_spec.validate_noise_spec].

    Raises
    ------
    TypeError
        If ``obj`` is none of the supported input forms.
    ValueError
        If the resolved specification fails validation.
    """
    if isinstance(obj, Mapping):
        raw = obj
    elif isinstance(obj, (str, os.PathLike)):
        raw = json.loads(Path(obj).read_text(encoding="utf-8"))
    elif not isinstance(obj, type) and callable(
        getattr(obj, "to_noise_spec", None)
    ):
        raw = obj.to_noise_spec()
    else:
        raise TypeError(
            "Expected a noise-spec mapping, a path to a JSON file, or an "
            f"object with to_noise_spec(), got {type(obj).__name__}."
        )
    return validate_noise_spec(raw)


def instrument_match(data, spec):
    """Check whether ``data``'s recorded instrument matches ``spec``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Compares ``data``'s ``insname`` (when present and non-empty) against
    ``spec["instrument"]`` by case-insensitive prefix/substring matching so
    generic instrument names such as ``"GRAVITY"`` or ``"MATISSE"`` match
    channel names such as ``"GRAVITY_SC"`` or ``"MATISSE_LM"``. Emits a
    ``UserWarning`` and returns ``False`` on mismatch, never raising.

    Parameters
    ----------
    data : OIData or Mapping
        Dataset (or record) optionally carrying an ``insname`` attribute or
        key.
    spec : Mapping
        Noise specification dictionary.

    Returns
    -------
    bool
        ``True`` when the instruments match or either side is unrecorded;
        ``False`` when they disagree.
    """
    spec_inst = spec.get("instrument") if isinstance(spec, Mapping) else None
    if spec_inst is None or not str(spec_inst).strip():
        return True
    spec_clean = str(spec_inst).strip()

    raw_ins = (
        data.get("insname")
        if isinstance(data, Mapping)
        else getattr(data, "insname", None)
    )
    if raw_ins is None:
        return True
    if isinstance(raw_ins, str):
        names = [raw_ins.strip()] if raw_ins.strip() else []
    elif isinstance(raw_ins, (tuple, list, set, frozenset, onp.ndarray)):
        names = [
            str(x).strip() for x in raw_ins if x is not None and str(x).strip()
        ]
    else:
        single = str(raw_ins).strip()
        names = [single] if single else []
    if not names:
        return True

    target = spec_clean.lower()
    matched = all(
        target in name.lower() or name.lower() in target for name in names
    )
    if not matched:
        shown = names[0] if len(names) == 1 else names
        warnings.warn(
            f"Noise spec instrument {spec_clean!r} does not match data "
            f"insname {shown!r}.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    return matched


def apply_noise_spec(data, spec):
    """Apply a validated ``virgil.noise_spec/1`` specification to ``data``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Validates ``spec``, checks [`instrument_match`][virgil.noise_spec.instrument_match],
    rescales quoted uncertainties on the observables present in ``data``, and
    attaches marginalized gains, closure-phase offsets and differential-phase
    common modes. Refuses to stack gains, closure-phase offsets or
    differential-phase modes onto data that already carry them.

    Parameters
    ----------
    data : OIData
        Prepared interferometric dataset without pre-existing gains or offsets
        for any group active in ``spec``.
    spec : Mapping
        Noise specification dictionary conforming to ``virgil.noise_spec/1``.

    Returns
    -------
    OIData
        Dataset with the specification's white scales and nuisance modes
        applied.

    Raises
    ------
    ValueError
        If ``spec`` is invalid, if ``data`` already carries a nuisance group
        that ``spec`` applies, or if ``data`` is incompatible with the
        requested nuisance modes.
    """
    spec = validate_noise_spec(spec)
    gain_kwargs = {
        k: v for k, v in spec["gains"].items() if v is not None and v > 0.0
    }
    offset_kwargs = {
        k: v
        for k, v in spec["closure_offsets"].items()
        if v is not None and v > 0.0
    }
    visphi_width = spec["common_mode"]["visphi"]
    visphi_active = visphi_width is not None and visphi_width > 0.0

    if gain_kwargs and data.gains is not None:
        raise ValueError(
            "Data already carry calibration gains (data.gains is not None); "
            "apply_noise_spec refuses to stack gains."
        )
    if offset_kwargs and data.phase_offsets is not None:
        raise ValueError(
            "Data already carry closure-phase offsets "
            "(data.phase_offsets is not None); apply_noise_spec refuses to "
            "stack offsets."
        )
    visphi_blocks = [b for b in data.extras if b.kind == "visphi"]
    if visphi_active and any(
        getattr(b, "modes", None) is not None
        or getattr(b, "raw_modes", None) is not None
        for b in visphi_blocks
    ):
        raise ValueError(
            "Data already carry differential-phase modes; apply_noise_spec "
            "refuses to stack common_mode.visphi."
        )

    instrument_match(data, spec)

    present_kinds = {b.kind for b in data.extras}
    if data.vis.size:
        present_kinds.add("vis")
    if data.phi.size:
        present_kinds.add("phi")
    present_white = {
        k: v for k, v in spec["white"].items() if k in present_kinds
    }

    out = data
    if present_white:
        out = out.with_error_scale(present_white)
    if gain_kwargs:
        out = out.with_gains(**gain_kwargs)
    if offset_kwargs:
        out = out.with_closure_offsets(**offset_kwargs)
    if visphi_active and visphi_blocks:
        if (
            "modes"
            not in inspect.signature(DifferentialPhase.build).parameters
        ):
            raise NotImplementedError(
                "Applying common_mode.visphi requires "
                "DifferentialPhase.build(..., modes=...)."
            )
        new_extras = []
        for b in out.extras:
            if b.kind != "visphi":
                new_extras.append(b)
                continue
            wavel = onp.asarray(b.wavel)
            row = onp.asarray(b.row)
            if b.continuum is None and b.lines is None:
                line = onp.ones(wavel.size, dtype=bool)
            elif b.lines is None:
                line = ~in_ranges(wavel, b.continuum)
            else:
                line = in_ranges(wavel, b.lines)
            mode_rows = [
                onp.where((row == r) & line, visphi_width, 0.0)
                for r in onp.unique(row)
                if ((row == r) & line).any()
            ]
            if mode_rows:
                b = b.rebuild(modes=onp.stack(mode_rows, axis=0))
            new_extras.append(b)
        out = eqx.tree_at(lambda d: d.extras, out, tuple(new_extras))
    return out
