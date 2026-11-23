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
import json
import math
from numbers import Real
import os
from pathlib import Path
import warnings

import numpy as onp

from ._deprecate import external_stacklevel
from .observables import KINDS

__all__ = [
    "apply_noise_spec",
    "instrument_match",
    "load_noise_spec",
    "validate_noise_spec",
]

__provisional__ = ("*",)

SCHEMA = "virgil.noise_spec/1"
SCHEMA_PREFIX = "virgil.noise_spec/"
WHITE_KEYS = ("vis", "phi", *KINDS)
GAIN_KEYS = ("telescope", "baseline", "chromatic")
OFFSET_KEYS = ("triangle", "baseline")
COMMON_MODE_KEYS = ("visphi",)
BOUND_VALUES = ("lower", "estimate")
_INST_DELIMS = ("_", "-", "/", " ", "(")
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
        if (
            raw_prov.get("bound") is not None
            and raw_prov["bound"] not in BOUND_VALUES
        ):
            problems.append(
                f"provenance.bound must be None or one of "
                f"{list(BOUND_VALUES)}, got {raw_prov['bound']!r}"
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


def instrument_match(data, spec, *, insname=None):
    """Check whether a dataset's recorded instrument matches ``spec``.

    Compares ``insname`` (or ``data["insname"]`` / ``getattr(data, "insname",
    None)`` when recorded and non-empty) against ``spec["instrument"]`` by
    case-insensitive exact or delimiter-bounded prefix matching so generic
    instrument names such as ``"GRAVITY"`` or ``"MATISSE"`` match sub-channel
    names such as ``"GRAVITY_SC"`` or ``"MATISSE_LM"`` without matching
    unrelated short names. Emits a ``UserWarning`` and returns ``False`` on
    mismatch, never raising.

    Parameters
    ----------
    data : OIData or Mapping
        Dataset or record optionally carrying an ``insname`` attribute or key.
    spec : Mapping
        Noise specification dictionary.
    insname : str or sequence of str, optional
        Explicit instrument name(s) overriding any ``insname`` on ``data``
        (useful when OIFITS ``INSNAME`` metadata are read separately from
        [`OIData`][virgil.oidata.OIData]).

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

    raw_ins = insname
    if raw_ins is None:
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
        if isinstance(raw_ins, onp.ndarray):
            raw_ins = raw_ins.reshape(-1)  # also a 0-d string array
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
        name.lower() == target
        or (
            len(name) > len(target)
            and name.lower().startswith(target)
            and name[len(target)] in _INST_DELIMS
        )
        for name in names
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


def apply_noise_spec(data, spec, *, insname=None):
    """Apply a validated ``virgil.noise_spec/1`` specification to ``data``.

    Validates ``spec``, refuses to apply onto data that already carry any
    marginalized nuisance modes, checks
    [`instrument_match`][virgil.noise_spec.instrument_match], rescales quoted
    uncertainties on the observables present in ``data``, and attaches
    marginalized gains and closure-phase offsets where the corresponding
    observables are present.

    Parameters
    ----------
    data : OIData
        Prepared interferometric dataset without pre-existing gains,
        closure-phase offsets, or differential-phase nuisance modes.
    spec : Mapping
        Noise specification dictionary conforming to ``virgil.noise_spec/1``.
    insname : str or sequence of str, optional
        The data's instrument name(s), checked by
        [`instrument_match`][virgil.noise_spec.instrument_match]; required
        when the spec names an instrument, as `OIData` does not record
        `INSNAME`.

    Returns
    -------
    OIData
        Dataset with the specification's white scales and nuisance modes
        applied. Closure-phase offsets are
        skipped on data without closure phases (baseline phases, projected
        kernel phases).

    Raises
    ------
    ValueError
        If ``spec`` is invalid, if ``data`` already carries marginalized
        gains, closure-phase offsets, differential-phase or flux-spectrum modes, if
        the spec names an instrument and ``insname`` is not given, or if
        ``data`` is incompatible with the requested nuisance modes.
    NotImplementedError
        If ``spec["common_mode"]["visphi"]`` is positive and ``data`` carries
        a ``"visphi"`` block.
    """
    spec = validate_noise_spec(spec)
    visphi_blocks = [b for b in data.extras if b.kind == "visphi"]
    if data.gains is not None:
        raise ValueError(
            "Data already carry calibration gains (data.gains is not None); "
            "apply_noise_spec refuses to stack onto data with existing "
            "nuisance modes."
        )
    if data.phase_offsets is not None:
        raise ValueError(
            "Data already carry closure-phase offsets "
            "(data.phase_offsets is not None); apply_noise_spec refuses to "
            "stack onto data with existing nuisance modes."
        )
    if any(
        getattr(b, "prior_width", None) is not None
        or getattr(b, "modes", None) is not None
        for b in visphi_blocks
    ):
        raise ValueError(
            "Data already carry differential-phase nuisance modes; "
            "apply_noise_spec refuses to stack onto data with existing "
            "nuisance modes."
        )
    if any(
        b.kind in ("flux", "nflux", "corrflux")
        and getattr(b, "modes", None) is not None
        for b in data.extras
    ):
        raise ValueError(
            "Data already carry flux-spectrum nuisance modes; "
            "apply_noise_spec refuses to stack onto data with existing "
            "nuisance modes."
        )

    if spec["instrument"] is not None and insname is None:
        raise ValueError(
            "The spec names an instrument but OIData does not record "
            "INSNAME: pass insname= so the instrument can be checked."
        )
    instrument_match(data, spec, insname=insname)

    present_kinds = {b.kind for b in data.extras}
    if data.vis.size:
        present_kinds.add("vis")
    if data.phi.size:
        present_kinds.add("phi")
    present_white = {
        k: v for k, v in spec["white"].items() if k in present_kinds
    }

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

    out = data
    if present_white:
        out = out.with_error_scale(present_white)
    has_gain_obs = bool(out.vis.size) or any(
        b.gain_legs() and onp.size(b.values) for b in out.extras
    )
    if gain_kwargs and has_gain_obs:
        out = out.with_gains(**gain_kwargs)
    if offset_kwargs and out.phi.size:
        if out.cp_flag and out.phi_mat is None:
            out = out.with_closure_offsets(**offset_kwargs)
        else:
            warnings.warn(
                f"closure_offsets {offset_kwargs} skipped: the data's phases "
                "are not closure phases (baseline or projected phases).",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
    if visphi_active and visphi_blocks:
        raise NotImplementedError(
            "Applying common_mode.visphi to DifferentialPhase is not yet "
            "implemented."
        )
    return out
