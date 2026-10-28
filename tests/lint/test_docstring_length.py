"""No docstring in ``src/virgil`` may exceed 60 lines (80 for >= 10 parameters).

Derivations belong in ``docs/`` and design history in ``design/``. See
"Docstring style" in AGENTS.md.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

MAX_LINES = 60
MAX_LINES_LARGE = 80
LARGE_PARAM_THRESHOLD = 10
MAX_PARAMS_PER_HEADER = 4
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
_PARAM_HEADER = re.compile(r"^([A-Za-z_*][\w*,\s]*?)(?:\s*:.*)?$")

# Legacy docstrings with > 4 parameters on a single numpydoc header line,
# scheduled for cleanup in Wave 2/3 (PR 2/4/6). No new entries may be added.
LEGACY_CRAMMED = {
    "virgil.aliases:fit_orbit_aliases",
    "virgil.arcs:unwrap_cycles",
    "virgil.ensemble:gp_members",
    "virgil.epochs:broad_loglike",
    "virgil.epochs:epoch_positions",
    "virgil.epochs:start_from_positions",
    "virgil.fitting:gauss_newton_diag",
    "virgil.fitting:gauss_newton_mass",
    "virgil.grid_fit:laplace_flux_uncertainty_grid",
    "virgil.grid_fit:linear_flux_grid",
    "virgil.limits:injection_limits",
    "virgil.orbits:KeplerOrbit.from_position_angle",
    "virgil.plotting:plot_field_draws",
    "virgil.svi:variational",
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _param_headers(doc: str) -> list[list[str]]:
    """Lists of parameter names on each numpydoc ``Parameters`` header line."""
    lines = doc.splitlines()
    in_params = False
    param_indent = 0
    headers: list[list[str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped == "Parameters"
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        ):
            in_params = True
            param_indent = len(line) - len(line.lstrip())
            continue
        if not in_params:
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        if (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and set(lines[i + 1].strip()) == {"-"}
            and len(lines[i + 1].strip()) == len(stripped)
        ):
            in_params = False
            continue
        indent = len(line) - len(line.lstrip())
        if stripped and indent == param_indent:
            m = _PARAM_HEADER.match(stripped)
            if m:
                names = [n.strip() for n in m.group(1).split(",") if n.strip()]
                headers.append(names)
    return headers


def _docstrings() -> dict[str, str]:
    """Every module, class and function docstring in ``src/virgil``."""
    docs = {}

    def visit(node, module, qualname):
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                name = [*qualname, child.name]
                doc = ast.get_docstring(child, clean=False)
                if doc:
                    docs[f"{module}:{'.'.join(name)}"] = doc
                visit(child, module, name)
            else:
                visit(child, module, qualname)

    for path in sorted(SRC.glob("virgil/**/*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree, clean=False)
        if doc:
            docs[f"{module}:<module>"] = doc
        visit(tree, module, [])
    return docs


def docstring_lengths() -> dict[str, int]:
    """Line count of every module, class and function docstring."""
    return {key: len(doc.splitlines()) for key, doc in _docstrings().items()}


def _line_limit(doc: str) -> int:
    n_params = sum(len(names) for names in _param_headers(doc))
    return MAX_LINES_LARGE if n_params >= LARGE_PARAM_THRESHOLD else MAX_LINES


def test_no_docstring_is_over_the_limit():
    over = []
    for key, doc in sorted(_docstrings().items()):
        n = len(doc.splitlines())
        limit = _line_limit(doc)
        if n > limit:
            over.append(f"{key} ({n} lines > {limit})")
    assert not over, (
        f"docstrings over the limit ({MAX_LINES} lines, or {MAX_LINES_LARGE} "
        f"for >= {LARGE_PARAM_THRESHOLD} parameters): {over}. Shorten them "
        "(derivations go in docs/, history in design/)."
    )


def test_no_crammed_parameter_headers():
    crammed = sorted(
        f"{key}: {', '.join(names)}"
        for key, doc in _docstrings().items()
        if key not in LEGACY_CRAMMED
        for names in _param_headers(doc)
        if len(names) > MAX_PARAMS_PER_HEADER
    )
    assert not crammed, (
        f"numpydoc parameter headers with > {MAX_PARAMS_PER_HEADER} names: "
        f"{crammed}. Document unrelated parameters on separate lines."
    )


def test_param_headers_parses_typed_and_untyped_lines():
    doc = (
        "Summary.\n\nParameters\n----------\n"
        "a, b : float\n    Typed pair.\n"
        "c, d, e, f, g\n    Untyped crammed line.\n"
    )
    assert _param_headers(doc) == [["a", "b"], ["c", "d", "e", "f", "g"]]
