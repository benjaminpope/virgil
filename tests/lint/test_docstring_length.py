"""No docstring in ``src/virgil`` may have more than 60 lines of prose.

Prose is everything except the numpydoc ``Parameters``, ``Returns`` and
``Raises`` sections, which document the signature and grow with it. Derivations belong in ``docs/`` and design history in ``design/``. See
"Docstring style" in AGENTS.md.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

MAX_LINES = 60
MAX_PARAMS_PER_HEADER = 4
UNCAPPED_SECTIONS = {"Parameters", "Returns", "Raises"}
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
_PARAM_HEADER = re.compile(r"^([A-Za-z_*][\w*,\s]*?)(?:\s*:.*)?$")

# Existing docstrings with > 4 parameters on one numpydoc header line.
# No new entries may be added: shrink this set, never grow it.
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


def prose_lines(doc: str) -> int:
    """Lines of ``doc`` outside its Parameters, Returns and Raises sections."""
    lines = doc.splitlines()
    skipping = False
    n = 0
    for i, line in enumerate(lines):
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt and set(nxt) == {"-"} and line.strip():
            skipping = line.strip() in UNCAPPED_SECTIONS
        if not skipping:
            n += 1
    return n


def docstring_lengths() -> dict[str, int]:
    """Prose line count of every module, class and function docstring."""
    return {key: prose_lines(doc) for key, doc in _docstrings().items()}


def test_no_docstring_is_over_the_limit():
    over = sorted(
        f"{key} ({n} lines)"
        for key, n in docstring_lengths().items()
        if n > MAX_LINES
    )
    assert not over, (
        f"docstrings over {MAX_LINES} prose lines: {over}. Shorten them (derivations "
        "go in docs/, history in design/)."
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


def _doc(prose: int, params: int) -> str:
    body = "\n".join(f"Prose line {i}." for i in range(prose))
    entries = "\n".join(f"p{i} : float\n    Entry." for i in range(params))
    return (
        f"Summary.\n\n{body}\n\nParameters\n----------\n{entries}\n\n"
        "Returns\n-------\nfloat\n    Value.\n\nRaises\n------\n"
        "ValueError\n    Bad input.\n"
    )


def test_long_parameters_with_short_prose_pass():
    doc = _doc(prose=5, params=60)
    assert len(doc.splitlines()) > MAX_LINES
    assert prose_lines(doc) <= MAX_LINES


def test_long_prose_fails():
    assert prose_lines(_doc(prose=MAX_LINES, params=2)) > MAX_LINES


def test_notes_count_as_prose():
    doc = "Summary.\n\nNotes\n-----\n" + "x\n" * 70
    assert prose_lines(doc) > MAX_LINES


def test_prose_resumes_after_uncapped_sections():
    doc = _doc(prose=0, params=3) + "\nNotes\n-----\n" + "x\n" * 70
    assert prose_lines(doc) > MAX_LINES
