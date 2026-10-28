"""Measure the shape of ``src/virgil``: private helpers, wrappers, duplicates.

Usage: ``python scripts/code_shape.py [--json out.json] [--extra DIR ...]
[--max-stmts 3] [--since REF]``.

Reads only. Prints a per-module table and the candidate lists of
``design/code_shape_audit.md``:

- (a) inline candidates (one call site, at most ``--max-stmts`` statements,
  undecorated, not mentioned in tests/docs/examples), and the keepers
  (decorated, or mentioned there);
- (b) identical normalized bodies, same-named helpers with their body
  similarity, and near-duplicates by line-shingle overlap;
- (d) private functions with no use.

Call sites are counted per module: the module's own ``Name`` uses, plus
``from x import _name`` and ``mod._name`` in other modules. A decorator that
registers the function (``@f.defjvp``, ``@x.register``) counts as a use.
``--extra`` adds directories (default tests, docs, examples) grepped for each
private name.

``--since REF`` is the CI gate (rule 8 in AGENTS.md): it prints each private
def (module-level, nested or method) that the tree adds relative to ``REF``
and that is called exactly once in the package, and exits 1 if there is any.
Exempt: a ``# deliberate: <reason>`` comment on the def line or the line above;
jit / custom_jvp / custom_vjp / ``.defjvp`` / ``.defvjp`` / ``.register``
decorators; a def referenced as a value (a callback); a name tests import or take as an attribute
(``tests/lint`` excluded). Calls count in the defining module plus other modules
that import the name or call it as ``mod._name``.
"""

import argparse
import ast
import collections
import copy
import difflib
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "virgil"
_REGISTERING = re.compile(r"\.(def\w*|register)$")


def _strip_doc(body):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _count(stmts):
    return sum(isinstance(n, ast.stmt) for s in stmts for n in ast.walk(s))


def _private(name):
    return name.startswith("_") and not name.startswith("__")


class _Norm(ast.NodeTransformer):
    """Rename arguments and locals by first use so the hash ignores names."""

    def __init__(self, names):
        self.map = {n: f"v{i}" for i, n in enumerate(names)}

    def visit_Name(self, node):
        return ast.copy_location(
            ast.Name(self.map.get(node.id, node.id), node.ctx), node
        )

    def visit_arg(self, node):
        return ast.arg(self.map.get(node.arg, node.arg), None)


def _norm_hash(fn):
    fn = copy.deepcopy(fn)  # the transformer rewrites in place
    names = []
    for n in ast.walk(fn):
        if isinstance(n, ast.arg) and n.arg not in names:
            names.append(n.arg)
        elif (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id not in names
        ):
            names.append(n.id)
    fn.body = _strip_doc(fn.body)
    fn.decorator_list = []
    fn.args.defaults = []
    fn.name = "f"
    fn = _Norm(names).visit(fn)
    return hashlib.md5(ast.dump(fn).encode()).hexdigest()[:10]


def _is_wrapper(fn):
    body = _strip_doc(fn.body)
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Call)
    )


def collect():
    """Private functions (with call-site counts) and sizes of ``src/virgil``."""
    defs, modules = [], {}
    own = collections.Counter()  # (module, name): Name loads in that module
    imported = collections.Counter()  # name: ``from x import name`` anywhere
    attrs = collections.defaultdict(
        collections.Counter
    )  # module: attribute names
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        text = path.read_text()
        tree = ast.parse(text)
        modules[rel] = dict(
            lines=len(text.splitlines()),
            pub=0,
            priv=0,
            methods_pub=0,
            methods_priv=0,
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                own[rel, node.id] += 1
            elif isinstance(node, ast.Attribute):
                attrs[rel][node.attr] += 1
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported[alias.name] += 1
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            private = _private(node.name)
            modules[rel]["priv" if private else "pub"] += 1
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef):
                        key = (
                            "methods_priv"
                            if _private(m.name)
                            else "methods_pub"
                        )
                        modules[rel][key] += 1
            elif private:
                body = _strip_doc(node.body)
                defs.append(
                    dict(
                        module=rel,
                        name=node.name,
                        line=node.lineno,
                        end=node.end_lineno,
                        stmts=_count(body),
                        wrapper=_is_wrapper(node),
                        decorators=[
                            ast.unparse(d) for d in node.decorator_list
                        ],
                        hash=_norm_hash(node),
                        body=ast.unparse(ast.Module(body, [])),
                    )
                )
    for d in defs:
        d["sites"] = (
            own[d["module"], d["name"]]
            + imported[d["name"]]
            + sum(c[d["name"]] for m, c in attrs.items() if m != d["module"])
            + sum(bool(_REGISTERING.search(x)) for x in d["decorators"])
        )
    return defs, modules


def grep_extra(names, dirs):
    """Count whole-word occurrences of each name in non-src files."""
    pats = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    counts = collections.Counter()
    for d in dirs:
        for p in pathlib.Path(d).expanduser().rglob("*"):
            if (
                p.suffix not in {".py", ".ipynb", ".md", ".toml", ".sh"}
                or not p.is_file()
            ):
                continue
            try:
                t = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for n, pat in pats.items():
                counts[n] += len(pat.findall(t))
    return counts


def _shingles(body):
    lines = [
        re.sub(r"[A-Za-z_]\w*", "v", s.strip()) for s in body.splitlines()
    ]
    return set(zip(lines, lines[1:]))


MAX_FUNC_LINES = 120
_EXEMPT_DECORATOR = re.compile(
    r"jit|custom_(jvp|vjp)|\.def(jvp|vjp)(\(.*\))?$|\.register(\(.*\))?$"
)


def _valid_deliberate(line):
    """A ``# deliberate:`` comment with a non-empty reason (specificity is for review)."""
    return "# deliberate:" in line and bool(
        line.split("# deliberate:", 1)[1].strip()
    )


def _all_defs(tree):
    """``(qualname, node)`` of every def: module-level, nested, method."""
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                out.append((q, child))
                walk(child, q + ".")
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _private_defs(tree):
    """``(qualname, node)`` of every private def: module-level, nested, method."""
    return [(q, n) for q, n in _all_defs(tree) if _private(n.name)]


def _func_body_lines(node):
    """Line span of a function's body excluding its docstring."""
    body = _strip_doc(node.body)
    if not body:
        return 0
    return node.end_lineno - body[0].lineno + 1


def _module_uses(tree):
    """Uses in one module: calls and value references, by plain name or attribute.

    Attributes on ``self``/``cls`` stay in the module (``own``); others may
    refer to another module's function (``ext``).
    """
    u = {k: collections.Counter() for k in ("name", "ext", "own")}
    uv = {k: collections.Counter() for k in ("name", "ext", "own")}
    called = set()

    def add(table, n):
        if isinstance(n, ast.Name):
            table["name"][n.id] += 1
        elif isinstance(n, ast.Attribute):
            own = isinstance(n.value, ast.Name) and n.value.id in {
                "self",
                "cls",
            }
            table["own" if own else "ext"][n.attr] += 1

    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            called.add(id(n.func))
            add(u, n.func)
    for n in ast.walk(tree):
        if id(n) not in called:
            add(uv, n)
    imported = {
        al.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for al in n.names
    }
    return u, uv, imported


def _reach(table, name, own, imported):
    """Uses of ``name`` in a module: all of them if it defines ``name``, else
    only those that can reach it (an import, or an attribute of a module)."""
    if own:
        return sum(table[k][name] for k in table)
    return table["ext"][name] + (table["name"][name] if imported else 0)


def _attr_chain(node):
    """Dotted identifier chain ``['a', 'b', 'c']`` for ``a.b.c``, else ``None``."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return parts[::-1]
    return None


def _add_qualified(names, rel_prefix, attr):
    names.add((f"{rel_prefix}.py", attr))
    names.add((f"{rel_prefix}/__init__.py", attr))


def _test_names(tests_text):
    """Names tests import or take as attributes (``mod._name``), qualified by module when known."""
    tree = ast.parse(tests_text)
    names = set()
    mod_aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.module == "virgil":
                for al in n.names:
                    mod_aliases[al.asname or al.name] = al.name
                    names.add(("__init__.py", al.name))
            elif n.module and n.module.startswith("virgil."):
                rel = n.module.removeprefix("virgil.").replace(".", "/")
                for al in n.names:
                    _add_qualified(names, rel, al.name)
                    mod_aliases[al.asname or al.name] = f"{rel}/{al.name}"
            else:
                names.update(al.name for al in n.names)
        elif isinstance(n, ast.Import):
            for al in n.names:
                if al.name.startswith("virgil."):
                    rel = al.name.removeprefix("virgil.").replace(".", "/")
                    key = al.asname or al.name.split(".")[-1]
                    mod_aliases[key] = rel
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            chain = _attr_chain(n)
            if not chain:
                continue
            if chain[0] == "virgil" and len(chain) >= 3:
                _add_qualified(names, "/".join(chain[1:-1]), chain[-1])
            elif chain[0] in mod_aliases:
                prefix = "/".join([mod_aliases[chain[0]], *chain[1:-1]])
                _add_qualified(names, prefix, chain[-1])
            elif len(chain) == 2:
                _add_qualified(names, chain[0], chain[1])
    return names


def find_single_use(head, base, tests_text=""):
    """Private defs in ``head`` (``{path: source}``) absent from ``base``, called once.

    Calls are counted in the defining module plus those that can reach it from
    other modules (an import of the name, or ``mod.name``). A def is new when
    its qualified name is not in the same module of ``base`` (nor, for a module
    absent from ``base``, in any), so moving a module is not an addition. Returns ``["path:line qualname", ...]``; see the module docstring
    for the exemptions.
    """
    trees = {p: ast.parse(t) for p, t in head.items()}
    old = {
        (p, q) for p, t in base.items() for q, _ in _private_defs(ast.parse(t))
    }
    moved = {q for _, q in old}
    uses = {p: _module_uses(t) for p, t in trees.items()}
    tested = _test_names(tests_text)
    found = []
    for path, tree in trees.items():
        lines = head[path].splitlines()
        for q, node in _private_defs(tree):
            name = node.name
            if (path, q) in old or (path, name) in tested or name in tested:
                continue
            if path not in base and q in moved:  # a moved module
                continue
            counts = [
                (
                    _reach(u, name, m == path, name in imp),
                    _reach(uv, name, m == path, name in imp),
                )
                for m, (u, uv, imp) in uses.items()
            ]
            if sum(c for c, _ in counts) != 1 or sum(v for _, v in counts):
                continue
            if any(
                _EXEMPT_DECORATOR.search(ast.unparse(d))
                for d in node.decorator_list
            ):
                continue
            top = min([node.lineno] + [d.lineno for d in node.decorator_list])
            near = (
                lines[top - 2 : top - 1] + lines[node.lineno - 1 : node.lineno]
            )
            if any(_valid_deliberate(x) for x in near):
                continue
            found.append(f"{path}:{node.lineno} {q}")
    return found


def find_oversized_functions(head, base, max_lines=MAX_FUNC_LINES):
    """Functions in ``head`` that are new or grew vs ``base`` and exceed ``max_lines`` body lines."""
    trees = {p: ast.parse(t) for p, t in head.items()}
    base_trees = {p: ast.parse(t) for p, t in base.items()}
    head_defs = {(p, q): n for p, t in trees.items() for q, n in _all_defs(t)}
    old_sizes = {
        (p, q): _func_body_lines(n)
        for p, t in base_trees.items()
        for q, n in _all_defs(t)
    }
    moved_sizes = {}
    for (old_p, q), sz in old_sizes.items():
        if (old_p, q) not in head_defs:
            moved_sizes[q] = max(moved_sizes.get(q, 0), sz)
    found = []
    for path, tree in trees.items():
        for q, node in _all_defs(tree):
            n_lines = _func_body_lines(node)
            if n_lines <= max_lines:
                continue
            old_lines = old_sizes.get((path, q), moved_sizes.get(q))
            if old_lines is not None and n_lines <= old_lines:
                continue
            found.append(
                f"{path}:{node.lineno} {q} ({n_lines} lines > {max_lines})"
            )
    return found


def _read_tree(ref=None):
    """``{path: source}`` of ``src/virgil`` on disk, or at git ``ref``."""
    if ref is None:
        return {
            str(p.relative_to(SRC)): p.read_text() for p in SRC.rglob("*.py")
        }
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "src/virgil"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {
        n[len("src/virgil/") :]: subprocess.run(
            ["git", "show", f"{ref}:{n}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for n in names
        if n.endswith(".py")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--max-stmts", type=int, default=3)
    ap.add_argument("--since")
    a = ap.parse_args()
    if a.since:
        tests = "\n".join(
            p.read_text()
            for p in (ROOT / "tests").rglob("*.py")
            if "lint" not in p.parts
        )
        head, base = _read_tree(), _read_tree(a.since)
        found = find_single_use(head, base, tests)
        oversized = find_oversized_functions(head, base)
        print(f"single-use private defs added since {a.since}: {len(found)}")
        for f in found:
            print(f"  {f}")
        if oversized:
            print(
                f"functions over {MAX_FUNC_LINES} body lines added/grown since {a.since}: {len(oversized)}"
            )
            for f in oversized:
                print(f"  {f}")
        raise SystemExit(bool(found or oversized))
    defs, modules = collect()
    extra = grep_extra(
        {d["name"] for d in defs},
        a.extra or [ROOT / "tests", ROOT / "docs", ROOT / "examples"],
    )
    for d in defs:
        d["extra"] = extra[d["name"]]
    small = [d for d in defs if d["sites"] == 1 and d["stmts"] <= a.max_stmts]
    inline = [d for d in small if not d["decorators"] and not d["extra"]]
    print(
        f"{'module':32s} {'lines':>6s} {'pub':>4s} {'priv':>5s} {'m_pub':>5s} {'m_priv':>6s}"
    )
    for m, v in sorted(modules.items(), key=lambda kv: -kv[1]["priv"]):
        print(
            f"{m:32s} {v['lines']:6d} {v['pub']:4d} {v['priv']:5d} "
            f"{v['methods_pub']:5d} {v['methods_priv']:6d}"
        )
    print(
        f"\n(a) inline: one use, <={a.max_stmts} statements, undecorated, "
        f"unmentioned in tests/docs/examples ({len(inline)}):"
    )
    for d in inline:
        print(
            f"  {d['module']}:{d['line']} {d['name']} "
            f"stmts={d['stmts']} wrapper={d['wrapper']}"
        )
    print(
        "\n(a) keep: the same, but decorated (jit) or mentioned in tests/docs/examples:"
    )
    for d in small:
        if d not in inline:
            why = ", ".join(
                w
                for w, on in (("jit", d["decorators"]), ("tests", d["extra"]))
                if on
            )
            print(f"  {d['module']}:{d['line']} {d['name']} ({why})")
    print("\n(d) no use in src:")
    for d in defs:
        if d["sites"] == 0:
            print(
                f"  {d['module']}:{d['line']} {d['name']} extra={d['extra']}"
            )
    print("\n(b) identical normalized bodies (>=3 statements):")
    by_hash = collections.defaultdict(list)
    for d in defs:
        if d["stmts"] >= 3:
            by_hash[d["hash"]].append(d)
    for g in by_hash.values():
        if len({x["module"] for x in g}) > 1:
            print("  " + "; ".join(f"{x['module']}:{x['name']}" for x in g))
    print("\n(b) same name, several modules (body similarity):")
    names = collections.Counter(d["name"] for d in defs)
    for n, c in names.items():
        if c > 1:
            g = [d for d in defs if d["name"] == n]
            sims = [
                difflib.SequenceMatcher(None, x["body"], y["body"]).ratio()
                for i, x in enumerate(g)
                for y in g[i + 1 :]
            ]
            print(
                f"  {n}: {len(g)} defs, max sim {max(sims):.2f}, "
                f"in {', '.join(d['module'] for d in g)}"
            )
    print(
        "\n(b) near-duplicates by line-shingle overlap (>=8 statements, >=0.3):"
    )
    big = [d for d in defs if d["stmts"] >= 8]
    for d in big:
        d["sh"] = _shingles(d["body"])
    pairs = []
    for i, x in enumerate(big):
        for y in big[i + 1 :]:
            j = len(x["sh"] & y["sh"]) / max(1, len(x["sh"] | y["sh"]))
            if j >= 0.3:
                pairs.append((j, x, y))
    for j, x, y in sorted(pairs, key=lambda t: -t[0]):
        print(
            f"  {j:.2f} {x['module']}:{x['name']} ~ {y['module']}:{y['name']}"
        )
    if a.json:
        for d in defs:
            d.pop("sh", None)
        pathlib.Path(a.json).write_text(
            json.dumps(dict(defs=defs, modules=modules), indent=1)
        )


if __name__ == "__main__":
    main()
