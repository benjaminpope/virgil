"""The single-use private helper gate (scripts/code_shape.py, rule 8 in AGENTS.md).

Pure stdlib: parses fixture snippets only.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "code_shape", ROOT / "scripts" / "code_shape.py"
)
shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape)


def _found(src, base="", tests=""):
    return shape.find_single_use({"m.py": src}, {"m.py": base}, tests)


def test_single_call_fails_at_any_size():
    src = "def _h(x):\n    a = x\n    b = a\n    c = b\n    d = c\n    return d\n\n\ndef f(x):\n    return _h(x)\n"
    assert _found(src) == ["m.py:1 _h"]


def test_nested_single_call_fails():
    src = "def f(x):\n    def _g(y):\n        return y\n\n    return _g(x)\n"
    assert _found(src) == ["m.py:2 f._g"]


def test_method_fails():
    src = "class C:\n    def _m(self):\n        return 1\n\n    def f(self):\n        return self._m()\n"
    assert _found(src) == ["m.py:2 C._m"]


def test_two_calls_pass():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h() + _h()\n"
    assert _found(src) == []


def test_existing_helper_passes():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    assert _found(src, base=src) == []


def test_deliberate_comment_on_line_or_above_passes():
    call = "\n\ndef f():\n    return _h()\n"
    above = "# deliberate: custom JVP helper\ndef _h():\n    return 1\n" + call
    same = "def _h():  # deliberate: custom JVP helper\n    return 1\n" + call
    assert _found(above) == [] and _found(same) == []
    assert _found(
        "# deliberate\ndef _h():\n    return 1\n" + call
    )  # no reason
    assert _found(
        "# deliberate:   \ndef _h():\n    return 1\n" + call
    )  # empty reason


def test_jit_kernels_pass():
    for deco in (
        "jax.jit",
        "partial(jax.jit, static_argnums=0)",
        "eqx.filter_jit",
    ):
        src = f"@{deco}\ndef _k(x):\n    return x\n\n\ndef f(x):\n    return _k(x)\n"
        assert _found(src) == [], deco


def test_custom_derivative_functions_and_rules_pass():
    src = (
        "@jax.custom_jvp\ndef _f(x):\n    return x\n\n\n"
        "@_f.defjvp\ndef _f_jvp(p, t):\n    return p, t\n\n\n"
        "def g(x):\n    return _f(x)\n"
    )
    assert _found(src) == []
    vjp = "@jax.custom_vjp\ndef _f(x):\n    return x\n\n\ndef g(x):\n    return _f(x)\n"
    assert _found(vjp) == []


def test_singledispatch_register_passes():
    src = "@disp.register\ndef _h(x: int):\n    return x\n\n\ndef f():\n    return _h(1)\n"
    assert _found(src) == []


def test_callback_passes():
    src = "def f(xs):\n    def _step(c, x):\n        return c, x\n\n    return lax.scan(_step, 0, xs)\n"
    assert _found(src) == []
    src = "def _h(x):\n    return x\n\n\ndef f(xs):\n    _h(0)\n    return map(_h, xs)\n"
    assert _found(src) == []


def test_name_used_in_tests_passes():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    assert _found(src, tests="from virgil.m import _h") == []


def test_register_with_arguments_passes():
    src = "@disp.register(int)\ndef _h(x):\n    return x\n\n\ndef f():\n    return _h(1)\n"
    assert _found(src) == []


def test_comment_above_decorator_passes():
    src = "# deliberate: cached\n@functools.cache\ndef _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    assert _found(src) == []


def test_move_to_another_module_is_not_an_addition():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    got = shape.find_single_use({"new.py": src}, {"old.py": src})
    assert got == []


def test_same_name_in_another_module_does_not_hide_a_new_helper():
    new = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    other = "def _h():\n    return 2\n\n\ndef g():\n    return _h() + _h()\n"
    got = shape.find_single_use(
        {"a.py": new, "b.py": other}, {"a.py": "", "b.py": other}, ""
    )
    assert got == ["a.py:1 _h"]


def test_cross_module_import_counts_as_a_second_call():
    a = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    b = "from a import _h\n\n\ndef g():\n    return _h()\n"
    assert shape.find_single_use({"a.py": a, "b.py": b}, {"b.py": b}) == []


def test_test_local_name_does_not_exempt():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    assert _found(src, tests="def _h():\n    return 0\n") == ["m.py:1 _h"]


def test_qualified_test_import_only_exempts_that_module():
    src = "def _h():\n    return 1\n\n\ndef f():\n    return _h()\n"
    got = shape.find_single_use(
        {"a.py": src, "b.py": src},
        {"a.py": "", "b.py": ""},
        "from virgil.a import _h\n",
    )
    assert got == ["b.py:1 _h"]
    # Package __init__.py and submodule attribute accesses via package import are also module-qualified.
    got_pkg = shape.find_single_use(
        {
            "pipeline/__init__.py": src,
            "pipeline/_checks.py": src,
            "other.py": src,
        },
        {},
        "from virgil.pipeline import _h, _checks\n_checks._h()\nsome_obj._h()\n",
    )
    assert got_pkg == ["other.py:1 _h"]


def test_find_oversized_functions():
    body = "".join(f"    x{i} = {i}\n" for i in range(125))
    big = f'def f():\n    """Docstring."""\n{body}    return x0\n'
    small = "def f():\n    return 1\n"
    assert shape.find_oversized_functions({"m.py": big}, {"m.py": small}) == [
        "m.py:1 f (126 lines > 120)"
    ]
    # Unchanged oversized function in base does not fail, nor one with a long docstring only.
    assert shape.find_oversized_functions({"m.py": big}, {"m.py": big}) == []
    long_doc = 'def f():\n    """' + "line\n" * 130 + '"""\n    return 1\n'
    assert (
        shape.find_oversized_functions({"m.py": long_doc}, {"m.py": ""}) == []
    )
    # A new function in a new module sharing a name with a base function that still exists is flagged.
    big_other = f"def f():\n{body}    return x0 + 1\n"
    assert shape.find_oversized_functions(
        {"old.py": big, "new.py": big_other}, {"old.py": big}
    ) == ["new.py:1 f (126 lines > 120)"]
    # Moving an oversized function from old.py to existing.py passes.
    assert (
        shape.find_oversized_functions(
            {"old.py": "", "existing.py": big_other},
            {"old.py": big, "existing.py": ""},
        )
        == []
    )


_STAMPED = "# deliberate: {}\ndef _h():\n    return 1\n\n\ndef f():\n    return _h()\n"


def test_each_vacuous_stem_is_rejected():
    for stem in shape._VACUOUS_STEMS:
        for reason in (
            stem,
            f"{stem.upper()}: unit-tested directly",
            f"the {stem}",
            f"A {stem}",
            stem.replace(" ", "-"),
            stem.replace(" ", "  "),
        ):
            assert _found(_STAMPED.format(reason)) == ["m.py:2 _h"], reason


def test_genuine_reason_passes():
    reason = "custom_jvp rule: the primal is reused by the tangent"
    assert _found(_STAMPED.format(reason)) == []
    assert (
        shape.find_vacuous_stamps({"m.py": _STAMPED.format(reason)}, []) == []
    )


def _key(reason, q="_h"):
    return ("m.py", q, reason)


def test_vacuous_stamp_off_the_allowlist_fails_and_on_it_passes():
    src = _STAMPED.format("named step")
    assert len(shape.find_vacuous_stamps({"m.py": src}, [])) == 1
    assert shape.find_vacuous_stamps({"m.py": src}, [_key("named step")]) == []


def test_changed_listed_stamp_is_judged_again():
    new = _STAMPED.format("test seam")
    got = shape.find_vacuous_stamps({"m.py": new}, [_key("named step")])
    assert len(got) == 2  # the new one is unlisted; the old entry has no stamp


def test_allowlist_may_only_shrink():
    src = _STAMPED.format("named step")
    k = _key("named step")
    assert shape.find_vacuous_stamps({"m.py": src}, [k], [k]) == []
    got = shape.find_vacuous_stamps({"m.py": src}, [k], [])
    assert len(got) == 1 and "may only shrink" in got[0]
    stale = shape.find_vacuous_stamps({"m.py": ""}, [k], [k])
    assert len(stale) == 1 and "remove it" in stale[0]


def test_duplicate_stamps_need_duplicate_entries():
    src = _STAMPED.format("named step") + _STAMPED.format("named step")
    assert (
        len(shape.find_vacuous_stamps({"m.py": src}, [_key("named step")]))
        == 1
    )


def test_allowlist_file_round_trips_and_checks_its_count():
    keys = [_key("named step"), _key("a :: b", "f")]
    text = shape.format_allowlist(keys)
    assert shape.read_allowlist(text) == sorted(keys)
    for bad in (text.replace("count: 2", "count: 1"), text + "x :: y :: z\n"):
        with pytest.raises(ValueError):
            shape.read_allowlist(bad)


def test_committed_allowlist_matches_the_tree():
    tree = shape._read_tree()
    listed = shape.read_allowlist(shape.STAMP_ALLOWLIST.read_text())
    assert shape.find_vacuous_stamps(tree, listed) == []


_MULTI = "{above}def _h({defl}\n    a,{mid}\n    b,\n):{close}\n    return 1\n\n\ndef f():\n    return _h(1, 2)\n"


def _multi(where, reason):
    parts = dict(above="", defl="", mid="", close="")
    mark = f"# deliberate: {reason}"
    parts[where] = mark + "\n" if where == "above" else "  " + mark
    return _MULTI.format(**parts)


def test_stamp_on_any_line_of_a_multiline_signature_counts():
    for where in ("above", "defl", "mid", "close"):
        good, bad = (
            _multi(where, "a lax.scan body"),
            _multi(where, "jit boundary"),
        )
        line = 2 if where == "above" else 1
        assert _found(good) == [], where
        assert _found(bad) == [f"m.py:{line} _h"], where
        assert len(shape._stamps({"m.py": bad})) == 1, where
        assert len(shape.find_vacuous_stamps({"m.py": bad}, [])) == 1, where


def test_body_comment_does_not_stamp_a_multiline_def():
    src = _MULTI.format(above="", defl="", mid="", close="").replace(
        "    return 1", "    # deliberate: a lax.scan body\n    return 1"
    )
    assert _found(src) == ["m.py:1 _h"]
    assert shape._stamps({"m.py": src}) == []


def test_annotated_signature_ends_at_its_own_colon():
    src = (
        "def _h(a: dict[str, int],\n       b: int = 1) -> dict:  # deliberate: a lax.scan body\n"
        "    return a\n\n\ndef f():\n    return _h({})\n"
    )
    assert _found(src) == []


def test_audited_vacuous_reasons_are_rejected():
    for reason in (
        "stage callable",
        "stage callable, patched by tests",
        "one named step per part",
        "dispatch-table callable (_SECTIONS)",
        "argparse callback",
        "registry callback",
        "float64 boundary (in_float64) around the NUTS run",
        "jit/vmap boundary, 59 lines",
        "speed",
        "readability",
        "60 lines",
        "85 lines, tested directly",
        "tests call it directly",
        "imported by aliases.py; unit-tested",
        "named check, keeps numpyro_model short",
    ):
        assert shape._vacuous(reason), reason


def test_specific_reasons_are_not_vacuous():
    for reason in (
        "custom_jvp rule, an implicit-function derivative",
        "lax.while_loop body, jitted with its solve",
        "70 lines; inlined it would bury score_orbits",
        "Gauss-Legendre quadrature in ln f, anchored at the bound nearest the peak",
    ):
        assert not shape._vacuous(reason), reason


def test_audit_counts_vacuous_and_kept():
    src = _STAMPED.format("test seam") + _STAMPED.format("a lax.scan body")
    got = shape._stamps({"m.py": src})
    assert [bool(shape._vacuous(r)) for _, _, r, _ in got] == [True, False]


def test_named_word_step_is_rejected():
    assert shape._vacuous("named pipeline step")
