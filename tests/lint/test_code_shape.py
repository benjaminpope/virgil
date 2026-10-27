"""The single-use private helper gate (scripts/code_shape.py, rule 8 in AGENTS.md).

Pure stdlib: parses fixture snippets only.
"""

import importlib.util
from pathlib import Path

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
    for generic in (
        "named step",
        "a named step",
        "test seam",
        "stage callable",
    ):
        assert _found(
            f"# deliberate: {generic}\ndef _h():\n    return 1\n" + call
        ) == ["m.py:2 _h"]


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
