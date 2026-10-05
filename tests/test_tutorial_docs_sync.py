from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sync_module(repo_root: Path):
    script_path = repo_root / "scripts" / "sync_tutorial_docs.py"
    spec = importlib.util.spec_from_file_location(
        "sync_tutorial_docs", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_markdown_is_synced_with_notebooks():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sync_module(repo_root)

    for nb_rel, doc_rel in module.MAPPINGS.items():
        nb_path = repo_root / nb_rel
        doc_path = repo_root / doc_rel

        expected = module.render_notebook_markdown(nb_path)
        actual = doc_path.read_text(encoding="utf-8")

        assert actual == expected, (
            f"{doc_rel} is out of sync with {nb_rel}. "
            "Run scripts/sync_tutorial_docs.py to regenerate tutorial docs."
        )


def test_tutorial_markdown_header_uses_repo_relative_notebook_path():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sync_module(repo_root)

    nb_rel = "notebooks/binary_search.ipynb"
    rendered = module.render_notebook_markdown(repo_root / nb_rel)
    header_line = next(line for line in rendered.splitlines() if line.strip())

    assert nb_rel in header_line
    assert str(repo_root.resolve()) not in header_line


def test_warning_locations_are_removed_from_published_output():
    module = _load_sync_module(Path(__file__).resolve().parents[1])
    raw = (
        "/var/folders/ab/T/ipykernel_86886/1114166516.py:8: RuntimeWarning: "
        "optimizer did not converge\n"
        "  opt_flux = optimized_flux_grid(\n"
        "result: 3\n"
    )
    assert module._sanitize_text(raw) == (
        "RuntimeWarning: optimizer did not converge\nresult: 3\n"
    )


def test_landing_page_is_generated_from_the_readme():
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_sync_module(repo_root)
    readme, index = (repo_root / path for path in module.LANDING_PAGE)
    assert index.read_text(encoding="utf-8") == module.render_landing_page(
        readme.read_text(encoding="utf-8")
    ), (
        "docs/index.md is out of sync with README.md: run scripts/sync_tutorial_docs.py."
    )


def test_landing_page_links_stay_inside_the_docs():
    module = _load_sync_module(Path(__file__).resolve().parents[1])
    url = module.DOCS_URL
    page = module.render_landing_page(
        f"[a]({url}data_io/) [b]({url}api/) [c]({url}) [d](CONTRIBUTING.md)"
    )
    assert "[a](data_io.md)" in page
    assert "[b](api/index.md)" in page
    assert f"[c]({url})" in page
    assert f"[d]({module.REPO_FILE_URL}CONTRIBUTING.md)" in page


def test_sanitize_text_strips_machine_paths():
    module = _load_sync_module(Path(__file__).resolve().parents[1])
    for prefix in ("/Users/ben/code/virgil", "/fred/oz1/x/virgil", "/a/b"):
        text = f"{prefix}/src/virgil/fitting.py:12: UserWarning: no\n"
        assert module._sanitize_text(text) == (
            "virgil/fitting.py:12: UserWarning: no\n"
        )
    other = "/home/me/.venv/lib/x.py:3: DeprecationWarning: old"
    assert module._sanitize_text(other) == "x.py:3: DeprecationWarning: old"
    assert module._sanitize_text("plain /Users/x/y") == "plain /Users/x/y"
