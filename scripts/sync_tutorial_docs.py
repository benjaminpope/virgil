from __future__ import annotations

import base64
import json
import re
from pathlib import Path


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# A warning raised from a notebook cell is printed as
# "<tmpdir>/ipykernel_<pid>/<hash>.py:<line>: <Category>: <message>" followed
# by the offending source line. The path is machine-specific and changes on
# every run, so the published pages keep only "<Category>: <message>".
_CELL_WARNING = re.compile(
    r"^\S*ipykernel_\d+[/\\]\d+\.py:\d+: (?P<warning>\w+: .*)\n"
    r"(?P<source>[ \t]+\S.*\n?)?",
    re.MULTILINE,
)


def _sanitize_text(text: str) -> str:
    """Strip terminal colours and machine-specific warning locations."""
    text = _ANSI_ESCAPE.sub("", text)
    return _CELL_WARNING.sub(lambda m: m.group("warning") + "\n", text)


MAPPINGS = {
    "notebooks/binary_search.ipynb": "docs/binary_search.md",
    "notebooks/data_io.ipynb": "docs/data_io.md",
    "notebooks/contrast_limits.ipynb": "docs/contrast_limits.md",
    "notebooks/hierarchical_inference.ipynb": "docs/hierarchical_inference.md",
    "notebooks/model_syntax.ipynb": "docs/model_syntax.md",
    "notebooks/source_models.ipynb": "docs/source_models.md",
    "notebooks/composition.ipynb": "docs/composition.md",
    "notebooks/amigo_disco.ipynb": "docs/amigo_disco.md",
    "notebooks/imaging_ami.ipynb": "docs/imaging_ami.md",
    "notebooks/imaging_rml.ipynb": "docs/imaging_rml.md",
    "notebooks/imaging_gp.ipynb": "docs/imaging_gp.md",
    "notebooks/imaging_composite.ipynb": "docs/imaging_composite.md",
    "notebooks/imaging_sampling.ipynb": "docs/imaging_sampling.md",
    "notebooks/imaging_clean.ipynb": "docs/imaging_clean.md",
    "notebooks/harmonix.ipynb": "docs/harmonix.md",
    "notebooks/limb_darkening.ipynb": "docs/limb_darkening.md",
    "notebooks/gravity_darkened_star.ipynb": "docs/gravity_darkened_star.md",
}


def _to_md_source(cell_source: list[str] | str) -> str:
    if isinstance(cell_source, list):
        return "".join(cell_source).rstrip()
    return str(cell_source).rstrip()


def _to_text(value) -> str:
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    return str(value)


def _output_text(output: dict) -> str:
    out_type = output.get("output_type")
    if out_type == "stream":
        return _to_text(output.get("text", ""))
    if out_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        if "text/plain" in data:
            return _to_text(data["text/plain"])
    if out_type == "error":
        return _to_text(output.get("traceback", ""))
    return ""


def _output_png_bytes(output: dict) -> bytes | None:
    if output.get("output_type") not in {"execute_result", "display_data"}:
        return None
    data = output.get("data", {})
    image_b64 = data.get("image/png")
    if image_b64 is None:
        return None
    image_b64 = _to_text(image_b64).replace("\n", "")
    return base64.b64decode(image_b64)


def render_notebook_markdown(
    nb_path: Path, *, write_images: bool = False
) -> str:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    repo_root = nb_path.resolve().parents[1]
    generated_dir = repo_root / "docs" / "generated"

    # Only the sync step should touch the filesystem; comparisons stay read-only.
    if write_images:
        generated_dir.mkdir(parents=True, exist_ok=True)
        for old_img in generated_dir.glob(f"{nb_path.stem}_cell*_out*.png"):
            old_img.unlink()

    lines: list[str] = []
    relative_nb_path = nb_path.resolve().relative_to(repo_root).as_posix()

    lines.append(
        f"<!-- AUTO-GENERATED FROM {relative_nb_path} "
        "by scripts/sync_tutorial_docs.py. -->"
    )

    for cell_index, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type")
        source = _to_md_source(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "markdown":
            lines.append(source.strip())
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            lines.append("")

            for output_index, output in enumerate(
                cell.get("outputs", []), start=1
            ):
                png_bytes = _output_png_bytes(output)
                if png_bytes is not None:
                    image_name = (
                        f"{nb_path.stem}_cell{cell_index:03d}"
                        f"_out{output_index:02d}.png"
                    )
                    if write_images:
                        image_path = generated_dir / image_name
                        image_path.write_bytes(png_bytes)
                    lines.append(
                        f"![{nb_path.stem} output {cell_index}.{output_index}]"
                        f"(generated/{image_name})"
                    )
                    lines.append("")
                    continue

                text_out = _sanitize_text(_output_text(output)).rstrip()
                if text_out:
                    lines.append("```text")
                    lines.append(text_out)
                    lines.append("```")
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


DOCS_URL = "https://benjaminpope.github.io/virgil/"
REPO_FILE_URL = "https://github.com/benjaminpope/virgil/blob/main/"
LANDING_PAGE = ("README.md", "docs/index.md")


def render_landing_page(readme_text: str) -> str:
    """The docs landing page, generated from ``README.md``.

    Links into the docs site become links between pages, so that they stay
    inside the site; links to files in the repository (e.g.
    ``CONTRIBUTING.md``) become links to them on GitHub. The site's root
    URL (in the badges) is kept as it is.
    """

    def docs_page(match: re.Match) -> str:
        page = match.group(1).strip("/")
        return f"]({page}/index.md)" if page == "api" else f"]({page}.md)"

    # Repository files first: the docs pages produced next are relative too.
    text = re.sub(
        r"\]\((?!https?://|#|mailto:)([^)]+)\)",
        lambda m: f"]({REPO_FILE_URL}{m.group(1)})",
        readme_text,
    )
    text = re.sub(rf"\]\({re.escape(DOCS_URL)}([^)#]+)\)", docs_page, text)
    header = (
        f"<!-- AUTO-GENERATED FROM {LANDING_PAGE[0]} by "
        "scripts/sync_tutorial_docs.py. Edit README.md, not this file. -->\n"
    )
    return header + text


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    readme, index = (repo_root / path for path in LANDING_PAGE)
    landing = render_landing_page(readme.read_text("utf-8"))
    if index.exists() and index.read_text("utf-8") == landing:
        print(f"unchanged {LANDING_PAGE[1]}")
    else:
        index.write_text(landing, encoding="utf-8")
        print(f"synced {LANDING_PAGE[1]} <- {LANDING_PAGE[0]}")

    for nb_rel, doc_rel in MAPPINGS.items():
        nb_path = repo_root / nb_rel
        doc_path = repo_root / doc_rel
        rendered = render_notebook_markdown(nb_path, write_images=True)
        if doc_path.exists() and doc_path.read_text("utf-8") == rendered:
            print(f"unchanged {doc_rel}")
            continue
        doc_path.write_text(rendered, encoding="utf-8")
        print(f"synced {doc_rel} <- {nb_rel}")


if __name__ == "__main__":
    main()
