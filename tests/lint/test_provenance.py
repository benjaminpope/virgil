"""Every source cited in ``src/`` or ``docs/`` is in ``references/provenance.toml``.

Checks arXiv IDs, DOIs, ADS bibcodes, GitHub/GitLab/Bitbucket repositories and
author-year citations ("Smith (2019)", "Smith et al. [2019]", "Smith, Jones &
Lee 2019") against the TOML (the single list behind ``docs/references.md``)
in the source, docs, notebooks, scripts and top-level Markdown; that each
``references.md#key`` anchor names a key; that no entry is an orphan; and that
each ``used_in`` symbol imports. A citation matches an entry by first-author
surname and ``year`` (or ``alt_years``, with ``alt_years_note`` saying why).
A source that cannot be checked on the arXiv abstract page stays
``verified = false``.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import tomllib
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFS = tomllib.loads(
    (ROOT / "references" / "provenance.toml").read_text(encoding="utf-8")
)["ref"]

# Links that are not sources of ideas: the project's own repositories and the
# libraries virgil is built on (credited on the Contributions page).
SELF_AND_DEPENDENCY_REPOS = {
    "benjaminpope/virgil",
    "benjaminpope/virgil-validation",
    "benjaminpope/virgil-vlti",
    "benjaminpope/claude-code-carbon-dashboard",
    "your-username-here/virgil",  # the placeholder in CONTRIBUTING.md
    "louisdesdoigts/zodiax_tutorials",
    "jax-ml/jax",
    "google-deepmind/optax",
    "patrick-kidger/equinox",
    "patrick-kidger/lineax",
    "patrick-kidger/optimistix",
    "pyro-ppl/numpyro",
}

# Plain-text forms of the identifiers. Versions ("v2") and the abs/pdf/html
# route are ignored; old-style IDs ("astro-ph/0508185") are kept whole.
_NEW_ID = r"\d{4}\.\d{4,5}"
_OLD_ID = r"[a-z][a-z\-]+(?:\.[A-Z]{2})?/\d{7}"
ARXIV = re.compile(
    rf"(?:arXiv[:/ ]\s*|arxiv\.org/(?:abs|pdf|html)/)({_NEW_ID}|{_OLD_ID})"
    rf"|\b((?:astro-ph|hep-[a-z]+|gr-qc|physics|math|stat|cs|cond-mat)"
    rf"(?:\.[A-Z]{{2}})?/\d{{7}})",
    re.I,
)
DOI = re.compile(r"(?<![\w.])(10\.\d{4,9}/[^\s\"'<>\]}|]+)")
BIBCODE = re.compile(
    r"(?<![\w.&])((?:18|19|20)\d\d[A-Za-z][\w.&]{4}[\w.]{4}[\w.][\w.]{4}[A-Z])"
    r"(?![A-Za-z\d])"
)
REPO = re.compile(
    r"(github\.com|gitlab\.com|bitbucket\.org)/([\w.-]+)/([\w.-]+)"
)
ANCHOR = re.compile(r"references\.md#([\w-]+)")
URL = re.compile(r"https?://\S+")
URL_GROUP = re.compile(r"(https?://\S+)")

# Author-year citations. A surname may carry a prefix ("van der Marel",
# "De Prins"); the first author is the one that is matched.
_PREFIX = (
    r"(?:(?:van|von|de|der|den|del|della|di|du|le|la|ter|ten|dos|da"
    r"|Van|Von|De|Del|Di|Du|Le|La|Da)\s)*"
)
_NAME = r"[A-ZÀ-Þ][a-zß-ÿ'’\-]+(?:[A-Z][a-zß-ÿ]+)?"
_SUR = rf"{_PREFIX}{_NAME}"
_YEAR = r"(?:18|19|20)\d\d"
CITATION = re.compile(
    rf"(?<![\w/.\-])(?P<first>{_SUR})"
    rf"(?:(?:,\s+{_SUR})+,?\s+(?:&|and)\s+{_SUR}"
    rf"|\s+(?:&|and)\s+{_SUR}|,?\s+et\.?\s?al\.?)?"
    rf"(?:['’]s)?,?\s+[(\[]?(?P<years>{_YEAR}[a-z]?"
    rf"(?:,\s*(?:{_YEAR}[a-z]?|[a-z]\b))*)(?!\d|-\d)"
)
# Capitalized words that precede a year without being an author. Kept short:
# a new false positive is added here with the reason, not silenced elsewhere.
NOT_AUTHORS = {
    # months
    *"January February March April May June July August September October "
    "November December".split(),
    # document structure and software versions
    *"Version Release Copyright Section Table Figure Fig Eq Cycle Python "
    "Changelog NumPy SciPy PyTorch By The".split(),
}
# Pairs that read as a citation but name a project or event, not a paper.
NOT_CITATIONS: set[tuple[str, str]] = set()
# The generated page restates the TOML, so it cannot check it.
EXCLUDED = {"docs/references.md"}
SKIP_DIRS = {
    ".git",
    ".venv",
    ".claude",
    ".jax-cache",
    ".ipynb_checkpoints",
    "site",
    "build",
    "dist",
    "node_modules",
}
SKIP_PREFIXES = ("notebooks/archive/",)


def _files() -> list[Path]:
    """Every file whose citations must be in the TOML."""
    paths = [
        *ROOT.glob("src/virgil/**/*.py"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("scripts/*.py"),
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        # hand-written prose; the generated page restates the TOML
        ROOT / "references" / "references_intro.md",
    ]
    for dirpath, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        paths += [Path(dirpath, n) for n in names if n.endswith(".ipynb")]
    keep = set()
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if rel not in EXCLUDED and not rel.startswith(SKIP_PREFIXES):
            keep.add(p)
    return sorted(keep)


def _text(path: Path) -> str:
    """The prose and code of a file (notebook outputs excluded)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    cells = json.loads(raw).get("cells", [])
    return "\n".join("".join(c.get("source", [])) for c in cells)


def _unwrap(text: str) -> str:
    """Join wrapped lines (and comment or quote markers) into one."""
    return re.sub(r"\s*\n[\s#>*]*", " ", text)


def _repo(url: str) -> str | None:
    """``owner/name`` on GitHub, ``host/owner/name`` elsewhere."""
    m = REPO.search(url)
    if not m:
        return None
    host, owner, name = m.groups()
    name = name.rstrip(".").removesuffix(".git").rstrip(".")
    key = f"{owner}/{name}".lower()
    return key if host == "github.com" else f"{host}/{key}"


def _ascii(text: str) -> str:
    text = text.replace("ß", "ss").replace("Enßlin", "Ensslin")
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )


def _last_name(name: str) -> str:
    return _ascii(re.sub(r"['’]s$", "", name.split()[-1]))


def _first_surnames(authors: str) -> set[str]:
    """Surnames a citation of this entry may start with.

    The first author, or for "Enßlin group (Arras, Edenhofer et al.)" the
    group and the first name in the brackets.
    """
    group = re.match(r"(.*?) group \((.*)\)", authors)
    if group:
        inner = re.split(r",|&| and |et al", group.group(2))[0]
        return {_last_name(group.group(1)), _last_name(inner)}
    first = re.split(r",|&| and |\bet al", authors)[0]
    return {_last_name(first)}


def _years(entry: dict) -> set[str]:
    """The year, plus ``alt_years`` (which must say why in ``alt_years_note``)."""
    years = {str(entry.get("year"))}
    return years | {str(y) for y in entry.get("alt_years", [])}


def find_citations(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """``(first-author surname, years)`` for each author-year citation."""
    found = []
    for m in CITATION.finditer(URL.sub(" ", _unwrap(text))):
        first = m.group("first")
        if first in NOT_AUTHORS:
            continue
        years = tuple(re.findall(rf"{_YEAR}", m.group("years")))
        found.append((_last_name(first), years))
    return found


def _known_arxiv() -> set[str]:
    return {e["arxiv"].lower() for e in REFS.values() if "arxiv" in e}


def _known_dois() -> set[str]:
    return {e["doi"].lower() for e in REFS.values() if "doi" in e}


def _known_bibcodes() -> set[str]:
    return {e["bibcode"] for e in REFS.values() if "bibcode" in e}


def _known_repos() -> set[str]:
    repos = {
        _repo(e[field])
        for e in REFS.values()
        for field in ("url", "repo")
        if field in e
    }
    return (repos - {None}) | SELF_AND_DEPENDENCY_REPOS


def _matches(surname: str, year: str) -> bool:
    return any(
        year in _years(e) and _ascii(surname) in _first_surnames(e["authors"])
        for e in REFS.values()
    )


def _arxiv_id(groups: tuple[str | None, ...]) -> str:
    return re.sub(r"v\d+$", "", next(g for g in groups if g)).lower()


def _clean_url(url: str) -> str:
    """A link without trailing markup or punctuation, for comparing with ``url``."""
    url = url.rstrip(".,;:")
    while url.endswith((")", "]", '"', "'", ">", "*", "`")):
        if url.endswith(")") and url.count("(") >= url.count(")"):
            break
        url = url[:-1].rstrip(".,;:")
    return url.rstrip("/").lower()


def _clean_doi(doi: str) -> str:
    """Drop trailing punctuation and a closing bracket that is not the DOI's."""
    doi = doi.rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi.lower()


_CACHE: dict[Path, str] = {}


def _scan(pattern: re.Pattern, unwrap: bool = False):
    hits = []
    for path in _files():
        if path not in _CACHE:
            _CACHE[path] = _text(path)
        text = _unwrap(_CACHE[path]) if unwrap else _CACHE[path]
        rel = path.relative_to(ROOT).as_posix()
        hits += [(rel, m.groups()) for m in pattern.finditer(text)]
    return hits


def _all_citations():
    for path in _files():
        if path not in _CACHE:
            _CACHE[path] = _text(path)
        rel = path.relative_to(ROOT).as_posix()
        for surname, years in find_citations(_CACHE[path]):
            yield rel, surname, years


def test_every_arxiv_id_is_in_the_toml():
    known = _known_arxiv()
    missing = sorted(
        {
            f"{rel}: {_arxiv_id(g)}"
            for rel, g in _scan(ARXIV)
            if _arxiv_id(g) not in known
        }
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_every_doi_is_in_the_toml():
    known = _known_dois()
    missing = sorted(
        {
            f"{rel}: {_clean_doi(g[0])}"
            for rel, g in _scan(DOI)
            if _clean_doi(g[0]) not in known
        }
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_every_ads_bibcode_is_in_the_toml():
    known = _known_bibcodes()
    missing = sorted(
        {f"{rel}: {g[0]}" for rel, g in _scan(BIBCODE) if g[0] not in known}
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_every_repository_is_in_the_toml():
    known = _known_repos()
    missing = sorted(
        {
            f"{rel}: {_repo('/'.join(('github.com', *g[1:])) if g[0] == 'github.com' else '/'.join(g))}"
            for rel, g in _scan(REPO)
            if _repo("/".join(g)) not in known
        }
    )
    assert not missing, (
        f"add these to references/provenance.toml (url or repo) or, for the "
        f"project's own links, to SELF_AND_DEPENDENCY_REPOS: {missing}"
    )


def test_every_author_year_citation_is_in_the_toml():
    missing = sorted(
        {
            f"{rel}: {surname} ({year})"
            for rel, surname, years in _all_citations()
            for year in years
            if not _matches(surname, year)
            and (surname, year) not in NOT_CITATIONS
        }
    )
    assert not missing, f"add these to references/provenance.toml: {missing}"


def test_every_references_anchor_names_a_toml_key():
    bad = sorted(
        {f"{rel}: #{g[0]}" for rel, g in _scan(ANCHOR) if g[0] not in REFS}
    )
    assert not bad, f"references.md anchors with no TOML key: {bad}"


def test_every_local_anchor_in_the_references_intro_names_a_toml_key():
    intro = (ROOT / "references" / "references_intro.md").read_text(
        encoding="utf-8"
    )
    bad = sorted(set(re.findall(r"\]\(#([\w-]+)\)", intro)) - set(REFS))
    assert not bad, f"references_intro.md anchors with no TOML key: {bad}"


def test_alt_years_say_why():
    bad = [
        k
        for k, e in REFS.items()
        if e.get("alt_years") and not e.get("alt_years_note")
    ]
    assert not bad, f"alt_years needs alt_years_note: {bad}"


_UNVERIFIED_NOTE = re.compile(
    r"from memory|not checked|check pending|cited in the audit",
    re.I,
)


def test_verified_entries_have_no_unverified_notes():
    bad = [
        k
        for k, e in REFS.items()
        if e.get("verified") is True
        and _UNVERIFIED_NOTE.search(
            f"{e.get('note', '')} {e.get('notes', '')}"
        )
    ]
    assert not bad, (
        f"verified = true entries carry unverified/audit notes: {bad}"
    )


def _cited_keys() -> set[str]:
    """Keys of the entries that something in the scanned files cites."""
    cited = set()
    arxiv = {_arxiv_id(g) for _, g in _scan(ARXIV)}
    dois = {_clean_doi(g[0]) for _, g in _scan(DOI)}
    codes = {g[0] for _, g in _scan(BIBCODE)}
    repos = {_repo("/".join(g)) for _, g in _scan(REPO)}
    anchors = {g[0] for _, g in _scan(ANCHOR)}
    urls = {_clean_url(g[0]) for _, g in _scan(URL_GROUP)}
    pairs = {(s, y) for _, s, years in _all_citations() for y in years}
    for key, e in REFS.items():
        hit = (
            key in anchors
            or e.get("arxiv", "").lower() in arxiv
            or e.get("doi", "").lower() in dois
            or e.get("bibcode") in codes
            or any(_repo(e[f]) in repos for f in ("url", "repo") if f in e)
            or ("url" in e and _clean_url(e["url"]) in urls)
            or any(
                (s, y) in pairs
                for s in _first_surnames(e["authors"])
                for y in _years(e)
            )
        )
        if hit:
            cited.add(key)
    return cited


def test_every_entry_is_used_cited_or_bibliography_only():
    cited = _cited_keys()
    orphans = sorted(
        k
        for k, e in REFS.items()
        if not e.get("used_in")
        and e.get("role") != "bibliography"
        and k not in cited
    )
    assert not orphans, (
        f"cited nowhere and no used_in: {orphans}. Cite them, add used_in, "
        'or set role = "bibliography".'
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Kipping (2013)", [("kipping", ("2013",))]),
        ("Kipping 2013", [("kipping", ("2013",))]),
        ("Kipping's (2013) prior", [("kipping", ("2013",))]),
        ("Smith et al. [2020]", [("smith", ("2020",))]),
        ("Smith et. al. 2019", [("smith", ("2019",))]),
        ("Smith, et al. 2019", [("smith", ("2019",))]),
        ("Smith et al.\n(2019)", [("smith", ("2019",))]),
        ("Smith et\n al. (2019)", [("smith", ("2019",))]),
        ("Smith et al.\n# (2019)", [("smith", ("2019",))]),
        (
            "(Smith et al. 2019; Jones 2020)",
            [("smith", ("2019",)), ("jones", ("2020",))],
        ),
        ("Smith, Jones & Lee (2019)", [("smith", ("2019",))]),
        ("Smith, Jones, and Lee 2019", [("smith", ("2019",))]),
        ("Smith & Lee 2019a,b", [("smith", ("2019",))]),
        ("Smith et al. (2019, 2021)", [("smith", ("2019", "2021"))]),
        ("van der Marel et al. 2013", [("marel", ("2013",))]),
        ("De Prins (2024)", [("prins", ("2024",))]),
        ("Enßlin 2009", [("ensslin", ("2009",))]),
        ("Python 3", []),
        ("Version 2024", []),
        ("in March 2020 we", []),
        ("JWST 2022", []),
        ("NumPy 2024", []),
        ("See https://example.org/Smith/2019 for", []),
        ("Figure 2020", []),
        ("Apple M1 Max, 2026-10-19 scans", []),
        ("van Hamme 1993", [("hamme", ("1993",))]),
    ],
)
def test_the_citation_matcher_reads_each_form(text, expected):
    assert find_citations(text) == expected


def test_the_identifier_matchers_read_each_form():
    assert (
        ARXIV.search("arXiv:astro-ph/0508185").group(1) == "astro-ph/0508185"
    )
    assert ARXIV.search("arXiv:hep-th/9901001")
    html = ARXIV.search("https://arxiv.org/html/2509.25433v2")
    assert _arxiv_id(html.groups()) == "2509.25433"
    assert _arxiv_id(ARXIV.search("astro-ph/0508185").groups()) == (
        "astro-ph/0508185"
    )
    assert DOI.search("doi.org/10.1086/444523.").group(1).startswith("10.1086")
    assert _clean_doi("10.1086/444523.") == "10.1086/444523"
    assert _clean_doi("10.1016/0167-2789(92)90242-F)") == (
        "10.1016/0167-2789(92)90242-f"
    )
    assert _clean_doi("10.1016/0167-2789(92") == "10.1016/0167-2789(92"
    assert BIBCODE.search("2026PASP..138e4504D")
    assert BIBCODE.search("see 2025AJ....169..137B.")
    assert BIBCODE.search("2022SPIE12183E..1NM")
    assert _clean_url("https://www.netlib.org/cephes/).") == (
        "https://www.netlib.org/cephes"
    )
    assert not BIBCODE.search("2025 is a year")
    assert _repo("https://gitlab.com/Foo/Bar.git") == "gitlab.com/foo/bar"
    assert _repo("github.com/Foo/Bar.") == "foo/bar"


def test_the_matcher_checks_the_first_author_and_the_year_only():
    assert _matches("Kammerer", "2020") and not _matches("Kammerer", "1999")
    assert _matches("Ensslin", "2009") and _matches("Enßlin", "2009")
    # a later author, or a year found only in a note, does not count
    assert not _matches("Kloppenburg", "2010") and not _matches("Pope", "1999")
    assert _first_surnames("Enßlin group (Arras, Edenhofer et al.)") == {
        "ensslin",
        "arras",
    }
    assert _first_surnames("De Cao, Titov & Aziz") == {"cao"}


def _used_in_symbols():
    for key, entry in REFS.items():
        for item in entry.get("used_in", []):
            yield key, item


@pytest.mark.parametrize(
    ("key", "item"), sorted(_used_in_symbols()), ids=lambda v: str(v)
)
def test_used_in_symbols_import(key, item):
    module, _, symbol = item.partition(":")
    obj = importlib.import_module(module)
    for part in filter(None, symbol.split(".")):
        assert hasattr(obj, part), f"{key}: {item} does not import"
        obj = getattr(obj, part)
