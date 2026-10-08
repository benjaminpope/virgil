"""The ``virgil-pipeline`` command.

    virgil-pipeline binary DATA... --output DIR [--through S] [--fresh]
                                   [--set key=value ...]
    virgil-pipeline info DIR
    virgil-pipeline quicklook DIR [--no-execute]
    virgil-pipeline validate DIR

``info`` and ``validate`` read only JSON and file hashes: they import
neither the pipeline classes nor the notebook tools, and do no JAX work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._io import read_json, sha256_file

_PIPELINES = {"binary": ("virgil.pipeline.binary", "BinaryPipeline")}


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _settings(pairs):
    settings = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value; got {pair!r}.")
        settings[key.strip()] = _parse_value(value)
    return settings


def _run(args):
    import importlib

    module, name = _PIPELINES[args.command]
    cls = getattr(importlib.import_module(module), name)
    pipeline = cls.from_oifits(
        args.data, output=args.output, **_settings(args.set)
    )
    res = pipeline.run(through=args.through, resume=not args.fresh)
    print(res.describe())
    return 0


def _info(args):
    folder = Path(args.dir)
    run = read_json(folder / "run.json")
    print(f"{run['class']} ({run['stability']}), schema {run['schema']}")
    print(f"status: {run['status']}")
    if run.get("error"):
        error = run["error"]
        print(
            f"error in {error['stage']}: {error['type']}: {error['message']}"
        )
    for name, entry in run["stages"].items():
        print(
            f"  {name:<10} {entry['status']:<9} {entry.get('seconds', 0):8.1f} s"
        )
    summary_file = folder / "summary.json"
    if summary_file.exists():
        summary = read_json(summary_file)
        for check in summary.get("checks", []):
            print(f"  [{check['status']}] {check['name']}: {check['message']}")
    return 0


def _validate(args):
    folder = Path(args.dir)
    problems = []
    try:
        run = read_json(folder / "run.json")
    except (OSError, ValueError) as err:
        print(f"invalid: cannot read run.json ({err})")
        return 1
    if run.get("schema") != "virgil-pipeline-run-v1":
        problems.append(f"unknown schema {run.get('schema')!r}")
    if run.get("status") not in ("running", "partial", "complete", "failed"):
        problems.append(f"unknown status {run.get('status')!r}")
    for key in ("pipeline", "config", "inputs", "provenance", "stages"):
        if key not in run:
            problems.append(f"run.json lacks {key!r}")
    try:
        summary = read_json(folder / "summary.json")
        if summary.get("schema") != run.get("schema"):
            problems.append("summary.json schema differs from run.json")
    except (OSError, ValueError) as err:
        problems.append(f"cannot read summary.json ({err})")
    for name, entry in run.get("stages", {}).items():
        if entry.get("status") != "complete":
            continue
        for rel in entry.get("outputs", []):
            if not (folder / rel).exists():
                problems.append(f"stage {name}: missing {rel}")
    for rel, digest in (run.get("files") or {}).items():
        path = folder / rel
        if not path.exists():
            problems.append(f"missing {rel}")
            continue
        h = sha256_file(path)
        if h != digest:
            problems.append(f"{rel} changed since the run")
    if problems:
        print("invalid:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"valid: {run['class']} run, status {run['status']}")
    return 0


def _quicklook(args):
    from ._quicklook import write_quicklook

    path = write_quicklook(args.dir, execute=not args.no_execute)
    print(path)
    return 0


def build_parser():
    """The argument parser of ``virgil-pipeline``."""
    parser = argparse.ArgumentParser(
        prog="virgil-pipeline",
        description="Run and inspect virgil pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in _PIPELINES:
        p = sub.add_parser(name, help=f"run the {name} pipeline")
        p.add_argument("data", nargs="+", help="OIFITS files, read together")
        p.add_argument("--output", "-o", default="run", help="run folder")
        p.add_argument("--through", help="last stage to run")
        p.add_argument(
            "--fresh",
            action="store_true",
            help="start over instead of resuming",
        )
        p.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a setting (VALUE is parsed as JSON if it can be)",
        )
        p.set_defaults(func=_run)
    p = sub.add_parser("info", help="print a run's status and checks")
    p.add_argument("dir")
    p.set_defaults(func=_info)
    p = sub.add_parser("validate", help="check a run folder is complete")
    p.add_argument("dir")
    p.set_defaults(func=_validate)
    p = sub.add_parser("quicklook", help="rebuild a run's quicklook notebook")
    p.add_argument("dir")
    p.add_argument("--no-execute", action="store_true")
    p.set_defaults(func=_quicklook)
    return parser


def main(argv=None):
    """Entry point of ``virgil-pipeline``."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
