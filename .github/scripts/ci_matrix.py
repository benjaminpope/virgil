"""Choose the job matrix of tests.yml.

    EVENT=pull_request FULL_CI=true PR=123 python3 .github/scripts/ci_matrix.py

(``NIGHTLY=true`` makes any event the nightly tier: a hand-dispatched
nightly. ``TIER=main`` makes a dispatch the main tier, which
lock-upgrade.yml uses; an empty ``TIER`` leaves a dispatch full.)

prints ``matrix=<json>`` for ``$GITHUB_OUTPUT``. Tiers:

| tier    | trigger                      | jobs                                    |
|---------|------------------------------|-----------------------------------------|
| gate    | pull request                 | py3.12 float32 and float64, 3 shards    |
|         |                              | each, no slow tests; a PR touching the  |
|         |                              | orbit modules or the disk pipeline adds |
|         |                              | their slow tests, one changing          |
|         |                              | pyproject.toml adds the floors          |
| main    | push to main                 | the gate, with the slow tests except    |
|         |                              | the `full_only` ones                    |
|         | dispatch with `tier=main`    | the same (the daily lock-upgrade PR)    |
| full    | `full-ci` label, dispatch    | main + the `full_only` tests in 6       |
|         |                              | float32 and 5 float64 shards, macOS     |
|         |                              | py3.11, py3.13 and the floors (the last |
|         |                              | two fast tier, 2 shards each)           |
| nightly | daily schedule               | full, all with slow tests except macOS  |
|         |                              | (fast tier); py3.13 and the floors in 6 |
|         |                              | shards each (24 jobs in all)            |

The float64 gate jobs skip the dtype-independent test files, so they add
about 0.57x the float32 suite, not 2x. The lock-upgrade PR tests the newest
dependencies before they merge, so the nightly has no separate canary.

The full tiers pack their shards by ``tests/test_durations_full.json`` (with
the ``full_only`` tests), the others by ``tests/test_durations.json``. The
shard counts differ between the gate and the full tiers, so a shard's job name
(``1/3``, ``1/6``) does too; the gate's names are the same on every pull
request. Standard library and the ``gh`` CLI only.
"""

import json
import os
import re
import subprocess

# The orbit modules, the disk pipeline and their tests: a pull request
# touching one runs the slow tests of these test files too
# (tests/conftest.py, VIRGIL_SLOW_FILES).
ORBIT_PATHS = re.compile(
    r"^(src/virgil/(aliases|arcs|epochs|orbit[a-z_]*|likelihood|tempering|_marginal|_precision|pipeline/(disk|image|orbit))"
    r"|tests/test_(aliases|arcs|epochs|epoch_coarse|orbit[a-z_]*|tempering|pipeline_(disk|image|orbit[a-z_]*)))\.py$"
)
ORBIT_SLOW_FILES = (
    "test_aliases.py test_arcs.py test_epochs.py test_epoch_coarse.py test_orbit*.py test_tempering.py "
    "test_pipeline_disk.py test_pipeline_image.py test_pipeline_orbit_gap.py"
)

FULL_F32_SHARDS = 6
FULL_X64_SHARDS = 5
NIGHTLY_FLOOR_SHARDS = 6

LINUX = {"os": "ubuntu-latest", "python-version": "3.12", "x64": "0"}


def changed_flags(files):
    """(pyproject changed, orbit module or test changed) for a file list."""
    return (
        "pyproject.toml" in files,
        any(ORBIT_PATHS.match(f) for f in files),
    )


def choose(
    event, full_label=False, pyproject=False, orbit_slow=False, tier=""
):
    """The job matrix for a workflow event, as ``{"include": [...]}``."""
    nightly = event == "schedule"
    main_tier = event == "push" or (
        event == "workflow_dispatch" and tier == "main"
    )
    full = (
        nightly
        or full_label
        or (event == "workflow_dispatch" and not main_tier)
    )
    slow = full or main_tier
    orbit_files = ORBIT_SLOW_FILES if orbit_slow and not slow else ""

    # The full tiers also run the `full_only` tests, which roughly triple the
    # float32 suite (tests/test_durations_full.json: about 220 estimated
    # minutes on one runner before sharding): more shards keep each job near
    # half of the nightly's 90-minute cap. The largest file,
    # test_orbit_cells.py, is about 44 minutes alone.
    n_shards = (
        {"0": FULL_F32_SHARDS, "1": FULL_X64_SHARDS}
        if full
        else {"0": 3, "1": 3}
    )

    # Whole files per shard, packed by tests/test_durations.json (the full
    # tiers: test_durations_full.json). The
    # per-test budget (tests/conftest.py) is enforced on these Linux py3.12
    # jobs, whose timings it was set from. x64 exercises the float64 paths
    # on the numerical test files only.
    include = [
        {
            **LINUX,
            "x64": x64,
            "shard": f"{k}/{n_shards[x64]}",
            "full_durations": full,
            "slow": slow,
            "orbit_slow": orbit_files,
            "budget": True,
            # A cold float32 main run with every slow test outran the
            # 45-minute limit; the `full_only` ones wait for the full tiers.
            "skip_full_only": slow and not full,
        }
        for x64 in ("0", "1")
        for k in range(1, n_shards[x64] + 1)
    ]
    # py3.13 and the floors run the slow and `full_only` tests in the nightly.
    floor_shards = NIGHTLY_FLOOR_SHARDS if nightly else 2
    if full:
        # A float32 platform check: macOS py3.11, fast tier. It does not
        # test x64 (the Linux gate jobs do).
        include.append(
            {
                "os": "macos-latest",
                "python-version": "3.11",
                "x64": "0",
                "slow": nightly,
                # Unsharded, its 3 cores took over 3 hours for the whole
                # suite: the nightly's 90-minute limit fits the fast tier.
                "fast_only": True,
            }
        )
        # Two shards each: unsharded, py3.13 and the floors outgrew the
        # 30-minute pull-request limit (tests.yml), which stays as the hang
        # detector.
        include += [
            {
                **LINUX,
                "python-version": "3.13",
                "slow": nightly,
                "shard": f"{k}/{floor_shards}",
                "full_durations": nightly,
            }
            for k in range(1, floor_shards + 1)
        ]
    if full or pyproject:
        # Every direct dependency at the oldest version pyproject.toml
        # allows.
        include += [
            {
                **LINUX,
                "python-version": "3.11",
                "lowest": True,
                "slow": nightly,
                "shard": f"{k}/{floor_shards}",
                "full_durations": nightly,
            }
            for k in range(1, floor_shards + 1)
        ]

    for job in include:
        shard = job.get("shard", "").replace("/", "of")
        variant = "-lowest" if job.get("lowest") else ""
        job["id"] = (
            f"{job['os']}-py{job['python-version']}-x64{job['x64']}"
            f"{variant}{'-shard' + shard if shard else ''}"
        )
    return {"include": include}


def main():
    event = os.environ["EVENT"]
    nightly = os.environ.get("NIGHTLY") == "true"
    pyproject = orbit_slow = False
    if event == "pull_request":
        files = subprocess.run(
            ["gh", "pr", "diff", os.environ["PR"], "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        pyproject, orbit_slow = changed_flags(files)
    matrix = choose(
        "schedule" if nightly else event,
        os.environ.get("FULL_CI") == "true",
        pyproject,
        orbit_slow,
        os.environ.get("TIER", ""),
    )
    print("matrix=" + json.dumps(matrix))


if __name__ == "__main__":
    main()
