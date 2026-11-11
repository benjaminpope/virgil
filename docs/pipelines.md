<!-- AUTO-GENERATED FROM notebooks/pipelines.ipynb by scripts/sync_tutorial_docs.py. -->
# Pipelines: one call, one folder

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

> **Install:** this tutorial needs the `pipeline` extra: `pip install "virgil-astro[pipeline]"`.

The other tutorials build an analysis one step at a time, which is how you learn it. Once you know the steps, you will want to run them the same way on every target and come back to the results a month later without rerunning anything. That is what a *pipeline* is for: a fixed sequence of named stages that each write their outputs into a run folder, finish with quality checks, and can be reloaded in a fresh session.

A pipeline adds no science. Every number it reports comes from a public virgil function that you can call yourself, as the "As a pipeline" sections of the [binary search](binary_search.md), [contrast limits](contrast_limits.md) and [limb darkening](limb_darkening.md) tutorials show. This page is about the machinery around those functions. We use the smallest problem that shows all of it: a simulated companion around a star, seen through a seven-hole aperture mask.

??? note "Imports and setup"

    ```python
    import json
    import subprocess
    import sys
    from pathlib import Path
    from tempfile import mkdtemp

    import jax
    import numpy as onp
    from IPython.display import Image, display

    from virgil.coverage import nrm_oidata
    from virgil.models import BinaryModelCartesian
    from virgil.pipeline import BinaryPipeline, ConfigMismatchError, load
    ```

## Simulate a binary

[`nrm_oidata`](api/coverage.md) gives the uv coverage of the NIRISS aperture mask with zero data, and `with_model` fills it from a model with Gaussian noise. Here the companion is 3% as bright as the star, about 117 mas away to the south-east.

```python
truth = BinaryModelCartesian(dra=100.0, ddec=-60.0, flux=0.03)
data = nrm_oidata().with_model(truth, key=jax.random.PRNGKey(0))
print(data.vis.size, "squared visibilities and", data.phi.size, "closure phases")
```

## Run it

A pipeline takes the data, a model template (its class sets the parameterization of the answer; the fit starts from the grid search's best point, so the template's values need no tuning), a run folder and keyword settings. We keep the sampler short with `num_warmup`, `num_samples` and `num_chains` so that the whole run takes a minute or so. A seven-hole mask has its holes on a triangular lattice, so its visibilities and closure phases cannot tell a companion from one displaced by a lattice vector, and the nearest of these are 776 mas away. On such coverage the default search grid is limited to 336 mas, half the smallest larger coordinate of an alias vector (672 mas), so that no two points of the square differ by a lattice vector; the summary records the limit as `search.alias_cap_mas`. Every other setting is the default.

```python
work = Path(mkdtemp())
template = BinaryModelCartesian(dra=0.0, ddec=0.0, flux=1e-3)

res = BinaryPipeline(
    data,
    template,
    output=work / "binary_run",
    num_warmup=500,
    num_samples=500,
    num_chains=2,
).run()
res.status
```

## The run folder

Everything the run produced is in `output`, as JSON, HDF5, NumPy, PNG and OIFITS files, with no pickles. Here is the layout (the files inside each stage's folder are left out):

```python
for p in sorted(res.path.rglob("*")):
    rel = p.relative_to(res.path)
    if len(rel.parts) <= 2:
        print(f"{'  ' * (len(rel.parts) - 1)}{rel.name}{'/' if p.is_dir() else ''}")
```

`run.json` records the resolved settings, the sha256 of the inputs, the versions of the libraries, and the status and timings of each stage. `summary.json` holds the key numbers and the checks. `plots/` has the figures, and `quicklook.ipynb` is an executed notebook that reloads the folder and shows them, so a collaborator can open the results without installing anything beyond Jupyter.

## Reading the results

`res` is a `Result`. Its `summary` is a plain dictionary of numbers by section, so you can pull out whatever you need:

```python
comp = res.summary["companion"]
print(f"fitted: {comp['sep_mas']:.1f} mas, PA {comp['pa_deg']:.1f} deg")
print(f"truth:  {(100.0**2 + 60.0**2) ** 0.5:.1f} mas, PA {onp.degrees(onp.arctan2(100.0, -60.0)):.1f} deg")
```

`describe()` prints the key numbers, the checks and a count of any warnings raised inside the stages in one block. Look at the checks first. The first is always the $\chi^2$ per independent datum on the *quoted* errors, because a fitted error scale makes $\chi^2$ close to one by construction and tells you nothing.

```python
print(res.describe())
```

`res.checks` has the same checks as objects with `name`, `status` (`pass`, `warn` or `fail`), `value`, `threshold` and `message`, which is what you want in a script that has to decide whether to trust a run:

```python
[(c.name, c.status) for c in res.checks]
```

The figures live in `res.plots`. This is the contrast curve from the `limits` stage: the faintest companion the data could have shown at each separation.

??? note "Plotting code"

    ```python
    print([p.name for p in res.plots])
    display(Image(filename=str(res.path / "plots" / "limits_contrast_curve.png")))
    ```

## Loading a run instead of rerunning it

In a fresh session, point `load` at the folder. Nothing is recomputed: each accessor reads the files the run wrote.

```python
again = load(res.path)
model = again.model()  # the best-fit model, a real virgil object
print(type(model).__name__, model.dra, model.ddec, model.flux)
print(again.samples()["dra"].shape)  # (chain, draw)
```

`data()` returns the data as fitted, `fit_result()` the maximum-likelihood fit (to continue with [`fit`](api/fitting.md)), and `grid()` the search maps. The [pipeline reference](pipeline.md) lists every accessor.

## Stopping early and resuming

A non-detection does not need a fit or a sampler. `run(through="limits")` stops after the named stage, and the run is then `partial`. Here is a fresh folder for the same data, run through the contrast limits only:

```python
short = BinaryPipeline(
    data, template, output=work / "short_run",
    num_warmup=500, num_samples=500, num_chains=2,
)
res2 = short.run(through="limits")
print(res2.status)
print({name: s["status"] for name, s in res2.run["stages"].items()})
```

Run it again without `through` and it carries on from where it stopped. A stage that completed, with its outputs still on disk, is skipped. Once a stage reruns, every later stage reruns too, since they depend on it.

```python
started = {name: s["started"] for name, s in res2.run["stages"].items()}
res2 = short.run()
for name, s in res2.run["stages"].items():
    print(f"{name:10s} {'skipped' if started.get(name) == s['started'] else 'ran'}")
print(res2.status)
```

A resumed run refuses to continue if anything it depends on has changed: the settings, the data or the model template. Without this you could mix the stages of two different analyses in one folder, and the summary would describe neither.

```python
changed = BinaryPipeline(
    data, template, output=work / "short_run",
    num_warmup=500, num_samples=500, num_chains=2, sigma=5.0,
)
try:
    changed.run()
except ConfigMismatchError as err:
    print(err)
```

The message names each difference. Either use a new folder, or start this one over with `run(resume=False)`.

## Settings

Settings resolve in three layers, and each overrides the one before: the class defaults, then a JSON config file, then keyword arguments. `defaults()` shows what you can set:

```python
BinaryPipeline.defaults()
```

A typo raises a `TypeError` that lists the valid names. For a config file, give the data (OIFITS paths, relative to the file), the run folder and any settings. We reuse the data the first run fitted, which it wrote as `data/processed.oifits`, so the config reads a real file:

```python
config = work / "config.json"
config.write_text(json.dumps({
    "data": [str(res.path / "data" / "processed.oifits")],
    "output": str(work / "config_run"),
    "settings": {"num_warmup": 500, "num_samples": 500, "num_chains": 2, "sigma": 2.0},
}, indent=2))

p = BinaryPipeline.from_config(config, template, sigma=4.0)  # keywords win
print(p.settings["sigma"], p.settings["num_chains"])
p.run(through="load")  # cheap: writes run.json with the resolved settings
print(load(p.output).run["config"]["sigma"])
```

The config file says `sigma` is 2, the keyword says 4, and the keyword wins; `num_chains` comes from the file; everything else is a default. The resolved settings are written to `run.json`, so the folder records what was actually used.

## From the command line

Everything above is also available from a shell as `virgil-pipeline`, which is convenient for batch jobs and for running over many OIFITS files:

```bash
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --set sigma=3 --set num_chains=2
virgil-pipeline binary hd1234.oifits -o runs/hd1234 --through limits --fresh
virgil-pipeline info runs/hd1234        # status and checks
virgil-pipeline validate runs/hd1234    # complete, and every file matches its hash
virgil-pipeline quicklook runs/hd1234   # rebuild the quicklook notebook
```

`--set KEY=VALUE` overrides a setting, `--fresh` starts over, and `info` and `validate` only read JSON and hashes. Here is the same thing on our run, called from the notebook:

```python
def cli(*args):
    out = subprocess.run(
        [sys.executable, "-m", "virgil.pipeline.cli", *map(str, args)],
        capture_output=True, text=True,
    )
    print(out.stdout + out.stderr)
    if out.returncode:
        raise RuntimeError(f"virgil-pipeline exited with {out.returncode}")

cli("info", res.path)
cli("validate", res.path)
```

```python
cli(
    "binary", res.path / "data" / "processed.oifits",
    "-o", work / "cli_run", "--through", "limits",
    "--set", "sigma=3", "--set", "n_flux=20",
)
cli("info", work / "cli_run")
```

## Where next

The [pipeline reference](pipeline.md) lists the stages and the checks, and the [`StarPipeline`](pipeline.md#starpipeline-stellar-diameters) and [`DiskPipeline`](pipeline.md#diskpipeline-rings-and-haloes-provisional) work the same way. Because every number comes from a public function, you can always take a run apart: `res.model()` and `res.data()` give you the objects to carry on with by hand.
