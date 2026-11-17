<!-- AUTO-GENERATED FROM notebooks/amigo_disco.ipynb by scripts/sync_tutorial_docs.py. -->
# AMIGO mixed-DISCO products

The AMIGO pipeline writes filter-keyed mixed-DISCO products. `virgil.amigo.load_oi_data` loads each filter into an `OIData` whose model vector uses the stored log-amplitude and phase projection operators.

This notebook is short on purpose: once loaded, these data use the same `joint_loglike` and `joint_prediction` interfaces as the hierarchical inference tutorial.

## Load the product

The product is the file `calibrated_visibility.npy`, in the [`data/` folder of the repository](https://github.com/benjaminpope/virgil/blob/main/data/calibrated_visibility.npy); AMIGO writes products of this form. It is a pickled dictionary, and `load_oi_data` unpickles it, which can run arbitrary code: **load only products you trust**, such as your own pipeline's output.

??? note "Imports and setup"

    ```python
    import warnings
    from pathlib import Path

    import jax.numpy as jnp

    from virgil.amigo import load_oi_data
    from virgil.likelihood import joint_prediction, whitened_residuals
    from virgil.models import BinaryModelCartesian

    warnings.filterwarnings("ignore", category=SyntaxWarning)  # upstream zodiax docstrings

    # The tutorial's product lives in the repository's data/ folder; a run from a
    # clone finds it there. With the package installed, point this at your own product.
    product_path = next(
        p / "data" / "calibrated_visibility.npy"
        for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "data" / "calibrated_visibility.npy").exists()
    )
    ```

```python
observations_by_filter = load_oi_data(product_path)
filter_names = tuple(observations_by_filter)
observations = tuple(observations_by_filter[name] for name in filter_names)

for name, oidata in observations_by_filter.items():
    print(f"Filter: {name}")
    print(f"Number of observables: {int(oidata.flatten_data()[0].size)}")
    print(f"Number of UV points: {int(oidata.u.size)}")
    print(f"Observable kind: {oidata.vis_encoding}")
    print()
```

```text
Filter: F380M
Number of observables: 206
Number of UV points: 47
Observable kind: mixed_log_complex

Filter: F430M
Number of observables: 210
Number of UV points: 47
Observable kind: mixed_log_complex

Filter: F480M
Number of observables: 194
Number of UV points: 47
Observable kind: mixed_log_complex
```

## Verify one mixed-DISCO projection

The product stores one independent data vector per filter. For a complex visibility $V$, its model vector is

$$y = A_{\mathrm{logamp}}\,\mathrm{Re}\log V + A_{\mathrm{phase}}\,\mathrm{Im}\log V,$$

where `A_logamp` and `A_phase` are the product's two projection matrices. `OIData.model` applies this for you.

```python
oidata = observations_by_filter["F430M"]
binary = BinaryModelCartesian(dra=-34.7, ddec=197.0, flux=1e-3)

prediction = oidata.model(binary)
data, errors = oidata.flatten_data()

prediction.shape, data.shape, errors.shape
```

```text
((210,), (210,), (210,))
```

## Use the shared hierarchical interface

The loaded filters are ordinary `OIData` objects. A hierarchical model supplies shared astrometry and one flux per filter; the hierarchical inference tutorial shows the full grid, optimization and HMC workflow.

```python
params = {
    "dra": jnp.array(-34.7),
    "ddec": jnp.array(197.0),
    "log10_flux": jnp.log10(jnp.full(len(observations), 1e-3)),
}


def binary_model(values, observation_index):
    return BinaryModelCartesian(
        values["dra"],
        values["ddec"],
        10.0 ** values["log10_flux"][observation_index],
    )


for i, name in enumerate(filter_names):
    r = whitened_residuals(binary_model(params, i), observations[i])
    print(f"{name}: chi2/N = {float(jnp.sum(r**2) / r.size):.1f} on the quoted errors (N = {r.size})")
print("joint prediction:", joint_prediction(params, binary_model, observations).shape)
```

```text
F380M: chi2/N = 1.2 on the quoted errors (N = 206)
F430M: chi2/N = 1.0 on the quoted errors (N = 210)
```

```text
F480M: chi2/N = 0.9 on the quoted errors (N = 194)
joint prediction: (610,)
```

$\chi^2/N$ is close to one in every filter on the quoted errors, so the parameters above describe this product well.
