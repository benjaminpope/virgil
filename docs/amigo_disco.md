<!-- AUTO-GENERATED FROM notebooks/amigo_disco.ipynb by scripts/sync_tutorial_docs.py. -->
# AMIGO mixed-DISCO products

AMIGO pipeline reductions write filter-keyed mixed-DISCO products. `virgil.amigo.load_oi_data` loads each filter into an `OIData` object whose standardized model vector uses the stored log-amplitude and phase projection operators.

This notebook is deliberately short: once loaded, these observations use the same `joint_loglike` and `joint_prediction` interfaces as the complete hierarchical inference tutorial.

## Load the product

```python
import sys
from pathlib import Path

import jax.numpy as jnp

repo_root = Path.cwd()
if not (repo_root / "src").exists():
    repo_root = repo_root.parent
for path in (repo_root, repo_root / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from virgil.amigo import load_oi_data
from virgil.likelihood import joint_loglike, joint_prediction
from virgil.models import BinaryModelCartesian

product_path = repo_root / "data" / "calibrated_visibility.npy"
observations_by_filter = load_oi_data(product_path)
filter_names = tuple(observations_by_filter)
observations = tuple(observations_by_filter[name] for name in filter_names)

for name, oidata in observations_by_filter.items():
    print(f"Filter: {name}")
    print(f"Number of observables: {int(oidata.flatten_data()[0].size)}")
    print(f"Number of UV points: {int(oidata.u.size)}")
    print(f"Observable kind: {oidata.observable_kind}")
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

The product stores one independent data vector per filter. For a complex visibility `cvis`, its model vector is

```python
A_logamp @ log(cvis).real + A_phase @ log(cvis).imag
```

`OIData.model` applies that relation automatically.

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

The loaded filters are ordinary `OIData` objects. A hierarchical model supplies shared astrometry and one flux per filter; the complete grid, optimization, and HMC workflow is shown in the hierarchical inference tutorial.

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


joint_prediction(params, binary_model, observations).shape, joint_loglike(
    params, binary_model, observations
)
```

```text
((610,), Array(14801.682, dtype=float32))
```
