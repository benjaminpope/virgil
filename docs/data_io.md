<!-- AUTO-GENERATED FROM notebooks/data_io.ipynb by scripts/sync_tutorial_docs.py. -->
# Data I/O

`virgil` reads and writes `.oifits` files, the data standard in interferometry, with `astropy.io.fits` (see `virgil.oifits`). The older writers derived from [`ImPlaneIA`](https://github.com/anand0xff/ImPlaneIA), in `virgil.legacy.oifits_implaneia`, are still available for existing scripts.

```python
import copy
import sys
from pathlib import Path

import numpy as np

from virgil.oidata import OIData
from virgil.oifits import write_oifits

notebook_dir = (
    (Path.cwd() / "notebooks")
    if (Path.cwd() / "notebooks").exists()
    else Path.cwd()
)
if str(notebook_dir) not in sys.path:
    sys.path.insert(0, str(notebook_dir))

from tutorial_helpers import find_repo_root, load_synthetic_workflow_module

repo_root = find_repo_root()
module = load_synthetic_workflow_module(repo_root)
```

## Simulate Data
Let's simulate some synthetic data and save it to an `.oifits` file. The synthetic data are a dictionary of OIFITS tables (`OI_WAVELENGTH`, `OI_VIS`, `OI_VIS2`, `OI_T3`, plus an `info` dictionary for the header), with phases in degrees as in OIFITS:

```python
out = (
    repo_root / "docs" / "generated" / "synthetic_binary_from_notebook.oifits"
).resolve()

synth_dict, truth, noise_settings = module._build_synthetic_oifits_dict(seed=4)
_ = write_oifits(synth_dict, out)
```

# Reading Data

Let's read the data - this is easy!

```python
# A path works, as does a file opened with astropy.io.fits or pyoifits.
oidata = OIData(out)
```

## OIData Object

OIData knows automatically whether you're using visibilities or squared visibilities, which are just saved as `oidata.vis` with uncertainty `oidata.d_vis` and toggled with `oidata.v2_flag`. 

Likewise OIData can tell if you're using closure phases or absolute phases, which are saved just as `oidata.phi` with uncertainty `oidata.d_phi` and toggled with `oidata.cp_flag`.

$u,v$ information is saved in `oidata.u` and `oidata.v`, with closure phase indices in `oidata.i_cps1`, `oidata.i_cps2`, `oidata.i_cps3`.

When you're using this, you will pass it a model object, which will automatically evaluate it at the appropriate arguments.

Phases are always stored in radians, whatever the unit in the file. Flagged points (the OIFITS `FLAG` column, or non-finite values) are left out of the observables.

```python
# let's have a tour of the oidata object
print(oidata)
```

```text
OIData(
  u=f32[6],
  v=f32[6],
  wavel=f32[1],
  vis=f32[6],
  d_vis=f32[6],
  phi=f32[4],
  d_phi=f32[4],
  i_cps1=i32[4],
  i_cps2=i32[4],
  i_cps3=i32[4],
  vis_mat=None,
  phi_mat=None,
  vis_index=None,
  phi_index=None,
  uv_grid=None,
  cp_noise=ClosureNoise(
    groups=i32[1,4](numpy),
    mask=bool[1,4](numpy),
    incidence=f64[1,4,6](numpy),
    basis=f64[1,3,4](numpy),
    chol=f64[1,3,3](numpy),
    valid=bool[1,3](numpy),
    keep=i32[3](numpy)
  ),
  dt=f32[6],
  frame=i32[6],
  stations=i32[6,2],
  gains=None,
  phase_offsets=None,
  extras=(),
  observable_kind='split',
  vis_mode='v2',
  v2_flag=True,
  cp_flag=True,
  t_ref=61000.0
)
```

## Verification
And just to verify, let's make sure all the keys are saved and loaded correctly:

```python
# Compare arrays written to OIFITS with arrays reloaded via OIData.
# OIData converts OIFITS phase columns from degrees to internal radians.
# We test all core OIData keys in one cell.
expected = {
    "u": np.asarray(synth_dict["OI_VIS2"]["UCOORD"]),
    "v": np.asarray(synth_dict["OI_VIS2"]["VCOORD"]),
    "vis": np.asarray(synth_dict["OI_VIS2"]["VIS2DATA"]),
    "d_vis": np.asarray(synth_dict["OI_VIS2"]["VIS2ERR"]),
    "phi": np.deg2rad(np.asarray(synth_dict["OI_T3"]["T3PHI"])),
    "d_phi": np.deg2rad(np.asarray(synth_dict["OI_T3"]["T3PHIERR"])),
    "wavel": np.atleast_1d(
        np.asarray(synth_dict["OI_WAVELENGTH"]["EFF_WAVE"])
    ),
}

reloaded = {
    "u": np.asarray(oidata.u),
    "v": np.asarray(oidata.v),
    "vis": np.asarray(oidata.vis),
    "d_vis": np.asarray(oidata.d_vis),
    "phi": np.asarray(oidata.phi),
    "d_phi": np.asarray(oidata.d_phi),
    "wavel": np.atleast_1d(np.asarray(oidata.wavel)),
}

equality = {
    key: bool(np.allclose(reloaded[key], expected[key], equal_nan=True))
    for key in expected
}

print("All arrays equal:", all(equality.values()))
for key, equal in equality.items():
    print(f"{key}: {'equal' if equal else 'not equal'}")
```

```text
All arrays equal: True
u: equal
v: equal
vis: equal
d_vis: equal
phi: equal
d_phi: equal
wavel: equal
```

## Several Wavelengths, Flags and Targets

Files with several wavelength channels work the same way: every (baseline, wavelength) sample becomes one entry of `oidata.u`, `oidata.v` and `oidata.wavel`, and closure phases are built from visibilities at their own wavelength. Here we copy the synthetic data into three channels (repeating the same values, just for illustration) and flag one squared visibility:

```python
multi = copy.deepcopy(synth_dict)
multi["OI_WAVELENGTH"] = {
    "EFF_WAVE": np.array([4.6e-6, 4.8e-6, 5.0e-6]),
    "EFF_BAND": np.full(3, 0.1e-6),
}
data_columns = {
    "OI_VIS": ["VISAMP", "VISAMPERR", "VISPHI", "VISPHIERR"],
    "OI_VIS2": ["VIS2DATA", "VIS2ERR"],
    "OI_T3": ["T3AMP", "T3AMPERR", "T3PHI", "T3PHIERR"],
}
for table, columns in data_columns.items():
    for column in columns:
        values = np.asarray(multi[table][column])
        multi[table][column] = np.repeat(values[:, None], 3, axis=1)
    multi[table].pop("FLAG")

flag = np.zeros(multi["OI_VIS2"]["VIS2DATA"].shape, dtype=bool)
flag[0, 1] = True  # first baseline, middle channel
multi["OI_VIS2"]["FLAG"] = flag

multi_out = out.with_name("synthetic_binary_three_channels.oifits")
write_oifits(multi, multi_out)
multi_data = OIData(multi_out)

print("samples:", multi_data.u.size, "wavelengths:", np.unique(multi_data.wavel))
print("V2 observables:", multi_data.vis.size, "(one flagged)")
print("closure phases:", multi_data.phi.size)
```

```text
samples: 18 wavelengths: [4.6e-06 4.8e-06 5.0e-06]
V2 observables: 17 (one flagged)
closure phases: 12
```

If a file holds data on several targets (e.g. a science target and its calibrator), pick one with `OIData(path, target="name")` or by `TARGET_ID`; `OIData` refuses to mix them silently.
