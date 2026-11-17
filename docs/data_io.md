<!-- AUTO-GENERATED FROM notebooks/data_io.ipynb by scripts/sync_tutorial_docs.py. -->
# Data I/O

`virgil` reads and writes `.oifits` files, the interferometry data standard, with `astropy.io.fits` (see `virgil.oifits`). The older writers derived from [`ImPlaneIA`](https://github.com/anand0xff/ImPlaneIA), in `virgil.legacy.oifits_implaneia`, remain for existing scripts. This page shows how to write a file, read it back into an `OIData`, and check what you got.

??? note "Imports and setup"

    ```python
    import copy
    import tempfile
    import warnings
    from itertools import combinations
    from pathlib import Path

    import numpy as np

    from virgil.models import BinaryModelCartesian
    from virgil.oidata import OIData, closure_phases, cp_indices
    from virgil.oifits import write_oifits

    warnings.filterwarnings("ignore", category=SyntaxWarning)  # upstream zodiax docstrings
    ```

## Simulate Data
First, simulate some data and save it to an `.oifits` file. The data are a dictionary of OIFITS tables (`OI_WAVELENGTH`, `OI_VIS`, `OI_VIS2`, `OI_T3`, plus an `info` dictionary for the header), with phases in degrees as OIFITS stores them:

??? note "Code"

    ```python
    # A four-station array, its six baselines and four closure triangles
    # (stations are numbered from 1, as OIFITS does), and a binary 0.32 per cent as bright.
    stations = np.array([[0.0, 0.0], [3.2, 0.2], [1.4, 2.6], [-1.1, 1.8]])
    pairs = np.array(list(combinations(range(1, 5), 2)))
    triangles = np.array(list(combinations(range(1, 5), 3)))
    uv = stations[pairs[:, 1] - 1] - stations[pairs[:, 0] - 1]
    leg1 = stations[triangles[:, 1] - 1] - stations[triangles[:, 0] - 1]
    leg2 = stations[triangles[:, 2] - 1] - stations[triangles[:, 1] - 1]

    truth = {"dra": 110.0, "ddec": -70.0, "flux": 3.2e-3}
    wavel = 4.8e-6
    cvis = np.asarray(BinaryModelCartesian(**truth).model(uv[:, 0], uv[:, 1], np.array([wavel])))
    cp = np.rad2deg(np.asarray(closure_phases(cvis, *cp_indices(pairs, triangles))))

    rng = np.random.default_rng(4)
    vis2, phi = np.abs(cvis) ** 2, np.rad2deg(np.angle(cvis))
    vis2_err = 1e-3 * np.median(vis2) * np.ones_like(vis2)
    cp_err = 4e-3 * max(np.median(np.abs(cp)), 5.0) * np.ones_like(cp)
    header = {"TARGET_ID": 1, "TIME": 0.0, "MJD": 61000.0, "INT_TIME": 1.0}

    synth_dict = {
        "info": {
            "TARGET": "UNKNOWN", "OBJECT": "UNKNOWN", "INSTRUME": "SYNTH",
            "MASK": "SYNTH_MASK", "ARRNAME": "SYNTH_MASK", "FILT": "F480M",
            "DATE-OBS": "2000-01-01", "TELESCOP": "SIM", "OBSERVER": "virgil-docs",
            "INSMODE": "NRM", "PA": 0.0, "MJD": 61000.0, "PSCALE": 65.0, "ISZ": 81,
            "STAXY": stations, "CTRS_EQT": stations.copy(),
        },
        "OI_WAVELENGTH": {"EFF_WAVE": wavel, "EFF_BAND": 0.3e-6},
        "OI_VIS": {
            **header,
            "VISAMP": np.abs(cvis) + 2e-3 * np.median(np.abs(cvis)) * rng.normal(size=6),
            "VISAMPERR": 2e-3 * np.median(np.abs(cvis)) * np.ones(6),
            "VISPHI": phi + 0.02 * rng.normal(size=6),
            "VISPHIERR": 0.02 * np.ones(6),
            "UCOORD": uv[:, 0], "VCOORD": uv[:, 1],
            "STA_INDEX": pairs, "FLAG": np.zeros(6, dtype=bool),
        },
        "OI_VIS2": {
            **header,
            "VIS2DATA": vis2 + vis2_err * rng.normal(size=6), "VIS2ERR": vis2_err,
            "UCOORD": uv[:, 0], "VCOORD": uv[:, 1],
            "STA_INDEX": pairs, "FLAG": np.zeros(6, dtype=bool),
        },
        "OI_T3": {
            **header,
            "T3AMP": np.ones(4), "T3AMPERR": np.ones(4),
            "T3PHI": cp + cp_err * rng.normal(size=4), "T3PHIERR": cp_err,
            "U1COORD": leg1[:, 0], "V1COORD": leg1[:, 1],
            "U2COORD": leg2[:, 0], "V2COORD": leg2[:, 1],
            "STA_INDEX": triangles, "FLAG": np.zeros(4, dtype=bool),
        },
    }

    out = Path(tempfile.mkdtemp()) / "synthetic_binary.oifits"
    _ = write_oifits(synth_dict, out)
    ```

# Reading Data

Reading the file back takes one line.

```python
# A path works, as does a file opened with astropy.io.fits or pyoifits.
oidata = OIData(out)
```

## OIData Object

`OIData` works out whether the file holds visibilities or squared visibilities, stored as `oidata.vis` with uncertainty `oidata.d_vis` and marked by `oidata.v2_flag`. Likewise, closure or absolute phases are stored as `oidata.phi` with uncertainty `oidata.d_phi` and marked by `oidata.cp_flag`.

The baselines are in `oidata.u` and `oidata.v`, and the closure-phase indices in `oidata.i_cps1`, `oidata.i_cps2` and `oidata.i_cps3`. You pass a model to the `OIData`, and it evaluates the model at these points.

Phases are always stored in radians, whatever the file's unit. Flagged points (the OIFITS `FLAG` column, or non-finite values) are left out of the observables.

```python
# let's have a tour of the oidata object
print(oidata)
```

```text
OIData(
  u=f32[6],
  v=f32[6],
  wavel=f32[1],
  bandwidth=None,
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
  vis_encoding='split',
  vis_mode='v2',
  v2_flag=True,
  cp_flag=True,
  t_ref=61000.0
)
```

## Verification
Check that every key survives the round trip:

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

Files with several wavelength channels work the same way: each (baseline, wavelength) sample is one entry of `oidata.u`, `oidata.v` and `oidata.wavel`, and closure phases are built from visibilities at their own wavelength. Here we copy the data into three channels (repeating the values, for illustration) and flag one squared visibility:

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

If a file holds several targets (a science target and its calibrator, say), pick one with `OIData(path, target="name")` or by `TARGET_ID`; `OIData` refuses to mix them silently.
