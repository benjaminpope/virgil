import warnings
from pathlib import Path

import jax.numpy as np
import jax.scipy as jsp
import numpy as onp
import pyoifits as oifits

from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData

fname = "NuHor_F480M.oifits"
ddir = str(Path(__file__).resolve().parents[1] / "data") + "/"

data = oifits.open(ddir + fname)

try:
    data.verify("silentfix")
except AttributeError:
    warnings.warn(
        "Skipping pyoifits verify due to upstream compatibility issue on this environment"
    )

oidata = OIData(data)
u, v = oidata.u / oidata.wavel, oidata.v / oidata.wavel
i_cps1, i_cps2, i_cps3 = oidata.i_cps1, oidata.i_cps2, oidata.i_cps3

params = ["dra", "ddec", "flux"]

# Coarse on purpose: the grid tests check shapes, axis order and agreement
# between code paths, not detection limits, and every point costs a fit.
samples_dict = {
    "dra": np.linspace(600.0, -600.0, 30),
    "ddec": np.linspace(-600.0, 600.0, 31),
    "flux": 10 ** np.linspace(-6, -1, 32),
}

true_values = [250.0, 150.0, 5e-4]
binary = BinaryModelCartesian(true_values[0], true_values[1], true_values[2])

cvis_sim = binary.model(oidata.u, oidata.v, oidata.wavel)

_rng = onp.random.default_rng(42)

sim_data = {
    "u": oidata.u,
    "v": oidata.v,
    "wavel": oidata.wavel,
    "vis": oidata.to_vis(cvis_sim)
    + _rng.standard_normal(oidata.vis.shape) * oidata.d_vis,
    "d_vis": oidata.d_vis,
    "phi": oidata.to_phases(cvis_sim)
    + _rng.standard_normal(oidata.phi.shape) * oidata.d_phi,
    "d_phi": oidata.d_phi,
    "i_cps1": oidata.i_cps1,
    "i_cps2": oidata.i_cps2,
    "i_cps3": oidata.i_cps3,
    "v2_flag": oidata.v2_flag,
    "cp_flag": oidata.cp_flag,
}

oidata_sim = OIData(sim_data)

perc = np.array([jsp.stats.norm.cdf(2.0)])
