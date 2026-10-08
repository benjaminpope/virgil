"""Generate the golden fixture ``data/elr_golden.npz`` from Shashank Dholakia's ELR code.

This runs the ORIGINAL, unmodified ``core/ELR.py`` and ``core/utils.py`` from
https://github.com/shashankdholakia/jax-interferometry at commit 70689ed
(full sha 70689ed3dba338d59c98d02e8126a07a3b4e86da) in a legacy environment
(his code uses ``from jax.config import config``, removed in modern jax), so a
port into virgil can be checked against it. It does not import virgil.
No patch to his code is needed with the pins below. His module enables x64
globally; all golden values are float64.

Run (from the repository root):

    uv run --no-project --python 3.11 \
        --with "jax==0.4.23" --with "jaxlib==0.4.23" --with "jaxopt==0.8.2" \
        --with "zodiax==0.4.1" --with "equinox==0.11.2" \
        --with "scipy<1.13" --with "numpy<2" --with matplotlib \
        scripts/make_elr_golden.py

Keys in ``data/elr_golden.npz`` (all float64 unless stated):

Solver (his ``solve_ELR_vec(omega, thetas)``, n_omega=6, n_theta=39)
  omegas              (6,)     [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
  thetas              (39,)    linspace(1e-4, pi-1e-4, 32) followed by 7 extra angles
  solver_rtw          (6, 39)  r/R_eq-type radius rtw (his eq30 solution)
  solver_teff_ratio   (6, 39)  Teff / Teff_pole-type ratio (Flux_ratio**0.25)
  solver_flux_ratio   (6, 39)  Flux_ratio
  eq32                (6,)     his eq32(omega) (equatorial/polar Teff ratio)

Mesh for ``ELR_Model(32, ...)``
  mesh_thetas         (32,)    latitudes (theta=0 pole)
  mesh_n              (32,)    int, points per latitude ring (utils.closest_polygon)
  mesh_phi            (sum n,) longitudes of all points, ring by ring
  mesh_triangulation  (T, 3)   int, ConvexHull simplices

Visibilities (n_sets=5, n_baselines=40, T triangles)
  vis_params          (5, 4)   rows (omega, r_eq_mas, inc_rad, obl_rad); his inc=0 is equator-on
  vis_u, vis_v        (40,)    baselines in metres (default_rng(0), uniform in disk of radius 330 m)
  vis_wavel           ()       0.7e-6 m
  vis2                (5, 40)  his ELR_Model.__call__ output, |V|^2
  cvis                (5, 40)  complex128; normalised complex visibility before |.|^2
  bary_x, bary_y      (5, T)   triangle barycentre x, y (mas) after his rotation
  weight              (5, T)   intensity*heaviside(cos)*cos (before normalisation)
  teff_tri            (5, T)   mean Teff ratio at triangle corners (as in his ``plot``)
"""

import importlib
import os
import sys
import tempfile
import urllib.request

import numpy as onp

SHA = "70689ed3dba338d59c98d02e8126a07a3b4e86da"
RAW = "https://raw.githubusercontent.com/shashankdholakia/jax-interferometry/%s/core/%s"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "elr_golden.npz")


def fetch(cache):
    pkg = os.path.join(cache, "core")
    os.makedirs(pkg, exist_ok=True)
    for name in ("__init__.py", "ELR.py", "utils.py"):
        path = os.path.join(pkg, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(RAW % (SHA, name), path)


def main():
    cache = os.path.join(tempfile.gettempdir(), "elr_shashank_" + SHA[:7])
    fetch(cache)
    sys.path.insert(0, cache)
    ELR = importlib.import_module("core.ELR")
    utils = importlib.import_module("core.utils")
    import jax
    import jax.numpy as np

    assert jax.config.jax_enable_x64

    omegas = onp.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    mesh_th = np.linspace(1e-4, np.pi - 1e-4, 32)
    thetas = np.concatenate(
        [mesh_th, np.array([0.05, 0.3, 0.8, 1.2, 1.5, 1.6, 2.5])]
    )
    rtw_a, t_a, f_a = [], [], []
    for om in omegas:
        r, t, f = ELR.solve_ELR_vec(float(om), thetas)
        rtw_a.append(r)
        t_a.append(t)
        f_a.append(f)
    eq32 = onp.array([float(ELR.eq32(o)) for o in omegas])

    rng = onp.random.default_rng(0)
    nb = 40
    rad = 330.0 * onp.sqrt(rng.uniform(size=nb))
    ang = rng.uniform(0, 2 * onp.pi, nb)
    u, v = rad * onp.cos(ang), rad * onp.sin(ang)
    uv = np.asarray(onp.stack([u, v], axis=1))
    wavel = 0.7e-6
    params = onp.array(
        [
            (0.5, 0.4, 0.3, 0.4),
            (0.9, 0.4, 1.0, 2.0),
            (0.95, 0.6, 0.0, 1.0),
            (0.7, 0.3, 1.4, 0.0),
            (0.2, 0.5, 0.7, 3.0),
        ]
    )

    model = ELR.ELR_Model(32, uv, wavel)
    vis2, cvis, bx, by, wt, tt = [], [], [], [], [], []
    for omega, r_eq, inc, obl in params:
        # replicate ELR_Model.__call__ step by step
        rtws, Ts, Fs = ELR.solve_ELR_vec(omega, model.thetas)
        rtw, T, F = (a.repeat(model.n) for a in (rtws, Ts, Fs))
        theta = model.thetas.repeat(model.n)
        x, y, z = utils.spherical_to_cartesian(rtw, theta, model.phi)
        pts = r_eq * np.stack([x, y, z], axis=1)
        pr = utils.rotate_point_cloud(pts, -inc, obl)
        normals = utils.triangle_normals(pr, model.triangulation)
        bary = utils.barycenter(pr, model.triangulation)
        intensity = np.mean(F[model.triangulation], axis=1)
        cosine = np.dot(np.array([0, 0, 1]), normals.T)
        weight = intensity * np.heaviside(cosine, 0) * cosine
        dftm = ELR.compute_DFTM1(bary[:, 0], bary[:, 1], model.uv, model.wavel)
        ft = ELR.apply_DFTM1(weight, dftm)
        ref = model(omega, r_eq, inc, obl)
        onp.testing.assert_allclose(
            onp.abs(ft) ** 2, ref, rtol=1e-12, atol=1e-14
        )
        vis2.append(ref)
        cvis.append(ft)
        bx.append(bary[:, 0])
        by.append(bary[:, 1])
        wt.append(weight)
        tt.append(np.mean(T[model.triangulation], axis=1))

    out = dict(
        omegas=omegas,
        thetas=onp.asarray(thetas),
        solver_rtw=onp.array(rtw_a),
        solver_teff_ratio=onp.array(t_a),
        solver_flux_ratio=onp.array(f_a),
        eq32=eq32,
        mesh_thetas=onp.asarray(model.thetas),
        mesh_n=onp.asarray(model.n),
        mesh_phi=onp.asarray(model.phi),
        mesh_triangulation=onp.asarray(model.triangulation),
        vis_params=params,
        vis_u=u,
        vis_v=v,
        vis_wavel=onp.float64(wavel),
        vis2=onp.array(vis2),
        cvis=onp.array(cvis),
        bary_x=onp.array(bx),
        bary_y=onp.array(by),
        weight=onp.array(wt),
        teff_tri=onp.array(tt),
    )
    for k, a in out.items():
        assert onp.all(onp.isfinite(a)), k
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    onp.savez_compressed(OUT, **out)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
