"""Figures for docs/radio_interferometry.md and docs/radio_clean.md.

Simulated, seeded data (the radio tutorial's scene); runs in about a minute on a CPU.
Writes PNGs to docs/figures/.  Usage: .venv/bin/python scripts/make_radio_docs_figures.py
"""

from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import numpyro.distributions as dist
from matplotlib.patches import Ellipse

from virgil import CalibrationNuisance, ComplexVisData, Image, fit, scenes
from virgil._geometry import image_visibilities
from virgil.imaging import MaxEntropy
from virgil.radio_clean import clean, restore

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
N, S, SIG = 65, 4.0, 0.05
CLUMP = (90.0, 60.0)  # (ΔRA, ΔDec) mas, East- and North-positive: the orientation marker
HALF = N * S / 2
EXTENT = [HALF, -HALF, -HALF, HALF]  # East left, North up; row 0 is North


def simulate():
    truth = onp.asarray(
        0.8 * scenes.ring(N, S, 60, 10, inc_deg=50, pa_deg=40)
        + 0.2 * scenes.gaussian_blob(N, S, 6, dra=CLUMP[0], ddec=CLUMP[1])
    )
    rng = onp.random.default_rng(1)
    bl = onp.exp(rng.uniform(onp.log(100), onp.log(1e4), (300, 2))) * rng.choice([-1, 1], (300, 2))
    H, dec, lat = onp.deg2rad(onp.linspace(-45, 45, 200)), onp.deg2rad(-37), onp.deg2rad(-23)
    u = ((bl[:, :1] * onp.cos(H) - bl[:, 1:] * onp.sin(lat) * onp.sin(H)) / 1.3e-3).ravel()
    v = (
        (bl[:, :1] * onp.sin(dec) * onp.sin(H) + bl[:, 1:] * (onp.sin(lat) * onp.sin(dec) * onp.cos(H) + onp.cos(lat) * onp.cos(dec)))
        / 1.3e-3
    ).ravel()
    vis = onp.asarray(image_visibilities(truth, u, v, S, ft="dft")) + SIG * (rng.normal(size=u.size) + 1j * rng.normal(size=u.size))
    return truth, ComplexVisData.from_arrays(u, v, vis, 1 / SIG**2, 230e9), u, v


def check_orientation(image):
    """The clump is East (left, low column) and North (up, low row) of the centre."""
    c = (N - 1) // 2
    east, north = int(round(CLUMP[0] / S)), int(round(CLUMP[1] / S))
    assert image[c - north, c - east] > 3 * image[c + north, c + east], "East/North orientation is wrong"
    assert image[c - north, c - east] > 3 * image[c + north, c - east]


def sky(ax, im, beam=None, title="", label="Jy/beam", marker=False, **kw):
    shown = ax.imshow(im, extent=EXTENT, origin="upper", **kw)
    ax.set(title=title, xlabel="ΔRA (mas)  ← East", ylabel="ΔDec (mas)  North ↑")
    ax.get_figure().colorbar(shown, ax=ax, label=label, shrink=0.85)
    if beam is not None:
        bmaj, bmin, pa = beam
        x0, y0 = HALF * 0.72, -HALF * 0.72
        ax.add_patch(Ellipse((x0, y0), bmin, bmaj, angle=-pa, fc="white", ec="k", lw=0.8))
    if marker:
        ax.plot(*CLUMP, "+", color="cyan", ms=10, mew=1.2)
    return shown


def diverging(ax, im, lim, **kw):
    return sky(ax, im, cmap="RdBu_r", vmin=-lim, vmax=lim, **kw)


def main():
    truth, data, u, v = simulate()
    OUT.mkdir(exist_ok=True)
    kw = dict(weighting="briggs", robust=0.5)
    beam_im = onp.asarray(data.dirty_beam(N, S, **kw))
    dirty = onp.asarray(data.dirty_image(N, S, **kw))
    check_orientation(dirty)
    res = clean(data, N, S, deconvolver="hogbom", threshold=3 / onp.sqrt(onp.sum(onp.asarray(data.weight))), usemask="auto-multithresh")
    beam = res.beam
    lam = 1e6

    # --- dirty trio: uv coverage, dirty beam, dirty image
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].plot(u / lam, v / lam, ",", color="C0", alpha=0.6)
    ax[0].plot(-u / lam, -v / lam, ",", color="C0", alpha=0.6)
    ax[0].set(title="uv coverage", xlabel="u (Mλ)", ylabel="v (Mλ)", aspect="equal")
    sky(ax[1], beam_im, beam, "dirty beam (Briggs 0.5)", "response", cmap="RdBu_r", vmin=-1, vmax=1)
    sky(ax[2], dirty, beam, "dirty image", cmap="magma", marker=True)
    fig.tight_layout()
    fig.savefig(OUT / "radio_dirty_trio.png", dpi=110)

    # --- CLEAN products
    lim = 5 * float(onp.std(res.residual))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    sky(ax[0], dirty, beam, "dirty image", cmap="magma", vmin=0, vmax=res.image.max())
    sky(ax[1], res.model, None, "CLEAN model", "Jy/pixel", cmap="magma")
    sky(ax[2], res.image, beam, "restored image", cmap="magma", vmin=0, vmax=res.image.max(), marker=True)
    diverging(ax[3], res.residual, lim, beam=beam, title="residual")
    fig.tight_layout()
    fig.savefig(OUT / "radio_clean_products.png", dpi=110)

    # --- CLEAN vs RML on the same data
    nuis = CalibrationNuisance(data, ref_obs=0, alpha=False, offsets=False, sigmas=False)
    priors = nuis.priors()
    scene = lambda values: Image(values["log_brightness"], pixel_scale_mas=S)
    start = onp.clip(res.model, 0, None) / onp.clip(res.model, 0, None).sum()
    init = nuis.defaults()
    init["log_brightness"] = onp.log(start + 1e-4 * start.max())
    init["cal.flux"] = float(res.model.sum())
    priors["log_brightness"] = dist.Normal(0.0, 3.0)
    term = data.term(scene, nuis, chi2="auto")
    result = fit(lambda **values: scene(values), priors, (), likelihoods=[term], regularizers=[MaxEntropy(1e3)], init=init, method="lbfgs", max_steps=5000)
    fit_image = onp.asarray(scene(result.values).brightness) * float(result.values["cal.flux"])
    zeros = onp.zeros((N, N))
    tr, rm = restore(truth, zeros, beam, S), restore(fit_image, zeros, beam, S)
    w = onp.asarray(data.weight)
    chi = lambda m: float(onp.sum(w * onp.abs(onp.asarray(data.vis) - onp.asarray(m)) ** 2) / (2 * len(data)))
    fit_vis = data.model(scene(result.values), result.values, nuis)
    pk = tr.max()
    stats = {
        "CLEAN": (chi(res.model_visibilities(data)), onp.sqrt(onp.mean((res.image - res.residual - tr) ** 2)) / pk),
        "RML (MaxEntropy)": (chi(fit_vis), onp.sqrt(onp.mean((rm - tr) ** 2)) / pk),
    }
    print({k: (round(a, 3), round(b, 4)) for k, (a, b) in stats.items()})
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))
    sky(ax[0, 0], tr, beam, "truth, restored", cmap="magma", vmin=0, vmax=pk, marker=True)
    sky(ax[0, 1], res.image - res.residual, beam, "CLEAN model, restored", cmap="magma", vmin=0, vmax=pk)
    sky(ax[0, 2], rm, beam, "RML fit, restored", cmap="magma", vmin=0, vmax=pk)
    ax[1, 0].axis("off")
    ax[1, 0].text(0.0, 0.5, "\n".join(f"{k}:  χ²/N = {a:.3f}\n   rms − truth = {b:.3f} of peak" for k, (a, b) in stats.items()), fontsize=11, va="center")
    l2 = 0.02 * pk
    diverging(ax[1, 1], res.image - res.residual - tr, l2, beam=beam, title="CLEAN − truth")
    diverging(ax[1, 2], rm - tr, l2, beam=beam, title="RML − truth")
    fig.tight_layout()
    fig.savefig(OUT / "radio_clean_vs_rml.png", dpi=110)


if __name__ == "__main__":
    main()
