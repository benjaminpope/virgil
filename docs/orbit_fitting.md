<!-- AUTO-GENERATED FROM notebooks/orbit_fitting.ipynb by scripts/sync_tutorial_docs.py. -->
# Fitting a binary's orbit from interferometric epochs

A long-baseline interferometer resolves a close binary into two stars, and if we observe it again over months and years we can watch the companion move around its primary. This tutorial fits a Keplerian orbit to such a sequence of observations, from start to finish, with no radial velocities.

The path is the classical one. We first measure the companion's position at each epoch separately, with a grid search and a fit of a binary to that night's squared visibilities and closure phases. A grid over period, eccentricity and time of periastron then gives starting orbits, because at fixed values of those three the positions are linear in the remaining elements. Finally we sample the orbit's posterior with NUTS, under priors that respect the symmetries of the problem, and look at the result as a corner plot and as an ensemble of orbits drawn on the sky.

The orbit conventions are those of [Conventions](conventions.md#orbits): `Omega` is the position angle of the node where the secondary recedes, `omega` is the secondary's argument of periastron, an inclination below 90° means the position angle increases with time, and times are days since a reference epoch `t_ref`.

We need numpyro for the sampler, virgil's coverage, fitting and orbit tools, and two plotting helpers: `plot_chainconsumer_diagnostics` for the corner plot and `plot_orbit_ensemble` for orbits on the sky. `set_style` applies the figure style used throughout the docs.

```python
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from tqdm.auto import tqdm

from virgil.coverage import vlti_oidata
from virgil.fitting import fit
from virgil.grid_fit import best_grid_point, likelihood_grid
from virgil.inference import laplace_cov
from virgil.likelihood import numpyro_model
from virgil.models import Attached, BinaryModelCartesian, PointSource, System
from virgil.orbits import (
    KeplerOrbit,
    PositionData,
    starting_orbits,
    total_mass,
)
from virgil.plotting import (
    plot_chainconsumer_diagnostics,
    plot_orbit_ensemble,
    set_style,
)
from virgil.simulate import simulate

warnings.filterwarnings("ignore", message="IProgress not found.*")
set_style()
```

## The system

Our binary has a period of about three years, a mildly eccentric orbit ($e = 0.3$), an angular semimajor axis of 25 mas and an inclination of 55°. The companion is 0.15 times as bright as the primary in the K band. We place it at 105 pc, which we will treat as known exactly when we turn the orbit into a total mass.

The eight epochs fall in two observing seasons and cover about half of the orbit. The periastron passage is not observed: the last epoch is about ten months before the next one. We will see that this is where the orbit is least certain.

```python
T_REF = 60000.0  # MJD of the first epoch; every time is measured from it
DISTANCE_PC = 105.0  # assumed known exactly, for the total mass
FLUX = 0.15  # companion / primary
TRUTH = dict(
    period=1100.0,
    dt_peri=-220.0,
    ecc=0.3,
    inc=55.0,
    omega=70.0,
    Omega=130.0,
    a_mas=25.0,
)
truth_orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
NIGHTS = T_REF + np.array([0.0, 35, 95, 150, 330, 385, 445, 560])

next_peri = T_REF + TRUTH["dt_peri"] + TRUTH["period"]
print(
    f"period {TRUTH['period'] / 365.25:.2f} yr, total mass "
    f"{float(total_mass(truth_orbit, DISTANCE_PC)):.2f} M_sun at "
    f"{DISTANCE_PC:.0f} pc"
)
print(
    f"the epochs span {np.ptp(NIGHTS) / TRUTH['period']:.0%} of the "
    f"orbit; the next periastron is at MJD {next_peri:.0f}"
)
```

## Simulating the observations

[`vlti_oidata`](api/coverage.md) builds the coverage of the four 8-m Unit Telescopes at Paranal for a target at declination −30°: three snapshots per night, two hours apart either side of transit, each with all six baselines and four closure triangles, in six K-band channels (2.0–2.4 μm). Passing `nights_mjd` repeats this on every night and stamps each snapshot with its own time. The errors are 0.05 on each $V^2$ and 3° on each closure phase, typical of calibrated VLTI data.

The scene is a primary point source and a companion `Attached` to the true orbit, so [`simulate`](api/simulate.md) evaluates the companion's position at the time of each sample and adds noise with a fixed seed. Splitting the result by epoch gives one dataset per night.

```python
template = vlti_oidata(
    declination_deg=-30.0,
    hour_angles_h=(-2.0, 0.0, 2.0),
    wavelengths_m=np.linspace(2.0e-6, 2.4e-6, 6),
    sigma_v2=0.05,
    sigma_cp_deg=3.0,
    nights_mjd=NIGHTS,
)
scene = System(
    primary=PointSource(), comp=Attached(PointSource(FLUX), truth_orbit)
)
data = simulate(scene, template, key=jax.random.PRNGKey(2026))
epochs = data.split_by_epoch()
print(
    f"{len(epochs)} epochs, each with {epochs[0].n_independent} "
    "independent V² and closure phases"
)
```

The left panel shows the uv coverage of the first night, coloured by wavelength: Earth rotation carries each baseline along a short track, and the spread of channels stretches it radially. The right panel shows the true orbit on the sky (East to the left, North up), with the companion's position at each of the eight epochs and the unobserved periastron marked.

```python
fig, (ax_uv, ax_sky) = plt.subplots(
    1, 2, figsize=(11, 4.6), constrained_layout=True
)
first = epochs[0]
uu, vv = (np.asarray(x / first.wavel) / 1e6 for x in (first.u, first.v))
wavel_um = np.asarray(first.wavel) * 1e6
for sign in (1, -1):
    points = ax_uv.scatter(
        sign * uu, sign * vv, c=wavel_um, cmap="plasma", s=12
    )
fig.colorbar(points, ax=ax_uv, label="Wavelength (μm)")
ax_uv.set(
    xlabel="u (Mλ)",
    ylabel="v (Mλ)",
    aspect="equal",
    title=f"uv coverage on MJD {NIGHTS[0]:.0f}",
)
ax_uv.invert_xaxis()  # East to the left

track = T_REF + TRUTH["dt_peri"] + np.linspace(0, TRUTH["period"], 400)
dra, ddec, _ = truth_orbit.relative(track)
night_dra, night_ddec, _ = truth_orbit.relative(NIGHTS)
ax_sky.plot(dra, ddec, color="0.6", lw=1.2)
ax_sky.scatter(
    night_dra,
    night_ddec,
    c=NIGHTS,
    cmap="viridis",
    s=40,
    edgecolors="k",
    linewidths=0.5,
    zorder=3,
    label="epochs",
)
ax_sky.plot(
    dra[0], ddec[0], "o", mfc="none", mec="k", ms=9, label="periastron"
)
ax_sky.plot(0, 0, "*", color="k", ms=14, label="primary")
ax_sky.set(
    xlabel="Δα (mas, East)",
    ylabel="Δδ (mas, North)",
    aspect="equal",
    title="True orbit",
)
ax_sky.invert_xaxis()
ax_sky.legend(frameon=False, fontsize="small")
plt.show()
```

## Astrometry at each epoch

For each night we first search a grid of companion positions and flux ratios (±40 mas in steps of 1 mas, finer than the resolution λ/B ≈ 3.5 mas of the longest baseline), and start a fit of `BinaryModelCartesian` from the best grid point. The positions have uniform priors, and the flux ratio a log-uniform (Jeffreys) prior, as it is a scale. The Laplace approximation at the best fit, [`laplace_cov`](api/inference.md), gives the covariance of the position, and each epoch's time is the mean time of its samples. Between a night's first and last snapshots the companion moves by under 0.02 mas, less than half its positional error, and the motion averages out at the mean time, so one position per night is adequate here.

We collect the positions and covariances into a [`PositionData`](api/orbits.md), virgil's container for relative astrometry.

```python
axis = np.linspace(-40.0, 40.0, 81)
grid = {"dra": axis, "ddec": axis, "flux": np.geomspace(0.02, 0.5, 6)}
binary_priors = {
    "dra": dist.Uniform(-50.0, 50.0),
    "ddec": dist.Uniform(-50.0, 50.0),
    "flux": dist.LogUniform(1e-3, 1.0),
}
names = ["dra", "ddec", "flux"]

rows = []
for night in tqdm(epochs, desc="epochs"):
    loglike_grid = likelihood_grid(night, BinaryModelCartesian, grid)
    start = best_grid_point(loglike_grid, grid)
    result = fit(BinaryModelCartesian(**start), binary_priors, night)
    values = [float(result.values[k]) for k in names]
    cov = np.asarray(laplace_cov(values, names, night, result.model))
    rows.append(
        dict(
            zip(names, values), mjd=float(np.mean(night.mjd)), cov=cov[:2, :2]
        )
    )

measured = PositionData(
    [r["mjd"] for r in rows],
    [r["dra"] for r in rows],
    [r["ddec"] for r in rows],
    np.array([r["cov"] for r in rows]),
    t_ref=T_REF,
)
```

The fitted positions agree with the true ones to within their errors of a few hundredths of a milliarcsecond. The last column is each epoch's χ² (two degrees of freedom) of the fitted position against the truth, using the Laplace covariance; their sum should be near twice the number of epochs if the covariances are right.

```python
true_dra, true_ddec, _ = (
    np.asarray(x) for x in truth_orbit.relative([r["mjd"] for r in rows])
)
print(
    "    MJD   Δα fitted (mas)  Δα true   Δδ fitted (mas)  Δδ true"
    "   flux    χ²"
)
total = 0.0
for r, ra, de in zip(rows, true_dra, true_ddec):
    err = np.sqrt(np.diag(r["cov"]))
    offset = np.array([r["dra"] - ra, r["ddec"] - de])
    chi2 = float(offset @ np.linalg.solve(r["cov"], offset))
    total += chi2
    print(
        f"{r['mjd']:7.1f}  {r['dra']:7.3f} ± {err[0]:.3f}  {ra:7.3f}"
        f"   {r['ddec']:7.3f} ± {err[1]:.3f}  {de:7.3f}"
        f"   {r['flux']:.3f}  {chi2:5.2f}"
    )
print(f"total χ² = {total:.1f} for {2 * len(rows)} degrees of freedom")
```

## Starting orbits

At a fixed period, eccentricity and time of periastron, the sky positions are linear in the four Thiele–Innes constants, so each point of a grid in those three is an exact weighted least-squares solve. [`starting_orbits`](api/orbits.md) runs this grid (here 120 periods from 300 to 5000 days, eccentricities from 0 to 0.9, and 36 times of periastron per period) and converts the best solutions back to Keplerian elements. It needs no random restarts and finds the right basin even when the arc is short. Positions alone fix the node only modulo 180°, so the returned `Omega` lies in [0°, 180°).

```python
starts = starting_orbits(
    measured, periods=np.geomspace(300.0, 5000.0, 120), n_best=3
)
columns = ("period", "ecc", "inc", "omega", "Omega", "a_mas")
print("        " + "".join(f"{c:>9}" for c in columns) + "       χ²")
print("truth   " + "".join(f"{TRUTH[c]:9.2f}" for c in columns))
for k, (orbit, chi2) in enumerate(starts):
    print(
        f"start {k} "
        + "".join(f"{float(getattr(orbit, c)):9.2f}" for c in columns)
        + f"  {chi2:7.1f}"
    )
best_start = starts[0][0]
```

## Priors

virgil's default priors are the invariant (Jeffreys) measures of the groups that act on each parameter, unless there is strong information otherwise:

| Parameter | Prior | Why it is the invariant choice |
|---|---|---|
| Period $P$ | log-uniform, 100–10⁴ d | A scale: the prior should not depend on the unit of time, so it is invariant under rescaling. |
| Semimajor axis $a$ | log-uniform, 1–300 mas | A scale, invariant under rescaling of angles. With $P$ this is $1/(aP)$, the invariant prior for independent rescalings of space and time. |
| Inclination | uniform in $\cos i$ on [−1, 1] | An isotropic orientation of the orbital plane is the uniform (Haar) measure on rotations, which is uniform in $\cos i$. |
| $\omega$, $\Omega$ | uniform on [0°, 360°) | The same rotation-invariant measure: no direction of the node or of periastron is preferred. |
| Mean anomaly at `t_ref` | uniform on [0°, 360°) | A location in time: invariant under time translation. It is the same as a uniform time of periastron over one period. |
| Eccentricity $e$ | uniform on [0, 1) | No group acts on $e$, so uniform $e$ is an interim prior, which a population prior can later reweight. |

Two details of the parameterisation matter for the sampler. virgil has no helper for an isotropic inclination yet, so we sample $\cos i$ itself and set $i = \arccos(\cos i)$ in the model; uniform $\cos i$ is then exactly the isotropic prior, with no Jacobian to add. And the three angles are periodic, but a sampler given `Uniform(0, 360)` sees walls at 0° and 360°. We therefore sample each angle as the direction of a 2-vector $v$. Any rotationally symmetric density for $v$ makes the direction exactly uniform over the full circle, with no boundary, and leaves the vector's length $r$ as a nuisance. The length still matters to the sampler. The data fix the angles to a fraction of a degree, so near a radius $r$ the posterior is a thin wedge, of width proportional to $r$. Under an isotropic Gaussian, $r$ ranges over more than a factor of ten, the wedge narrows into a funnel towards the origin, and no single step size suits it all: NUTS then diverges. So `Ring` gives $v$ the density $\exp[-(r - 1)^2/2s^2]$, with $s = 0.1$, which keeps $r$ within about 20% of 1 and the wedge's width nearly constant. virgil will provide this prior as `AngleVector` (virgil#211). The function `elements` maps the sampled values to the orbital elements; it works on single values inside the model and on whole arrays of samples afterwards.

```python
def angle(vec):
    # The direction of a 2-vector (last axis), in degrees in [0, 360).
    return jnp.mod(jnp.degrees(jnp.arctan2(vec[..., 1], vec[..., 0])), 360.0)


def elements(v):
    mean_anomaly = angle(v["phase_vec"])  # at t_ref
    return dict(
        period=v["period"],
        dt_peri=-v["period"] * mean_anomaly / 360.0,
        ecc=v["ecc"],
        inc=jnp.degrees(jnp.arccos(v["cos_inc"])),
        omega=angle(v["omega_vec"]),
        Omega=angle(v["Omega_vec"]),
        a_mas=v["a_mas"],
    )


class Ring(dist.Distribution):
    # A 2-vector with a uniform direction and a length near 1: the density
    # exp(-(r - 1)^2 / 2 s^2), normalised in the plane.
    support = dist.constraints.real_vector

    def __init__(self, width=0.1):
        self.width = width
        z = width**2 * jnp.exp(-0.5 / width**2) + width * jnp.sqrt(
            jnp.pi / 2
        ) * (1 + jax.scipy.special.erf(1 / (width * jnp.sqrt(2))))
        self.log_z = jnp.log(2 * jnp.pi * z)
        super().__init__(event_shape=(2,))

    def log_prob(self, v):
        r = jnp.linalg.norm(v, axis=-1)
        return -0.5 * ((r - 1) / self.width) ** 2 - self.log_z

    def sample(self, key, sample_shape=()):
        # Uniform direction; the length from N(1, s), which is close to the
        # ring's radial density for small s (used only to initialise).
        k1, k2 = jax.random.split(key)
        theta = jax.random.uniform(k1, sample_shape, maxval=2 * jnp.pi)
        r = jnp.abs(1 + self.width * jax.random.normal(k2, sample_shape))
        return r[..., None] * jnp.stack([jnp.cos(theta), jnp.sin(theta)], -1)


direction = Ring(width=0.1)
orbit_priors = {
    "period": dist.LogUniform(100.0, 1e4),
    "a_mas": dist.LogUniform(1.0, 300.0),
    "ecc": dist.Uniform(0.0, 1.0),
    "cos_inc": dist.Uniform(-1.0, 1.0),
    "omega_vec": direction,
    "Omega_vec": direction,
    "phase_vec": direction,
}


def run_nuts(model, seed, num_warmup, num_samples, num_chains=1, init=None):
    strategy = (
        {} if init is None else {"init_strategy": init_to_value(values=init)}
    )
    kernel = NUTS(model, dense_mass=True, target_accept_prob=0.95, **strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc
```

### Prior check

A sampler should reproduce its prior when there are no data, Jacobians included. [`numpyro_model`](api/likelihood.md) builds the model from the same priors with no data and no likelihood terms (the scene is empty, since the positions are not visibilities), and a short NUTS run samples it. The histograms of the derived elements match the stated densities (dashed): log-uniform $P$ and $a$, uniform $e$ and $\cos i$, the $\sin i$ density that uniform $\cos i$ implies, and flat angles.

```python
no_scene = lambda **values: None  # the positions carry all the data
prior_model = numpyro_model(no_scene, orbit_priors, ())
prior = run_nuts(prior_model, seed=1, num_warmup=500, num_samples=4000)
draws = prior.get_samples()
derived = {k: np.asarray(v) for k, v in elements(draws).items()}
mean_anomaly = np.asarray(angle(draws["phase_vec"]))

degrees = np.linspace(0, 360, 2)
panels = [
    ("log₁₀ P (d)", np.log10(derived["period"]), [2, 4], [0.5, 0.5]),
    (
        "log₁₀ a (mas)",
        np.log10(derived["a_mas"]),
        [0, np.log10(300)],
        [1 / np.log10(300)] * 2,
    ),
    ("e", derived["ecc"], [0, 1], [1, 1]),
    ("cos i", np.asarray(draws["cos_inc"]), [-1, 1], [0.5, 0.5]),
    (
        "i (deg)",
        derived["inc"],
        np.linspace(0, 180, 100),
        np.sin(np.radians(np.linspace(0, 180, 100))) * np.pi / 360,
    ),
    ("ω (deg)", derived["omega"], degrees, [1 / 360] * 2),
    ("Ω (deg)", derived["Omega"], degrees, [1 / 360] * 2),
    ("mean anomaly at t_ref (deg)", mean_anomaly, degrees, [1 / 360] * 2),
]
fig, axes = plt.subplots(2, 4, figsize=(13, 5), constrained_layout=True)
for ax, (label, samples, x, density) in zip(axes.flat, panels):
    ax.hist(samples, bins=30, density=True, color="#3c6e9f", alpha=0.6)
    ax.plot(x, density, "k--", lw=1.2)
    ax.set(xlabel=label, yticks=[])
fig.suptitle("Prior draws from NUTS with no data (dashed: stated prior)")
plt.show()
```

## Sampling the orbit

Now we add the positions. `measured.term(orbit_fn)` is a likelihood term, the Gaussian log density of the positions given an orbit, which `numpyro_model` adds to the prior. We start four chains at the best Thiele–Innes orbit (with $e$ and $\cos i$ moved slightly off the edges of their priors), and use a dense mass matrix, since period, eccentricity and time of periastron are strongly correlated when the periastron is unobserved. The run is short: a thousand warm-up steps and a thousand samples per chain. Afterwards we check the sampler: the number of divergent transitions, the largest split-$\hat R$ over the sampled sites and the smallest effective sample size.

```python
def orbit_fn(values):
    return KeplerOrbit(**elements(values), t_ref=T_REF)


def direction_of(degrees):
    return jnp.array(
        [jnp.cos(jnp.radians(degrees)), jnp.sin(jnp.radians(degrees))]
    )


posterior_model = numpyro_model(
    no_scene, orbit_priors, (), likelihoods=[measured.term(orbit_fn)]
)
init = {
    "period": float(best_start.period),
    "a_mas": float(best_start.a_mas),
    "ecc": float(np.clip(best_start.ecc, 0.02, 0.95)),
    "cos_inc": float(np.clip(np.cos(np.radians(best_start.inc)), -0.99, 0.99)),
    "omega_vec": direction_of(best_start.omega),
    "Omega_vec": direction_of(best_start.Omega),
    "phase_vec": direction_of(-360.0 * best_start.dt_peri / best_start.period),
}
mcmc = run_nuts(
    posterior_model,
    seed=2,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    init=init,
)
divergences = int(mcmc.get_extra_fields()["diverging"].sum())
stats = summary(mcmc.get_samples(group_by_chain=True))
r_hat = max(float(np.max(s["r_hat"])) for s in stats.values())
n_eff = min(float(np.min(s["n_eff"])) for s in stats.values())
print(f"{divergences} divergent transitions in {4 * 1000} samples")
print(f"largest r_hat {r_hat:.3f}, smallest effective sample size {n_eff:.0f}")
```

A healthy run has no divergences, $\hat R$ below about 1.01 and an effective sample size in the thousands. If there are more than a handful of divergences, do not trust the samples, because the divergences mark regions the sampler could not explore: reparameterise (as `Ring` does for the angles), start closer to the mode or raise `target_accept_prob`. Do not simply drop the divergent samples.

Each sample is turned into orbital elements, and into a total mass at the assumed distance by Kepler's third law, $M = a^3/P^2$ with $a$ in au and $P$ in years. Positions cannot tell $(\Omega, \omega)$ from $(\Omega + 180°, \omega + 180°)$, which only flips the line-of-sight direction, so the full-circle prior gives two mirror-image modes of equal height. The chains stay in the mode where they started; we report it with $\Omega$ in [0°, 180°), the usual convention for visual orbits, and radial velocities would be needed to choose between the two. The table compares the posterior medians and 68% intervals with the truth.

```python
samples = mcmc.get_samples()
post = {k: np.asarray(v, dtype=float) for k, v in elements(samples).items()}
flip = post["Omega"] >= 180.0  # fold onto the Omega < 180 mirror image
post["Omega"] = np.where(flip, post["Omega"] - 180.0, post["Omega"])
post["omega"] = np.mod(post["omega"] - 180.0 * flip, 360.0)

table = pd.DataFrame(
    {
        "P (d)": post["period"],
        "a (mas)": post["a_mas"],
        "e": post["ecc"],
        "i (deg)": post["inc"],
        "ω (deg)": post["omega"],
        "Ω (deg)": post["Omega"],
        "t_peri (MJD)": T_REF + post["dt_peri"],
        "M_tot (M☉)": np.asarray(
            total_mass(KeplerOrbit(**post, t_ref=T_REF), DISTANCE_PC)
        ),
    }
)
truths = dict(
    zip(
        table.columns,
        [
            TRUTH["period"],
            TRUTH["a_mas"],
            TRUTH["ecc"],
            TRUTH["inc"],
            TRUTH["omega"],
            TRUTH["Omega"],
            T_REF + TRUTH["dt_peri"],
            float(total_mass(truth_orbit, DISTANCE_PC)),
        ],
    )
)
low, mid, high = np.percentile(table, [16, 50, 84], axis=0)
for name, lo, m, hi in zip(table.columns, low, mid, high):
    print(
        f"{name:13s} {m:10.3f}  +{hi - m:7.3f} −{m - lo:7.3f}"
        f"   truth {truths[name]:10.3f}"
    )
```

The corner plot shows the joint posterior, with the truth marked. The total mass assumes the distance of 105 pc exactly; with a parallax, its error would add to the mass's through $M \propto D^3$. With the periastron unobserved, the period, eccentricity, time of periastron, inclination and node are strongly correlated: a slightly longer period with a lower eccentricity and an earlier periastron fits the observed half orbit almost as well. The total mass inherits these through $a^3/P^2$. Only the next periastron passage will break the correlations.

```python
_, corner_fig, walks_fig = plot_chainconsumer_diagnostics(
    {"NUTS": table},
    columns=list(table.columns),
    truth=truths,
    colors=["#3c6e9f"],
)
plt.close(walks_fig)  # the chains' traces are not needed here
plt.show()
```

## Orbits drawn from the posterior

A corner plot shows the elements; what an observer needs to know is where the companion can be. We draw 150 orbits from the posterior and plot them on the sky with [`plot_orbit_ensemble`](api/plotting.md), together with the measured positions (coloured by epoch, with their 1σ error ellipses, which are far smaller than the markers) and the true orbit. Where the epochs cover the orbit the draws coincide; around the unobserved periastron, just south of the primary, they fan out by up to a few milliarcseconds.

```python
pick = np.random.default_rng(7).choice(len(table), 150, replace=False)
ensemble = KeplerOrbit(**{k: v[pick] for k, v in post.items()}, t_ref=T_REF)
fig, ax = plot_orbit_ensemble(ensemble, measured, truth_orbit)
plt.show()
```

The same draws as functions of time show when the uncertainty matters. Separation and position angle are tightly pinned during the two observed seasons and spread out towards the next periastron (dotted line), which is therefore the most valuable time for the next observation. The position angle increases with time, as it should for an inclination below 90°.

```python
times = T_REF + np.linspace(-150.0, 1050.0, 600)
epoch_mjd = measured.t_ref + np.asarray(measured.dt)


def unwrapped_pa(pa):
    # Position angles as continuous curves, starting in [0, 360).
    return np.degrees(np.unwrap(np.radians(np.asarray(pa)), axis=-1))


sep, pa = jax.vmap(lambda orbit: orbit.separation_pa(times))(ensemble)
true_sep, true_pa = truth_orbit.separation_pa(times)
true_pa = unwrapped_pa(true_pa)
pa = unwrapped_pa(pa)
pa += 360.0 * np.round((true_pa[0] - pa[:, :1]) / 360.0)
data_sep = np.hypot(measured.dra, measured.ddec)
data_pa = np.degrees(np.arctan2(measured.dra, measured.ddec))
near = np.interp(epoch_mjd, times, true_pa)  # same branch as the curves
data_pa += 360.0 * np.round((near - data_pa) / 360.0)

fig, axes = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True, constrained_layout=True
)
curves = (
    (np.asarray(sep), np.asarray(true_sep), data_sep, "Separation (mas)"),
    (pa, true_pa, data_pa, "Position angle (deg)"),
)
for ax, (draws_t, true_t, observed, label) in zip(axes, curves):
    ax.plot(times, draws_t.T, color="#3c6e9f", lw=0.6, alpha=0.06)
    ax.plot(times, true_t, color="#d1495b", lw=1.5, label="true orbit")
    ax.scatter(
        epoch_mjd,
        observed,
        c=epoch_mjd,
        cmap="viridis",
        s=36,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
        label="measured",
    )
    ax.axvline(next_peri, color="k", ls=":", lw=1)
    ax.set_ylabel(label)
axes[0].legend(frameon=False, fontsize="small")
axes[1].set_xlabel("MJD")
plt.show()
```

## Summary

We simulated eight epochs of VLTI-like squared visibilities and closure phases of a binary with a three-year orbit, measured the companion's position at each epoch with a grid search, a fit and its Laplace covariance, found starting orbits with the Thiele–Innes grid, and sampled the orbit's posterior with NUTS under invariant priors: log-uniform period and semimajor axis, an isotropic orientation, a uniform phase and a uniform eccentricity. The prior check confirmed that the sampler reproduces those priors without data. The posterior recovers the truth, and the ensemble of orbits shows where the companion's position is still uncertain: around the unobserved periastron.

Where to go next:

- Fitting the orbit to the visibilities themselves, jointly with the rest of the scene, avoids the per-epoch step, which can be biased when the source is more than two point stars. Tie components to an orbit with `Attached`; the orbit worked example [`notebooks/mwe/mwe_orbit.ipynb`](https://github.com/benjaminpope/virgil/blob/main/notebooks/mwe/mwe_orbit.ipynb) does this for a companion with a disc, and the design is in [`design/orbit_scene_joint_fitting.md`](https://github.com/benjaminpope/virgil/blob/main/design/orbit_scene_joint_fitting.md).
- The four Thiele–Innes constants enter the positions linearly, so they can be marginalised analytically, leaving a three-dimensional sampling problem in $P$, $e$ and $t_{\rm peri}$. This marginalised sampler is designed in [`design/thiele_innes_marginalisation.md`](https://github.com/benjaminpope/virgil/blob/main/design/thiele_innes_marginalisation.md) but not built yet.
- [Binary search](binary_search.md) covers the grid and the sampler for one epoch in more detail, and [Conventions](conventions.md#orbits) the orbit conventions and the degeneracies of visual orbits.
