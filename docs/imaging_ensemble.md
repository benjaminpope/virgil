# Imaging, part 7: ensembles of reconstructions

Every reconstruction in parts 2–6 rests on choices the data do not make: the regulariser, its weight, the pixel size, the field and the starting image. Part 2 chose them carefully, with L-curves and the discrepancy principle, but a different careful choice would give a somewhat different image. Which features survive any reasonable choice, and which belong to one choice alone?

Drevon et al. (2025, [arXiv:2609.15365](https://arxiv.org/abs/2609.15365)) won the 2024 interferometric imaging contest by not choosing. Their PYRA runs thousands of reconstructions with these settings drawn at random, and their MYTHRA keeps the ones that fit the data and averages them, adding members only while the average still fits. The mean is less sensitive to any one choice than a single reconstruction is, and the spread of the members shows where the image depends on the choices. [`virgil.ensemble`](api/ensemble.md) does the same with virgil's own fits. It is written from the paper's description, not from their code.

!!! note "This page is not executed"
    Unlike parts 1–6, which are generated from executed notebooks, the code here is not run when the documentation is built: a useful ensemble is tens to hundreds of fits. Run it on a GPU, or on a cluster one group per task (below).

## What is drawn

An [`EnsembleSpec`](api/ensemble.md#virgil.ensemble.EnsembleSpec) says what to draw. Each **group** of reconstructions draws:

- a regulariser family: total variation, total squared variation, maximum entropy or the starlet L1 norm of part 6;
- a pixel size, the Nyquist scale over 2, 3 or 4;
- a field, ½, ¾ or all of `field_of_view(data)`;
- a starting image: the Gaussian envelope of `starting_image`, or a flat one (or, for data with phases, the dirty image);
- several weights, log-uniform over a range per data point. A weight is a scale, so log-uniform is the invariant (Jeffreys) choice.

The group is then one L-curve ([`l_curve`](api/imaging.md#virgil.imaging.l_curve)) over its weights, warm-started from the strongest. That is PYRA's random weight, at the cost of one fit sequence rather than independent fits.

**Why groups?** An [`Image`](api/models/classes/image.md)'s pixel size and shape are static, so each new geometry compiles the fit afresh, which takes longer than the fit. The weights are traced, so the fits of a group share one compilation, and so do groups that share a family and a geometry. With a short list of pixel sizes and fields, an ensemble of hundreds of fits compiles a handful of times. `draw_groups` sorts the groups by geometry so that those sharing one run together.

## Selection and the mean

[`combine`](api/ensemble.md#virgil.ensemble.combine) follows MYTHRA:

1. **The L-curve window.** In each group, keep the weights from the L-curve's corner up to `window_dex` (1 by default) above it: from where more freedom stops improving the fit, to somewhat stronger regularisation. Weaker weights fit the noise.
2. **χ².** Drop a member if, on any dataset, its raw χ² per data point is more than `chi2_ratio` (2) times the best member's, or above `max_chi2_red`. Then drop outliers in total χ², more than `mad_cut` (5) robust standard deviations above the median.
3. **A common grid.** Resample the survivors to the finest pixels and the largest field among them, conserving flux ([`metrics.resample`](api/metrics.md#virgil.metrics.resample)). Without a star, recentre each on the best member ([`metrics.align`](api/metrics.md#virgil.metrics.align)); with one, the star fixes the position.
4. **The iterative mean.** In order of χ², add members to a running mean one at a time, keeping each only if the mean's χ² does not rise on any dataset. The mean is judged as the mixture of the members' images on their own grids, which is exact; the common grid of step 3 is for display, since resampling smooths the images and on precise data can raise their χ² several-fold. On data the best member already fits to the noise, this strict rule may keep that member alone, and the spread is then zero; `EnsembleSpec(mean_rtol=...)` lets the mean's χ² rise by that fraction, of order the χ²/N noise √(2/N), to keep more. Give the squared visibilities and closure phases as separate datasets, and each is judged on its own, as in the paper.

With a star, the members are averaged as whole normalised skies: the star's share of the flux is their mean, and the image's pixels the mean of their fluxes.

## A small example

A star with two blobs of dust, observed with the AMI coverage of part 1.

```python
import jax
import matplotlib.pyplot as plt

from virgil.coverage import ami_grid_record
from virgil.ensemble import EnsembleSpec, ensemble
from virgil.imaging import beam
from virgil.metrics import score
from virgil.models import Image, PointSource, System
from virgil.oidata import OIData
from virgil.plotting import plot_model
from virgil.scenes import gaussian_blob

npix, scale = 24, 19.0
blobs = gaussian_blob(npix, scale, 25.0, dra=60.0) + 0.6 * gaussian_blob(
    npix, scale, 35.0, dra=-40.0, ddec=50.0
)
truth = System(
    star=PointSource(), env=Image.from_brightness(blobs, scale, flux=0.5)
)
template = OIData(ami_grid_record(wavelength_m=4.8e-6, rotation_deg=-6.9))
data = template.with_model(truth, key=jax.random.PRNGKey(7))

spec = EnsembleSpec(n_weights=6)
result = ensemble(data, 8, jax.random.PRNGKey(1), spec=spec)
print(result.summary())
```

`summary()` lists each group (its family, geometry, start, L-curve corner and how many members were kept), why the others were dropped, and the raw χ² per data point of the best member and of the mean on each dataset. Those are χ² with the quoted errors, not rescaled: a value well above one means the data are not fitted, not that the errors need inflating. The iterative mean never raises any dataset's χ² above the best member's, and `result.trace` records it after each member joins.

```python
fov = npix * scale
resolution = beam(data)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
plot_model(truth.env, fov_mas=fov, npix=npix, ax=axes[0], title="truth")
plot_model(result.mean, fov_mas=fov, npix=npix, ax=axes[1], title="ensemble mean", beam=resolution)
spread = Image.from_brightness(result.std, result.mean.pixel_scale_mas)
plot_model(spread, fov_mas=fov, npix=npix, ax=axes[2], title="ensemble standard deviation")
plt.tight_layout()

print(score(result.mean, truth.env, max_shift_mas=resolution.major_mas))
```

[`metrics.score`](api/metrics.md#virgil.metrics.score) compares the mean with the truth: it resamples it onto the truth's grid and reports the normalised cross-correlation, the 2024 contest's L1 score and the older contests' metrics.

## Ensemble spread is not a posterior

The standard deviation map answers "how much does the image change between reasonable reconstruction choices?" It is large where the regularisers disagree and nearly zero where every choice agrees. It does not include the noise in the data: an ensemble of reconstructions of noiseless data still has a spread, and one of very noisy data can have little if every regulariser smooths the noise the same way.

The posterior standard deviation of [part 5](imaging_sampling.md) answers a different question: given one prior, how uncertain is the image because of the noise? The two are complementary. A feature that is significant in the posterior but absent from many ensemble members depends on the prior; a feature present in every member but with a broad posterior is robust to the method but not well measured.

## On a cluster

`ensemble` runs its groups one after another. For a large ensemble, run one group per task of a job array and combine them at the end. The draws depend only on the key, so every task can draw them all and pick its own:

```python
import pickle
import sys

from virgil.ensemble import draw_groups, reference_starts, run_group

task = int(sys.argv[1])
draws = draw_groups(data, 200, jax.random.PRNGKey(1), spec)
starts = reference_starts(data, True, spec.starts)
group = run_group(data, draws[task], starts=starts)
with open(f"group_{task:03d}.pkl", "wb") as f:
    pickle.dump(group, f)
```

and then, in one more job,

```python
from virgil.ensemble import combine

groups = [pickle.load(open(f"group_{i:03d}.pkl", "rb")) for i in range(200)]
result = combine(data, groups, spec=spec)
```

Group tasks that share a geometry each compile it once. JAX's persistent compilation cache can share those compilations between tasks.

## Summary

- **`ensemble`** draws regulariser families, weights, pixel sizes, fields and starts, fits each group as an L-curve, and selects and averages the members (PYRA and MYTHRA, Drevon et al. 2025).
- **Selection** keeps each group's L-curve window, drops members that fit any dataset badly and χ² outliers, and adds members to the mean only while no dataset's χ² rises.
- **The result** is a mean image, a per-pixel standard deviation across the kept members, and the raw χ² per data point of the mean on each dataset.
- **Compilation** is per geometry and family, not per fit; on a cluster, use `draw_groups`, `run_group` and `combine`.
