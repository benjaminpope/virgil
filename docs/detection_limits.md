# Detection limits: methods and choices

[`virgil.limits`](api/limits.md) turns a non-detection into a flux limit at
each position of a grid. This page says what each method computes and which
to use; [Contrast limits](contrast_limits.md) runs them on simulated data,
and [Detection ROC curves](detection_roc.md) calibrates a detection threshold
for a grid search, look-elsewhere effect included.
[Detection statistics](detection_statistics.md#detection-plots) explains the
plots.

## Units {#units}

A companion's **flux** is relative to the primary (companion/primary, 0.01
for a companion 100 times fainter), and limits come back in these units.
Results are usually reported as a **contrast**, primary/companion (100 here),
or a magnitude difference $\Delta m = 2.5\log_{10}(\text{contrast})$ (5 mag
here). [`flux_to_contrast`][virgil.limits.flux_to_contrast],
[`flux_to_delta_mag`][virgil.limits.flux_to_delta_mag] and their inverses
convert; zero or negative fluxes map to a large finite contrast, not to
infinity.

## Which method? {#which-method}

| | [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] | [`absil_limits`][virgil.limits.absil_limits] | [`injection_limits`][virgil.limits.injection_limits] |
|---|---|---|---|
| Statistics | Bayesian, flux $\ge 0$ prior | frequentist $\chi^2$ test | frequentist $\chi^2$ test, as CANDID |
| Input | best-fit flux and its Laplace error on a grid | model, data, grid | model, data, grid |
| Answers | what flux is credible at each position, given a source there | what flux would have been detected at $n\sigma$ | the same, with the companion injected into the data |

Use the **Ruffio** limit for a posterior statement (a percentile of the flux,
or a median and credible interval), conditioned on the astrometry being right
and a source being there. Use **Absil** or **injection** for a detection-style
limit: "a companion brighter than this would have been detected at
$n\sigma$". The two differ only in the sign of the cross term between the
data residuals and the signal: Absil uses
$\chi^2(\text{data} - s(f))/\chi^2(\text{data})$, and injection (Gallenne et
al. 2015, section 3.2, as in CANDID's `detectionLimit(methods=["injection"])`)
uses $\chi^2(\text{data} + s(f))/\chi^2(\text{data})$ for the null model.

## Ruffio et al. (2018) {#ruffio}

The unconstrained (possibly negative) best-fit flux at a position, with its
Laplace uncertainty, is the Gaussian $N(\text{mean}, \sigma)$; the positivity
prior truncates it to $f \ge 0$ (eqn 8 of
[Ruffio et al. 2018](https://arxiv.org/abs/1809.08261)).
[`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] returns a quantile of
the truncated posterior: `percentile=norm.cdf(2)` gives a 2-sigma-equivalent
upper limit, and 0.16, 0.5, 0.84 a median and credible interval. Take `mean`
from [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid] and `sigma`
from
[`laplace_flux_uncertainty_grid`][virgil.grid_fit.laplace_flux_uncertainty_grid].

The flat $f \ge 0$ prior is the published convention, and deliberately not
the Jeffreys prior for a scale: a log-uniform prior on the flux makes the
posterior improper as $f \to 0$, leaving no finite upper limit.

The quantile is taken from the upper tail, $\text{mean} + \sigma z$ with
$Q(z) = (1 - \text{percentile})\,Q(-\text{mean}/\sigma)$ and $Q$ the standard
normal survival function, in log space, so it stays accurate when the best
fit is many sigma below zero. Where the tail underflows, the large-deviation
limit $z = \sqrt{a^2 - 2\log(1 - \text{percentile})}$, with
$a = -\text{mean}/\sigma$, starts two Newton steps on $\log Q(z)$.

## Absil and injection limits {#absil-injection}

At each grid position both find the flux whose significance, a chi-squared
ratio converted by [`nsigma`][virgil.limits.nsigma], first reaches `sigma`.

- **Absil** ([Absil et al. 2011](https://arxiv.org/abs/1110.1178)): the
  ratio of the model with the companion to the no-companion model.
- **Injection**: a companion of flux $f$ is added to every observable
  (extras included, as the model with the companion minus the no-companion
  model) and the no-companion model is fitted to the result. The companion
  model fits the injected data exactly as the null model fits the original,
  so the ratio is $\chi^2(\text{data} + s(f))/\chi^2(\text{data})$ for the
  null model. With gains or OI_FLUX data, whose whitening depends on the
  model, the companion model's $\chi^2$ is computed in full.

The no-companion model sets every parameter in `grid` to zero. The degrees of
freedom are the number of data points; fitted parameters are not subtracted.

**The search.** The flux steps up from the lower bound in quarter decades
until the significance reaches `sigma`, and that step is bisected in log
flux, so a significance that crosses `sigma` and falls back within a decade
is not stepped over. For a normalized scene the significance falls again once
the companion outshines the primary (flux well above 1), and for injection it
can dip at very faint fluxes because of the cross term, so an unbounded
search (`flux_bounds=None`) starts from the smallest positive value of the
flux axis and goes at most 40 decades. Limits outside `flux_bounds` are
returned at the nearer bound, with a `RuntimeWarning` counting them.

**Range of `sigma`.** It must exceed the significance of a chi-squared ratio
of 1 (about 0.67 for many degrees of freedom) and stay below the largest
significance the float type can represent (about 12.9 in float32, 37 in
float64; enable float64 with `jax.config.update("jax_enable_x64", True)`).

**Differences from CANDID.** CANDID refits the primary's diameter to the
injected data before computing the null chi-squared (for V² and T3); virgil's
null model is exactly the one with every grid parameter at zero. CANDID
brackets the limit in flux steps of 1.4 and interpolates linearly in
significance; virgil bisects. The chi-squared and degrees of freedom are
virgil's.

## Significance {#significance}

[`nsigma`][virgil.limits.nsigma] converts a reduced-chi-squared ratio to a
Gaussian-equivalent significance, as in Absil et al. (2011): it compares
$x = \text{ndof}\,\chi^2_{r,\text{test}}/\chi^2_{r,\text{true}}$ with a
chi-squared distribution of `ndof` degrees of freedom and expresses the
upper-tail probability as a two-sided Gaussian significance. The tail is
computed directly (with the regularized incomplete gamma function), not as
$1 - \text{cdf}$, so significances stay finite and accurate far beyond 8σ,
even in float32 (up to about 13σ).

[`chi2ppf`][virgil.limits.chi2ppf] is $2\,\text{gammaincinv}(\text{df}/2, p)$,
inverting `jax.scipy.special.gammainc` by Halley's method from a
Wilson-Hilferty start (with the residual in the upper tail for $p > 1/2$).
It is accurate to about $10^{-14}$ relative in float64 for $p$ in
$[10^{-10}, 1 - 10^{-10}]$, needs no optional dependency, and is
differentiable and `jit`-able in both arguments. $p$ is clipped to
$[\epsilon, 1-\epsilon]$ of its float type and loses precision near $p = 1$,
so use `nsigma` for small tail probabilities.
