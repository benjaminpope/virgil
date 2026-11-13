# Closure-phase noise model (`_closure.py`)

Moved from the `_closure` module docstring, which keeps a short summary.

The closure phase of triangle (a, b, c) is φ_ab + φ_bc − φ_ac. Noise on the
baseline phases makes the closure phases of triangles that share a baseline
correlated, and with N telescopes only (N−1)(N−2)/2 of the triangles of one
frame and channel are independent: three of four, for four telescopes.
Treating all four as independent counts the closure phases 4/3 times.

## Covariance

The model is that of Kammerer et al. (2020, A&A 644, A110, §2.2): the
reported variances σ² on the diagonal, and a fixed correlation between
triangles from equal noise on every baseline,

    C = D^½ R D^½,   R = T Tᵀ / 3,   D = diag(σ²),

where T is the triangle-by-baseline matrix of +1, +1, −1 entries; R has
unit diagonal and ±1/3 between triangles sharing a baseline. C keeps every
reported error, is never indefinite, and scales with the errors, so
rescaled or inflated errors carry through.

## Approximation

This is an approximation whenever the σ within a group differ. True closure
phases are exact closures T b of baseline phases b, so their noise lies in
the column space of T, while C spans D^½ times it: the two agree only for
equal σ. With unequal σ the χ² of real closure-phase noise is slightly off
its nominal distribution (for four telescopes with one noisy baseline, a
mean of 2.87 against the 3 expected for three independent closure phases),
and `ClosureNoise.sample` draws noise from C that is not the closure of any
set of baseline phases. The exact model would be `C = T diag(s) Tᵀ` with
baseline variances s chosen to reproduce σ². The correlation is built from
the per-triangle errors σ, so it is exact only when the baseline phase
errors are equal; with unequal baseline errors the whitened χ² comes out
slightly low, by about 2% with one baseline three times noisier than the
others. OIFITS carries no per-baseline phase errors, so the data do not
allow better.

## Whitening

`ClosureNoise` groups the triangles that share baselines (one frame and
channel each). It whitens residuals r by dividing by σ, projecting onto an
orthonormal basis Q of the column space of T (the independent combinations)
and solving with the Cholesky factor of M = Q R Qᵀ, which is fixed. For r in
the column space of C this is exactly rᵀ C⁺ r, and the normalization is
½ log pdet C = ½ (log det M + log det Q D Qᵀ).

## Index dtype

The arrays are NumPy, built once and reused in either x64 mode. JAX (0.10
and later) caches the converted copy of a NumPy array by identity, whatever
the mode it was made in, so a float32 fit followed by a float64 one could be
handed int indices of the wrong width. The indices are therefore int32,
which is the same in both modes, and the floats are cast to the data's dtype
at use.

## Register: marginalized closure-phase offsets and wrap-penalty scaling (`gains.ClosureOffsets`, `likelihood._whiten_phases`, `epochs._Surface`, `orbit_search._Scorer`)

- **Claim:** When `OIData.with_closure_offsets` (`ClosureOffsets`, `scale_penalty=True` by default) marginalizes zero-mean Gaussian instrumental closure-phase offsets $\delta \sim \mathcal{N}(0, M \operatorname{diag}(\tau^2) M^{\mathsf{T}})$, the periodic wrap-penalty rows $2\sin^2(\Delta_i/2)$ in `likelihood._whiten_phases`, `epochs._Surface`, and `orbit_search._Scorer` divide by the nominal marginal standard deviation $\sqrt{\sigma_i^2 + \sigma_{\text{off},i}^2}$ with $\sigma_{\text{off},i}^2 = \sum_j (\tau_{\text{nom},j} m_{ij})^2$ rather than the photon-only $\sigma_i$. Passing `scale_penalty=False` keeps the unscaled `/σ_i` denominator for comparison.
- **Status:** *derived* (follows directly from Meimon, Mugnier & Le Besnerais 2005 and Lachaume 2021 / Luger, Foreman-Mackey & Hogg 2017).
- **Derivation:**
  1. In Meimon, Mugnier & Le Besnerais (2005, JOSA A 22, 2348, §3, Eq. 12), the convex unit-phasor approximation represents each closure-phase residual $\Delta_i = \phi_{\text{pred},i} - \phi_{\text{obs},i}$ by the complex residual $e^{i\Delta_i} - 1 = -2\sin^2(\Delta_i/2) + i\sin\Delta_i$ in the data-aligned frame, with circular variance $\sigma_i^2$ shared by the tangential ($\sin\Delta_i$) and radial ($-2\sin^2(\Delta_i/2)$) components. The radial row removes the false minimum of $\sin\Delta_i$ at $\Delta_i = \pi$ while contributing only $O(\Delta_i^4 / \sigma_i^2)$ near $\Delta_i = 0$.
  2. Under linear Gaussian closure-phase offset modes $\delta = M z$, $z \sim \mathcal{N}(0, \operatorname{diag}(\tau^2))$ (Lachaume 2021, PASA 38, e033; Luger, Foreman-Mackey & Hogg 2017, RNAAS 1, 7), the marginal variance of closure phase $i$ increases from $\sigma_i^2$ to $\sigma_i^2 + \sigma_{\text{off},i}^2$ where $\sigma_{\text{off},i}^2 = \sum_j (\tau_j m_{ij})^2$. Leaving the radial denominator at $\sigma_i$ when $\sigma_{\text{off},i} \gg \sigma_i$ violates the isotropic phasor variance and adds a $-\Delta_i^4 / (8\sigma_i^2)$ bias whenever $|\delta_i|^4 \gtrsim \sigma_i^2$ (for example $10^\circ$ offsets at $0.1^\circ$ photon noise).
  3. Matching the radial scale to the marginal phase variance replaces $\sigma_i$ by $\sqrt{\sigma_i^2 + \sigma_{\text{off},i}^2}$. Because the auxiliary radial rows carry constant normalization (`penalty_errors = 1/√(2π)`) so degrees of freedom are not counted twice, evaluating $\sigma_{\text{off},i}$ at a fitted width $\tau$ (`phi_offset_*`) would introduce an unnormalized $+\tfrac{3}{4} N \tau^4$ upward shift in $\mathbb{E}[\hat\tau^2]$ across $N$ shared channels; $\sigma_{\text{off},i}$ is therefore evaluated at the fixed nominal `offsets.widths` declared on `OIData.with_closure_offsets`, keeping `model_loglike` an exact smooth scalar objective for L-BFGS, Gauss–Newton, and NUTS.
- **Traceability:**
  - Unit tests (`tests/test_closure_offsets.py`, `tests/test_orbit_search.py`): `test_offsets_match_the_dense_whitened_covariance`, `test_independent_dense_kammerer_plus_offset_loglike_and_penalty`, `test_fit_with_nonzero_injected_closure_offsets_recovers_flux_and_width`, `test_wrap_penalty_scaling_gradients_and_unbiased_fit_at_tiny_photon_noise`, `test_fitted_offset_width_is_unbiased_across_many_channels`, and `test_scorer_scales_wrap_penalty_and_recovers_flux_with_closure_offsets`.
  - `virgil-validation` check: frequentist coverage check on synthetic high-SNR ($0.1^\circ$ photon noise) binary closure phases with injected $10^\circ$ frame/triangle instrumental offsets recovering companion flux, astrometry, and $\tau$ (to-do on `virgil-validation`).
- **References (ADS-verified):**
  - Meimon, Mugnier & Le Besnerais 2005, JOSA A 22, 2348 (`2005JOSAA..22.2348M`, doi:10.1364/JOSAA.22.002348);
  - Kammerer et al. 2020, A&A 644, A110 (`2020A&A...644A.110K`, arXiv:2011.01209);
  - Lachaume 2021, PASA 38, e033 (`2021PASA...38...29L`, arXiv:2104.07082);
  - Luger, Foreman-Mackey & Hogg 2017, RNAAS 1, 7 (`2017RNAAS...1....7L`, arXiv:1710.11136).
