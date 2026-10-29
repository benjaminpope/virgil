# Toeplitz (dirty-beam) form of the radio visibility χ²

Status: PR 1 ("Core") built; fitted α (Taylor moments), exact HVPs / Newton-CG and the large-data
validation runs are later PRs. Sections 1–8 below are the scoping note of 2026-10-23 (written against
`origin/main` at `3374bd0`; line references are to that commit), kept as the design record; where they and
"As built" differ, "As built" is what the code does. IM Lup is the worked example only.

## As built (PR 1)

**Status of the methods** (until 1.0 virgil makes no unsupported innovations):

- *Established:* the Toeplitz form of the normal operator `AᴴWA` on a 2N grid (Fessler et al. 2005) and the
  dirty-beam/dirty-image form of χ² (the CLEAN minor cycle). This is the default path.
- *Derived, opt-in:* free per-observation offsets from the stored spread grid (`offsets="spread"`). The
  offset ramp cancels in `|M|²`, so `B` is offset-free; with the spread grid `b`,
  `D(x + δ) = Σ_l b_l e^{2πil(x+δ)/n_f} / ψ̂(x + δ)`, a phase ramp on `b` and the kernel transform at a
  non-integer position. Evaluating a type-1 grid off the integer lattice is a type-3 NUFFT evaluation (Lee &
  Greengard 2005; Barnett et al. 2019, Sect. 3.3, for `ψ̂` at non-integer frequencies). We know of no prior use
  of this combination (per-observation phase-ramped grids with an offset-free beam term, differentiated
  through δ); radio practice rephases the visibilities instead (CASA `fixvis`). Default `offsets="none"`.

**API.** `ComplexVisData.toeplitz(image, nuisance, mask=None, offsets="none", eps=1e-14,
max_offset_mas=None, chunk_size=65536)` returns `virgil.radio_toeplitz.ToeplitzVisData`; its
`term(scene, nuisance, sigma=None)` goes into `fit(likelihoods=[...])`. Also `rebuild(vis, weight)` (jitted,
no recompilation), `direct_term`, `chi2_by_ddid`, `crosscheck`, `nyquist_fraction` and `margin` (the "guard
fraction" of section 7). A fixed spectral index comes from `CalibrationNuisance(alpha=<number>)`, not from an
`alpha=` argument of the precompute; there is no `spectral_order` yet.

**Precompute.** Points are sorted by ddid into chunks that never straddle a group; one `lax.scan` spreads each
chunk with nufftax's ES kernel (through the private adapter `virgil._nufft_spread`, version-checked at
nufftax ≥ 0.6.1) and flushes a finished group's grid.

- `B` is evaluated from its spread grid (`n_f = grid_size(2N)`) at the integer lags `m ∈ [−N, N)`, exact
  at any baseline. Lag `−N` is zeroed (no pair of pixels is `N` apart; keeping it would break the
  symmetry of the circulant embedding), then `ifftshift` and `rfft2` give the real `B̂` of shape
  `(n_ddid, 2N, N+1)`.
- Without offsets, `D` is evaluated at integer positions on `n_f = grid_size(N)` and stored as an image.
- With offsets, the grids `b_d` are stored on `n_f = grid_size(N + 2⌈δ_max⌉)` (`M = N + 2⌈δ_max⌉`) and
  evaluated per observation; the precompute refuses `max|t|/π > 1 − w/n_f` (the kernel would wrap).

**Term.** `χ²_d = (C_d − 2 g_d IᵀD_d(δ_e) + g_d² IᵀB_d I)/s_d²` with `g_d` = flux × gain of the ddid's
observation. `C_d` is dropped from the traced loss and added back as a constant (`loss_constant`), so the
float64 differences that matter are not swamped by `Σw|V|²`.

- Fixed σ (`sigmas=False`): constant `ΣC`. Fitted σ: `Σ(C + Δ)/s² + Σ 2N ln s`, no constant.
- Profiled σ: `ŝ² = (C + Δ)/2N` (held by `stop_gradient`); the traced loss is `Σ[Δ/ŝ² − sg(Δ/ŝ²)]` plus
  `ΣN sg(log1p(Δ/C))`, with constants `Σ2N` and `ΣN ln(C/2N)`: the same value and gradient as the direct
  profiled loss, without cancellation.
- Scene checks: the scene must return one `Image` (alone, or the only part of a `System`) with the template's
  shape, pixel scale, rotation and smoothing, at the template's fixed `dra`, `ddec` (a known shift, such as
  MPoL's half-pixel map convention, folded into the visibilities at the precompute as CASA's `fixvis` would;
  checked at run time with `eqx.error_if`, since they are traced under `jit`); offsets beyond the precompute's bound also raise at run
  time.
- Refused: antenna gains, a fitted α, other components, float32 (at trace time), and, with offsets, baselines
  past the margin.

**Fit protocol** (additive; existing terms untouched). A term with `chi2_and_log_norm(values)` is a scalar
term: `fit` uses it for L-BFGS and Adam, adds `loss_constant` back in the reported loss, counts `n_data`
(= 2 × the points, as the direct term), and LM, `gauss_newton_mass` and the Gauss–Newton covariance raise
`NoLeastSquaresForm`. For `precondition="diag"`, `curvature_params(values)` maps the values to
`θ = (I, shift, g, σ)` and `curvature_loss(θ)` is the loss in θ; the diagonal is `diag(JᵀH_θJ)` (exact for
small problems, Rademacher probes otherwise), floored at 0. It is the exact Gauss–Newton diagonal of the direct
term for the pixel, flux and gain blocks, and the loss-Hessian diagonal for offsets and σ (moved forward from
PR 2, because PR 1 has to give `precondition="diag"` meaningful curvature).

**Numerical choices.**

| Choice | Value | Why |
|---|---|---|
| Kernel tolerance `eps` | 1e-14 (w = 16, β ≈ 37.6) | the χ² differences that matter are ~1e-8 of `C`; 1e-14 leaves margin |
| Upsampling | σ = 2, even 2-3-5-smooth `n_f ≥ max(2M, 2w)` | nufftax's default; the Poisson-summation guarantee is for `|x| ≤ M/2` |
| `ψ̂` quadrature | 160 Gauss–Legendre nodes on `[−w/2, w/2]` | 4e-14 at 64² with 20-pixel shifts; differentiable in `x` |
| Lag −N of `B` | zeroed | no pixel pair has that lag; keeps the 2N circulant embedding symmetric |
| Offset margin | refuse `max|t|/π > 1 − w/n_f` | past it the kernel wraps and fractional shifts are wrong (integer lags are unaffected) |
| `M` | `N + 2⌈δ_max⌉`, `δ_max` from the offset prior box (rotated to the image frame) | shifted positions must stay within `M/2` |
| Precision | float64 only, `C` analytic | `C` ≈ twice the number of points (~1e8 for IM Lup) against differences of order 1 |

**Measured (CPU, float64), Toeplitz against the direct term.**

| Case | loss (rel) | loss differences (rel to compensated direct) | gradient (rel) |
|---|---|---|---|
| 15², 1500 points, offsets | 6e-16 | ≤ 1.1e-13 (direct's own: ≤ 1.2e-13) | ≤ 1.3e-14 |
| 64², 1e5 points, offsets, rotation 33°, σ fixed / fitted / profiled | ≤ 3e-16 | ≤ 3.2e-13 (direct: ≤ 5e-13) | ≤ 2.3e-14; fitted `cal.sigma` 1.4e-12 |

Fits from the same start: Adam agrees step by step to 1e-14; L-BFGS reaches the same optimum (loss to 1e-10,
values to ~1e-5, profiled σ to 1e-8).

**Measured (A100, float64), IM Lup at the config-R solution** (42,421,131 points, 26 groups, 1028² × 5 mas,
free offsets, `n_f` = 4320 for `B` and 2160 for the offset grids; one run of the section 7 check):

| Quantity | Toeplitz | Direct |
|---|---|---|
| χ² per group and loss, relative difference | 1.8e-13 | (reference) |
| Gradient, relative: flux, gains, offsets, pixels | ≤ 2.0e-10 | (reference) |
| Gradient, relative: `cal.sigma` | 2.1e-6 (∂L/∂s ≈ 0 at the optimum, so relative to a near-zero gradient) | (reference) |
| Loss difference to a perturbed image, absolute | 1.5e-8 (roundoff) | (reference) |
| Nyquist fraction / margin | 0.486 / 0.993 | — |
| Precompute | 8.1 s; arrays 2.38 GB | — |
| Value+grad | 21.8 ms | 734.7 ms (eps 1e-6), 997.7 ms (eps 1e-12): 34×, 46× |
| Compile | 0.6 s | 3.0 s |
| Peak GPU memory | 12.98 GB | 21.92 GB |

The 21.8 ms is at the top of the section 4 estimate (10–25 ms with free offsets) and above its 5–10 ms
image-only part; the estimates below are kept as written.


## Summary

A pixel image `I` (N × N, regular grid) seen through `A`, the type-2 NUFFT at the data's (u, v), has

```
χ² = Σ_k w_k |V_k − g A_k I|²  =  g² Iᵀ B I − 2 g Iᵀ D + C,
B = Re(Aᴴ W A),  D = Re(Aᴴ W V),  C = Σ_k w_k |V_k|².
```

`B` is a (block-)Toeplitz matrix: `B[p, p'] = B(x_p − x_p')` with the dirty beam
`B(Δ) = Σ_k w_k cos 2π(u_k Δx + v_k Δy)`, so `Iᵀ B I` is a convolution done with two FFTs on a 2×-padded
grid, and `D` is the dirty image. Both are computed once per data group with one adjoint NUFFT each.
Every value + gradient afterwards costs a few N² FFTs and no pass over the visibilities. It is exact
to the NUFFT tolerance used for the precompute (the MRI "Toeplitz embedding" trick), not a gridding
approximation.

**Verdict: difficulty M** (three PRs, no blocker). Per-step cost for IM Lup (42.4 M points,
26 ddids, 1028² × 5 mas) should fall from ~0.45–0.5 s to ~10–25 ms per value+grad with free offsets (estimate), 20–50× (5–10 ms and ~50–100× with offsets fixed).
Measured on an A100: 21.8 ms against 734.7 ms (direct, eps 1e-6), 34× (see "As built");
the number of L-BFGS steps is unchanged, so a cold start goes from about an hour to about a minute,
not to a different algorithm. Per-observation offsets can be fitted exactly (opt-in, "derived"; see "As built"): the spread (fine)
grid of the type-1 NUFFT is stored once per group and evaluated at any shift (section 2, tested to 4e-13). The one
nuisance that costs extra is a **fitted spectral index** (a handful of moment beams per ddid). The term can
also be rebuilt from arbitrary effective data and weights, which keeps the door open for EM schemes
(section 6).

## 1. Where the radio likelihood lives today

| Piece | Location |
|---|---|
| `ComplexVisData` (u, v in wavelengths per channel, `vis`, `weight = 1/σ²` of each of Re and Im, `freq`, `ddid`, `obs`) | `radio.py:64` |
| Model of every point: `M = F g_e (ν/ν₀)^α exp(-2πi(δα_e u + δδ_e v)) T(u,v) V_scene(u λ, v λ, λ)` | `ComplexVisData.model` `radio.py:392`, `_model_points` `radio.py:1262` |
| Nuisances (`cal.flux`, `.gain` per obs, `.offset` per obs with `offset_beams` prior, `.sigma` per ddid or `"profile"`, `.alpha`), taper `taper_fwhm_mas` | `CalibrationNuisance` `radio.py:1048`, `_unpack` `:1230`, `_taper` `:1256`, priors `:1159` |
| The loss term: `_RadioTerm`, chunked `lax.map` over padded point arrays, whitened residual vector `(vis − M) √w / s_d` | `radio.py:1294`, `_residuals_and_sigma` `:1358` |
| Profiled σ: `ŝ_d = √(χ²_d(1)/2N_d)`, `stop_gradient` | `_profile_sigma` `radio.py:1392` |
| Normalization `Σ 2 ln s_d`, `loglike` (used by numpyro) | `log_norm` `radio.py:1431`, `loglike` `:1444` |
| The pixel image model: `Image` (`softmax(log_brightness)`, optional static `smooth_fwhm_mas`, `rotation_deg`, component `flux`, `dra`, `ddec`); each pixel a delta at its centre, `V = Σ I exp(-2πi(u x + v y))`, centres from `pixel_offsets` | `models.py:1892`, `model_on_grid` `:2073`, `_centred_cvis` `:2066`; `_geometry.py:41, 50` |
| The ft path: `nufft2d2` at `eps_f64 = 1e-12`, `ft="auto"` | `_geometry.py:118-127, 206-224` |
| Primary beam: not a data-side factor; `primary_beam()` is added to the Image's log-brightness (sky × beam, one frequency) | `radio.py:1458`, docs in the `Image` docstring |
| An adjoint NUFFT already exists (display only): `_adjoint`, `_type1`, `dirty_image`, `dirty_beam`, `_uv_taper` (CASA uvtaper, not the nuisance taper) | `radio.py:690, 919, 723, 759, 942` |

How `fit` builds the loss from terms (`fitting.py`):

* `_Objective.loss` (`:545-582`) loops over `likelihoods`; a term with `residuals_and_log_norm`
  gives `(r, log_norm)` and `chi2 += Σ r²` (`:556-565`); loss is `½ χ² + log_norm + penalty − log_prior`.
* The term is also used as a residual vector in `residuals` (`:397`), `curvature_rows` (`:451-471`, the
  Gauss–Newton rows), `fit`'s data count (`ndata += int(np.size(term(start)))`, `:737`), and
  `_forward_summary` (`:1185-1194`, per-term χ²). `_check_least_squares` (`:366`) already raises
  `NoLeastSquaresForm` for fitted σ, which is how a term without residuals can opt out of LM.
* numpyro uses `term.loglike` (`likelihood.py:813-821`), so `loglike` carries over unchanged.
* Preconditioning: `fit(precondition="diag")` → `_fisher_diagonal` (`:1485-1522`): `_gn_diagonal`
  (`:1430`, exact JVP per coordinate if n ≤ 1024, else 16 Rademacher probes, each a full vjp through
  the 42 M-point model), plus prior and regulariser curvature (`_hessian_diagonal`, `:1461`); a regulariser
  with `metric_rows` (`MaxEntropy`, `imaging.py:202, 269`) supplies its own positive curvature `w·b` that
  *replaces* the data JᵀJ on the pixel coordinates.

Consequence for the design: the one real interface change is that a Toeplitz term has **no residual
vector**, only χ² (and a log-norm), so the term protocol needs a scalar entry point.

## 2. Which nuisances map

Let a *group* be one `ddid` label (a `(DATA_DESC_ID, obs)` pair, `radio.py:80-90`), so each group has one
gain `g_e`, one offset `δ_e`, one σ scale `s_d`. All other per-point factors multiply the model, and the
quadratic form needs `|factor|²`.

| Nuisance | Handling | Exact? |
|---|---|---|
| Flux `F`, gain `g_e` | scalar on the image of group d: `G_d = F g_e` | yes |
| σ (fitted, fixed or profiled) | scalar `1/s_d²` on group d's `χ²_d = G²Q_d − 2G L_d + C_d`; profiling needs only `χ²_d` | yes |
| Taper `T(u,v)` (`taper_fwhm_mas`, `Image.smooth_fwhm_mas`) | real, static, per point: fold into the weights, `w → w T²` in B and `w T V` in D. Not a convolution before A, just a different weight | yes |
| Image rotation (`Image.rotation_deg`, static) | rotate (u, v) at precompute | yes |
| Masks (`term(mask=)`) | zero weights before the precompute; one precompute per fold | yes |
| Spectral index, **fixed** | `(ν_k/ν₀)^α` is a known real factor per point: fold into weights like the taper (`w f²` in B, `w f V` in D) | yes |
| Spectral index, **fitted** | see below | to an order `J`; error ≲ 1e-10 absolute at `J = 4` in the prototype (±0.4 % bandwidth) |
| Offsets `δ_e` | `Iᵀ B I` does **not** depend on δ (the phase ramp cancels in `\|M\|²`); only the cross term does: `L = ⟨I, D(· + δ_e)⟩`. Exact for any real δ from the stored spread grid (below) | yes, to the NUFFT `eps`, no refresh |
| Primary beam | already inside the Image (single frequency, single pointing); nothing to do. A per-spw beam (`PB_d` differing by ν) means one padded FFT per group instead of one in total (~26 extra FFTs) | yes |
| `Image.dra/ddec`, `flux` in a `System` | one more shift, added to `δ_e`; flux is a scalar | as offsets |
| Other components in the scene (`PointSource`, analytic disks) | not representable as `A I`. A point source adds a rank-one piece whose cross terms with I are the beam sampled off-grid (a tiny direct NUFFT); out of scope for the first version, the term should refuse such scenes | n/a |

**Offsets in detail.** The model of group e is `Σ_p I_p exp(−2πi u·(x_p + δ_e))`, the image at the
shifted positions, so the cross term is `Σ_p I_p D_e(x_p + δ_e)` with the continuous dirty image
`D_e(x) = Re Σ_k a_k e^{i t_k·x}`, `a_k = w_k T_k V_k`, `t_k = 2π u_k` (rad per pixel).

*Free, exact offsets from the spread grid (recommended; prototype `toeplitz_shift_proto.py`).* A type-1
NUFFT is spread → FFT → divide by `ψ̂`. Spread the `a_k` once onto the fine grid (`n_f ≈ 2M` points a side,
`ES` kernel of width `w`), `b_j = Σ_k a_k ψ(j − s_k)`, `s_k = t_k n_f/2π`, and keep `b` (one `n_f²` complex
array per group, or per group and moment). Poisson summation,
`Σ_j ψ(j − s) e^{iθj} = Σ_m ψ̂(θ + 2πm) e^{is(θ+2πm)}`, holds for **any real** `x`, with `θ = 2πx/n_f`, not
only the integer modes the library returns, so with `x = m + f` (integer part `m`, fractional part
`f = 1/2 + δ`; pixel centres are half-integers for even N)

```
D_e(x_p + δ) = Re[ (−1)^m · IDFT( b · e^{2πi (j − n_f/2) f / n_f} )(m) / ψ̂(2π(m + f)/n_f) ],
```

separable in x and y, the ramp and `ψ̂` being 1-D factors. One FFT of `b·ramp` per *observation* (groups
sharing `δ_e` are linearly combined first: `Σ_d coef_d b_d`, coefficients `g s_d⁻² c_j(α)`), crop to N²,
divide, dot with `I`. The δ-gradient is by autodiff: the adjoint FFT of `I/ψ̂`, then a spectral dot with
`(2πi (j − n_f/2)/n_f) b e^{…}`, plus the 1-D `∂ψ̂/∂x`; about 2–3 FFTs per observation. The quadratic term
needs nothing (the ramp cancels in `|M|²`), and no outer refresh is needed.

Prototype (64² image, 1e5 random points, nufftax 0.6.1 spreader, `eps = 1e-12` so `w = 14`, `n_f = 216`,
`M = N + 2·20` so that |shift| ≤ 20 px, 31 % of N/2; quadrature `ψ̂` with 160 Gauss–Legendre nodes):

| shift (px) | max \|D_nufft − D_exact\| / max \|D\| | δ-gradient of `L`, relative error |
|---|---|---|
| (0, 0) | 3.4e-14 | 3.3e-14 |
| (0.3, −0.2) sub-pixel | 3.8e-14 | 2.2e-14 |
| (−0.77, 0.41) | 3.9e-14 | 4.6e-14 |
| (2.2, −1.6) | 3.9e-14 | 1.3e-14 |
| (7.5, 11.3) | 7.1e-14 | 2.5e-14 |
| (20, −20) | 3.6e-13 | 6.7e-14 |
| (28, 0), beyond the design band | 8.4e-10 | n/a (graceful, not a cliff) |

At `eps = 1e-8` the same table reads 4e-10 … 4e-9 (gradient ≤ 1e-9), i.e. the error follows the requested
`eps`, as in the library. This replaces the earlier interpolation and refresh options (a Fourier shift of the
padded dirty image gives only 3e-4, a Kaiser–Bessel-windowed sinc 2e-6 to 3e-9, both far worse).

Conditions, all checkable when the term is built:

* **`M = N + 2⌈δ_max⌉` modes** for the spread/deconvolution (the usual guarantee is for |x| ≤ M/2): for IM Lup
  with |δ| ≤ 100 mas = 20 px, `M = 1068` against 1028, a 4 % larger fine grid.
* **|t_k| ≤ π (1 − w/n_f)**, i.e. every baseline within the Nyquist band of the pixel grid, with a margin of the
  kernel footprint (`≈ 0.5 · 14/2136` ≈ 0.3 % for IM Lup). This is a requirement *only* because the shift is
  fractional: wrapping `t` by `2π` is harmless at integer `x`, but changes `e^{itx}` for fractional `x`, and
  a kernel footprint straddling the periodic seam picks up the wrong phase. Measured (`eps = 1e-12`):
  disc radius 0.40 and 0.45 cycles/pixel, errors 4e-13; 0.47 (just inside the margin) 3e-12; 0.49,
  above the margin, 1e-3 to 4e-3; 0.70 (beyond Nyquist) errors of order 1. The term must compute
  `max|t| / π` (and refuse above `1 − w/n_f`), as it must for any use of fractional shifts. IM Lup is at 0.62
  (16 km at 240 GHz = 1.3e7 λ against Nyquist 2.1e7 λ for 5 mas pixels), so it is fine. The quadratic term has
  no such requirement (it is exact for any `u`).
* nufftax 0.6.1 exposes the spread grid only through private functions: `nufftax.core.spread._spread_2d_dispatch`
  (`x, y, c, nf1, nf2, kernel_params`, batched in `c`, GPU Pallas path included),
  `core.kernel.compute_kernel_params`, `es_kernel`, and `utils.grid.compute_grid_size`; the public API stops
  at `nufft2d1`. The grid position of `t = 0` is `n_f/2`. Pin `nufftax >= 0.6.1` and import these in one
  small adapter module (with a version assertion), or write our own spreader (ES kernel, scatter-add of
  `w²` taps per point, ~40 lines; the nufftax one is faster on GPU).
* `ψ̂` at non-integer `x` is not provided (`kernel_fourier_series` returns integer-mode values only): evaluate
  the cosine transform of the kernel by Gauss–Legendre quadrature of `w/2·∫φ(z) cos(2πxz/n_f)dz`, a
  1-D `(N, nodes)` array per axis, ~2·10³ × 160 flops per evaluation, negligible and differentiable in δ.
* Memory: `n_f² × 16 B` per group (and per moment): 216² = 0.75 MB here; IM Lup `n_f ≈ 2136` is 73 MB, so
  26 groups = 1.9 GB (×5 with a fitted α: 9.5 GB, still fine on 40 GB; combine per observation before the
  FFT so only the reads grow). A Hermitian-folded grid (spread `a` at `t` and `ā` at `−t`, so `b` is
  conjugate-symmetric and `D` real) would halve memory and use `c2r` FFTs at twice the spreading cost; untested.
* Cost per evaluation: the shifted cross term needs one `n_f²` FFT per observation (5–8 for IM Lup),
  ≈ 5–10 ms on an A100 added to the quadratic term's ≈ 5 ms, so ≈ 10–20 ms per value+grad (the
  50× estimate becomes ≈ 25–40×; measured 21.8 ms, 34×). With **profiled σ** each ddid's `L_d` is needed separately each step: 26 FFTs,
  ≈ 30–40 ms, so a fixed or fitted `s_d` is much cheaper than profiling here.
* Precompute is one spread per group in place of the `D` adjoint (no FFT, so slightly cheaper); `D_e` at zero
  shift is just the same stored grid evaluated at `δ = 0`.

**Multi-frequency and fitted α.** Within a ddid the frequency spans ε = ν/ν_d − 1 ≲ ±0.4 % (2 GHz spws at
225–233 GHz). Expand about the group's centre: `f_k = r_d^α (1 + ε_k)^α`, `f_k² = r_d^{2α}(1+ε_k)^{2α}`,
`(1+ε)^a = Σ_j C(a, j) ε^j` (generalized binomial). Then

```
Q_d(α) = r_d^{2α} Σ_j C(2α, j) B_{d,j},  B_{d,j}(Δ) = Σ w T² ε^j cos 2π(u·Δ)
L_d(α) = r_d^{α}  Σ_j C(α, j)  ⟨I, D_{d,j}(· + δ)⟩,  D_{d,j} = Σ w T ε^j Re[V e^{2πi u·x}]
```

`J + 1` beams and dirty images per ddid, with `C(a, j)` differentiable in α (a polynomial). This is
CASA's MT-MFS idea but expanded narrowband around each ddid's centre rather than over the whole ±8 % band
(which is why a few terms suffice). Prototype (±0.4 %, α = 2.1, χ² = 2.3e5): absolute χ² error 0.27 (J = 0),
1.5 (J = 1), 2e-4 (J = 2), 5e-6 (J = 3), **9e-11 (J = 4)**, with gradients to 1e-15 relative at J = 5. For
integer α the series terminates at J = 2α. Plan J = 4 as the default (to be chosen from the fractional
bandwidth, which the precompute knows); memory and precompute ×5. With α **fixed** no expansion is
needed. Chromatic scenes (a scene whose V depends on λ beyond a scalar factor) are not handled.

## 3. Precision

The constant `C = Σ w |V|²` is ~1e8 for IM Lup (noise-dominated: ≈ 2N), while the differences that matter
are ~1.

* **Drop the constant from the optimized loss.** Autodiff never sees `C` anyway; the term returns
  `Δχ² = χ² − C = G²Q − 2GL` (magnitude ≈ −(signal power) ~ 1e5–1e7, not 1e8) and adds `C_d` back
  only when a number is *reported* (float64 on the host, `math.fsum`, or kept as a Python float).
  The optimizer's loss then differs from the old one by a constant, which must be documented (loss
  values in `FitResult.info` stay comparable if `C` is added back there).
* **Profiled σ** is the only place `C` enters the loss: `½ χ²_d/ŝ² + 2N_d ln ŝ → N_d(1 + ln(χ²_d/2N_d))`.
  Write `ln(χ²_d/2N_d) = ln(C_d/2N_d) + log1p(Δχ²_d/C_d)`: the first is a constant, the second is
  accurate to a relative 1e-16 of a ratio that is small near a good fit.
* **FFT accuracy.** A float64 FFT convolution has error ~1e-15 × the largest term, i.e. ~1e-8 absolute
  on a 1e7 signal power, 8 orders below the differences of interest. Float32 is not usable (error ~1
  in the loss); the term must require `jax_enable_x64` (virgil's radio path already uses f64).
* **NUFFT tolerance of the precompute is a coherent, systematic error** (the same B every step), of
  order `eps × Σ_d w_d g² / s_d²` in loss units, which for `eps = 1e-12` is acceptable only if `Σw` is
  ≲ 1e9. This is the main precision risk (section 8): measure it by comparing `χ²` from the Toeplitz
  and direct paths on IM Lup before trusting a 1e-3 loss tolerance; lowering `eps` to 1e-14 costs a
  few more kernel widths in nufftax.
* The quadratic form is **exact for any u**: B at integer lags is a sum of `cos 2π u m`, which aliases
  the same way the delta-pixel model does, so no band limit is needed for `Q` (only fractional
  shifts need the Nyquist margin, section 2).
* Build the kernel with the lag `−N` row and column zeroed: the 2N lags are `−N..N−1`, and the unreachable
  lag `−N` breaks the point-reflection symmetry that makes `FFT(B)` real (the prototype's `Im/Re` was
  1.7 % without it); zeroing it also lets the code use `rfft2/irfft2` (half the work).

## 4. Memory and cost for IM Lup (26 groups, 1028², 2056² padded)

Per evaluation, with `Î = rfft2(pad(I))` computed once (shared by all groups):

* `Q_d = (1/M) Σ B̂_d |Î|²` (spectral dot, `M = (2N)²`), summed over groups with their scalars
  `G_d²/s_d² · c_j(α)`; one **combined kernel** `K̂ = Σ_d (G_d²/s_d²) B̂_d` gives the gradient
  `2 irfft2(K̂ Î)[:N,:N]`, so there is one forward and one inverse FFT per value+grad.
* Cross terms: one FFT of the ramped spread grid per observation (section 2), crop, divide by `ψ̂`, dot with
  `I`. (A Fourier-domain Parseval form with the ramp applied to the *padded dirty image* is not used: that
  image is not periodic, giving the 3e-4 error quoted above.)
* Elementwise work: 26 × (J + 1) reads of a 2056 × 1029 f64 array.

| Item | Size |
|---|---|
| `B̂` (f64 half spectrum, 17 MB each) × 26 groups × (J+1) = 1 (fixed α) … 5 (fitted α) | 0.44 … 2.2 GB |
| spread grid `b` (n_f² ≈ 2136² c128, 73 MB) × 26 × (J+1), in place of the dirty images `D` (1028² f64, 8.5 MB each, 0.22 … 1.1 GB, if offsets are not free) | 1.9 … 9.5 GB |
| Working arrays (`Î`, `K̂`, gradients; the L-BFGS history of 100 × 1M f64 = 0.8 GB is already present) | ~1 GB extra |

Fits an A100 40 GB comfortably. If σ is fixed (MPoL's objective, `s = 1`) and α fixed, the groups of one
observation could share a kernel and memory drops ~5×; keep per group otherwise (σ and the profiled ŝ are per
ddid).

**Estimated cost per value+grad** (A100, f64; measured: 21.8 ms with free offsets, see "As built"): two 2056² real-to-complex FFTs ≈ 1–2 ms,
spectral reductions over 0.44–2.2 GB at ~1.5 TB/s ≈ 0.3–1.5 ms, pixel-space softmax/entropy/autodiff on 1M
pixels ≈ 2–4 ms, so **≈ 5–10 ms** for the image part; free offsets add 2–3 complex 2136² FFTs per observation
(5–8 observations, ≈ 5–15 ms) and the spread-grid reads. Total **≈ 10–25 ms** against the measured
0.45–0.5 s: **~20–50×** (range 15–100×); an L-BFGS step (≈ 1.6 evaluations, 0.8 s today) → ~20–40 ms. With
offsets frozen (no shifted FFTs) it is the earlier ≈ 5–10 ms.

**Precompute** (one-off): per group a type-1 NUFFT onto 2056² modes for `B` (fine grid 4112² f64 = 270 MB) and
one spread (no FFT) for the offsets' grid or one onto 1028² for `D`, per moment. The cost is the spreading, ∝ N_vis, about that of today's adjoint
(≈ 0.2–0.3 s for all 42.4 M points, `nufftax` f64, eps 1e-12; a guess): ≈ 0.5–1 s for fixed α, 3–5 s for fitted α
(J = 4). Break-even after ~5–10 old-style steps. Chunk over points as `_adjoint` already does
(`chunk_size`, `radio.py:690`), accumulating in f64.

## 5. Hessian-vector products

`χ²` is exactly quadratic in `I` for fixed nuisances, so the data Hessian in pixel space is the constant
operator `K = Σ_d (G_d²/s_d²) B_d` (for the loss `½ χ²`), applied with one forward and one inverse FFT
(~2 ms). Through the softmax `I = F softmax(η)` the Gauss–Newton Hessian is `JᵀKJ`,
`J = F(diag s − s sᵀ)`: a few elementwise ops around the same two FFTs. `jax.jvp(jax.grad(loss))` on the
Toeplitz term gives the exact Hessian-vector product at ~3–4 value+grad ≈ 20–30 ms (against ~1.5 s
now), and the pure Gauss–Newton product can be written by hand from `K`. Truncated Newton / CG (and the
existing matrix-free `lm` path, whose `cg_steps` apply `JᵀJ v`) become cheap: a CG iteration costs about one
L-BFGS step. This is where the *number* of steps, not just their cost, could fall, since the experiments in
`precond-experiments.md` show L-BFGS needs thousands of steps at weak entropy weight.

What `precondition="diag"` does today (section 1): the pixel coordinates' curvature is `w·b` from
`MaxEntropy.metric_rows` (the data JᵀJ is dropped there), but the diagonal is still computed by
`_gn_diagonal` over *all* coordinates (pixels and nuisances), hence 16 Rademacher probes, each a vjp
through all 42 M points (8–16 s per refresh). With the Toeplitz term:

* The exact GN diagonal for the pixels is closed form and costs one convolution:
  `(JᵀKJ)_pp = F² s_p² [K_pp − 2 (K s)_p + sᵀK s]`, `K_pp = Σ_d G_d²/s_d² B_d(0)` (constant, by translation
  invariance). No probes.
* Nuisance diagonals: `(JᵀJ)_{g_e g_e} = Q_e` (already computed), flux the same sum, `δ_e` needs the
  second moments of the beam (`B_uu, B_uv, B_vv`, three more kernels per observation), `s_d` and α
  are closed form from `Q_d, L_d, C_d`. Simplest first version: estimate these few from
  `_hessian_diagonal` (`v ⊙ H v`, `fitting.py:1461`) on the Toeplitz loss, floored at zero as for
  regularizers: 16 probes now cost ~0.3 s.
* The term must therefore provide either `curvature_rows` (impossible: no residual vector) or a
  `curvature_diag(values, probes)` hook; `_fisher_diagonal` (`:1485`) calls the hook for terms without
  rows. `fit_preconditioning.md` §D (a dirty-beam preconditioner in `I`) becomes cheap with the same
  kernel, but whether it helps still needs the Lanczos measurement that note asks for.

## 6. API shape

A separate precomputed object, not an option on `ComplexVisData.term`, because the precompute depends on
the *image grid* (N, pixel scale, rotation, smoothing), the mask and the nuisance configuration, none of
which the data object knows:

```python
tdata = data.toeplitz(
    image,                 # template virgil.models.Image: N, pixel scale, rotation, smoothing
    nuisance,              # CalibrationNuisance: taper, nu0, sigmas mode, alpha fitted/fixed, ref_obs
    mask=None,             # as term(mask=)
    spectral_order=None,   # J for fitted alpha; None = from the fractional bandwidth
    eps=None,              # NUFFT tolerance of the precompute (default _FT_DEFAULTS["eps_f64"])
    offsets="spread",      # "spread" (exact free shifts from the stored spread grid) | "none" (all δ = 0, no grid)
    max_offset_mas=None,   # δ_max: sets M = N + 2 δ_max / pixel; default the nuisance's offset prior bound
    alpha=None,            # value to fold in when nuisance.alpha is False
)
term = tdata.term(lambda values: Image(...), nuisance)   # same call shape as ComplexVisData.term
fit(model, priors, likelihoods=[term], regularizers=[MaxEntropy(...)], ...)
```

`ToeplitzVisData` (an `eqx.Module`, arrays only as pytree leaves so `fit` passes them as arguments like
`_RadioTerm.arrays`): `B̂` `(n_group, J+1, 2N, N+1)`, the spread grids `b` `(n_group, J+1, n_f, n_f)`
(or dirty images `D` with `offsets="none"`), `C_d`, `N_d` (unmasked counts), `Σ ln w`, and group metadata
(`ddid_obs`, centre frequency ratio `r_d`, moments, kernel parameters, guard fraction). The original
`ComplexVisData` is not held by the pytree (kept by the builder only for `direct_term`).

**Rebuilding from effective data and weights** (requested by the antenna-gain EM work, `design/antenna_gains.md`
"Stage 1b": the M-step's expected χ² is a plain χ² with `d_eff = d·conj(E[c])/E|c|²` and `w_eff = w·E|c|²`,
so each EM iteration needs `B̂` and `b` for new per-visibility `(d_eff, w_eff)`):

```python
tdata = tdata.rebuild(vis, weight)   # new per-visibility arrays (N,), same layout as the build; returns a new ToeplitzVisData
```

Design consequences: (i) the builder is split into a pure, jitted function
`_toeplitz_arrays(plan, u, v, freq, ddid, obs, vis, weight) → (B̂, b, C, N_d)` and the public constructors
around it; `plan` is the static part (grid, kernel parameters, group count, `M`, `J`, taper and α folding
factors, chunk size), the visibility arrays are arguments of fixed padded shape (the `_RadioTerm` padding and
`lax.map` over chunks), so EM iterations reuse one compilation; the taper, `(ν/ν₀)^α` factor and mask are
folded inside from the static point arrays held in `plan`. (ii) `w_eff` enters both `B` (quadratic, ∝ `w T²`)
and the data term (`a = w_eff d_eff T`); `C_d = Σ w_eff |d_eff|²` is recomputed, since an EM objective
is not χ² of the raw data (its constant differs, and `log_norm` uses `Σ ln w_eff`, a different, per-iteration
quantity: the caller owns that bookkeeping). (iii) The cost of a rebuild is the precompute, ≈ 1–5 s for IM
Lup, one pass over the points, so an EM iteration costs the E-step plus one rebuild plus a cheap image solve.
(iv) **Offsets and EM:** offsets are per observation and enter only through the shifted cross term, so they
commute with the effective data: `d_eff`, `w_eff` are independent of δ (they depend on antenna gains only),
the spread grids `b_e` are rebuilt per iteration along with `B̂_d`, and δ remains a free parameter of the
image solve, no per-observation dirty images at fixed δ are needed. (v) Visibility-dependent weights
break nothing in the Toeplitz structure: `B` is still the transform of the (now per-visibility) weights.

Protocol additions in `fitting.py` (all additive, existing terms untouched):

* `term.chi2_and_log_norm(values) → (χ² − C_dropped, log_norm)` preferred by `_Objective.loss`
  (`:556`) over `residuals_and_log_norm`; `term.constant` returns the dropped `ΣC` for reporting.
* `term.size` (or `n_data`) for the `ndata` scaling (`:737`) instead of `np.size(term(start))`.
* `_forward_summary` (`:1193`) per-term χ² through the same method.
* `has_residual_form = False`: `residuals`, `curvature_rows`, `_check_least_squares` raise
  `NoLeastSquaresForm` (`method="lm"` unavailable; `lbfgs`/`adam`/numpyro available); a `curvature_diag`
  hook for `precondition="diag"` (section 5).
* `loglike` is unchanged in form; `derived()` returns the profiled σ as now.

Scene handling: the callable returns an `Image` (or a `System` with exactly one `Image` and no other
component); the term checks that the grid, rotation and smoothing equal the template's, reads `flux`,
`dra`, `ddec` off the component, and raises otherwise (no silent fallback: a wrong grid would give a
valid-looking but wrong χ²). A small hook such as `Image._linear_pixels() → (pixels, shift_mas)` keeps
this out of `radio.py`.

Incompatible nuisances are refused at construction: `data.toeplitz` raises `ValueError` if
`getattr(nuisance, "antenna_gains", False)` is true (the marginalized antenna-gain path of the radio
antenna-gains work has a per-visibility, per-solution-interval gain that the image-plane form cannot
absorb; its EM bridge goes through `rebuild` instead). This design touches `_RadioTerm` and
`CalibrationNuisance` only by reading attributes; the new code lives in a new module.

Diagnostics and residual maps stay on the direct path: `tdata.direct_term(scene, nuisance)` returns
the ordinary `_RadioTerm` of the retained `ComplexVisData`, so `dirty_image(scene=…)`, `chi2_by_ddid`,
`sigma_hat`, per-point residuals and cross-validation reporting work unchanged;
`tdata.chi2_by_ddid(values)` (cheap) and `tdata.crosscheck(values)` (Toeplitz vs direct, relative χ²
difference) give the agreement check, and `fit(...)` could finish by running the direct term once at the
solution and reporting any disagreement. Pipelines (`virgil.pipeline`) and `imaging.py` do not build radio
terms today (`likelihoods=` is used only in `fitting`, `svi`, `likelihood`, `crossval`), so nothing
there changes; a future radio pipeline would pick the Toeplitz term by size and the direct NUFFT for
diagnostics.

## 7. Validation plan

**Prototype already run** (`toeplitz_proto.py`, not in the repo): 64²
image, K = 1e5 random points, 2 observations × 2 ddids, random gains, an offset, σ scales, a Gaussian
taper, channels ±0.4 % around each ddid centre, α = 2.1, direct DFT in JAX x64 on CPU.
χ² agreement `3.8e-16` relative (9e-11 absolute on 2.3e5) at J = 4, gradients with respect to `η`, flux,
gains, offsets, σ and α all 1e-15–3e-14 relative, using the exactly re-evaluated shifted `D`. (The
prototype's timing is meaningless: it re-evaluates `D` per call.) Findings folded in above: the offset
spread-grid result and its Nyquist-margin condition, the J requirement, and the lag `−N` symmetry detail.

**Free-offset prototype also run** (`toeplitz_shift_proto.py`): the table in section 2; includes δ-gradients
by `jax.grad` against the brute-force sum, and the Nyquist-margin failures.

**CPU unit test to add** (`tests/test_radio_toeplitz.py`, ≈ 10 s): 64² image, 1e5 points from a disc of
radius ≤ 0.45 cycles/pixel (inside the Nyquist margin; a second case beyond it must be refused), 2–3 observations with random gains,
offsets, σ per ddid, taper, α fitted, a mask; compare (i) χ² − C and `jax.grad` with respect to all fitted
values, (ii) the profiled-σ loss, (iii) `loglike`, against `_RadioTerm`, with `rtol = 1e-10` (free offsets with
`eps = 1e-14`-class settings at 64², J = 4, direct DFT as the reference to avoid the NUFFT tolerance, shifts up to 20 px); a second case with `ft="nufft"` as the
reference at `rtol = 1e-9`; an HVP check (`jvp(grad)` against finite differences of the gradient); a
refusal test (scene with a point source, mismatched grid); a `fit` smoke test (a few L-BFGS steps give the
same trajectory as the direct term to ~1e-8). Mark the 1e5-point cases slow if the CI budget needs it (the
lean-CI rule).

**GPU check** (draft, for a cluster job): fixed
inputs from the existing `alma_imlup_maxent` warm start (`values.npz`, MPoL solution, all data):

```python
# draft: precompute, agreement and timing on IM Lup (A100, float64). Run as vv's alma_imlup_toeplitz_check.py; results in "As built".
import json, time, jax, numpy as np
jax.config.update("jax_enable_x64", True)
import virgil
from virgil.radio import ComplexVisData, CalibrationNuisance
data = ...  # as in alma_imlup_maxent: same selection, taper, nu0
nuis = CalibrationNuisance(data, sigmas=False, alpha=ALPHA_FITTED, taper_fwhm_mas=30.0)
values, image = ...  # warm start (values.npz): gains, offsets, flux, alpha, 1028² log-brightness
t = time.time(); tdata = data.toeplitz(image, nuis, offsets="spread", max_offset_mas=100.0); jax.block_until_ready(tdata); t_pre = time.time() - t
tt, td = tdata.term(scene, nuis), tdata.direct_term(scene, nuis)
def bench(f, n=10):
    f(values)[0].block_until_ready(); t = time.time()
    for _ in range(n): f(values)[0].block_until_ready()
    return (time.time() - t) / n
vg_t = jax.jit(jax.value_and_grad(lambda v: tt.chi2_and_log_norm(v)[0]))
vg_d = jax.jit(jax.value_and_grad(lambda v: 0.5 * jax.numpy.sum(td(v) ** 2)))   # chi2 as the loss sees it
report = dict(
    precompute_s=t_pre,
    chi2_direct=..., chi2_toeplitz=... + tdata.constant,        # relative difference printed
    grad_rel_diff=..., s_per_vg_direct=bench(vg_d), s_per_vg_toeplitz=bench(vg_t),
    s_per_hvp=..., gpu_mem_gb=jax.local_devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9,
    guard_fraction=tdata.guard_fraction, sum_w=float(np.sum(data.weight)),
)
json.dump(report, open("out/toeplitz_check.json", "w"), indent=1)
# then: cold-start fit with the Toeplitz term, same optimizer settings as R, 3 seeds; report steps, wall time and final
# loss against the direct warm-start loss (72k better than MPoL), comparing losses as differences (chi2 - C).
```

Acceptance: relative χ² difference ≤ 1e-9 and loss difference between the two paths (constant added
back) ≲ 1e-2 in absolute terms; gradient agreement ≤ 1e-8 relative; ≥ 30× speed-up per value+grad;
memory < 20 GB; then a cold-start fit that reaches the same loss as the direct run.

## 8. Effort, PR breakdown, risks

| PR | Content | Size |
|---|---|---|
| 1. Core | `toeplitz()` precompute (reuse `_adjoint`/`_type1`, `radio.py:690, 919`, with `normalize=False`, the 2N lag grid, lag −N zeroed, `rfft2`; a nufftax-spreader adapter and the `ψ̂` quadrature), `ToeplitzVisData` with a pure jitted `_toeplitz_arrays` and `rebuild`, `term` with groups, gains, flux, taper, mask, fixed or profiled or fitted σ, **fixed α**, **free offsets from the spread grid** (with the margin and `M` checks), scene checks, protocol additions in `fitting.py` (`chi2_and_log_norm`, `size`, no-residual opt-out), `direct_term`/`crosscheck`, unit tests, a Radio-guide section | M (~700 lines + tests) |
| 2. Spectral index and curvature | fitted α by the J-moment expansion, `Image` `dra/ddec`, `curvature_diag` hook and the closed-form pixel GN diagonal for `precondition="diag"`, exact HVP helper | M (~350 lines) |
| 3. Validation and use | OzSTAR scripts (check + cold-start fit), timings, docs numbers, optional Newton-CG / matrix-free GN solver on `K` | S–M, mostly cluster time |

Optional later: per-spw primary beams (one FFT per group), point-source add-ons, a Toeplitz-based
dirty-beam preconditioner (`fit_preconditioning.md` §D), CLEAN major cycles on the same beams.

Risks, roughly by importance:

1. **Systematic NUFFT error in `B`** scaling with `Σw` (section 3): may force `eps ≤ 1e-13` or an
   extended-precision kernel; measure first, with the A100 script, before PR 2.
2. **Reliance on nufftax internals** for the spread grid (private `_spread_2d_dispatch`, kernel parameters):
   pin the version behind one adapter with an assertion and a CPU test; our own spreader is the fallback.
   Free offsets also require every baseline inside the Nyquist band minus a kernel margin (`1 − w/n_f`), a
   hard limit on the pixel scale (the term refuses otherwise); IM Lup is at 62 %.
3. **Fitted α** multiplies memory and precompute by `J+1 = 5`; expect 2–3 GB, fine on an A100, tight on
   small GPUs.
4. **Scope of scenes:** one `Image` only; composite scenes fall back to the direct term, so users must
   choose consciously (the term raises rather than guessing).
5. **Protocol change in `fitting.py`** (scalar terms) touches `loss`, `ndata`, `_forward_summary` and the
   preconditioner; keep it behind `hasattr` so existing terms are untouched, and test every existing
   term type for regression.
6. **Loss constant** changes the reported loss; keep `FitResult.info["loss"]` comparable by adding `C` back
   there, and say so in the docs.
7. The speed-up lowers the cost per step, not the number of steps: if L-BFGS needs 5000 steps, that is
   still minutes, but a cold start that is not converging for conditioning reasons needs the Newton-CG
   follow-up (section 5) to benefit.
8. Float32 and CPU: the term is f64-only; on CPU the FFT cost at 2056² (~30 ms) still gives a ~10×
   improvement, not 50×.

*Note (0.4.2): `_residuals_and_sigma`, `_type1` and `_uv_taper` cited above were inlined into their one caller each (virgil#535); line numbers are from before.*
