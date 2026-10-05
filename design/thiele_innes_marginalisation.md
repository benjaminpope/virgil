# Analytic marginalisation of the Thiele–Innes constants in orbit fits

Status: **design**, 2026-10-05 (approved by Ben; decisions D1–D6 recorded in §6). Nothing here is implemented yet. It builds on Stage 6a.1 ([`imaging_plan.md`](imaging_plan.md)) and on the orbit note ([`orbit_scene_joint_fitting.md`](orbit_scene_joint_fitting.md), "O" below), whose conventions (O §2.1) and Thiele–Innes definitions (O §2.4) it uses unchanged. The algebra below is checked numerically by [`sketches/thiele_innes_marginal_check.py`](sketches/thiele_innes_marginal_check.py), which uses NumPy only and runs in a few seconds; its checks are cited as [S1]–[S7].

## 0. Summary

- **Relative positions are linear in the Thiele–Innes constants ψ = (A, B, F, G) at fixed φ = (P, e, t_peri).** With a Gaussian prior on ψ, ψ integrates out in closed form (Luger, Foreman-Mackey & Hogg 2017). A sampler or optimiser then sees three nonlinear parameters instead of seven, and ψ is recovered exactly afterwards from its conditional Gaussian. `starting_orbits` already does the profile (least-squares) version of this. The marginal version adds the prior and the log-determinant, which depends on φ and must be kept (§1.2).
- **The catch is the prior.** No Gaussian prior on ψ, and no scale mixture of isotropic Gaussians, gives isotropic orientations. The isotropic zero-mean Gaussian implies p(cos i) = (1 − cos² i)/(1 + cos² i)², which gives orbits within 30° of face-on about 1% of the prior mass instead of 13% (§2.2, [S3]). That matters most for short arcs, where marginalisation would help most.
- **Recommendation (§2.4):** marginalise under an isotropic Gaussian prior, then **importance-reweight** the recovered Campbell elements to the prior the user asked for (option b), and report the Pareto-smoothed importance-sampling diagnostic k̂. When k̂ > 0.7, fall back to full-parameter NUTS (option c), initialised from the reweighted draws. Accepting the implied prior (option a) is dropped (D1): the Gaussian is only a computational proposal. Face-on systems are expected to need the fallback.
- **Scope (§3):** exact for positions alone. In joint position + RV fits the line-of-sight constants C and H are functions of ψ, so ψ no longer marginalises. Two partial schemes exist: marginalise the RV amplitude and zero points given (φ, ψ), as an extension of #192; or follow Wright & Howard (2009) and sample (i, Ω) as well, keeping (C, H) linear. Fits to closure phases admit no marginalisation. The method still supplies their starting points, a fast intermediate posterior from per-epoch positions, and an importance proposal for the joint fit.

## 1. The mathematics

### 1.1 The linear model
Per O §2.4, with X = cos E − e and Y = √(1 − e²) sin E (E the eccentric anomaly, a function of t through φ):

$$
\mathrm{dra}(t) = B\,X(t) + G\,Y(t), \qquad \mathrm{ddec}(t) = A\,X(t) + F\,Y(t).
$$

For epochs k = 1…n, stack the data as d = (dra₁, ddec₁, …, draₙ, ddecₙ) (length 2n, the row order of `PositionData` and `_thiele_innes_fit`). The design matrix D(φ) is 2n × 4, with columns ordered (A, B, F, G) and one 2 × 4 block per epoch:

$$
D_k(\varphi) = \begin{pmatrix} 0 & X_k & 0 & Y_k \\ X_k & 0 & Y_k & 0 \end{pmatrix}.
$$

The model is d = D(φ) ψ + noise, and the noise covariance C is block diagonal with the per-epoch 2 × 2 blocks `PositionData.cov`. These blocks may be correlated: a Laplace covariance of a binary fit is generally not diagonal in (dra, ddec). `PositionData` already stores the whitener W_k = L_k⁻¹ (C_k = L_k L_kᵀ). Write d̃ = W d and D̃ = W D for the whitened data and design. With diagonal C the problem splits into two 2 × 2 problems, (A, F) from ddec and (B, G) from dra, which share one normal matrix when the two coordinates have equal weights (Lucy 2014, App. A). A correlated C_k couples them, and the general 4 × 4 form below covers both.

D depends on (P, e, t_peri) only. It does not depend on i, ω or Ω, so nothing in §1.2–1.3 degenerates at face-on or edge-on orbits. Those cases are problems only for the map from ψ to Campbell elements (§1.4), and for the prior (§2).

### 1.2 The marginal likelihood
Take the prior ψ ~ N(μ, Λ) with Λ = S Sᵀ (S diagonal in practice; μ and S may depend on φ, §2.3). Then d | φ ~ N(D μ, C + D Λ Dᵀ), a dense 2n × 2n Gaussian. By the Woodbury identity and the matrix determinant lemma, it reduces to a 4 × 4 problem in the prior-whitened basis. With

$$
K = \tilde D S \;(2n \times 4), \quad \tilde r = \tilde d - \tilde D \mu, \quad \tilde M = I_4 + K^\top K, \quad u = \tilde M^{-1} K^\top \tilde r,
$$

the log marginal likelihood is

$$
\ln Z(\varphi) = -\tfrac12\left(\tilde r^\top \tilde r - \tilde r^\top K u\right) - \tfrac12 \ln\det \tilde M - \tfrac12 \ln\det C - n \ln 2\pi. \tag{M1}
$$

[S4] checks (M1) against the dense 2n-dimensional Gaussian and against brute-force 4-D quadrature, with correlated per-epoch covariances. The closed form equals the dense Gaussian to 1e-8 in ln Z, and the quadrature to better than 1e-5.

**Keep the log-determinant.** ½ ln det M̃ = ½ ln det(I + Λ Dᵀ C⁻¹ D) depends on φ. It is the Occam factor: the ratio of prior to posterior volume in ψ. It is what makes this a marginal likelihood rather than a profile likelihood, which is what `starting_orbits` ranks by today (χ² at the least-squares ψ). Dropping it is the error that The Joker's authors corrected in their 2020 update (Price-Whelan et al. 2020, §3.1 and App. A). Lucy (2014) used the profile likelihood with a delta function in ψ, and found credible intervals for the ψ-derived elements (a, i, ω, Ω, and the mass) too narrow by factors of 1.1–2.1. He pointed to the conditional Gaussian (§1.3) as the remedy.

**The prior must be proper.** For a flat prior (Λ → ∞), ln Z diverges by the constant −½ ln det Λ. The finite part, −½ χ²_min − ½ ln det(D̃ᵀD̃), also diverges wherever D̃ is rank-deficient: when every epoch falls at the same phase (period aliases, P = Δt/k), or in other near-degenerate geometries. A flat prior therefore produces spurious modes at exactly the aliases a short arc is prone to. As in #192 for the RV zero points, only finite prior widths are supported.

**Numerics.** All eigenvalues of M̃ are at least 1, so its Cholesky factor exists and ln det M̃ ≥ 0 for any data, however degenerate the design. Its condition number is about 1 + (σ_prior/σ_post)², though, which in float32 overflows the 1e-7 precision once the prior is 10⁴ times wider than the posterior. Use the QR factorisation of the stacked (2n + 4) × 4 matrix [K; I₄] instead: its R satisfies RᵀR = M̃, so ln det M̃ = 2 Σ ln |R_jj|, and its condition number is only the square root of M̃'s. Fitting already runs in float64 (`_precision`). The QR route keeps float32 sampling usable too.

**Residuals for `fit`.** virgil's fitters consume one residual vector (`AGENTS.md`). The marginal has an exact one, from variable projection (Golub & Pereyra 2003):

$$
\rho(\varphi) = \begin{pmatrix} \tilde r - K u \\ u \end{pmatrix}, \qquad \rho^\top\rho = \tilde r^\top \tilde r - \tilde r^\top K u .
$$

This is the stacked data-plus-prior residual at the conditional mode of ψ, of length 2n + 4, and its squared norm is exactly the quadratic form in (M1). The φ-dependent ½ ln det M̃ goes into the term's `log_norm`, the mechanism #192 adds for jitter and RV zero points. `fit` then minimises with L-BFGS (LM raises `TypeError` for terms with a `log_norm`). Its gradient needs no special treatment: u minimises the stacked residual, so by the envelope theorem the derivative of ρᵀρ with respect to φ equals the partial derivative at fixed u. JAX's derivative through the solve gives the same.

### 1.3 The conditional posterior of ψ
Given φ,

$$
\psi \mid d, \varphi \sim \mathcal N\!\left(\mu + S u,\; S \tilde M^{-1} S^\top\right). \tag{M2}
$$

[S4] checks (M2) against the moments of the brute-force quadrature. The posterior over all seven elements is then exact by composition: draw φ from p(φ | d) ∝ Z(φ) p(φ) (by NUTS, a dense 3-D grid, or rejection sampling, §4), and then ψ from (M2), one or more draws per φ. Lucy (2018) builds the same cascade on a grid: φ on a grid, and Monte Carlo draws of ψ and of the RV parameters.

### 1.4 From ψ to the Campbell elements, and the node
`ThieleInnesOrbit.to_kepler` already inverts §2.4, and every ψ draw goes through it. The inversion rests on two complex numbers whose moduli are the "prograde" and "retrograde" radii:

$$
z_1 = (A + G) + \mathrm{i}(B - F) = a(1 + \cos i)\,e^{\mathrm{i}(\omega + \Omega)}, \qquad
z_2 = (A - G) - \mathrm{i}(B + F) = a(1 - \cos i)\,e^{\mathrm{i}(\omega - \Omega)},
$$

so a = (|z₁| + |z₂|)/2 and cos i = (|z₁| − |z₂|)/(|z₁| + |z₂|). These are the half-sum and half-difference forms the code uses.
- **The node.** ψ is invariant under (Ω, ω) → (Ω + 180°, ω + 180°) ([S2]; only C and H change sign), so the map from Campbell elements to ψ is two to one. `to_kepler` returns Ω ∈ [0°, 180°). Positions alone cannot choose: report both nodes with equal weight (Ω and Ω + 180°, with dz reversed), unless RVs or a front–back-asymmetric scene component decide (O §2.3, §3.2 below).
- **Face-on (i → 0):** |z₂| → 0, so ω − Ω is undefined and only ϖ = ω + Ω is measured (O §2.3.4). The ψ posterior is perfectly regular there, and only the derived angles are degenerate. Report ϖ, and ω and Ω separately only when |z₂| is significantly non-zero. The retrograde case, i → 180°, is the same with |z₁| → 0.
- **Edge-on (i → 90°):** AG − BF = a² cos i → 0, so (A, B) is parallel to (F, G) and the sky orbit is a segment. Nothing is singular, but i and 180° − i are then nearly degenerate, and the sense of rotation is poorly measured (O §2.3.3).

### 1.5 Degenerate and periodic cases in φ
- **e → 0.** X = cos M and Y = sin M, so a shift in t_peri is a rotation of (X, Y), which ψ absorbs. With an isotropic prior (§2.3) the marginal likelihood is then **exactly** independent of t_peri at e = 0 ([S5]: a spread of 5e-12 in ln Z over seven t_peri). This is correct, but it is a flat circle in φ. Parameterise the phase as the mean anomaly at t_ref, M_ref = 2π(−dt_peri)/P, which is periodic. The pair (e cos M_ref, e sin M_ref) makes the likelihood analytic at e = 0: by the d'Alembert property, terms of order e^k carry harmonics of M_ref up to k. A uniform prior on e then needs a density ∝ 1/e on that plane. The usual (√e cos M_ref, √e sin M_ref) gives a uniform-e prior for free, but at the origin the likelihood is then only C¹ (its first-order term is e cos(M_ref − α)). Both are acceptable for NUTS. The second matches `KeplerOrbit`'s √e cos ω convention, so it is the default (decision D4: uniform e as the interim prior, with population e priors applied by reweighting).
- **e → 1.** D is finite; the Kepler solver's derivatives become steep. This is unchanged from the full-parameter fit.
- **Too few epochs.** There are 2n data and 4 linear parameters. With n = 2, every φ fits exactly and p(φ | d) is prior × Occam factor only. With n = 3, a one-parameter family of φ fits exactly. Only n ≥ 4 epochs at distinct phases constrain all three of φ through χ², which is the classical statement. The proper prior keeps ln Z finite in all these cases (§1.2).
- **Period aliases.** P and P/k (or P = Δt/k for regularly spaced epochs) give near-equal ln Z. This is genuine multimodality, which the 3-D marginal lets us map completely (§4).

## 2. Priors: the crux

### 2.1 The Jacobian
With ψ = ψ(a, i, ω, Ω) from O §2.4,

$$
\left|\frac{\partial(A, B, F, G)}{\partial(a, i, \omega, \Omega)}\right| = a^3 \sin^3 i,
\qquad\text{equivalently}\qquad
\mathrm{d}A\,\mathrm{d}B\,\mathrm{d}F\,\mathrm{d}G = a^3 (1 - \cos^2 i)\,\mathrm{d}a\,\mathrm{d}\!\cos i\,\mathrm{d}\omega\,\mathrm{d}\Omega
$$

(over Ω ∈ [0, π), where the map is one to one). Derivation: the map ψ → (z₁, z₂) is linear with |det| = 4. Polar coordinates give dz₁ dz₂ = |z₁||z₂| d|z₁| d|z₂| d arg z₁ d arg z₂, with |z₁||z₂| = a² sin² i. Then ∂(|z₁|, |z₂|)/∂(a, cos i) has |det| = 2a, and ∂(ω + Ω, ω − Ω)/∂(ω, Ω) has |det| = 2. [S1] checks the result by finite differences at 200 random points (to 4e-8).

Lucy (2014) adopts uniform priors on ψ, and Lucy (2018) "weak, non-informative" ones. By the Jacobian, a uniform ψ prior means p(a, cos i) ∝ a³ sin² i, which favours large and edge-on orbits. It is also improper (§1.2).

### 2.2 What a Gaussian prior on ψ implies
For the isotropic zero-mean prior ψ ~ N(0, s² I₄), using |ψ|² = (|z₁|² + |z₂|²)/2 = a²(1 + cos² i):

$$
p(a, \cos i, \omega, \Omega) = \frac{a^3 (1 - \cos^2 i)}{(2\pi s^2)^2}\,\exp\!\left[-\frac{a^2 (1 + \cos^2 i)}{2 s^2}\right], \qquad \Omega \in [0, \pi).
$$

- **ω and Ω are uniform and independent of the rest.** This is the reason for choosing s² I (§2.3).
- **Inclination.** Integrating out a gives

  $$
  p(\cos i) = \frac{1 - \cos^2 i}{(1 + \cos^2 i)^2},
  $$

  which is normalised on [−1, 1]: substitute cos i = tan t, and the integrand becomes cos 2t. Its density is 1 at edge-on, twice the isotropic ½, and 0 at face-on. The prior mass within 30° of face-on (i < 30° or i > 150°) is 1 − sin(2 arctan cos 30°) = 0.0103, against 1 − cos 30° = 0.134 for isotropic orientations ([S3], by 2 × 10⁶ draws).
- **Semimajor axis.** Given cos i, a = s χ₄ / √(1 + cos² i), with χ₄ the chi distribution with 4 degrees of freedom. Its mode is at a² = 3s²/(1 + cos² i), and its relative width is about ±35%. So the prior on a is informative at the scale s, and a and i are coupled: at fixed |ψ|, edge-on orbits are larger.
- **No bounded prior on ψ gives isotropic orientations.** By §2.1, any prior density g(ψ) on ψ implies p(a, cos i, ω, Ω) = g a³ (1 − cos² i). If g is bounded, as every Gaussian is, this vanishes at face-on, whereas the isotropic density is uniform in cos i.
- **No scale mixture fixes the inclination either.** Suppose the prior depends on ψ only through |ψ|², as for any isotropic Gaussian or scale mixture of them, including a hierarchical s. Then substitute ϱ = a√(1 + cos² i), and p(ϱ, cos i) factorises with the same p(cos i) as above, whatever the radial profile. A broad hyperprior on s can make the prior on a close to log-uniform, but nothing radial makes the orientation isotropic.
- **The isotropic target in ψ space is not Gaussian.** An isotropic orientation with a prior p(a) has density ∝ p(a)/(a |z₁||z₂|) in ψ, which is singular on |z₁| = 0 and |z₂| = 0, the face-on orbits. No closed-form marginal exists for it.

This is the known pattern for linear parameters in orbit fits. The Joker marginalises (K, v₀) under a Gaussian prior. Its 2020 update scales σ_K with P and e (Price-Whelan et al. 2020, eq. 4), so that at fixed primary mass the implied prior on the companion mass does not depend on P or e. orvara marginalises only parameters whose Gaussian priors are physically natural: parallax, barycentre proper motion and RV zero points (Brandt et al. 2021). It samples the orientation with its usual priors. orbitize! samples Campbell elements (or its alternative bases), and also γ and jitter, with no linear marginalisation (Blunt et al. 2020). Its OFTI algorithm instead scales a and rotates Ω on prior draws to match one epoch, then rejection-samples (Blunt et al. 2017). Wright & Howard (2009) profile the linear parameters inside Levenberg–Marquardt, with no prior.

### 2.3 The prior to marginalise with
- **Shape: zero mean and isotropic, ψ ~ N(0, s² I₄).** Changing Ω by δ rotates (A, B) and (F, G) by δ, and changing ω rotates (A, F) and (B, G). Both are orthogonal maps of ψ, and only zero-mean s² I is invariant under both. It is therefore the only Gaussian that keeps ω and Ω uniform. It also makes the e = 0 invariance of §1.5 exact.
- **Scale tied to the period (a proposal setting, D6).** For angular orbits Kepler's law gives a ∝ P^{2/3} (M_tot ϖ³)^{1/3}, so take s(P) = s₀ (P/P₀)^{2/3}, by analogy with The Joker's σ_K(P, e). Here s₀ is set from a plausible mass and distance, or from the data's own scale (e.g. the median separation). In the prior-whitened form (M1) this φ-dependence is already inside −½ ln det M̃, since K = D̃S(φ); do not add a separate −½ ln det Λ(φ). That term appears only in the equivalent decomposition ln det M̃ = ln det Λ + ln det(Λ⁻¹ + D̃ᵀD̃). Either way it must be kept (§1.2).
- **s is a proposal setting, not a prior (D6).** It is fixed at s₀(P/P₀)^{2/3}, or adapted per (P, e, T₀) for efficiency, with the importance weights of §2.4(b) using the same s. A hierarchical s, sampled as a fourth nonlinear parameter, is ruled out: it would replace the invariant prior with a radial mixture, which cannot give isotropic orientations (§2.2).

### 2.4 Options, and the recommendation
**(a) Accept the implied prior and state it (dropped; D1).** The rule is that priors are the invariant measures unless there is strong information otherwise (`imaging_plan.md`, standing design decision 9), and the implied prior is non-isotropic, p(cos i) ∝ (1 − cos² i)/(1 + cos² i)². The Gaussian is only a computational proposal. The rest of this paragraph records why the option is unsafe. It would be valid when the data pin i and a far more tightly than the prior varies: well-covered orbits, in Lucy's terms an orbital coverage f_orb ≳ 0.6. The choice of prior then barely matters (Lucy 2014, §5.4). It is wrong in exactly the regime where marginalisation pays most: short arcs, where i is poorly constrained, the prior pushes orbits toward edge-on and so biases a and the dynamical mass a³/P². It must never be silent.

**(b) Importance-reweight to the desired prior.** For each draw (φ, ψ) from §1.3, converted to Campbell elements θ = (a, i, ω, Ω), the weight is

$$
w \;\propto\; \frac{\pi_{\rm target}(\varphi, a, i, \omega, \Omega)}{\pi(\varphi)\,\mathcal N(\psi;\,0,\,s(\varphi)^2 I)\;a^3 \sin^3 i},
$$

where π(φ) is the prior on φ used in the marginal, normally the target's own, so that it cancels. The target density in Ω is folded onto [0, π), since the weights cannot see the node. The weights apply to any target: isotropic i, log-uniform a, a prior on the total mass with a distance prior, or external element priors (O R5). [S6] checks the identity on prior draws. Weighting N(0, s² I) draws to isotropic i and log-uniform a over a decade recovers both to within 3%, at an effective sample size of 4%.
- **Failure mode 1, face-on support.** Near face-on, w ∝ 1/sin² i. If the posterior has support at i → 0 or 180°, the variance of the weights diverges (logarithmically in cos i). The estimate is still consistent, but unreliable.
- **Failure mode 2, the Gaussian tail.** The factor exp(+|ψ|²/2s²) blows up when the posterior a lies far above s.
- **Diagnostic.** Both failures are diagnosable. Report Pareto-smoothed importance sampling's k̂ and the effective sample size (Vehtari et al. 2024). k̂ < 0.7 is reliable, and k̂ > 0.7 means fall back.
- **Fixed a.** A delta-function prior on a (a fixed scale) cannot be reached by reweighting. It is also a nonlinear constraint on ψ, so it needs (c).

**(c) Marginalise only for exploration, and sample the full parameters with NUTS.** The marginal gives starting points, MAP fits in 3-D, a map of the modes, and a mass matrix: the conditional covariance (M2) is the ψ block, and the 3-D Laplace covariance is the φ block. The reported posterior then comes from 7-D NUTS with the user's priors, in `ThieleInnesOrbit` or `KeplerOrbit` coordinates. This is always correct, but it inherits 7-D NUTS's difficulties on short arcs (O R4). Worse, separate chains stuck in separate modes do **not** give the relative weights of the modes.

**Recommendation, adopted as D1: (b), with (c) as the fallback and the cross-check.** The fallback is full 7-D NUTS initialised from the reweighted draws, used when PSIS k̂ > 0.7. Face-on systems are expected to need it.
- Option (b) gets mode weights right automatically. The modes live in 3-D φ space, where p(φ | d) can be mapped exhaustively (§4). Only (b) and (c) honour the user's prior, and (c) cannot weight modes.
- Its failure modes are diagnosable per run (k̂).
- The fallback reuses the same machinery: chains initialised from the reweighted draws in every mode with non-negligible weight, and the mode weights taken from (b) when k̂ is acceptable within each mode.
- Option (a) is not offered (D1). The target prior is set by D2.

**Default target prior (D2).**
- Log-uniform a and P, equivalently log-uniform P and μ = a³/P².
- Isotropic orientation (uniform cos i, ω, Ω).
- Uniform phase.
- Uniform e, as the interim prior (D4).
- Stated bounds on a and P, with a check that the posterior is proper.

A parallax prior (Gaia) overrides this as strong information. With the distance known, the scale prior goes on the total mass (1/M).

## 3. Scope and limits

### 3.1 Positions alone
The method is exact and complete: `PositionData` with correlated per-epoch covariances, any number of epochs, and either node reported (§1.4). This includes the R0 fast route, where per-epoch Laplace positions stand in for the visibilities (O R0, R7).

### 3.2 Joint positions and radial velocities
`RVData` predicts v = γ + share · D κ · v_z, with v_z = C Ẋ + H Ẏ and κ the km/s per mas/day at 1 pc. C and H are the line-of-sight constants of O §2.4. They are **not free**: (A, B, C)/a and (F, G, H)/a are orthonormal, so given ψ they are fixed up to a common sign, which is the node. The RV term is therefore a nonlinear function of ψ, and ψ cannot be marginalised exactly in a joint fit. Wright & Howard (2009, §3.3) note the same overconstraint: the six linear constants A, B, F, G, C, H depend on only five Keplerian elements. Three partial schemes keep some of the gain.
1. **Zero points only (exact; #192).** The RV zero points (γ and per-instrument offsets) are linear in every case, and #192 marginalises them. This combines with any treatment of the orbit.
2. **RV amplitudes and zero points given (φ, ψ) (exact; extends #192).** At fixed (φ, ψ), v_z(t) is known, so the RV is linear in the amplitude s = share · D κ and in the zero points. Add v_z(t) as one more column of #192's design matrix, or two for a double-lined binary, where the amplitudes Dκ q/(1+q) and Dκ/(1+q) are Lucy's (2018) K₁ and K₂.
   - The sampler sees (φ, ψ): 7 dimensions instead of 7 + (q, D, γ, offsets).
   - **The node is the sign of s.** (C, H) → −(C, H) with s → −s leaves the RV unchanged. The physical sign of s (negative for the primary in the §2.1 convention) selects the node. A half-Gaussian prior on s keeps the marginal closed-form (an erfc factor), and the posterior node probability is the ratio of the two half-space evidences.
   - **Priors.** The Gaussian prior on s must be reweighted to the physical priors on q and D, as in §2.4(b). In a single-lined binary only s is identified, not q and D separately.
3. **Wright & Howard's scheme (exact, but singular face-on).** Promote (i, Ω) to nonlinear parameters. Then at fixed (φ, i, Ω), ψ is linear in (C, H) ([S7]):

   $$
   \begin{aligned}
   A &= (H\cos\Omega - C\sin\Omega\cos i)/\sin i, & B &= (H\sin\Omega + C\cos\Omega\cos i)/\sin i,\\
   F &= -(C\cos\Omega + H\sin\Omega\cos i)/\sin i, & G &= -(C\sin\Omega - H\cos\Omega\cos i)/\sin i.
   \end{aligned}
   $$

   - With D and q fixed or sampled, positions and RVs are then jointly linear in (C, H) and the zero points. That leaves 5 nonlinear orbit parameters plus the RV scale.
   - The coefficients diverge as 1/sin i, so the scheme fails near face-on. Face-on orbits are exactly where RVs carry no information anyway.
   - A Gaussian prior on (C, H) implies uniform ω and a Rayleigh prior on a sin i, so it needs reweighting too.

Scheme 2 is preferred. It is exact, regular at all inclinations, and a small extension of #192. **Decision D5: deferred until a real system needs it.** When built, K is a scale parameter, so a Gaussian on K is again only a proposal and needs reweighting (§2.4(b)).

### 3.3 Joint fits to interferometric data
Closure phases, V² and kernel phases are nonlinear in the companion's position, and the flux ratio does not enter linearly either: the binary's normalised visibility has the flux in its denominator. There is nothing to marginalise in a joint visibility fit (O R7), but the method still helps in four ways:
- **Starting points:** per-epoch positions, then the marginal over φ on a grid, as `starting_orbits` does now. It keeps the χ² ranking by default (D3): starting points make no inferential claim. Ranking by posterior comes only once reweighting exists.
- **Intermediate data.** When per-epoch Laplace positions are close to sufficient (two point sources well inside the field; O R0), the position-based posterior of §1.3 *is* the answer to good accuracy. `simulate`/`bias_test` decides when that holds.
- **An importance proposal for the joint fit.** Reweight draws from the position-based posterior by L_vis(θ)/L_pos(θ). This is valid where the positions capture most of the visibilities' information, and it is diagnosed by k̂ like §2.4(b). When it fails, it still supplies initialisations and a mass matrix for joint NUTS.
- **Mode weights** for multimodal short arcs, which joint NUTS chains cannot provide (§2.4).

## 4. Cost and benefit
- **Dimension, 7 → 3.** For NUTS on a well-conditioned Gaussian target, the gradient evaluations per effective sample grow only as about d^{1/4}, so the dimension alone is worth about (7/3)^{1/4} ≈ 1.2. **The real gains are geometric.** The marginal removes:
  - the curved Campbell-element ridges of short arcs (O R4);
  - the ω/Ω degeneracy of near-face-on orbits;
  - the (Ω, ω) + 180° bimodality, since ψ is unique.

  What remains is the genuine multimodality of (P, t_peri). These gains have to be measured (§5.2), not assumed.
- **Exhaustive 3-D maps.** p(φ | d) can be evaluated on a dense grid: Lucy (2014) used 800 × 200 × 200. Or it can be rejection-sampled from the prior, as The Joker does. Each point is independent and vmappable, so a GPU evaluates 10⁷–10⁸ points directly. This gives exact posteriors and mode weights without MCMC mixing, which is the main benefit for short arcs.
- **Cost per evaluation.** The Kepler solve at n epochs is the same as in the full-parameter likelihood. On top of it come forming K (O(n)), a QR of a (2n + 4) × 4 matrix (O(n)), and two 4 × 4 triangular solves. Per gradient the cost is expected to be within a factor of 2 of the full likelihood. NUTS on GPUs is launch-bound at these sizes anyway.
- **What breaks.**
  - A flat prior gives spurious alias modes (§1.2), which is why it is not supported.
  - In float32, the Cholesky of M̃ fails once σ_prior/σ_post ≳ 10³–10⁴. Use the QR route (§1.2).
  - Gradients through ln det M̃ are bounded, since tr(M̃⁻¹ dM̃) with M̃ ≥ I. The derivatives of D̃ with respect to e still grow near e → 1, as in the full-parameter fit.
  - At e = 0 the t_peri direction is flat. Use the parameterisation of §1.5.
  - The importance weights fail near face-on (§2.4(b)). This is diagnosed by k̂.
  - Converting a draw of ψ that is very nearly zero (a data set with no orbit) is refused by `to_kepler`. That is correct.

## 5. Proposed API and test plan

### 5.1 API (in `src/virgil/orbits.py`)
```python
prior = ThieleInnesPrior(sd=5.0, period_ref=None)   # N(0, sd² I); with
#   period_ref=P0 the sd scales as (P / P0)^(2/3), §2.3
term = positions.marginal_term(elements, prior)      # elements(values) -> (period, dt_peri, ecc)
result = fit(lambda **kw: None, priors, (), likelihoods=[term])  # 3-D; L-BFGS
mean, cov = term.posterior(values)                   # (M2): ψ | φ, shape (4,), (4, 4)
orbits = term.draw(samples, key, n_per=1)            # φ samples -> ThieleInnesOrbit draws
                                                     #   -> .to_kepler(), both nodes
log_w, k_hat = reweight(orbits, prior, target_log_prior)  # §2.4(b), PSIS
best = starting_orbits(positions, periods, prior=prior)   # χ² ranking by default (D3)
```
- **One solver.** Refactor `_thiele_innes_fit`'s whitened design into a private `_thiele_innes_system(dt, whitener, period, dt_peri, ecc) -> (D̃, d̃)`. The existing least-squares route uses it, so `starting_orbits` is unchanged when `prior=None`. So does a new `_thiele_innes_marginal(D̃, d̃, mu, sd) -> (ln Z, ρ, log_norm, mean, cov)` (QR of [K; I₄], §1.2).
- **`PositionData` methods,** named like `RVData`'s in #192: `marginal_loglike(period, dt_peri, ecc, prior)`, `marginal_whitened_residuals`, `marginal_log_norm`, `thiele_innes_posterior`.
- **`marginal_term`** returns a `_Term`-like module with `__call__` (ρ), `has_log_norm`/`log_norm`, `loglike` (for `numpyro_model`) and `posterior`. No change to `fitting.py` or `likelihood.py` beyond what #192 adds.
- **`reweight`** returns normalised log weights, the effective sample size and k̂. PSIS uses SciPy's generalised Pareto fit, which adds no dependency.
- **The import rule holds:** `orbits` still imports only `_utils` (and jaxoplanet lazily).

### 5.2 Tests
Fast, in `tests/test_orbits.py` (float64 in a local `jax.enable_x64`, plus a float32 case):
1. `marginal_loglike` equals the dense 2n-dimensional Gaussian (SciPy), with correlated per-epoch covariances, at random φ (to 1e-10).
2. `thiele_innes_posterior` equals dense Gaussian conditioning.
3. ρᵀρ + 2 `log_norm` + constants = −2 ln Z exactly. L-BFGS on the term reaches the 3-D MAP found by a dense grid.
4. At e = 0, ln Z is independent of t_peri. It is unchanged by any i, since D does not depend on i.
5. A degenerate design (all epochs at one phase): ln Z and its gradient are finite with a proper prior, and a flat prior is rejected.
6. float32 agrees with float64 for a well-conditioned case. The QR route stays finite where the Cholesky route fails (cond M̃ ~ 1e8).
7. Prior-only reweighting (no data) reproduces the target prior: uniform cos i and log-uniform a, as in [S6]. k̂ is large for a target at face-on.
   - **No-data prior test** (`imaging_plan.md`, standing design decision 9; Hogg+2010 §4): sampling with an empty dataset, through the marginal, the conditional draws and the reweighting, reproduces the D2 default prior, Jacobians included: log-uniform P and μ, isotropic orientation, uniform phase and uniform e.
8. `starting_orbits(prior=None)` is exactly the current χ² ranking (backward compatible), and stays the default (D3). `starting_orbits(prior=...)` keeps the true grid point among the best (O §5.2.8), and its ln Z per grid point equals the dense Gaussian evidence. A wide prior does **not** recover the χ² ranking: the Occam factor tends to −½ ln det(D̃ᵀD̃) plus a constant, which still depends on φ.

Independent, in [virgil-validation](https://github.com/benjaminpope/virgil-validation) (the independence rule of `AGENTS.md`): the brute-force 4-D quadrature of [S4], and the Jacobian and implied prior of [S1] and [S3], written from scratch there. An Issue requesting this will be opened on that repository when the code PR lands.

Slow, on **OzSTAR** (CPU and GPU; never on a laptop), on simulated systems from `simulate`:
1. **Exact equivalence.** A well-covered orbit (f_orb ≈ 0.8), with the same Gaussian ψ prior in both: 3-D NUTS plus (M2) draws against 7-D NUTS in `ThieleInnesOrbit` coordinates. All seven marginals must agree within Monte Carlo error (quantiles, and a two-sample KS test per parameter).
2. **Short arcs** (f_orb ≈ 0.1–0.3), multimodal in P. A dense 3-D grid, 3-D NUTS and 7-D NUTS (with the same prior) are compared for mode weights and quantiles.
3. **Reweighting.** Case 2, reweighted to isotropic i and log-uniform a, against 7-D NUTS in `KeplerOrbit` coordinates with those priors. Report k̂ and the effective sample size.
4. **Edge cases:** near face-on (i = 5°, where (b) is expected to fail and fall back), edge-on, e = 0.01 and e = 0.95.
5. **Performance:** effective samples per second and per gradient, for 3-D against 7-D NUTS, on an A100 and on CPU.
6. **Calibration**, in the manner of Lucy (2014, §5.5): coverage of the 1σ and 2σ intervals of log M_tot over about 300 noise realisations of case 2. Lucy's profile likelihood was too narrow; the marginal should not be.

### 5.3 Effort (agent hours)

| Item | Effort | Depends on |
|---|---|---|
| `_thiele_innes_system`/`_thiele_innes_marginal`, `PositionData.marginal_*`, `marginal_term`, `ThieleInnesPrior`; fast tests 1–6 | 3–4 | #192 (the `log_norm` term) |
| `draw`, `reweight` (PSIS), `starting_orbits(prior=)`; fast tests 7–8 | 2–3 | the above |
| OzSTAR comparisons 1–6 | 3–4 | the above; `simulate` |
| RV amplitude columns (§3.2 scheme 2), half-Gaussian node evidence | 2–3 | #192 merged |
| validation Issue and its independent checks (in virgil-validation) | 1–2 | — |

## 6. Decisions (Ben, 2026-10-05)
These follow the standing rule that priors are Jeffreys priors under the relevant group actions, unless there is strong information otherwise (`imaging_plan.md`, standing design decision 9).
1. **D1, the prior policy.** (b), importance reweighting of the conditional (A, B, F, G) draws to the invariant target prior (§2.4), with (c), full 7-D NUTS initialised from the reweighted draws, as the fallback when PSIS k̂ > 0.7. Option (a), accepting the Gaussian-implied prior, is dropped entirely: that prior is non-isotropic, p(cos i) ∝ (1 − cos² i)/(1 + cos² i)² (§2.2), and the Gaussian is only a computational proposal. Face-on systems are expected to need the fallback.
2. **D2, the default target prior** (§2.4):
   - log-uniform a and P, equivalently log-uniform P and μ;
   - isotropic orientation;
   - uniform phase;
   - uniform e (interim);
   - stated bounds, with a check that the posterior is proper.

   A parallax prior (Gaia) overrides as strong information. With the distance known, the scale prior goes on total mass (1/M).
3. **D3, `starting_orbits`** keeps χ² ranking by default (§3.3). Starting points make no inferential claim, and ln Z under the Gaussian proposal is not the posterior either. Ranking by posterior comes only once reweighting exists.
4. **D4, the phase parameterisation:** uniform e as the interim prior, sampled as (√e cos M_ref, √e sin M_ref), which is Jacobian-free for uniform e (§1.5). Population e priors are applied by reweighting.
5. **D5, joint RV fits:** the RV amplitude as a linear column (scheme 2 of §3.2) is deferred until a real system needs it. When built, K is a scale parameter, so a Gaussian on K is again only a proposal and needs reweighting.
6. **D6, the Gaussian width s** is a proposal setting, not a prior (§2.3). It is fixed, s₀(P/P₀)^{2/3}, or adapted per (P, e, T₀) for efficiency, with the importance weights using the same s. A hierarchical s, sampled as a parameter, is ruled out, because it would replace the invariant prior.

## References
- Blunt, S., et al. 2017, AJ 153, 229 (arXiv:1703.10653): Orbits for the Impatient (OFTI).
- Blunt, S., et al. 2020, AJ 159, 89 (arXiv:1910.01756): orbitize!.
- Brandt, T. D., et al. 2021, AJ 162, 186 (arXiv:2105.11671): orvara. It marginalises parallax, barycentre proper motion and RV zero points.
- Golub, G. & Pereyra, V. 2003, Inverse Problems 19, R1: separable nonlinear least squares, variable projection.
- Lucy, L. B. 2014, A&A 563, A126 (arXiv:1309.2868): Thiele–Innes reduction of the visual-binary posterior to 3-D (P, e, τ), with uniform priors on ψ.
- Lucy, L. B. 2018, A&A 618, A100 (arXiv:1710.07544): combined astrometric and spectroscopic orbits, with a grid in φ and Monte Carlo in ψ and (γ, K₁, K₂).
- Luger, R., Foreman-Mackey, D. & Hogg, D. W. 2017, RNAAS 1, 7 (arXiv:1710.11136): linear models for systematics and nuisances.
- Price-Whelan, A. M., et al. 2017, ApJ 837, 20 (arXiv:1610.07602): The Joker.
- Price-Whelan, A. M., et al. 2020, ApJ 895, 2 (arXiv:2002.00014): The Joker's corrected marginal likelihood, and the period-scaled prior on K.
- Vehtari, A., et al. 2024, JMLR 25, 72 (arXiv:1507.02646): Pareto smoothed importance sampling.
- Wright, J. T. & Howard, A. W. 2009, ApJS 182, 205 (arXiv:0904.3725): linear parameters in Keplerian fits, including Thiele–Innes astrometry and the joint RV–astrometry scheme with (C, H).
