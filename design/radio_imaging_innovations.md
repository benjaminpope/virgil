# Radio imaging in virgil: departures from MPoL, and their provenance

This note lists what virgil's interferometric imaging (`virgil.radio`, `virgil.radio_clean`, `virgil.fit`)
does beyond a standard regularized maximum-likelihood (RML) imager such as MPoL. For each item it gives what
it is, where it lives, its status and evidence, and the prior art it builds on. IM Lup (DSHARP; 42.4 M
visibilities, 26 spectral windows, 1028² × 5 mas) is the worked example only. The methods are general.

**Policy until 1.0.** virgil implements the state of the art and makes no unsupported innovations. Each item
B1–B8 therefore carries a **Status**:

- *established*: published practice, cited;
- *derived*: follows from published results by a short derivation, given with the papers it rests on;
- *novel / experimental*: no precedent found.

Each item also carries **Traceability**: the virgil PRs and tests (checked on `origin/main` on 2026-10-23), and
the virgil-validation (vv) check, or "none yet". Re-audited against `origin/main` of both repositories on
2026-10-23 (virgil#531); the D-ledger below records the findings.

**Status of citations.** Every reference below was checked on 2026-10-23, against NASA ADS (bibcode given) or,
for non-ADS items, Crossref/DOI, the publisher or arXiv (DOI given). Two items could not be verified and are
marked *(unverified)*. Corrections are listed at the end.

## A. Baseline: what is shared with MPoL (replication, not innovation)

- **The objective.** −ln L of the visibilities (Gaussian, per-point weights), with per-spectral-window σ
  scales, per-observation amplitude gains and phase-centre offsets, a Gaussian uv taper (30 mas), and a
  maximum-entropy regularizer at MPoL's λ.
  - Our implementation matches MPoL's own value at MPoL's solution to 14 in 1.08e8 (1.3e-7).
  - Started from MPoL's solution, our optimizer goes on to beat it by 72,099 in the objective. MPoL's
    nuisance gradients there are 10³–10⁶ times ours, so MPoL had not converged in flux, gains and offsets.
- **Prior art:**
  - MPoL: Zawadzki et al. 2023, PASP 135, 064503 (`2023PASP..135f4503Z`); software paper Czekala et al. 2025,
    arXiv:2502.00100 (`2025arXiv250200100C`).
  - RML imaging generally: Narayan & Nityananda 1986, ARA&A 24, 127 (`1986ARA&A..24..127N`).
  - EHT RML imagers: eht-imaging, Chael et al. 2016, ApJ 829, 11 (`2016ApJ...829...11C`) and 2018, ApJ 857, 23
    (`2018ApJ...857...23C`); SMILI, Akiyama et al. 2017, ApJ 838, 1 (`2017ApJ...838....1A`) and AJ 153, 159
    (`2017AJ....153..159A`).

## B. Departures

### B1. Full-batch second-order-ish optimization in float64 (vs minibatch Adam in float32)
- **What:**
  - L-BFGS on all 42.4 M points per step, with an exact NUFFT (nufftax, eps 1e-12) in float64;
  - a convergence criterion (C8: loss change < 0.1 over ≥ 100 steps) instead of a fixed epoch count;
  - L-BFGS memory 100, capped at a 1 GiB history.
- **Why:** the loss is about 1e8 while the differences that matter are of order 1, so float32 cannot resolve
  them. MPoL's stochastic AdamW stops short of the optimum (above).
- **Where:** virgil#426 (nufftax backend), #510 (memory); virgil-validation scripts/alma_imlup_maxent.py.
- **Status:** established. L-BFGS and a FINUFFT-type NUFFT, applied at full-batch scale.
- **Traceability:**
  - PRs #426, #508, #510 (merged). Tests: `tests/test_nufft.py::test_nufft_matches_the_direct_sum`,
    `::test_gradient_matches_the_dft_gradient`; `tests/test_fitting.py::test_lm_and_lbfgs_agree_on_a_tsv_image`,
    `::test_grad_norm_is_small_at_convergence_and_large_when_stopped`,
    `::test_the_lbfgs_history_is_100_unless_it_would_pass_one_gibibyte`,
    `::test_the_lbfgs_history_is_sized_by_the_widest_leaf`.
  - vv: `scripts/alma_imlup_maxent.py`, `tests/test_alma_maxent.py` (R1: the objective equals MPoL's),
    `tests/test_alma_mpol_ref.py`; comparison on the shared objective, vv#176/#177.
- **Prior art:**
  - L-BFGS: Liu & Nocedal 1989, Math. Prog. 45, 503, doi:10.1007/BF01589116; bound-constrained L-BFGS-B, Byrd,
    Lu, Nocedal & Zhu 1995, SIAM J. Sci. Comput. 16, 1190 (`1995SJSC...16.1190B`).
  - NUFFT: the exponential-of-semicircle kernel of FINUFFT (Barnett, Magland & af Klinteberg 2019, SIAM J.
    Sci. Comput. 41, C479, `2019SJSC...41C.479B`); nufftax is a JAX port of it.
  - eht-imaging also uses L-BFGS(-B), through SciPy, on far smaller data sets (Chael et al. 2016, 2018; checked
    in the papers' text).

### B2. Diagonal Gauss–Newton preconditioning with an entropy metric
- **What:** `precondition="diag"` rescales L-BFGS coordinates by an estimate of the Gauss–Newton diagonal.
  - The data part is estimated stochastically with Rademacher probes (exact by JVP below 1024 coordinates).
  - On image pixels, the data part is replaced by the entropy's own metric (`MaxEntropy.metric_rows`): w·b in
    ln b, which is w·b(1−b) in the Image's softmax logits, the coordinates the fit uses (D13).
  - `prefers_lbfgs` gates which terms use it.
- **Where:** virgil#474, #505.
- **Status:** derived (the parts are established).
  - Rests on: the Gauss–Newton curvature of a least-squares term (Martens 2010; Schraudolph 2002), its diagonal by
    Hutchinson-type probes (Bekas et al. 2007), and the entropy metric of MEM (Skilling & Bryan 1984).
  - Derivation, in one line: Skilling & Bryan's metric is Σ db²/b in image space, which is Σ b dη² in η = ln b;
    with weight w it is w·b in ln b. The Image's coordinates are the logits η of a softmax over the support, so the
    metric is pulled back through db_i/dη_j = b_i(δ_ij − b_j) and its diagonal is w·b(1−b) (D13). On pixels where the entropy dominates the curvature, this replaces the
    probe estimate of the data part.
- **Traceability:**
  - PRs #474, #486, #505 (merged). Tests, all in `tests/test_fit_precondition.py`:
    `test_the_diagonal_is_that_of_the_gauss_newton_matrix`,
    `test_the_regularizer_curvature_is_included_and_estimated_by_probes`,
    `test_max_entropy_curvature_is_the_weight_times_the_image`,
    `test_the_entropy_metric_replaces_the_data_curvature_on_its_pixels`,
    `test_a_known_diagonal_metric_is_removed`, `test_the_optimum_and_the_gradient_norm_are_unchanged`.
  - vv checks (`tests/test_radio_innovations.py` on vv `main`, mathematics root): the diagonal equals the dense
    diagonal of JᵀJ; the probe estimate is within its stochastic error of the exact one; with `MaxEntropy(w)` the
    diagonal is w·b(1−b), the Skilling–Bryan metric pulled back through the softmax, to 1e-9 (the vv test also
    shows it differs from w·b by up to 20% on a 6×6 image). Plus `--precondition {none,diag}` in
    `scripts/alma_imlup_maxent.py` (a switch and per-round stats). The vv evidence file `latest.jsonl` (0.4.1 run
    of 2026-10-23) has no record for these tests, as they skip on the pinned virgil: they are not yet pinned
    evidence.
- **Prior art:**
  - Hutchinson-type diagonal estimation: Bekas, Kokiopoulou & Saad 2007, Appl. Numer. Math. 57, 1214
    (`2007ApNM...57.1214B`); Rademacher probes, Hutchinson 1989, Commun. Stat. Simul. Comput. 18, 1059,
    doi:10.1080/03610918908812806 (`1989CoSSC..18.1059H`; ADS also lists a 1990 record of the same title,
    vol. 19, 433, `1990CoSSC..19..433H`). We measured about 35% error with Gaussian probes on our nearly diagonal
    metric, which is why we use Rademacher.
  - The entropy metric as the natural preconditioner in MEM: Skilling & Bryan 1984, MNRAS 211, 111
    (`1984MNRAS.211..111S`); Cornwell & Evans 1985, A&A 143, 77 (`1985A&A...143...77C`).
  - Gauss–Newton curvature in Hessian-free optimization: Martens 2010, ICML 27, 735–742 (no DOI, not on ADS;
    title and author checked in the ICML 2010 PDF; the paper uses the Gauss–Newton matrix as its curvature matrix); Schraudolph 2002, Neural Comput. 14, 1723,
    doi:10.1162/08997660260028683.

### B3. Nuisance parameterization for conditioning and gauge
- **What:**
  - Image = flux × softmax(log-brightness), so total flux is a separate parameter.
  - Gains are fixed at 1 for a reference observation (gauge), with Jeffreys (1/g) priors on the others.
  - Offsets are expressed in units of each observation's beam and bounded at ±3 beams, a real prior that is
    reported when it is hit (virgil#473).
  - σ per spectral window is profiled, ŝ = √(χ²/2N), with stop-gradient (virgil#474).
  - `--ref-obs auto` (the most heavily weighted observation) is in progress, as are 100 mas offset bounds for
    cold starts (vv#178, ozstar_scripts#64).
- **Diagnosis that motivated it:** the old cold start ended with every free gain pinned at the old prior's
  limit of 2.0 and one offset 6.8″ away, outside the field.
- **Status:** established (each part), combined jointly with the image.
  - DSHARP aligns astrometric offsets (CASA `fixvis`) and rescales flux between observations as a separate
    pre-step; virgil fits them jointly. GR (GRAVITY imaging) fixes the position of one point source as an anchor
    for the same translation gauge.
- **Traceability:**
  - PRs #441, #473, #474 (merged); `--ref-obs auto` is in progress. Tests in `tests/test_radio.py`:
    `test_fit_recovers_nuisances`, `test_gradients_match_finite_differences`, `test_offset_ramp_is_a_shift_of_the_scene`,
    `test_sigma_scale_recovers_known_scales`, `test_sigma_scale_detects_scrambled_time`;
    `tests/test_fit_precondition.py::test_a_proper_pixel_prior_is_not_given_the_gauge_projection` (softmax gauge).
  - vv check: none yet specific to these (vv#178 in progress).
- **Prior art:**
  - Jeffreys priors: Jeffreys 1946, Proc. R. Soc. A 186, 453, doi:10.1098/rspa.1946.0056.
  - Profile likelihood of a noise scale: standard, e.g. Pawitan 2001, *In All Likelihood*, OUP,
    doi:10.1093/oso/9780198507659.001.0001.
  - Reference-antenna gauge in self-calibration: Cornwell & Wilkinson 1981, MNRAS 196, 1067
    (`1981MNRAS.196.1067C`).
  - ALMA weight rescaling: the CASA `statwt` task and the CASA "Data Weights" documentation (pages checked
    2026-10-23). A specific ALMA memo on weight miscalibration could not be identified *(unverified)*.
  - Offsets and flux-scale mismatches between ALMA observations: Andrews et al. 2018, ApJ 869, L41, Sect. 4.4
    (`2018ApJ...869L..41A`); exoALMA II, Loomis et al. 2025, ApJ 984, L7 (`2025ApJ...984L...7L`).

### B4. Cold-start strategies (in progress, vv#178 / ozstar_scripts#64)
- **CLEAN start.** The MaxEnt fit starts from our own CLEAN components (virgil.radio_clean), not beam-smoothed,
  because smoothing biases the long-baseline gains. An image-fixed stage fits gains and offsets first
  (self-calibration), then everything is released.
- **Image-first staging.** The reverse order, for flat or Gaussian starts. L-BFGS's early steps otherwise
  throw the nuisances around; Adam cannot, because its learning rate caps each step.
- **Coarse-to-fine** (planned). Each level cuts the uv data to its grid's Nyquist radius.
- **Status:**
  - CLEAN start and self-calibration staging: established (hybrid mapping; DSHARP self-calibrates on CLEAN
    models).
  - Image-first staging: established in spirit (eht-imaging re-starts from a blurred previous image; Chael et al.
    2018); our specific L-BFGS argument is an observation, not a result from the literature.
  - Coarse-to-fine uv truncation: planned, not built. I did not search the multiresolution-imaging literature
    for it beyond DoG-HiT and multiscale CLEAN, which are different schemes. Treat as novel / experimental
    until checked.
- **Traceability:**
  - PR #472 (radio_clean, merged). Tests in `tests/test_radio_clean.py`: `test_point_source_is_recovered`,
    `test_two_sources_with_a_negative_residual_sidelobe_cleaned`, `test_multiscale_recovers_an_extended_source`,
    `test_auto_mask_finds_the_source_and_nsigma_stops`. The staging itself has no test yet.
  - vv: `tests/test_alma_clean.py` and `scripts/alma_imlup_clean.py` (CLEAN against CASA, IM Lup); staging check
    none yet.
- **Prior art:**
  - CLEAN: Högbom 1974, A&AS 15, 417 (`1974A&AS...15..417H`); Clark 1980, A&A 89, 377
    (`1980A&A....89..377C`); Cotton–Schwab, Schwab 1984, AJ 89, 1076 (`1984AJ.....89.1076S`); multiscale CLEAN,
    Cornwell 2008, IEEE J. Sel. Top. Signal Process. 2, 793 (`2008ISTSP...2..793C`).
  - Hybrid mapping / self-calibration loops: Readhead & Wilkinson 1978, ApJ 223, 25 (`1978ApJ...223...25R`);
    Pearson & Readhead 1984, ARA&A 22, 97 (`1984ARA&A..22...97P`).
  - Staged RML imaging with blurred re-starts: Chael et al. 2018 (`2018ApJ...857...23C`); EHT M87* Paper IV,
    2019, ApJL 875, L4 (`2019ApJ...875L...4E`).
  - Multiscale in VLBI: DoG-HiT, Müller & Lobanov 2022, A&A 666, A137 (`2022A&A...666A.137M`); DoB-CLEAN,
    Müller & Lobanov 2023, A&A 672, A26 (`2023A&A...672A..26M`).

### B5. Exact Toeplitz (dirty-beam) likelihood with free offsets (virgil#516, merged)
- **What:** χ² = Σ_k [g_k² IᵀB_k I − 2 g_k IᵀD_k(Δ_k)] + C, where k runs over (observation, spectral window),
  g_k is the observation's gain, B_k the dirty beam (a convolution done with 2×-padded FFTs) and D_k the dirty
  image. The σ profiling needs χ² for each k too. All B_k and D_k are precomputed once, so each evaluation is
  independent of the number of visibilities but grows with the number of groups k: an estimated 5–10 ms against
  0.45–0.5 s, still to be measured.
- **Free per-observation offsets, exactly.**
  - The beam term does not depend on the offsets, because the phase ramp cancels in |M|².
  - The cross term needs D at shifted positions. These are evaluated from the stored NUFFT spread grid, with
    a fractional phase ramp and ψ̂ at non-integer positions by quadrature. Poisson summation holds for any
    real position, so this is exact to the NUFFT's eps: prototype errors 4e-14 up to 20 pixels, gradients
    5e-14.
  - Condition: every baseline must lie within the grid's Nyquist limit minus the kernel margin.
  - The usual alternatives fall far short: a Fourier shift of the padded dirty image is good to 3e-4, a
    windowed sinc to 2e-6–3e-9.
- **Automatic choice (virgil#532):** `ComplexVisData.term(chi2="auto")`; `_RadioTerm.resolve`, called once in `fit`,
  swaps in the Toeplitz term when it is exact (one fixed `Image`, `offsets=False`, fixed `alpha`, no antenna gains,
  float64, not LM) and the problem is big (`set_ft_defaults`: 50 000 points, 1 per pixel and ddid, 4 GB of beams). Free offsets stay explicit.
  numpyro, the fits of `cross_validate` and svi stay direct; Toeplitz cross-validation (a precompute per fold) is a possible
  follow-up.
- **Also planned:** a fitted spectral index through Taylor-moment beams, and exact Hessian-vector products
  for truncated Newton-CG.
- **Where:** virgil#516 (merged): `virgil.radio_toeplitz` and `virgil._nufft_spread`. The design
  note `design/toeplitz_likelihood.md` shipped with it.
- **Status:**
  - The Toeplitz quadratic and cross terms: established (Fessler et al. 2005).
  - Free offsets (`offsets="spread"`): opt-in, never the default, and derived. They are a type-3 evaluation
    (Lee & Greengard 2005, `2005JCoPh.206....1L`; Barnett et al. 2019, Sect. 3.3; see "Prior-art search"
    below); we found no published use of the combination.
  - Derivation, in two lines. With spread grid ĝ_l, the dirty image at pixel x is D(x) = Σ_l ĝ_l e^{ilx}/ψ̂(x)
    (type 1). At x + Δ the exponential gains a factor e^{ilΔ}, a phase ramp on the stored grid, and the
    correction becomes 1/ψ̂(x + Δ), the kernel transform at a non-integer position by Gauss–Legendre quadrature.
    Poisson summation (Barnett et al. 2019, Sect. 3) makes this exact to the kernel's aliasing error for any
    real Δ. The model visibility is M_j = e^{−2πi u_j·Δ} F(I)(u_j), so |M_j|² does not depend on Δ and the
    beam term is offset-free.
- **Traceability:**
  - virgil#516 (merged): `virgil.radio_toeplitz`, `virgil._nufft_spread`. Tests: `tests/test_radio_toeplitz.py`
    (13 fast, 9 slow; the "Tests" table of the PR lists each guarantee), including
    `test_matches_direct_term` (loss differences and gradients against `ComplexVisData.term`, ≤ 1e-10, with and
    without offsets), `test_matches_direct_term_at_scale`, `test_spread_grid_against_nufftax_and_direct_sums`,
    `test_fit_reaches_the_direct_optimum`. Building blocks on main (PR #441, merged), `tests/test_radio.py`:
    `test_dirty_image_matches_brute_force`, `test_dirty_beam_peak_and_briggs_limits`,
    `test_dirty_residual_of_true_model_is_zero`.
  - vv checks on vv `main`: `tests/test_radio_innovations.py` (free-offset χ² equals an independent direct DFT
    χ² up to the documented bounds, with a control that drops the offsets and a guard that refuses baselines
    past the margin; mathematics root) and `tests/test_alma_toeplitz_check.py` (the check script on a small
    synthetic set). The IM Lup A100 run (`scripts/alma_imlup_toeplitz_check.py`: Toeplitz vs direct at the
    R_mpol solution, timing) is not yet in the evidence; the 5–10 ms estimate is unmeasured and large-data
    agreement is not demonstrated. As for B2, the evidence file has no record of these tests.
- **Prior art:**
  - Toeplitz embedding for NUFFT normal operators: Wajer & Pruessmann 2001, Proc. ISMRM 9, 767 *(unverified:
    the abstract could not be found online; the citation is from memory)*; Fessler, Lee, Olafsson, Shi, Noll
    2005, IEEE Trans. Signal Process. 53, 3393 (`2005ITSP...53.3393F`, doi:10.1109/TSP.2005.853152).
  - The dirty-beam convolution is the CLEAN minor cycle (Högbom 1974; Clark 1980).
  - Radio uses of the normal operator or dirty-image fits: UVMULTIFIT `immultifit` (fits a model convolved with the
    PSF to the dirty image; Martí-Vidal et al. 2014, A&A 563, A136, `2014A&A...563A.136M`); the Wiaux group
    (PURIFY / SARA; Carrillo et al. 2012, MNRAS 426, 1223, `2012MNRAS.426.1223C`; Pratley et al. 2018, MNRAS
    473, 1038, `2018MNRAS.473.1038P`); resolve/NIFTy (Arras et al. 2021, A&A 646, A84, `2021A&A...646A..84A`;
    it applies the full response, not a Toeplitz normal operator).
  - Truncated Newton: Nash 2000, J. Comput. Appl. Math. 124, 45, doi:10.1016/S0377-0427(00)00426-X.
  - Multi-term multi-frequency synthesis (Taylor-moment beams): Rau & Cornwell 2011, A&A 532, A71
    (`2011A&A...532A..71R`).

#### Prior-art search for the exact free offsets (2026-10-23)

- **Searched:** ADS (title and abstract queries on Toeplitz, NUFFT, off-grid, shift, type 3), the arXiv API
  (partly rate-limited), and web searches on MRI motion correction with Toeplitz operators, shifted or
  non-integer NUFFT outputs, and radio phase-centre fitting.
- **Mathematical precedent: yes.** Evaluating a type-1 transform at output positions off the integer lattice is
  a type-3 NUFFT (nonuniform to nonuniform).
  - Lee & Greengard 2005, J. Comput. Phys. 206, 1 (`2005JCoPh.206....1L`) introduced it.
  - Barnett et al. 2019, Sect. 3.3 (`2019SJSC...41C.479B`): Step 3 divides by ψ̂ evaluated at non-uniform
    (non-integer) frequencies, by quadrature of the exponential-of-semicircle kernel, because "there is no phase
    winding trick" there. Their Remark on translation shows that translated data are handled by pre-phasing.
  - Standard type 1: Dutt & Rokhlin 1993, SIAM J. Sci. Comput. 14, 1368, doi:10.1137/0914081; Greengard & Lee
    2004, SIAM Rev. 46, 443, doi:10.1137/S003614450343200X; Fessler & Sutton 2003, IEEE Trans. Signal Process.
    51, 560, doi:10.1109/TSP.2002.807005. They all correct at integer outputs only, as far as I read.
- **Application precedent: not found.** I found no paper that (1) keeps the type-1 spread grid per observation,
  (2) evaluates the dirty image at fractional per-observation offsets by a phase ramp on that grid with ψ̂ at
  shifted positions, (3) pairs it with an offset-independent beam term in a Toeplitz likelihood, and (4)
  differentiates through the offsets. The Toeplitz MRI papers I found (Fessler 2005; open-source packages)
  treat off-resonance or fixed trajectories, not per-shot fractional shifts. Radio practice shifts the phase
  centre by rephasing the visibilities (CASA `fixvis`, as DSHARP did), which costs O(N_vis) per offset.
- **Verdict.** The mathematics has precedent (type-3 NUFFT); the combination is, as far as this search goes,
  new. It is a derived reuse of published results, not a new transform. The note should claim only that. The
  search was not exhaustive (no access to the ISMRM proceedings, no full-text search of the MRI motion
  literature), so "we know of no prior use" is the strongest honest wording.

### B6. Float32 MaxEnt via a linear-δV delta loss (experiment)
- **What:** in float32, compute the residual about a float64 reference, δV = F(I_ref·expm1(z − z_ref)), to
  avoid catastrophic cancellation.
- **Status:** derived; experimental. It helps on a 32² AMI test case down to λ = 1, but still stalls at the two
  weakest λ. Not yet used for ALMA.
  - Derivation: I = I_ref·e^{z−z_ref} = I_ref(1 + expm1(z−z_ref)), and F is linear, so V = V_ref + F(I_ref·expm1(·)).
    V_ref is computed once in float64 and the low-precision part carries only the small difference. This is
    residual-based mixed-precision iterative refinement.
- **Traceability:** the δV loss itself is not on `origin/main`, so no test. Related: PR #441 and
  `tests/test_radio.py::test_model_and_term_in_float32`; commit 80d71d1 (float32 against float64 image
  visibilities on long baselines). vv check: `tests/test_radio_innovations.py`
  (`crosscheck.radio_innovations.DeltaProblem`, the method alone: the float32 δV loss matches the float64 χ²
  difference and is exact in float64); it exercises vv's own code, since virgil has none.
- **Prior art:** mixed-precision iterative refinement: Wilkinson 1963, *Rounding Errors in Algebraic Processes*
  (Prentice-Hall; SIAM reissue doi:10.1137/1.9781611977523); Higham 2002, *Accuracy and Stability of Numerical
  Algorithms*, 2nd ed., SIAM, doi:10.1137/1.9780898718027.

### B7. Marginalized antenna gains (`design/antenna_gains.md`; Stage 1 merged, virgil#513)
- **What:** complex antenna gains per solution interval, marginalized with a Laplace approximation about a
  per-interval Gauss–Newton solution, with the log-determinant in the normalization. Stage 2 makes them
  correlated in time with a state-space Gaussian process (Kalman/RTS).
- **Status:** established for Stage 1 (built, opt-in: `CalibrationNuisance(antenna_gains=True)`, default
  `False`); derived for Stage 2 (a plan, not built).
  - Stage 1 is the Laplace gain marginalization of Themis (Broderick et al. 2020, Sect. 6.8) applied to ALMA
    complex visibilities.
  - Stage 2 combines gains as Gaussian processes in time (Arras et al. 2019, A&A 627, A134) with the
    state-space form of a temporal GP (Hartikainen & Särkkä 2010). The derivation is the standard Kalman/RTS
    smoother for a Markov prior.
- **Traceability:**
  - Stage 1: PR #513 (merged), `src/virgil/_antenna_gains.py`. Tests in `tests/test_radio_gains.py`:
    `test_calibrated_limit`, `test_closure_invariance`, `test_snapshot_position_curvature_vanishes`,
    `test_solved_gains_match_injected`, `test_recovery_with_large_phase_errors`,
    `test_gradient_matches_finite_differences`, `test_lm_is_refused`, `test_intervals_never_cross_scans`.
    The VLTI/AMI marginal-gain code is a different surface: PR #453,
    `tests/test_marginal_gains.py::test_two_blocks_with_gains_and_offsets_match_dense_integrals`,
    `::test_quadrature_matches_a_dense_integral_and_cholesky_logdet`, `tests/test_gains.py`.
  - vv check: none for the radio gains (the vv graph has no antenna-gain node); the gain widths 0.1 and 0.5 rad
    are choices, not validated values (`design/antenna_gains.md`).
- **Prior art:**
  - Themis's Laplace gain marginalization: Broderick et al. 2020, ApJ 897, 139, Sect. 6.8
    (`2020ApJ...897..139B`; see corrections: this was cited as ApJ 898, 9). The hybrid imaging paper is
    Broderick et al. 2020, ApJ 898, 9 (`2020ApJ...898....9B`).
  - Comrade.jl's gain modelling: Tiede 2022, JOSS 7, 4457 (`2022JOSS....7.4457T`).
  - Closure-only imaging as the alternative: Chael et al. 2018 (`2018ApJ...857...23C`); Blackburn et al. 2020,
    ApJ 894, 31 (`2020ApJ...894...31B`).
  - The state-space GP: Hartikainen & Särkkä 2010, IEEE MLSP, pp. 379–384, doi:10.1109/MLSP.2010.5589113.

### B8. Ordered-subsets / minibatch acceleration (other session)
- **What:** cycle through subsets of the data per step, as in OS-EM.
- **Status:** established; not built. The scoping note `design/ordered_subsets.md` (not yet on `main`) is "scoping only, nothing built" (2026-10-23), and
  proposes it only for joint gain and image fitting.
- **Traceability:** no PR, no test. vv check: none yet.
- **Prior art:** Hudson & Larkin 1994, IEEE Trans. Med. Imaging 13, 601 (`1994ITMI...13..601H`); SVRG, Johnson &
  Zhang 2013, NeurIPS 26, 315–323 (no DOI; papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).

### B8a. Per-step projection hook for Adam (`fit(post_step=)`)
- **What:** a jax-traceable map of the free parameters applied after every Adam update, so the next gradient sees
  the projected parameters; the caller supplies the map (e.g. a subpixel blur of an image's log-brightness).
  Adam moments are left as they are; not available for L-BFGS, whose curvature pairs a projection breaks.
- **Status:** established. Post-step smoothing of the image parameters is the published recipe of dorito
  (Charles et al. 2026, PASA 43, 48, `2026PASA...43...48C`: Gaussian σ ≈ 0.25 pix on the linear image, then
  renormalised, each epoch).
- **Traceability:** `tests/test_fitting.py::test_post_step_none_and_identity_are_bit_identical`,
  `::test_post_step_clip_holds_at_every_step`, `::test_post_step_needs_plain_adam`. vv check: none yet.

### B9. Validation discipline
- Every comparison leads with raw χ²/N.
- The comparison with MPoL is made on the shared objective, the same data (26 of 26 spectral windows
  matched), and identical cross-validation folds (vv#176).
- Speed is measured as time to reach MPoL's final objective (vv#177).
- Independent checks live in virgil-validation.

### Summary table

| Item | Status | Key references | virgil tests | vv check |
|---|---|---|---|---|
| B1 full-batch L-BFGS, float64 | established | Liu & Nocedal 1989; Barnett+ 2019 | `test_nufft.py`, `test_fitting.py` (L-BFGS memory, convergence) | `alma_imlup_maxent.py`, `test_alma_maxent.py`, `test_alma_mpol_ref.py` |
| B2 diagonal GN + entropy metric | derived | Bekas+ 2007; Skilling & Bryan 1984; Martens 2010 | `test_fit_precondition.py` | `test_radio_innovations.py` (dense JᵀJ, probes, entropy metric w·b(1−b)); not yet in pinned evidence |
| B3 nuisance parameterization | established | Cornwell & Wilkinson 1981; Jeffreys 1946; DSHARP Sect. 4.4 | `test_radio.py` (nuisances, σ scales, offsets) | none yet (vv#178) |
| B4 cold starts | CLEAN start and staging established; coarse-to-fine novel / experimental | Readhead & Wilkinson 1978; Chael+ 2018; Cornwell 2008 | `test_radio_clean.py`; staging none | `test_alma_clean.py`; staging none |
| B5 Toeplitz likelihood, free offsets | Toeplitz terms established; free offsets (`offsets="spread"`, opt-in) derived (type-3 NUFFT; combination not found) | Fessler+ 2005; Lee & Greengard 2005; Barnett+ 2019 | `test_radio_toeplitz.py` (fast and slow; virgil#516 merged) | `test_radio_innovations.py`, `test_alma_toeplitz_check.py`; IM Lup A100 run not yet in evidence |
| B6 float32 δV loss | derived; experimental | Wilkinson 1963; Higham 2002 | none on main (no code) | `test_radio_innovations.py` (the method alone) |
| B7 marginalized gains | established (Stage 1, built, opt-in), derived (Stage 2, plan) | Broderick+ 2020; Arras+ 2019; Hartikainen & Särkkä 2010 | `test_radio_gains.py` (Stage 1) | none yet |
| B8 ordered subsets | established; not built | Hudson & Larkin 1994; Johnson & Zhang 2013 | none | none yet |

### Provenance gaps

Closed on 2026-10-23 (virgil#531): Zawadzki et al. 2023, Cornwell & Wilkinson 1981, Cornwell & Evans 1985,
Martens 2010, Schraudolph 2002 and Pawitan 2001 are in `references/provenance.toml`, with `Source:` lines in
`virgil.radio` and `virgil.fitting.gauss_newton_diag`; Barnett et al. 2019, Hutchinson 1989 and Bekas et al. 2007
were already there. Cornwell & Evans 1985 has no DOI or arXiv ID and is `verified` on its ADS record, as Skilling & Bryan is;
Martens 2010 is not on ADS, has no DOI, and stays `verified = false` (title and author read in the ICML PDF).
The other references of B1–B8 that no code cites (for example Chael 2016, Hudson & Larkin 1994, Readhead &
Wilkinson 1978) are in the prose only and need entries when code cites them.

## C. Comparison with the state of the art

Literature checked through ADS to October 2026. Claims about a paper's content come from its abstract or, where
marked, its full text. Nothing here is a ranking: the fields differ in data, array and goal.

### C1. EHT and VLBI imaging

- **eht-imaging** (Chael et al. 2016, `2016ApJ...829...11C`; 2018, `2018ApJ...857...23C`).
  - RML with maximum entropy, total variation, L1 and flux/centroid terms; L-BFGS(-B) through SciPy.
  - Polarimetric imaging (2016) and imaging directly with closure phases and closure amplitudes (2018), so no
    self-calibration. Re-starts the imager from a blurred previous image.
- **SMILI** (Akiyama et al. 2017, `2017ApJ...838....1A`, `2017AJ....153..159A`): sparse modelling, L1 + TSV,
  full polarimetry. Related: Honma et al. 2014 (`2014PASJ...66...95H`); Wiaux et al. 2009
  (`2009MNRAS.395.1733W`).
- **EHT pipelines.**
  - M87* Paper IV (2019, `2019ApJ...875L...4E`): two stages. Four blind teams (CLEAN and RML), then a survey of
    imaging parameters on synthetic data against ground truth, which selects the parameters used on M87*.
  - Sgr A* Paper III (2022, `2022ApJ...930L..14E`): four scripted pipelines (DIFMAP for CLEAN; eht-imaging and
    SMILI for RML; Themis for Bayesian posterior sampling), parameter surveys on synthetic data ("Top Sets"),
    several scattering mitigations, a variability noise budget, and clustering of the resulting morphologies.
    Read in the paper's text.
- **Bouman and colleagues.**
  - CHIRP (Bouman et al. 2016, CVPR, doi:10.1109/CVPR.2016.105, `2016cvpr.conf..913B`): Bayesian VLBI imaging
    with patch priors.
  - StarWarps (Bouman et al. 2018, IEEE Trans. Comput. Imaging 4, 512, doi:10.1109/TCI.2018.2838452): dynamic
    imaging, a Gaussian model in time and space solved by a Kalman-type elimination. Used in the Sgr A* dynamic
    analyses (Paper III text).
  - DPI and α-DPI (Sun & Bouman 2021, AAAI 35, 2628, doi:10.1609/aaai.v35i3.16366; Sun et al. 2022, ApJ 932, 99,
    `2022ApJ...932...99S`): normalizing-flow variational posteriors over images, with importance reweighting in
    α-DPI.
  - BH-NeRF (Levis et al. 2022, CVPR, arXiv:2204.03715, `2022arXiv220403715L`) and its polarimetric successor
    for an Sgr A* flare (Levis et al. 2024, Nature Astron. 8, 765, `2024NatAs...8..765L`): 3D emission
    tomography with lensing and Keplerian dynamics.
  - Score-based priors: Feng et al. 2023, ICCV (`2023iccv.conf..965F`); Feng & Bouman 2023, arXiv:2309.01949
    (`2023arXiv230901949F`; published venue not verified); Sun et al. 2024, IEEE TCI 10, 1290
    (`2024ITCI...10.1290S`); M87* under different priors, Feng et al. 2024, ApJ 975, 201
    (`2024ApJ...975..201F`).
- **Other EHT methods.**
  - Themis (Broderick et al. 2020, `2020ApJ...897..139B`) and hybrid imaging and modelling (Broderick et al.
    2020, `2020ApJ...898....9B`): full Bayesian posterior over images, Laplace-marginalized gains.
  - Comrade.jl (Tiede 2022, `2022JOSS....7.4457T`); hierarchical Bayesian imaging with MRF priors and joint
    calibration (HIBI, Tiede et al. 2026, ApJ 997, 262, `2026ApJ...997..262T`).
  - PRIMO (Medeiros et al. 2023, ApJ 943, 144, `2023ApJ...943..144M`; the M87* image, ApJL 947, L7,
    `2023ApJ...947L...7M`): principal components of a GRMHD image library fitted to the visibilities.
  - DoG-HiT (Müller & Lobanov 2022, `2022A&A...666A.137M`), DoB-CLEAN (2023, `2023A&A...672A..26M`) and
    dynamic and polarimetric extensions (2023, `2023A&A...673A.151M`): multiscale wavelet imaging, closure-only
    first, then self-calibration restricted to the multiresolution support.
  - resolve / NIFTy: unified calibration and imaging with uncertainty (Arras et al. 2019,
    `2019A&A...627A.134A`); Cygnus A against CLEAN (Arras et al. 2021, `2021A&A...646A..84A`); M87* in space,
    time and frequency (Arras et al. 2022, Nature Astron. 6, 259, `2022NatAs...6..259A`); direction-dependent
    calibration (Roth et al. 2023, `2023A&A...678A.177R`). Posterior by MGVI (Knollmüller & Enßlin 2019,
    arXiv:1901.11033, `2019arXiv190111033K`).
  - Chael's multifrequency imaging for the ngEHT (Chael et al. 2023, ApJ 945, 40, `2023ApJ...945...40C`): joint
    RML images and spectral-index maps, in eht-imaging. I did not find a published paper on "kine" on ADS, so I
    do not cite one.
  - ngEHT analysis challenges (Roelofs et al. 2023, Galaxies 11, 12, `2023Galax..11...12R`); science goals
    (Johnson et al. 2023, `2023Galax..11...61J`).
  - Deep learning from closure invariants (Lai et al. 2025, MNRAS 536, 446, `2025MNRAS.536..446L`).
  - Scattering mitigation framework: Johnson 2016 (`2016ApJ...833...74J`); in Paper III, division by a
    diffractive kernel plus a refractive noise budget.

### C2. ALMA protoplanetary-disk imaging

- **DSHARP** (Andrews et al. 2018, `2018ApJ...869L..41A`; read in the paper's text): `tclean` multiscale
  multi-frequency CLEAN, Briggs robust 0.5; iterative phase then amplitude self-calibration on CLEAN models;
  astrometric alignment with `fixvis` and flux-scale alignment of observations (mismatches of 15–25% in many
  datasets) before combining. This is the practice virgil's B3 and B4 replace with a joint fit.
- **frank** (Jennings et al. 2020, `2020MNRAS.495.3209J`): 1D radial brightness by a Gaussian process, fitted to
  the visibilities directly; applied to all DSHARP discs (Jennings et al. 2022, `2022MNRAS.509.2780J`) and the
  Taurus survey (`2022MNRAS.514.6053J`).
- **MPoL** (Zawadzki et al. 2023, `2023PASP..135f4503Z`; software paper `2025arXiv250200100C`) and its use on
  exoALMA kinematics (Zawadzki et al. 2025, ApJL 984, L14, `2025ApJ...984L..14Z`); exoALMA's own calibration and
  imaging (Loomis et al. 2025, `2025ApJ...984L...7L`), which studied the order of alignment, flux scaling and
  self-calibration.
- **Sparse modelling in 2D:** Yamaguchi et al. 2020, 2021, 2024 (`2020ApJ...895...84Y`, `2021ApJ...923..121Y`,
  `2024PASJ...76..437Y`; SMILI-like, images two to three times sharper than CLEAN in the 2024 sample).
- **Bayesian with learned priors:** IRIS (Dia et al. 2025, arXiv:2501.02473, `2025arXiv250102473D`): a
  score-based prior trained on optical galaxy images, posterior samples for DSHARP data, coverage tests on
  simulations.
- **Parametric model fitting on GPU:** galario (Tazzari et al. 2018, `2018MNRAS.476.4527T`).

### C3. GRAVITY and the Galactic Centre

- **Deep images** (GRAVITY Collaboration 2022, A&A 657, A82, `2022A&A...657A..82G`; read in the paper's text).
  - The tool is GR (GRAVITY-RESOLVE): Bayesian, in information field theory, built on resolve, with MGVI for the
    posterior. CLEAN is used as an independent confirmation (S300).
  - Prior: Sgr A* as a point source with a log-normal flux that may vary in time; further point sources with
    Gaussian position priors and log-normal fluxes; diffuse pixels with an inverse-Gamma prior.
  - Data: closure phases and visibility amplitudes, with per-exposure amplitude self-calibration (Gaussian
    scaling prior, mean 1, σ 0.1). One point source (usually Sgr A*) is fixed as an anchor because both
    observables are translation-invariant.
  - Instrument model: single-mode fibre damping, phase and amplitude maps, bandwidth smearing.
  - Faint S-stars: found as point-source components in the posterior; ten runs with different seeds per night,
    keeping the cleanest; positions then refined by point-source fits with priors from orbit predictions or
    the pixel position. Injection tests give about K ≈ 19.7 as the universal limit, below 20 under favourable
    conditions.
  - Imaging versus model fitting: imaging is used to search for unknown stars and as the starting point;
    the final astrometry comes from model fits.
- **Why CLEAN is stretched** (Mang et al. 2024, EAS abstract, `2024eas..conf.2299M`): the central 50 mas holds
  bright and very faint sources, and the spin of Sgr A* needs stars on short orbits fainter than any yet seen.
- **Recent GRAVITY results (2023–2026)**, all by model fitting of positions, not imaging:
  - polarimetry and astrometry of flares: hot spots at about 9 R_g with a Keplerian period (GRAVITY 2023, A&A
    677, L10, `2023A&A...677L..10G`; earlier, GRAVITY 2018, A&A 618, L10, `2018A&A...618L..10G`);
  - S2 and other stars: Schwarzschild precession detected (2020, `2020A&A...636L...5G`; about 10σ with S29, S38,
    S55 in GRAVITY 2024, A&A 692, A242, `2024A&A...692A.242G`), the extended mass (2022, `2022A&A...657L..12G`;
    Heißel et al. 2022, `2022A&A...660A..13H`), a fifth-force bound (2025, `2025A&A...698L..15G`);
  - theory for the astrometry: post-Newtonian relativistic astrometry (GRAVITY+ 2026, A&A 713, A226,
    `2026A&A...713A.226G`).
- **GRAVITY+** (adaptive optics first light, GRAVITY+ Collaboration 2026, A&A 707, A115, `2026A&A...707A.115G`):
  more sensitivity and sky coverage, so fainter stars and a larger field.
- **What limits imaging and astrometry** (from the 2022 paper, text): dynamic range against Sgr A*; the fibre
  field of view and fibre damping of off-axis sources; instrument calibration (aberration maps, bandwidth
  smearing) whose errors a high-dimensional Bayesian model is vulnerable to; sparse uv coverage (four
  telescopes, six baselines); no absolute phase, hence the anchor.

### C4. B1–B8 against the state of the art

- **B1 full-batch L-BFGS in float64.** Same as eht-imaging's optimizer (L-BFGS through SciPy), at far larger
  scale. Different from MPoL, which describes stochastic optimizers (software paper abstract). resolve uses MGVI,
  a natural-gradient/Newton-type scheme. Established.
- **B2 diagonal Gauss–Newton with entropy metric.** Preconditioning by a curvature estimate is the idea behind
  MGVI's Fisher metric in resolve; a diagonal Rademacher-probe version for L-BFGS in RML imaging is not in the
  EHT papers I read. Derived from published parts.
- **B3 nuisance parameterization.** The state of the art splits it: DSHARP and exoALMA align offsets and flux
  scales by hand before self-calibration; Themis and Comrade fit gains with priors; GR fixes one source as an
  anchor and fits amplitude scalings with priors. Joint fitting with a reference gauge matches the Bayesian
  tools' practice; I found no ALMA RML tool doing it with a reported bound hit.
- **B4 cold starts.** CLEAN models as the first self-calibration model is DSHARP practice; blurred re-starts are
  eht-imaging's; DoG-HiT starts from closure-only data. Coarse-to-fine uv truncation is not in anything I read.
- **B5 Toeplitz likelihood.** Established in MRI (Fessler 2005). Not used in the EHT or ALMA RML tools I read:
  MPoL and eht-imaging transform every visibility (or grid them) each step, resolve applies the full response.
  The nearest radio relatives are UVMULTIFIT's dirty-image fits and gridding-kernel work (Ye et al. 2020,
  `2020MNRAS.491.1146Y`). Exact fractional offsets: see the prior-art search; no precedent for the combination.
- **B6 float32 δV.** I found no radio-imaging precedent in this literature pass, only the numerical-analysis
  results it rests on. Lacks in the state of the art, because the others run float32 or float64.
- **B7 marginalized gains.** Done in Themis (Laplace, Sect. 6.8) and Comrade; closure-only methods remove the
  gains instead; resolve puts GP priors on gains (Arras et al. 2019). Nobody in the ALMA RML tools I read does
  it; ALMA practice is self-calibration (DSHARP).
- **B8 ordered subsets.** Not found in the radio imagers read here; minibatching is a deep-learning and medical
  imaging practice (Hudson & Larkin 1994). MPoL's optimizers are stochastic.

### C5. What they have that virgil lacks

- **Closure-only imaging:** eht-imaging (Chael 2018), DoG-HiT, GR (closure phases plus amplitudes); statistics
  of closure quantities (Blackburn 2020).
- **Scattering mitigation:** diffractive kernel division and refractive noise budgets (EHT Paper III);
  stochastic optics (Johnson 2016).
- **Variability handling:** the variability noise budget (Paper III).
- **Polarimetric imaging:** eht-imaging (Chael 2016), SMILI (Akiyama 2017b), polarimetric DoG-HiT (Müller &
  Lobanov 2023).
- **Dynamic imaging:** StarWarps (Bouman 2018), RML dynamic imaging (Johnson et al. 2017, ApJ 850, 172,
  `2017ApJ...850..172J`), resolve in time (Arras 2022), BH-NeRF (Levis 2022, 2024).
- **Posterior sampling of images:** Themis, Comrade/HIBI, resolve (MGVI), DPI/α-DPI, IRIS.
- **Learned priors:** score-based (Feng 2023, 2024; Sun 2024; Dia 2025), PRIMO's GRMHD library, DIRECT (Lai 2025).
- **Multi-epoch / multi-frequency:** Chael 2023 (multifrequency, spectral-index maps), Arras 2022.
- **Parameter surveys against synthetic truth:** EHT Papers IV and III choose imaging parameters this way.
- **1D super-resolution of axisymmetric discs:** frank.
- **Wide-field and direction-dependent effects:** resolve's wide-field response (Arras et al. 2021,
  `2021A&A...646A..58A`) and direction-dependent calibration (Roth 2023).
- **Instrument-specific forward models:** GR's fibre damping and aberration maps.

#### Scope decisions (2026-10-23)

Ben's calls on the list above:

- **Closure-only imaging:** wanted eventually; being scoped now. The other route is antenna-gain marginalization
  with broad priors (B7; another PR, in progress).
- **Scattering and variability:** out of scope for now.
- **Polarimetry:** wanted, but after 0.4.2.
- **Learned score-based priors:** out of scope (a separate PhD project).
- **Multi-frequency:** a global spectral index (SPARCO-like; `virgil.spectra`, `cal.alpha`) exists. Per-pixel
  spectral-index maps (MT-MFS, Rau & Cornwell 2011) are a later Toeplitz PR.
- **Posterior sampling of images:** wanted; GP imaging work is in progress in another session.

### C6. Where virgil could contribute

Defensible claims only. Each is conditional on the validation shown in vv.

- **A converged full-batch ALMA RML baseline.** On the shared objective MPoL's reported solution is not
  converged in flux, gains and offsets (B1; vv#176/#177). This is a statement about that solution, not about
  MPoL's software.
- **Speed to a given objective** on 42 M visibilities, from the Toeplitz likelihood once built and measured (B5).
  Not yet demonstrated.
- **Reproducible cross-checks:** a shared objective, identical folds, and independent checks in virgil-validation
  (B9). The EHT papers use synthetic-data surveys; this is complementary.
- **One JAX code path** from ALMA RML to GRAVITY/AMI model fitting and orbits, with exact gradients through the
  NUFFT. The tools above are separate codes per instrument. A convenience, not a scientific advance.
- **Joint nuisance fitting with a gauge** for ALMA (B3), in place of manual alignment, with the prior reported
  when it is hit.
- **Gain marginalization for ALMA continuum** (B7), if built: the Themis approach, which ALMA RML tools lack.

## D. Audit ledger (2026-10-23, virgil#531)

### D13. The B2 diagonal: w·b or w·b(1−b)?

- **Question.** The register wrote the diagonal of the entropy metric as w·b; the code (autodiff of
  `MaxEntropy.metric_rows` through `gauss_newton_diag`) returns w·b(1−b).
- **Parameterisation.** An `Image` stores logits η and its brightness is b = softmax(η) over the support
  (`models.py`, `Image.brightness`: unit sum). The fit's coordinates are η (the flux is a separate parameter).
  `metric_rows` returns rows r_i = 2√(w b_i).
- **Derivation.** With db_i/dη_j = b_i(δ_ij − b_j) and dr_i/db_i = √(w/b_i):
  (JᵀJ)_jj = Σ_i (w/b_i) b_i² (δ_ij − b_j)² = w Σ_i b_i (δ_ij − b_j)²
  = w [b_j (1−b_j)² + b_j² Σ_{i≠j} b_i] = w b_j (1−b_j)[(1−b_j) + b_j] = w b_j (1−b_j).
  In independent log-brightness coordinates (unnormalised b = e^η) the same rows give w·b, the
  Skilling–Bryan metric Σ b dη². The softmax's unit-sum constraint projects out the mean direction, and the
  full matrix is w(diag b − bbᵀ).
- **Numerical check** (scratch script, float64, central differences, 6 logits, w = 3.7; not in the repo): the
  finite-difference diagonal of JᵀJ equals w·b(1−b) to 2e-10 relative and differs from w·b by up to 20%; in
  unnormalised ln b coordinates it equals w·b to 1e-10.
- **Independent confirmation.** virgil-validation
  `tests/test_radio_innovations.py::test_entropy_metric_replaces_the_data_curvature_with_the_pulled_back_skilling_bryan_metric`
  asserts w·b(1−b) against virgil's `gauss_newton_diag` at 1e-9, from its own numpy code.
- **Verdict: the code is right and the doc was wrong.** The doc gave the ln b metric, not the diagonal in the
  coordinates the code uses. B2 and the docstrings of `fit`, `gauss_newton_diag` and `MaxEntropy.metric_rows` now
  say w·b(1−b). The two differ by under 1e-6 on a million-pixel image (b ≪ 1) and matter only on small images.
  With `smoothed=True` the rows are in the smoothed pixels and the diagonal is not w·b(1−b); no claim covers
  that case.

### Defaults audit

Every keyword default and auto-selection rule on a radio path, against the register (checked on `main`).
**No experimental or derived-unvalidated method is on by default.** Findings follow the table.

| Default | Where | Register item, status | Opt-in? |
|---|---|---|---|
| `fit(method=None)`: L-BFGS for a regularizer with `prefers_lbfgs` (MaxEntropy), else LM | `fitting.fit` | B1, established | n/a |
| `lbfgs_memory=None` → 100, cut to a 1 GiB history; `dtype="float64"`; `gtol=1e-4` | `fitting.fit` | B1, established | n/a |
| `precondition=None` (off) | `fitting.fit` | B2, derived | yes |
| `ft="auto"`: NUFFT above size thresholds (eps 1e-12 f64, 1e-5 f32), else DFT | `_geometry` | B1 (NUFFT), established; thresholds are timings | n/a |
| `ref_obs=0`; `gains`, `offsets`, `sigmas`, `alpha` all `True` (fitted, with more than one observation) | `CalibrationNuisance` | B3, established per part; the joint fit has no vv check (F2) | no: on |
| `sigmas=True` fits the scales; `"profile"` sets ŝ = √(χ²/2N) with stop-gradient | `CalibrationNuisance` | B3, established | `"profile"` opt-in |
| `offset_beams=3.0` (offset prior ±3 beams) | `CalibrationNuisance` | B3, a choice (F3) | no: on |
| `antenna_gains=False`; `solint="scan"`; widths 0.1, 0.5 rad; `fit_gain_widths=False`; `gain_steps=5` | `CalibrationNuisance` | B7 Stage 1, established; widths are choices (F3) | yes |
| `taper_fwhm_mas=None` (no taper) | `CalibrationNuisance` | A, established | yes |
| `offsets="none"` (needs `offsets=False`); `"spread"` | `ComplexVisData.toeplitz`, `ToeplitzVisData` | B5: Toeplitz terms established; `"spread"` derived | `"spread"` opt-in |
| `eps=1e-14`, `chunk_size=65536` | `toeplitz` | B5, numerical settings | n/a |
| `conjugate=None`, `u_sign=None` → `MS_CONJUGATE=True`, `MS_U_SIGN=1.0` | `ComplexVisData.read_ms` | convention; vv checks it against CASA | n/a |
| `weighting="natural"`, `robust=0.5`, `uvtaper=None`, `use_sigma=False` | `dirty_beam`, `dirty_image` | Briggs 1995; CASA practice (F4) | n/a |
| `deconvolver="hogbom"`, `weighting="natural"`, `robust=0.5`, `niter=1000`, `gain=0.1`, `threshold=0.0`, `nsigma=0.0`, `smallscalebias=0.6`, `psfcutoff=0.35`, `usemask="user"`, `chunk_size=1e6`, NUFFT model visibilities | `radio_clean.clean` | B4 (Cotton–Schwab CLEAN), established; names and defaults after `tclean` | n/a |
| `k=5`, `n_q=12`, `n_phi=16`, `seed=0` | `dartboard_folds` | none (F4) | n/a |

**Findings (not fixed here).**

- **F1.** The B2 code is right and the doc was wrong (D13); no code change.
- **F2.** `CalibrationNuisance` fits gains, offsets and spectral index jointly by default (B3). Each part is
  established; the joint fit has no vv check (vv#178 in progress). The default rests on published practice for
  the parts, not on validation of the combination.
- **F3.** Three defaults are choices with no source or validation: `offset_beams=3`, the gain widths 0.1 and
  0.5 rad (B7; the design note calls them a starting point), and the 100 mas offset bound planned for cold
  starts. They are documented as choices, and `at_bound` reports an offset that hits its bound.
- **F4.** Radio surfaces with no register entry: the spectral-index nuisance `alpha` (fitted by default),
  `ComplexVisData.sigma_scale` (an error-scale estimate from pair differences), `dartboard_folds`,
  `primary_beam`, and the Briggs and uv-taper options of `dirty_image`. They are not claimed as innovations
  here. Each needs a register entry, or a line saying it is established practice, with its tests and vv check.
- **F5.** B4's staging (image-fixed gains first, then everything) and coarse-to-fine live in scripts, not in
  library defaults.
- **F6.** The vv tests for B2, B5 and B6 are on vv `main` but absent from `trust/evidence/latest.jsonl` (the
  0.4.1 run skipped them on the pinned virgil). They are not release evidence until the evidence is rerun on a
  virgil that has them.

## Citation corrections

- **Broderick et al. 2020, ApJ 898, 9 (B7)** was cited for Themis's Laplace gain marginalization. That paper is
  the hybrid imaging and modelling paper (`2020ApJ...898....9B`). The Laplace marginalization of gains is in
  Sect. 6.8 of the Themis framework paper, Broderick et al. 2020, ApJ 897, 139 (`2020ApJ...897..139B`), as
  cited by EHT Paper III's Appendix A. Both are now cited.
- **Bouman et al. 2016 (CHIRP) as "Multiscale" (B4)** was a mis-attribution: CHIRP is a patch-prior Bayesian
  method, not a multiscale one. Replaced by Cornwell 2008 (multiscale CLEAN), DoG-HiT and DoB-CLEAN.
- **Arras et al. 2021, A&A 646, A84 (B5)** is correct, but its title is "Comparison of classical and Bayesian
  imaging in radio interferometry: Cygnus A with CLEAN and resolve", not a Toeplitz paper. It is cited only as
  the resolve application, with the remark that resolve applies the full response. Arras et al. 2019 (A&A 627,
  A134) added for gains as Gaussian processes.
- **Zawadzki et al. 2023** verified as PASP 135, 064503 (title: "Regularized Maximum Likelihood Image Synthesis
  and Validation for ALMA Continuum Observations of Protoplanetary Disks"), no change.
- **Wilkinson 1963 / Higham 2002 / Pawitan 2001** are books. Added the Higham and Pawitan DOIs and the SIAM
  reissue DOI for Wilkinson; the 1963 Prentice-Hall edition has no DOI.
- **Hudson & Larkin 1994, Bekas et al. 2007, Hutchinson 1989, Skilling & Bryan 1984, Cornwell & Evans 1985,
  Cornwell & Wilkinson 1981, Högbom 1974, Clark 1980, Schwab 1984, Readhead & Wilkinson 1978, Pearson &
  Readhead 1984, EHT Paper IV, Chael 2016/2018, Akiyama 2017 (both), Carrillo 2012, Pratley 2018, Rau &
  Cornwell 2011, Barnett 2019, Tiede 2022, Blackburn 2020, Hartikainen & Särkkä 2010, Fessler 2005, Nash 2000,
  Liu & Nocedal 1989, Jeffreys 1946, Martens 2010, Johnson & Zhang 2013, Narayan & Nityananda 1986:** verified
  as cited (authors, year, venue, volume, page); bibcode or DOI added.
- **Unverified:**
  - Wajer & Pruessmann 2001, Proc. ISMRM 9, 767: the ISMRM archive page returned 404 and web search found no
    record of the abstract. The authors, venue and year are from memory. Fessler et al. 2005 carries the same
    claim and is verified.
  - "ALMA memos" on weight miscalibration (B3): no specific memo was identified. The CASA `statwt` and "Data
    Weights" documentation are cited instead.
- **Count:** 40 distinct references in the original *(verify)* entries: 37 verified as cited, 1 corrected
  (Broderick), 2 unverified (Wajer & Pruessmann; ALMA memos). CHIRP is a real paper (verified) but was
  mis-attributed to multiscale imaging, and is now cited in C1 only.
