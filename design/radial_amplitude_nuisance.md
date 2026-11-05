# Radial log-amplitude nuisance for projected DISCO data

Register entry (AGENTS.md, "Methods beyond published practice").

**Claim.** On DISCO data a radially symmetric log-amplitude error (a PSF-shape mismatch between target and
calibrator) is common to every amplitude template of a faint companion, so it biases amplitude-only flux
maps in one sign, at every position. Marginalizing a smooth radial profile on log |V| removes it.

**Method.** The profile is `Σ_k c_k P_k(x)`, `x = 2q/q_max − 1`, `q = |u|/λ`, `k ≤ order` (default 4); `P_k` are
Legendre polynomials, the same span as the powers `1, q, …, q^order` but better conditioned. With a flat
prior on `c` the likelihood is the Gaussian marginal under covariance `D + N Nᵀ τ²`, `N = vis_mat · P` (the
projected modes, the Jacobian of a log-amplitude being 1). It is the existing gain marginalization
(`virgil.gains`, Lachaume 2021, arXiv:2104.07082) with one block of projected modes; as τ → ∞ it is the QR
projection of N out of the model gradient and the residual. The default width 10 is a near-flat stand-in;
the width is the noise term `vis_gain_radial`. The modes are never drawn when simulating.

**Cost.** Amplitude sensitivity falls, most at small separations (about half inside 200 mas, a quarter beyond,
on the bare-star test products). A real companion's azimuthally symmetric amplitude signal is projected out
with the nuisance; its peak position is unchanged. Phases are unaffected.

**Why order 4.** Offset plus `q²` left the bias in place on a bare-star product; degree 4 removed it.
Whether a particular dataset needs fewer is for the user to check.

**Cause.** A PSF-shape mismatch, perhaps the brighter-fatter effect: a hypothesis, not shown here.

**Pins.** `tests/test_radial_amplitude.py` (independent numpy QR projection onto the powers to 1e-9;
bias of either sign removed; companion SNR falls by the projection's factor and its peak stays; width fit
finite; float32). virgil-validation check: none yet.

**Status.** *derived* (a published marginalization applied to one more mode set); opt-in, never a default.
