# `virgil.gains`

Calibration gains correlated across the channels of a frame, as low-rank
blocks of the visibility covariance that the likelihood marginalizes
analytically. Add them with
[`OIData.with_gains`][virgil.oidata.OIData.with_gains], and fit their widths
with the noise terms `vis_gain_telescope`, `vis_gain_baseline`,
`vis_gain_chromatic` and `vis_gain_modes` (e.g.
`fit(..., noise={"vis_gain_telescope": dist.LogUniform(1e-4, 0.1)})`). Widths
are scale parameters, so their default (Jeffreys) prior is log-uniform on
stated bounds; a `Uniform(0, ...)` would favour large widths.

Closure-phase offsets common to a frame's channels work the same way, on the
whitened closure phases: add them with
[`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets] and
fit their widths with `phi_offset_baseline`, `phi_offset_triangle` and
`phi_offset_modes`. They are off by default.

::: virgil.gains
    options:
      members:
        - GainModes
        - gain_modes
        - GAIN_GROUPS
        - ClosureOffsets
        - closure_offsets
        - OFFSET_GROUPS
