# `virgil.gains`

Calibration gains correlated across the channels of a frame, as low-rank
blocks of the visibility covariance that the likelihood marginalises
analytically. Add them with
[`OIData.with_gains`][virgil.oidata.OIData.with_gains], and fit their widths
with the noise terms `vis_gain_telescope`, `vis_gain_baseline`,
`vis_gain_chromatic` and `vis_gain_modes` (e.g.
`fit(..., noise={"vis_gain_telescope": dist.Uniform(0, 0.1)})`).

::: virgil.gains
    options:
      members:
        - GainModes
        - gain_modes
        - GAIN_GROUPS
