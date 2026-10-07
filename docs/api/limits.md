# `virgil.limits`

Contrast limits, significance, and conversions between flux ratios
(companion/primary), contrasts (primary/companion) and magnitudes. A companion
100 times fainter than the primary has flux 0.01, contrast 100 and Δmag 5.

::: virgil.limits
    options:
      members:
        - flux_to_contrast
        - contrast_to_flux
        - flux_to_delta_mag
        - delta_mag_to_flux
        - ruffio_upperlimit
        - absil_limits
        - injection_limits
        - chi2ppf
        - nsigma
        - radial_profile
