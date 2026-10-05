# `virgil.imaging`

Regularisers, priors and helpers for image reconstruction with
[`Image`](models/classes/image.md). How to choose a regularisation weight is
discussed in the [design note](https://github.com/benjaminpope/virgil/blob/main/design/regulariser_weight_selection.md): the L-curve's corner, the
discrepancy principle, classic MaxEnt (`LCurve.classic_maxent`) and, for
Gaussian-field images, the Laplace evidence (`log_evidence`).

::: virgil.imaging
    options:
      members:
        - TSV
        - TV
        - MaxEntropy
        - Laplacian
        - StarletL1
        - LogSum
        - starlet
        - clean
        - CleanResult
        - Centroid
        - starting_image
        - dirty_image
        - image_priors
        - nyquist_pixel_scale
        - field_of_view
        - beam
        - Beam
        - convolve_beam
        - l_curve
        - LCurve
        - log_evidence
        - laplace_samples
        - error_scale
        - diagnose
        - Diagnosis

::: virgil._geometry
    options:
      members:
        - pixel_offsets
