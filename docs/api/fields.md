# `virgil.fields`

Gaussian-process priors for pixel images. A `GaussianField` can take the place
of an `Image`'s log-brightness array: the log-brightness is then a stationary
Gaussian process about an optional template, parameterized by whitened
coefficients on the image's cosine basis, which
`imaging.image_priors` gives standard-normal priors.

::: virgil.fields
    options:
      members:
        - GaussianField
        - field_spectrum
