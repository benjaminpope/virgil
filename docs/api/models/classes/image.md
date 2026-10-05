# `Image`: pixelised brightness

A pixel image is one more [component](composition.md) of a `System`, for
image reconstruction. Its visibilities are the exact Fourier transform of
the pixels. The design is recorded in `design/image_reconstruction.md`.

::: virgil.models.Image
    options:
      show_root_heading: true
      heading_level: 2
      members:
        - brightness
        - eta
        - from_brightness
        - from_model

::: virgil.models.circular_support
    options:
      show_root_heading: true
      heading_level: 2
