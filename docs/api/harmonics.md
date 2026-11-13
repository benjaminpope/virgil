# `virgil.harmonics`

Azimuthal harmonics of a [`MultipoleDisk`](models/classes/multipole_disk.md): the image
harmonics $I_m$ that every form maps to and from (the linear, positive and von
Mises forms of the azimuthal profile and the legacy `az_amps`/`az_pas` of
`ModulatedGaussianRim` and `TruncatedCone`), and the reflection-coefficient
machinery of the positive form.

At each radius, every azimuthal profile of a [`MultipoleDisk`][virgil.models.MultipoleDisk] is a real function of the azimuth $\theta$ (in the disk plane, from the projected major axis towards `pa + 90`) with the Fourier series

$$
\frac{I(\theta)}{I_0} = \sum_{m=-M}^{M} I_m e^{im\theta}, \qquad I_0 = 1, \quad I_{-m} = \bar I_m,
$$

whose *image harmonics* $I_m$ are complex, $m = 0 \dots M$, normalized to the azimuthal mean. Every form maps to and from them with [`to_image_harmonics`][virgil.harmonics.to_image_harmonics] and [`from_image_harmonics`][virgil.harmonics.from_image_harmonics] (or `Harmonics.convert`):

- **linear**: $1 + \sum_m c_m \cos m\theta + s_m \sin m\theta$, so $c_m = 2\,\mathrm{Re}\,I_m$ and $s_m = -2\,\mathrm{Im}\,I_m$ (exact);
- **positive**: $|h(e^{i\theta})|^2 / \sum_j |h_j|^2$ with $h(z) = 1 + h_1 z + \dots + h_M z^M$, whose roots all lie outside the unit circle. It is built from complex reflection coefficients $k_j$ with $|k_j| < 1$ by the Levinson step-up recursion $h^{(j)}_i = h^{(j-1)}_i + k_j \overline{h^{(j-1)}_{j-i}}$, with $k = w \tanh|w| / |w|$ from unconstrained latents $w_j \in \mathbb{C}$. Then $I_m = \sum_j h_{j+m} \bar h_j / \sum_j |h_j|^2$ (exact); the inverse is a spectral factorization (`from_image_harmonics`);
- **von Mises**, $\exp[\kappa\cos(\theta - \theta_0)]$, truncated at the model's degree;
- the legacy amplitude and azimuth pairs `az_amps`/`az_pas` of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim] and [`TruncatedCone`][virgil.models.TruncatedCone].

Use the conversions to report models in one convention, warm-start a positive-form fit from a linear one, or compare models. Report image harmonics, not the positive form's internal reflection coefficients.

References: Dumitrescu 2007, *Positive Trigonometric Polynomials and Signal Processing Applications* (Springer, doi:10.1007/978-1-4020-5125-8), for the Fejér–Riesz theorem, spectral factorization and the Schur–Levinson recursion; Barndorff-Nielsen & Schou 1973, J. Multivariate Anal. 3, 408 (doi:10.1016/0047-259X(73)90030-4), for the unconstrained parametrization of a minimum-phase polynomial by reflection (partial autocorrelation) coefficients, here with a complex tanh map.

::: virgil.harmonics
    options:
      members:
        - to_image_harmonics
        - from_image_harmonics
        - from_legacy
        - to_legacy
        - rim_coefficients
        - cone_coefficients
        - reflection_coefficients
        - step_up
        - step_down
        - latents_from_reflection
        - spectral_factor
        - image_harmonics_from_h
