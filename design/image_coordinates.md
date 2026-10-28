# Image coordinate convention

> Reference for agents, split out of [AGENTS.md](../AGENTS.md); the rules there still apply. Keep this file in step with the code.

The user-facing version of the conventions (coordinates, Fourier sign, observables, fluxes, times) is `docs/conventions.md`; keep it in step with this section.

**This convention must never be violated.** It has been the direct cause of real
bugs (see below), and any new coordinate, rendering, or plotting code must be
verified against it with a direct orientation test — not just a
finiteness/normalization check.

- Rendered images and `(dra, ddec)` grids are 2D arrays using standard array
  indexing: the first index (row) runs top-to-bottom, the second index (column)
  runs left-to-right. That part is generic.
- The **physical sky coordinates** assigned to those array positions follow
  astronomical convention, not a generic image convention: **East points left,
  North points up**.
  - `x` (a right-ascension-like offset, e.g. `dra`) **increases to the left**
    (East is positive `x`).
  - `y` (a declination-like offset, e.g. `ddec`) **increases upward** (North
    is positive `y`).
  - Consequently: as the **row index increases, `y` decreases** (row 0 =
    North, the top row). As the **column index increases, `x` decreases**
    (column 0 = East, the leftmost column).
- The coordinate origin `(x, y) = (0, 0)` sits at the **geometric center of
  the pixel grid** — the center of the center pixel(s), not a pixel edge.
- **Position angles** (binary separations, an ellipse's projected major axis,
  azimuthal modulation phases) are measured **counter-clockwise from the top**,
  i.e. **North-to-East**: PA=0° points North, PA=90° points East.

The canonical, tested reference implementation is `image_coordinates` in
`src/virgil/_geometry.py` (image-plane pixel coordinates) and the elliptical
rotation/stretch helpers in the same module (`undo_`/`apply_elliptical_transf_coord`
and `..._spat_freq`). Every geometric `SourceModel`'s `model()`/`render()` must
be dimensionally consistent with these, and should have a direct regression
test analogous to `test_gaussian_disk_render_uses_interferometric_image_orientation`,
`test_uniform_disk_render_uses_interferometric_image_orientation`, or
`test_modulated_gaussian_rim_render_follows_north_to_east_pa_convention` in
`tests/test_models_sources.py` — i.e. one that asserts flux/argmax lands at the
*correct* pixel for a known offset or PA, not just that the output is finite.

**Plotting rule:** any `imshow`-based display of a sky-coordinate image or grid
must end up with the x-axis increasing toward the left (East) and the y-axis
increasing toward the top (North) *regardless of how the underlying
array/extent/origin was constructed*. Do not assume a particular `dra`/`ddec`
axis ordering (ascending vs. descending) — different parts of this codebase
build these axes both ways. Use `_enforce_sky_orientation` in
`src/virgil/plotting.py`, which corrects an `Axes`' final displayed limits
regardless of the plotted array's construction, rather than hand-tuning
`origin`/`extent` per call site.

Historical motivation: this exact convention was violated in three independent
places found in one audit — `BinaryModelAngular`'s position angle was mirrored
about the North-South axis (a real bug in `model()`/`render()`, not just
display); `notebooks/source_models.ipynb`'s render plots used `origin="lower"`
with a non-reversed extent; and two of `plotting.py`'s five grid-plotting
functions displayed `dra` backwards. None of these were caught by existing
tests because none of them asserted orientation directly — see
`add_modulated_gaussian_rim_and_uniform_disk`'s commit history for the fixes.

Every model's `render()` must also be consistent with its `model()`: the
Fourier transform of the rendered image must reproduce the visibilities
(`test_render_fourier_transform_matches_model_visibilities`). Add new models
to that test.
