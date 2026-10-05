"""Sketch: an orbit driving a companion and a component attached to it.

Design sketch for design/orbit_scene_joint_fitting.md, not library code. It
shows the proposed ``KeplerOrbit.relative`` (on jaxoplanet, an optional
dependency) and ``Attached(...).at(mjd)``, using only existing virgil
components: a companion whose position follows the orbit, and a disc around
the secondary that lies in the orbital plane (its projected major axis along
the line of nodes, inclined by i) and is brighter on the side facing the
primary. Every sample is evaluated at its own MJD. Run with::

    uv run --python .venv/bin/python --with jaxoplanet \
        python design/sketches/orbit_attached.py

Conventions (section 2 of the note): r points from the primary (the origin)
to the secondary, as (dra, ddec, dz) in mas with dz positive away from the
observer; omega is the secondary's argument of periastron; Omega is the
position angle (North through East) of the node where the secondary recedes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody

from virgil.models import ModulatedGaussianRim, PointSource, System

jax.config.update("jax_enable_x64", True)
DEG = np.pi / 180.0
YEAR = 365.25


class KeplerOrbit(eqx.Module):
    """Relative orbit of a secondary about a primary, in angular units."""

    period: jax.Array  # days
    dt_peri: (
        jax.Array
    )  # days after t_ref (relative, so float32 keeps it precise)
    ecc: jax.Array
    inc: jax.Array  # deg; < 90 means the position angle increases
    omega: jax.Array  # deg, the secondary's argument of periastron
    Omega: jax.Array  # deg, PA of the node where the secondary recedes
    a_mas: jax.Array
    t_ref: float = eqx.field(static=True)  # float64 MJD, kept on the host

    def _body(self):
        # jaxoplanet's omega_peri is the primary's: the secondary's minus 180.
        # Its units (R_sun, M_sun, days) cancel once divided by the semimajor
        # axis; never pass its parallax argument (it scales by au, not 1/au).
        body = Body(
            period=self.period,
            time_peri=self.dt_peri,
            eccentricity=self.ecc,
            omega_peri=(self.omega - 180.0) * DEG,
            inclination=self.inc * DEG,
            asc_node=self.Omega * DEG,
        )
        return OrbitalBody(Central(mass=1.0, radius=1.0), body)

    def relative(self, t):
        """(dra, ddec, dz) of the secondary at MJD t, in mas."""
        body = self._body()
        north, east, toward = body.relative_position(t - self.t_ref)
        scale = self.a_mas / body.semimajor
        return east * scale, north * scale, -toward * scale

    def frame(self, t):
        """Angles of the binary frame at MJD t, in degrees."""
        dra, ddec, dz = self.relative(t)
        line_pa = jnp.degrees(jnp.arctan2(dra, ddec))
        return dict(
            line_pa=line_pa,  # primary -> secondary
            towards_primary=line_pa + 180.0,
            line_tilt=jnp.degrees(jnp.arctan2(dz, jnp.hypot(dra, ddec))),
            node_pa=self.Omega + 0.0 * line_pa,
            inc=self.inc + 0.0 * line_pa,  # physical, 0-180
            # The projected tilt, for components whose inc is apparent (0-90).
            apparent_inc=jnp.degrees(
                jnp.arccos(jnp.abs(jnp.cos(self.inc * DEG)))
            )
            + 0.0 * line_pa,
        )


class Attached(eqx.Module):
    """A component placed and oriented in the binary frame of an orbit.

    ``anchor`` is "primary" (the origin) or "secondary"; ``bind`` maps the
    component's angle attributes to frame angles (``KeplerOrbit.frame``),
    and ``offsets`` adds fitted offsets to them (e.g. a skew).
    """

    component: eqx.Module
    orbit: KeplerOrbit
    offsets: dict
    bind: tuple = eqx.field(static=True)
    anchor: str = eqx.field(static=True, default="secondary")

    def __init__(
        self, component, orbit, bind, offsets=None, anchor="secondary"
    ):
        self.component, self.orbit, self.anchor = component, orbit, anchor
        self.bind = tuple(bind.items())
        self.offsets = {k: jnp.asarray(0.0) for k, _ in self.bind} | dict(
            offsets or {}
        )

    def at(self, t):
        dra, ddec, _ = self.orbit.relative(t)
        if self.anchor == "primary":
            dra, ddec = 0.0 * dra, 0.0 * ddec
        frame = self.orbit.frame(t)
        out = eqx.tree_at(
            lambda c: (c.dra, c.ddec), self.component, (dra, ddec)
        )
        # Bind the geometry (pa, inc, ...) first: azimuthal angles depend on it.
        for attr, quantity in sorted(
            self.bind, key=lambda b: b[0] == "az_pas"
        ):
            old = getattr(out, attr)
            value = frame[quantity] + self.offsets[attr]
            if attr == "az_pas":
                value = _disc_angle(value, out.pa, out.inc)
            new = jnp.broadcast_to(value, jnp.shape(old))
            out = eqx.tree_at(lambda c: getattr(c, attr), out, new)
        return out


def _disc_angle(sky_pa, pa, inc):
    """The rim angle whose projection points at sky position angle ``sky_pa``.

    ``ModulatedGaussianRim`` measures ``az_pas - pa`` in the deprojected
    disc, then compresses the minor axis by cos(inc); a sky angle must be
    deprojected first, or a modulation aimed at the primary misses it.
    """
    d = (sky_pa - pa) * DEG
    return pa + jnp.degrees(
        jnp.arctan2(jnp.sin(d) / jnp.cos(inc * DEG), jnp.cos(d))
    )


T_REF = 60500.0
orbit = KeplerOrbit(
    period=jnp.asarray(2.0 * YEAR),
    dt_peri=jnp.asarray(-100.0),  # periastron at MJD T_REF - 100
    ecc=jnp.asarray(0.3),
    inc=jnp.asarray(50.0),
    omega=jnp.asarray(60.0),
    Omega=jnp.asarray(120.0),
    a_mas=jnp.asarray(20.0),
    t_ref=T_REF,
)
disc = Attached(
    ModulatedGaussianRim(
        3.0, 1.0, 0.0, 0.0, az_amps=0.5, az_pas=0.0, flux=0.05
    ),
    orbit,
    bind={"pa": "node_pa", "inc": "apparent_inc", "az_pas": "towards_primary"},
)


def scene_at(t):
    dra, ddec, _ = orbit.relative(t)
    return System(
        primary=PointSource(),
        secondary=PointSource(0.3, dra, ddec),
        disc=disc.at(t),
    )


# Per-sample evaluation: each sample has its own MJD (one per frame).
rng = np.random.default_rng(1)
n = 300
length = rng.uniform(10.0, 130.0, n)
angle = rng.uniform(0.0, np.pi, n)
u, v = length * np.sin(angle), length * np.cos(angle)
wavel = np.full(n, 2.2e-6)
mjd = T_REF + rng.choice([0.0, 120.0, 240.0], n) + rng.uniform(0, 0.2, n)

cvis = jax.vmap(lambda t, u, v, w: scene_at(t).model(u, v, w))(
    mjd, u, v, wavel
)
print(f"{n} samples evaluated at their own MJDs; |V| in", end=" ")
print(f"[{float(jnp.min(jnp.abs(cvis))):.3f}, 1]")

print("MJD - t_ref   sep    PA   tilt   disc pa  inc  bright side")
for dt in (0.0, 120.0, 240.0):
    t = T_REF + dt
    dra, ddec, dz = orbit.relative(t)
    frame = orbit.frame(t)
    attached = disc.at(t)
    print(
        f"{dt:10.0f}  {float(jnp.hypot(dra, ddec)):5.2f}"
        f"  {float(frame['line_pa']) % 360:5.1f}"
        f"  {float(frame['line_tilt']):5.1f}"
        f"  {float(attached.pa) % 360:7.1f}"
        f"  {float(attached.inc):4.0f}"
        f"  {float(attached.az_pas[0]) % 360:6.1f}"
    )

# Orientation check: the disc must brighten towards the primary on the sky.
# At this orbit's 50° inclination the deprojected rim angle differs from the
# sky angle by up to ~13° (the table above), so binding the sky angle to
# az_pas directly would aim the modulation that far off.
npix, fov = 128, 8.0
xs = (np.arange(npix) - npix / 2 + 0.5) * fov / npix
for dt in (0.0, 120.0, 240.0):
    t = T_REF + dt
    attached = disc.at(t)
    centred = eqx.tree_at(lambda c: (c.dra, c.ddec), attached, (0.0, 0.0))
    image = np.asarray(centred.render(npix, fov))
    # render(): column 0 is the most positive dra (East left), row 0 North.
    # The flux-weighted centroid of a ring modulated as 1 + a cos(θ - φ)
    # points along the projection of φ (projection is linear). The brightest
    # pixel does too, since the rim is blurred in its own plane, but only to
    # the nearest pixel, so the centroid is the sharper check.
    dra = (-xs)[None, :] * np.ones((npix, 1))
    ddec = (-xs)[:, None] * np.ones((1, npix))
    bright_pa = (
        np.degrees(np.arctan2(np.sum(image * dra), np.sum(image * ddec))) % 360
    )
    want = float(orbit.frame(t)["towards_primary"]) % 360
    miss = (bright_pa - want + 180) % 360 - 180
    print(
        f"MJD - t_ref {dt:5.0f}: disc brightens towards PA {bright_pa:5.1f};"
        f" the primary is at {want:5.1f}"
    )
    assert abs(miss) < 2.0, miss
