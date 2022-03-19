from .lagrangian_points import xl11, xl12, xl1
from .methods import rtsafe
from .potentials import rpot1, rpot2, rpot
from .vector import Vec3
from .types import Star
from jax import numpy as jnp
from jax import grad, jit
from functools import partial


from jax.experimental import loops


def rlobe_eggleton(q):
    """
    The volume-averaged Roche lobe radius, as defined by Peter Eggelton's formula.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    Returns
    -------
    rl1_a: float, `jnp.DeviceArray`
        The volume equivalent radius of the Roche-lobe. This is the radius of a
        sphere that has the same volume as the Roche lobe.
    """
    q3 = q ** (1.0 / 3.0)
    return 0.49 * q3 * q3 / (0.6 * q3 * q3 + jnp.log(1 + q3))


def ref_sphere(q: float, star: Star, spin: float, ffac: float):
    """
    Computes the radius of a reference sphere just touching a Roche distorted star.

    `ref_sphere` computes the radius of a reference sphere just touching a Roche-distorted
    star along the line of centres and centred upon its centre of mass. This sphere, which
    is guaranteed to enclose the equipotential in question can then be used to define
    regions for searching for equipotential crossing when computing eclipses.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    star: `jax_roche.types.Star`
        Compute for primary or secondary star?

    spin: float, `jnp.DeviceArray`
        spin = ratio of angular/orbital frequency of primary/secondary

    ffac: float, `jnp.DeviceArray`
        Filling factor. Defined as the distance from the centre of mass
        of the star to its surface in the direction of the L1 point,
        divided by the distance to the L1 point. `ffac=1` defines a
        Roche-lobe filling star.

    Returns
    -------
    rref: float, `jnp.DeviceArray`
        Radius of the reference sphere

    pref: float,  `jnp.DeviceArray`
        The reference potential. This is the Roche potential on surface of 
        the distorted star.
    """
    if star == Star.PRIMARY:
        rref = ffac * xl11(q, spin)
        pref = rpot1(q, spin, Vec3(rref, 0.0, 0.0))
    if star == Star.SECONDARY:
        rref = ffac * (1 - xl12(q, spin))
        pref = rpot2(q, spin, Vec3(1 - rref, 0.0, 0.0))
    return rref, pref


@jit
def lobe2(q, n=200):

    rl1 = xl1(q)
    cpot = rpot(q, Vec3(rl1, 0.0, 0.0))
    upper = 1.0 - rl1
    lower = upper / 4

    @jit
    def root_func(t, dx, dy):
        """
        t is a step in direction dx, dy
        """
        p = Vec3(1.0 + t * dx, t * dy, 0.0)
        return rpot(q, p) - cpot

    gradf = jit(grad(root_func))
    with loops.Scope() as s:
        s.x = jnp.zeros(n)
        s.y = jnp.zeros(n)
        for i in s.range(n):
            theta = 2.0 * jnp.pi * i / (n - 1)
            dx = -jnp.cos(theta)
            dy = jnp.sin(theta)

            f = partial(root_func, dx=dx, dy=dy)
            df = partial(gradf, dx=dx, dy=dy)
            rad = rtsafe(f, df, lower, upper)
            s.x = s.x.at[i].set(1.0 + rad * dx)
            s.y = s.y.at[i].set(rad * dy)
    return s.x, s.y
