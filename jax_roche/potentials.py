from .vector import Vec3
from jax import numpy as jnp
from jax import jit


@jit
def rpot(q: float, p: Vec3):
    """
    Computes the Roche Potential at a given point.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    p: `jax_roche.Vec3`
        The point in question (all units scaled by separation)

    Returns
    -------
    rpot: `jnp.DeviceArray`
        The Roche Potential at `p`
    """
    mu = q / (1 + q)
    comp = 1 - mu
    x2y2 = p.x ** 2 + p.y ** 2
    z2 = p.z ** 2
    r1sq = x2y2 + z2
    r1 = jnp.sqrt(r1sq)
    r2 = jnp.sqrt(r1sq + 1 - 2 * p.x)
    return -comp / r1 - mu / r2 - (x2y2 + mu * (mu - 2 * p.x)) / 2


@jit
def rpot1(q: float, spin: float, p: Vec3):
    """
    Computes the Roche Potential at a given point, allowing asynchronous rotation of primary.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    spin: float, `jnp.DeviceArray`
        spin = ratio of angular/orbital frequency of primary

    p: `jax_roche.Vec3`
        The point in question (all units scaled by separation)

    Returns
    -------
    rpot: `jnp.DeviceArray`
        The Roche Potential at `p`
    """
    mu = q / (1 + q)
    comp = 1 - mu
    x2y2 = p.x ** 2 + p.y ** 2
    z2 = p.z ** 2
    r1sq = x2y2 + z2
    r1 = jnp.sqrt(r1sq)
    r2 = jnp.sqrt(r1sq + 1 - 2 * p.x)
    return -comp / r1 - mu / r2 - spin * spin * x2y2 / 2.0 + mu * p.x


@jit
def rpot2(q: float, spin: float, p: Vec3):
    """
    Computes the Roche Potential at a given point, allowing asynchronous rotation of secondary.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    spin: float, `jnp.DeviceArray`
        spin = ratio of angular/orbital frequency of secondary

    p: `jax_roche.Vec3`
        The point in question (all units scaled by separation)

    Returns
    -------
    rpot: `jnp.DeviceArray`
        The Roche Potential at `p`
    """
    mu = q / (1 + q)
    comp = 1 - mu
    x2y2 = p.x ** 2 + p.y ** 2
    z2 = p.z ** 2
    r1sq = x2y2 + z2
    r1 = jnp.sqrt(r1sq)
    r2 = jnp.sqrt(r1sq + 1 - 2 * p.x)
    return -comp / r1 - mu / r2 - spin * spin * (0.5 + 0.5 * x2y2 - p.x) - comp * p.x
