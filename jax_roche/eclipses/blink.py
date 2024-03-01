from ..vector import xhat, intersection_line_sphere, Vec3
from ..lagrangian_points import xl1
from ..potentials import rpot
from ..roche_lobes import ref_sphere
from .sphere_eclipse import sphere_eclipse

from functools import partial
from jax.lax import cond, while_loop
from jax import numpy as jnp
from jax import jit, grad, vmap

# gradient of roche potential w.r.t position
gradp = grad(rpot, argnums=1)


@jit
def calc_cache(q):
    xcm = 1 / (1 + q)
    # Find Roche potential at L1 point
    rl1 = xl1(q)
    crit = rpot(q, Vec3(rl1, 0, 0))

    # the donor lies entirely within the sphere centred on its
    # center of mass and reaching the inner Lagrangian point
    rsphere = 1 - rl1
    pp = rsphere * rsphere

    return dict(xcm=xcm, rsphere=rsphere, pp=pp, crit=crit)


@jit
def blink(q, position, earth, acc=0.1):
    """
    This routine finds if a point in a semi-detached binary is occulted by the donor star.
    It works by first eliminating as many cases as possible that are far from being
    occulted, before applying a brute force method of stepping along the line-of-sight
    to that position and checking if it ever falls within the Roche Lobe.

    To speed things up, `blink` can accept a cache of stored values which need to
    be re-computed every time q changes. If the cache is not provided, these
    values are computed from scratch.

    Parameters
    ----------
    q: float, `jax.Array`
        Mass ratio M2/M1
    position: `jax_roche.Vec3`
        The position vector of the point in the binary, scaled by binary separation
    earth: `jax_roche.Vec3`
        A unit vectory pointing towards the Observer (Earth).
        Usually, this is (sini(i)*cos(phi), -sin(i)*sin(phi), cos(i)),
        where phi is the orbital phase and i is the inclination.
    acc: float, `jax.Array` (default=0.1)
        Step size parameter.
        acc specifies the size of steps taken when trying to see if the photon path
        goes inside the Roche lobe. The step size is roughly acc times the radius
        of the lobe filling star. This means that the photon path could mistakenly
        not be occulted if it passed less than about (acc**2)/8 of the radius below
        the surface of the Roche lobe. acc of order 0.1 should therefore do the job.

    Returns
    -------
    eclipsed: bool
        True if the point is eclipsed, False if not
    cache: tuple
        Stored values
    """
    cache = calc_cache(q)
    state = dict()

    # step size is roughly acc * Roche lobe radius
    state["step"] = acc * cache["rsphere"]

    # evaluate closest approach distance to reference sphere.
    # dist1 is smaller, dist2 is larger
    # if both nan, then the line does not intersect the sphere
    # if one is nan the line just touches the sphere
    # if dist2 is negative then they both are and there is no eclipse (the
    # sphere is behind the position from the PoV of the observer).
    dist1, dist2 = intersection_line_sphere(xhat, cache["rsphere"], position, earth)
    eclipsed, dist1, dist2 = sphere_eclipse(earth, position, xhat, cache["rsphere"])

    state["dist1"], state["dist2"] = dist1, dist2
    return cond(
        ~eclipsed,
        lambda *args: False,
        _blink_step1,
        q,
        position,
        earth,
        state,
        cache,
    )


def _blink_step1(q, position, earth, state, cache):
    """
    Does the harder work of checking for eclipses: the photon enters the reference sphere
    """
    # first rule out an easy case - is the closest approach at the centre of the
    # donor

    # a = position - xhat is vector from centre of donor to position
    # closest approach to xhat is a distance -a.dot(earth) along LOS from position
    state["dist"] = (xhat - position).dot(earth)
    # closest is vector to closest approach
    closest = position + state["dist"] * earth
    state["closest"] = closest

    # point at COM of red star, definitely eclipsed.
    # returning true here avoids division by zero later
    return cond(
        closest == xhat,
        lambda *args: True,
        _blink_step2,
        q,
        position,
        earth,
        state,
        cache,
    )


@jit
def _blink_step2(q, position, earth, state, cache):
    # are we deeper in the well than lagrangian? If so eclipsed.
    # this should catch most cases
    pot = rpot(q, state["closest"])
    return cond(
        pot < cache["crit"],
        lambda *args: True,
        _blink_step3,
        q,
        position,
        earth,
        state,
        cache,
    )


@jit
def _blink_step3(q, position, earth, state, cache):
    """
    OK, all easy cases handled - now we step
    """
    # step direction established by evaluating first derivative of R.P at closest approach
    p = state["closest"]

    # positive if potential is increasing along LOS towards earth
    derivative = gradp(q, p).dot(earth)

    # if Roche Potential is increasing along LOS towards earth, step away from Earth
    p1 = state["dist"]
    p2 = jnp.where(derivative >= 0.0, state["dist1"], state["dist2"])
    nstep = jnp.floor(0.5 + jnp.fabs(p2 - p1) / state["step"]).astype("int32")

    def continue_cond(state):
        # stop if we've reached the end or the potential is less than the critical potential
        n, min_pot = state
        return (n < nstep) & (min_pot >= cache["crit"])

    def step(state):
        n, min_pot = state
        gamma = p1 + (p2 - p1) * n / nstep
        newpos = position + gamma * earth
        min_pot = jnp.minimum(min_pot, rpot(q, newpos))
        return (n + 1, min_pot)

    nsteps_taken, min_pot = while_loop(continue_cond, step, (0, 1.0))
    return min_pot < cache["crit"]
