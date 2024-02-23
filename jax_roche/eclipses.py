from .vector import Vec3, xhat, intersection_line_sphere
from .lagrangian_points import xl1

from jax.lax import cond
from jax import numpy as jnp
from jax import jit, grad
from functools import partial


def calc_cache(q):
    """
    Cache values of quantities that depend on q
    """
    xcm = 1 / (1 + q)
    c1 = 2 * xcm
    xcm *= q
    c2 = 2 * xcm
    # Find Roche potential at L1 point
    rl1 = xl1(q)
    r2 = 1 - rl1
    xc = rl1 - xcm
    crit = c1 / rl1 + c2 / r2 + xc * xc

    # the donor lies entirely within the sphere centred on its
    # center of mass and reaching the inner Lagrangian point
    rsphere = 1 - rl1
    pp = rsphere * rsphere

    return dict(xcm=xcm, c1=c1, c2=c2, rsphere=rsphere, pp=pp, crit=crit)


def _potential(p, cache):
    """
    Calculate Roche Potential at point p
    """
    # calculate roche potential at point p
    xm = p.x - 1
    yy = p.y * p.y
    rr = yy + p.z * p.z
    rs2 = (xm * xm) + rr  # squared dist from centre of donor to p
    xc = p.x - cache["xcm"]
    r1 = jnp.sqrt(p.x * p.x + rr)  # dist from centre of mass to p
    r2 = jnp.sqrt(rs2)  # dist from centre of donor to p
    # guard against division by zero
    return jnp.where(
        r2 == 0, jnp.inf, cache["c1"] / r1 + cache["c2"] / r2 + xc * xc + yy
    )


_gradp = grad(_potential)


def blink(q, position, earth, acc=0.1, cache_precision=10, cache=None):
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
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1
    position: `jax_roche.Vec3`
        The position vector of the point in the binary, scaled by binary separation
    earth: `jax_roche.Vec3`
        A unit vectory pointing towards the Observer (Earth).
        Usually, this is (sini(i)*cos(phi), -sin(i)*sin(phi), cos(i)),
        where phi is the orbital phase and i is the inclination.
    acc: float, `jnp.DeviceArray` (default=0.1)
        Step size parameter.
        acc specifies the size of steps taken when trying to see if the photon path
        goes inside the Roche lobe. The step size is roughly acc times the radius
        of the lobe filling star. This means that the photon path could mistakenly
        not be occulted if it passed less than about (acc**2)/8 of the radius below
        the surface of the Roche lobe. acc of order 0.1 should therefore do the job.
    cache_precision: int
        Number of decimal places to use as key cache. Values of q that agree to within
        this number of decimal places will use cached values for quantities like the
        critical potential instead of recalculating them.
    cache: tuple
        Stored values of quantities like the critical potential

    Returns
    -------
    eclipsed: bool
        True if the point is eclipsed, False if not
    cache: tuple
        Stored values
    """
    if cache is None or cache["qlast"] != q:
        # TODO: use cache_precision to round q to a certain number of decimal places
        cache = calc_cache(q)
        cache["qlast"] = q

    state = dict()

    # step size is roughly acc * Roche lobe radius
    state["step"] = acc * cache["rsphere"]

    #  evaluate closest approach distance to reference sphere.
    dist1, dist2 = intersection_line_sphere(xhat, cache["rsphere"], position, earth)

    state["dist1"], state["dist2"] = dist1, dist2

    return cond(
        (jnp.isnan(dist1) and jnp.isnan(dist2)) or (dist2 <= 0.0),
        lambda *args: 0,
        _blink_step1,
        q,
        position,
        earth,
        state,
        cache,
    )


@jit
def _blink_step1(q, position, earth, state, cache):
    """
    Does the harder work of checking for eclipses: called when the photon enters the reference sphere
    """
    # first rule out an easy case - is the closest approach at the centre of the
    # donor
    state["dist"] = (position - xhat).dot(earth)
    closest = (xhat - position) + state["dist"] * earth
    zero = Vec3(0.0, 0.0, 0.0)
    state["closest"] = closest

    # point at COM of red star, definitely eclipsed.
    # returning true here avoids division by zero later
    return cond(
        closest == zero,
        lambda *args: 1,
        _blink_step2,  # call next step in cascade
        q,
        position,
        earth,
        state,
        cache,
    )


@jit
def _blink_step2(q, position, earth, state, cache):
    # are we deeper in the well than lagrangian? If so eclipsed.
    c = _potential(state["closest"], cache)
    return cond(
        c > cache["crit"],
        lambda *args: 2,
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

    TODO: FINISH!
    """
    # step direction established by evaluating first derivative of R.P at closest approach
    p = state["closest"]
    derivative = _gradp(p, cache).dot(earth)
    p1 = state["dist"]
    # if RP is increasing along LOS towards earth, step towards Earth
    # otherwise, step away
    p2 = jnp.where(derivative > 0.0, state["dist2"], state["dist1"])
    nstep = jnp.floor(0.5 + jnp.fabs(p2 - p1) / state["step"]).astype("int32")
    step_size = (p2 - p1) / nstep
    # loop over nstep, evaluate potential here.
    # if we are above cache['crit'] then we are eclipsed
    return 3
