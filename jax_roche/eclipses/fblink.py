from ..vector import xhat, Vec3
from ..lagrangian_points import xl1
from ..potentials import rpot_val
from ..roche_lobes import ref_sphere
from .sphere_eclipse import sphere_eclipse

from functools import partial
from jax.lax import cond, while_loop
from jax import numpy as jnp
from jax import jit, grad, vmap


def fblink(q, position, earth, ffac=1.0, acc=0.0001, star=2, spin=1):
    """
    Computes whether a point in a semi-detached binary is eclipsed or not.

    Parameters
    ----------
    q: float
        Mass ratio M2/M1
    position: `jax_roche.Vec3`
        The position vector of the point in the binary.
    earth: `jax_roche.Vec3`
        A unit vectory pointing towards the Observer (Earth).
    ffac: float (default=1.0)
        Filling factor of the star
    acc: float (default=0.0001)
        accuracy of location of minimum potential, units of separation.
        The accuracy in height relative to the Roche potential is acc*acc/(2*R)
        where R is the radius of curvature of the Roche potential surface,
        so don't be too picky. 1.e-4 would be more than good enough in most cases.
    star: int (default=2)
        Which star to consider. 1 for the primary, 2 for the secondary.
    spin: int (default=1)
        Ratio of spin to orbital frequency.

    Returns
    -------
    eclipsed: bool
        True if the point is eclipsed, False if not
    """
    # radius and potential of reference sphere
    rref, pref = ref_sphere(q, star, spin, ffac)
    # centre of mass of star
    cofm = jnp.where(star == 1, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
    cofm = Vec3(*cofm)

    # multipliers cutting the reference sphere (if any)
    eclipsed, lam1, lam2 = sphere_eclipse(earth, position, cofm, rref)

    # evaluate closest approach distance to reference sphere.
    # dist1 is smaller, dist2 is larger
    # if both nan, then the line does not intersect the sphere

    state = dict(
        lam1=lam1,
        lam2=lam2,
        position=position,
        earth=earth,
        rref=rref,
        pref=pref,
        acc=acc,
        star=star,
        spin=spin,
        ffac=ffac,
        q=q,
    )

    return cond(
        eclipsed,  # not eclipsed
        lambda state: state["lam1"] == 0,  # inside sphere
        _step_and_solve,
        state,
    )


def _step_and_solve(state):

    def f(lam):
        # function to find potential at point on line
        return rpot_val(
            state["q"],
            state["star"],
            state["spin"],
            state["earth"],
            state["position"],
            lam,
        )

    df = grad(f)
