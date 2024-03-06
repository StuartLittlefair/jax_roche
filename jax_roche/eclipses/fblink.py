from ..vector import xhat, Vec3
from ..potentials import rpot_val
from ..roche_lobes import ref_sphere
from ..methods import brent
from .sphere_eclipse import sphere_eclipse

from jax.lax import cond, while_loop
from jax import numpy as jnp
from jax import jit, grad, vmap


@jit
def fblink(q, position, earth, ffac=1.0, acc=0.0001, star=2, spin=1.0):
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
    # radius of reference sphere and corresponding Roche potential
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
        ~eclipsed,  # not eclipsed
        lambda state: False,
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

    # evaluate potential between end points
    pot = f(0.5 * (state["lam1"] + state["lam2"]))
    f1 = f(state["lam1"])
    f2 = f(state["lam2"])

    # find minimum potential
    def minpot(state):
        minpos = brent(
            f,
            0.5 * (state["lam1"] + state["lam2"]),
            state["lam1"],
            state["lam2"],
            MAXIT=100,
            ACC=state["acc"],
        )
        return f(minpos)

    return cond(
        pot < state["pref"],
        lambda state: True,  # below reference potential
        lambda state: cond(  # not below reference potential, but
            (pot < f1) & (pot < f2),  # below potential at ends
            # hard case - use brent algorithm to compare min to pref
            lambda state: minpot(state) < state["pref"],
            lambda state: False,  # minimum not bracketed, so no eclipse
            state,
        ),
        state,
    )
