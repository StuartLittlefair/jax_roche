from ..vector import xhat, Vec3, set_earth
from ..potentials import rpot_val
from ..roche_lobes import ref_sphere
from ..methods import brent
from .sphere_eclipse import sphere_eclipse_any

from jax.lax import cond, while_loop
from jax import numpy as jnp
from jax import jit, grad, vmap


@jit
def ingress_egress(q, iangle, x, spin=1.0, ffac=1.0, delta=1.0e-7, star=2):
    """
    ingress_egress tests for whether a given point is eclipsed by a Roche-distorted star.

    If it is, it computes the ingress and egress phases using a binary chop.
    The accuracy on the phase should be set to be below the expected
    uncertainties of the phases of your data.

    Parameters
    ----------
    q : float
        mass ratio
    iangle : float
        inclination angle in degrees
    x : Vec3
        position of the point
    spin : float, optional
        ratio of spin to orbit of eclipsing star, by default 1.0
    ffac : float, optional
        filling factor of the star, by default 1.0
    delta : float, optional
        desired accuracy of the result, by default 1.e-7
    star : int, optional
        which star, primary (1) or secondary(2), is doing the eclipsing, by default 2

    Returns
    -------
    eclipsed  : bool
        True if the point is eclipsed
    ingress, egress  : float
        ingress and egress phases. NaN if not eclipsed.
    """
    # radius of reference sphere and corresponding Roche potential
    rref, pref = ref_sphere(q, star, spin, ffac)

    # centre of mass of star
    cofm = jnp.where(star == 1, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
    cofm = Vec3(*cofm)

    # some quantities to allow quick calculation of vector to earth
    ri = jnp.radians(iangle)
    cosi, sini = jnp.cos(ri), jnp.sin(ri)

    # OK - is the point eclipsed by reference sphere?
    eclipsed, phi1, phi2, lam1, lam2 = sphere_eclipse_any(cosi, sini, x, cofm, rref)
    state = dict()
    return cond(
        ~eclipsed,  # not eclipsed
        lambda state: eclipsed,
        phi1,
        phi2,
        lam1,
        lam2,  # return false and nans
        _ingress_egress_step1,
        state,
    )


def _ingress_egress_step1(state):

    # A certain accuracy of position corresponds to an accuracy in phase
    ACC = 2.0 * jnp.sqrt(2 * jnp.pi * (state["lam2"] - state["lam1"]) * state["delta"])
