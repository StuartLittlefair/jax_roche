from jax import jit
from jax.lax import cond
from jax import numpy as jnp


@jit
def sphere_eclipse(earth, position, centre, rsphere):
    """
    Does a sphere eclipse a point or not?

    This will also return the multiplier values giving the cut points.
    These can then be used as starting points for Roche lobe computations.

    Points inside the sphere are regarded as being eclipsed with the lower multiplier set = 0.

    See https://en.wikipedia.org/wiki/Line–sphere_intersection for method

    Parameters
    ----------
    earth: `jax_roche.Vec3`
        A unit vectory pointing towards the Observer (Earth).
    position: `jax_roche.Vec3`
        The position vector of the point in the binary.
    centre: `jax_roche.Vec3`
        The centre of the sphere
    rsphere: float
        The radius of the sphere

    Returns
    -------
    eclipsed: bool
        True if the point is eclipsed, False if not
    lam1: float
        The multiplier giving the first cut point
    lam2: float
        The multiplier giving the second cut point
    """
    # vector from centre of sphere to position
    d = position - centre

    # does line of sight point towards sphere?
    BQUAD = earth.dot(d)

    CQUAD = abs(d) - rsphere * rsphere
    fac = BQUAD * BQUAD - CQUAD

    no_solution = (
        BQUAD >= 0.0  # line of sight points away from sphere, no eclipse
    ) | (
        fac < 0.0  # no solution to quadratic if less than 0.0
    )

    return cond(
        no_solution,
        lambda *args: (False, jnp.nan, jnp.nan),
        _sphere_eclipse_solve,
        BQUAD,
        CQUAD,
        fac,
    )


def _sphere_eclipse_solve(BQUAD, CQUAD, fac):
    """
    Solve the quadratic to find the multipliers giving the cut points.

    Specialisation of NR method to avoid subtraction of similar sized quantities.
    """
    fac = jnp.sqrt(fac)
    lam2 = -BQUAD + fac
    lam1 = CQUAD / lam2
    lam1 = jnp.where(lam1 < 0.0, 0.0, lam1)
    return True, lam1, lam2
