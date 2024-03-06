from jax import jit
from jax.lax import cond
from jax import numpy as jnp


@jit
def sphere_eclipse_any(cosi, sini, position, centre, rsphere):
    """
    Does a sphere eclipse a point at any phase?

    If the answer is yes, it will also return with four parameters
    to define the phase range and the multiplier range delimiting the
    region within which the spheres surface is crossed. These can be used
    for later computation.

    The multiplier must be positive: in other words the routine does not
    project backwards. If the point in inside the sphere, phi1=0.0 and
    phi2=1.0, lam1 = 0, and lam2 = the largest value of the multiplier lambda.

    Parameters
    ----------
    cosi: float
        The cosine of the inclination angle
    sini: float
        The sine of the inclination angle
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
    phi1: float
        The phase giving the first cut point.
        This is the start of the phase range (0-1) where the point is
        eclipsed. If the point is inside the sphere, phi1=0.0
    phi2: float
        The phase giving the second cut point.
        This is the end of the phase range (0-1) where the point is
        eclipsed. If the point is inside the sphere, phi2=1.0
    lam1: float
        The multiplier giving the first cut point (>=0). A point in the
        binary system is defined as the position vector plus the multiplier
        times the line of sight vector towards Earth.

        If the point is inside the sphere, lam1=0.0
    lam2: float
        The multiplier giving the second cut point (>lam1). A point in the
        binary system is defined as the position vector plus the multiplier
        times the line of sight vector towards Earth.

        If the point is inside the sphere, lam2 is the largest value of
        the multiplier lambda.
    """
    # vector from centre of sphere to position
    d = position - centre

    # distance from origin in X-Y plane
    PDIST = jnp.sqrt(d.x**2 + d.y**2)

    """
    This is half the minimum value of the linear coefficient in the 
    quadratic that determines whether an intersection occurs. We use 
    the minimum to increase chance of positive roots for the multiplier.
    """
    BQUAD = d.z * cosi - PDIST * sini

    CQUAD = abs(d) - rsphere * rsphere
    fac = BQUAD * BQUAD - CQUAD

    no_solution = (
        BQUAD >= 0.0  # line of sight points away from sphere, no eclipse
    ) | (
        fac < 0.0  # no solution to quadratic if less than 0.0
    )

    # start conditional path
    return cond(
        no_solution,
        lambda *args: (False, jnp.nan, jnp.nan, jnp.nan, jnp.nan),
        _sphere_eclipse_any_case1,
        *(d, cosi, sini, fac, BQUAD, CQUAD, PDIST),
    )


def _sphere_eclipse_any_case1(d, cosi, sini, fac, BQUAD, CQUAD, PDIST):
    # inside here we have a solution
    fac = jnp.sqrt(fac)
    # Specialisation of NR method to avoid
    # subtraction of similar sized quantities.
    lam2 = -BQUAD + fac
    val = CQUAD / lam2
    lam1 = jnp.where(val < 0.0, 0.0, val)

    # now compute the phase range where the eclipse occurs
    def _compute_phases(d, cosi, sini, CQUAD, PDIST):
        delta = jnp.arccos(cosi * d.z - jnp.sqrt(CQUAD)) / (sini * PDIST)
        phi = jnp.arctan2(d.y, -d.x)
        phi1 = (phi - delta) / (2.0 * jnp.pi)
        phi1 -= jnp.floor(phi1)
        phi2 = phi1 + 2.0 * delta / (2.0 * jnp.pi)
        return phi1, phi2

    phi1, phi2 = cond(
        CQUAD < 0.0,
        lambda *args: (0.0, 1.0),
        _compute_phases,
        d,
        cosi,
        sini,
        CQUAD,
        PDIST,
    )
    return True, phi1, phi2, lam1, lam2


@jit
def sphere_eclipse(earth, position, centre, rsphere):
    """
    Does a sphere eclipse a point at a given phase?

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
