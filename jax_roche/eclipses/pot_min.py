def pot_min(
    q, cosi, sini, star, spin, position, phi1, phi2, lam1, lam2, rref, pref, acc
):
    """
    Find minimum potential across a range of phases and points within the binary.

    The line of sight to any fixed point in a binary sweeps out a cone as the binary
    rotates. Positions on the cone can be parameterised by the orbital phase phi and
    the multiplier ('lambda') needed to get from a fixed point along the line of sight.
    The question pot_min tries to solve is "does the cone intersect a surface of fixed
    Roche potential lying within a Roche lobe?". It does so by minimisation over a
    region of phi and lambda. It stops as soon as any potential below a critical value
    is found. The initial range of phi and lambda can be determined using sphere_eclipse
    which calculates them for a sphere.

    Parameters
    ----------
    q : float
        mass ratio
    cosi : float
        The cosine of the inclination angle
    sini : float
        The sine of the inclination angle
    star : int
        which star, primary (1) or secondary(2), is doing the eclipsing.
    spin : float
        Ratio of spin to orbital frequency.
    position : `jax_roche.Vec3`
        The position vector of the point of origin in the binary.
    phi1 : float
        The phase giving the first cut point.
    phi2 : float
        The phase giving the second cut point.
    lam1 : float
        The multiplier giving the first cut point (>=0).
    lam2 : float
        The multiplier giving the second cut point (>lam1).
    rref : float
        The radius of the reference sphere.
    pref : float
        The Roche potential we want to be below.
    acc : float
        Desireed accuracy in position.

    Returns
    -------
    eclipsed: bool
        True if potential drops below pref, ie point is eclipsed.
    phi: float
        The phase giving the minimum potential.
        Ingress occurs between phi1 and phi if there is an eclipse. Egress
        occurs between phi and phi2.
    lam: float
        lambda at minimum potential
    """
    raise NotImplementedError


def linmin():
    """
    linmin minimises the roche potential along a line in phase, lambda space.

    A range of points in the binary can be specified with a starting point, a
    vector pointing to the observer (earth), and a range of multipliers along
    that line of sight.

    For fixed inclination, the lines of sight to the observer at at range of phases
    can be thought of as a line in phase (phi) and multiplier (lambda) space. linmin
    finds the minimum potential along that line.

    It returns at the minimum or as soon as the potential drops below a reference value.
    It is assumed that the potential is dropping with x at the starting point.
    """
    raise NotImplementedError
