from jax import jit

from .methods import newton


@jit
def xl1(q):
    """
    Find inner lagrangian point xl1/a

    Currently 50x slower than python-wrapped C++ code. But this works with `jax.grad`.

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    Returns
    -------
    xl1_a: `jnp.DeviceArray`
        Distance between primary and inner lagrangian point, scaled by orbital
        separation.
    """

    # setup polynomial coefficients
    mu = q / (1 + q)
    a1 = -1 + mu
    a2 = 2 - 2 * mu
    a3 = a1
    a4 = 1 + 2 * mu
    a5 = -2 - mu
    a6 = 1

    d1 = a2
    d2 = 2 * a3
    d3 = 3 * a4
    d4 = 4 * a5
    d5 = 5 * a6

    x = mu / q

    def f(x):
        return x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1

    def gradf(x):
        return x * (x * (x * (x * d5 + d4) + d3) + d2) + d1

    return newton(f, gradf, x, MAXIT=1000, EPS=1.0e-8)


@jit
def xl11(q, spin):
    """
    Find inner lagrangian point xl1/a allowing for asynchronous rotation of the primary

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    spin: float, `jnp.DeviceArray`
        spin = ratio of angular/orbital frequency of primary

    Returns
    -------
    xl1_a: `jnp.DeviceArray`
        Distance between primary and inner lagrangian point, scaled by orbital
        separation.
    """
    ssq = spin * spin
    # setup polynomial coefficients
    mu = q / (1 + q)
    a1 = -1 + mu
    a2 = 2 - 2 * mu
    a3 = a1
    a4 = ssq + 2 * mu
    a5 = -2 * ssq - mu
    a6 = ssq
    d1 = a2
    d2 = 2 * a3
    d3 = 3 * a4
    d4 = 4 * a5
    d5 = 5 * a6
    x = 1 / (1 + q)

    def f(x):
        return x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1

    def gradf(x):
        return x * (x * (x * (x * d5 + d4) + d3) + d2) + d1

    return newton(f, gradf, x, MAXIT=1000, EPS=1.0e-8)


@jit
def xl12(q, spin):
    """
    Find inner lagrangian point xl1/a allowing for asynchronous rotation of the secondary

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    spin: float, `jnp.DeviceArray`
        spin = ratio of angular/orbital frequency of secondary

    Returns
    -------
    xl1_a: `jnp.DeviceArray`
        Distance between primary and inner lagrangian point, scaled by orbital
        separation.
    """
    ssq = spin * spin
    # setup polynomial coefficients
    mu = q / (1 + q)
    a1 = -1 + mu
    a2 = 2 - 2 * mu
    a3 = -ssq + mu
    a4 = 3 * ssq + 2 * mu - 2
    a5 = 1 - mu - 3 * ssq
    a6 = ssq
    d1 = a2
    d2 = 2 * a3
    d3 = 3 * a4
    d4 = 4 * a5
    d5 = 5 * a6
    x = 1 / (1 + q)

    def f(x):
        return x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1

    def gradf(x):
        return x * (x * (x * (x * d5 + d4) + d3) + d2) + d1

    return newton(f, gradf, x, MAXIT=1000, EPS=1.0e-8)


@jit
def xl2(q):
    """
    Find L2 distance from primary point xl2/a

    L2 is defined here as the point on the side of the primary on the
    same side as the secondary and therefore xl2/a > 1

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    Returns
    -------
    xl2_a: `jnp.DeviceArray`
        Distance between primary and inner lagrangian point, scaled by orbital
        separation.
    """
    # setup polynomial coefficients
    mu = q / (1 + q)
    a1 = -1 + mu
    a2 = 2 - 2 * mu
    a3 = -1 - mu
    a4 = 1 + 2 * mu
    a5 = -2 - mu
    a6 = 1
    d1 = a2
    d2 = 2 * a3
    d3 = 3 * a4
    d4 = 4 * a5
    d5 = 5 * a6
    x = 1.5

    def f(x):
        return x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1

    def gradf(x):
        return x * (x * (x * (x * d5 + d4) + d3) + d2) + d1

    return newton(f, gradf, x, MAXIT=1000, EPS=1.0e-8)


@jit
def xl3(q):
    """
    Find L3 distance from primary point xl2/a

    L2 is defined here as the point on the side of the primary opposite the
    secondary and therefore xl3/a < 0

    Parameters
    ----------
    q: float, `jnp.DeviceArray`
        Mass ratio M2/M1

    Returns
    -------
    xl2_a: `jnp.DeviceArray`
        Distance between primary and inner lagrangian point, scaled by orbital
        separation.
    """
    # setup polynomial coefficients
    mu = q / (1 + q)
    a1 = 1 - mu
    a2 = -2 + 2 * mu
    a3 = 1 - mu
    a4 = 1 + 2 * mu
    a5 = -2 - mu
    a6 = 1
    d1 = a2
    d2 = 2 * a3
    d3 = 3 * a4
    d4 = 4 * a5
    d5 = 5 * a6
    x = -1.0

    def f(x):
        return x * (x * (x * (x * (x * a6 + a5) + a4) + a3) + a2) + a1

    def gradf(x):
        return x * (x * (x * (x * d5 + d4) + d3) + d2) + d1

    return newton(f, gradf, x, MAXIT=1000, EPS=1.0e-8)

