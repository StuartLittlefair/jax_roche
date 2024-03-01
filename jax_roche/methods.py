"""
Numerical Methods for root finding and minimisation.

Note to self: we cannot jit functions that accept functions as arguments.
"""

from jax.lax import cond, scan, while_loop
from jax import numpy as jnp
from functools import partial
from jax import jit


@partial(jit, static_argnames=["f", "gradf", "MAXIT", "EPS"])
def newton(f, gradf, x0, MAXIT=100, EPS=1.0e-8):
    """
    Newton-Raphson Solver

    Parameters
    ----------
    f: Callable
        Function f(x) to find root of
    gradf: Callable
        Gives df/dx
    x0: float, `jnp.DeviceArray`
        Initial guess for root
    MAXIT: int
        Maximum number of iterations
    EPS: float, `jnp.DeviceArray`
        Desired relative precision in root

    Returns
    -------
    x: `jnp.DeviceArray`
        Root of f, so that f(x) = 0
    """

    def stop_cond(state):
        x, xold, n = state
        return (n < MAXIT) & (jnp.fabs(x - xold) > EPS * jnp.fabs(x))

    def step(state):
        x, xold, n = state
        xold = x
        fx = f(x)
        dfx = gradf(x)
        step = fx / dfx
        new_state = x - step, xold, n + 1
        return new_state

    state = (x0, 0.0, 0)
    state = while_loop(stop_cond, step, state)
    return state[0]


@partial(jit, static_argnames=["f", "gradf", "MAXIT", "ACC"])
def rtsafe(f, gradf, lower, upper, MAXIT=100, ACC=1.0e-7):
    """
    Safe root-finding algorithm.

    Switches between Newton and Bisection as appropriate
    """

    def newton_step(state):
        """
        A Newton-Raphson step
        """
        # unpack state
        x, xold, dx, dxold, n, xh, xl = state
        xold = x
        dxold = dx

        # make step
        fx = f(x)
        dfx = gradf(x)
        dx = fx / dfx
        x = x - dx

        # update new state
        fx = f(x)
        xl = jnp.where(fx < 0.0, x, xl)
        xh = jnp.where(fx < 0.0, xh, x)

        new_state = x, xold, dx, dxold, n + 1, xh, xl
        return new_state

    def bisect_step(state):
        """
        A Bisect step
        """
        # unpack state
        x, xold, dx, dxold, n, xh, xl = state
        xold = x
        dxold = dx

        # make step
        dx = 0.5 * (xh - xl)
        x = xl + dx
        fx = f(x)

        # update new state
        below = fx < 0.0
        xl = jnp.where(below, x, xl)
        xh = jnp.where(below, xh, x)

        new_state = x, xold, dx, dxold, n + 1, xh, xl
        return new_state

    def choose_step(state):
        """
        Choose between Newton and Bisection
        """
        x, _, _, dxold, _, xh, xl = state
        fx = f(x)
        dfx = gradf(x)
        # newton would take us outside range = use bisection
        conditionA = ((x - xh) * dfx - fx) * ((x - xl) * dfx - fx) >= 0.0
        # not going fast enough - use bisection
        conditionB = jnp.fabs(2.0 * fx) > jnp.fabs(dxold * dfx)
        return conditionA | conditionB

    def take_step(state):
        """
        A single optimisation step

        Chooses between N-R and Bisection depending on the position
        """
        return cond(choose_step(state), bisect_step, newton_step, state)

    def stop_cond(state):
        """
        When we have exceeded MAXIT or ACC is exceeded, switch to noop
        """
        x, xold, _, _, n, _, _ = state
        return (n < MAXIT) & (jnp.fabs(x - xold) > ACC * jnp.fabs(x))

    # setup

    # make sure lower and upper limits are the write way around
    # depending on wether f(lower) is +ve or not
    fl = f(lower)
    # xl, xh = cond(
    #    fl < 0.0, lambda x: (x[0], x[1]), lambda x: (x[1], x[0]), (lower, upper)
    # )
    xl = jnp.where(fl < 0.0, lower, upper)
    xh = jnp.where(fl < 0.0, upper, lower)

    # make an initial guess
    x0 = 0.5 * (lower + upper)
    dx = dxold = jnp.fabs(upper - lower)
    state = (x0, 0.0, dx, dxold, 0, xh, xl)
    state = while_loop(stop_cond, take_step, state)
    return state[0]


@partial(jit, static_argnames=["f", "MAXIT", "ACC"])
def brent(f, start, lower, upper, MAXIT=100, ACC=1.0e-7):
    """
    Safe minimisation algorithm.

    Switches between SPI and golden section as appropriate
    """
    rho = 0.61803399
    one_min_rho = 1.0 - rho

    def loop_cond(state):
        a, b, x, _, _, _, _, _, _, niter, _, _, tol2, _, xmid = state
        return (niter < MAXIT) & (jnp.fabs(x - xmid) > tol2 - 0.5 * (b - a))

    def initial_calcs(state):
        _, _, x, w, v, fx, fw, fv, deltax, _, _, _, _, ratio, _ = state
        tmp1 = (x - w) * (fx - fv)
        tmp2 = (x - v) * (fx - fw)
        p = (x - v) * tmp2 - (x - w) * tmp1
        tmp2 = 2.0 * (tmp2 - tmp1)
        p = jnp.where(tmp2 > 0.0, -p, p)
        tmp2 = jnp.fabs(tmp2)
        dx_temp = deltax
        deltax = ratio
        return tmp2, p, dx_temp, deltax

    def choose_step(state):
        """
        Choose between SPI and Golden Section
        """
        a, b, x, _, _, _, _, _, deltax, _, _, tol1, _, _, _ = state
        # if true, take golden step
        condition = jnp.fabs(deltax) <= tol1

        def careful_check(state):
            tmp2, p, dx_temp, _ = initial_calcs(state)
            condition = (
                (p <= tmp2 * (a - x))
                | (p >= tmp2 * (b - x))
                | (jnp.fabs(p) >= jnp.fabs(0.5 * tmp2 * dx_temp))
            )
            return condition  # if False, SPI step is useful

        return cond(
            condition,  # if true, take golden step, otherwise check carefully
            lambda state: True,
            careful_check,
            state,
        )

    def golden_step(state):
        """
        Golden section step, updates deltax, ratio
        """
        a, b, x, w, v, fx, fw, fv, deltax, niter, nfeval, tol1, tol2, ratio, xmid = (
            state
        )
        deltax = jnp.where(x >= xmid, a - x, b - x)
        ratio = rho * deltax
        state = (
            a,
            b,
            x,
            w,
            v,
            fx,
            fw,
            fv,
            deltax,
            niter,
            nfeval,
            tol1,
            tol2,
            ratio,
            xmid,
        )
        return common_step(state)

    def spi_step(state):
        """
        SPI part of step, just updates deltax, ratio
        """
        a, b, x, w, v, fx, fw, fv, deltax, niter, nfeval, tol1, tol2, ratio, xmid = (
            state
        )
        tmp2, p, _, deltax = initial_calcs(state)
        ratio = p / tmp2
        u = x + ratio
        ratio = jnp.where(
            ((u - a) < tol2) | ((b - u) < tol2), jnp.sign(xmid - x) * tol1, ratio
        )
        state = (
            a,
            b,
            x,
            w,
            v,
            fx,
            fw,
            fv,
            deltax,
            niter,
            nfeval,
            tol1,
            tol2,
            ratio,
            xmid,
        )
        return common_step(state)

    def common_step(state):
        a, b, x, w, v, fx, fw, fv, deltax, niter, nfeval, tol1, tol2, ratio, xmid = (
            state
        )
        # update by at least tol1
        u = jnp.where(jnp.fabs(ratio) < tol1, x + jnp.sign(ratio) * tol1, x + ratio)
        fu = f(u)
        nfeval += 1

        def stepA(vals):
            a, b, v, w, x, fv, fw, fx = vals
            # a = jnp.where(u < x, u, a)
            # b = jnp.where(u < x, b, u)
            a, b = jnp.where(u < x, jnp.array([u, b]), jnp.array([a, u]))

            cond1 = (fu <= fw) | (w == x)
            # v = jnp.where(cond1, w, v)
            # w = jnp.where(cond1, u, w)
            # fv = jnp.where(cond1, fw, fv)
            # fw = jnp.where(cond1, fu, fw)
            v, w, fv, fw = jnp.where(
                cond1, jnp.array([w, u, fw, fu]), jnp.array([v, w, fv, fw])
            )

            cond2 = (fu <= fv) | (v == x) | (v == w)
            # v = jnp.where(cond2, u, v)
            # fv = jnp.where(cond2, fu, fv)
            v, fv = jnp.where(cond2, jnp.array([u, fu]), jnp.array([v, fv]))
            return a, b, v, w, x, fv, fw, fx

        def stepB(vals):
            a, b, v, w, x, fv, fw, fx = vals
            # a = jnp.where(u >= x, x, a)
            # b = jnp.where(u >= x, b, x)
            a, b = jnp.where(u >= x, jnp.array([x, b]), jnp.array([a, x]))
            return a, b, w, x, u, fw, fx, fu

        a, b, v, w, x, fv, fw, fx = cond(
            fu > fx, stepA, stepB, (a, b, v, w, x, fv, fw, fx)
        )

        tol1 = ACC * jnp.fabs(x)
        tol2 = 2.0 * tol1
        return (
            a,
            b,
            x,
            w,
            v,
            fx,
            fw,
            fv,
            deltax,
            niter + 1,
            nfeval,
            tol1,
            tol2,
            ratio,
            0.5 * (a + b),
        )

    def take_step(state):
        """
        A single optimisation step

        Chooses between N-R and Bisection depending on the position
        """
        return cond(choose_step(state), golden_step, spi_step, state)

    # setup
    xa = lower
    xb = start
    xc = upper
    fb = f(xb)
    x = w = v = xb
    fw = fv = fx = fb
    # ensure a consistent ordering of xa, xb, xc
    a = jnp.where(xa < xc, xa, xc)
    b = jnp.where(xa < xc, xc, xa)
    niter = 0
    nfeval = 1
    deltax = 0.0
    xmid = 0.5 * (a + b)
    ratio = 0.0

    tol1 = ACC * jnp.fabs(x)
    tol2 = 2.0 * tol1
    state = (a, b, x, w, v, fx, fw, fv, deltax, niter, nfeval, tol1, tol2, ratio, xmid)

    state = while_loop(loop_cond, take_step, state)
    return 0.5 * (state[0] + state[1])
