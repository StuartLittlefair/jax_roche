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

    def noop(state):
        """
        Does nothing - called when stop_cond is satisfied
        """
        return state

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
    # state, _ = scan(body, state, None, length=MAXIT)
    state = while_loop(stop_cond, take_step, state)
    return state[0]
