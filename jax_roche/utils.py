from .vector import set_earth, Vec3, xhat, yhat, zhat
from .eclipses import fblink

from jax import jit
from jax.lax import cond, while_loop
from jax import numpy as jnp


@jit
def findi(q, deltaphi, acc=1.0e-4, di=1.0e-5):
    """
    computes inclination for a given mass ratio and phase width.

    Parameters
    ----------
    q : float
        mass ratio
    deltaphi : float
        phase width
    acc : float, optional
        accuracy of the result, by default 1.e-4
    di : float, optional
        accuracy in the result. Between 0 and 10, by default 1.e-5
    """
    i_low, i_high = 65.0, 90.0
    # |phase| at ingress and egress
    phi = 0.5 * deltaphi
    earth_low = set_earth(i_low, phi)
    earth_high = set_earth(i_high, phi)
    origin = Vec3(0.0, 0.0, 0.0)
    # check if eclipse occurs at the lower and upper limits
    eclipse_low = fblink(q, origin, earth_low, acc=acc)
    eclipse_high = fblink(q, origin, earth_high, acc=acc)

    def _search_i(state):
        def loop_cond(state):
            i_low, i_high = state
            return i_high - i_low > di

        def step(state):
            i_low, i_high = state
            iangle = 0.5 * (i_low + i_high)
            phi = 0.5 * deltaphi
            earth = set_earth(iangle, phi)
            eclipse = fblink(q, origin, earth, acc=acc)
            i_low, i_high = jnp.where(
                eclipse, jnp.array([i_low, iangle]), jnp.array([iangle, i_high])
            )
            return i_low, i_high

        state = while_loop(loop_cond, step, state)
        return 0.5 * (state[0] + state[1])

    state = (i_low, i_high)
    iangle = cond(
        eclipse_low & eclipse_high,
        lambda state: -2.0,
        lambda state: cond(
            ~eclipse_low & ~eclipse_high,
            lambda state: -1.0,
            _search_i,
            state,
        ),
        state,
    )
    return iangle


def findphi(q, i):
    raise NotImplementedError


@jit
def findq(iangle, dphi, acc=1.0e-4, dq=1.0e-5, q_low=0.001, q_high=2.0):
    """
    computes mass ratio for a given inclination and phase width.

    Parameters
    ----------
    iangle : float
        inclination
    deltaphi : float
        phase width
    acc : float, optional
        accuracy to use with blink, by default 1.e-4
    dq : float, optional
        accuracy in the result. Between 0 and 0.1, by default 1.e-5
    q_low : float, optional
        lower limit of mass ratio, by default 0.001
    q_high : float, optional
        upper limit of mass ratio, by default 2.0
    """
    # |phase| at ingress and egress
    phi = 0.5 * dphi
    earth = set_earth(iangle, phi)
    origin = Vec3(0.0, 0.0, 0.0)

    # check if eclipse occurs at the lower and upper limits
    eclipse_low = fblink(q_low, origin, earth, acc=acc)
    eclipse_high = fblink(q_high, origin, earth, acc=acc)

    def _search_q(state):
        def loop_cond(state):
            q_low, q_high = state
            return q_high - q_low > dq

        def step(state):
            q_low, q_high = state
            q = 0.5 * (q_low + q_high)
            eclipse = fblink(q, origin, earth, acc=acc)
            q_low, q_high = jnp.where(
                eclipse, jnp.array([q_low, q]), jnp.array([q, q_high])
            )
            return q_low, q_high

        state = while_loop(loop_cond, step, state)
        return 0.5 * (state[0] + state[1])

    state = (q_low, q_high)
    q = cond(
        eclipse_low & eclipse_high,
        lambda state: -2.0,
        lambda state: cond(
            ~eclipse_low & ~eclipse_high,
            lambda state: -1.0,
            _search_q,
            state,
        ),
        state,
    )
    return q


def qirbs(deltaphi, pbi, pbe, ilo=78.0, ihi=90.0, rlo=0.1):
    raise NotImplementedError


def jacobi(q, position, velocity):
    raise NotImplementedError
