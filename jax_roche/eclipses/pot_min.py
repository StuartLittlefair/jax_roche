"""
In this module we implement routines to find if a point in the binary is eclipsed
by the Roche Lobe, given a range of possible orbital phases ϕ and multipliers λ.

For a point of origin O in the binary, the line of sight to the observer is described
by P = O + λ * E, where E is the unit vector pointing to the observer. For an inclination
i, E = [sin(i)cos(ϕ), -sin(i)sin(ϕ), cos(i)].

Therefore, for some limits ϕ1, ϕ2 and λ1, λ2, the line of sight to the observer sweeps
out a cone. The routines below consider the Roche potential as a function of ϕ and λ,
and ask the question "does any region in (ϕ, λ) space drop below the critical potential?
"""

from collections import namedtuple
from jax import grad, jit, vmap
from jax.lax import cond, while_loop
from jax import numpy as jnp

from ..vector import Vec3
from ..potentials import rpot1, rpot2


# @partial(jit, static_argnames=["extras"]) # needs hash of vec3 to work
@jit
def rpot_philam(params, extras):
    """
    Parameterisation of roche potential in terms of ϕ and λ.

    This is used to search for the minimum potential across a range of phases and
    multipliers.

    Parameters
    ----------
    params : tuple, jnp.ndarray
        Array of (ϕ, λ)
    extras : tuple, jnp.ndarray
        Array of (q, star, spin, cosi, sini, origin)
    """
    phi, lam = params
    phi_r = 2 * jnp.pi * phi
    cosp = jnp.cos(phi_r)
    sinp = jnp.sin(phi_r)
    q, star, spin, cosi, sini, origin = extras
    earth = Vec3(sini * cosp, -sini * sinp, cosi)
    return cond(star == 1, rpot1, rpot2, q, spin, origin + earth * lam)


# partial derivative of rpot_philam with respect to ϕ and λ
drpot = grad(rpot_philam, argnums=0)


PotMinState = namedtuple(
    "PotMinState",
    [
        "direction",
        "pmin",
        "pot",
        "params",
        "gradient",
        "bounds",
        "H_k",
        "jammed",
        "n_iters",
    ],
)


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
        The phase giving the minimum potential found.
        Ingress occurs between phi1 and phi if there is an eclipse. Egress
        occurs between phi and phi2.
    lam: float
        lambda at minimum potential found.
    """
    phi = (phi1 + phi2) / 2.0
    lam = (lam1 + lam2) / 2.0
    rp = 2.0 * jnp.pi * phi
    cosp, sinp = jnp.cos(rp), jnp.sin(rp)

    # these do not vary
    extras = (q, star, spin, cosi, sini, position)
    bounds = jnp.array([[phi1, lam1], [phi2, lam2]])

    # initial potential
    params = jnp.array([phi, lam])
    pot = rpot_philam(params, extras)

    # already below critical potential?

    MAXITER = 200
    # given ccuracy in position corresponds to an accuracy in potential
    acc_pot = q / (1.0 + q) * (acc / rref) ** 2 / 2.0

    # now perform the minimisation, choosing a search direction and
    # carrying out line minimisation in that direction at each step
    def _loop_cond(state):
        # found an eclipsing point
        cond1 = state.pmin < pref
        # stuck on boundary or potential not decreasing any more (no eclipse)
        cond2 = state.jammed | jnp.fabs(state.pmin - state.pot) < acc_pot
        # gradient is zero (no eclipse)
        cond3 = jnp.linalg.norm(state.gradient) == 0.0
        # too many iterations
        cond4 = state.n_iters >= MAXITER
        # carry on if all False
        return ~(cond1 | cond2 | cond3 | cond4)

    def _loop_body(state):
        # perform line search for the minimum potential along best direction
        pmin, lammin, jammed = linmin(
            state.params, state.direction, state.bounds, extras
        )

        # calculate the position and potential at new position
        params = jnp.array([phi, lammin])
        pot = rpot_philam(params, extras)
        gradient = drpot(params, extras)
        n_iters = state.n_iters + 1

        # bfgs update to direction (see eq 6.17 Nocedal and Wright)
        yk = (gradient - state.gradient)[..., None]
        sk = (params - state.params)[..., None]
        # scale
        rho_k = 1.0 / (yk.T @ sk)
        update = jnp.eye(2) - rho_k * sk @ yk.T
        new_H_k = update @ state.H_k @ update.T + rho_k * sk @ sk.T
        direction = -new_H_k @ gradient

        # update the state
        state = state._replace(
            direction=direction,
            pmin=pmin,
            pot=pot,
            params=params,
            gradient=gradient,
            H_k=new_H_k,
            jammed=jammed,
            n_iters=n_iters,
        )
        return state

    def _perform_minimisation(state):
        state = while_loop(_loop_cond, _loop_body, state)
        # TODO: JAMMED ETC
        eclipsed = state.pmin < pref
        phi, lam = state.params
        return eclipsed, phi, lam

    state = PotMinState(
        direction=drpot(params, extras),
        pmin=pot,
        pot=pot,
        params=params,
        gradient=drpot(params, extras),
        H_k=jnp.eye(2),
        bounds=bounds,
        jammed=False,
        n_iters=0,
    )
    return cond(
        pot < pref, lambda state: (True, phi, lam), _perform_minimisation, state
    )


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
