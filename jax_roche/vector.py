"""
A Cartesian vector class that JAX understands

Based on https://github.com/google/jax/blob/63c06ef77e84bb5b3582fe23b17d8dfd2f5ecd0c/tests/custom_object_test.py
"""
import jax.numpy as jnp
from jax.lax import cond
from jax import tree_util
from jax import jit


@tree_util.register_pytree_node_class
class Vec3:
    def __init__(self, x=0, y=0, z=0):
        # TODO: properly handle shapes of inputs with error
        self.x = x
        self.y = y
        self.z = z

    def tree_flatten(self):
        """
        Specifies a flattening recipe.

        Returns
        -------
            a pair of an iterable with the children to be flattened recursively,
            and some opaque auxiliary data to pass back to the unflattening recipe.
            The auxiliary data is stored in the treedef for use during unflattening.
            The auxiliary data could be used, e.g., for dictionary keys.
        """
        children = (self.x, self.y, self.z)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def shape(self):
        return self.x.shape if hasattr(self.x, "shape") else ()

    @jit
    def __rmul__(self, other):
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    @jit
    def __mul__(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    @jit
    def __rtruediv__(self, other):
        self.x /= other
        self.y /= other
        self.z /= other
        return self

    @jit
    def __truediv__(self, other):
        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    @jit
    def __radd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    @jit
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @jit
    def __rsub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __eq__(self, other):
        return (self.x == other.x) & (self.y == other.y) & (self.z == other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    @jit
    def dot(self, other):
        """
        Returns the scalar or dot product of self with other, a 3-vector
        """
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    @jit
    def cross(self, other):
        """
        Computes the vector or cross product of self with other, a 3-vector
        """
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __abs__(self):
        return self.dot(self)

    def __getitem__(self, i):
        return [self.x, self.y, self.z][i]

    @jit
    def norm(self):
        """
        Returns vector as a unit vector in same direction
        """
        mag = jnp.sqrt(abs(self))
        return self * (1.0 / jnp.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Vec3({self.x}, {self.y}, {self.z})"


xhat = Vec3(1.0, 0.0, 0.0)
yhat = Vec3(0.0, 1.0, 0.0)
zhat = Vec3(0.0, 0.0, 1.0)


@jit
def intersection_line_sphere(centre, radius, origin, dirn):
    """
    Calculate the intersection of a sphere and a line

    See https://en.wikipedia.org/wiki/Line–sphere_intersection for method

    Parameters
    ----------
    centre: Vec3
        The centre of the sphere
    radius: float
        The radius of the sphere
    origin: Vec3
        Origin of the line
    dirn: Vec3
        Unit vector in the direction of the line

    Returns
    -------
    solutions: tuple(float)
        The distance along the line from origin to the intersections between the
        line and the sphere. Can be None if one or no solutions exist
    """
    # intersection of line and sphere is solution to quadratic ax^2 + bx + c
    # this is x = alpha +/- sqrt(delta)
    o_min_c = origin - centre  #  vec from centre of sphere to origin of line
    alpha = -dirn.dot(o_min_c)
    delta = alpha**2 - o_min_c.dot(o_min_c) + radius**2

    def find_soln(alpha, delta):
        # at least one solution, find it
        def caseA(alpha, delta):
            # two solutions
            return (alpha - jnp.sqrt(delta), alpha + jnp.sqrt(delta))

        def caseB(alpha, delta):
            # one solution
            return (alpha, jnp.nan)

        return cond(delta == 0.0, caseB, caseA, alpha, delta)

    return cond(delta < 0.0, lambda *args: (jnp.nan, jnp.nan), find_soln, alpha, delta)
