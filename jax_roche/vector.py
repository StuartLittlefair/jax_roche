"""
A Cartesian vector class that JAX understands

Based on https://github.com/google/jax/blob/63c06ef77e84bb5b3582fe23b17d8dfd2f5ecd0c/tests/custom_object_test.py
"""
import jax.numpy as jnp
from jax import tree_util
from jax import jit


class Vec3:
    def __init__(self, x=0, y=0, z=0):
        # TODO: properly handle shapes of inputs with error
        self.x = x
        self.y = y
        self.z = z

    @property
    def shape(self):
        return self.x.shape if hasattr(self.x, "shape") else ()

    @jit
    def __imul__(self, other):
        self.x *= other
        self.y *= other
        self.z *= other
        return self

    @jit
    def __mul__(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    @jit
    def __itruediv__(self, other):
        self.x /= other
        self.y /= other
        self.z /= other
        return self

    @jit
    def __truediv__(self, other):
        return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)

    @jit
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    @jit
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @jit
    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

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


tree_util.register_pytree_node(
    Vec3,
    lambda vec: ((vec.x, vec.y, vec.z), None),
    lambda _, xvec: Vec3(xvec[0], xvec[1], xvec[2]),
)
