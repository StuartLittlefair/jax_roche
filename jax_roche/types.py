"""
Useful types
"""
from enum import Enum
from jax import tree_util


class Star(Enum):
    PRIMARY = 1
    SECONDARY = 2


# register as an official JAX type by
# telling JAX how to strip down and build up
tree_util.register_pytree_node(
    Star, lambda star: ((star.value,), None), lambda _, values: Star(values[0]),
)
