"""Protocol type definition for Rax callables."""

from typing import Optional, Tuple, Union
import jax.numpy as jnp

# Protocol is a python 3.8+ feature. For older versions, we can use
# typing_extensions, which provides the same functionality.
try:
  from typing import Protocol  # pylint: disable=g-import-not-at-top
except ImportError:
  from typing_extensions import Protocol  # pylint: disable=g-import-not-at-top


class RankFn(Protocol):
  """Computes the ranks for the given scores."""

  def __call__(self, scores: jnp.ndarray, mask: Optional[jnp.ndarray],
               rng_key: Optional[jnp.ndarray]) -> jnp.ndarray:
    pass


class CutoffFn(Protocol):
  """Computes an indicator that selects the `n` largest values of `a`."""

  def __call__(self, a: jnp.ndarray, n: Optional[int]) -> jnp.ndarray:
    pass


class ReduceFn(Protocol):
  """Reduces a tensor across one or more dimensions."""

  def __call__(self, a: jnp.ndarray, where: Optional[jnp.ndarray],
               axis: Optional[Union[int, Tuple[int, ...]]]) -> jnp.ndarray:
    pass


class LossFn(Protocol):
  """A Rax loss function."""

  def __call__(self, scores: jnp.ndarray, labels: jnp.ndarray, *,
               mask: Optional[jnp.ndarray], **kwargs) -> jnp.ndarray:
    pass


class MetricFn(Protocol):
  """A Rax metric function."""

  def __call__(self, scores: jnp.ndarray, labels: jnp.ndarray, *,
               mask: Optional[jnp.ndarray], **kwargs) -> jnp.ndarray:
    pass
