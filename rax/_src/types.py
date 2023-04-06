# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rax-specific types and protocols.

.. note::

  Types and protocols are provided for **type-checking** convenience only. You
  do **not** need to instantiate, subclass or extend them.
"""

from typing import Optional, Tuple, Union
import jax

# Protocol is a python 3.8+ feature. For older versions, we can use
# typing_extensions, which provides the same functionality.
try:
  from typing import Protocol  # pylint: disable=g-import-not-at-top
except ImportError:
  from typing_extensions import Protocol  # pylint: disable=g-import-not-at-top

# Type alias for a JAX array.
Array = jax.numpy.ndarray


class RankFn(Protocol):
  """:class:`typing.Protocol` for rank functions."""

  def __call__(
      self,
      scores: Array,
      where: Optional[Array],
      key: Optional[Array],
      segments: Optional[Array] = None,
  ) -> Array:
    """Computes 1-based ranks based on the given scores.

    Args:
      scores: The scores to compute the 1-based ranks for.
      where: An optional :class:`jax.numpy.ndarray` of the same shape as ``a``
        that indicates which elements to rank. Other elements will be ranked
        last.
      key: An optional :func:`~jax.random.PRNGKey` used for random operations.
      segments: An optional :class:`jax.numpy.ndarray` of the same shape as
        ``a`` that indicates which elements to group together.

    Returns:
      A :class:`jax.numpy.ndarray` of the same shape as ``scores`` that
      represents the 1-based ranks.
    """
    pass


class CutoffFn(Protocol):
  """:class:`typing.Protocol` for cutoff functions."""

  def __call__(
      self, a: Array, n: Optional[int], segments: Optional[Array] = None
  ) -> Array:
    """Computes cutoffs based on the given array.

    Args:
      a: The array for which to compute the cutoffs.
      n: The position of the cutoff.
      segments: An optional :class:`jax.numpy.ndarray` of the same shape as
        ``a`` that indicates which elements to group together.

    Returns:
      A binary :class:`jax.numpy.ndarray` of the same shape as ``a`` that
      represents which elements of ``a`` should be selected for the topn cutoff.
    """
    pass


class ReduceFn(Protocol):
  """:class:`typing.Protocol` for reduce functions."""

  def __call__(self, a: Array, where: Optional[Array],
               axis: Optional[Union[int, Tuple[int, ...]]]) -> Array:
    """Reduces an array across one or more dimensions.

    Args:
      a: The array to reduce.
      where: An optional :class:`jax.numpy.ndarray` of the same shape as ``a``
        that indicates which elements to include in the reduction.
      axis: One or more axes to use for the reduction. If ``None`` this reduces
        across all available axes.

    Returns:
      A :class:`jax.numpy.ndarray` that represents the reduced result of ``a``
      over given ``axis``.
    """
    pass


class LossFn(Protocol):
  """:class:`typing.Protocol` for loss functions."""

  def __call__(self, scores: Array, labels: Array, *, where: Optional[Array],
               **kwargs) -> Array:
    """Computes a loss.

    Args:
      scores: The score of each item.
      labels: The label of each item.
      where: An optional :class:`jax.numpy.ndarray` of the same shape as
        ``scores`` that indicates which elements to include in the loss.
      **kwargs: Optional loss-specific keyword arguments.

    Returns:
      A :class:`jax.numpy.ndarray` that represents the loss computed on the
      given ``scores`` and ``labels``.
    """
    pass


class MetricFn(Protocol):
  """:class:`typing.Protocol` for metric functions."""

  def __call__(self, scores: Array, labels: Array, *, where: Optional[Array],
               **kwargs) -> Array:
    """Computes a metric.

    Args:
      scores: The score of each item.
      labels: The label of each item.
      where: An optional :class:`jax.numpy.ndarray` of the same shape as
        ``scores`` that indicates which elements to include in the metric.
      **kwargs: Optional metric-specific keyword arguments.

    Returns:
      A :class:`jax.numpy.ndarray` that represents the metric computed on the
      given ``scores`` and ``labels``.
    """
    pass


class LambdaweightFn(Protocol):
  """:class:`typing.Protocol` for lambdaweight functions."""

  def __call__(self, scores: Array, labels: Array, *, where: Optional[Array],
               weights: Optional[Array], **kwargs) -> Array:
    """Computes lambdaweights.

    Args:
      scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
        score of each item.
      labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
        relevance label for each item.
      where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
        indicating which items are valid for computing the lambdaweights. Items
        for which this is False will be ignored when computing the
        lambdaweights.
      weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
        indicating the weight for each item.
      **kwargs: Optional lambdaweight-specific keyword arguments.

    Returns:
      A :class:`jax.numpy.ndarray` that represents the lambda weights.
    """
    pass
