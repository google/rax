# Copyright 2022 Google LLC.
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

"""Utility functions for Rax."""

import functools

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from rax._src.types import Array


def safe_reduce(a: Array,
                where: Optional[Array] = None,
                reduce_fn: Optional[Callable[..., Array]] = None) -> Array:
  """Reduces the values of given array while preventing NaN in the output.

  For `jnp.mean` reduction, this additionally prevents NaN in the output if all
  elements are masked. This can happen with pairwise losses where there are no
  valid pairs because all labels are the same. In this situation, 0 is returned
  instead.

  When there is no `reduce_fn`, this will set elements of `a` to 0 according to
  the `where` mask.

  Args:
    a: The array to reduce.
    where: Which elements to include in the reduction.
    reduce_fn: The function used to reduce. If None, no reduction is performed.

  Returns:
    The result of reducing the values of `a` using given `reduce_fn`.
  """
  # Reduce values if there is a reduce_fn, otherwise keep the values as-is.
  output = reduce_fn(a, where=where) if reduce_fn is not None else a

  if reduce_fn is jnp.mean:
    # For mean reduction, we have to check whether the input contains any NaN
    # values, to ensure that masked mean reduction does not hide them (see
    # below).
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(a)))

    # The standard jnp.mean implementation returns NaN if `where` is False
    # everywhere. This can happen in our case, e.g. pairwise losses with no
    # valid pairs. Instead, we prefer that the loss returns 0 in these cases.
    # Note that this only hides those NaN values if the input did not contain
    # any NaN values. Otherwise it just returns the output as-is.
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0., output)

  if reduce_fn is None and where is not None:
    # When there is no reduce_fn (i.e. we are returning an unreduced
    # loss/metric), set the values of `a` to 0 for invalid (masked) items.
    # This makes sure that manual sum reduction on an unreduced loss works as
    # expected:
    # `jnp.sum(loss_fn(reduce_fn=None)) == loss_fn(reduce_fn=jnp.sum)`
    output = jnp.where(where, output, 0.)

  return output


def normalize_probabilities(unscaled_probabilities: Array,
                            where: Optional[Array] = None,
                            axis: int = -1) -> Array:
  """Normalizes given unscaled probabilities so they sum to one in given axis.

  This will scale the given unscaled probabilities such that its valid
  (non-masked) elements will sum to one along the given axis. Note that the
  array should have only non-negative elements.

  For cases where all valid elements along the given axis are zero, this will
  return a uniform distribution over those valid elements.

  For cases where all elements along the given axis are invalid (masked), this
  will return a uniform distribution over those invalid elements.

  Args:
    unscaled_probabilities: The probabilities to normalize.
    where: The mask to indicate which entries are valid.
    axis: The axis to normalize on.

  Returns:
    Given unscaled probabilities normalized so they sum to one for the valid
    (non-masked) items in the given axis.
  """
  if where is None:
    where = jnp.ones_like(unscaled_probabilities, dtype=jnp.bool_)

  # Sum only the valid items across the given axis.
  unscaled_probabilities_sum = jnp.sum(
      unscaled_probabilities, axis=axis, keepdims=True, where=where)

  # Sum mask for correctly normalizing all-zero elements with mask.
  where_sum = jnp.sum(where, axis=axis, keepdims=True)
  where_sum = jnp.where(where_sum == 0.,
                        jnp.sum(jnp.ones_like(where), axis=axis, keepdims=True),
                        where_sum)

  # Compute output
  return jnp.where(
      unscaled_probabilities_sum != 0.,
      unscaled_probabilities / unscaled_probabilities_sum,
      jnp.ones_like(where, dtype=unscaled_probabilities.dtype) / where_sum)


def logcumsumexp(x: Array,
                 *,
                 axis: int = -1,
                 where: Optional[Array] = None,
                 reverse: bool = False):
  """Computes the cumulative logsumexp.

  This is a numerically safe and efficient implementation of a cumulative
  logsumexp operation.

  Args:
    x: The :class:`~jax.numpy.ndarray` to compute the cumulative logsumexp for.
    axis: The axis over which the cumulative sum should take place.
    where: An optional :class:`~jax.numpy.ndarray` of the same shape as ``x``
      indicating which items are valid for computing the cumulative logsumexp.
    reverse: Whether to compute the cumulative sum in reverse.

  Returns:
    An :class:`~jax.numpy.ndarray` of the same shape as ``x`` representing the
    cumulative logsumexp of the values of ``x``.
  """
  if where is None:
    where = jnp.ones_like(x, dtype=jnp.bool_)

  # Flip the inputs if the cumulative sum needs to be in reverse.
  if reverse:
    x = jnp.flip(x, axis=axis)
    where = jnp.flip(where, axis=axis)

  # Mask out invalid entries so they do not count in the summation.
  x = jnp.where(where, x, -jnp.inf)

  # Make the axis on which to perform the operation the leading axis which is
  # necessary for `jax.lax.scan` used below.
  x = jnp.swapaxes(x, axis, 0)
  where = jnp.swapaxes(where, axis, 0)

  # Compute cumulative maximum.
  m = jax.lax.cummax(x, axis=0)

  # Compute `exp(x_i - m_i)` for each i.
  x_shifted = jnp.exp(x - m)
  x_shifted = jnp.where(where, x_shifted, 0.)

  # Compute `exp(m_{i-1} - m_i)` for each i. This is used to perform an
  # efficient version of the internal cumulative sumation (see below).
  # Note that `m_{i-1} <= m_i` for all i because m_i is a cumulative maximum, so
  # this is numerically safe.
  m_diffs = jnp.exp(jnp.minimum(0., jnp.roll(m, 1, axis=0) - m))
  m_diffs = jnp.where(where, m_diffs, 1.)

  # We wish to compute the following output values (for each i):
  #
  #   out[i] = sum_{j=1}^{i} exp(x_j - m_i)
  #
  # This can be implemented in a vectorized way using a pairwise broadcasted
  # expansion of x and m and computing all pairs `exp(x_j - m_i)` with
  # appropriating masking. This approach would have an O(n^2) complexity.
  # The O(n^2) can be avoided by using a recursive definition of out[i]:
  #
  #   out[1] = exp(x_1 - m_1)
  #   out[i] = exp(x_i - m_i) + exp(m_{i-1} - m_i) * out[i-1]
  #
  # This recursive formulation allows for an o(n) complexity implementation
  # using `jax.lax.scan` and is used here.
  #
  # TODO(jagerman): Investigate using log-space summation instead of product.
  def f(previous, x):
    out_i = x[0] + previous * x[1]
    return out_i, out_i

  initial = jnp.zeros(x.shape[1:], dtype=x.dtype)
  out = jax.lax.scan(f, initial, (x_shifted, m_diffs))[1]

  # Compute the log of the cumulative sum and correct for the cumulative
  # maximum shift.
  tiny = jnp.finfo(x.dtype).tiny
  out = jnp.log(out + tiny) + m

  # Swap axes back and flip output if the cumulative sum needs to be in reverse.
  out = jnp.swapaxes(out, 0, axis)
  if reverse:
    out = jnp.flip(out, axis=axis)

  return out


def sort_by(scores: Array,
            tensors_to_sort: Sequence[Array],
            axis: int = -1,
            where: Optional[Array] = None,
            key: Optional[Array] = None) -> Sequence[Array]:
  """Sorts given list of tensors by given scores.

  Each of the entries in the `tensors_to_sort` sequence must be a tensor that
  matches the shape of the `scores`.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score for each item.
    tensors_to_sort: A sequence of tensors of the same shape as scores. These
      are the tensors that will be sorted in the order of scores.
    axis: The axis to sort on, by default this is the last axis.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid. Invalid entries are pushed to the end.
    key: An optional :func:`jax.random.PRNGKey`. If provided, ties will be
      broken randomly using this key. If not provided, ties will retain the
      order of their appearance in the `scores` array.

  Returns:
    A tuple containing the tensors of ``tensors_to_sort`` sorted in the order of
    ``scores``.
  """
  # Sets up the keys we want to sort on.
  sort_operands = []
  if where is not None:
    sort_operands.append(jnp.logical_not(where))
  sort_operands.append(-scores)
  if key is not None:
    sort_operands.append(jax.random.uniform(key, scores.shape))
  num_keys = len(sort_operands)

  # Adds the tensors that we want sort.
  sort_operands.extend(tensors_to_sort)

  # Performs sort and returns the sorted tensors.
  sorted_values = jax.lax.sort(sort_operands, dimension=axis, num_keys=num_keys)
  return sorted_values[num_keys:]


def ranks(scores: Array,
          *,
          where: Optional[Array] = None,
          axis: int = -1,
          key: Optional[Array] = None) -> Array:
  """Computes the ranks for given scores.

  Note that the ranks returned by this function are not differentiable due to
  the sort operation having no gradients.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid.
    axis: The axis to sort on, by default this is the last axis.
    key: An optional :func:`jax.random.PRNGKey`. If provided, ties will be
      broken randomly using this key. If not provided, ties will retain the
      order of their appearance in the `scores` array.

  Returns:
    A tensor with the same shape as `scores` that indicates the 1-based rank of
    each item.
  """
  # Construct `arange` tensor and broadcast it to the shape of `scores`.
  arange_broadcast_dims = list(range(len(scores.shape)))
  del arange_broadcast_dims[axis]
  arange = jnp.arange(scores.shape[axis])
  arange = jnp.expand_dims(arange, axis=arange_broadcast_dims)
  arange = jnp.broadcast_to(arange, scores.shape)

  # Perform an argsort on the scores along given axis. This returns the indices
  # that would sort the scores. Note that we can not use the `jnp.argsort`
  # method here as it does not support masked arrays or randomized tie-breaking.
  indices = sort_by(scores, [arange], axis=axis, where=where, key=key)[0]

  # Perform an argsort on the indices to get the 1-based ranks.
  return jnp.argsort(indices, axis=axis) + 1


def approx_ranks(scores: Array,
                 *,
                 where: Optional[Array] = None,
                 key: Optional[Array] = None,
                 step_fn: Callable[[Array], Array] = jax.nn.sigmoid) -> Array:
  """Computes approximate ranks.

  This can be used to construct differentiable approximations of metrics. For
  example:

  >>> import functools
  >>> approx_ndcg = functools.partial(
  ...     rax.ndcg_metric, rank_fn=rax.utils.approx_ranks)
  >>> scores = jnp.asarray([-1., 1., 0.])
  >>> labels = jnp.asarray([0., 0., 1.])
  >>> print(approx_ndcg(scores, labels))
  0.63092977
  >>> print(jax.grad(approx_ndcg)(scores, labels))
  [-0.03763788 -0.03763788  0.07527576]

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid.
    key: An optional :func:`jax.random.PRNGKey`. Unused by ``approx_ranks``.
    step_fn: A callable that approximates the step function ``x >= 0``.

  Returns:
    A :class:`~jax.numpy.ndarray` of the same shape as ``scores``, indicating
    the 1-based approximate rank of each item.
  """

  del key  # unused for approximate ranks.

  score_i = jnp.expand_dims(scores, axis=-1)
  score_j = jnp.expand_dims(scores, axis=-2)
  score_pairs = step_fn(score_j - score_i)

  # Build mask to prevent counting (i == j) pairs.
  triangular_mask = (jnp.triu(jnp.tril(jnp.ones_like(score_pairs))) == 0.0)
  if where is not None:
    where = jnp.expand_dims(where, axis=-1) & jnp.expand_dims(where, axis=-2)
    triangular_mask &= where

  return jnp.sum(score_pairs, axis=-1, where=triangular_mask, initial=1.0)


def cutoff(a: Array,
           n: Optional[int] = None,
           *,
           where: Optional[Array] = None,
           step_fn: Callable[[Array], Array] = lambda x: x >= 0) -> Array:
  """Computes a binary array to select the largest ``n`` values of ``a``.

  This function computes a binary :class:`jax.numpy.ndarray` that selects the
  ``n`` largest values of ``a`` across its last dimension.

  Note that the returned indicator may select more than ``n`` items if ``a`` has
  ties.

  Args:
    a: The :class:`jax.numpy.ndarray` to select the topn from.
    n: The cutoff value. If None, no cutoff is performed.
    where: A mask to indicate which values to include in the topn calculation.
    step_fn: A function that computes ``x >= 0`` or an approximation.

  Returns:
    A :class:`jax.numpy.ndarray` of the same shape as ``a``, where the
    ``n`` largest values are set to 1, and the smaller values are set to 0.
  """
  # When the shape of `a` is smaller than `n`, there is no cut off.
  if n is None or n >= a.shape[-1]:
    return jnp.ones_like(a)

  # When `n` is 0, everything is cut off.
  if n == 0:
    return jnp.zeros_like(a)

  # Get the sorted value at `n` and `n+1`, which is where the cutoff is supposed
  # to happen.
  a_topn = sort_by(a, [a], where=where)[0][..., :n][..., -1]
  a_topnp1 = sort_by(a, [a], where=where)[0][..., :n + 1][..., -1]

  # Place the cutoff point between a[n] and a[n+1]. For exact `step_fn`, this
  # does not make a difference. But for approximate step functions (e.g.
  # sigmoid), this ensures the cutoff value at `n` can be close to 1, and the
  # cutoff value at `n+1` can be close to 0.
  a_cutoff = jax.lax.stop_gradient((a_topnp1 + a_topn) / 2)
  cutoffs = step_fn(a - jnp.expand_dims(a_cutoff, -1))

  if where is not None:
    # A mask can indicate a different list size for each list in `a`, so this
    # checks if a valid cutoff exists for each list.
    valid_cutoffs = n < jnp.sum(where, axis=-1, keepdims=True)
    cutoffs = jnp.where(valid_cutoffs, cutoffs, jnp.ones_like(cutoffs))

    # Also mask out invalid entries.
    cutoffs = jnp.where(where, cutoffs, jnp.zeros_like(cutoffs))

  return cutoffs


approx_cutoff = jax.util.wraps(cutoff)(
    functools.partial(cutoff, step_fn=jax.nn.sigmoid))


def compute_pairs(a: Array, op: Callable[[Array, Array], Array]) -> Array:
  """Computes pairs based on values of `a` and the given pairwise `op`.

  Args:
    a: The array used to form pairs. The last axis is used to form pairs.
    op: The binary op to map a pair of values to a single value.

  Returns:
    A new array with the same leading dimensions as `a`, but with the last
    dimension expanded so it includes all pairs `op(a[..., i], a[..., j])`
  """
  a_i = jnp.expand_dims(a, -1)
  a_j = jnp.expand_dims(a, -2)
  result_shape = jnp.broadcast_shapes(a_i.shape, a_j.shape)
  result = jnp.broadcast_to(op(a_i, a_j), result_shape)
  out_shape = tuple(result.shape[:-2]) + (result.shape[-2] * result.shape[-1],)
  return jnp.reshape(result, out_shape)
