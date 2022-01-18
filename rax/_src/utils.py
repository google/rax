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

# Type alias for a tensor shape:
Shape = Sequence[Optional[int]]


def safe_reduce(
    a: jnp.ndarray,
    where: Optional[jnp.ndarray] = None,
    reduce_fn: Optional[Callable[..., jnp.ndarray]] = None) -> jnp.ndarray:
  """Reduces the values of given array while preventing NaN in the output.

  Args:
    a: The array to reduce.
    where: Which elements to include in the reduction.
    reduce_fn: The function used to reduce. If None, no reduction is performed.

  Returns:
    The result of reducing the values of `a` using given `reduce_fn`. Any NaN
    values in the output are converted to 0. If `reduce_fn` is None, no
    reduction is performed and `a` is returned as-is.
  """
  if reduce_fn is not None:
    # Perform reduction.
    a = reduce_fn(a, where=where)

    # Set NaN to zero. This is common for mean-reduction with `where` being
    # False everywhere (e.g. pairwise losses with no valid pairs).
    a = jnp.where(jnp.isnan(a), 0., a)
  return a


def normalize_probabilities(unscaled_probabilities: jnp.ndarray,
                            mask: Optional[jnp.ndarray] = None,
                            axis: int = -1) -> jnp.ndarray:
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
    mask: The mask to indicate which entries are valid.
    axis: The axis to normalize on.

  Returns:
    Given unscaled probabilities normalized so they sum to one for the valid
    (non-masked) items in the given axis.
  """
  if mask is None:
    mask = jnp.ones_like(unscaled_probabilities, dtype=jnp.bool_)

  # Sum only the valid items across the given axis.
  unscaled_probabilities_sum = jnp.sum(
      unscaled_probabilities, axis=axis, keepdims=True, where=mask)

  # Sum mask for correctly normalizing all-zero elements with mask.
  mask_sum = jnp.sum(mask, axis=axis, keepdims=True)
  mask_sum = jnp.where(mask_sum == 0.,
                       jnp.sum(jnp.ones_like(mask), axis=axis, keepdims=True),
                       mask_sum)

  # Compute output
  return jnp.where(
      unscaled_probabilities_sum != 0.,
      unscaled_probabilities / unscaled_probabilities_sum,
      jnp.ones_like(mask, dtype=unscaled_probabilities.dtype) / mask_sum)


def sort_by(scores: jnp.ndarray,
            tensors_to_sort: Sequence[jnp.ndarray],
            axis: int = -1,
            mask: Optional[jnp.ndarray] = None,
            rng_key: Optional[jnp.ndarray] = None) -> Sequence[jnp.ndarray]:
  """Sorts given list of tensors by given scores.

  Each of the entries in the `tensors_to_sort` sequence must be a tensor that
  matches the shape of the `scores`.

  Args:
    scores: A tensor, representing scores by which to sort.
    tensors_to_sort: A sequence of tensors of the same shape as scores. These
      are the tensors that will be sorted in the order of scores.
    axis: The axis to sort on, by default this is the last axis.
    mask: An optional tensor of the same shape as scores, indicating which
      entries are valid for sorting. Invalid entries are pushed to the end.
    rng_key: An optional jax rng key. If provided, ties will be broken randomly
      using this key. If not provided, ties will retain the order of their
      appearance in the `scores` array.

  Returns:
    A tuple containing the tensors of `tensors_to_sort` sorted in the order of
    `scores`.
  """
  # Sets up the keys we want to sort on.
  sort_operands = []
  if mask is not None:
    sort_operands.append(jnp.logical_not(mask))
  sort_operands.append(-scores)
  if rng_key is not None:
    sort_operands.append(jax.random.uniform(rng_key, scores.shape))
  num_keys = len(sort_operands)

  # Adds the tensors that we want sort.
  sort_operands.extend(tensors_to_sort)

  # Performs sort and returns the sorted tensors.
  sorted_values = jax.lax.sort(sort_operands, dimension=axis, num_keys=num_keys)
  return sorted_values[num_keys:]


def sort_ranks(scores: jnp.ndarray,
               *,
               axis: int = -1,
               mask: Optional[jnp.ndarray] = None,
               rng_key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Returns the ranks for given scores via sorting.

  Note that the ranks returned by this function are not differentiable due to
  the sort operation having no gradients.

  Args:
    scores: A tensor, representing scores to be ranked.
    axis: The axis to sort on, by default this is the last axis.
    mask: An optional tensor of the same shape as scores, indicating which
      entries are valid for ranking. Invalid entries are ranked last.
    rng_key: An optional jax rng key. If provided, ties will be broken randomly
      using this key. If not provided, ties will retain the order of their
      appearance in the `scores` array.

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
  indices = sort_by(scores, [arange], axis=axis, mask=mask, rng_key=rng_key)[0]

  # Perform an argsort on the indices to get the 1-based ranks.
  return jnp.argsort(indices, axis=axis) + 1


def approx_ranks(
    scores: jnp.ndarray,
    *,
    mask: Optional[jnp.ndarray] = None,
    rng_key: Optional[jnp.ndarray] = None,
    step_fn: Callable[[jnp.ndarray],
                      jnp.ndarray] = jax.nn.sigmoid) -> jnp.ndarray:
  """Computes approximate ranks.

  This can be used to construct differentiable approximations of metrics. For
  example:

  >>> import functools
  >>> approx_ndcg = functools.partial(rax.ndcg_metric, rank_fn=rax.approx_ranks)
  >>> scores = jnp.asarray([-1., 1., 0.])
  >>> labels = jnp.asarray([0., 0., 1.])
  >>> approx_ndcg(scores, labels)
  DeviceArray(0.63092977, dtype=float32)
  >>> jax.grad(approx_ndcg)(scores, labels)
  DeviceArray([-0.03763788, -0.03763788,  0.07527576], dtype=float32)

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score for each item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid.
    rng_key: An optional jax rng key. Unused by approx_ranks.
    step_fn: A callable that approximates the step function `x >= 0`.

  Returns:
    A [..., list_size]-jnp.ndarray, indicating the 1-based approximate rank of
    each item.
  """

  del rng_key  # unused for approximate ranks.

  score_i = jnp.expand_dims(scores, axis=-1)
  score_j = jnp.expand_dims(scores, axis=-2)
  score_pairs = step_fn(score_j - score_i)

  # Build mask to prevent counting (i == j) pairs.
  triangular_mask = (jnp.triu(jnp.tril(jnp.ones_like(score_pairs))) == 0.0)
  if mask is not None:
    mask = jnp.expand_dims(mask, axis=-1) & jnp.expand_dims(mask, axis=-2)
    triangular_mask &= mask

  return jnp.sum(score_pairs, axis=-1, where=triangular_mask, initial=1.0)


def cutoff(
    a: jnp.ndarray,
    n: Optional[int] = None,
    *,
    where: Optional[jnp.ndarray] = None,
    step_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x >= 0
) -> jnp.ndarray:
  """Computes an indicator (or approximation) to select the largest `n` values.

  This function computes an indicator (or approximation) that selects the `n`
  largest values of `a` across its last dimension.

  Note that the returned indicator may select more than `n` items if `a` has
  ties.

  Args:
    a: The array to select the topn from.
    n: The cutoff value. If None, no cutoff is performed.
    where: A mask to indicate which values to include in the topn calculation.
    step_fn: A function that computes `x >= 0` or an approximation.

  Returns:
    A {0, 1}-tensor (or approximation thereof) of the same shape as `a`, where
    the `n` largest values are set to 1, and the smaller values are set to 0.
  """

  if n is None:
    return jnp.ones_like(a)
  a_topn = sort_by(a, [a], mask=where)[0][..., :n]
  if a_topn.shape[-1] == 0:
    return jnp.zeros_like(a)
  return step_fn(a - jnp.expand_dims(a_topn[..., -1], -1))


approx_cutoff = jax.util.wraps(cutoff)(
    functools.partial(cutoff, step_fn=jax.nn.sigmoid))
