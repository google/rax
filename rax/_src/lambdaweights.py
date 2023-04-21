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

"""Implementations of lambdaweight functions for Rax pairwise losses.

Lambdaweight functions dynamically adjust the weights of a pairwise loss based
on the scores and labels. Rax provides a number of lambdaweight functions as JAX
functions that are implemented according to the
:class:`~rax.types.LambdaweightFn` interface.

Example usage:

>>> scores = jnp.array([1.2, 0.4, 1.9])
>>> labels = jnp.array([1.0, 2.0, 0.0])
>>> loss = rax.pairwise_logistic_loss(
...     scores, labels, lambdaweight_fn=rax.labeldiff_lambdaweight)
>>> print(loss)
1.8923712
"""

import operator
from typing import Callable, Optional

import jax.numpy as jnp
from rax._src import metrics
from rax._src import utils
from rax._src.types import Array


def labeldiff_lambdaweight(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    weights: Optional[Array] = None
) -> Array:
  r"""Absolute label difference lambdaweights.

  Definition:

  .. math::
      \lambda_{ij}(s, y) = |y_i - y_j|

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the lambdaweights. Items
      for which this is False will be ignored when computing the lambdaweights.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The loss will only be computed on
      items that share the same segment.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.

  Returns:
    Absolute label difference lambdaweights.
  """
  del scores, weights  # Unused.

  results = jnp.abs(utils.compute_pairs(labels, operator.sub))

  if where is not None:
    results = jnp.where(utils.compute_pairs(where, operator.eq), results, 0.0)

  if segments is not None:
    results = jnp.where(
        utils.compute_pairs(segments, operator.eq), results, 0.0
    )

  return results


def dcg_lambdaweight(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    weights: Optional[Array] = None,
    topn: Optional[int] = None,
    normalize: bool = False,
    gain_fn: Callable[[Array], Array] = metrics.default_gain_fn,
    discount_fn: Callable[[Array], Array] = metrics.default_discount_fn
) -> Array:
  r"""DCG lambdaweights.

  Definition :cite:p:`burges2006learning`:

  .. math::
      \lambda_{ij}(s, y) = |\operatorname{gain}(y_i) - \operatorname{gain}(y_j)|
      \cdot |\operatorname{discount}(\operatorname{rank}(s_i)) -
      \operatorname{discount}(\operatorname{rank}(s_j))|

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the lambdaweights. Items
      for which this is False will be ignored when computing the lambdaweights.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The loss will only be computed on
      items that share the same segment.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    topn: The topn cutoff. If ``None``, no cutoff is performed.
    normalize: Whether to use the normalized DCG formulation.
    gain_fn: A function mapping labels to gain values.
    discount_fn: A function mapping ranks to discount values.

  Returns:
    DCG lambdaweights.
  """
  ranks = utils.ranks(scores, where=where, segments=segments)
  gains = gain_fn(labels)
  if weights is not None:
    gains *= weights

  if normalize:
    ideal_dcg = metrics.dcg_metric(
        gains,
        labels,
        where=where,
        segments=segments,
        topn=topn,
        weights=weights,
        gain_fn=gain_fn,
        discount_fn=discount_fn,
        reduce_fn=None,
    )
    ideal_dcg = jnp.where(ideal_dcg == 0.0, 1.0, ideal_dcg)
    if segments is None:
      ideal_dcg = jnp.expand_dims(ideal_dcg, -1)
    gains /= ideal_dcg

  gains_abs_diffs = jnp.abs(utils.compute_pairs(gains, operator.sub))

  if where is not None:
    valid_pairs = utils.compute_pairs(where, operator.and_)
  else:
    valid_pairs = jnp.ones_like(gains_abs_diffs, dtype=jnp.bool_)

  discounts = discount_fn(ranks)

  if topn is not None:
    discounts = jnp.where(ranks <= topn, discounts, 0.0)

  discounts_abs_diffs = jnp.abs(utils.compute_pairs(discounts, operator.sub))
  discounts_abs_diffs = jnp.where(valid_pairs, discounts_abs_diffs, 0.0)

  # Scale up the lambdaweights by the constant list size to avoid too small
  # values.
  weight_scalar = labels.shape[-1]

  results = discounts_abs_diffs * gains_abs_diffs * weight_scalar
  if segments is None:
    return results
  else:
    return jnp.where(utils.compute_pairs(segments, operator.eq), results, 0.0)


def dcg2_lambdaweight(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    weights: Optional[Array] = None,
    topn: Optional[int] = None,
    normalize: bool = False,
    gain_fn: Callable[[Array], Array] = metrics.default_gain_fn,
    discount_fn: Callable[[Array], Array] = metrics.default_discount_fn
) -> Array:
  r"""DCG v2 ("lambdaloss") lambdaweights.

  Definition :cite:p:`wang2018lambdaloss`:

  .. math::
      \lambda_{ij}(s, y) = |\operatorname{gain}(y_i) - \operatorname{gain}(y_j)|
      \cdot |\operatorname{discount}(
          |\operatorname{rank}(s_i) - \operatorname{rank}(s_j)|) -
      \operatorname{discount}(
          |\operatorname{rank}(s_i) - \operatorname{rank}(s_j)| + 1)|

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the lambdaweights. Items
      for which this is False will be ignored when computing the lambdaweights.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The loss will only be computed on
      items that share the same segment.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    topn: The topn cutoff. If ``None``, no cutoff is performed. Topn cutoff is
      uses the method described in :cite:p:`jagerman2022optimizing`.
    normalize: Whether to use the normalized DCG formulation.
    gain_fn: A function mapping labels to gain values.
    discount_fn: A function mapping ranks to discount values.

  Returns:
    DCG v2 ("lambdaloss") lambdaweights.
  """
  ranks = utils.ranks(scores, where=where, segments=segments)
  gains = gain_fn(labels)
  if weights is not None:
    gains *= weights

  if normalize:
    ideal_dcg = metrics.dcg_metric(
        gains,
        labels,
        where=where,
        topn=topn,
        weights=weights,
        gain_fn=gain_fn,
        discount_fn=discount_fn,
        reduce_fn=None,
    )
    ideal_dcg = jnp.where(ideal_dcg == 0.0, 1.0, ideal_dcg)
    if segments is None:
      ideal_dcg = jnp.expand_dims(ideal_dcg, -1)
    gains /= ideal_dcg

  gains_abs_diffs = jnp.abs(utils.compute_pairs(gains, operator.sub))

  if where is not None:
    valid_pairs = utils.compute_pairs(where, operator.and_)
  else:
    valid_pairs = jnp.ones_like(gains_abs_diffs, dtype=jnp.bool_)

  ranks_abs_diffs = jnp.abs(utils.compute_pairs(ranks, operator.sub))
  ranks_max = utils.compute_pairs(ranks, jnp.maximum)

  discounts = jnp.abs(
      discount_fn(ranks_abs_diffs) - discount_fn(ranks_abs_diffs + 1)
  )
  discounts = jnp.where(ranks_abs_diffs != 0, discounts, 0.0)

  if topn is not None:
    topn_weights = 1.0 / (1.0 - discount_fn(ranks_max))
    discounts *= jnp.where(ranks_max > topn, topn_weights, 1.0)

  discounts = jnp.where(valid_pairs, discounts, 0.0)

  # Scale up the lambdaweights by the constant list size to avoid too small
  # values.
  weight_scalar = labels.shape[-1]

  results = discounts * gains_abs_diffs * weight_scalar
  if segments is None:
    return results
  else:
    return jnp.where(utils.compute_pairs(segments, operator.eq), results, 0.0)
