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

"""Ranking metrics in JAX.

A ranking metric function is a callable that accepts a scores tensor, a labels
tensor, and, optionally a mask tensor. In addition to these arguments, metric
functions *may* accept additional optional keyword arguments, e.g. for weights
or topn, however this depends on the specific metric function and is not
required.

The metric functions operate on the last dimension of its inputs. The leading
dimensions are considered batch dimensions. To compute per-list metrics, for
example to apply per-list weighting or for distributed computing of metrics
across devices, please use standard JAX transformations such as `jax.vmap` or
`jax.pmap`.

Example usage:
>>> import jax
>>> import rax
>>> scores = jnp.asarray([[0., 1., 3.], [1., 2., 0.]])
>>> labels = jnp.asarray([[0., 0., 1.], [1., 0., 0.]])
>>> rax.mrr_metric(scores, labels)
DeviceArray(0.75, dtype=float32)

"""

from typing import Callable, Optional

import jax.numpy as jnp

from rax._src import utils
from rax._src.protocols import CutoffFn
from rax._src.protocols import RankFn
from rax._src.protocols import ReduceFn


def _retrieved_items(scores: jnp.ndarray,
                     ranks: jnp.ndarray,
                     *,
                     mask: Optional[jnp.ndarray] = None,
                     topn: Optional[int] = None,
                     cutoff_fn: CutoffFn = utils.cutoff) -> jnp.ndarray:
  """Computes an array that indicates which items are retrieved.

  Args:
    scores: A [..., list_size]-jnp.ndarray, indicating the score of each item.
    ranks: A [..., list_size]-jnp.ndarray, indicating the 1-based rank of each
      item.
    mask: An optional [..., list_size]-jnp.ndarray, indicating which items are
      valid.
    topn: An optional integer value indicating at which rank items are cut off.
      If None, no cutoff is performed.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.

  Returns:
    A [..., list_size]-jnp.ndarray, indicating for each item whether they are a
    retrieved item or not.
  """
  # Items with score > -inf are considered retrieved items.
  retrieved_items = jnp.float32(jnp.logical_not(jnp.isneginf(scores)))

  # Masked items should always be ignored and not retrieved.
  if mask is not None:
    retrieved_items *= jnp.float32(mask)

  # Only consider items in the topn as retrieved items.
  retrieved_items *= cutoff_fn(-ranks, n=topn)

  return retrieved_items


def default_gain_fn(label: jnp.ndarray) -> jnp.ndarray:
  r"""Default gain function used for gain-based metrics.

  Definition:

  .. math::
      \operatorname{gain}(y) = 2^{y} - 1

  Args:
    label: The label to compute the gain for.

  Returns:
    The gain value for given label.
  """
  return jnp.power(2., label) - 1.


def default_discount_fn(rank: jnp.ndarray) -> jnp.ndarray:
  r"""Default discount function used for discount-based metrics.

  Definition:

  .. math::
      \operatorname{discount}(r) = 1. / \log_2(r + 1)

  Args:
    rank: The 1-based rank.

  Returns:
    The discount value for given rank.
  """
  return 1. / jnp.log2(rank + 1)


def mrr_metric(scores: jnp.ndarray,
               labels: jnp.ndarray,
               *,
               mask: Optional[jnp.ndarray] = None,
               topn: Optional[int] = None,
               rng_key: Optional[jnp.ndarray] = None,
               rank_fn: RankFn = utils.sort_ranks,
               cutoff_fn: CutoffFn = utils.cutoff,
               reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Mean reciprocal rank (MRR).

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.mrr_metric(scores, labels)
  DeviceArray(0.5, dtype=float32)

  Usage with a batch of data:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> rax.mrr_metric(scores, labels)
  DeviceArray(0.75, dtype=float32)

  Usage with `jax.vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(rax.mrr_metric)(scores, labels, mask=mask)
  DeviceArray([1., 1.], dtype=float32)

  Definition:

  .. math::
      \operatorname{mrr}(s, y) = \max_i \frac{y_i}{\operatorname{rank}(s_i)}

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using `rank_fn`.

  This metric converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If None, no cutoff is performed.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The MRR metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)
  retrieved_items = _retrieved_items(
      scores, ranks, mask=mask, topn=topn, cutoff_fn=cutoff_fn)

  # Compute reciprocal ranks.
  reciprocal_ranks = jnp.reciprocal(jnp.where(ranks == 0., jnp.inf, ranks))

  # Get the maximum reciprocal rank.
  values = jnp.max(
      relevant_items * retrieved_items * reciprocal_ranks,
      axis=-1,
      where=mask,
      initial=0.)
  return utils.safe_reduce(values, reduce_fn=reduce_fn)


def recall_metric(scores: jnp.ndarray,
                  labels: jnp.ndarray,
                  *,
                  topn: Optional[int] = None,
                  mask: Optional[jnp.ndarray] = None,
                  rng_key: Optional[jnp.ndarray] = None,
                  rank_fn: RankFn = utils.sort_ranks,
                  cutoff_fn: CutoffFn = utils.cutoff,
                  reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Recall.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 1., 0.])
  >>> rax.recall_metric(scores, labels, topn=2)
  DeviceArray(0.5, dtype=float32)

  Usage with a batch of data:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 1., 0.], [1., 0., 1.]])
  >>> rax.recall_metric(scores, labels, topn=2)
  DeviceArray(0.75, dtype=float32)

  Usage with `jax.vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 1.], [0., 1., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(functools.partial(rax.recall_metric, topn=2))(
  ...     scores, labels, mask=mask)
  DeviceArray([1. , 0.5], dtype=float32)

  Definition:

  .. math::
      \operatorname{recall@n}(s, y) =
      \frac{1}{\sum_i y_i} \sum_i y_i \operatorname{rank}(s_i) \leq n

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using `rank_fn`.

  This metric converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If None, no cutoff is performed.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The recall metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)
  retrieved_items = _retrieved_items(
      scores, ranks, mask=mask, topn=topn, cutoff_fn=cutoff_fn)

  # Compute number of retrieved+relevant items and relevant items.
  n_retrieved_relevant = jnp.sum(
      retrieved_items * relevant_items, where=mask, axis=-1)
  n_relevant = jnp.sum(relevant_items, where=mask, axis=-1)

  # Compute recall but prevent division by zero.
  n_relevant = jnp.where(n_relevant == 0, 1., n_relevant)
  values = n_retrieved_relevant / n_relevant
  return utils.safe_reduce(values, reduce_fn=reduce_fn)


def precision_metric(scores: jnp.ndarray,
                     labels: jnp.ndarray,
                     *,
                     topn: Optional[int] = None,
                     mask: Optional[jnp.ndarray] = None,
                     rng_key: Optional[jnp.ndarray] = None,
                     rank_fn: RankFn = utils.sort_ranks,
                     cutoff_fn: CutoffFn = utils.cutoff,
                     reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Precision.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 1., 0.])
  >>> rax.precision_metric(scores, labels, topn=3)
  DeviceArray(0.6666667, dtype=float32)

  Usage with a batch of data:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 1., 0.], [1., 0., 1.]])
  >>> rax.precision_metric(scores, labels, topn=2)
  DeviceArray(0.75, dtype=float32)

  Usage with `jax.vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(functools.partial(rax.precision_metric, topn=None))(
  ...     scores, labels, mask=mask)
  DeviceArray([0.5       , 0.33333334], dtype=float32)

  Definition:

  .. math::
      \operatorname{precision@n}(s, y) =
      \frac{1}{n} \sum_i y_i \operatorname{rank}(s_i) \leq n

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using `rank_fn`.

  This metric converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If not provided, no cutoff is performed.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The precision metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)
  retrieved_items = _retrieved_items(
      scores, ranks, mask=mask, topn=topn, cutoff_fn=cutoff_fn)

  # Compute number of retrieved+relevant items and retrieved items.
  n_retrieved_relevant = jnp.sum(
      retrieved_items * relevant_items, where=mask, axis=-1)
  n_retrieved = jnp.sum(retrieved_items, where=mask, axis=-1)

  # Compute precision but prevent division by zero.
  n_retrieved = jnp.where(n_retrieved == 0, 1., n_retrieved)
  values = n_retrieved_relevant / n_retrieved
  return utils.safe_reduce(values, reduce_fn=reduce_fn)


def ap_metric(scores: jnp.ndarray,
              labels: jnp.ndarray,
              *,
              topn: Optional[int] = None,
              mask: Optional[jnp.ndarray] = None,
              rng_key: Optional[jnp.ndarray] = None,
              rank_fn: RankFn = utils.sort_ranks,
              cutoff_fn: CutoffFn = utils.cutoff,
              reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Average Precision.

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 1., 0.])
  >>> rax.ap_metric(scores, labels)
  DeviceArray(0.5833334, dtype=float32)

  Usage with `jax.vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 1., 0.], [1., 0., 1.]])
  >>> rax.ap_metric(scores, labels)
  DeviceArray(0.7916667, dtype=float32)

  Usage with `jax.vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(rax.ap_metric)(scores, labels, mask=mask)
  DeviceArray([1., 1.], dtype=float32)

  Definition:

  .. math::
      \operatorname{ap}(s, y) =
      \frac{1}{y_i} \sum_i y_i \operatorname{precision@i}(s, y)

  where :math:`\operatorname{precision@i}(s, y)` indicates the precision at
  :math:`i`.

  This metric converts graded relevance to binary relevance by considering items
  with `label >= 1` as relevant and items with `label < 1` as non-relevant.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If not provided, no cutoff is performed.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The average precision metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)
  retrieved_items = _retrieved_items(
      scores, ranks, mask=mask, topn=topn, cutoff_fn=cutoff_fn)

  # Compute ranks.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)

  # Compute a matrix of all precision@k values
  relevant_i = jnp.expand_dims(relevant_items, axis=-1)
  relevant_j = jnp.expand_dims(relevant_items, axis=-2)
  ranks_i = jnp.expand_dims(ranks, axis=-1)
  ranks_j = jnp.expand_dims(ranks, axis=-2)
  prec_at_k = ((ranks_i >= ranks_j) * relevant_i * relevant_j) / ranks_i

  # Only include precision@k for retrieved items.
  sum_prec_at_k = jnp.sum(
      prec_at_k * jnp.expand_dims(retrieved_items, -1), axis=(-2, -1))

  # Compute number of relevant items.
  n_relevant = jnp.sum(relevant_items, where=mask, axis=-1)

  # Compute average precision but prevent division by zero.
  n_relevant = jnp.where(n_relevant == 0, 1., n_relevant)
  values = sum_prec_at_k / n_relevant
  return utils.safe_reduce(values, reduce_fn=reduce_fn)


def dcg_metric(scores: jnp.ndarray,
               labels: jnp.ndarray,
               *,
               mask: Optional[jnp.ndarray] = None,
               topn: Optional[int] = None,
               weights: Optional[jnp.ndarray] = None,
               rng_key: Optional[jnp.ndarray] = None,
               gain_fn: Callable[[jnp.ndarray], jnp.ndarray] = default_gain_fn,
               discount_fn: Callable[[jnp.ndarray],
                                     jnp.ndarray] = default_discount_fn,
               rank_fn: RankFn = utils.sort_ranks,
               cutoff_fn: CutoffFn = utils.cutoff,
               reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Discounted cumulative gain (DCG).

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.dcg_metric(scores, labels)
  DeviceArray(0.63092977, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.dcg_metric)(scores, labels)
  DeviceArray([0.63092977, 1.        ], dtype=float32)

  Usage with `vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(rax.dcg_metric)(scores, labels, mask=mask)
  DeviceArray([1., 1.], dtype=float32)

  Definition:

  .. math::
      \operatorname{dcg}(s, y) =
      \sum_i \operatorname{gain}(y_i)
      \cdot \operatorname{discount}(\operatorname{rank}(s_i))

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using `rank_fn`.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item. All labels are assumed to be positive when using the default gain
      function.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If not provided, no cutoff is performed.
    weights: An optional [list_size]-jnp.ndarray indicating the per-example
      weights.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    gain_fn: A callable that maps relevance label to gain values.
    discount_fn: A callable that maps 1-based ranks to discount values.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The DCG metric.
  """
  # Get the retrieved items.
  ranks = rank_fn(scores, mask=mask, rng_key=rng_key)
  retrieved_items = _retrieved_items(
      scores, ranks, mask=mask, topn=topn, cutoff_fn=cutoff_fn)

  # Computes (weighted) gains.
  gains = gain_fn(labels)
  if weights is not None:
    gains *= weights

  # Computes rank discounts.
  discounts = discount_fn(ranks)

  # Compute DCG.
  values = jnp.sum(retrieved_items * gains * discounts, axis=-1, where=mask)
  return utils.safe_reduce(values, reduce_fn=reduce_fn)


def ndcg_metric(scores: jnp.ndarray,
                labels: jnp.ndarray,
                *,
                mask: Optional[jnp.ndarray] = None,
                topn: Optional[int] = None,
                weights: Optional[jnp.ndarray] = None,
                rng_key: Optional[jnp.ndarray] = None,
                gain_fn: Callable[[jnp.ndarray], jnp.ndarray] = default_gain_fn,
                discount_fn: Callable[[jnp.ndarray],
                                      jnp.ndarray] = default_discount_fn,
                rank_fn: RankFn = utils.sort_ranks,
                cutoff_fn: CutoffFn = utils.cutoff,
                reduce_fn: Optional[ReduceFn] = jnp.mean) -> jnp.ndarray:
  r"""Computes normalized discounted cumulative gain (NDCG).

  Standalone usage:

  >>> scores = jnp.array([2., 1., 3.])
  >>> labels = jnp.array([1., 0., 0.])
  >>> rax.ndcg_metric(scores, labels)
  DeviceArray(0.63092977, dtype=float32)

  Usage with `vmap` batching:

  >>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> jax.vmap(rax.ndcg_metric)(scores, labels)
  DeviceArray([0.63092977, 1.        ], dtype=float32)

  Usage with `vmap` batching and a mask to indicate valid items:

  >>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
  >>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
  >>> mask = jnp.array([[True, True, False], [True, True, True]])
  >>> jax.vmap(rax.ndcg_metric)(scores, labels, mask=mask)
  DeviceArray([1., 1.], dtype=float32)

  Definition:

  .. math::
      \operatorname{ndcg}(s, y) =
      \operatorname{dcg}(s, y) / \operatorname{dcg}(y, y)

  where :math:`\operatorname{dcg}` is the discounted cumulative gain metric.

  Args:
    scores: A [list_size]-jnp.ndarray, indicating the score of each item. Items
      for which the score is `-inf` are treated as unranked items.
    labels: A [list_size]-jnp.ndarray, indicating the relevance label for each
      item. All labels are assumed to be positive when using the default gain
      function.
    mask: An optional [list_size]-jnp.ndarray, indicating which items are valid
      for computing the metric.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If not provided, no cutoff is performed.
    weights: An optional [list_size]-jnp.ndarray indicating the per-example
      weights.
    rng_key: An optional jax rng key. If provided, any random operations in this
      metric will be based on this key.
    gain_fn: A callable that maps relevance label to gain values.
    discount_fn: A callable that maps 1-based ranks to discount values.
    rank_fn: A callable that maps scores to 1-based ranks. The callable should
      accept a `scores` argument and optional `mask` and `rng_key` keyword
      arguments and return a `ranks` jnp.ndarray of the same shape as `scores`.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` jnp.ndarray and an `n` integer and return
      a cutoffs jnp.ndarray of the same shape as `ranks`.
    reduce_fn: An optional Callable that reduces the loss values. The callable
      should accept a loss tensor and an optional `where` tensor indicating
      which elements to include in the reduction. Can be `jnp.sum` or
      `jnp.mean`. If `None`, no reduction is performed.

  Returns:
    The NDCG metric.
  """
  # Compute regular dcg.
  regular_dcg = dcg_metric(
      scores,
      labels,
      mask=mask,
      topn=topn,
      weights=weights,
      rng_key=rng_key,
      gain_fn=gain_fn,
      discount_fn=discount_fn,
      rank_fn=rank_fn,
      cutoff_fn=cutoff_fn,
      reduce_fn=None)

  # The ideal dcg is computed by ordering items by their (weighted) gains.
  ideal_scores = gain_fn(labels)
  if weights is not None:
    ideal_scores *= weights
  ideal_dcg = dcg_metric(
      ideal_scores,
      labels,
      mask=mask,
      topn=topn,
      weights=weights,
      rng_key=None,
      gain_fn=gain_fn,
      discount_fn=discount_fn,
      rank_fn=utils.sort_ranks,
      cutoff_fn=utils.cutoff,
      reduce_fn=None)

  # Compute the result as `dcg / ideal_dcg` while preventing division by zero.
  ideal_dcg = jnp.where(ideal_dcg == 0., 1., ideal_dcg)
  values = regular_dcg / ideal_dcg
  return utils.safe_reduce(values, reduce_fn=reduce_fn)
