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

"""Implementations of common ranking metrics in JAX.

A ranking metric expresses how well a ranking induced by item scores matches a
ranking induced from relevance labels. Rax provides a number of ranking metrics
as JAX functions that are implemented according to the
:class:`~rax.types.MetricFn` interface.

Metric functions are designed to operate on the last dimension of its inputs.
The leading dimensions are considered batch dimensions. To compute per-list
metrics, for example to apply per-list weighting or for distributed computing of
metrics across devices, please use standard JAX transformations such as
:func:`jax.vmap` or :func:`jax.pmap`.

Standalone usage of a metric:

>>> import jax
>>> import rax
>>> scores = jnp.array([2., 1., 3.])
>>> labels = jnp.array([2., 0., 1.])
>>> print(rax.ndcg_metric(scores, labels))
0.79670763

Usage with a batch of data and a mask to indicate valid items:

>>> scores = jnp.array([[2., 1., 3.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[2., 0., 1.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> print(rax.ndcg_metric(scores, labels))
0.8983538

Usage with :func:`jax.vmap` batching and a mask to indicate valid items:

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> print(jax.vmap(rax.ndcg_metric)(scores, labels, where=where))
[1. 1.]

"""

from typing import Callable, Optional

import jax.numpy as jnp

from rax._src import utils
from rax._src.types import Array
from rax._src.types import CutoffFn
from rax._src.types import RankFn
from rax._src.types import ReduceFn


def _retrieved_items(
    scores: Array,
    ranks: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    topn: Optional[int] = None,
    cutoff_fn: CutoffFn = utils.cutoff
) -> Array:
  """Computes an array that indicates which items are retrieved.

  Args:
    scores: A [..., list_size]-Array, indicating the score of each item.
    ranks: A [..., list_size]-Array, indicating the 1-based rank of each item.
    where: An optional [..., list_size]-Array, indicating which items are valid.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The retrieved items will only be
      computed on items that share the same segment.
    topn: An optional integer value indicating at which rank items are cut off.
      If None, no cutoff is performed.
    cutoff_fn: A callable that computes a cutoff tensor indicating which
      elements to include and which ones to exclude based on a topn cutoff. The
      callable should accept a `ranks` Array and an `n` integer and return a
      cutoffs Array of the same shape as `ranks`.

  Returns:
    A [..., list_size]-Array, indicating for each item whether they are a
    retrieved item or not.
  """
  # Items with score > -inf are considered retrieved items.
  retrieved_items = jnp.float32(jnp.logical_not(jnp.isneginf(scores)))

  # Masked items should always be ignored and not retrieved.
  if where is not None:
    retrieved_items *= jnp.float32(where)

  # Only consider items in the topn as retrieved items.
  retrieved_items *= cutoff_fn(-ranks, segments=segments, n=topn)

  return retrieved_items


def default_gain_fn(label: Array) -> Array:
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


def default_discount_fn(rank: Array) -> Array:
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


def mrr_metric(scores: Array,
               labels: Array,
               *,
               where: Optional[Array] = None,
               topn: Optional[int] = None,
               key: Optional[Array] = None,
               rank_fn: RankFn = utils.ranks,
               cutoff_fn: CutoffFn = utils.cutoff,
               reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Mean Reciprocal Rank (MRR).

  .. note::

    This metric converts graded relevance to binary relevance by considering
    items with ``label >= 1`` as relevant and items with ``label < 1`` as
    non-relevant.

  Definition:

  .. math::
      \operatorname{mrr}(s, y) = \max_i \frac{y_i}{\operatorname{rank}(s_i)}

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using ``rank_fn``.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The MRR metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, where=where, key=key)
  retrieved_items = _retrieved_items(
      scores, ranks, where=where, topn=topn, cutoff_fn=cutoff_fn)

  # Compute reciprocal ranks.
  reciprocal_ranks = jnp.reciprocal(jnp.where(ranks == 0., jnp.inf, ranks))

  # Get the maximum reciprocal rank.
  values = jnp.max(
      relevant_items * retrieved_items * reciprocal_ranks,
      axis=-1,
      where=where,
      initial=0.)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  if where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)


def recall_metric(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    topn: Optional[int] = None,
    key: Optional[Array] = None,
    rank_fn: RankFn = utils.ranks,
    cutoff_fn: CutoffFn = utils.cutoff,
    reduce_fn: Optional[ReduceFn] = jnp.mean
) -> Array:
  r"""Recall.

  .. note::

    This metric converts graded relevance to binary relevance by considering
    items with ``label >= 1`` as relevant and items with ``label < 1`` as
    non-relevant.

  Definition:

  .. math::
      \operatorname{recall@n}(s, y) =
      \frac{1}{\sum_i y_i}
      \sum_i y_i \cdot \mathbb{I}\left[\operatorname{rank}(s_i) \leq n\right]

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using `rank_fn`.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The metric will only be computed on
      items that share the same segment.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The recall metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, where=where, segments=segments, key=key)
  retrieved_items = _retrieved_items(
      scores,
      ranks,
      where=where,
      segments=segments,
      topn=topn,
      cutoff_fn=cutoff_fn,
  )

  # Compute number of retrieved+relevant items and relevant items.
  if segments is not None:
    n_retrieved_relevant = utils.segment_sum(
        retrieved_items * relevant_items, segments, where=where
    )
    n_relevant = utils.segment_sum(relevant_items, segments, where=where)
  else:
    n_retrieved_relevant = jnp.sum(
        retrieved_items * relevant_items, where=where, axis=-1
    )
    n_relevant = jnp.sum(relevant_items, where=where, axis=-1)

  # Compute recall but prevent division by zero.
  n_relevant = jnp.where(n_relevant == 0, 1., n_relevant)
  values = n_retrieved_relevant / n_relevant

  # In the segmented case, values retain their list dimension. This constructs
  # a mask so that only the first item per segment is used in reduce_fn.
  if segments is not None:
    where = utils.first_item_segment_mask(segments, where=where)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  elif where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)


def precision_metric(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    topn: Optional[int] = None,
    key: Optional[Array] = None,
    rank_fn: RankFn = utils.ranks,
    cutoff_fn: CutoffFn = utils.cutoff,
    reduce_fn: Optional[ReduceFn] = jnp.mean
) -> Array:
  r"""Precision.

  .. note::

    This metric converts graded relevance to binary relevance by considering
    items with ``label >= 1`` as relevant and items with ``label < 1`` as
    non-relevant.

  Definition:

  .. math::
      \operatorname{precision@n}(s, y) =
      \frac{1}{n}
      \sum_i y_i \cdot \mathbb{I}\left[\operatorname{rank}(s_i) \leq n\right]

  where :math:`\operatorname{rank}(s_i)` indicates the rank of item :math:`i`
  after sorting all scores :math:`s` using ``rank_fn``.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The metric will only be computed on
      items that share the same segment.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The precision metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, where=where, segments=segments, key=key)
  retrieved_items = _retrieved_items(
      scores,
      ranks,
      where=where,
      segments=segments,
      topn=topn,
      cutoff_fn=cutoff_fn,
  )

  # Compute number of retrieved+relevant items and retrieved items.
  if segments is not None:
    n_retrieved_relevant = utils.segment_sum(
        retrieved_items * relevant_items, segments, where=where
    )
    n_retrieved = utils.segment_sum(retrieved_items, segments, where=where)
  else:
    n_retrieved_relevant = jnp.sum(
        retrieved_items * relevant_items, where=where, axis=-1
    )
    n_retrieved = jnp.sum(retrieved_items, where=where, axis=-1)

  # Compute precision but prevent division by zero.
  n_retrieved = jnp.where(n_retrieved == 0, 1., n_retrieved)
  values = n_retrieved_relevant / n_retrieved

  # In the segmented case, values retain their list dimension. This constructs
  # a mask so that only the first item per segment is used in reduce_fn.
  if segments is not None:
    where = utils.first_item_segment_mask(segments, where=where)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  elif where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)


def ap_metric(scores: Array,
              labels: Array,
              *,
              where: Optional[Array] = None,
              topn: Optional[int] = None,
              key: Optional[Array] = None,
              rank_fn: RankFn = utils.ranks,
              cutoff_fn: CutoffFn = utils.cutoff,
              reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Average Precision.

  .. note::

    This metric converts graded relevance to binary relevance by considering
    items with ``label >= 1`` as relevant and items with ``label < 1`` as
    non-relevant.

  Definition:

  .. math::
      \operatorname{ap}(s, y) =
      \frac{1}{\sum_i y_i}
      \sum_i y_i \operatorname{precision@rank}_{s_i}(s, y)

  where :math:`\operatorname{precision@rank}_{s_i}(s, y)` indicates the
  precision at the rank of item :math:`i`.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The average precision metric.
  """
  # Get the relevant items.
  relevant_items = jnp.where(labels >= 1, jnp.ones_like(labels),
                             jnp.zeros_like(labels))

  # Get the retrieved items.
  ranks = rank_fn(scores, where=where, key=key)
  retrieved_items = _retrieved_items(
      scores, ranks, where=where, topn=topn, cutoff_fn=cutoff_fn)

  # Compute ranks.
  ranks = rank_fn(scores, where=where, key=key)

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
  n_relevant = jnp.sum(relevant_items, where=where, axis=-1)

  # Compute average precision but prevent division by zero.
  n_relevant = jnp.where(n_relevant == 0, 1., n_relevant)
  values = sum_prec_at_k / n_relevant

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  if where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)


def dcg_metric(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    topn: Optional[int] = None,
    weights: Optional[Array] = None,
    key: Optional[Array] = None,
    gain_fn: Callable[[Array], Array] = default_gain_fn,
    discount_fn: Callable[[Array], Array] = default_discount_fn,
    rank_fn: RankFn = utils.ranks,
    cutoff_fn: CutoffFn = utils.cutoff,
    reduce_fn: Optional[ReduceFn] = jnp.mean
) -> Array:
  r"""Discounted cumulative gain (DCG).

  Definition :cite:p:`jarvelin2002cumulated`:

  .. math::
      \operatorname{dcg}(s, y) =
      \sum_i \operatorname{gain}(y_i)
      \cdot \operatorname{discount}(\operatorname{rank}(s_i))

  where :math:`\operatorname{rank}(s_i)` indicates the 1-based rank of item
  :math:`i` as computed by ``rank_fn``, :math:`\operatorname{gain}(y)` indicates
  the per-item gains as computed by ``gain_fn``, and,
  :math:`\operatorname{discount}(r)` indicates the per-item rank discounts as
  computed by ``discount_fn``.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The metric will only be computed on
      items that share the same segment.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the per-item weights.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    gain_fn: A function that maps relevance label to gain values.
    discount_fn: A function that maps 1-based ranks to discount values.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The DCG metric.
  """
  # Get the retrieved items.
  ranks = rank_fn(scores, where=where, segments=segments, key=key)
  retrieved_items = _retrieved_items(
      scores,
      ranks,
      segments=segments,
      where=where,
      topn=topn,
      cutoff_fn=cutoff_fn,
  )

  # Computes (weighted) gains.
  gains = gain_fn(labels)
  if weights is not None:
    gains *= weights

  # Computes rank discounts.
  discounts = discount_fn(ranks)

  # Compute DCG.
  if segments is not None:
    values = utils.segment_sum(
        retrieved_items * gains * discounts, segments, where=where
    )
  else:
    values = jnp.sum(retrieved_items * gains * discounts, axis=-1, where=where)

  # In the segmented case, values retain their list dimension. This constructs
  # a mask so that only the first item per segment is used in reduce_fn.
  if segments is not None:
    where = utils.first_item_segment_mask(segments, where=where)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  elif where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)


def ndcg_metric(
    scores: Array,
    labels: Array,
    *,
    where: Optional[Array] = None,
    segments: Optional[Array] = None,
    topn: Optional[int] = None,
    weights: Optional[Array] = None,
    key: Optional[Array] = None,
    gain_fn: Callable[[Array], Array] = default_gain_fn,
    discount_fn: Callable[[Array], Array] = default_discount_fn,
    rank_fn: RankFn = utils.ranks,
    cutoff_fn: CutoffFn = utils.cutoff,
    reduce_fn: Optional[ReduceFn] = jnp.mean
) -> Array:
  r"""Normalized discounted cumulative gain (NDCG).

  Definition :cite:p:`jarvelin2002cumulated`:

  .. math::
      \operatorname{ndcg}(s, y) =
      \operatorname{dcg}(s, y) / \operatorname{dcg}(y, y)

  where :math:`\operatorname{dcg}` is the discounted cumulative gain metric.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item. Items for which the score is :math:`-\inf` are treated
        as unranked items.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the metric.
    segments: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating segments within each list. The metric will only be computed on
      items that share the same segment.
    topn: An optional integer value indicating at which rank the metric cuts
      off. If ``None``, no cutoff is performed.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the per-item weights.
    key: An optional :func:`~jax.random.PRNGKey`. If provided, any random
      operations in this metric will be based on this key.
    gain_fn: A function that maps relevance label to gain values.
    discount_fn: A function that maps 1-based ranks to discount values.
    rank_fn: A function that maps scores to 1-based ranks.
    cutoff_fn: A function that maps ranks and a cutoff integer to a binary array
      indicating which items are cutoff.
    reduce_fn: An optional function that reduces the metric values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The NDCG metric.
  """
  # Compute regular dcg.
  regular_dcg = dcg_metric(
      scores,
      labels,
      where=where,
      segments=segments,
      topn=topn,
      weights=weights,
      key=key,
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
      where=where,
      segments=segments,
      topn=topn,
      weights=weights,
      key=None,
      gain_fn=gain_fn,
      discount_fn=discount_fn,
      rank_fn=utils.ranks,
      cutoff_fn=utils.cutoff,
      reduce_fn=None)

  # Compute the result as `dcg / ideal_dcg` while preventing division by zero.
  ideal_dcg = jnp.where(ideal_dcg == 0., 1., ideal_dcg)
  values = regular_dcg / ideal_dcg

  # In the segmented case, values retain their list dimension. This constructs
  # a mask so that only the first item per segment is used in reduce_fn.
  if segments is not None:
    where = utils.first_item_segment_mask(segments, where=where)

  # Setup mask to ignore lists with only invalid items in reduce_fn.
  elif where is not None:
    where = jnp.any(where, axis=-1)

  return utils.safe_reduce(values, where=where, reduce_fn=reduce_fn)
