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

"""Implementations of common ranking losses in JAX.

A ranking loss is a differentiable function that expresses the cost of a ranking
induced by item scores compared to a ranking induced from relevance labels. Rax
provides a number of ranking losses as JAX functions that are implemented
according to the :class:`~rax.types.LossFn` interface.

Loss functions are designed to operate on the last dimension of its inputs. The
leading dimensions are considered batch dimensions. To compute per-list losses,
for example to apply per-list weighting or for distributed computing of losses
across devices, please use standard JAX transformations such as :func:`jax.vmap`
or :func:`jax.pmap`.

Standalone usage:

>>> scores = jnp.array([2., 1., 3.])
>>> labels = jnp.array([1., 0., 0.])
>>> rax.softmax_loss(scores, labels)
DeviceArray(1.4076059, dtype=float32)

Usage with a batch of data and a mask to indicate valid items.

>>> scores = jnp.array([[2., 1., 0.], [1., 0.5, 1.5]])
>>> labels = jnp.array([[1., 0., 0.], [0., 0., 1.]])
>>> where = jnp.array([[True, True, False], [True, True, True]])
>>> rax.pairwise_hinge_loss(
...     scores, labels, where=where, reduce_fn=jnp.mean)
DeviceArray(0.16666667, dtype=float32)

To compute gradients of each loss function, please use standard JAX
transformations such as :func:`jax.grad` or :func:`jax.value_and_grad`:

>>> scores = jnp.asarray([[0., 1., 3.], [1., 2., 0.]])
>>> labels = jnp.asarray([[0., 0., 1.], [1., 0., 0.]])
>>> jax.grad(rax.softmax_loss)(scores, labels, reduce_fn=jnp.mean)
DeviceArray([[ 0.02100503,  0.0570976 , -0.07810265],
             [-0.37763578,  0.33262047,  0.04501529]], dtype=float32)

"""

import operator
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from rax._src import metrics
from rax._src import utils
from rax._src.types import Array
from rax._src.types import ReduceFn


def softmax_loss(scores: Array,
                 labels: Array,
                 *,
                 where: Optional[Array] = None,
                 weights: Optional[Array] = None,
                 label_fn: Callable[..., Array] = lambda a, where: a,
                 reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Softmax loss.

  Definition:

  .. math::
      \ell(s, y) =
      - \sum_i y_i \log \frac{\exp(s_i)}{\sum_j \exp(s_j)}

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    label_fn: A label function that maps labels to probabilities. Default keeps
      labels as-is.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The softmax loss.
  """
  # Applies mask so that masked elements do not count towards the loss.
  if where is not None:
    labels = jnp.where(where, labels, jnp.zeros_like(labels))
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Scales labels and scores to match the cross entropy loss.
  labels_probabilities = label_fn(labels, where=where)
  scores_log_softmax = jax.nn.log_softmax(scores, axis=-1)

  # Computes per-element cross entropy.
  softmax_cross_entropy = labels_probabilities * scores_log_softmax

  # Reduces softmax cross-entropy loss.
  loss = -jnp.sum(softmax_cross_entropy, axis=-1, where=where)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def poly1_softmax_loss(scores: Array,
                       labels: Array,
                       *,
                       epsilon: float = 1.0,
                       where: Optional[Array] = None,
                       weights: Optional[Array] = None,
                       reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Poly1 softmax loss.

  Definition :cite:p:`leng2022polyloss`:

  .. math::
      \ell(s, y) = softmax(s, y) + \epsilon * (1 - pt)

  where :math:`softmax` is the standard softmax loss as implemented in
  :func:`~rax.softmax_loss` and :math:`pt` is the target softmax probability
  defined as:

  .. math::
      pt = \sum_i \frac{y_i}{\sum_j y_j} \frac{\exp(s_i)}{\sum_j \exp(s_j)}

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    epsilon: A float hyperparameter indicating the weight of the leading
      polynomial coefficient in the poly loss.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The poly1 softmax loss.
  """
  # Compute softmax cross-entropy loss without batch reduction.
  ce = softmax_loss(
      scores, labels, where=where, weights=weights, reduce_fn=None)

  # Applies mask so that masked elements do not count towards the loss.
  if where is not None:
    labels = jnp.where(where, labels, jnp.zeros_like(labels))
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)

  # Apply weights to labels.
  if weights is not None:
    labels *= weights

  # Compute target probabilities.
  scores_softmax = jax.nn.softmax(scores)
  labels_normalized = utils.normalize_probabilities(labels, where=where)
  pt = jnp.sum(labels_normalized * scores_softmax, where=where, axis=-1)

  # For lists where all items are masked, this sets pt to 1 so that the term
  # (1 - pt) is set to 0 for the loss computation.
  if where is not None:
    pt = jnp.where(jnp.all(jnp.logical_not(where), axis=-1), 1., pt)

  # Compute and return the poly1 loss.
  loss = ce + epsilon * (1. - pt)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def unique_softmax_loss(scores: Array,
                        labels: Array,
                        *,
                        where: Optional[Array] = None,
                        weights: Optional[Array] = None,
                        gain_fn: Optional[Callable[
                            [Array], Array]] = metrics.default_gain_fn,
                        reduce_fn: ReduceFn = jnp.mean) -> Array:
  r"""Unique softmax loss.

  Definition :cite:p:`zhu2020listwise`:

  .. math::
      \ell(s, y) =
      - \sum_i \operatorname{gain}(y_i)
      \log \frac{\exp(s_i)}{\exp(s_i) + \sum_{j : y_j < y_i} \exp(s_j)}

  where :math:`\operatorname{gain}(y_i)` is a user-specified gain function
  applied to label :math:`y_i` to boost items with higher relevance.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    gain_fn: An optional function that maps relevance labels to gain values. If
      provided, the per-item losses are multiplied by ``gain_fn(label)`` to
      boost the importance of relevant items.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The unqiue softmax loss.
  """
  # Construct pairwise matrices for scores and labels. The labels matrix will
  # indicate, for each item, which other items have a smaller label.
  labels_lt = jnp.expand_dims(labels, -2) < jnp.expand_dims(labels, -1)
  scores_repeated = jnp.repeat(
      jnp.expand_dims(scores, -2), scores.shape[-1], axis=-2)

  # Build an identity mask to select items during softmax computation.
  identity_mask = jnp.identity(scores.shape[-1], dtype=jnp.bool_)

  # Apply mask to ignore invalid items.
  if where is not None:
    labels_lt &= jnp.expand_dims(where, -2)
    identity_mask &= jnp.expand_dims(where, -2)

  # Compute log-softmax for each of the items' unique softmax distribution. This
  # effectively computes the log_softmax for each item using only itself and
  # items with a smaller relevance label in the denominator. The computed
  # log_softmax distribution is reduced by selecting its diagonal elements.
  log_softmax = jax.nn.log_softmax(
      scores_repeated,
      axis=-1,
      where=(identity_mask | labels_lt),
      initial=jnp.min(scores))
  log_softmax = jnp.diagonal(log_softmax, axis1=-2, axis2=-1)

  # Apply per-item weights.
  if weights is not None:
    log_softmax *= weights

  # Apply per-item `gain_fn` boost.
  if gain_fn is not None:
    log_softmax *= gain_fn(labels)

  # Compute per-list loss and return a reduced loss.
  loss = -jnp.sum(log_softmax, axis=-1, where=where)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


def listmle_loss(scores: Array,
                 labels: Array,
                 *,
                 key: Optional[Array] = None,
                 where: Optional[Array] = None,
                 reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""ListMLE Loss.

  .. note::

    This loss performs sorting using the given labels. If the labels contain
    multiple identical values, you should provide a :func:`~jax.random.PRNGKey`
    to the ``key`` argument to make sure ties are broken randomly during the
    sorting operation.

  Definition :cite:p:`xia2008listwise`:

  .. math::
      \ell(s, y) =
      - \sum_i \log
        \frac{\exp(s_i)}{\sum_j I[rank(y_j) \ge rank(y_i)] \exp(s_j)}

  where :math:`\operatorname{rank}(y_i)` indicates the rank of item :math:`i`
  after sorting all labels :math:`y`.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    key: An optional :func:`~jax.random.PRNGKey` to perform random tie-breaking.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The listmle loss.
  """
  # Sort scores and mask by labels.
  if where is None:
    where = jnp.ones_like(scores, dtype=jnp.bool_)

  scores_sorted, where_sorted = utils.sort_by(
      labels, [scores, where], where=where, key=key)

  # Compute cumulative logsumexp.
  lse = utils.logcumsumexp(
      scores_sorted, axis=-1, where=where_sorted, reverse=True)

  # Reduce list MLE loss.
  loss = -jnp.sum(scores_sorted - lse, axis=-1, where=where_sorted)
  return utils.safe_reduce(loss, reduce_fn=reduce_fn)


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


def pairwise_hinge_loss(scores: Array,
                        labels: Array,
                        *,
                        where: Optional[Array] = None,
                        weights: Optional[Array] = None,
                        reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Pairwise hinge loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j I[y_i > y_j] \max(0, 1 - (s_i - s_j))

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise hinge loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.gt)

  # Compute hinge function on scores.
  scores = jax.nn.relu(1. - scores)

  # Apply mask to labels, so only valid items are considered.
  if where is not None:
    where = compute_pairs(where, operator.and_)
    labels &= where

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    scores *= weights

  return utils.safe_reduce(scores, where=labels, reduce_fn=reduce_fn)


def pairwise_logistic_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Pairwise logistic loss.

  Definition :cite:p:`burges2005learning`:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j I[y_i > y_j] \log(1 + \exp(-(s_i - s_j)))

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise logistic loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.gt)

  # Compute numerically stable version of logistic function on scores.
  scores = jax.nn.relu(-scores) + jnp.log1p(jnp.exp(-jnp.abs(scores)))

  # Apply mask to labels, so only valid items are considered.
  if where is not None:
    where = compute_pairs(where, operator.and_)
    labels &= where

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    scores *= weights

  return utils.safe_reduce(scores, where=labels, reduce_fn=reduce_fn)


def pointwise_sigmoid_loss(scores: Array,
                           labels: Array,
                           *,
                           where: Optional[Array] = None,
                           weights: Optional[Array] = None,
                           reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Sigmoid cross entropy loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i y_i * -log(sigmoid(s_i)) + (1 - y_i) * -log(1 - sigmoid(s_i))

  This loss converts graded relevance to binary relevance by considering items
  with ``label >= 1`` as relevant and items with ``label < 1`` as non-relevant.

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional [..., list_size]-Array, indicating which items are valid
      for computing the loss. Items for which this is False will be ignored when
      computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The sigmoid cross entropy loss.
  """

  # Convert to binary relevance labels.
  labels = jnp.where(labels >= 1, jnp.ones_like(labels), jnp.zeros_like(labels))

  # A numerically stable version of sigmoid cross entropy.
  loss = (
      jax.nn.relu(scores) - scores * labels +
      jnp.log(1. + jnp.exp(-jnp.abs(scores))))

  if weights is not None:
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)


def pointwise_mse_loss(scores: Array,
                       labels: Array,
                       *,
                       where: Optional[Array] = None,
                       weights: Optional[Array] = None,
                       reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Mean squared error loss.

  Definition:

  .. math::
      \ell(s, y) = \sum_i (y_i - s_i)^2

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The mean squared error loss.
  """
  loss = jnp.square(scores - labels)

  if weights is not None:
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)


def pairwise_mse_loss(scores: Array,
                      labels: Array,
                      *,
                      where: Optional[Array] = None,
                      weights: Optional[Array] = None,
                      reduce_fn: Optional[ReduceFn] = jnp.mean) -> Array:
  r"""Pairwise mean squared error loss.

  Definition:

  .. math::
      \ell(s, y) =
      \sum_i \sum_j ((y_i - y_j) - (s_i - s_j))^2

  Args:
    scores: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      score of each item.
    labels: A ``[..., list_size]``-:class:`~jax.numpy.ndarray`, indicating the
      relevance label for each item.
    where: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating which items are valid for computing the loss. Items for which
      this is False will be ignored when computing the loss.
    weights: An optional ``[..., list_size]``-:class:`~jax.numpy.ndarray`,
      indicating the weight for each item.
    reduce_fn: An optional function that reduces the loss values. Can be
      :func:`jax.numpy.sum` or :func:`jax.numpy.mean`. If ``None``, no reduction
      is performed.

  Returns:
    The pairwise mean squared error loss.
  """
  # Expand scores and labels into pairwise versions.
  scores = compute_pairs(scores, operator.sub)
  labels = compute_pairs(labels, operator.sub)

  # Compute pairwise mask.
  if where is not None:
    where = compute_pairs(where, operator.and_)

  # Compute squared error between score pairs and label pairs.
  loss = jnp.square(scores - labels)

  # Apply weights to scores.
  if weights is not None:
    weights = compute_pairs(weights, lambda x, y: x)
    loss *= weights

  return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)
